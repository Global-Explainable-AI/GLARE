#!/usr/bin/env python3
"""
Validate NL-SQL dataset for GLARE.

Checks whether the 50,000 synthetic (NL question, SQL query) pairs are
faithful translations of each other.  Because the generation code is
template-based and not shipped in this repo, validation must be done
post-hoc on the generated JSONL files.

Checks performed
────────────────
1. Structural integrity   – JSON parses, expected fields present
2. SQL syntax             – every SQL parses via sqlite3
3. SQL safety             – only SELECT statements (no writes)
4. Fence markers          – SQL_START / SQL_END present and well-formed
5. Entity consistency     – object/class names in the NL question appear
                            in the SQL query AND in the allowed lists
6. Schema compliance      – SQL references only tables/columns in the
                            known schema
7. Allowed-list compliance – SQL literal strings are in allowed_objects
                            or allowed_classes
8. Template diversity     – distribution of query types is reported
9. Duplicate detection    – flags exact-duplicate NL or SQL entries

Usage
─────
    python validate_nl_sql_dataset.py [--data-dir data/] [--verbose]
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ── Known schema ──────────────────────────────────────────────────────

KNOWN_TABLES = {
    "objects", "object_aliases", "classes", "images", "image_objects",
}
KNOWN_COLUMNS = {
    # objects
    "object_id", "canonical_name", "ratio", "train_count", "val_count",
    # object_aliases
    "alias_id", "alias_name",
    # classes
    "class_id", "class_name",
    # images
    "image_id", "image_name", "probability", "split",
    # image_objects  (composite PK reuses image_id, object_id)
    # SQL aliases used in training data
    "count", "total", "percent", "percentage", "cooccur_count",
    "percent_of_anchor", "count_without", "obj_count", "image_count",
    "result", "avg_objects", "min_objects", "max_objects",
    "obj1", "obj2", "obj3", "obj4", "obj5",
}

# ── Helpers ───────────────────────────────────────────────────────────

def extract_fence(assistant_text: str, user_text: str = ""):
    """Return the SQL between SQL_START and SQL_END.

    In the training data the fence is split: SQL_START is at the end of
    the user message and SQL_END at the end of the assistant message.
    We handle both styles (split and unified).
    """
    # Style 1: both markers in assistant text
    if "SQL_START" in assistant_text and "SQL_END" in assistant_text:
        m = re.search(r"SQL_START\s*\n?(.*?)\s*SQL_END", assistant_text, re.DOTALL)
        if m:
            return m.group(1).strip(), "unified"

    # Style 2: SQL_START in user text, SQL_END in assistant text (training data)
    if user_text.rstrip().endswith("SQL_START"):
        sql = assistant_text
        if sql.rstrip().endswith("SQL_END"):
            sql = sql.rstrip()[: -len("SQL_END")].rstrip()
        return sql.strip(), "split"

    # Style 3: SQL_END only in assistant text (SQL_START was in user prompt)
    if "SQL_END" in assistant_text:
        sql = re.sub(r"\s*SQL_END\s*$", "", assistant_text)
        return sql.strip(), "split"

    # Fallback: treat entire content as SQL
    return assistant_text.strip(), "none"


def parse_user_message(content: str):
    """Extract question, allowed_objects, and allowed_classes from user msg."""
    question = ""
    allowed_objects = []
    allowed_classes = []

    # Question line
    m = re.search(r"question:\s*(.+?)(?:\n|$)", content)
    if m:
        question = m.group(1).strip()

    # Allowed objects
    m = re.search(r"allowed_objects.*?:\s*(\[.*?\])", content, re.DOTALL)
    if m:
        try:
            allowed_objects = json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Allowed classes
    m = re.search(r"allowed_classes.*?:\s*(\[.*?\])", content, re.DOTALL)
    if m:
        try:
            allowed_classes = json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    return question, set(allowed_objects), set(allowed_classes)


def extract_sql_string_literals(sql: str):
    """Return all single-quoted string literals from SQL."""
    return re.findall(r"'([^']*)'", sql)


def classify_query_type(question: str, sql: str):
    """Heuristic classification of the query template type."""
    q = question.lower()
    s = sql.lower()

    if "rarest" in q and "object" in q:
        if "4-object" in q: return "rarest_4tuple"
        if "3-object" in q: return "rarest_3tuple"
        if "2-object" in q or "pair" in q: return "rarest_2tuple"
        return "rarest_ntuple"
    if re.search(r"(most common|top \d+|most frequent)", q):
        return "top_k_objects"
    if re.search(r"(least common|bottom|rarest object)", q):
        return "bottom_k_objects"
    if "co-occurrence" in q or "appear with" in q or "objects appear" in q:
        return "cooccurrence"
    if "percentage" in q or "what %" in q or "percent" in q:
        if "but no" in q or "but not" in q:
            return "pct_and_not"
        if " and " in q:
            return "pct_boolean_and"
        return "pct_single"
    if "how many" in q:
        if "exactly" in q:
            return "exact_count"
        if "don't have" in q or "don't contain" in q or "lacking" in q or "without" in q:
            return "count_without"
        return "count_query"
    if "does any" in q or "do any" in q or "is there" in q:
        return "existence_check"
    if "which class" in q or "where is" in q or "which has more" in q:
        return "class_ranking"
    if "what objects" in q or "show me" in q or "list" in q:
        return "list_objects"
    if "max object" in q or "min object" in q or "avg" in q or "average" in q:
        return "aggregate_stats"
    if "how many class" in q or "how many object" in q or "how many unique" in q:
        return "schema_introspection"
    if "image" in q and ("with" in q or "contain" in q):
        if "but no" in q:
            return "images_with_and_not"
        return "images_with_object"
    if "count of" in q and "lacking" in q:
        return "count_without"
    if "distinguishing" in q or "unique to" in q or "only in" in q:
        return "distinguishing_features"
    if "threshold" in q or "at least" in q:
        return "threshold_filter"
    return "other"


def sql_syntax_check(sql: str) -> str | None:
    """Try to prepare the SQL via sqlite3. Returns error string or None."""
    # We wrap in EXPLAIN so it validates syntax without needing actual tables
    try:
        conn = sqlite3.connect(":memory:")
        # Create stub tables so column references resolve
        conn.executescript("""
            CREATE TABLE objects (
                object_id INTEGER PRIMARY KEY,
                canonical_name TEXT,
                ratio REAL,
                train_count INTEGER,
                val_count INTEGER
            );
            CREATE TABLE object_aliases (
                alias_id INTEGER PRIMARY KEY,
                object_id INTEGER,
                alias_name TEXT
            );
            CREATE TABLE classes (
                class_id INTEGER PRIMARY KEY,
                class_name TEXT UNIQUE
            );
            CREATE TABLE images (
                image_id INTEGER PRIMARY KEY,
                image_name TEXT,
                class_id INTEGER,
                probability REAL,
                split TEXT
            );
            CREATE TABLE image_objects (
                image_id INTEGER,
                object_id INTEGER,
                PRIMARY KEY (image_id, object_id)
            );
        """)
        conn.execute(f"EXPLAIN {sql}")
        conn.close()
        return None
    except Exception as e:
        return str(e)


def check_sql_safety(sql: str) -> list[str]:
    """Return list of safety warnings (non-SELECT statements, etc.)."""
    warnings = []
    s = sql.strip().upper()
    if not s.startswith("SELECT"):
        # Allow subqueries starting with WITH, but flag truly unsafe ops
        if s.startswith("WITH"):
            pass  # CTE is fine
        else:
            warnings.append(f"SQL does not start with SELECT: starts with {s[:20]}")
    for keyword in ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE"]:
        # Avoid false positives from string literals or column names
        if re.search(rf"\b{keyword}\b", sql, re.IGNORECASE):
            # Double check it's not inside a string literal
            stripped = re.sub(r"'[^']*'", "", sql)
            if re.search(rf"\b{keyword}\b", stripped, re.IGNORECASE):
                warnings.append(f"SQL contains potentially unsafe keyword: {keyword}")
    return warnings


def check_entity_consistency(question: str, sql: str,
                              allowed_objects: set, allowed_classes: set):
    """
    Check that entity names in the NL question also appear in the SQL,
    and vice versa.  Returns (warnings, info).
    """
    warnings = []
    info = []

    sql_literals = set(extract_sql_string_literals(sql))
    q_lower = question.lower().replace("_", " ")

    # Check: every string literal in SQL should be either an allowed object
    # or allowed class
    for lit in sql_literals:
        if lit not in allowed_objects and lit not in allowed_classes:
            # Some literals are structural: 'Yes', 'No', etc.
            if lit not in {"Yes", "No", "yes", "no"}:
                warnings.append(
                    f"SQL literal '{lit}' not in allowed_objects or allowed_classes"
                )

    # Check: class names in SQL should appear in the question
    sql_classes = sql_literals & allowed_classes
    for cls in sql_classes:
        cls_readable = cls.replace("_", " ")
        if cls_readable not in q_lower and cls not in question.lower():
            warnings.append(
                f"Class '{cls}' in SQL but not found in question: '{question}'"
            )

    # Check: object names in SQL should appear in the question
    sql_objects = sql_literals & allowed_objects
    for obj in sql_objects:
        obj_readable = obj.replace("_", " ")
        if obj_readable not in q_lower and obj not in question.lower():
            warnings.append(
                f"Object '{obj}' in SQL but not found in question: '{question}'"
            )

    # Check: named entities in the question should appear in the SQL
    # (This is harder — we look for known objects/classes mentioned in question)
    for obj in allowed_objects:
        obj_readable = obj.replace("_", " ")
        if (obj in question.lower() or obj_readable in q_lower):
            # Check this object is in SQL (either as literal or as column ref)
            if obj not in sql_literals:
                # Might be in a subquery or referenced differently
                info.append(
                    f"Object '{obj}' appears in question but not as SQL literal"
                )

    return warnings, info


# ── Main validation ───────────────────────────────────────────────────

def validate_file(filepath: str, verbose: bool = False):
    """Validate a single JSONL file. Returns a summary dict."""
    results = {
        "file": filepath,
        "total": 0,
        "valid_json": 0,
        "valid_structure": 0,
        "fence_detected": 0,
        "sql_syntax_ok": 0,
        "sql_safe": 0,
        "entity_consistent": 0,
        "allowed_list_compliant": 0,
        "warnings": [],
        "errors": [],
        "query_type_dist": Counter(),
        "nl_duplicates": 0,
        "sql_duplicates": 0,
    }

    seen_questions = Counter()
    seen_sqls = Counter()

    with open(filepath) as f:
        for line_num, line in enumerate(f, 1):
            results["total"] += 1
            line = line.strip()
            if not line:
                continue

            # 1. JSON parse
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                results["errors"].append(f"Line {line_num}: JSON parse error: {e}")
                continue
            results["valid_json"] += 1

            # 2. Structure check
            msgs = record.get("messages", [])
            if len(msgs) < 3:
                results["errors"].append(
                    f"Line {line_num}: Expected >=3 messages, got {len(msgs)}"
                )
                continue
            system_msg = msgs[0].get("content", "")
            user_msg = msgs[1].get("content", "")
            assistant_msg = msgs[2].get("content", "")

            if msgs[0].get("role") != "system":
                results["errors"].append(f"Line {line_num}: First msg not system role")
                continue
            if msgs[1].get("role") != "user":
                results["errors"].append(f"Line {line_num}: Second msg not user role")
                continue
            if msgs[2].get("role") != "assistant":
                results["errors"].append(f"Line {line_num}: Third msg not assistant role")
                continue
            results["valid_structure"] += 1

            # 3. Parse user message
            question, allowed_objects, allowed_classes = parse_user_message(user_msg)
            if not question:
                results["warnings"].append(
                    f"Line {line_num}: Could not extract question from user msg"
                )

            # 4. Fence detection
            sql, fence_style = extract_fence(assistant_msg, user_msg)
            if fence_style in ("unified", "split"):
                results["fence_detected"] += 1
            else:
                results["warnings"].append(
                    f"Line {line_num}: Missing SQL_START/SQL_END fence markers"
                )

            if not sql:
                results["errors"].append(f"Line {line_num}: Empty SQL after extraction")
                continue

            # 5. SQL syntax
            syntax_err = sql_syntax_check(sql)
            if syntax_err is None:
                results["sql_syntax_ok"] += 1
            else:
                results["errors"].append(
                    f"Line {line_num}: SQL syntax error: {syntax_err}"
                )
                if verbose:
                    print(f"  SQL: {sql[:200]}")

            # 6. SQL safety
            safety_warnings = check_sql_safety(sql)
            if not safety_warnings:
                results["sql_safe"] += 1
            else:
                for w in safety_warnings:
                    results["warnings"].append(f"Line {line_num}: {w}")

            # 7. Entity consistency
            entity_warnings, entity_info = check_entity_consistency(
                question, sql, allowed_objects, allowed_classes
            )
            if not entity_warnings:
                results["entity_consistent"] += 1
            else:
                for w in entity_warnings:
                    results["warnings"].append(f"Line {line_num}: {w}")

            # 8. Allowed-list compliance
            sql_literals = set(extract_sql_string_literals(sql))
            all_allowed = allowed_objects | allowed_classes | {"Yes", "No", "yes", "no"}
            violations = sql_literals - all_allowed
            if not violations:
                results["allowed_list_compliant"] += 1
            else:
                results["warnings"].append(
                    f"Line {line_num}: SQL literals not in allowed lists: {violations}"
                )

            # 9. Query type classification
            qtype = classify_query_type(question, sql)
            results["query_type_dist"][qtype] += 1

            # 10. Duplicate tracking
            seen_questions[question] += 1
            seen_sqls[sql] += 1

    # Count duplicates
    results["nl_duplicates"] = sum(c - 1 for c in seen_questions.values() if c > 1)
    results["sql_duplicates"] = sum(c - 1 for c in seen_sqls.values() if c > 1)
    results["unique_questions"] = len(seen_questions)
    results["unique_sqls"] = len(seen_sqls)

    return results


def print_report(results: dict, verbose: bool = False):
    """Pretty-print the validation report."""
    total = results["total"]
    if total == 0:
        print(f"  (empty file)")
        return

    def pct(n):
        return f"{n}/{total} ({100*n/total:.1f}%)"

    print(f"\n{'='*70}")
    print(f"  VALIDATION REPORT: {results['file']}")
    print(f"{'='*70}")

    print(f"\n  Total records:              {total}")
    print(f"  Valid JSON:                 {pct(results['valid_json'])}")
    print(f"  Valid structure:            {pct(results['valid_structure'])}")
    print(f"  Fence markers detected:     {pct(results['fence_detected'])}")
    print(f"  SQL syntax valid:           {pct(results['sql_syntax_ok'])}")
    print(f"  SQL safety check passed:    {pct(results['sql_safe'])}")
    print(f"  Entity names consistent:    {pct(results['entity_consistent'])}")
    print(f"  Allowed-list compliant:     {pct(results['allowed_list_compliant'])}")

    print(f"\n  Unique NL questions:        {results['unique_questions']}")
    print(f"  Unique SQL queries:         {results['unique_sqls']}")
    print(f"  Duplicate NL questions:     {results['nl_duplicates']}")
    print(f"  Duplicate SQL queries:      {results['sql_duplicates']}")

    # Query type distribution
    print(f"\n  Query Type Distribution:")
    print(f"  {'Type':<30} {'Count':>6} {'Pct':>7}")
    print(f"  {'-'*30} {'-'*6} {'-'*7}")
    for qtype, count in results["query_type_dist"].most_common():
        print(f"  {qtype:<30} {count:>6} {100*count/total:>6.1f}%")

    # Errors
    if results["errors"]:
        print(f"\n  ERRORS ({len(results['errors'])}):")
        for e in results["errors"][:20]:
            print(f"    - {e}")
        if len(results["errors"]) > 20:
            print(f"    ... and {len(results['errors'])-20} more")

    # Warnings
    if results["warnings"]:
        n_warn = len(results["warnings"])
        print(f"\n  WARNINGS ({n_warn}):")
        # Group by type
        warn_types = Counter()
        entity_mismatch_examples = []
        for w in results["warnings"]:
            # Extract the warning type (after "Line N: ")
            m = re.match(r"Line \d+: (.+?)(?::|$)", w)
            if m:
                key = m.group(1).split(":")[0].strip()
                warn_types[key] += 1
                # Collect entity mismatch examples for special display
                if "in SQL but not found in question" in w:
                    entity_mismatch_examples.append(w)
            else:
                warn_types[w[:60]] += 1
        for wtype, wcount in warn_types.most_common():
            print(f"    - {wtype}: {wcount} occurrences")

        # Always show entity mismatch details (these are genuine data bugs)
        if entity_mismatch_examples:
            print(f"\n  ENTITY MISMATCH DETAILS (NL-SQL translation errors):")
            for w in entity_mismatch_examples[:20]:
                print(f"    - {w}")

        if verbose:
            print(f"\n  All warnings (first 30):")
            for w in results["warnings"][:30]:
                print(f"    - {w}")

    # Overall assessment
    print(f"\n  {'─'*50}")
    critical_pass = (
        results["valid_json"] == total
        and results["valid_structure"] == total
        and results["sql_syntax_ok"] == total
        and results["sql_safe"] == total
        and results["fence_detected"] == total
    )
    entity_pass_rate = results["entity_consistent"] / total * 100 if total else 0
    allowed_pass_rate = results["allowed_list_compliant"] / total * 100 if total else 0

    if critical_pass and entity_pass_rate > 95:
        print(f"  OVERALL: PASS")
    elif critical_pass and entity_pass_rate > 80:
        print(f"  OVERALL: PASS WITH WARNINGS")
    else:
        print(f"  OVERALL: ISSUES DETECTED")

    print(f"    - Critical checks (JSON/structure/syntax/safety/fences): "
          f"{'ALL PASS' if critical_pass else 'FAILURES DETECTED'}")
    print(f"    - Entity consistency: {entity_pass_rate:.1f}%")
    print(f"    - Allowed-list compliance: {allowed_pass_rate:.1f}%")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Validate GLARE NL-SQL dataset for correctness"
    )
    parser.add_argument(
        "--data-dir", default="data/",
        help="Directory containing JSONL files (default: data/)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed warnings and failing SQL"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: data directory '{data_dir}' not found", file=sys.stderr)
        sys.exit(1)

    jsonl_files = sorted(data_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"Error: no .jsonl files found in '{data_dir}'", file=sys.stderr)
        sys.exit(1)

    print(f"GLARE NL-SQL Dataset Validator")
    print(f"Scanning {len(jsonl_files)} file(s) in {data_dir}/\n")

    all_results = []
    for fpath in jsonl_files:
        print(f"Validating {fpath.name} ...")
        results = validate_file(str(fpath), verbose=args.verbose)
        all_results.append(results)
        print_report(results, verbose=args.verbose)

    # Combined summary
    if len(all_results) > 1:
        total = sum(r["total"] for r in all_results)
        print(f"\n{'='*70}")
        print(f"  COMBINED SUMMARY ({total} total records)")
        print(f"{'='*70}")
        for key in ["valid_json", "valid_structure", "fence_detected",
                     "sql_syntax_ok", "sql_safe", "entity_consistent",
                     "allowed_list_compliant"]:
            s = sum(r[key] for r in all_results)
            print(f"  {key:<30} {s}/{total} ({100*s/total:.1f}%)")

        combined_dist = Counter()
        for r in all_results:
            combined_dist.update(r["query_type_dist"])
        print(f"\n  Combined Query Type Distribution:")
        for qtype, count in combined_dist.most_common():
            print(f"    {qtype:<30} {count:>6} {100*count/total:>6.1f}%")


if __name__ == "__main__":
    main()
