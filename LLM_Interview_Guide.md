# LLM Interview Questions & Answers for GLARE

A comprehensive guide for answering LLM-related interview questions in the context of the GLARE project (*GLARE: A Natural Language Interface for Querying Global Explanations*).

---

## 1. Core Concept Questions

### Q: What is GLARE and how does it use LLMs differently from typical AI systems?

**Answer:** GLARE treats LLMs as *semantic parsers*, not content generators. Most systems use LLMs to directly generate answers or explanations, which introduces hallucination risk. GLARE instead uses a fine-tuned LLM (e.g., Gemma 2-9B) to translate natural language questions into SQL queries over a relational database that stores pre-computed global explanations. The LLM handles the *form* (structural mapping from question to query), while a deterministic database handles the *content* (actual explanation data). This means the core data retrieval pipeline is hallucination-free because results come from verified ground-truth data, not LLM generation.

### Q: Why use LLMs as semantic parsers instead of generators? What are the trade-offs?

**Answer:**
- **Advantages**: Eliminates hallucination in the data retrieval layer; answers are grounded in actual model explanation data. Provides formal correctness guarantees since every output is a valid SQL query executed against a verified database. Enables auditability — you can inspect the generated SQL to verify the answer.
- **Trade-offs**: Constrains the system to questions expressible in the template taxonomy (24 query types). Open-ended or philosophical questions about the model can't be answered. A final LLM step that formats results into natural language still carries some residual hallucination risk at the presentation layer.
- **Why it works here**: Explainability queries are inherently structured (frequencies, comparisons, set operations), making them a natural fit for SQL. The query space is well-defined and can be covered by templates.

### Q: How does GLARE reduce hallucination risk compared to directly prompting an LLM about model behavior?

**Answer:** If you ask an LLM "what features does this classifier use for bedrooms?", it generates plausible-sounding text that may not reflect the model's actual decision process. GLARE eliminates this by: (1) pre-computing actual explanations (Minimal Sufficient Explanations / MSX) and storing them in a relational database, (2) using the LLM *only* to translate questions into SQL, never to reason about model behavior, and (3) executing SQL deterministically against ground-truth data. The LLM never sees the explanation data during query translation. A downstream LLM call formats results into natural language, so the presentation layer has some residual risk, but the underlying facts are verified.

---

## 2. Architecture & Pipeline Questions

### Q: Walk me through the GLARE pipeline end-to-end.

**Answer:** Four steps:
1. **Query Parsing** — The fine-tuned LLM receives the user's natural language question along with a system prompt containing the database schema and allowed entity names. It identifies the query template type and extracts parameters (class names, object names, thresholds).
2. **SQL Generation** — The LLM outputs SQL between special fence markers (`SQL_START` / `SQL_END`). For example, for "What percentage of bedroom images contain both bed and wall?", it generates a SQL query with appropriate JOINs, WHERE clauses, and aggregations.
3. **Validation & Execution** — The generated SQL is parsed for syntactic correctness, checked for safety (e.g., no DROP/DELETE), and executed against the explanation database.
4. **Response Generation** — Results are formatted into natural language with supporting evidence (e.g., example images with highlighted objects).

### Q: Why store explanations in a relational database rather than a vector store or knowledge graph?

**Answer:** Global explanations have inherently *relational* structure — they describe relationships between objects, classes, images, and decision boundaries. A relational database is the natural fit because:
- Explanations decompose into rows (one per local explanation / MSX) with typed columns (image_id, class_name, object_name, confidence).
- SQL provides exact, deterministic answers to frequency, ranking, comparison, and set-operation queries — no approximate retrieval.
- Unlike vector stores, there's no embedding similarity threshold to tune, and answers are provably correct given the data.
- Unlike knowledge graphs, the schema is simpler and SQL is a more mature target for LLM generation.

### Q: How does the database schema work?

**Answer:** The core tables are:
- **images**: Maps `image_id` to `class_name` (the predicted/ground-truth scene category).
- **image_objects**: Maps `image_id` to `object_name`, representing which objects appear in each image's explanation (MSX).

This simple two-table schema supports a wide range of queries through JOINs and aggregations: object frequency per class, co-occurrence, cross-class comparison, absence analysis, and more. The simplicity is deliberate — it keeps the SQL generation task tractable for small LLMs.

---

## 3. Training & Fine-Tuning Questions

### Q: How did you generate training data without manual annotation?

**Answer:** Fully synthetic data generation. We defined 24 query templates organized into three tiers:
- **Core**: Object frequency, boolean combinations, top-k ranking, co-occurrence
- **Extended**: N-way combinations, cross-class comparison, set operations, conditional co-occurrence
- **Contrastive**: Absence analysis, threshold filtering, distinguishing features, counterfactual reasoning

Each template is a function that takes random parameters (sampled class names, object names, thresholds) and outputs both a natural language question and the corresponding ground-truth SQL. We generated 50,000 training pairs this way — zero human annotation, fully reproducible, and trivially extensible to new query types.

### Q: What is fence-based loss masking and why is it critical?

**Answer:** During fine-tuning, we wrap the target SQL in special markers (`SQL_START` / `SQL_END`). The training loss is computed *only on the SQL tokens between the fences*, not on the prompt tokens. This is critical because:
- Without it, the model could memorize associations between specific entity names (e.g., "bed") and SQL patterns, rather than learning the *structural mapping* from question patterns to SQL patterns.
- With loss masking, the model receives no gradient signal from entity names in the prompt, forcing it to learn *relational algebra* — the abstract structure of how questions map to queries.
- **Proof it works**: A model trained entirely on ADE20K entities (bed, wall, stove) transfers to Pascal VOC entities (lear, torso, rhand) at 90.6% accuracy — it learned structure, not vocabulary.

### Q: What fine-tuning approach did you use? LoRA, full fine-tuning, or something else?

**Answer:** The models were fine-tuned with the fence-based loss masking strategy on 50,000 synthetic (question, SQL) pairs. The key innovation isn't the optimizer or parameter-efficient method — it's the *loss masking*. By zeroing out loss on non-SQL tokens, we get a model that generalizes across domains. The training achieves 100% fence detection, 100% SQL parse rate, and 100% execution rate on the fine-tuned models.

---

## 4. Model Selection & Scaling Questions

### Q: Why does a 2B parameter model match a 27B model on this task?

**Answer:** GLARE's query mapping is a *well-defined structural task*, not open-ended generation. The model needs to learn 24 template patterns and map natural language variations onto them. This is a bounded mapping problem — once a model has enough capacity to represent all 24 templates and their parameter extraction patterns, additional parameters don't help. Results:
- Gemma 2 (2B): 95.4%
- Gemma 2 (9B): 95.2%
- Gemma 2 (27B): ~95%

**Key insight for interviews**: For well-constrained structural tasks, model capacity saturates early. This has practical implications — you can deploy on smaller, cheaper hardware with no quality loss. The lesson generalizes: if you can define your task precisely enough, you can serve it with a small model.

### Q: Is there a minimum model size? What happens below 2B parameters?

**Answer:** Yes. Qwen 2.5 at 0.5B achieves only 4.4% accuracy. Below ~2B parameters, models can't reliably learn the mapping from natural language to SQL for 24 template types. This suggests ~2B is the capacity floor for this complexity of structural mapping. This is valuable knowledge for deployment cost optimization.

### Q: Why did base models (without fine-tuning) score near 0%?

**Answer:** Even Gemma 2-27B scores 0.0% without fine-tuning. This is because the task requires learning: (1) the specific database schema (table names, column names, relationships), (2) the fence format (`SQL_START`/`SQL_END`), and (3) the 24 template structures. Base models can write generic SQL but cannot generate schema-specific, correctly-formatted queries for this domain without training. This demonstrates that the synthetic training pipeline is essential — prompt engineering alone is insufficient.

---

## 5. Robustness & Generalization Questions

### Q: How robust is GLARE to noisy or informal user input?

**Answer:** Tested against 7 perturbation types:
| Perturbation | Robustness |
|:---|:---:|
| Synonym substitution | 100% |
| Verbose padding | 97% |
| Spelling errors | 89% |
| Telegraphic style | 84% |
| Word swap | 82% |
| Grammar corruption | 76% |
| Word drop | 48% |

The model is robust to meaning-preserving perturbations (synonyms, verbosity, minor errors). Word drop scores lowest (48%), but this is *correct behavior* — dropping entity names (e.g., removing "bed" from the query) destroys the information needed for correct SQL. Low robustness to information-destroying perturbations is a feature, not a bug.

### Q: How does zero-shot cross-dataset transfer work?

**Answer:** A model trained entirely on ADE20K (scene classification with objects like wall, bed, stove) is tested on Pascal VOC (part-based object recognition with entities like lear, torso, rhand) — a completely different domain and vocabulary. Results:
- Gemma 2 (27B): 90.6%
- Gemma 2 (2B): 90.0%

This works because of fence-based loss masking. The model learned *SQL compositional structure* (how to compose JOINs, WHERE clauses, aggregations based on question patterns), not entity-specific associations. At inference, new entity names are provided in the prompt, and the model slots them into the learned structural patterns.

---

## 6. Failure Modes & Limitations Questions

### Q: Where does GLARE fail? What are the weakest query types?

**Answer:** The model achieves 100% on 18 of 25 query types. The two weakest:
- **N-way combinations (49%)**: Requires multi-way self-joins with precise ordering constraints — structurally complex SQL that the model sometimes gets subtly wrong.
- **Exact count queries (43%)**: Requires HAVING clauses with exact equality conditions — a rarely-seen SQL pattern in the training distribution.

**How to fix**: Increase the sampling frequency for these templates in the synthetic data generator. The failure modes are well-understood and the fix is straightforward engineering, not a fundamental limitation.

### Q: What are GLARE's fundamental limitations?

**Answer:**
1. **Single-turn only**: No conversational context or follow-up questions. Real explanation-seeking is iterative.
2. **Template-bounded**: Can only answer questions that map to one of the 24 defined templates. Novel query structures require adding new templates.
3. **One explanation type**: Currently only supports DNF-based explanations (Minimal Sufficient Explanations). Doesn't handle concept-based, prototype, or counterfactual explanation formats.
4. **No user studies**: SQL accuracy was evaluated but not whether users actually understand models better with GLARE vs. traditional dashboards.
5. **Presentation-layer risk**: The final LLM step that formats SQL results into natural language still carries hallucination risk, even though the data retrieval itself is deterministic.

---

## 7. Broader LLM Concept Questions (with GLARE Context)

### Q: What is the difference between RAG and GLARE's approach?

**Answer:** RAG (Retrieval-Augmented Generation) retrieves relevant text chunks via embedding similarity and feeds them to an LLM for answer generation. GLARE differs in key ways:
- **Retrieval**: RAG uses approximate vector similarity; GLARE uses exact SQL queries. No retrieval noise.
- **Generation**: RAG still relies on the LLM to synthesize an answer from retrieved chunks (hallucination risk). GLARE's answers come directly from SQL execution (deterministic).
- **Structure**: RAG treats knowledge as unstructured text. GLARE treats it as structured relational data, enabling precise aggregations, comparisons, and set operations that RAG struggles with.
- **Best for**: GLARE is better when data is inherently structured and queries need exact numerical answers. RAG is better for unstructured knowledge where approximate retrieval is acceptable.

### Q: How does GLARE relate to Text-to-SQL research?

**Answer:** GLARE is an application of Text-to-SQL in the XAI domain, but with specific innovations:
- **Domain-specific templates**: Rather than arbitrary SQL generation (like Spider/WikiSQL benchmarks), GLARE constrains to 24 explanation-relevant templates, trading generality for reliability.
- **Synthetic training**: Unlike standard Text-to-SQL datasets that require manual annotation, GLARE generates all training data programmatically.
- **Fence-based loss masking**: A novel training technique that enables cross-domain transfer — not standard in Text-to-SQL literature.
- **Safety-critical framing**: Template constraints provide formal guarantees (every SQL is a valid template instance), which matters for trustworthy AI applications.

### Q: How would you extend GLARE to support multi-turn dialogue?

**Answer:** This is listed as future work. Approaches to consider:
1. **Context window approach**: Include previous (question, SQL, result) turns in the prompt, letting the LLM resolve coreferences ("What about kitchens?" after asking about bedrooms).
2. **State tracking**: Maintain explicit dialogue state (current class, objects of interest, comparison context) and inject it into the prompt.
3. **Query refinement**: Allow users to modify the previous SQL rather than generating from scratch ("Now filter to only images with confidence > 0.8").
4. **Challenges**: Context window growth, coreference resolution accuracy, and maintaining the no-hallucination guarantee across turns.

### Q: If you were deploying GLARE in production, what would your architecture look like?

**Answer:**
1. **Frontend**: Web interface with natural language input, query history, and visualization of results (images with highlighted objects).
2. **LLM Service**: The fine-tuned 2B model (since it matches larger models) served via a lightweight inference server (vLLM, TGI, or ONNX Runtime). Low latency since 2B models are fast.
3. **Database**: PostgreSQL or SQLite for the explanation database. Pre-computed MSX data loaded at startup.
4. **Validation Layer**: SQL parsing and safety checks (block DROP, DELETE, UPDATE — only SELECT allowed). Timeout enforcement.
5. **Response Formatter**: A separate LLM call (or template-based formatting) to convert SQL results to natural language.
6. **Monitoring**: Log generated SQL for debugging, track query type distribution, flag low-confidence or failed parses.

---

## 8. Quick-Fire Interview Questions

| Question | Key Point to Hit |
|:---|:---|
| "What's novel about GLARE?" | LLM as parser not generator; explanations as databases; fence-based loss masking for domain transfer |
| "How do you handle hallucination?" | LLM never sees explanation data; SQL execution is deterministic; only the presentation layer has residual risk |
| "Why not just use GPT-4?" | Base models score 0% without fine-tuning; task requires schema-specific training; 2B fine-tuned beats 27B base |
| "How do you evaluate?" | 500 held-out queries; result-match accuracy (not just SQL-match); per-template breakdown; robustness perturbations; cross-dataset transfer |
| "What's the training cost?" | Zero annotation cost (synthetic data); 50K examples generated programmatically; fine-tuning a 2B model is cheap |
| "How would you improve it?" | Multi-turn dialogue; broader explanation types; user studies; more templates for weak query types; informal language support |
| "What's the 'LLM as parser' pattern?" | Use LLMs for structural mapping (NL to formal language), let deterministic systems handle content. Applies to medical, legal, financial domains. |

---

## Tips for Answering

1. **Lead with the insight, not the implementation.** "We use LLMs as semantic parsers to eliminate hallucination in the data retrieval pipeline" is better than "We fine-tuned Gemma 2 on synthetic SQL data."
2. **Know your numbers.** 95% accuracy, 90% transfer, 2B = 27B, 24 templates, 50K training examples, 100% on 18/25 query types.
3. **Acknowledge limitations honestly.** Single-turn, template-bounded, no user studies, presentation-layer hallucination risk. Interviewers respect candor.
4. **Connect to broader patterns.** GLARE isn't just an XAI tool — it demonstrates a general "LLM as parser" pattern applicable across domains where you need flexible input + faithful output.
5. **Have a "what's next" ready.** Multi-turn dialogue, concept-based explanations, and user studies are concrete, well-motivated next steps.
