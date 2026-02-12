# Stop Explaining Models. Start Letting Users Ask Questions.

*How treating global explanations as databases—and LLMs as semantic parsers, not generators—creates an XAI interface that achieves 95% accuracy, transfers to new datasets at 90%, and sidesteps the hallucination problem entirely.*

---

## The Problem Nobody Talks About in XAI

We've gotten remarkably good at *generating* explanations for deep learning models. Grad-CAM, SHAP, LIME, concept bottleneck models, prototype networks—the zoo of explanation methods grows every year. But here's the uncomfortable truth:

**Most explanations go unused.**

Not because they're wrong. Not because practitioners don't care about interpretability. But because the format is wrong. We hand users a static artifact—a saliency map, a set of rules, a list of prototypes—and say "here, understand your model." This is the equivalent of handing someone an encyclopedia when they asked a question.

Consider what happens when you compute a global explanation for a scene classifier trained on ADE20K. You might get hundreds of logical rules per class, expressed as disjunctive normal form (DNF) formulas:

> **bedroom** ← (bed ∧ wall) ∨ (bed ∧ curtain ∧ lamp) ∨ (bed ∧ pillow ∧ nightstand) ∨ ...
> **kitchen** ← (stove ∧ cabinet) ∨ (oven ∧ counter ∧ sink) ∨ (refrigerator ∧ floor ∧ cabinet) ∨ ...
> *... hundreds more rules per class, across 35 scene categories*

No human is going to read all of this. But every human who encounters this model will have *specific questions*: Does the model use background features for classification? What distinguishes kitchen from dining room? How often does "bed" appear in bedroom explanations?

The fundamental mismatch is that **explanations are generated as monologues, but understanding happens through dialogue**.

---

## The Idea: Explanations as Databases

Our paper, *GLARE: A Natural Language Interface for Querying Global Explanations*, starts from a simple observation:

> Global explanations have relational structure. They describe relationships between objects, classes, images, and decision boundaries. This is exactly the kind of data that databases are designed to store and query.

So instead of presenting explanations as a document to be read, we ingest them into a relational database. Each local explanation (a Minimal Sufficient Explanation, or MSX) becomes a row in a table. Objects, classes, confidence scores, and image IDs become columns. Suddenly, questions that would require a PhD student hours of manual analysis become SQL queries that execute in milliseconds.

The second observation is about LLMs. Everyone is using LLMs to *generate* explanations. We think this is the wrong job for them. LLMs hallucinate. When you ask an LLM to explain a model's behavior, it produces plausible-sounding text that may have nothing to do with how the model actually works.

**KEY INSIGHT**: Use LLMs as *semantic parsers*, not generators. The LLM translates natural language questions into SQL. The answers come from the actual explanation data. The LLM never touches the content of the explanation—it only handles the *form* of the query.

This gives you the best of both worlds: the flexibility of natural language input with the formal correctness of database queries. No hallucination possible, because every answer is computed deterministically from ground-truth data.

---

## How It Works: A Walkthrough

Let me trace through a concrete example to make this tangible.

A user asks:

> "What percentage of bedroom images contain both bed and wall?"

Here's what happens inside GLARE:

**Step 1: Query parsing.** The fine-tuned LLM (Gemma 2-9B) receives the question along with a system prompt containing the database schema and allowed entity names. It identifies this as a "boolean AND percentage" query template and extracts the parameters: class = `bedroom`, objects = `bed`, `wall`.

**Step 2: SQL generation.** The LLM outputs SQL between special fence markers:

```sql
SQL_START
SELECT ROUND(COUNT(DISTINCT CASE WHEN io1.object_name = 'bed'
  AND io2.object_name = 'wall' THEN i.image_id END) * 100.0
  / COUNT(DISTINCT i.image_id), 2) AS percentage
FROM images i
JOIN image_objects io1 ON i.image_id = io1.image_id
JOIN image_objects io2 ON i.image_id = io2.image_id
WHERE i.class_name = 'bedroom'
SQL_END
```

**Step 3: Validation and execution.** The SQL is parsed for syntactic correctness, checked for safety, and executed against the explanation database.

**Step 4: Response generation.** The result is formatted as: *"73.2% of bedroom images contain both bed and wall."* alongside supporting evidence images with highlighted objects.

The critical property: **the LLM never sees the explanation data**. It only translates the question into a formal query. The data remains the single source of truth.

---

## The Training Pipeline: How You Teach an LLM SQL Without Any Manual Annotation

One of the biggest challenges was generating training data. We needed thousands of (question, SQL) pairs, but manually writing them would be tedious and wouldn't scale to new datasets.

Our solution: **fully synthetic training data**.

We define 24 query templates organized into three tiers:

- **Core queries**: Object frequency, boolean combinations, top-k ranking, co-occurrence
- **Extended queries**: N-way combinations (self-joins), cross-class comparison, set operations, conditional co-occurrence
- **Contrastive queries**: Absence analysis, threshold filtering, distinguishing features, counterfactual reasoning

Each template is a function that takes random parameters (class names, object names, thresholds) and outputs both a natural language question and the corresponding SQL. We generate 50,000 training examples this way. No human annotation. No expensive labeling.

But there's a subtlety. If you fine-tune on these examples naively, the model might memorize associations between specific entity names and SQL patterns. It would learn "when you see 'bed', put 'bed' in the WHERE clause" rather than "when you see an object name, put it in the WHERE clause."

Our solution is **fence-based loss masking**. During fine-tuning, we wrap the target SQL in special markers (`SQL_START` / `SQL_END`) and compute the training loss *only on the SQL tokens*. The model never receives gradient signal for the prompt tokens (which contain entity names and natural language). This forces it to learn the *relational algebra* of querying—the structural mapping from question patterns to SQL patterns—rather than dataset-specific vocabulary associations.

The proof is in the transfer results: a model trained entirely on ADE20K (with objects like "wall", "bed", "stove") transfers to Pascal VOC (with objects like "lear", "torso", "rhand") at **90.6% accuracy**. It has never seen these entity names during training. It learned the structure, not the vocabulary.

---

## Results: What Actually Works (and What Doesn't)

### The Headline Numbers

On 500 held-out test queries from the ADE20K domain:

- **Gemma 2 (9B) fine-tuned**: 95.2% result-match accuracy, 100% fence detection, 100% SQL parse rate, 100% execution rate
- **Gemma 2 (2B) fine-tuned**: 95.4% — identical to the 27B model
- **Qwen 2.5 (14B) fine-tuned**: 95.4%
- **Base models (no fine-tuning)**: Near 0% across the board

Several things stand out:

1. **Performance saturates early.** 2B, 9B, and 27B all hit ~95%. GLARE's query mapping is learnable even by small models.
2. **Base models achieve near zero.** Even Gemma 2-27B scores 0.0% on result-match. Our task requires learning the specific schema and template structures. The synthetic training pipeline is essential.
3. **There's a minimum capacity threshold.** Qwen 2.5 at 0.5B achieves only 4.4%. Below ~2B parameters, models can't reliably learn the mapping.

### Per-Query-Type Breakdown

The model achieves **100% accuracy on 18 of 25 query types**. The two weakest:

- **N-way combinations** (49%): Multi-way self-joins with ordering constraints
- **Exact count queries** (43%): HAVING clauses with exact equality

These failure modes are informative: the model struggles precisely where SQL requires exact structural precision in rarely-seen patterns. The fix is straightforward—increase sampling frequency for these templates.

### Robustness

Tested against 7 perturbation types:

| Perturbation | Robustness |
|:-------------|:----------:|
| Synonym substitution | 100% |
| Verbose padding | 97% |
| Spelling errors | 89% |
| Telegraphic | 84% |
| Word swap | 82% |
| Grammar corruption | 76% |
| Word drop | 48% |

The learned SQL mapping is largely invariant to lexical and syntactic surface variation. Word drop (48%) is the outlier, but this makes sense: removing entity names destroys the information needed for correct SQL. Low robustness to information-destroying perturbations is a correct property, not a bug.

### Zero-Shot Cross-Dataset Transfer

Trained on ADE20K, tested on Pascal VOC (completely different vocabulary):

| Model | Transfer Accuracy |
|:------|:-----------------:|
| Gemma 2 (27B) | 90.6% |
| Gemma 2 (2B) | 90.0% |
| Qwen 2.5 (14B) | 90.0% |
| Gemma 2 (9B) | 89.6% |
| Qwen 2.5 (7B) | 87.2% |

Even 2B models transfer at 90%. The model learned SQL's compositional structure, not entity associations.

---

## The Deeper Lessons: What Generalizes Beyond This Paper

### 1. Use LLMs for Structure, Not Content

When you need both flexibility and faithfulness, **use the LLM to handle structure and let a deterministic system handle content**. GLARE uses the LLM to parse natural language into SQL (structural mapping). The answers come from database execution (deterministic content). This eliminates hallucination by construction.

This pattern applies broadly:
- **Medical diagnosis support**: LLM parses symptoms into database queries over clinical guidelines
- **Legal research**: LLM translates legal questions into structured searches over case law
- **Financial analysis**: LLM converts questions into queries over verified financial data

### 2. Synthetic Data + Structural Loss Masking = Domain-Agnostic Learning

The combination of synthetic training data and fence-based loss masking produces a model that learns *task structure* rather than *domain vocabulary*. The recipe:

1. Generate training examples synthetically by sampling parameters from templates
2. Mask the loss to focus only on structural output (SQL), not parameter values
3. At deployment, provide new parameter values in the prompt

This is viable for any domain where *structure* is fixed but *vocabulary* changes: code generation for different APIs, form filling for different schemas, query generation for different databases.

### 3. Template Taxonomies Beat End-to-End Generation

We chose 24 templates over arbitrary SQL generation. This gives:
- **Perfect coverage** on supported types (100% on 18/25)
- **Graceful degradation** on unsupported types (99.3% execution rate)
- **Easy extensibility**: one template + regenerate data
- **Formal guarantees**: every SQL is a valid template instance

The principle: **constrained generation with broad template coverage beats unconstrained generation for safety-critical applications**.

### 4. Explanation is a Query Problem, Not a Generation Problem

The broadest conceptual contribution: rethink what "explainability" means. The dominant paradigm is generation—compute an explanation, present it. But this treats users as passive consumers.

GLARE reframes explanation as **query answering**. The explanation data exists (computed once). The interface helps users ask the right questions and retrieve relevant slices. This aligns with research showing human explanation-seeking is iterative, question-driven, and contrastive.

### 5. The Minimum Viable Model Is Smaller Than You Think

2B parameters = 27B parameters for this task. For well-defined structural tasks (as opposed to open-ended generation), model capacity saturates quickly. If you can define your task precisely enough, you can serve it cheaply.

---

## What We'd Do Differently (and What's Next)

**Multi-turn dialogue.** GLARE handles single questions. Real explanation-seeking is iterative. Supporting conversational context is the natural next step.

**Broader explanation types.** We demonstrated with DNF-based explanations. The database schema should extend to concept-based, prototype, and counterfactual explanation formats.

**User studies.** We evaluated SQL accuracy but haven't measured whether users *understand models better* with GLARE vs. traditional dashboards. Task-level human evaluation is the missing piece.

**Handling informal language.** The 12% accuracy on informal register is a practical gap. Expanding the synthetic data generator's register coverage is straightforward engineering.

---

## Summary

GLARE demonstrates that the bottleneck in explainable AI is not generating better explanations—it's making existing explanations accessible. By treating global explanations as relational databases and using LLMs as semantic parsers (not content generators), we build an interface that:

- Achieves 95% query accuracy across diverse question types
- Transfers to new datasets at 90.6% without retraining
- Handles spelling errors, synonyms, and verbose input gracefully
- Eliminates hallucination by construction, not by hope
- Runs on 2B-parameter models with no performance loss

The broader lesson: when you need trustworthy answers from flexible input, let the LLM handle structure and let a verified system handle content. This "LLM as parser" pattern sidesteps the hallucination problem entirely and applies far beyond explainability.

The even broader lesson: if your users are drowning in information, the solution isn't better information—it's a better question-and-answer interface. Explanations aren't documents. They're databases waiting for the right query.

---

*Paper and code available at the project website.*
