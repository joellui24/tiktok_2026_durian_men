# Hackathon Context — Conversational Shopping Agent

## 1. Background

Traditional e-commerce search engines heavily rely on static keyword matching, failing to capture the fluid shifts of genuine consumer psychology and the distinction between open-ended browsing and high-intent buying.

In modern conversational commerce, constructing an intelligent agent that leverages **dynamic context programming** is critical to bridging the gap between ambiguous user queries and complex product catalogs. Solving this challenge directly impacts core industrial metrics.

---

## 2. Problem Statement

Participants are challenged to architect an intelligent, next-generation shopping agent capable of navigating real-world customer dynamics.

Moving beyond rigid search filters, the engineered system must demonstrate:

- Deep cognitive understanding
- Runtime architectural agility
- Commercial efficiency

The system should use the provided [Amazon Reviews 2023 dataset](https://amazon-reviews-2023.github.io/) and be built around the following four core pillars.

### I. Core Architecture: Intent Routing & Hybrid Pipeline

#### Dual-Track Routing

Instantly detect the user's underlying intent and route the conversation into one of two tracks:

- **Buying track** — use high-precision filtering to lock hard constraints for targeted, high-intent purchases.
- **Browsing track** — use diverse dense retrieval to support open-ended, cross-category, scenario-based discovery.

#### Pipeline Base

Construct an in-memory retrieval and ranking pipeline:

**Multi-Route Retrieval → LLM Semantic Ranking**

Retrieval can combine:

- Keyword similarity
- Category matching
- Vector similarity

### II. Dialog Strategy: Multi-Turn Scenario Evolution

#### Dynamic State Machine

Build a robust conversational state tracker that can handle:

- **Information Accumulation** — incrementally add user constraints and preferences across turns.
- **Intent Override** — detect when the user changes direction and erase, replace, or rewrite outdated slots.

#### Proactive Guidance

When a query is too general and retrieval produces an excessively large candidate pool, trigger an **immediate retrieval cutoff**.

Instead of continuing with weak recommendations, the agent should generate structured clarification questions that help the user converge toward a smaller and more relevant product set.

### III. Self-Evolution: Dynamic Context Programming

#### Runtime Adaptation

Use accumulated dialog history for **Personalized Context Distillation**.

The system should continuously update:

- Short-term session state
- Long-term user profile

#### Adaptive Orchestration

Use dynamic context programming to adapt the workflow at runtime.

The agent should be capable of:

- Re-orchestrating its retrieval and reasoning strategy
- Aligning the pipeline with the evolving user intent
- Iteratively refining its own guidance logic

### IV. Evaluation Matrix: Product & Efficiency Metrics

Evaluation is anchored on the **final purchased product** contained in the Amazon evaluation session.

Performance is measured across three dimensions:

#### Coverage — Hit Rate@K

Measures retrieval-stage recall and whether the purchased product appears within the candidate set.

#### Precision — MRR / Top-K Hit Rate

Measures how accurately the ranking stage pushes the exact purchased product toward the top of the recommendation list.

#### Efficiency — MTTC

**MTTC = Mean Turns to Conversion**

Systems are rewarded for guiding the user to the correct product in fewer conversation turns and penalized for unnecessary conversational load.

---

## 3. Constraints & Scope

| Category | Constraints & Scope |
| --- | --- |
| **In Scope** | Designing sensitive intent-detection modules to split traffic into **Buying** and **Browsing** tracks. |
|  | Implementing heterogeneous retrieval routing, including dynamic weights, custom truncation, and slot decay over time. |
|  | Engineering runtime-adaptive memory layers for personalized context distillation. |
|  | Fine-tuning prompt strategies or local scoring logic for the LLM ranking stage to compress decision paths. |
| **Out of Scope** | UI/UX development. Evaluation is performed through automated backend APIs and headless pipelines. |
|  | Training or full-parameter fine-tuning of base foundational LLMs. |
|  | Deploying heavy external industrial vector database clusters. The solution must run entirely in-memory for lightweight execution. |
|  | Multi-modal processing. The challenge is restricted to text catalogs, structured metadata, and text dialogs. |
| **Limits** | **Maximum turns:** 10 turns per session. Exceeding this limit forces termination and results in a zero score. |
|  | **Catalog mutation:** The Amazon product dataset is read-only. Structural mutations and mock ASIN injections are not allowed. |
| **Allowed Assumptions** | Inputs are pre-cleaned text strings. Spelling correction, typo handling, and ASR noise do not need to be addressed. |
|  | Product catalog, pricing, and category trees remain static during the hackathon. |
|  | Each evaluation session is an isolated single-user interaction. Multi-user concurrency stress testing is not required. |

---

## 4. Available Resources & Data

Participants receive a frozen and reproducible competition kit derived from the Amazon Reviews 2023 dataset.

### Competition Data

- **Product catalog:** 50,000 products from the Amazon Reviews 2023 `Clothing_Shoes_and_Jewelry` category.
- **Public development sessions:** 200 labeled sessions for local testing and iteration.
- **Private evaluation sessions:** 800 additional sessions retained by the organizer for final evaluation.
- Public and private evaluation sessions use **separate users and target products**.

### Participant Resources

The competition provides:

- A weak **BM25 starter Agent** implemented in Python.
- A deterministic local evaluator for:
  - Hit Rate@10
  - MRR
  - MTTC
  - Efficiency
  - Combined `TechnicalScore`
- A published Python Agent interface.
- A machine-readable API contract.
- Evaluation configuration.
- Reproducible baseline results.
- Data documentation.
- Submission rules.
- A SHA256 checksum file for validating the downloaded catalog.

Participants may modify or completely replace the starter Agent while continuing to use the official local evaluator.

The participant kit supports:

- Keyword retrieval
- Rule-based methods
- Dense retrieval
- Hybrid retrieval
- Reranking
- Local models
- External model APIs

### Model / API Access

The organizer does **not** provide:

- Hosted model access
- API keys
- Model tokens
- Third-party API credits

A paid LLM is **not required** to complete the challenge.

Teams that choose to use external services are responsible for:

- Their own credentials
- Usage limits
- API costs
- Ensuring secrets are not published in public repositories

### Resources

- Participant repository: https://github.com/TechJam2026/techjam-conversational-search
- Participant Kit Release: https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit
- Original Amazon Reviews 2023 documentation: https://amazon-reviews-2023.github.io/

The competition catalog and evaluation sessions are already prepared and frozen by the organizer. Participants do **not** need to download or reconstruct the full upstream Amazon Reviews 2023 dataset.

---

## 5. Generated Retrieval Indexes

Two derived SQLite databases have been created from the immutable
`data/catalog.jsonl` product catalog. They do not modify the competition data or
introduce new products. They reorganize existing catalog fields into indexed
lookup structures for faster candidate selection.

### Category Index

- **File:** `data/category_index.sqlite3`
- **Builder and query API:** `starter/category_index.py`
- **Approximate size:** 24 MB

The category index contains:

| Data | Count |
| --- | ---: |
| Products | 50,000 |
| Category hierarchy nodes | 1,832 |
| Product-to-category mappings | 238,403 |

It preserves the complete Amazon category hierarchy rather than using only the
final category name. This prevents collisions where the same name, such as
`T-Shirts`, appears under several departments or branches.

The database supports:

- Exact normalized category-name lookup.
- Full category-path lookup.
- Category-to-product lookup.
- Product-ASIN-to-category-path lookup.
- Parent/child category traversal.
- Leaf-category filtering.

### `ask_attribute` Index

- **File:** `data/attribute_index.sqlite3`
- **Committed archive:** `data/attribute_index.sqlite3.gz`
- **Builder and query API:** `starter/attribute_index.py`
- **Approximate size:** 149 MB

This index contains a product posting list for every non-null value accepted by
`ask_attribute`. The stored values follow the local evaluator's constraint
classification policy where applicable.

| `ask_attribute` | Product/value mappings | Distinct normalized values |
| --- | ---: | ---: |
| `category` | 426,680 | 2,693 |
| `material` | 72,898 | 8,927 |
| `color` | 53,107 | 4,984 |
| `size` | 3,989 | 3,703 |
| `style` | 8,684 | 3,894 |
| `brand` | 69,776 | 23,225 |
| `budget` | 10,675 | 2,662 |
| `feature` | 103,688 | 37,326 |
| `use_case` | 866 | 804 |
| `other` | 196,201 | 59,763 |

There is one SQLite view per attribute, including `category_values`,
`material_values`, `color_values`, and `size_values`. The value `null` has no
view because it means that the agent is not asking for an attribute.

The query API supports:

- Attribute value to matching product ASINs.
- Product ASIN and attribute to known values.
- Numeric minimum/maximum budget filtering.
- Intersecting multiple filters into one remaining product-ID candidate set.
- Loading one selected attribute into an in-memory hash map.

The catalog does not contain transaction order IDs. The stable identifier used
for these mappings is `parent_asin`, which is also the product ID scored by the
evaluator. Each indexed attribute value therefore corresponds to one or more
`parent_asin` values.

For example, `maximum_price=100` removes every product priced at $100 or above.
Adding `material=cotton` then intersects that result with the cotton posting
list. Only product IDs satisfying both constraints remain eligible for ranking.
Across attributes the filter operation uses AND; multiple accepted values for a
single attribute use OR.

### Why the Derived Databases Are Larger

The original JSONL catalog is approximately 58 MB. The generated SQLite files
use approximately 173 MB in total because they deliberately duplicate compact
keys across posting lists and maintain additional B-tree indexes. This is a
space-for-speed tradeoff: the source catalog is optimized for storage and
sequential reading, while the derived databases are optimized for repeated
retrieval during conversations.

The intended retrieval pipeline is:

1. Parse the current customer constraint and select its `ask_attribute`.
2. Perform an indexed category or attribute lookup.
3. Reduce the original 50,000 products to a small candidate set.
4. Run BM25, semantic retrieval, or LLM reranking only on those candidates.

SQLite index access is technically **O(log n)** because it uses B-trees. When
strict average **O(1)** exact-key access is useful, `AttributeIndex.load_hashmap()`
loads only the required attribute into memory. Loading one targeted map avoids
placing the entire 149 MB database in memory and keeps startup cost proportional
to the current retrieval route.

Both databases are deterministic build artifacts and can be regenerated after
the catalog is downloaded:

```bash
python -m starter.category_index --force
python -m starter.attribute_index --force
```

The expanded attribute database exceeds GitHub's normal per-file limit, so the
repository stores a compressed 34 MB archive. It can be restored with
`gzip -dk data/attribute_index.sqlite3.gz`; the expanded local copy is ignored
by Git.

---

## 6. Deliverables

### 1. Written Project Description — Devpost

Provide a clear written description covering:

- How the solution addresses the problem statement
- Development tools used
  - Example: VS Code, Colab, Jupyter
- APIs used
  - Example: OpenAI GPT-4o, Google Maps API
- Libraries and frameworks used
  - Example: Hugging Face Transformers, PyTorch, scikit-learn, pandas
- Datasets and assets used
  - Example: Google Local Reviews dataset, manually labelled data

### 2. Public Code / GitHub Repository

Submit a public repository containing:

- Well-structured and commented code covering all major solution components.
- A `README.md` containing:
  - Project overview
  - Setup and installation instructions
  - Steps to reproduce results
  - Brief reflection on solution limitations
  - What would be improved with more time
  - Team member contributions, if applicable

### 3. Demo Video

Submit a short video that:

- Demonstrates the solution working end-to-end.
- May show:
  - API usage
  - Inference examples
  - Model predictions
  - Evaluation results
  - Result analysis
- Is uploaded to YouTube.
- Is publicly viewable.
- Is linked in the Devpost description.
- Does not include third-party trademarks or copyrighted content without permission.

For backend or NLP tracks where a front-end interface is not applicable, a walkthrough demonstrating API usage, inference examples, or evaluation analysis is accepted.

---

## 7. Judging Criteria

| Judging Criteria | Definition | Weight |
| --- | --- | ---: |
| **Technical Execution** | The solution demonstrates strong engineering fundamentals, including well-structured code, thoughtful architecture, and effective use of APIs or models. The demo runs reliably, and the technical complexity reflects deliberate and capable decision-making. | **35%** |
| **Innovation & Problem Insight** | The project demonstrates originality in both idea and approach. It stands out through a clear understanding of the problem, why it matters, and how directly the solution addresses it. | **20%** |
| **Impact & Relevance** | The project demonstrates clear potential to create value for real users or stakeholders, with meaningful reach, tangible benefits, and relevance beyond the hackathon prompt. | **20%** |
| **Feasibility & Practicality** | The solution is realistic and buildable beyond a prototype. Resource usage is proportionate, the architecture is sustainable under real-world conditions, and the implementation is grounded rather than speculative. | **15%** |
| **Presentation & Communication** | **Final Event Only.** The team communicates the project clearly, presents a coherent problem-to-solution story, and can respond to questions with depth and genuine understanding of the implementation. | **10%** |

---

## Key Challenge Summary

The core task is to build a conversational shopping agent that can:

1. Determine whether the user is **buying** or **browsing**.
2. Dynamically decide whether to **retrieve products immediately** or **ask a clarification question**.
3. Maintain and update user constraints across multiple turns.
4. Handle sudden changes in intent.
5. Combine multiple retrieval approaches.
6. Rank retrieved products accurately.
7. Minimize unnecessary conversation turns.
8. Run efficiently within the competition's lightweight, in-memory constraints.
9. Maximize the probability that the user's final purchased product appears near the top of the recommendation list.
10. Complete the interaction within the strict **10-turn maximum**.
