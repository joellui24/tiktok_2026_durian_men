# Intent-Routed Conversational Product Search

This is the submission bundle for the TechJam Conversational E-Commerce Search Challenge. The private evaluator entry point is [`agent.py`](agent.py), which exports the required `Agent` class. All commands below must be run from this repository root.

The agent is fully offline at inference time. It makes no API calls, requires no credentials, and reports zero prompt and completion tokens.

## Architecture

The agent maintains isolated conversation state for each `session_id`, applies deterministic hard constraints before ranking, and asks the next attribute using the selected ranker's posterior distribution.

```text
first user message
  -> intent and constraint parser
  -> exact category/attribute filtering
  -> Buying: frozen Linear ranker
     Browsing / Boundary / Intent Override / unknown: Hybrid FM ranker
  -> free-form requests: local BGE dense retrieval + rank fusion
     (SQLite FTS5 fallback)
  -> clarification question + ordered Top-10 products
```

Intent Override messages atomically replace obsolete state. Free-form hard filters such as exact category, material, color, brand, negation, alternatives, and numeric price ceilings remain authoritative; semantic retrieval can only reorder products that survive them.

## Submission contents

```text
.
├── agent.py                       # private-evaluator entry point
├── requirements.txt               # complete Python dependency manifest
├── requirements-training.txt      # optional retraining dependency
├── README.md                      # setup, method, results, and limitations
├── SHA256SUMS                     # bundled-asset integrity checks
├── src/                           # runtime implementation and helper modules
├── artifacts/
│   ├── linear_model.sqlite3       # Buying route
│   └── fm_model.sqlite3           # Hybrid FM route
├── data/
│   ├── attribute_index.sqlite3.gz # exact-filter index, expanded temporarily
│   ├── category_index.sqlite3
│   ├── lexical_index.sqlite3      # dependency-free semantic fallback
│   └── semantic_embeddings.npz
├── models/bge-small-en-v1.5/      # local quantized ONNX encoder and license
└── training/                      # optional FM trainer and frozen config
```

The 149 MB expanded attribute index is deliberately stored as a 35.7 MB gzip archive. `Agent()` extracts it to an isolated temporary directory once per agent instance and removes it when `close()` is called. The submission never changes the frozen product catalog or evaluator files.

## Setup

Python 3.11 or later is required; Python 3.13 is recommended and Python 3.12/3.13 were used during development.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
shasum -a 256 -c SHA256SUMS       # Windows alternative: Get-FileHash
```

Inference does not need network access. Dependency installation is the only step that may access a package index.

The private tester does not need the `training/` directory. Its catalog-only retraining instructions are documented separately in [`training/README.md`](training/README.md).

## Private tester

Run the organizer's harness with this repository root as its working directory or on `PYTHONPATH`. The harness should import exactly:

```python
from agent import Agent

agent = Agent()                    # Agent(catalog_path) is also supported
agent.reset(session_id, user_profile)
response = agent.respond(session_id, user_message, turn, top_k)
agent.close()
```

The required methods and return shape are:

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None: ...
    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict: ...
```

Each response contains a customer-facing `message`, an allowed `ask_attribute` or `None`, up to ten ordered unique `parent_asin` recommendations, and non-negative token usage.

One-command smoke test:

```bash
python -B -c "from agent import Agent; a=Agent(); a.reset('demo', {}); print(a.respond('demo', 'Something breathable for summer', 1, 10)); a.close()"
```

## Reproduced public result

The frozen 200-session public evaluator produced:

| Metric | Result |
|---|---:|
| Hit Rate@10 | 0.995000 |
| MRR | 0.704468 |
| MTTC | 2.120000 |
| Efficiency | 0.888000 |
| Technical Score | 0.886440 |
| Reported tokens | 0 |

Scenario Hit Rate@10 was 0.9875 for Buying and 1.0000 for Browsing, Boundary, and Intent Override. The focused parser, state, lexical, dense, and retrieval regression suite passed 80 tests (one environment-dependent skip) before submission cleanup.

## Demonstrated multi-turn session

This frozen public Intent Override session shows state replacement and eventual conversion. Product identifiers below are the actual top-three outputs; the hidden target appeared at rank 5 on turn 4.

| Turn | Customer | Agent action | Top three |
|---:|---|---|---|
| 1 | “I'm looking for Accessories Belts. Buckle closure” | Asked `other` | `B00X5042IS`, `B07RZ33BCK`, `B00N4CEGEW` |
| 2 | “For that, what matters is: leather; 100% Leather.” | Applied constraints; asked `color` | `B07RQBD7MY`, `B01727EVCQ`, `B071P5SP48` |
| 3 | “Actually, ignore my earlier preference. What I need is: leather.” | Replaced obsolete intent; asked `feature` | `B00I1080VW`, `B0052T0SGK`, `B08TWV2QCN` |
| 4 | “For that, what matters is: Imported; Buckle closure.” | Applied new intent; target reached rank 5 | `B0119U5Z90`, `B00QINL7ZA`, `B0C5F4BMLV` |

## Models, latency, and cost

- Structured ranking uses one frozen Linear artifact and one frozen second-order Factorization Machine with regularized explicit context-item crosses.
- Natural free-form retrieval uses the local `BAAI/bge-small-en-v1.5` FastEmbed model, revision `52398278842ec682c6f32300af41344b1c0b0bb2`, as a quantized ONNX model. The model license is bundled.
- Inference network calls: none.
- API/LLM cost: USD 0.
- Reported LLM tokens: 0.
- Indicative free-form CPU latency: 2.1–2.9 seconds cold; 0.586-second warm median; 1.218-second observed warm p95.
- Indicative semantic peak memory: about 346 MB. Dense assets load lazily and are not loaded for formatted evaluator messages.

## Limitations and next improvements

- The deterministic parser is conservative and does not normalize every unseen category paraphrase, word-number budget, or implicit size expression.
- Parent-product metadata has incomplete or multi-valued brand, color, material, and size fields; unsupported feature/use-case exclusions therefore remain soft.
- Cross-field OR clauses are approximated as independent same-field alternatives.
- The current policy does not yet use the aggregate user profile for personalization.
- The compressed attribute index reduces repository size but needs about 149 MB of temporary disk space and adds cold-start extraction time.
- With more time, the next experiment would add a confidence-gated catalog entity linker for unseen paraphrases and safe profile-aware priors, followed by a genuinely untouched private-style validation split.

## Contribution split

- Intent routing contributor: Linear + Hybrid FM integration, scenario routing from the first prompt, ranking validation, and official-regression evaluation.
- Parsing/retrieval contributor: free-form constraint parser, conversational edit handling, dense semantic retrieval, lexical fallback, and generalization diagnostics.

## Data and reproducibility

The indexes and model artifacts contain only data derived from the frozen 50,000-product Amazon Reviews 2023 Clothing, Shoes and Jewelry catalog supplied for the challenge. Private evaluation sessions, organizer-only files, credentials, outputs, and API keys are not included. The original dataset is published by McAuley Lab, UCSD; the bundled BGE model is MIT licensed.
