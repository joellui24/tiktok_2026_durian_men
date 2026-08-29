# TechJam Conversational E-Commerce Search Challenge

Build an AI shopping agent that asks useful follow-up questions and recommends the customer's hidden target product within at most 10 turns.

## What You Receive

- A frozen catalog of 50,000 products from the `Clothing_Shoes_and_Jewelry` category of Amazon Reviews 2023.
- 200 labeled public sessions for local development.
- A weak BM25 starter agent and deterministic local evaluator.
- The Agent API contract and scoring rules.

The organizer keeps 800 additional sessions private for final evaluation.

## Task

For each session, your agent receives an anonymized preference profile and a short customer message. Raw user IDs, review text, timestamps, and purchase history are never disclosed. On every turn the agent may:

- ask a natural clarification question in `message` and identify one requested field in `ask_attribute`;
- return a ranked list of up to 10 catalog `parent_asin` values;
- do both in the same response.

The session ends when the target product appears in the scored Top 10 or after turn 10. Sessions cover Buying, Browsing, Intent Override, and Boundary behavior.

## Download the Catalog

Download `catalog.jsonl.gz` from the GitHub Release attached to this repository, then run:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Verify the downloaded file using the published `SHA256SUMS` file.

## Run the Starter

Python 3.10 or later is recommended. The starter uses only the Python standard library.

```bash
python3 -m evaluator.local_evaluator
```

Edit `starter/agent.py` to implement your system. Do not edit the evaluator or public labels when reporting your local score.
The command writes per-session results and aggregate metrics to `results.json`.

The included weak BM25 starter scores Hit Rate@10 `0.125`, MRR `0.068034`, and
MTTC `9.81` on the released public set. See `docs/baseline_results.json`.

## Agent Interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": "B000..."},
                {"parent_asin": "B001..."}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}
        }
```

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, `other`, or `null`. See `docs/agent_api_contract.json`.

## Technical Metrics

- **Hit Rate@10:** fraction of sessions that find the target within 10 turns.
- **MRR:** mean reciprocal rank of the target; a miss contributes zero.
- **MTTC:** mean first-hit turn; a miss is assigned turn 11.
- **Reported token usage:** prompt and completion tokens returned by the team's model client.

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

Only exact `parent_asin` equality produces a hit. Core metrics are also reported by scenario.

## Model Choice and Cost

Teams may use any legally accessible LLM API or local model. Teams manage their own credentials and must never commit API keys. Model choice, estimated cost, token usage, and latency must be disclosed. Token usage is a feasibility metric, not part of the core technical score. The organizer may reimburse model costs through prizes instead of issuing API keys.

## Files

```text
data/public_set.jsonl             200 labeled development sessions
data/category_index.sqlite3       generated product/category lookup database
data/attribute_index.sqlite3      generated ask_attribute lookup database
data/attribute_index.sqlite3.gz   compressed, GitHub-safe attribute database
docs/competition_specification.md participant rules and evaluation protocol
docs/agent_api_contract.json      machine-readable Agent contract
docs/evaluation_config.json       scoring configuration
docs/baseline_results.json        reproducible weak-starter reference score
starter/agent.py                  editable weak starter
starter/category_index.py         category-index builder and query helpers
starter/attribute_index.py        ask_attribute-index builder and query helpers
evaluator/local_evaluator.py      public-set simulator and scorer
```

Build the category index after downloading the catalog:

```bash
python -m starter.category_index
```

The database preserves the full category tree (so identical category names under
different branches do not collide) and supports both category-to-product and
product-to-category lookups through `starter.category_index.CategoryIndex`.

Build the lookup index for every valid `ask_attribute` value:

```bash
python -m starter.attribute_index
```

Alternatively, restore the committed prebuilt database:

```bash
gzip -dk data/attribute_index.sqlite3.gz
```

`starter.attribute_index.AttributeIndex` provides indexed SQLite lookups for
category, material, color, size, style, brand, budget, feature, use case, and
other constraints. Its `load_hashmap()` method loads any one attribute into
memory when average O(1) exact-key lookup is required.

Every value maps to the catalog's scored product identifier, `parent_asin`.
Filters can be intersected so only product IDs satisfying every selected
constraint remain:

```python
from starter.attribute_index import AttributeIndex

with AttributeIndex() as index:
    candidate_ids = index.filter_products(
        {"material": "cotton", "color": ["black", "blue"]},
        maximum_price=100,
    )
```

Different attributes use AND semantics; multiple values within one attribute
use OR semantics. `maximum_price=100` means strictly below $100.

## Approach 1: Hybrid Factorization Machine

The `fm` branch replaces lexical candidate ordering with a catalog-trained
second-order Factorization Machine plus regularized explicit context–item
crosses. Exact filters remain authoritative, while the model ranks survivors
and supplies the probability distribution for information-gain questions.
Intent Override is handled by atomically replacing obsolete active state.

The evaluator-facing `Agent` loads the committed standard-library-compatible
SQLite artifact automatically. Offline training and complete reproduction
instructions are in [`approach 1/README.md`](<../approach 1/README.md>).

Run the frozen public evaluator wrapper from this directory:

```bash
/usr/local/bin/python3.12 "../approach 1/evaluate_fm.py"
```

The reported official hybrid result is 197/200 correct, MRR `0.658440`, MTTC
`2.200000`, and Technical Score `0.866032`.

## Judging and Submission Policy

- Participant submission requirements: `docs/submission_rules.md`
- Participant release checklist: `docs/participant_release_checklist.md`
- Organizer-only final judging controls: `organizer/JUDGING_RUNBOOK.md`
- Organizer private release checklist: `organizer/private_release_checklist.md`
- Judging day operations SOP: `organizer/JUDGING_DAY_SOP.md`

## Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data.
Sessions are sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the frozen catalog.
