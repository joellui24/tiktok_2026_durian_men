# Approach 1 — Hybrid Factorization Machine

This experiment keeps exact category/attribute filtering as a safety boundary,
then ranks the surviving products with a second-order Factorization Machine plus
regularized explicit context–item cross terms. The same posterior drives an
answerability-aware information-gain question policy.

The runtime supports all four evaluator scenarios: Buying, Browsing, Boundary,
and Intent Override. An override atomically removes obsolete active filters,
restores the original category candidate set, applies the replacement value,
and reranks in the same turn.

Initial routing uses the intent template as follows:

| Scenario | Initially available evidence |
|---|---|
| Buying | Exact coarse category plus one requirement, classified into an evaluator attribute |
| Browsing | Exact coarse category only; the user is still exploring |
| Boundary | Initially identical to Browsing; detected after the first use-your-judgment reply |
| Intent Override | Exact coarse category plus a provisional preference; the later replacement becomes the active constraint |

The evaluator exposes 10 attributes: `category`, `material`, `color`, `size`,
`style`, `brand`, `budget`, `feature`, `use_case`, and `other`. Coarse category
is also indexed separately for exact initial candidate retrieval.

## Data and leakage boundary

- Training examples come only from evaluator-compatible intent cards generated
  from the 50,000-product catalog.
- Products are split by SHA-256 before synthetic states are generated.
- The public 200-session targets and outcomes are never read by the trainer.
- `public_set.jsonl` and `local_evaluator.py` remain frozen.
- Runtime inference uses only Python's standard library.

## Model

The score has three components:

```text
item linear/base score
+ latent FM context × item interaction
+ regularized explicit context-value × item-value crosses
```

The committed hybrid artifact contains 16-dimensional factors and all product
vectors in SQLite. Explicit crosses require at least 20 positive catalog
co-occurrences and 20 same-context hard-negative comparisons. Values with
catalog support below five use a typed rare-value fallback.

The model was trained with BPR, eight fixed same-category hard negatives per
state, Adam at `0.01`, FM L2 `1e-5`, explicit-cross L2 `1e-4`, and seed `2026`.

## Reproduce

Training requires NumPy and SciPy. The checked environment used Python 3.12,
NumPy 2.0.1, and SciPy 1.14.0.

```bash
cd techjam-conversational-search

# Production hybrid model
/usr/local/bin/python3.12 "../approach 1/train_fm.py"

# Independently trained matched ablations
/usr/local/bin/python3.12 "../approach 1/train_fm.py" \
  --variant linear \
  --output "../approach 1/linear_model.sqlite3" \
  --metrics "../approach 1/linear_training_metrics.json" \
  --cross-audit "../approach 1/linear_cross_weights.csv"

/usr/local/bin/python3.12 "../approach 1/train_fm.py" \
  --variant fm \
  --output "../approach 1/fm_only_model.sqlite3" \
  --metrics "../approach 1/fm_only_training_metrics.json" \
  --cross-audit "../approach 1/fm_only_cross_weights.csv"

# Public cohorts, full-horizon trace, matched ablation, and bootstrap
/usr/local/bin/python3.12 "../approach 1/evaluate_fm.py"

# Per-field explicit-cross removal
/usr/local/bin/python3.12 "../approach 1/analyze_interactions.py"

# Tests
PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3.12 -m unittest discover -v
```

`starter/agent.py` automatically loads `approach 1/fm_model.sqlite3`. If the
artifact is missing or belongs to another catalog, it falls back to a stable
catalog-valid ordering instead of failing.

## Main result

The hybrid production agent answers 197 of 200 official public cases correctly
(accuracy/Hit Rate@10 `0.985`), with MRR `0.658440`, MTTC `2.200000`, efficiency
`0.880000`, and Technical Score `0.866032`. It answers all 30 Intent Override
cases correctly.

Held-out catalog ranking improves substantially as interactions are added:

| Model | Test MRR | Pairwise accuracy |
|---|---:|---:|
| Linear | 0.421438 | 0.577704 |
| FM | 0.563704 | 0.725913 |
| FM + explicit crosses | **0.591509** | **0.747067** |

The 200-session public sample does not provide statistically conclusive evidence
that the hybrid beats the independently trained linear ranker: its paired
bootstrap intervals include zero for accuracy, MRR, and Technical Score. The
held-out catalog experiment supports the usefulness of interactions; the public
experiment shows that this does not automatically guarantee a better end-to-end
conversation policy on a small evaluation set.

Five-seed analysis also shows that only 50.57% of individual explicit crosses
retain the same nonzero sign. Interpret individual weights as exploratory;
held-out model ablation and field-group removal are the primary importance
tests.

See the repository [findings](../findings.md) for the complete result tables and
links to every CSV.
