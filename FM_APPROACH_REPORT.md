# Approach 1: Factorization Machine Handover Report

## Executive summary

Approach 1 combines exact attribute filtering with learned candidate ranking.
The system uses a standard second-order Factorization Machine (FM) plus
regularized explicit context–item cross terms. It supports Buying, Browsing,
Boundary, and Intent Override sessions.

The production hybrid FM answered **197 of 200** official public cases
correctly, producing an accuracy/Hit Rate@10 of **0.985000** and a Technical
Score of **0.866032**.

The interaction model performed better than simpler models on the controlled
offline hard-negative benchmark. It did **not**, however, outperform the linear
model in the 200-session official conversational evaluation. These are two
different experiments and their MRR values should not be compared directly.

## Project objective

The agent must identify a hidden target product from a 50,000-product catalog
within at most 10 conversation turns. On each response, it can ask one
attribute question and return up to 10 `parent_asin` recommendations.

The evaluator measures:

- **Accuracy / Hit Rate@10:** whether the target appears in a recommendation
  list;
- **MRR:** how highly the target is ranked;
- **MTTC:** the average turn on which the target is first found, with misses
  charged as turn 11;
- **Efficiency:** `(11 - MTTC) / 10`, bounded to `[0, 1]`;
- **Technical Score:** `0.50 × Accuracy + 0.30 × MRR + 0.20 × Efficiency`.

The maximum possible Technical Score is **1.0**. This requires finding every
target at rank 1 on turn 1.

## Available intent and attributes

The evaluator exposes 10 attributes:

`category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`,
`use_case`, and `other`.

Coarse category is also stored in a separate exact category index used to form
the initial candidate set.

| Scenario | Evidence initially available |
|---|---|
| Buying | Exact coarse category and one key requirement |
| Browsing | Exact coarse category; the user is still exploring |
| Boundary | Initially identical to Browsing |
| Intent Override | Exact coarse category and a provisional preference |

Boundary becomes distinguishable after the first no-preference response asking
the agent to use its judgment. Intent Override becomes definitive when the user
replaces the provisional preference with a new requirement.

## System architecture

### 1. Exact filtering

The category and attribute indexes establish a safety boundary:

1. Retrieve products matching the exact coarse category.
2. Normalize each disclosed value.
3. Retrieve its attribute posting list.
4. Intersect accepted constraints using AND semantics.
5. Commit an intersection only if at least one candidate remains.
6. Roll back an empty intersection and record the value as unindexed.

The learned model only ranks the surviving set. It cannot restore a product
removed by an accepted hard constraint.

### 2. Hybrid FM ranking

Each candidate receives the score:

```text
item linear and item–item base score
+ latent FM context × item interaction
+ regularized explicit context-value × item-value crosses
```

The latent FM learns low-rank relationships between sparse attributes. The
explicit crosses provide directly inspectable weights for relationships such
as:

```text
ctx:use_case=running × item:feature=cushioned
ctx:color=black × item:category=shoes
ctx:material=cotton × item:style=casual
```

The production artifact contains:

| Component | Value |
|---|---:|
| Products | 50,000 |
| Synthetic conversation states | 400,000 |
| Context features | 2,255 |
| Item features | 3,855 |
| Latent dimensions | 16 |
| Explicit crosses | 24,900 |
| Hard negatives per state | 8 |
| Training seed | 2026 |

Training uses pairwise Bayesian Personalized Ranking loss, fixed same-category
hard negatives, Adam optimization, product-level train/validation/test splits,
and validation-based temperature calibration. The public conversation dataset
is not read by the trainer.

### 3. Question selection

The FM scores are converted into a probability distribution over surviving
items. For every remaining attribute, the agent groups candidates by their
predicted answer and calculates response entropy. A separate no-answer bucket
prevents sparse attributes from looking artificially useful.

The policy favors an attribute that divides likely candidates into informative,
answerable response groups. Questions stop after turn 9 or when no more than 10
candidates remain.

### 4. Intent Override

When an override is detected, the agent atomically:

1. restores the original category candidate set;
2. clears obsolete active constraints and unindexed values;
3. changes the scenario state to `intent_override`;
4. applies the replacement constraint;
5. reranks the updated survivors in the same turn.

An incomplete override message does not destroy the existing valid state.

## Evaluation results

### Production hybrid FM

| Cohort | Correct | Accuracy | MRR | MTTC | Efficiency | Technical Score |
|---|---:|---:|---:|---:|---:|---:|
| Buying/Browsing/Boundary | **167/170** | 0.982353 | 0.648394 | 1.835294 | 0.916471 | 0.868989 |
| Official 200 sessions | **197/200** | 0.985000 | 0.658440 | 2.200000 | 0.880000 | 0.866032 |

Official results by scenario:

| Scenario | Correct | Accuracy | MRR | MTTC | Technical Score |
|---|---:|---:|---:|---:|---:|
| Buying | 77/80 | 0.962500 | 0.602029 | 1.700000 | 0.847859 |
| Browsing | 80/80 | 1.000000 | 0.682996 | 1.862500 | 0.887649 |
| Boundary | 10/10 | 1.000000 | 0.742500 | 2.700000 | 0.888750 |
| Intent Override | **30/30** | **1.000000** | 0.715370 | 4.266667 | 0.849278 |

The three missed cases are `public_0028`, `public_0067`, and `public_0083`.
All are Buying sessions. The target remained in the filtered survivor set, so
the failures came from ranking or candidate width rather than destructive
filtering.

## What the interaction experiments show

### Controlled offline benchmark

The offline benchmark ranks each held-out target against eight fixed
same-category hard negatives using synthetic conversation states.

| Model | Offline test MRR | Pairwise accuracy |
|---|---:|---:|
| Linear | 0.421438 | 0.577704 |
| Standard FM | 0.563704 | 0.725913 |
| FM plus explicit crosses | **0.591509** | **0.747067** |

On this controlled benchmark:

- latent FM interactions add `0.142266` MRR over the linear model;
- explicit crosses add another `0.027805` MRR over the standard FM.

This is evidence that catalog attribute interactions contain learnable ranking
signal under the sampled hard-negative protocol.

### Official conversational benchmark

The official benchmark evaluates full conversations, including filtering,
question selection, larger survivor sets, state transitions, and stopping.

| Model | Correct | MRR | MTTC | Technical Score |
|---|---:|---:|---:|---:|
| Linear | **199/200** | **0.672881** | **2.040000** | **0.878564** |
| Standard FM | 197/200 | 0.645863 | 2.180000 | 0.862659 |
| FM plus explicit crosses | 197/200 | 0.658440 | 2.200000 | 0.866032 |

Therefore:

- the hybrid improves official MRR over the standard FM by `0.012577`;
- it does not outperform the linear model on the official 200-session set;
- the linear model currently has the highest official Technical Score;
- paired bootstrap intervals include zero for the important hybrid comparisons,
  so the public sample does not establish a statistically reliable interaction
  advantage.

The offline MRR and official MRR are not directly comparable. The offline
experiment ranks nine candidates in synthetic states; the official experiment
evaluates complete conversations and changing survivor sets.

## Interaction importance and stability

Removing explicit field-pair groups one at a time shows that interactions are
not uniformly useful. For example, `feature × material` improves state-level
MRR by `0.019297` across 306 active public states, while some popularity and
brand interaction groups slightly hurt ranking.

Across seeds 2026–2030, 12,593 of 24,900 explicit crosses—`50.57%`—retain the
same nonzero sign. Individual weights should therefore be treated as
exploratory associations, not causal facts. Field-level ablation and held-out
ranking provide stronger evidence than isolated cross weights.

## Turn-10 candidate analysis

The full-horizon diagnostic continues all sessions through turn 10, including
sessions that would normally stop after an earlier hit.

| Diagnostic | Cases |
|---|---:|
| Exactly 10 survivors on turn 10 | **4** |
| At most 10 survivors on turn 10 | **184** |
| More than 10 survivors on turn 10 | 16 |
| Sessions naturally reaching turn 10 | 3 |

The exactly-10 cases are `public_0038`, `public_0089`, `public_0092`, and
`public_0115`. None of the three sessions naturally reaching turn 10 has
exactly 10 survivors.

## Verification

- All 19 unit tests pass.
- The official frozen evaluator independently reproduces the committed
  197/200 result byte-for-byte.
- The production artifact loads under Python standard-library-only inference.
- `public_set.jsonl` and `local_evaluator.py` retain their original SHA-256
  hashes and were not modified.
- Recommendations are unique, catalog-valid, limited to 10, and drawn only
  from the current survivor set.

## Conclusions and recommended next steps

Approach 1 demonstrates that an FM can learn relationships between observed
attributes and candidate-item attributes. The controlled offline experiment
shows a clear interaction benefit. The official public experiment shows that
this signal has not yet translated into a better complete conversational policy
than the simpler linear ranker.

Recommended next steps:

1. Use the linear model as the current score-maximizing public baseline.
2. Tune FM and explicit-cross regularization on product-held-out validation
   categories rather than individual public sessions.
3. Prune or strongly shrink interaction field groups that hurt held-out
   ranking.
4. Replace exact long-form values with semantic clusters or normalized feature
   families to reduce sparsity.
5. Evaluate question selection separately from ranking so improvements and
   regressions can be attributed correctly.
6. Increase the number and diversity of held-out conversational sessions before
   concluding that the interaction model improves end-to-end performance.

## Handover files

- [Approach implementation and reproduction guide](<approach 1/README.md>)
- [Training code](<approach 1/train_fm.py>)
- [Evaluation code](<approach 1/evaluate_fm.py>)
- [Production model artifact](<approach 1/fm_model.sqlite3>)
- [Official 200-session CSV](<approach 1/results/official_200.csv>)
- [170-session CSV](<approach 1/results/non_override_170.csv>)
- [Model ablation CSV](<approach 1/results/model_ablation.csv>)
- [Bootstrap comparison CSV](<approach 1/results/model_ablation_bootstrap.csv>)
- [Interaction field-pair analysis](<approach 1/results/field_pair_importance.csv>)
- [Turn-10 full-horizon CSV](<approach 1/results/full_horizon_200.csv>)
- [Complete project findings](findings.md)

The implementation is on Git branch `fm`, commit `71a1e2e`.
