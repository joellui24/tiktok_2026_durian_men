# Findings

For the full project objective, proposed relational-learning architecture, model
comparison, and experiment roadmap, see [Project Aim and Machine-Learning
Research Direction](PROJECT_AIM_AND_ML_RESEARCH.md).

## Approach 1 FM branch: hybrid learned ranking

Approach 1 now combines exact hard filtering with a second-order Factorization
Machine, 24,900 regularized explicit context–item crosses, and FM-weighted
information-gain questions. It returns a deterministic learned Top 10 on every
turn and implements atomic Intent Override state replacement.

- [Approach documentation](<approach 1/README.md>)
- [Offline trainer](<approach 1/train_fm.py>)
- [Committed portable FM artifact](<approach 1/fm_model.sqlite3>)
- [170-session result CSV](<approach 1/results/non_override_170.csv>)
- [Official 200-session result CSV](<approach 1/results/official_200.csv>)
- [Full-horizon turn trace](<approach 1/results/full_horizon_200.csv>)
- [Matched model ablation](<approach 1/results/model_ablation.csv>)
- [Paired bootstrap intervals](<approach 1/results/model_ablation_bootstrap.csv>)
- [Individual learned-cross audit](<approach 1/cross_weights.csv>)
- [Field-pair importance](<approach 1/results/field_pair_importance.csv>)
- [Cross-weight seed stability](<approach 1/results/cross_seed_stability.csv>)
- [Field-pair seed stability](<approach 1/results/field_pair_seed_stability.csv>)
- [Frozen-input and model checksums](<approach 1/results/artifact_checksums.json>)

### Production hybrid result

| Cohort | Correct | Accuracy (Hit Rate@10) | MRR | MTTC | Efficiency | Technical Score |
|---|---:|---:|---:|---:|---:|---:|
| Buying/Browsing/Boundary | **167 / 170** | 0.982353 | 0.648394 | 1.835294 | 0.916471 | 0.868989 |
| Official public set | **197 / 200** | 0.985000 | 0.658440 | 2.200000 | 0.880000 | 0.866032 |

Official results by scenario:

| Scenario | Correct | Accuracy | MRR | MTTC | Technical Score |
|---|---:|---:|---:|---:|---:|
| Buying | 77 / 80 | 0.962500 | 0.602029 | 1.700000 | 0.847859 |
| Browsing | 80 / 80 | 1.000000 | 0.682996 | 1.862500 | 0.887649 |
| Boundary | 10 / 10 | 1.000000 | 0.742500 | 2.700000 | 0.888750 |
| Intent Override | **30 / 30** | **1.000000** | 0.715370 | 4.266667 | 0.849278 |

The three misses are `public_0028`, `public_0067`, and `public_0083`, all in
Buying. The target remains in the filtered survivor set throughout every
scored state; these are ranking/candidate-width misses rather than destructive
filter failures.

### Are interaction terms important?

Yes for the held-out catalog ranking task. Three models were trained separately
from the same product splits, states, and hard negatives:

| Offline model | Held-out test MRR | Held-out pairwise accuracy |
|---|---:|---:|
| Linear, no interactions | 0.421438 | 0.577704 |
| Standard FM | 0.563704 | 0.725913 |
| FM + explicit crosses | **0.591509** | **0.747067** |

This separates the effects cleanly: latent FM interactions add `0.142266` test
MRR over the linear model, and explicit crosses add another `0.027805` over the
FM.

The 200-session end-to-end result is less conclusive:

| Model | Correct | MRR | MTTC | Technical Score |
|---|---:|---:|---:|---:|
| Linear | **199 / 200** | **0.672881** | **2.040000** | **0.878564** |
| Standard FM | 197 / 200 | 0.645863 | 2.180000 | 0.862659 |
| FM + explicit crosses | 197 / 200 | 0.658440 | 2.200000 | 0.866032 |

The hybrid improves MRR over the FM by `0.012577`, but its paired 95% bootstrap
interval is `[-0.018464, 0.043948]`. Hybrid-versus-linear MRR is `-0.014440`
with interval `[-0.069728, 0.041550]`. These intervals include zero. The honest
finding is therefore:

> Attribute interactions contain strong transferable signal in catalog-held-out
> ranking, but this 200-session public set does not establish that the current
> hybrid policy beats the simpler linear model end to end.

Removing one explicit field-pair group at a time shows that the contribution is
not uniform. `feature×material` improves state-level MRR by `0.019297` across
306 active public states, while several popularity or brand crosses slightly
hurt ranking. The next iteration should shrink or prune harmful groups using
held-out data rather than assuming every interaction family is valuable.

Across seeds 2026–2030, 12,593 of 24,900 individual crosses (`50.57%`) keep the
same nonzero direction. This makes individual weights exploratory rather than
causal or universally stable. Same-field relationships are more reliable—for
example, 17/18 `color×color` crosses retain their sign—but field-level ablation
and held-out ranking remain the stronger evidence.

### Turn-10 candidate counts

The full-horizon diagnostic continues all 200 sessions even after their first
hit so every case has a comparable turn-10 state.

| Diagnostic | Cases |
|---|---:|
| Exactly 10 survivors at turn 10 | **4** |
| At most 10 survivors at turn 10 | **184** |
| More than 10 survivors at turn 10 | 16 |
| Sessions naturally reaching turn 10 | 3 |

The four exactly-10 cases are `public_0038`, `public_0089`, `public_0092`, and
`public_0115`. None of the three sessions that naturally reaches turn 10 has
exactly 10 survivors.

## Previous baseline: filter to at most 10 choices without guessing

The progressive attribute-filtering agent now returns no `parent_asin` values
while more than 10 candidates survive. It continues asking roadmap questions
and recommends products only when the filtered survivor set contains 10 or
fewer choices. This prevents an intermediate Top-10 sample from ending a
session through guessing.

- [Turn-10 analysis code](<approach 1/analyze_turn10.py>)
- [Per-test-case CSV results](<approach 1/turn10_results.csv>)
- [Progressive filtering agent](techjam-conversational-search/starter/agent.py)
- [170-session evaluation report](techjam-conversational-search/results_no_override.json)
- [Official 200-session evaluation report](techjam-conversational-search/results_official.json)

### Evaluation results

| Cohort | Correct | Incorrect | Accuracy (Hit Rate@10) | MRR | MTTC | Efficiency | Technical Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Supported Buying/Browsing/Boundary | **164 / 170** | 6 | 0.964706 | 0.730378 | 4.347059 | 0.665294 | 0.834525 |
| Official public set | **194 / 200** | 6 | 0.970000 | 0.731085 | 4.580000 | 0.642000 | 0.832726 |

Correct answers by scenario:

| Scenario | Correct | Total | Accuracy |
|---|---:|---:|---:|
| Buying | 75 | 80 | 0.937500 |
| Browsing | 79 | 80 | 0.987500 |
| Boundary | 10 | 10 | 1.000000 |
| Intent Override | 30 | 30 | 1.000000 |

In that previous baseline, Intent Override logic was explicitly unsupported.
Its public-set score did not demonstrate implemented override semantics.

The six incorrect official cases are:

| Sample ID | Scenario | Survivors at turn 10 |
|---|---|---:|
| `public_0026` | Buying | 14 |
| `public_0067` | Buying | 11 |
| `public_0083` | Buying | 54 |
| `public_0087` | Browsing | 48 |
| `public_0161` | Buying | 11 |
| `public_0174` | Buying | 14 |

Every incorrect case still had more than 10 candidates after the final turn,
so the no-guessing policy correctly returned no recommendation.

### Comparison with the earlier intermediate-guessing run

| Cohort | Policy | Correct | MRR | Technical Score |
|---|---|---:|---:|---:|
| Supported 170 | Intermediate Top-10 guesses | 169 | 0.566146 | 0.827020 |
| Supported 170 | Recommend only at ≤10 | 164 | 0.730378 | 0.834525 |
| Official 200 | Intermediate Top-10 guesses | 199 | 0.583530 | 0.827359 |
| Official 200 | Recommend only at ≤10 | 194 | 0.731085 | 0.832726 |

Removing intermediate guesses reduced the number of correct answers by five in
each cohort, but increased both MRR and the overall Technical Score.

### Turn-10 survivor analysis

The official evaluator normally stops when the target appears in a returned
recommendation list. To make every test case comparable, the analysis script
also continues all cases through a hypothetical turn 10 and records the full
survivor count.

The supported cohort contains the 170 Buying, Browsing, and Boundary cases. The
official cohort contains all 200 cases, including 30 Intent Override cases.

| Cohort | Cases | Exactly 10 survivors | At most 10 survivors | More than 10 survivors | Responses with 10 recommendations |
|---|---:|---:|---:|---:|---:|
| Supported Buying/Browsing/Boundary | 170 | 1 | 164 | 6 | 1 |
| Official public set | 200 | 2 | 194 | 6 | 2 |

The cases with exactly 10 surviving `parent_asin` values are:

| Sample ID | Scenario | Included in supported 170 |
|---|---|---|
| `public_0064` | Intent Override | No |
| `public_0092` | Browsing | Yes |

Under normal evaluator stopping, 17 sessions reach turn 10: 11 become correct
on turn 10 and six remain incorrect. None of those 17 has exactly 10 survivors
at that turn; the cases with exactly 10 survivors had already narrowed and
ended earlier.

### Reproduce

From the repository root, run:

```bash
python3 "approach 1/analyze_turn10.py"
```

To rerun the complete 170- and 200-session evaluations:

```bash
cd techjam-conversational-search
python3 evaluate_progressive.py
```

The scripts read the frozen public dataset in memory and rewrite the CSV and
JSON reports. They do not modify `public_set.jsonl` or the frozen evaluator.
