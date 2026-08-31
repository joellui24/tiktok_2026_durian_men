# Linear v3 six-hour sprint

## Decision

Do **not** promote the retrained candidate. It improves the preregistered
ranker-only validation metrics, but regresses the official end-to-end Technical
Score and accuracy. The frozen `approach 1/linear_model.sqlite3` remains the
best linear runtime candidate and was not modified.

## Training protocol

- Variant: item-only linear pairwise ranker
- Trajectory/split seed: 2026 / 2026
- Supervision: `downweight_ties`, tie weight 0.10
- Evidence weighting: category-only 0.05, saturation at 3 constraints
- Negatives: 32 per active state
- Sampler: 0% model-hard, 50% near-match, 50% random
- Negative mode: dynamic current-survivor sampling
- Feature encoding: dual `OTHER`, minimum value support 5
- Learning rate: 0.01
- Maximum epochs / patience: 60 / 7
- Selection metric: product-held-out trajectory-macro full-survivor validation MRR

## 25k L2 sweep

| Linear L2 | Selected epoch | Validation MRR | Validation conditional HR@10 | Test MRR | Test conditional HR@10 |
|---:|---:|---:|---:|---:|---:|
| 1e-5 | 1 | 0.437198 | 0.200581 | 0.438560 | 0.199051 |
| 1e-4 | 1 | 0.437215 | 0.200330 | 0.438752 | 0.199051 |
| **1e-3** | **3** | **0.437851** | 0.199623 | **0.439482** | **0.203027** |

The validation rule selected linear L2 `1e-3` for confirmation.

## 50k three-seed confirmation

| Model seed | Selected epoch | Validation MRR | Validation conditional HR@10 | Test MRR | Test conditional HR@10 |
|---:|---:|---:|---:|---:|---:|
| 2026 | 7 | 0.439090 | 0.202463 | 0.437405 | 0.199325 |
| **2027** | **14** | **0.439768** | 0.202798 | 0.437772 | 0.200957 |
| 2028 | 11 | 0.439105 | **0.205278** | **0.438752** | 0.198638 |
| Mean | - | **0.439321** | **0.203513** | **0.437976** | **0.199640** |

Seed 2027 was selected strictly by validation MRR. Its artifact SHA-256 is
`95b0de6bf0fd5570a52ccbf892f91e2daaab992c8ddeaed4893ed7f14a5195b0`.

## Paired ranker-only evaluation

The selected candidate was replayed on the frozen 25k validation cohort: 2,500
trajectories and 9,757 states.

| Model | MRR | HR@1 | HR@5 | Conditional HR@10 | Mean rank |
|---|---:|---:|---:|---:|---:|
| Frozen linear | 0.426977 | 0.357537 | 0.493683 | 0.188341 | 63.814743 |
| Retrained candidate | **0.439092** | **0.368682** | **0.508446** | **0.210592** | **62.884982** |
| Candidate delta | **+0.012115** | **+0.011145** | **+0.014763** | **+0.022250** | **-0.929760** |

The trajectory-bootstrap 95% interval for the MRR delta is
`[0.005318, 0.018932]`, excluding zero. HR@1, HR@5, conditional HR@10,
rank-percentile, and pairwise-accuracy improvements also exclude zero.

## Official 200-session evaluation

| Model | Correct | Accuracy | MRR | MTTC | Technical Score |
|---|---:|---:|---:|---:|---:|
| Frozen linear | **199 / 200** | **0.995** | **0.672881** | **2.040** | **0.878564** |
| Retrained candidate | 195 / 200 | 0.975 | 0.661450 | 2.460 | 0.856735 |
| Candidate delta | -4 | -0.020 | -0.011431 | +0.420 | -0.021829 |

The candidate misses `public_0002`, `public_0028`, `public_0080`,
`public_0083`, and `public_0096`. Three of the five misses are Intent Override
sessions. Its Intent Override result falls to 27/30 and MRR `0.562354`, which is
the main end-to-end regression. The paired bootstrap interval for Technical
Score is `[-0.045310, -0.000590]`, excluding zero.

## Conclusion

The revised training protocol successfully improves held-out survivor ranking,
but the item-only linear score and its calibrated posterior produce a worse
question/ranking policy in complete conversations, especially after intent
overrides. Keep the frozen linear artifact. A future attempt should isolate
question selection from ranking or add scenario-conditioned linear residuals
before another official evaluation.
