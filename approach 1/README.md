# Approach 1 — Factorization Machine ranking

> **Current status (31 August 2026):** the trajectory-aligned FM redesign is an
> experiment and has **not** been promoted. V3 selects down-weighted ties, 32
> negatives, no model-hard negatives, and dual `OTHER`. E7 is complete at 25k,
> 50k, and 100k across model seeds 2026–2028. The registered plateau rule
> selects 100k. Work stopped after E7 by user direction: v3 E8 tuning and final
> candidate retraining were not run. A later user-authorized exploratory
> official evaluation of the best-validation E7 artifact (100k, model seed
> 2028) scored 195/200, MRR `0.652365`, and Technical Score `0.855410`; it did
> not beat the frozen hybrid incumbent and was not promoted. The runtime still
> loads the frozen incumbent artifact, `fm_model.sqlite3`.

This directory contains two deliberately separate bodies of work:

1. the frozen Linear, independent-FM, and hybrid artifacts used for the
   historical public evaluation and current runtime; and
2. the experimental trajectory-aligned training/evaluation pipeline designed to align FM
   supervision with real conversation states.

The redesign plan is implemented at the data, training, evaluation, and
orchestration layers through E7. The learning curve and plateau decision are
complete, but the user-directed terminal scope is E7. E8, final-candidate,
and promotion-gate stages were not run. The direct E7 official evaluation is
exploratory evidence, not a registered final-candidate or promotion run.
Nothing below `results/redesign/` was copied over a frozen root artifact.

## Frozen runtime artifacts versus experimental redesign

| Scope | Artifacts | Status |
|---|---|---|
| Runtime incumbent | `fm_model.sqlite3` | Frozen hybrid FM plus explicit crosses; loaded by `starter/agent.py` |
| Frozen references | `linear_model.sqlite3`, `fm_only_model.sqlite3` | Read-only baselines for every redesign comparison |
| Legacy training reports | Root-level `*_training_metrics.json`, `cross_weights.csv`, and historical result CSVs | Historical evidence; not current redesign results |
| Historical redesign evidence | `results/redesign/v2/` and `results/redesign/experiments_v2/` | E0, smoke, and superseded set-valued/n16 experiments; not loaded by runtime |
| Current v3 evidence | `results/redesign/experiments_v3_downweight/` | Versioned, resumable down-weight/n32 research outputs; not loaded by runtime |

The experiment runner rejects frozen artifact paths and only writes below its
dedicated versioned output root. `--force` may replace an already generated
experiment run, but it still cannot replace a frozen artifact. Avoid `--force`
unless a specific versioned run is intentionally being regenerated.

The frozen hybrid's official result remains 197/200, MRR `0.658440`, and
Technical Score `0.866032`. The frozen Linear reference remains the strongest
official result at 199/200, MRR `0.672881`, and Technical Score `0.878564`.

## What the redesign changes

The redesign pipeline trains an **independent** FM; it is not a residual correction
over Linear and does not use the Linear score during training. Exact filtering
remains the safety boundary. The FM only orders products that survive the
visible constraints at a given conversation state.

The main implementation is split across:

- [`trajectory_data.py`](trajectory_data.py): deterministic product splits,
  complete trajectory simulation, runtime-compatible transitions/filtering,
  and compact state-specific survivor sets;
- [`fm_training.py`](fm_training.py): train-fitted vocabulary, weighted BPR,
  information-aware supervision, dynamic negatives, and exact held-out
  evaluation;
- [`train_fm.py`](train_fm.py): stable trainer entry point;
- [`evaluate_fm.py`](evaluate_fm.py): exact full-survivor and official
  evaluation, breakdowns, paired comparisons, and trajectory bootstrap;
- [`run_fm_experiments.py`](run_fm_experiments.py): dry-run-by-default,
  resumable E0–E8 orchestration and aggregation; and
- [`conversation_features.py`](../techjam-conversational-search/starter/conversation_features.py):
  shared runtime/training state and `OTHER` feature construction.

## Leakage-safe data protocol

The unit hierarchy is product → trajectory → observed state → positive/negative
pair. A trajectory, rather than each correlated prefix, is the independent
statistical unit.

- `parent_asin` values are split before simulation into category-stratified,
  deterministic 80% train, 10% validation, and 10% internal-test sets. The
  three product sets are asserted disjoint.
- Trajectory seed `2026` and split seed `2026` remain fixed across learning-curve
  sizes and model seeds. The 25k data are an exact prefix of 50k, which is an
  exact prefix of 100k.
- Training uses the 50,000-product catalogue and the aggregate public scenario
  proportions only: 40% Buying, 40% Browsing, 5% Boundary, and 15% Intent
  Override. It never reads public targets, dialogue paths, or outcomes.
- Browsing and Boundary start as `exploring_unknown`. No future reply or true
  scenario label is placed in the observed state.
- Questions are sampled reproducibly among valid informative attributes.
  Filtering uses runtime AND semantics and the same empty-intersection rollback.
  Intent Override clears obsolete evidence, restores the category pool, and
  applies only the replacement evidence.
- States cover early, middle, and late turns. Ten percent of trajectories are
  controlled extensions for late-state coverage. Each retained state has
  trajectory weight `1 / retained_states_in_trajectory`, so every trajectory
  contributes equal total state weight.
- Every state stores its complete survivor set. Negatives can only come from
  that state's product split and survivor set, excluding the target.
- Meaningful `OTHER` replies are dual encoded through shared code: an OTHER
  source marker, inferred typed features when available, and a retained
  normalized value with minimum-support fallback.

The public 200-session set is reserved for the final end-to-end handoff. Using
official results to choose data size, supervision, negative sampling, or FM
hyperparameters would break this boundary.

## Selected v3 training settings

The current learning-curve program fixes the following settings explicitly;
they are not inferred from the runner's generic CLI defaults.

| Setting | Default |
|---|---:|
| Supervision | `downweight_ties` |
| Tie weight | `0.10` |
| Category-only evidence weight | `0.05` |
| Evidence saturation | 3 meaningful constraints |
| Negatives per active state | 32 |
| Negative mixture | 0% model-hard, 50% near-match, 50% random |
| Reproducible hard-negative pre-pool | 128 survivors |
| `OTHER` representation | `dual` |
| Latent dimension | 16 |
| Learning rate | `0.01` |
| Latent / linear L2 | `1e-5` / `1e-5` |
| Maximum epochs / early-stop patience | 60 / 7 validation checks |
| Data / split seed | 2026 / 2026 |
| Learning-curve model seeds | 2026, 2027, 2028 |

The 800-trajectory smoke comparison favored set-valued positives after one
epoch, but the full 25k early-stopped comparison reversed that direction:
down-weighting reached validation MRR `0.451263`, versus `0.432232` for
skip-ties and `0.427706` for set-valued positives. That full-cohort evidence
drives the v3 pivot. N32 then led n16 by `+0.002613` validation MRR, and the
matched three-seed sampler comparison selected no-model-hard with mean
validation MRR `0.461145 ± 0.001312`.

## Evaluation protocol

Model selection uses product-held-out validation states and exact full-survivor
ranking. For each state the evaluator scores every current survivor and records
the exact target rank. The primary aggregate is trajectory-macro, preventing
long conversations from dominating the result.

Reports include MRR, HR@1/5/10, mean and median rank, rank percentile,
candidate width, and pairwise accuracy as a secondary diagnostic. Breakdowns
cover scenario, observed scenario state, early/middle/late turn, survivor-width
band, `OTHER`, and supervision band.

The reported **conditional HR@10** is calculated only for states with more than
10 survivors. `hit_rate_at_10_raw` includes narrow states where success can be
automatic and must not be used as ranking evidence.

Two evaluations remain distinct:

- ranker-only evaluation freezes trajectory states and survivors to isolate
  ordering quality; and
- official evaluation runs complete conversations and includes question
  selection, filtering, MTTC, and Technical Score.

The official handoff compares the frozen Linear model, frozen incumbent FM,
and a third model explicitly labelled `candidate`; occupying the evaluator's
historical hybrid slot does not make the candidate a hybrid.

## E0: frozen full-survivor baseline

E0 evaluates all frozen references on the same corrected-v2 25k validation
states: 2,500 held-out trajectories and 9,757 retained states. Metrics below
are trajectory-macro; HR@10 is conditional on survivor width greater than 10.

| Frozen model | MRR | HR@1 | HR@5 | Conditional HR@10 |
|---|---:|---:|---:|---:|
| Linear | 0.426977 | 0.357537 | 0.493683 | 0.188341 |
| Independent FM | 0.435062 | 0.363853 | 0.502263 | 0.210309 |
| Hybrid FM + crosses | **0.437229** | **0.365596** | **0.502550** | **0.218858** |

The complete report is
[`results/redesign/v2/e0_25k/ranker_only/summary.json`](results/redesign/v2/e0_25k/ranker_only/summary.json).
The accompanying official baseline reproduces the historical results:

| Frozen model | Correct | MRR | MTTC | Technical Score |
|---|---:|---:|---:|---:|
| Linear | **199 / 200** | **0.672881** | 2.040000 | **0.878564** |
| Independent FM | 197 / 200 | 0.645863 | 2.180000 | 0.862659 |
| Hybrid FM + crosses | 197 / 200 | 0.658440 | 2.200000 | 0.866032 |

## Current v3 selection evidence

The full 25k policy comparison, rather than the one-epoch smoke ordering,
selects down-weighted ties. With that policy fixed, the requested negative-count
comparison selects n32 on validation MRR.

| Selection stage | Alternatives | Selected result |
|---|---|---|
| Supervision | set-valued `0.427706`; skip `0.432232`; down-weight `0.451263` validation MRR | `downweight_ties`, tie weight `0.10` |
| Negative count | n8 `0.449345`; n16 `0.451263`; n32 `0.453876` validation MRR | n32 |

The matched three-seed sampler comparison then selects no-model-hard. `±` is
the population standard deviation over model seeds 2026–2028.

| Sampler | Validation MRR | Validation conditional HR@10 | Internal-test MRR | Internal-test conditional HR@10 |
|---|---:|---:|---:|---:|
| Balanced 34/33/33 | 0.459686 ± 0.003903 | 0.243441 | 0.465994 | **0.262985** |
| **No-model-hard 0/50/50** | **0.461145 ± 0.001312** | **0.244621** | **0.467995** | 0.259970 |

No-model-hard improves mean validation MRR by `+0.001459` with lower seed
dispersion. Its mean internal-test conditional HR@10 is `−0.003015` versus
balanced, which remains a disclosed tradeoff. The hard-heavy 75/12.5/12.5
profile has only a seed-2026 exploratory result and is not presented as
three-seed evidence.

Two further checks preserve the approved design:

- E2 balanced-scenario sensitivity is effectively flat on MRR versus public mix
  (`−0.000087` validation, `+0.000081` test) and lower on both conditional-HR@10
  measures, so the public 40/40/5/15 mix is retained.
- E6 legacy versus dual `OTHER` is mixed and tiny: dual changes validation MRR
  by `−0.000455` but test MRR by `+0.000379` and test conditional HR@10 by
  `+0.003307`. Dual remains selected because it implements the approved shared
  train/runtime semantics, not because of a decisive performance win.

Detailed rows and links are in the [execution report](FM_REDESIGN_EXECUTION.md).

## E7: completed three-seed learning curve

All nine E7 rows are terminal and validated. The 25k point uses strict,
digest-bound provenance records for the exact-compatible selected E4 mixture
evidence; the source metrics and models remain in `E4_negative_mixture` and
were not copied into E7 directories. The 50k and 100k rows are direct E7 runs.

| Size / model seed | Selected / completed epochs | Validation MRR | Validation conditional HR@10 | Internal-test MRR | Internal-test conditional HR@10 | Status |
|---|---:|---:|---:|---:|---:|---|
| 25k / 2026 | 48 / 55 | 0.463000 | 0.242064 | 0.465932 | 0.255509 | Complete |
| 25k / 2027 | 45 / 52 | 0.460245 | 0.247887 | 0.467805 | 0.264660 | Complete |
| 25k / 2028 | 38 / 45 | 0.460190 | 0.243910 | 0.470249 | 0.259743 | Complete |
| 50k / 2026 | 15 / 22 | 0.465244 | 0.255151 | 0.465438 | 0.253229 | Complete |
| 50k / 2027 | 13 / 20 | 0.465718 | 0.262378 | 0.464155 | 0.248523 | Complete |
| 50k / 2028 | 24 / 31 | 0.467449 | 0.255937 | 0.465947 | 0.253903 | Complete |
| 100k / 2026 | 25 / 32 | 0.471058 | 0.264343 | 0.470870 | 0.264739 | Complete |
| 100k / 2027 | 18 / 25 | 0.469067 | 0.262654 | 0.470067 | 0.258373 | Complete |
| 100k / 2028 | 27 / 34 | 0.472437 | 0.262811 | 0.470657 | 0.260735 | Complete |

The exact three-seed aggregates below are trajectory-macro means. `±` is the
population standard deviation of validation MRR. Deltas are against the next
smaller completed E7 point.

| Size | Mean validation MRR ± population SD | Mean validation conditional HR@10 | Mean internal-test MRR | Mean internal-test conditional HR@10 | Validation-MRR delta | Test-MRR delta |
|---:|---:|---:|---:|---:|---:|---:|
| 25k | 0.4611451735195582 ± 0.0013117036819392104 | 0.2446205431291301 | 0.4679953319320041 | 0.2599703619580787 | — | — |
| 50k | 0.4661368213075574 ± 0.0009476216686816433 | 0.2578220518067759 | 0.46517985986286964 | 0.25188508820886835 | +0.004991647787999209 | −0.0028154720691344615 |
| **100k** | **0.47085403640704176 ± 0.0013833569250072744** | **0.2632694080808186** | **0.47053107772159003** | **0.26128202194931865** | **+0.004717215099484362** | **+0.005351217858720392** |

From 25k→50k, mean validation conditional HR@10 changes by
`+0.013201508677645829` and mean test conditional HR@10 by
`−0.008085273749210364`. From 50k→100k, the corresponding changes are
`+0.005447356274042692` and `+0.009396933740450308`. Internal-test metrics are
diagnostic and do not determine the data-size selection. Overall, 25k→100k
changes validation MRR, validation conditional HR@10, test MRR, and test
conditional HR@10 by `+0.009708862887483571`, `+0.01864886495168852`,
`+0.0025357457895859303`, and `+0.001311659991239944`, respectively.

The registered rule selects a smaller size only when every subsequent
matched-seed validation-MRR gain is below `0.002`. The mean gains are
`+0.004991647787999209` for 25k→50k and `+0.004717215099484362` for
50k→100k; every matched-seed gain is also above the threshold. E7 therefore
selects **100k trajectories**.

Evidence: [aggregate manifest](<results/redesign/experiments_v3_downweight/aggregation_manifest.json>),
[plateau decision](<results/redesign/experiments_v3_downweight/plateau_decision.json>),
[learning-curve CSV](<results/redesign/experiments_v3_downweight/learning_curve.csv>),
[learning-curve SVG](<results/redesign/experiments_v3_downweight/learning_curve.svg>),
and the 25k provenance records for [seed 2026](<results/redesign/experiments_v3_downweight/E7_learning_curve/025k_public_tseed2026_mseed2026/reused_training_evidence.json>),
[2027](<results/redesign/experiments_v3_downweight/E7_learning_curve/025k_public_tseed2026_mseed2027/reused_training_evidence.json>),
and [2028](<results/redesign/experiments_v3_downweight/E7_learning_curve/025k_public_tseed2026_mseed2028/reused_training_evidence.json>).

## The 97% indistinguishable-pair limitation

Visible-evidence equivalence still makes about 97% of examined
target–competitor candidates ties. V3 does not discard them: it retains them at
weight `0.10`, which materially increases active training coverage while
limiting arbitrary exact-product supervision. The selected 25k seed-2026 run
reports a tie-candidate rate of `0.970159`, active-state rate `0.681432`, and
about 1.302 million sampled pairs per epoch. More trajectories can add pair
mass, but cannot create distinctions absent from the dialogue.

## E7 terminal scope, E8, and promotion status

- **E7:** complete at 25k, 50k, and 100k for seeds 2026–2028; the registered
  plateau decision selects 100k.
- **V3 E8:** **not run**. The completed v2 E8 grid used the superseded
  set-valued/n16/50-25-25 settings and is not v3 tuning evidence.
- **Final candidate retraining:** **not run**. The selected E7 data size does
  not turn any E7 artifact into a final candidate.
- **Exploratory E7 official evaluation:** **complete** for the best-validation
  100k/model-seed-2028 artifact. It scored 195/200, MRR `0.652365`, MTTC
  `2.390000`, and Technical Score `0.855410`, below the frozen hybrid's
  197/200, MRR `0.658440`, MTTC `2.200000`, and Technical Score `0.866032`.
  This was a direct diagnostic, not a registered final-candidate run. Evidence:
  [summary](<results/redesign/experiments_v3_downweight/E7_official_candidate/100k_tseed2026_mseed2028/summary.json>),
  [paired model table](<results/redesign/experiments_v3_downweight/E7_official_candidate/100k_tseed2026_mseed2028/model_ablation.csv>),
  and [paired bootstrap](<results/redesign/experiments_v3_downweight/E7_official_candidate/100k_tseed2026_mseed2028/model_ablation_bootstrap.csv>).
- **Promotion:** **not run and not approved**. Frozen runtime artifacts remain
  unchanged.

Training stopped after E7 by user direction. The later exploratory official
evaluation did not resume E8, final retraining, or promotion.

A dataset or launching manifest is not a completed run. Completion requires a
terminal manifest plus validated metrics, diagnostics, and SQLite artifact.

## Evidence and reproduction boundary

The aggregate artifacts linked above are the terminal E7 record. Every direct
50k/100k `run_manifest.json` records its resolved leaf command, configuration,
source digests, and terminal validation state. The 25k provenance records bind
each E7 target specification to the live, exact-compatible E4 source and its
artifact, metrics, diagnostics, manifests, and training/generation source-file
digests. No model or metric files were copied into the 25k E7 target
directories.

No E8, final-seed, formal candidate-gate, or promotion command is presented as
a completed reproduction step because those stages were not run. The direct
E7 diagnostic is preserved under `E7_official_candidate/`; the frozen runtime
and its root artifacts were not modified.

## Tests and further results

From `techjam-conversational-search/`:

```bash
PYTHONDONTWRITEBYTECODE=1 python3.12 -m unittest discover -s tests -q
```

The redesign tests cover leakage, scenario allocation, disjoint splits,
rollback and override, ten-turn behavior, survivor-safe/reproducible negatives,
trajectory weights, ambiguous pairs, `OTHER` parity, exact full-survivor
metrics, strict adoption/reuse, and machine promotion evidence.

See the repository [findings](../findings.md) for the preserved legacy result
tables and the current redesign status link.
