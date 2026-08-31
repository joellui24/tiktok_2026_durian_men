# FM training-set redesign: execution report

**Plan:** `FM_TRAINING_SET_REDESIGN_PLAN.md` (30 August 2026)

**Current evidence root:** [`results/redesign/experiments_v3_downweight`](<results/redesign/experiments_v3_downweight>)

**Status at 31 August 2026:** the v3 policy, negative-count, sampler-mixture, scenario-sensitivity, `OTHER`, and E7 learning-curve stages are complete. All three model seeds are validated at 25k, 50k, and 100k, and the registered plateau rule selects 100k. Training stopped after E7 by user direction. E8, final candidate seeds, and promotion gates were not run. A later user-authorized direct official evaluation of the best-validation E7 artifact was completed as an exploratory diagnostic and did not beat the incumbent.

This report uses trajectory-macro exact full-survivor validation MRR for selection. Internal-test metrics are diagnostic only. Selecting the 100k E7 data point does not designate an E7 artifact as a final candidate. The 100k/model-seed-2028 E7 artifact was evaluated directly on the official public 200 sessions after the E7 stop; it scored 195/200 and was not promoted. No promotion was attempted, and the runtime continues to load the unchanged frozen incumbent artifact.

## 1. Implementation coverage

| Approved design decision | Implemented mechanism | Evidence |
| --- | --- | --- |
| Complete conversational trajectories | Deterministic state-by-state simulation with target, visible constraints, pending question, survivor snapshot, scenario state, turn bucket, and intent epoch | [`trajectory_data.py`](trajectory_data.py), [`test_redesign_invariants.py`](<../techjam-conversational-search/tests/test_redesign_invariants.py>) |
| Browsing/Boundary begin unknown | Both begin as `exploring_unknown`; observable state excludes the pending reply and future scenario label | Future-label and pending-reply invariants in [`test_redesign_invariants.py`](<../techjam-conversational-search/tests/test_redesign_invariants.py>) |
| Public 40/40/5/15 mix and product-disjoint 80/10/10 splits | Exact deterministic scenario allocation; products split before trajectory simulation; per-split hashes recorded | [25k](<results/redesign/datasets/public_025k_manifest.json>), [50k](<results/redesign/datasets/public_050k_manifest.json>), [100k](<results/redesign/datasets/public_100k_manifest.json>) |
| Survivor-set dynamic negatives | State/epoch deterministic sampler; target and filtered products excluded; pre-pool 128; mixture is configurable | [`fm_training.py`](fm_training.py), [selected 25k negative audit](<results/redesign/experiments_v3_downweight/E4_negative_mixture/mixture_no_model_hard_n32_25k_tseed2026_mseed2026/negative_audit.csv>) |
| Full-survivor validation/test | Exact target rank over every current survivor; trajectory-macro MRR is primary; conditional HR@10 excludes states with at most 10 survivors | [`evaluate_fm.py`](evaluate_fm.py), [E0 summary](<results/redesign/v2/e0_25k/ranker_only/summary.json>) |
| Early/middle/late coverage and equal trajectory influence | All turn buckets retained, including controlled extensions; each trajectory's retained-state weights sum to 1.0 | Dataset manifests above |
| Information-aware supervision | `skip_ties`, `downweight_ties`, and `set_valued_positives`; v3 selects down-weighting from a full 25k comparison | Section 3 and [`fm_training.py`](fm_training.py) |
| Dual `OTHER` encoding | Shared train/runtime normalization emits answer source, inferred typed features, and a support-gated normalized value | [`conversation_features.py`](<../techjam-conversational-search/starter/conversation_features.py>), [`test_conversation_features.py`](<../techjam-conversational-search/tests/test_conversation_features.py>) |
| Nested 25k/50k/100k learning curve | Fixed catalog/configuration and trajectory/split seeds; three model seeds required at every size; strict digest-bound reuse of exact-compatible 25k evidence | [`run_fm_experiments.py`](run_fm_experiments.py), [`test_experiment_runner.py`](<../techjam-conversational-search/tests/test_experiment_runner.py>) |
| Independent FM and artifact isolation | Every redesign run uses `--variant fm`; generated artifacts stay under a versioned result root and never replace frozen scorers | [selected 25k manifest](<results/redesign/experiments_v3_downweight/E4_negative_mixture/mixture_no_model_hard_n32_25k_tseed2026_mseed2026/run_manifest.json>) |

## 2. E0: frozen full-survivor baseline

The historical E0 cohort contains 2,500 validation trajectories and 9,757 retained validation states. These trajectory-macro metrics remain the like-for-like frozen references.

| Frozen model | MRR | HR@1 | HR@5 | Conditional HR@10 | Mean rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear | 0.426977 | 0.357537 | 0.493683 | 0.188341 | 63.815 |
| Independent FM | 0.435062 | 0.363853 | 0.502263 | 0.210309 | 61.691 |
| FM + explicit crosses | **0.437229** | **0.365596** | **0.502550** | **0.218858** | **61.532** |

Evidence: [ranker-only summary](<results/redesign/v2/e0_25k/ranker_only/summary.json>) and [paired trajectory bootstrap](<results/redesign/v2/e0_25k/ranker_only/paired_trajectory_bootstrap.csv>). The same E0 run reproduces the frozen official references—Linear 199/200 and MRR 0.672881; independent FM 197/200 and MRR 0.645863; Hybrid 197/200 and MRR 0.658440. They are baselines, not v3 candidate results.

## 3. Supervision evidence and the down-weight pivot

The 800-trajectory smoke test originally favored set-valued positives after one epoch. That result remains useful as a pipeline check, but the full 25k, early-stopped comparison is the selection evidence. All rows below use the same public cohort, model seed 2026, 16 dynamic survivor negatives, 50/25/25 sampling, and dual `OTHER` encoding.

| Full 25k policy | Selected / completed epochs | Validation MRR | Validation conditional HR@10 | Test MRR | Test conditional HR@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Set-valued positives | 9 / 16 | 0.427706 | 0.185953 | 0.437650 | 0.195549 |
| Skip ties | 9 / 16 | 0.432232 | 0.196339 | 0.441073 | 0.196160 |
| **Down-weight ties** | **12 / 19** | **0.451263** | **0.236057** | **0.459161** | **0.250530** |

Down-weighting improves validation MRR by `+0.023557` and validation conditional HR@10 by `+0.050104` over the set-valued row. The v3 program therefore pivots to `downweight_ties` with tie weight `0.10`; the smoke winner is not carried forward.

Evidence: [set-valued](<results/redesign/experiments_v2/E4_negatives/count_n16_25k_tseed2026_mseed2026/metrics.json>), [skip](<results/redesign/experiments_v2/E5_supervision/skip_ties_25k_tseed2026_mseed2026/metrics.json>), and [down-weight](<results/redesign/experiments_v2/E5_supervision/downweight_ties_25k_tseed2026_mseed2026/metrics.json>). These earlier-root outputs are exact inputs to the v3 decision; their embedded configurations and terminal manifests, rather than the directory label, establish comparability.

## 4. Versioned datasets, manifests, and adoption safety

| Complete trajectories | Retained states | Buying / Browsing / Boundary / Override | Early / Middle / Late states | Weight error | Manifest |
| ---: | ---: | --- | --- | ---: | --- |
| 25,000 | 97,798 | 10,000 / 10,000 / 1,250 / 3,750 | 65,742 / 24,724 / 7,332 | 0.0 | [25k](<results/redesign/datasets/public_025k_manifest.json>) |
| 50,000 | 195,641 | 20,000 / 20,000 / 2,500 / 7,500 | 131,480 / 49,611 / 14,550 | 0.0 | [50k](<results/redesign/datasets/public_050k_manifest.json>) |
| 100,000 | 390,707 | 40,000 / 40,000 / 5,000 / 15,000 | 262,684 / 99,200 / 28,823 | 0.0 | [100k](<results/redesign/datasets/public_100k_manifest.json>) |

Completed v3 rows have terminal manifests, metrics, a dataset manifest, required audits, and a validated SQLite artifact. Existing outputs can be adopted or reused only when their exact command/configuration, catalog and dataset hashes, model metadata, diagnostics, SQLite integrity/schema, and artifact digest validate. A launching manifest or dataset manifest alone is not a completed result.

Representative evidence: [adopted 25k seed 2026](<results/redesign/experiments_v3_downweight/E4_negative_mixture/mixture_no_model_hard_n32_25k_tseed2026_mseed2026/run_manifest.json>) and [completed 50k seed 2026](<results/redesign/experiments_v3_downweight/E7_learning_curve/050k_public_tseed2026_mseed2026/run_manifest.json>).

## 5. V3 sampler/configuration selection

### E4 requested negatives per active state

All rows use down-weighted ties, the 50/25/25 sampler, dual `OTHER`, and model seed 2026.

| Negatives | Selected / completed epochs | Validation MRR | Validation conditional HR@10 | Test MRR | Test conditional HR@10 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 16 / 23 | 0.449345 | 0.221903 | 0.457280 | 0.244693 |
| 16 | 12 / 19 | 0.451263 | **0.236057** | 0.459161 | 0.250530 |
| **32** | **11 / 18** | **0.453876** | 0.225970 | **0.460040** | **0.256190** |

The preregistered primary metric selects 32 negatives. Its validation MRR is `+0.002613` over n16 and `+0.004530` over n8; the mixed validation-HR@10 movement is retained as a diagnostic rather than used to overturn the primary-metric decision.

Evidence: [n8](<results/redesign/experiments_v3_downweight/E4_negatives/count_n08_25k_tseed2026_mseed2026/metrics.json>), [n16](<results/redesign/experiments_v2/E5_supervision/downweight_ties_25k_tseed2026_mseed2026/metrics.json>), and [n32](<results/redesign/experiments_v3_downweight/E4_negatives/count_n32_25k_tseed2026_mseed2026/metrics.json>).

### E4 negative-mixture comparison

The matched three-seed decision compares the balanced sampler against no-model-hard. The hard-heavy profile was run only as a seed-2026 exploratory check and is not represented as three-seed evidence.

| Sampler (hard / near / random) | Val MRR seeds 2026 / 2027 / 2028 | Mean validation MRR ± population SD | Mean validation conditional HR@10 | Mean test MRR | Mean test conditional HR@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| Hard-heavy (0.75 / 0.125 / 0.125), exploratory only | 0.455653 / — / — | — | — | — | — |
| Balanced (0.34 / 0.33 / 0.33) | 0.463273 / 0.454259 / 0.461525 | 0.459686 ± 0.003903 | 0.243441 | 0.465994 | **0.262985** |
| **No-model-hard (0 / 0.50 / 0.50)** | **0.463000 / 0.460245 / 0.460190** | **0.461145 ± 0.001312** | **0.244621** | **0.467995** | 0.259970 |

No-model-hard is selected: mean validation MRR is `+0.001459` above balanced with lower seed dispersion, and mean test MRR is `+0.002001`. Its mean test conditional HR@10 is `−0.003015` versus balanced, a small disclosed tradeoff. This establishes the v3 mixture as 0% model-hard, 50% near-match, and 50% random.

Evidence: all terminal metrics/manifests in [`E4_negative_mixture`](<results/redesign/experiments_v3_downweight/E4_negative_mixture>).

### E2 scenario-distribution sensitivity

| 25k mix, seed 2026 | Validation MRR | Validation conditional HR@10 | Test MRR | Test conditional HR@10 |
| --- | ---: | ---: | ---: | ---: |
| Public 40/40/5/15 | **0.463000** | **0.242064** | 0.465932 | **0.255509** |
| Balanced 25/25/25/25 | 0.462913 | 0.239205 | **0.466013** | 0.236577 |

Balanced changes validation MRR by only `−0.000087` and test MRR by `+0.000081`, while both conditional-HR@10 values are lower. The main sequence retains the approved public scenario proportions.

Evidence: [public comparator](<results/redesign/experiments_v3_downweight/E4_negative_mixture/mixture_no_model_hard_n32_25k_tseed2026_mseed2026/metrics.json>) and [balanced sensitivity](<results/redesign/experiments_v3_downweight/E2_sensitivity/balanced_25k_tseed2026_mseed2026/metrics.json>).

### E6 legacy versus dual `OTHER`

| Encoding, seed 2026 | Validation MRR | Validation conditional HR@10 | Test MRR | Test conditional HR@10 |
| --- | ---: | ---: | ---: | ---: |
| Legacy | **0.463455** | 0.240870 | 0.465553 | 0.252202 |
| Dual | 0.463000 | **0.242064** | **0.465932** | **0.255509** |

The result is mixed and tiny: dual changes validation MRR by `−0.000455`, validation conditional HR@10 by `+0.001194`, test MRR by `+0.000379`, and test conditional HR@10 by `+0.003307`. Dual is retained because it implements the approved train/runtime semantics, not because this single-seed comparison is a decisive performance win.

Evidence: [legacy](<results/redesign/experiments_v3_downweight/E6_other_encoding/other_legacy_25k_tseed2026_mseed2026/metrics.json>) and [dual](<results/redesign/experiments_v3_downweight/E4_negative_mixture/mixture_no_model_hard_n32_25k_tseed2026_mseed2026/metrics.json>).

## 6. E7 learning curve: terminal three-seed result

Every E7 row uses down-weighted ties, n32, no-model-hard sampling, dual `OTHER`, dimension 16, learning rate 0.01, latent/linear L2 `1e-5`, trajectory/split seed 2026, maximum 60 epochs, and patience 7.

All nine E7 rows are terminal and validated. The 25k point is represented through exact-compatible, digest-bound reuse of the selected E4 mixture evidence; metrics and models remain in their source directories rather than being copied. The 50k and 100k rows are direct E7 runs.

| Size / seed | Selected / completed epochs | Validation MRR | Validation conditional HR@10 | Test MRR | Test conditional HR@10 | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 25k / 2026 | 48 / 55 | 0.463000 | 0.242064 | 0.465932 | 0.255509 | Complete |
| 25k / 2027 | 45 / 52 | 0.460245 | 0.247887 | 0.467805 | 0.264660 | Complete |
| 25k / 2028 | 38 / 45 | 0.460190 | 0.243910 | 0.470249 | 0.259743 | Complete |
| 50k / 2026 | 15 / 22 | 0.465244 | 0.255151 | 0.465438 | 0.253229 | Complete |
| 50k / 2027 | 13 / 20 | 0.465718 | 0.262378 | 0.464155 | 0.248523 | Complete |
| 50k / 2028 | 24 / 31 | 0.467449 | 0.255937 | 0.465947 | 0.253903 | Complete |
| 100k / 2026 | 25 / 32 | 0.471058 | 0.264343 | 0.470870 | 0.264739 | Complete |
| 100k / 2027 | 18 / 25 | 0.469067 | 0.262654 | 0.470067 | 0.258373 | Complete |
| 100k / 2028 | 27 / 34 | 0.472437 | 0.262811 | 0.470657 | 0.260735 | Complete |

Exact trajectory-macro three-seed means are below. `±` is population standard deviation over validation MRR; each delta is relative to the next smaller E7 point.

| Size | Mean validation MRR ± population SD | Mean validation conditional HR@10 | Mean test MRR | Mean test conditional HR@10 | Validation-MRR delta | Test-MRR delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25k | 0.4611451735195582 ± 0.0013117036819392104 | 0.2446205431291301 | 0.4679953319320041 | 0.2599703619580787 | — | — |
| 50k | 0.4661368213075574 ± 0.0009476216686816433 | 0.2578220518067759 | 0.46517985986286964 | 0.25188508820886835 | +0.004991647787999209 | −0.0028154720691344615 |
| **100k** | **0.47085403640704176 ± 0.0013833569250072744** | **0.2632694080808186** | **0.47053107772159003** | **0.26128202194931865** | **+0.004717215099484362** | **+0.005351217858720392** |

From 25k→50k, validation conditional HR@10 changes by `+0.013201508677645829` and test conditional HR@10 by `−0.008085273749210364`. From 50k→100k, the corresponding changes are `+0.005447356274042692` and `+0.009396933740450308`. Overall, 25k→100k changes validation MRR, validation conditional HR@10, test MRR, and test conditional HR@10 by `+0.009708862887483571`, `+0.01864886495168852`, `+0.0025357457895859303`, and `+0.001311659991239944`, respectively. Test metrics remain diagnostic rather than selection evidence.

The registered rule selects a smaller dataset only when every subsequent matched-seed validation-MRR gain is below `0.002`. Mean validation MRR rises by `+0.004991647787999209` from 25k→50k and by `+0.004717215099484362` from 50k→100k. The matched-seed gains are respectively `+0.0022440098283588172`, `+0.005472326192820953`, `+0.007258607342817747`, then `+0.005813940559396602`, `+0.003349291149675415`, and `+0.00498841358938118`; all exceed the threshold. The machine decision therefore selects **100k trajectories**.

Evidence: [aggregate manifest](<results/redesign/experiments_v3_downweight/aggregation_manifest.json>), [plateau decision](<results/redesign/experiments_v3_downweight/plateau_decision.json>), [learning-curve CSV](<results/redesign/experiments_v3_downweight/learning_curve.csv>), [learning-curve SVG](<results/redesign/experiments_v3_downweight/learning_curve.svg>), [direct 50k/100k runs](<results/redesign/experiments_v3_downweight/E7_learning_curve>), and reused 25k provenance for [seed 2026](<results/redesign/experiments_v3_downweight/E7_learning_curve/025k_public_tseed2026_mseed2026/reused_training_evidence.json>), [2027](<results/redesign/experiments_v3_downweight/E7_learning_curve/025k_public_tseed2026_mseed2027/reused_training_evidence.json>), and [2028](<results/redesign/experiments_v3_downweight/E7_learning_curve/025k_public_tseed2026_mseed2028/reused_training_evidence.json>).

## 7. User-directed stop after E7

The E7 learning curve and plateau decision are the terminal scope of this
execution. Work stopped after E7 by user direction after the machine decision
selected 100k trajectories.

- **V3 E8 tuning was not run.** The completed nine-row E8 table under
  `experiments_v2` belongs to the superseded set-valued/n16/50-25-25 program;
  it is historical evidence, not a v3 result.
- **Final candidate retraining was not run.** No E7 training seed or selected
  E7 artifact is designated a final candidate.
- **An exploratory E7 official evaluation was run later.** The best-validation
  100k/model-seed-2028 artifact scored 195/200, MRR `0.652365`, MTTC
  `2.390000`, and Technical Score `0.855410`. It did not beat the frozen hybrid
  incumbent's 197/200, MRR `0.658440`, MTTC `2.200000`, and Technical Score
  `0.866032`. Because E8 and final-candidate retraining were skipped, this is a
  direct diagnostic rather than registered promotion evidence.
- **Promotion was not run or approved.** The frozen runtime and all frozen
  root artifacts remain unchanged.

## 8. Artifact safety and evidence hardening

- Frozen Linear, FM, and Hybrid references remain inputs only. The runner refuses frozen paths as generated outputs.
- Valid-result adoption requires exact manifest/spec identity plus embedded configuration and catalog hashes, complete diagnostics, and a minimally valid SQLite model. A zero-return subprocess with incomplete outputs fails execution.
- E7 reuse is source-to-target exact: it verifies the source terminal manifest and output digests, permits differences only in destination arguments, rejects partial direct target output, and gives a complete direct E7 result precedence over reused evidence.
- Promotion evidence is machine-validated and bound to an exact cohort and artifact digests; arbitrary self-attested text cannot satisfy a gate.
- A successful official run must generate paired session evidence and a post-official decision artifact. It never automatically overwrites a frozen scorer.

Implementation: [`run_fm_experiments.py`](run_fm_experiments.py), [`evaluate_fm.py`](evaluate_fm.py), [`fm_training.py`](fm_training.py), and [`trajectory_data.py`](trajectory_data.py). Tests: [`tests`](<../techjam-conversational-search/tests>).

## 9. Terminal evidence and reproduction boundary

The [aggregate manifest](<results/redesign/experiments_v3_downweight/aggregation_manifest.json>) records nine complete learning-curve rows and three complete sizes. The [plateau decision](<results/redesign/experiments_v3_downweight/plateau_decision.json>) binds the matched configuration and seed cohorts to the 100k selection; the [CSV](<results/redesign/experiments_v3_downweight/learning_curve.csv>) and [SVG](<results/redesign/experiments_v3_downweight/learning_curve.svg>) are its tabular and visual views.

Each direct 50k/100k terminal manifest records the exact resolved leaf argv and working directory. Each 25k reuse record binds its E7 target to an exact E4 source, including the model, metrics, diagnostics, dataset and runner manifests, catalog, and training/generation source-file SHA-256 digests. This is provenance reuse, not artifact copying or retraining.

No host-specific launch guidance is retained here because training stopped after E7. E8, final-seed, formal candidate-gate, and promotion commands were not run. The later direct E7 official diagnostic is preserved separately and is not represented as promotion evidence.

## 10. Not run: final three-seed candidate retraining

**NOT RUN — no E8 tuning winner or E7 artifact is a final candidate.**

| Seed | Run manifest | Artifact SHA-256 | Validation MRR | Incumbent delta | HR@10 gate |
| ---: | --- | --- | ---: | ---: | --- |
| 2026 | **NOT RUN** | — | — | — | — |
| 2027 | **NOT RUN** | — | — | — | — |
| 2028 | **NOT RUN** | — | — | — | — |

## 11. Not run: machine promotion gates

| Gate | Status | Evidence |
| --- | --- | --- |
| Mean full-survivor validation MRR improves over frozen FM | **NOT RUN** | No final-candidate cohort |
| No material conditional-HR@10 regression by scenario or turn bucket | **NOT RUN** | No final-candidate cohort |
| Filtering, rollback, intent override, and turn-10 correctness | **NOT RUN** | No candidate gate bundle |
| Improvement direction is consistent across all three final seeds | **NOT RUN** | Final seeds were not trained |

No promotion evaluation or approval is claimed. Frozen runtime artifacts remain unchanged.

## 12. Exploratory E7 official 200-session result

The best-validation existing E7 artifact (100k trajectories, model seed 2028,
selected epoch 27, SHA-256
`3a742a4443d89690df2d062977baed352edd16f64c59d3aaea1a6400b2e48687`)
was evaluated directly without retraining. It was not designated or promoted
as the formal final candidate.

| Model | Correct | MRR | MTTC | Technical Score | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Frozen Linear | **199/200** | **0.672881** | **2.040000** | **0.878564** | Strongest public result |
| Frozen hybrid incumbent | 197/200 | 0.658440 | 2.200000 | 0.866032 | Runtime remains unchanged |
| E7 100k, seed 2028 | 195/200 | 0.652365 | 2.390000 | 0.855410 | Do not promote |

The E7 misses are `public_0017`, `public_0028`, `public_0083`,
`public_0087`, and `public_0174`. Intent Override remains 30/30, while the
losses are four Buying sessions and one Browsing session. Evidence: [summary](<results/redesign/experiments_v3_downweight/E7_official_candidate/100k_tseed2026_mseed2028/summary.json>), [official sessions](<results/redesign/experiments_v3_downweight/E7_official_candidate/100k_tseed2026_mseed2028/official_200.json>), [paired model table](<results/redesign/experiments_v3_downweight/E7_official_candidate/100k_tseed2026_mseed2028/model_ablation.csv>), and [paired bootstrap](<results/redesign/experiments_v3_downweight/E7_official_candidate/100k_tseed2026_mseed2028/model_ablation_bootstrap.csv>).
