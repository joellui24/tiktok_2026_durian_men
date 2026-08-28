# Findings

## Approach 1: Filter to at most 10 choices without guessing

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

Intent Override logic remains explicitly unsupported. Its public-set score does
not demonstrate that override semantics are implemented.

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
