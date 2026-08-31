# Buying-Linear / Hybrid-Else routed evaluation

Evaluated on the released 200-session set on 31 August 2026. Runtime intent
routing uses only visible messages and session state; `scenario_type` labels
were used afterward solely to audit routing and report scenario metrics.

| Model | Correct / 200 | Accuracy | MRR | Average first-hit turn | Efficiency | Technical score |
|---|---:|---:|---:|---:|---:|---:|
| Routed | **199** | **0.995** | **0.704468** | **2.12** | **0.888** | **0.886440** |
| Frozen Linear | 199 | 0.995 | 0.672881 | 2.04 | 0.896 | 0.878564 |
| Frozen Hybrid | 197 | 0.985 | 0.658440 | 2.20 | 0.880 | 0.866032 |

## Routing audit

| Ground-truth scenario | Observed opening state | Runtime model | Count |
|---|---|---|---:|
| Buying | `buying` | Linear | 80 |
| Browsing | `exploring_unknown` | Hybrid | 80 |
| Boundary | `exploring_unknown` | Hybrid | 10 |
| Intent Override | `provisional_override` | Hybrid | 30 |

There were zero public-set routing errors. Browsing and Boundary intentionally
share the opening state and model; Boundary becomes observable only after the
customer asks the agent to use its judgment.

The complete evaluator output, including per-session rows, is in
`official_200.json`. Both frozen SQLite artifacts were treated as read-only.
