# Project Aim and Machine-Learning Research Direction

## Executive recommendation

The next version of this project should not be a single end-to-end neural
network. The strongest practical design for the available data is:

1. **Hard constraint filtering** to guarantee that disclosed requirements are
   respected.
2. **A second-order Factorization Machine (FM) plus regularized explicit
   context–item crosses** to learn sparse relationships among categories,
   disclosed attributes, and candidate-item attributes.
3. **A Bayesian expected-information-gain policy** to choose the next attribute
   question using the FM's current item probabilities and an explicit model of
   whether the user can answer that question.
4. **A knowledge-graph or LightGCN experiment later**, after the simpler model
   proves that graph propagation improves held-out ranking and question value.

This is a hybrid because ranking and questioning are related but different
problems. The ranker estimates which items are likely; the question policy asks
which observation would improve that estimate most.

The most important research correction is that ordinary covariance is not
enough. These catalog attributes are sparse, categorical, multi-valued, and
strongly affected by category frequency. We should measure relationships with
category-conditioned co-occurrence, lift, normalized pointwise mutual
information, conditional mutual information, and learned latent embeddings.

## Project overview

### Competition objective

Build a conversational shopping agent that finds a hidden target product from
a frozen catalog as early and as highly ranked as possible. Each conversation
has at most 10 turns. The agent can ask one structured attribute question per
turn and may return up to 10 `parent_asin` values.

The primary project outcomes are:

- infer the user's product intent from an initially coarse request;
- ask useful follow-up questions rather than follow a fixed script;
- preserve every accepted hard constraint;
- keep narrowing the candidate set while returning a learned Top 10 every turn;
- rank every survivor set using learned item–attribute relationships;
- handle Buying, Browsing, Boundary, and Intent Override behavior;
- improve Hit Rate@10, MRR, MTTC, efficiency, and Technical Score.

### Available data

| Resource | Size or scope |
|---|---:|
| Catalog | 50,000 products |
| Coarse categories | 1,115 |
| Attribute-index rows | 946,564 |
| Public labeled conversations | 200 |
| Private evaluation conversations | 800 |
| Buying / Browsing / Override / Boundary public mix | 80 / 80 / 30 / 10 |
| Maximum turns | 10 |
| Maximum returned recommendations | 10 |

The evaluator supports 10 attributes:

`category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`,
`use_case`, and `other`.

The indexed coverage is highly uneven:

| Attribute | Unique normalized values | Products with an indexed value |
|---|---:|---:|
| Brand | 23,225 | 49,733 |
| Budget | 2,662 | 10,555 |
| Category | 2,693 | 50,000 |
| Color | 4,984 | 21,377 |
| Feature | 37,326 | 47,859 |
| Material | 8,927 | 28,651 |
| Other | 59,763 | 50,000 |
| Size | 3,703 | 3,765 |
| Style | 3,894 | 8,075 |
| Use case | 804 | 812 |

This unevenness is central to question selection. For example, a theoretically
high-entropy `size` or `use_case` question may still be poor if most candidates
have no answerable indexed value.

### Implemented Approach 1 system

The current agent:

- parses the exact coarse category from the initial request;
- maintains a per-session survivor set and known constraints;
- applies exact posting-list intersections with rollback on empty results;
- ranks surviving candidates with a 16-dimensional FM and 24,900 regularized
  explicit context–item crosses;
- chooses unrestricted remaining attributes using FM-weighted response entropy
  with an explicit no-answer bucket;
- returns a deterministic learned Top 10 on every turn;
- atomically replaces stale state when Intent Override is detected;
- uses only standard-library inference from a committed SQLite artifact.

Approach 1 official public-set performance is **197/200 correct**, with MRR
`0.658440`, MTTC `2.20`, and Technical Score `0.866032`. All 30 Intent Override
sessions are correct. Full results and matched ablations are in
[findings.md](findings.md).

### Current limitations

1. Exact long-form feature strings still create severe sparsity.
2. The aggregate user profile is deliberately unused because catalog-only
   self-supervision does not provide defensible profile–target labels.
3. Question selection depends on posterior calibration and simulator-compatible
   reply signatures.
4. The 200 public conversations are too few to safely train a large policy
   network or deep conversational model directly.
5. Held-out catalog ranking benefits strongly from interactions, but the public
   200-session bootstrap does not establish that the hybrid beats the separately
   trained linear ranker end to end.

## The machine-learning problem

Let `I` be the hidden target item, `E_t` be all evidence observed through turn
`t`, and `A_t` be the next attribute that could be asked.

The system needs to estimate:

```text
P(I = item | coarse category, profile, accepted values,
                 rejected values, no-preference answers, turn)
```

It then needs to select a question that maximally improves that posterior:

```text
question* = argmax_a ExpectedUtility(a | E_t)
```

A useful information-gain objective is:

```text
IG(a | E_t)
  = H(I | E_t)
    - sum over replies r of P(r | a, E_t) * H(I | E_t, a, r)
```

The production utility should also account for answerability and cost:

```text
Utility(a | E_t)
  = IG(a | E_t)
    - lambda_no_answer * P(no useful answer | a, E_t)
    - lambda_repeat * Redundancy(a, already_asked)
    - lambda_turn * TurnCost(t)
```

This is a form of active feature acquisition: the system chooses which missing
feature to acquire for the current instance. Work on [active feature acquisition
with generative surrogate models](https://proceedings.mlr.press/v139/li21p.html)
similarly learns dependencies among features to estimate the value of acquiring
new information.

## What “relationship covariance” should mean here

### Why raw covariance is insufficient

For two binary attribute values `x` and `y`, covariance is:

```text
Cov(x, y) = P(x, y) - P(x)P(y)
```

This is useful as a diagnostic, but it has four weaknesses in this project:

- frequent values dominate the statistic;
- most catalog values are rare;
- products can have multiple values for the same attribute;
- category is a confounder—for example, material and style may appear related
  only because both are common in one product category.

### Recommended relationship statistics

Build statistics within each coarse category and at the global level:

```text
support(x, y) = count(items containing both x and y)

lift(x, y) = P(x, y) / (P(x)P(y))

NPMI(x, y) = log(P(x, y)/(P(x)P(y))) / -log(P(x, y))

CMI(x; y | observed evidence or category)
```

Use Bayesian/Laplace smoothing and minimum-support thresholds so one or two
co-occurrences do not look like strong relationships. Conditional mutual
information is specifically designed to find information that remains useful
after accounting for already-selected features; see Fleuret's primary study on
[fast binary feature selection with conditional mutual
information](https://www.jmlr.org/papers/v5/fleuret04a.html).

### Recommended relational representation

Construct a heterogeneous bipartite graph:

```text
item nodes  <---- typed edges ---->  attribute-value nodes

Examples:
B0... --has_material--> cotton
B0... --has_color-----> black
B0... --has_brand-----> Nike
B0... --has_use_case--> running
```

Keep the attribute type on each edge. The string `black` under `color` must not
be treated as the same semantic relation as the word `black` inside a verbose
feature string.

This graph supports three levels of modeling:

1. exact co-occurrence and conditional-probability tables;
2. low-rank item and attribute-value embeddings;
3. later, typed graph message passing.

## Model research and recommendation

### 1. Conditional information gain: required baseline

This is the best next-question baseline and should be implemented before a
neural policy.

At each turn:

1. assign a probability to every surviving item;
2. simulate each possible answer bucket for every remaining attribute;
3. compute expected posterior entropy or expected survivor count;
4. include an explicit `no_preference_or_unindexed` outcome;
5. ask the attribute with the largest expected utility.

With a uniform item prior, this improves the current largest-bucket heuristic.
With learned item probabilities, it asks questions that separate likely items,
not merely questions that split item counts evenly.

The greedy policy has a theoretical connection to [adaptive
submodularity](https://arxiv.org/abs/1003.3967), where adaptive greedy selection
can be competitive with an optimal policy under suitable diminishing-return
conditions. A recent agentic-recommendation preprint also uses entropy for
preference elicitation; treat its findings as directional because it is new and
not yet a mature benchmark ([Tran et al.,
2026](https://arxiv.org/abs/2603.11399)).

### 2. Factorization Machine: recommended first learned model

A second-order FM should be the first learned relationship and ranking model.
[Rendle's Factorization Machines paper](https://doi.org/10.1109/ICDM.2010.127)
introduces factorized pairwise interactions that remain estimable under severe
sparsity and can be computed in linear time in the number of non-zero features.

Represent a candidate scoring example with sparse features such as:

```text
candidate item ID
candidate coarse category
candidate attribute values
observed accepted attribute values
observed rejected attribute values
user profile tags
scenario state
turn number
```

The second-order FM learns latent interactions such as:

```text
running × lightweight
cotton × casual_style
profile:comfort × soft_material
category:sneakers × use_case:gym
observed:black × candidate:black
```

Approach 1 also learns directly inspectable, separately regularized scalar
crosses. This hybrid lets a supported relationship such as
`ctx:use_case=running × item:feature=cushioned` receive a specific weight while
the FM provides a low-rank fallback for sparse or unseen combinations.

Recommended training construction:

1. Treat each catalog item as a positive target.
2. Randomly mask subsets of its category-compatible attribute values to create
   synthetic conversation states.
3. Sample hard negatives from the same coarse category, prioritizing products
   that match some observed values but violate held-out values.
4. Train with a pairwise ranking loss so the positive item outranks each hard
   negative.
5. Calibrate the scores into probabilities on held-out products before using
   them for expected information gain.

The FM should **rank**, not replace hard filtering. A product that violates a
committed hard constraint must never be restored merely because its model score
is high.

### 3. Graph models: valuable second experiment

An item–attribute graph is a natural match for the relationship goal. [KGAT
explicitly models high-order paths between items and attributes](https://arxiv.org/abs/1905.07854)
and uses attention to vary the importance of neighbors. [LightGCN](https://arxiv.org/abs/2002.02126)
shows that simple normalized embedding propagation can outperform heavier graph
convolutions in collaborative recommendation.

For this project, adapt these ideas to an item–attribute bipartite graph and
train by reconstructing held-out item–attribute edges or ranking positive edges
above negative ones.

Graph modeling may learn relationships like:

```text
item -> material -> related items -> use_case
item -> style -> related items -> brand
item -> feature -> related items -> color
```

However, graph models should be an ablation after the FM because:

- there are only 200 labeled conversations;
- extracted feature values are noisy and extremely high-cardinality;
- popular attribute nodes can dominate message passing;
- a graph ranker does not automatically solve next-question selection;
- graph training, negative sampling, and calibration add complexity.

### 4. Conversational policy models: later, when interaction data grows

The [Estimation–Action–Reflection
framework](https://arxiv.org/abs/2002.09102) separates preference estimation,
question/recommend actions, and learning from rejection. That architecture maps
well to this project now that Intent Override state replacement is implemented.

[Conversational Thompson Sampling](https://arxiv.org/abs/2005.12979) unifies
attributes and items as bandit arms and addresses the explore/exploit decision.
It is attractive once real or high-quality simulated interaction rewards are
available.

Do not start with deep reinforcement learning or a contextual bandit for the
current public data. Two hundred labeled sessions do not provide enough policy
coverage, and offline reward optimization could overfit evaluator-specific
templates. First establish a strong model-based information-gain policy; later
compare EAR, Thompson Sampling, or graph-based RL against it.

### Model decision table

| Model | Learns cross-attribute relationships | Selects next question | Ranks items | Data risk | Recommendation |
|---|---|---|---|---|---|
| Conditional counts + CMI | Yes, explicitly | Yes | Basic posterior | Low | Build first |
| Factorization Machine | Yes, latent pairwise | Through posterior + IG | Yes | Low–medium | Primary learned model |
| Low-rank SVD on item–value matrix | Yes, latent | Not by itself | Yes | Low | FM ablation |
| Bayesian network / Chow–Liu tree | Yes, probabilistic | Yes | Yes | Medium with high cardinality | Try on clustered values |
| LightGCN item–attribute graph | Yes, multi-hop | Not by itself | Yes | Medium | Second-stage experiment |
| KGAT | Yes, typed and attentive | Not by itself | Yes | Medium–high | Research ablation |
| EAR / contextual bandit | Learns policy interactions | Yes | Yes | High with 200 sessions | Defer |
| End-to-end deep RL | Potentially | Yes | Yes | Very high | Do not start here |

## Proposed architecture

```text
Initial message + profile
          |
          v
Scenario/category parser
          |
          v
Hard-filter survivor set <---------------------------+
          |                                           |
          v                                           |
FM or graph posterior ranker                          |
P(item | evidence)                                    |
          |                                           |
          v                                           |
Expected-information-gain question policy             |
          |                                           |
          v                                           |
Ask attribute -> normalize answer -> commit/rollback--+
          |
          v
Posterior-ranked Top 10 is returned on every turn
If survivors > 10 and turn < 10: also ask the best next question
If survivors <= 10: return all survivors and stop questioning
```

### Question policy pseudocode

```python
def choose_question(state, posterior):
    best_attribute = None
    best_utility = float("-inf")

    for attribute in state.remaining_attributes:
        outcomes = simulate_reply_outcomes(
            attribute=attribute,
            candidates=state.surviving_candidates,
            posterior=posterior,
            include_no_preference=True,
        )
        expected_entropy = sum(
            outcome.probability * entropy(outcome.item_posterior)
            for outcome in outcomes
        )
        information_gain = entropy(posterior) - expected_entropy
        utility = (
            information_gain
            - NO_ANSWER_COST * outcomes.no_answer_probability
            - REPEAT_COST * redundancy(attribute, state.known_constraints)
        )
        if utility > best_utility:
            best_attribute = attribute
            best_utility = utility

    return best_attribute
```

### Ranking pseudocode

```python
def rank_survivors(state, model):
    return sorted(
        state.surviving_candidates,
        key=lambda item: model.posterior_score(item, state),
        reverse=True,
    )[:10]
```

## Training and evaluation plan

### Phase 1: relationship tables and information gain

Create:

- per-category attribute coverage and answerability tables;
- value-pair support, covariance, lift, NPMI, and smoothed conditional
  probabilities;
- an information-gain question selector with `no_preference` as an outcome;
- diagnostic explanations showing why each question was selected.

Primary experiment:

```text
Current largest-bucket roadmap
vs.
Unrestricted expected information gain
vs.
Expected information gain with answerability penalty
```

### Phase 2: Factorization Machine reranker

Train on masked catalog states with category-matched hard negatives. Compare:

```text
lexical parent_asin order
popularity/rating heuristic
linear logistic ranker
matrix factorization
second-order Factorization Machine
```

Use the FM posterior both for final ranking and as the item distribution in the
question policy.

### Phase 3: graph embedding ablation

Train LightGCN-style and KGAT-style item–attribute embeddings. Test whether they
improve:

- held-out attribute prediction;
- hard-negative ranking within coarse categories;
- calibration of item probabilities;
- next-question information gain;
- long-tail performance.

Do not adopt the graph model unless it improves the full conversational metrics,
not just link-prediction accuracy.

### Phase 4: learned action policy

After generating many diverse, leakage-controlled simulator trajectories, test:

- contextual bandit/Thompson Sampling;
- EAR-style action selection;
- graph-based policy learning.

Keep the deterministic information-gain policy as the safety fallback.

## Data splitting and leakage controls

The public 200 sessions must be treated primarily as evaluation data, not as a
large policy-training set.

Recommended controls:

1. Train self-supervised relation models from catalog structure.
2. Split catalog products by `parent_asin`, stratified by coarse category.
3. Generate masked states only from the training products.
4. Tune on held-out products, not on the 200 public target IDs repeatedly.
5. Report the frozen 170 supported cohort and official 200 cohort only after
   model selection.
6. Never use ground-truth target IDs at inference time.
7. Keep `public_set.jsonl` and `local_evaluator.py` unchanged.

Because the same product supplies both an input's attributes and its synthetic
training label, ordinary random row splitting would leak nearly identical masked
versions of one product across train and validation. Split by product before
generating masks.

## Metrics

### Relationship model

- masked attribute Recall@K and NDCG;
- category-conditioned link-prediction AUC or average precision;
- pairwise ranking accuracy against hard negatives;
- posterior log loss and calibration error;
- performance by head, medium, and long-tail values.

### Question policy

- expected and actual survivors after every turn;
- percentage of sessions reaching `<=10` by turns 5, 7, 9, and 10;
- probability that a question receives a usable answer;
- number of redundant/no-preference questions;
- target-retention rate after every committed filter;
- question distribution by scenario and coarse category.

### End-to-end

- Hit Rate@10, labeled as accuracy;
- MRR;
- MTTC;
- efficiency;
- Technical Score;
- per-scenario metrics;
- a deterministic learned Top 10 on every turn.

The implemented hybrid exceeds the prior Technical Score (`0.866032` versus
`0.832726`) and improves accuracy from 194/200 to 197/200 while reducing MTTC
from 4.58 to 2.20.

## Key experiments and ablations

1. **Question objective:** largest bucket vs entropy vs expected survivors vs
   calibrated information gain.
2. **Answerability:** no penalty vs catalog coverage vs learned reply
   probability.
3. **Relationship measure:** covariance vs lift vs NPMI vs conditional mutual
   information.
4. **Ranker:** lexical vs popularity vs linear vs FM vs graph embedding.
5. **Negative samples:** random catalog vs same-category vs partially matching
   hard negatives.
6. **Evidence semantics:** accepted only vs accepted/rejected/no-preference.
7. **Profile:** ignored vs hand-mapped priors vs learned profile interactions.
8. **Value representation:** exact strings vs normalized clusters vs text
   embeddings plus exact-value features.
9. **Graph depth:** zero-layer matrix factorization vs one/two/three propagation
   layers.
10. **Stopping:** return Top 10 throughout, but stop clarification questions at
    `<=10`, attribute exhaustion, or turn 10.

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Only 200 labeled conversations | Policy overfitting | Self-supervised catalog training; product-level splits |
| Verbose values create near-duplicates | Sparse unreliable relations | Normalize, cluster, retain raw exact key for filtering |
| Category confounding | Spurious attribute correlation | Compute category-conditioned statistics |
| Multi-valued attributes | Outcomes are not disjoint | Simulate actual reply rules instead of assuming one value |
| Missing attribute coverage | High-entropy but unanswerable questions | Explicit no-answer bucket and penalty |
| Popular nodes dominate graph learning | Weak long-tail ranking | Degree normalization and popularity-aware negative sampling |
| Model removes the target | Catastrophic miss | Hard-filter rollback; audit target retention offline |
| Intent Override invalidates evidence | Stale constraints | Track provenance and support retract/replace operations |
| Posterior is poorly calibrated | Wrong information-gain estimates | Temperature/isotonic calibration on held-out products |
| Public-score chasing | Weak private generalization | Freeze selection protocol and limit public evaluations |

## Completed implementation and next research steps

1. Completed product-level train/validation/test splitting and 400,000 synthetic
   evaluator-compatible states.
2. Completed independently trained linear, FM, and hybrid comparisons.
3. Completed posterior calibration, information-gain questions, and all four
   scenario routes including Intent Override.
4. Completed per-cross and per-field ablations plus paired public bootstrap.
5. Next, improve value normalization and test category-conditioned interaction
   shrinkage before adding model complexity.
6. Only then test LightGCN/KGAT embeddings.
7. Defer bandit/RL policy learning until substantially more real trajectories
   exist.

## Research sources

- Steffen Rendle, [Factorization
  Machines](https://doi.org/10.1109/ICDM.2010.127), ICDM 2010.
- François Fleuret, [Fast Binary Feature Selection with Conditional Mutual
  Information](https://www.jmlr.org/papers/v5/fleuret04a.html), JMLR 2004.
- Daniel Golovin and Andreas Krause, [Adaptive Submodularity: Theory and
  Applications in Active Learning and Stochastic
  Optimization](https://arxiv.org/abs/1003.3967), 2010/2011.
- Yang Li and Junier Oliva, [Active Feature Acquisition with Generative
  Surrogate Models](https://proceedings.mlr.press/v139/li21p.html), ICML 2021.
- Wenqiang Lei et al., [Estimation–Action–Reflection: Towards Deep Interaction
  Between Conversational and Recommender Systems](https://arxiv.org/abs/2002.09102),
  WSDM 2020.
- Shijun Li et al., [Seamlessly Unifying Attributes and Items: Conversational
  Recommendation for Cold-Start Users](https://arxiv.org/abs/2005.12979), 2020.
- Xiang Wang et al., [KGAT: Knowledge Graph Attention Network for
  Recommendation](https://arxiv.org/abs/1905.07854), KDD 2019.
- Xiangnan He et al., [LightGCN: Simplifying and Powering Graph Convolution
  Network for Recommendation](https://arxiv.org/abs/2002.02126), SIGIR 2020.
- Shengbo Guo and Scott Sanner, [Real-time Multiattribute Bayesian Preference
  Elicitation with Pairwise Comparison
  Queries](https://proceedings.mlr.press/v9/guo10b.html), AISTATS 2010.
- Dat Tran et al., [Entropy Guided Diversification and Preference Elicitation
  in Agentic Recommendation Systems](https://arxiv.org/abs/2603.11399), 2026
  preprint.

## Final conclusion from Approach 1

Retain **answerability-aware expected information gain plus the hybrid FM
reranker** as a reproducible research system, while keeping the independently
trained linear model as an important end-to-end baseline.

This choice directly addresses all three desired capabilities:

- relationships among items and attributes are learned through factorized
  interactions;
- the next question is selected by its expected reduction in item uncertainty;
- every survivor set is ranked by a learned posterior rather than arbitrary
  identifier order.

On held-out catalog states, test MRR improves from `0.421438` (linear) to
`0.563704` (FM) to `0.591509` (hybrid), with pairwise accuracy improving from
`0.577704` to `0.725913` to `0.747067`. This is strong evidence that interactions
carry predictive signal in the catalog-derived ranking problem. On the much
smaller 200-session public evaluation, the hybrid-versus-linear confidence
interval includes zero, so the honest conclusion is that interaction terms are
important offline but not yet proven to improve the complete conversation
policy on this public sample.
