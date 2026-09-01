# Free-form conversational retrieval report

## Selected production design

The selected free-form path is:

```text
official template detector
  -> deterministic hard-constraint parser
  -> fail-closed recognized-hard-constraint filtering
  -> BGE-small dense retrieval over the surviving products
  -> reciprocal-rank fusion with the existing routed Linear/FM ranker
  -> clarification policy
  -> conversation state
```

The SQLite FTS5 lexical index is retained as an automatic fallback when the
dense dependency, local ONNX model, or embedding artifact is
unavailable. It is not added to the normal fusion because that variant scored
worse on the frozen development split.

The official formatted-query branch exits before free-form query construction,
optional imports, model loading, or retrieval fusion. Its result JSON before
and after this implementation is byte-for-byte identical.

## Semantics and safety

Deterministic code remains authoritative for:

- numeric maximum prices, including strict versus inclusive ceilings;
- exact catalogue categories, brands, colors, and materials;
- `not`, `except`, `avoid`, and `without` exclusions;
- same-field `or` alternatives for exact hard fields;
- `instead` replacement;
- preference removal and budget replacement;
- conversation category changes and non-destructive category refinement;
- short clarification answers interpreted against the requested field.

Recognized free-form hard constraints fail closed. If the deterministic parser
recognizes an incompatible category, brand, color, material, exclusion, OR, or
numeric ceiling, the agent returns no contradictory recommendations and asks
for clarification. Dense and lexical rankers receive only identifiers that
survived those filters, so neither can restore a removed product. This guarantee
does not apply to a hard phrase the parser failed to recognize. Raw turns with
parsed logical operators are not sent to the embedding model; retrieval instead
uses the current positive structured state.

Soft concepts such as `breathable`, `comfortable`, `summer`, `travel`, and
`beach holiday` are allowed to influence ordering without becoming invented
hard constraints. OR alternatives for feature, use case, and size likewise stay
soft when the parent-product catalogue cannot verify them reliably.

## Model and artifacts

- Model: `BAAI/bge-small-en-v1.5`, FastEmbed snapshot
  `52398278842ec682c6f32300af41344b1c0b0bb2`, quantized ONNX
- License: MIT
- Embedding dimension: 384
- Catalogue document: title, category, store/brand, leading features, and a
  bounded description; maximum 300 characters
- Search: exact cosine similarity over the hard-filtered subset
- Network/API calls at inference: none
- Model/API cost: USD 0
- Reported LLM tokens: 0

The loader validates unique identifiers, row/vector alignment, metadata count,
dimension and model name, finite normalized vectors, and exact identifier-set
agreement with the live frozen catalogue. A failure activates lexical fallback.

Required free-form assets:

| File | Bytes | SHA-256 |
|---|---:|---|
| `data/semantic_embeddings.npz` | 35,995,508 | `79F78C70F5F37385034BCF03E6A3F1B73E1EF6ACDBE5D2D8D7A7A24B25A0FD14` |
| `data/lexical_index.sqlite3` | 28,672,000 | `74D90D0A5F273D13B4D0CDE365001C82072C808170349D8AB32EB9EB1CEE4199` |
| `models/bge-small-en-v1.5/model_optimized.onnx` | 66,465,124 | `51F1BD0ADDD6E859E42C2C8021A5E5461385BB676A649F4B269AA445449F2431` |

## Frozen benchmark method

`tests/free_form_retrieval_benchmark.py` contains 60 natural queries across
paraphrase, category-synonym, hard-buying, browsing, vague-use-case, and
feature groups. Proxy relevance is derived independently from expected fields
plus frozen exact catalogue/lexical evidence; it never uses parser output,
retrieved products, or model scores. This proxy can miss valid synonyms and can
spuriously match generic evidence words. Odd case IDs form the 30-query
development split and even IDs form the 30-query confirmation split.

Seven queries have no grade-2-or-higher proxy-relevant item and are retained for
safety reporting but excluded from relevance metrics; some still have eligible
grade-1 products. Size is reported as unverifiable rather than treated as a hard
violation because parent-ASIN metadata does not expose a dependable normalized
variant size. Hard-violation rate is item-weighted. Raw pairwise category
diversity is secondary because irrelevant variety can inflate it.

Development was used to select one of four practical architectures:

| Development variant | nDCG@10 | Success@10 | First-relevant MRR | P@10 | Hard violation rate |
|---|---:|---:|---:|---:|---:|
| Deterministic + existing ranker | 0.4347 | 0.7037 | 0.4418 | 0.2778 | 0.0000 |
| Existing ranker + lexical | 0.4972 | 0.8148 | 0.4997 | 0.3519 | 0.0000 |
| Existing ranker + dense (selected) | **0.7210** | **1.0000** | **0.8040** | **0.5667** | **0.0000** |
| Existing ranker + lexical + dense | 0.7002 | 1.0000 | 0.7824 | 0.5556 | 0.0000 |

The frozen pre-change agent scored nDCG@10 `0.2182`, Success@10 `0.4444`,
first-relevant MRR `0.1943`, P@10 `0.1370`, and hard-violation rate `0.3552`
on the same development split.

After selection was frozen, the chosen dense variant was run on the confirmation
split. A generic brand/attribute collision guard was then found by a separate
160-query parser diagnostic, and an independent state/operator safety review
later found additional deterministic bugs. Those fixes were selected from their
reproductions, not from confirmation outcomes, and the final code was remeasured.
The confirmation split is therefore a useful frozen regression split, not a
pristine untouched holdout or statistically significant proof of generalization:

| Confirmation | nDCG@10 | Success@10 | First-relevant MRR | P@10 | Hard violation rate |
|---|---:|---:|---:|---:|---:|
| Frozen pre-change agent | 0.2504 | 0.5000 | 0.2927 | 0.2115 | 0.4023 |
| Selected dense pipeline | **0.4502** | **0.8462** | **0.6132** | **0.4731** | **0.2784** |

The lower confirmation result is evidence of remaining paraphrase and
hard-slot generalization limits; no architecture or weight was retuned against
that split.

## Official public evaluator regression

Command:

```powershell
py -3.13 -B -m evaluator.local_evaluator --output results.json
```

Result on all 200 public sessions:

| Metric | Before | After |
|---|---:|---:|
| Hit Rate@10 | 0.995000 | 0.995000 |
| MRR | 0.704468 | 0.704468 |
| MTTC | 2.120000 | 2.120000 |
| Technical Score | 0.886440 | 0.886440 |

Both full JSON files have SHA-256
`1FE0B8857F524DAD853E7A68CEFFE2BC01F1011F306BF93F894AE095F3310719`.

## Runtime

An informal 12-query spot measurement on the development Windows CPU/Python
3.13 machine gave:

- cold first free-form response: 2.136 seconds;
- warm median: 0.586 seconds;
- observed warm p95: 1.218 seconds;
- optional API/LLM calls: zero.

The original query list and timing harness were not saved, so these values are
indicative rather than independently reproducible. A separate submission audit
observed cold starts between roughly 2.1 and 2.9 seconds and semantic-only peak
memory around 346 MB. The model and vectors are loaded lazily on the first
free-form request and are never loaded by the official formatted-query evaluator.

## Reproduction

The selected dense free-form path requires Python 3.11+ (tested on 3.13).
Install its offline semantic runtime:

```powershell
py -3.13 -m pip install -r requirements-semantic.txt
```

Run unit and isolation tests:

```powershell
py -3.13 -B -m unittest tests.test_free_form_parser tests.test_agent tests.test_semantic_retrieval tests.test_lexical_retrieval tests.test_free_form_retrieval_benchmark
```

That focused submission suite currently passes all 80 tests. Full legacy test
discovery runs 178 tests on this Windows environment but reports 28 cleanup
errors: older experiment-runner tests leave temporary SQLite model connections
open and Windows refuses to delete those files. The failures are all
`PermissionError: [WinError 32]` during temporary-directory cleanup; the focused
free-form, retrieval, isolation, and official evaluator checks pass.

Run the development retrieval diagnostic:

```powershell
py -3.13 -B tests\free_form_retrieval_benchmark.py --split development --retrieval-mode dense --output retrieval-dense-development.json
```

Rebuild the lexical artifact if required:

```powershell
py -3.13 -B -m starter.lexical_retrieval --catalog data\catalog.jsonl --output data\lexical_index.sqlite3 --force
```

The committed dense artifact should normally be used. A CPU rebuild is an
offline preparation step and can take tens of minutes or longer:

```powershell
py -3.13 -B -m starter.semantic_retrieval --catalog data\catalog.jsonl --output data\semantic_embeddings.npz --model-path models\bge-small-en-v1.5 --local-files-only --threads 8 --batch-size 64
```

## Known limitations

- The deterministic parser remains conservative and does not normalize every
  category paraphrase, word-number budget, or implicit size expression.
- Unsupported negation such as `not flashy` cannot be guaranteed from this
  catalogue schema; operator text is withheld from dense retrieval rather than
  pretending embeddings enforce negation.
- Parent-level color/material/brand indexes can be multi-valued and incomplete.
- Cross-field OR is not represented as grouped Boolean clauses: a request like
  `black shoes or white sandals` is approximated by independent category and
  color alternatives. Same-field OR is exact.
- Feature/use-case/size exclusions remain soft where parent-level metadata is
  not dependable, so those negative preferences are not hard guarantees.
- Exact brand extraction uses catalogue brands occurring at least twice to
  suppress noisy singleton stores. A singleton brand can still be accepted as
  an exact answer to a brand clarification, but may be missed in an opening.
- Exact parser accuracy is not the optimization target; the separate 160-query
  diagnostic currently has 105/160 fully exact queries (65.625%) even though
  end-to-end retrieval improves. All 16 multi-turn state scenarios pass.
- Dense embeddings plus the model add roughly 103 MB (decimal); including the
  lexical failover index makes the complete optional retrieval bundle roughly
  132 MB. Dense inference adds sub-second to low-single-second CPU latency.
- Confirmation failures cluster around unseen category paraphrases and worded
  budgets. The next experiment should be a confidence-gated catalogue entity
  linker that proposes soft category candidates without turning uncertain
  synonyms into hard filters.
