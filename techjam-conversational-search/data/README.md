# Competition Data

## `public_set.jsonl`

Contains 200 labeled development sessions: 80 Buying, 80 Browsing, 30 Intent Override, and 10 Boundary sessions.

Each session contains a safe aggregate `user_profile` and public labels for local development. Direct user identifiers, timestamps, free-text reviews, raw purchase history, hidden intent cards, and simulator-policy internals are not shipped in this participant file.

## `catalog.jsonl`

Download `catalog.jsonl.gz` from the GitHub Release and decompress it as `catalog.jsonl` in this directory. Expected row count: 50,000.

Never place API keys, private evaluation data, or participant outputs in this directory.

## `category_index.sqlite3`

Preserves the original category hierarchy and product mappings. Its `products`
table also stores the exact `coarse_category` string produced for each
`parent_asin` by the local evaluator. Regenerate it from the immutable catalog
with:

```bash
python -m starter.category_index --force
```

## Free-form retrieval artifacts

`semantic_embeddings.npz` contains the precomputed normalized BGE-small vectors
for all 50,000 frozen products. `lexical_index.sqlite3` contains the matching
SQLite FTS5 documents and is used as the no-ML fallback. Both contain only
catalogue-derived data and are read-only at agent runtime.

Checksums, rebuild commands, model disclosure, and benchmark evidence are in
[`../docs/free_form_retrieval_report.md`](../docs/free_form_retrieval_report.md).
