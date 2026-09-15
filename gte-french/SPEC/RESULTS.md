# Results — GTE embeddings, French vs. English (STS)

Run of `benchmark_gte_french.ipynb` per [`SPECS.md`](./SPECS.md).

- **Endpoint:** `databricks-gte-large-en` (Databricks FMAPI)
- **Dataset:** `PhilipMay/stsb_multi_mt`, `test` split, 1,379 pairs per language
- **Metric:** Spearman correlation of cosine similarity vs. gold score (`cosine_spearman`)
- **Compute:** serverless, workspace `e2-demo-field-eng`
- **Date:** 2026-09-15

## Headline

| Language | Pairs | cosine_spearman | cosine_pearson |
|----------|------:|----------------:|---------------:|
| English  | 1,379 | **0.8310** | 0.8338 |
| French   | 1,379 | **0.7032** | 0.7133 |

**FR/EN cosine_spearman ratio: 0.846**

## Verdict

The English score (0.831) matches GTE-large-en's published STS-B result, which validates
the harness. French comes in at 0.703 — a solid correlation, so the endpoint is **usable
for French**, but with a **~15% relative quality drop** vs. English. For French-heavy
retrieval/search where precision matters, this gap is worth weighing against a
French-capable or multilingual embedding model (out of scope for this GTE-only benchmark).

## Reproduce

Run `benchmark_gte_french.ipynb` on serverless. STS-B is cached to
`/Volumes/lucasbruand_catalog/gte_french_bench/data/stsb/`; the results table is written to
`results_gte_french.csv` in the same Volume. Re-runs read the cache and skip the download.
