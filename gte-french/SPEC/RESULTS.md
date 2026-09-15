# Results — GTE embeddings, French vs. English (STS)

Run of `benchmark_gte_french.ipynb` per [`SPECS.md`](./SPECS.md).

- **Models:** `databricks-gte-large-en` (FMAPI) and `Alibaba-NLP/gte-multilingual-base`
  (downloaded from HuggingFace, run locally on CPU via `sentence-transformers`)
- **Dataset:** `PhilipMay/stsb_multi_mt`, `test` split, 1,379 pairs per language
- **Metric:** Spearman correlation of cosine similarity vs. gold score (`cosine_spearman`)
- **Compute:** serverless (CPU), workspace `e2-demo-field-eng`
- **Date:** 2026-09-15

## Headline

| Model | EN spearman | EN pearson | FR spearman | FR pearson | FR/EN (spearman) |
|-------|------------:|-----------:|------------:|-----------:|-----------------:|
| `gte-large-en` (FMAPI) | 0.8309 | 0.8338 | 0.7032 | 0.7133 | 0.846 |
| `gte-multilingual-base` (CPU) | 0.8641 | 0.8553 | 0.8411 | 0.8389 | 0.973 |

**French Spearman: multilingual − English-only = +0.1379.**

## Verdict

The English-only `gte-large-en` score (0.831) matches its published STS-B result,
validating the harness. On French it drops to 0.703 — a ~15% relative penalty.

`gte-multilingual-base` largely closes the gap: French Spearman 0.841 (+0.138 over the
English model), an FR/EN ratio of 0.97 vs. 0.85, and it is marginally stronger on English
too (0.864). For French-heavy retrieval/search the multilingual model is the clear choice,
at the cost of self-hosting it (CPU here) rather than using a managed FMAPI endpoint.

## Reproduce

Run `benchmark_gte_french.ipynb` on serverless. STS-B is cached to
`/Volumes/lucasbruand_catalog/gte_french_bench/data/stsb/`, the multilingual model to
`/Volumes/.../data/models/gte-multilingual-base/`, and the results table to
`results_gte_french.csv` in the same Volume. Re-runs read the caches and skip the downloads.

### Getting the multilingual model to run on CPU (serverless)

- Pin `transformers>=4.41,<5` — the model's custom code calls `ModuleUtilsMixin` helpers
  (`get_extended_attention_mask`, …) that transformers 5.x removed.
- Cache the model in the UC Volume by downloading to local disk and copying in with symlinks
  dereferenced — HuggingFace can't download onto a FUSE-mounted Volume.
- Load with `low_cpu_mem_usage=False` and rebuild the non-persistent `position_ids` and RoPE
  (`inv_freq`/`cos_cached`/`sin_cached`) buffers; otherwise they can be uninitialized and
  cause an `IndexError` in the RoPE branch.
