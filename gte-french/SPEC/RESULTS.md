# Results — GTE / BGE embeddings, French vs. English (STS)

Run of `benchmark_gte_french.ipynb` per [`SPECS.md`](./SPECS.md).

- **Models:** `databricks-gte-large-en` (FMAPI), `databricks-bge-large-en` (FMAPI), and
  `Alibaba-NLP/gte-multilingual-base` (downloaded from HuggingFace, run locally on CPU via
  `sentence-transformers`)
- **Dataset:** `PhilipMay/stsb_multi_mt`, `test` split, 1,379 pairs per language
- **Metric:** Spearman correlation of cosine similarity vs. gold score (`cosine_spearman`)
- **Compute:** serverless (CPU), workspace `e2-demo-field-eng`
- **Date:** 2026-09-15

## Headline

| Model | EN spearman | EN pearson | FR spearman | FR pearson | FR/EN (spearman) |
|-------|------------:|-----------:|------------:|-----------:|-----------------:|
| `gte-large-en` (FMAPI) | 0.8310 | 0.8338 | 0.7033 | 0.7133 | 0.846 |
| `bge-large-en` (FMAPI) | 0.8751 | 0.8620 | 0.7006 | 0.7120 | 0.801 |
| `gte-multilingual-base` (CPU) | 0.8641 | 0.8553 | 0.8411 | 0.8389 | 0.973 |

**Best on French: `gte-multilingual-base` (0.8411).**

## Verdict

Both English-tuned FMAPI endpoints land at ~0.70 on French despite strong English scores.
`bge-large-en` is the best model on English (0.875) yet drops the hardest on French (FR/EN
0.80, French 0.701) — a higher English score does not predict French quality.

`gte-multilingual-base` is the clear choice for French: French Spearman 0.841 (+0.14 over
either English-only model), an FR/EN ratio of 0.97. The trade-off is self-hosting it (CPU
here) rather than a managed FMAPI endpoint.

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

(The two FMAPI models need none of this — they are queried as serving endpoints.)
