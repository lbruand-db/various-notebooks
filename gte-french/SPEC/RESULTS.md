# Results — GTE / BGE embeddings, English vs. French / Italian / Spanish (STS)

Run of `benchmark_gte_french.ipynb` per [`SPECS.md`](./SPECS.md).

- **Models:** `databricks-gte-large-en` (FMAPI), `databricks-bge-large-en` (FMAPI), and
  `Alibaba-NLP/gte-multilingual-base` (downloaded from HuggingFace, run locally on CPU via
  `sentence-transformers`)
- **Dataset:** `PhilipMay/stsb_multi_mt`, `test` split, 1,379 pairs per language, languages
  `en` (reference), `fr`, `it`, `es`
- **Metric:** Spearman correlation of cosine similarity vs. gold score (`cosine_spearman`)
- **Compute:** serverless (CPU), workspace `e2-demo-field-eng`
- **Date:** 2026-09-15

## Headline (cosine_spearman)

| Model | EN | FR | IT | ES | retention* |
|-------|---:|---:|---:|---:|-----------:|
| `gte-large-en` (FMAPI) | 0.8310 | 0.7033 | 0.6947 | 0.7311 | 0.854 |
| `bge-large-en` (FMAPI) | 0.8751 | 0.7006 | 0.6944 | 0.7154 | 0.803 |
| `gte-multilingual-base` (CPU) | 0.8641 | 0.8411 | 0.8228 | 0.8489 | 0.969 |

\* retention = mean(FR, IT, ES) ÷ EN.

**Best per language:** EN → `bge-large-en` (0.8751); FR → `gte-multilingual-base` (0.8411);
IT → `gte-multilingual-base` (0.8228); ES → `gte-multilingual-base` (0.8489).

## Verdict

Both English-tuned FMAPI endpoints fall to ~0.69–0.73 on French, Italian, and Spanish despite
strong English scores. `bge-large-en` is the best on English (0.875) yet has the worst
cross-lingual retention (0.80) — a higher English score does not predict quality in other
languages.

`gte-multilingual-base` wins FR, IT, and ES outright (0.82–0.85) and retains ~97% of its
English quality across them. It is the clear choice for non-English work, at the cost of
self-hosting it (CPU here) rather than a managed FMAPI endpoint.

## Reproduce

Run `benchmark_gte_french.ipynb` on serverless. Languages are set by `LANGS` in the config
cell. STS-B is cached per language to
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
