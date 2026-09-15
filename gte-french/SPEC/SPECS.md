# Benchmark: GTE embeddings — French vs. English

## 1. Motivation

We will serve embeddings through the **Databricks Foundation Model API (FMAPI)** GTE
endpoint, **`databricks-gte-large-en`** — the English-tuned GTE-large model. The `-en`
suffix signals that the model was trained and evaluated primarily on English.

Many of our use cases involve **French** text (RAG over French documents, French
classification, French semantic search). Before relying on this endpoint for French, we
need evidence: **does the English-oriented GTE FMAPI endpoint produce useful embeddings
for French, and how much quality do we lose relative to English?**

This spec defines a reproducible benchmark that answers that question.

> **Scope constraint:** the benchmark evaluates **only** GTE via FMAPI. No other embedding
> model (no `system.ai` pyfunc, no multilingual baseline, no local sentence-transformers)
> is in scope. The comparison is French-vs-English of the *same* endpoint.

## 2. Goals & non-goals

**Goals**
- Quantify GTE (FMAPI) embedding quality on **French** vs. **English** on equivalent tasks.
- Use **parallel / aligned** data so the two languages differ only in language, not in
  content or difficulty.
- Produce a small set of headline metrics + a short written verdict that a reader can
  act on.

**Non-goals**
- Not a general MTEB leaderboard reproduction.
- Not fine-tuning or adapting the models.
- Not a latency/throughput/cost benchmark (tracked separately if needed).

## 3. Models under test

Three models, compared on the same pairs and metric:

| Model | Access | Dimension |
|-------|--------|-----------|
| **`databricks-gte-large-en`** | Databricks FMAPI serving endpoint (pay-per-token) | 1024 |
| **`databricks-bge-large-en`** | Databricks FMAPI serving endpoint (pay-per-token) | 1024 |
| **`Alibaba-NLP/gte-multilingual-base`** | downloaded from HuggingFace, run **locally on CPU** via `sentence-transformers` | 768 |

**FMAPI invocation:** MLflow deployments client —
`deploy_client.predict(endpoint="<endpoint>", inputs={"input": [texts...]})`
→ `data[i].embedding`, for both the GTE and BGE endpoints.

**Local multilingual model (CPU):** `sentence-transformers`. Getting it to run on serverless
CPU requires: pin `transformers>=4.41,<5` (its custom code calls `ModuleUtilsMixin` helpers
removed in v5); cache the model in a UC Volume via a symlink-dereferenced copy (HF can't
download onto a FUSE Volume); load with `low_cpu_mem_usage=False` and rebuild the
non-persistent `position_ids`/RoPE buffers.

Both: L2-normalize before cosine similarity; batch requests.

## 4. Dataset — standard choice

**Primary (standard): the STS Benchmark (STS-B), used parallel in EN and FR.**

- **English:** STS-Benchmark test split — the canonical STS task in the
  [MTEB](https://github.com/embeddings-benchmark/mteb) leaderboard.
- **French:** [`stsb_multi_mt`](https://huggingface.co/datasets/PhilipMay/stsb_multi_mt),
  `fr` config — the *same* STS-B pairs machine-translated (DeepL) into French. This is the
  dataset MTEB uses for its French STS task (`STSBenchmarkMultilingualSTS`, part of the
  **MTEB-French** suite, [Ciancone et al. 2024](https://arxiv.org/abs/2405.20468)).

Why this is the right standard:
- It is the recognized MTEB STS task, so results are comparable to published numbers.
- EN and FR are the **same 1,379 test pairs with identical gold similarity scores**
  (0.0–5.0 scale), so the EN-vs-FR difference is attributable to language alone — a clean
  apples-to-apples comparison.
- No labeling or corpus-construction work on our side.

**Secondary / optional (only if a retrieval signal is wanted later):** `SICKFr` (French
STS) as a second STS point, or an MTEB-French retrieval task (e.g. `MintakaFr`,
`AlloProfRetrieval`) evaluated FR-only against the EN equivalent. Not required for v1.

> **Sizes:** use the full STS-B test split (1,379 pairs per language). Record N.

## 5. Metric — standard choice

**Primary (standard): Spearman rank correlation between the cosine similarity of the two
sentence embeddings and the gold similarity score** — reported per language.

This is MTEB's official main metric for STS tasks (`cosine_spearman`). Procedure per
language:
1. Embed both sides of each pair via the FMAPI endpoint; L2-normalize.
2. Cosine similarity per pair.
3. Spearman correlation between those cosines and the gold scores.

Secondary: Pearson correlation (`cosine_pearson`) for reference. (If a retrieval task is
added later, its standard MTEB main metric is **nDCG@10**.)

**Headline comparison — the FR/EN ratio:** `spearman_fr / spearman_en`. A ratio near 1.0
means French is served about as well as English; a large drop means the English GTE
endpoint is a poor fit for French. Report both raw scores and the ratio.

## 6. Method

1. Set up the FMAPI client for `databricks-gte-large-en` (§3) — a small `embed(texts)`
   helper that batches, retries on rate limits, and returns an `(n, 1024)` array.
2. For each task and each language: embed all texts (batched), L2-normalize.
3. Compute the task metric.
4. Assemble a results table: rows = tasks, columns = EN score, FR score, FR/EN ratio.
5. Write a short **verdict** paragraph: is GTE (FMAPI) usable for French, and under what
   caveats?

## 7. Deliverables

- `benchmark_gte_french.ipynb` (or `.py` Databricks notebook) implementing §6.
- A results table (printed + saved, e.g. as a Delta table or CSV under the project).
- A markdown **Results & verdict** section appended to this repo (or to this spec's
  companion `RESULTS.md`).

## 8. Reproducibility

- Record: endpoint name (`databricks-gte-large-en`), dataset name + split + N, date,
  runtime, client library versions.
- Fixed random seed where sampling is used.
- Deterministic ordering of inputs.

## 9. Success criteria (for the benchmark itself)

- Both languages evaluated on identical gold labels (STS).
- Metrics reproduce within noise on a re-run.
- The notebook runs end-to-end on a standard Databricks ML runtime without manual steps
  beyond those already in the existing GTE notebook.

## 10. Decisions & open questions

- **v1 scope: STS-only** (STS-B, EN vs FR, Spearman). Retrieval deferred.
- **FMAPI access:** MLflow deployments client (`get_deploy_client("databricks")`) — works
  with notebook auth on serverless, no extra config.
- **Execution:** run `benchmark_gte_french.ipynb` on the **e2-demo-field-eng** workspace
  using **serverless** compute.
- **Data caching:** the notebook downloads STS-B from HuggingFace once into a **UC Volume**
  (`/Volumes/<catalog>/<schema>/<volume>/stsb/`) and reads the cached parquet on subsequent
  runs (no re-download).
- Results: printed table + `results_gte_french.csv` written to the same UC Volume;
  promoting to a Delta table is a later step.
