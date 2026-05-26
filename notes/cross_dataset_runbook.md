# Cross-Dataset Runbook

Last updated: 2026-05-25

Purpose: keep the operational commands from the dataset chats in one place. Per-dataset folders still contain the detailed READMEs; this file is the compact checklist to resume work without searching old chats.

## Global Rules

- Always clear stale dataset env vars before switching datasets:

```bash
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
unset HF_HOME HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE HF_HUB_CACHE TRANSFORMERS_CACHE XDG_CACHE_HOME
```

- Then source exactly one dataset env:

```bash
source <dataset>/env_hpc.sh
```

- Use page recall as the main metric for exact page-labeled datasets. Doc recall is useful but secondary.
- Use `plain_top224_ret1000_prediction.json` as the dense input for SPLADE fusion and Graph-PPR unless a dataset note says otherwise.
- Use this frozen page-labeled Graph-PPR default first:

```bash
GRAPH_PROFILE=page_rank_probe
FINAL_TOP_PAGES=1000
PER_DOC_PAGE_LIMIT=0
DENSE_WEIGHT=1.25
SPARSE_WEIGHT=0.75
RESTART_PROB=0.15
PPR_ITERS=30
PAGE_DOC_EDGE_WEIGHT=1.0
SAME_DOC_WINDOW=1
ADJACENT_PAGE_EDGE_WEIGHT=0.25
FINAL_PAGE_SEED_WEIGHT=1.0
FINAL_PPR_PAGE_WEIGHT=0.5
FINAL_PPR_DOC_WEIGHT=0.25
```

This corresponds to the row we call `denseheavy125_medium_both`. ViDoSeek has a dataset-specific heavier best row, but `denseheavy125_medium_both` is the current single general config.

## Current Status Snapshot

| Dataset | Status | Next action |
| --- | --- | --- |
| M3DocVQA/MMQA | Baseline, MaxSim+, SPLADE, and Graph Page Preserve runs exist. | Optional clean apples-to-apples ret4 vs ret1000 QA check. |
| MMDocIR | Prepared, embedded, plain_top224, SPLADE, sweep, and Graph-PPR results exist. | None unless rerunning for reproducibility. |
| SciEGQA | Prepared, embedded, plain_top224, SPLADE, sweep, and Graph-PPR results exist. | None unless rerunning for reproducibility. |
| ViDoRe V3 | Prepared, embedded, plain_top224, SPLADE, sweep, and Graph-PPR results exist. | None unless rerunning for reproducibility. |
| ViDoSeek | Prepared, embedded, plain_top224, SPLADE, sweep, and Graph-PPR results exist. | None unless rerunning for reproducibility. |
| OpenDocVQA | Full OCR-backed Graph-PPR completed. Full-dev no-support `pairwise_content_posterior` was negative: page hit@4 `26173 -> 22592`, net `-3581`. | Keep Graph-PPR as the full-dev result; use content posterior only as a hard-subset diagnostic/rescue component. |
| MMLongBench DocQA | Prepared: 708 docs, 30,917 pages, 14,466 QAs, 19 missing gold pages; embedding job was submitted. | Check embedding completion, then index, dense retrieval, plain_top224, SPLADE, Graph-PPR. |
| DUDE | Prepared with `Amazon_original`; OCR sanity passed: 4,020/4,086 nonempty pages, 66 empty, 0 missing images. Dense baseline, plain_top224, and SPLADE/doc-RRF are complete. SPLADE/doc-RRF improves doc@4 to `0.6631` but lowers page@4 to `0.5312`. | Run Graph-PPR. |

## Common Sanity Checks

### Converted Dataset

```bash
"$REPO_ROOT/env/bin/python" - <<'PY'
import json, os
from pathlib import Path
DATASET_ROOT_NAME = "dude"  # change this, e.g. "vidore-v3" or "sci-egqa-bench"
root = Path(os.environ["LOCAL_DATA_DIR"]) / DATASET_ROOT_NAME
doc_ids = json.loads((root / "dev_doc_ids.json").read_text())
pages = [json.loads(line) for line in (root / "doc_pages_dev.jsonl").open() if line.strip()]
qas = [json.loads(line) for line in (root / "MMQA_dev.jsonl").open() if line.strip()]
page_uids = {row["page_uid"] for row in pages}
missing = []
for row in qas:
    for uid in row.get("metadata", {}).get("gold_page_uids", []):
        if uid not in page_uids:
            missing.append((row["qid"], uid))
print("docs", len(doc_ids))
print("pages", len(pages))
print("qas", len(qas))
print("missing_gold_pages", len(missing))
print("first_missing", missing[:5])
PY
```

### Text Coverage For SPLADE

```bash
"$REPO_ROOT/env/bin/python" - <<'PY'
import json, os
from pathlib import Path
DATASET_ROOT_NAME = "dude"  # change this, e.g. "vidore-v3" or "mmlongbench-docqa"
root = Path(os.environ["LOCAL_DATA_DIR"]) / DATASET_ROOT_NAME
n = nonempty = empty = missing_img = 0
samples = []
for line in open(root / "doc_pages_dev.jsonl"):
    r = json.loads(line)
    n += 1
    text = (r.get("text") or r.get("ocr_text") or "").strip()
    nonempty += bool(text)
    empty += not bool(text)
    if text and len(samples) < 5:
        samples.append((r["page_uid"], len(text), text[:160]))
    if not (root / r["image_path"]).exists():
        missing_img += 1
print("pages", n)
print("nonempty_text_pages", nonempty)
print("empty_text_pages", empty)
print("empty_fraction", empty / n if n else None)
print("missing_images", missing_img)
print("samples", samples)
PY
```

Do not run SPLADE if text is empty or clearly metadata garbage.

### Embedding Completion

```bash
find "$LOCAL_EMBEDDINGS_DIR/<embedding_name>" -name "*.safetensors" | wc -l
"$REPO_ROOT/env/bin/python" - <<'PY'
import json, os
DATASET_ROOT_NAME = "dude"  # change this
p = os.environ["LOCAL_DATA_DIR"] + f"/{DATASET_ROOT_NAME}/dev_doc_ids.json"
print(len(json.load(open(p))))
PY
```

The counts should match.

### Evaluation

```bash
"$REPO_ROOT/env/bin/python" mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$PRED" \
  --gold "$GOLD" \
  --recall-k 1 2 4 5 10 20 50 100
```

## Standard Page-Labeled Pipeline

Use this shape for MMDocIR, SciEGQA, ViDoRe V3, ViDoSeek, MMLongBench DocQA, DUDE, and OpenDocVQA after each dataset is prepared.

### Build Index

```bash
"$REPO_ROOT/env/bin/python" mmdocir/run_indexing_mmdocir.py \
  --data-root "$DATA_ROOT" \
  --embedding-dir "$EMBEDDING_DIR" \
  --output-dir "$INDEX_DIR" \
  --faiss-index-type ivfflat
```

### Dense Retrieval

```bash
mkdir -p "$LOCAL_OUTPUT_DIR/$DATA_NAME"

"$REPO_ROOT/env/bin/python" mmdocir/run_retrieval_mmdocir.py \
  --data-root "$DATA_ROOT" \
  --embedding-dir "$EMBEDDING_DIR" \
  --index-dir "$INDEX_DIR" \
  --output-json "$LOCAL_OUTPUT_DIR/$DATA_NAME/baseline_ret1000.json" \
  --n-retrieval-pages 1000 \
  --faiss-nprobe 4
```

### Plain Top-224

Prefer the dataset wrapper when it exists:

```bash
bash <dataset>/run_plain_top224_<dataset>.sh
```

The output should be:

```text
$LOCAL_OUTPUT_DIR/<dataset>/plain_top224_ret1000_prediction.json
```

### SPLADE + Doc-RRF

```bash
DATA_NAME=<dataset> \
DATA_ROOT="$DATA_ROOT" \
DENSE_PRED="$LOCAL_OUTPUT_DIR/<dataset>/plain_top224_ret1000_prediction.json" \
OUT_DIR="$LOCAL_OUTPUT_DIR/<dataset>/doc_rrf_plain_top224_splade" \
DENSE_WEIGHT=1.25 \
SPARSE_WEIGHT=0.75 \
RRF_K=10 \
SPLADE_DEVICE=auto \
bash scripts/run_external_doc_rrf_pipeline.sh
```

If page text was already exported manually, add:

```bash
PAGE_TEXT_JSONL="$LOCAL_OUTPUT_DIR/<dataset>/doc_rrf_plain_top224_splade/<dataset>_page_text_dev.jsonl" \
SKIP_EXPORT=1
```

### Graph-PPR Default

```bash
DATA_NAME=<dataset> \
DATA_ROOT="$DATA_ROOT" \
DENSE_PRED="$LOCAL_OUTPUT_DIR/<dataset>/plain_top224_ret1000_prediction.json" \
SPARSE_PRED="$LOCAL_OUTPUT_DIR/<dataset>/doc_rrf_plain_top224_splade/<dataset>_splade_ret1000.prediction.json" \
OUT_DIR="$LOCAL_OUTPUT_DIR/<dataset>/graph_ppr_plain_top224_splade" \
GRAPH_PROFILE=page_rank_probe \
GRAPH_LABEL="<dataset>_denseheavy125_medium_both" \
FINAL_TOP_PAGES=1000 \
PER_DOC_PAGE_LIMIT=0 \
DENSE_WEIGHT=1.25 \
SPARSE_WEIGHT=0.75 \
RESTART_PROB=0.15 \
PPR_ITERS=30 \
PAGE_DOC_EDGE_WEIGHT=1.0 \
SAME_DOC_WINDOW=1 \
ADJACENT_PAGE_EDGE_WEIGHT=0.25 \
FINAL_PAGE_SEED_WEIGHT=1.0 \
FINAL_PPR_PAGE_WEIGHT=0.5 \
FINAL_PPR_DOC_WEIGHT=0.25 \
bash scripts/run_external_graph_ppr_pipeline.sh
```

## Dataset-Specific Entries

### MMDocIR

```bash
cd /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
source mmdocir/env_hpc.sh
```

Prepare:

```bash
"$REPO_ROOT/env/bin/python" mmdocir/prepare_mmdocir.py \
  --download \
  --snapshot-dir "$MMDocIR_WORK_ROOT/hf_snapshot/MMDocIR_Evaluation_Dataset" \
  --output-root "$LOCAL_DATA_DIR/mm-docir"
```

Key paths:

```text
DATA_ROOT="$LOCAL_DATA_DIR/mm-docir"
EMBEDDING_DIR="$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_mm-docir_dev"
INDEX_DIR="$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_mm-docir_dev_pageindex_ivfflat"
DENSE_PRED="$LOCAL_OUTPUT_DIR/mmdocir/plain_top224_ret1000_prediction.json"
```

Run `plain_top224`:

```bash
bash mmdocir/run_plain_top224_mmdocir.sh
```

### SciEGQA

```bash
cd /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
source sciegqa/env_hpc.sh
```

Prepare:

```bash
"$REPO_ROOT/env/bin/python" sciegqa/prepare_sciegqa_bench.py \
  --download \
  --snapshot-dir "$SciEGQA_WORK_ROOT/hf_snapshot/SciEGQA-Bench" \
  --output-root "$LOCAL_DATA_DIR/sci-egqa-bench"
```

Embed:

```bash
sbatch sciegqa/sbatch_embed_sciegqa_array.sh
```

Run `plain_top224`:

```bash
bash sciegqa/run_plain_top224_sciegqa.sh
```

Key paths:

```text
DATA_ROOT="$LOCAL_DATA_DIR/sci-egqa-bench"
EMBEDDING_DIR="$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_sci-egqa-bench_dev"
INDEX_DIR="$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_sci-egqa-bench_dev_pageindex_ivfflat"
DENSE_PRED="$LOCAL_OUTPUT_DIR/sciegqa/plain_top224_ret1000_prediction.json"
```

### ViDoRe V3

```bash
cd /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
source vidore/env_hpc.sh
```

Prepare all public V3 domains:

```bash
"$REPO_ROOT/env/bin/python" vidore/prepare_vidore_v3.py \
  --download \
  --cache-dir "$VIDORE_WORK_ROOT/hf_cache" \
  --output-root "$LOCAL_DATA_DIR/vidore-v3"
```

Embed:

```bash
sbatch --time=12:00:00 --array=0-7 --export=ALL,NUM_SHARDS=8,BATCH_SIZE=2 \
  vidore/sbatch_embed_vidore_v3_array.sh
```

Use sharded retrieval/plain_top224 for full ViDoRe:

```bash
sbatch --time=24:00:00 --array=0-15%4 \
  --export=ALL,NUM_SHARDS=16,TOP_PAGES=1000,FAISS_NPROBE=4,SAVE_EVERY=25 \
  vidore/sbatch_retrieval_vidore_v3_array.sh

"$REPO_ROOT/env/bin/python" mmdocir/merge_retrieval_predictions.py \
  --input-glob "$LOCAL_OUTPUT_DIR/vidore-v3/baseline_ret1000_shards/shard_*_of_16.json" \
  --output-json "$LOCAL_OUTPUT_DIR/vidore-v3/baseline_ret1000.json" \
  --gold "$LOCAL_DATA_DIR/vidore-v3/MMQA_dev.jsonl"

sbatch --time=24:00:00 --array=0-15%4 \
  --export=ALL,NUM_SHARDS=16,TOP_PAGES=1000,BASE_ONLY_PAGE_BATCH_SIZE=64 \
  vidore/sbatch_plain_top224_vidore_v3_array.sh

"$REPO_ROOT/env/bin/python" mmdocir/merge_retrieval_predictions.py \
  --input-glob "$LOCAL_OUTPUT_DIR/vidore-v3/plain_top224_ret1000_shards/shard_*_of_16_prediction.json" \
  --output-json "$LOCAL_OUTPUT_DIR/vidore-v3/plain_top224_ret1000_prediction.json" \
  --gold "$LOCAL_DATA_DIR/vidore-v3/MMQA_dev.jsonl"
```

### ViDoSeek

```bash
cd /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
source vidoseek/env_hpc.sh
```

Prepare:

```bash
"$REPO_ROOT/env/bin/python" vidoseek/prepare_vidoseek.py \
  --download \
  --snapshot-dir "$VIDOSEEK_WORK_ROOT/hf_snapshot/ViDoSeek" \
  --output-root "$LOCAL_DATA_DIR/vidoseek"
```

Embed:

```bash
sbatch --array=0-7 --export=ALL,NUM_SHARDS=8,BATCH_SIZE=2 \
  vidoseek/sbatch_embed_vidoseek_array.sh
```

Run `plain_top224`:

```bash
bash vidoseek/run_plain_top224_vidoseek.sh
```

Note: ViDoSeek is saturated. `denseheavy150_m3best_pagepreserve` is the best individual row recorded so far, but use `denseheavy125_medium_both` for uniform cross-dataset reporting unless intentionally optimizing ViDoSeek alone.

### OpenDocVQA

```bash
cd /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
unset HF_HOME HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE HF_HUB_CACHE TRANSFORMERS_CACHE XDG_CACHE_HOME
source opendocvqa/env_hpc.sh
```

OpenDocVQA-Corpus is gated and large. Full prepared state observed:

```text
docs=3223
pages=206267
qas=41017
missing_gold_pages=0
```

OCR status from the completed EasyOCR run:

```text
rows=206267
unique_page_uids=206267
duplicates=0
nonempty_text_pages=205573
empty_text_pages=694
empty_fraction=0.00336
```

Merge OCR shards:

```bash
"$REPO_ROOT/env/bin/python" scripts/merge_jsonl_shards.py \
  --input-glob "$LOCAL_OUTPUT_DIR/opendocvqa/easyocr_page_text_shards/shard_*_of_64.jsonl" \
  --output-jsonl "$LOCAL_OUTPUT_DIR/opendocvqa/doc_rrf_plain_top224_splade/opendocvqa_page_text_dev.jsonl" \
  --output-summary-json "$LOCAL_OUTPUT_DIR/opendocvqa/doc_rrf_plain_top224_splade/opendocvqa_page_text_dev_merge_summary.json" \
  --dedupe-key page_uid
```

OCR-backed SPLADE/doc-RRF already produced:

```text
opendocvqa_splade_ret1000.prediction.json
opendocvqa_exact_dense_splade_doc_rrf.prediction.json
page@4=0.5825
page@20=0.7395
doc@4=0.6069
doc@20=0.7818
```

OpenDocVQA Graph-PPR completed with the frozen `denseheavy125_medium_both` profile:

```text
qid_count 41017
page_recall@1 0.3988
page_recall@4 0.5863
page_recall@20 0.7662
doc_recall@4 0.6035
doc_recall@20 0.7922
page_hit@4 26173
doc_hit@4 26901
```

Negative full-dev boundary test:

```text
opendocvqa_boundary_pairwise_content_posterior_nosupport
n 41017
accepted 10209
base_page_hit_at_4_count 26173
page_hit_at_4_count 22592
recovered 146
lost 3727
net_recovered -3581
page_recall@4 0.5027
doc_recall@4 0.5683
```

Conclusion: do not use unconditional boundary reranking on full OpenDocVQA. Keep the OCR-backed
Graph-PPR prediction as the full-dataset result until a non-oracle selector can identify likely
right-document/wrong-page cases before applying boundary rescue.

Cross-dataset conditioning test for the boundary method:

```bash
ROUTER_DIR=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/query_subtype_router
mkdir -p "$ROUTER_DIR"

python scripts/learn_query_subtype_router.py \
  --run vidore "$VIDORE_GOLD" "$VIDORE_BASE" \
  --candidate vidore content "$BOUNDARY_DIR/vidore_boundary_pairwise_content_posterior.prediction.json" "$BOUNDARY_DIR/vidore_boundary_pairwise_content_posterior.cases.json" \
  --run mmdocir "$MMDOCIR_GOLD" "$MMDOCIR_BASE" \
  --candidate mmdocir content "$BOUNDARY_DIR/mmdocir_boundary_pairwise_content_posterior.prediction.json" "$BOUNDARY_DIR/mmdocir_boundary_pairwise_content_posterior.cases.json" \
  --run opendocvqa "$OPENDOC_GOLD" "$OPENDOC_BASE" \
  --candidate opendocvqa content "$BOUNDARY_DIR/opendocvqa_boundary_pairwise_content_posterior_nosupport.prediction.json" "$BOUNDARY_DIR/opendocvqa_boundary_pairwise_content_posterior_nosupport.cases.json" \
  --hit-k 4 \
  --cv-mode leave_run_out \
  --run-weighting equal_run \
  --output-json "$ROUTER_DIR/content_boundary_router_loro_equalrun.json" \
  --output-md "$ROUTER_DIR/content_boundary_router_loro_equalrun.md" \
  --output-routed-dir "$ROUTER_DIR/routed_content_boundary_loro_equalrun"
```

This is the current non-one-dataset check. The router learns observable query/candidate features from
the other runs and tests on the held-out run. `--run-weighting equal_run` prevents OpenDocVQA's 41k
queries from overpowering ViDoRe/MMDocIR during selector learning.

Unsupervised cluster-conditioned version:

```bash
python scripts/learn_cluster_conditioned_router.py \
  --run vidore "$VIDORE_GOLD" "$VIDORE_BASE" \
  --candidate vidore content "$BOUNDARY_DIR/vidore_boundary_pairwise_content_posterior.prediction.json" "$BOUNDARY_DIR/vidore_boundary_pairwise_content_posterior.cases.json" \
  --run mmdocir "$MMDOCIR_GOLD" "$MMDOCIR_BASE" \
  --candidate mmdocir content "$BOUNDARY_DIR/mmdocir_boundary_pairwise_content_posterior.prediction.json" "$BOUNDARY_DIR/mmdocir_boundary_pairwise_content_posterior.cases.json" \
  --run opendocvqa "$OPENDOC_GOLD" "$OPENDOC_BASE" \
  --candidate opendocvqa content "$BOUNDARY_DIR/opendocvqa_boundary_pairwise_content_posterior_nosupport.prediction.json" "$BOUNDARY_DIR/opendocvqa_boundary_pairwise_content_posterior_nosupport.cases.json" \
  --hit-k 4 \
  --cv-mode leave_run_out \
  --run-weighting equal_run \
  --cluster-count 0 \
  --min-clusters 1 \
  --max-clusters 0 \
  --min-cluster-n 5 \
  --doc-policy nonnegative \
  --output-json "$ROUTER_DIR/cluster_conditioned_content_loro_equalrun.json" \
  --output-md "$ROUTER_DIR/cluster_conditioned_content_loro_equalrun.md" \
  --output-routed-dir "$ROUTER_DIR/routed_cluster_conditioned_content_loro_equalrun"
```

This clusters observable features without gold labels. Labels are used only after clustering to
estimate held-out cluster utility and decide whether a cluster should route to the candidate or keep
base.

### Exact MaxSim Boundary Verifier

Use this targeted non-OCR test when the current Graph-PPR result has likely boundary failures. It
does not learn thresholds or use gold at decision time. For each query, it computes exact ColPali
MaxSim on only the current top-4 pages plus rank 5, then swaps rank 5 into top 4 only when exact
MaxSim scores it above the weakest current top-4 page.

Full-dev command shape:

```bash
MAXSIM_BOUNDARY_DIR=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/boundary_exact_maxsim
mkdir -p "$MAXSIM_BOUNDARY_DIR"

python scripts/rerank_graph_boundary_exact_maxsim.py \
  --gold "$VIDORE_DATA/MMQA_dev.jsonl" \
  --base-prediction "$VIDORE_OUT/vidore-v3_dev_graph_ppr_base.prediction.json" \
  --embedding-dir "$VIDORE_ROOT/embeddings/colpali-v1.2_vidore-v3_dev" \
  --hit-k 4 \
  --boundary-rank 5 \
  --output-prediction-json "$MAXSIM_BOUNDARY_DIR/vidore_exact_maxsim_boundary.prediction.json" \
  --output-summary-json "$MAXSIM_BOUNDARY_DIR/vidore_exact_maxsim_boundary.summary.json" \
  --output-case-json "$MAXSIM_BOUNDARY_DIR/vidore_exact_maxsim_boundary.cases.json"

python scripts/rerank_graph_boundary_exact_maxsim.py \
  --gold "$MMDOCIR_DATA/MMQA_dev.jsonl" \
  --base-prediction "$MMDOCIR_OUT/mmdocir_dev_graph_ppr_base.prediction.json" \
  --embedding-dir "$MMDOCIR_ROOT/embeddings/colpali-v1.2_mm-docir_dev" \
  --hit-k 4 \
  --boundary-rank 5 \
  --output-prediction-json "$MAXSIM_BOUNDARY_DIR/mmdocir_exact_maxsim_boundary.prediction.json" \
  --output-summary-json "$MAXSIM_BOUNDARY_DIR/mmdocir_exact_maxsim_boundary.summary.json" \
  --output-case-json "$MAXSIM_BOUNDARY_DIR/mmdocir_exact_maxsim_boundary.cases.json"

python scripts/rerank_graph_boundary_exact_maxsim.py \
  --gold "$OPENDOC_DATA/MMQA_dev.jsonl" \
  --base-prediction "$OPENDOC_OUT/opendocvqa_denseheavy125_medium_both.prediction.json" \
  --embedding-dir "$OPENDOC_ROOT/embeddings/colpali-v1.2_opendocvqa_dev" \
  --hit-k 4 \
  --boundary-rank 5 \
  --output-prediction-json "$MAXSIM_BOUNDARY_DIR/opendocvqa_exact_maxsim_boundary.prediction.json" \
  --output-summary-json "$MAXSIM_BOUNDARY_DIR/opendocvqa_exact_maxsim_boundary.summary.json" \
  --output-case-json "$MAXSIM_BOUNDARY_DIR/opendocvqa_exact_maxsim_boundary.cases.json"
```

For a cheap smoke test before a full run, add `--sample-qids 300 --sample-seed 17`. If the full run
loses many existing top-4 hits, keep it as an audit result and do not replace Graph-PPR.

Gated subset smoke test after an unsafe full run:

```bash
MAXSIM_BOUNDARY_DIR=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/boundary_exact_maxsim
mkdir -p "$MAXSIM_BOUNDARY_DIR"

python scripts/rerank_graph_boundary_exact_maxsim.py \
  --gold "$MMDOCIR_DATA/MMQA_dev.jsonl" \
  --base-prediction "$MMDOCIR_OUT/mmdocir_dev_graph_ppr_base.prediction.json" \
  --embedding-dir "$MMDOCIR_ROOT/embeddings/colpali-v1.2_mm-docir_dev" \
  --hit-k 4 \
  --boundary-rank 5 \
  --sample-qids 300 \
  --sample-seed 17 \
  --boundary-doc-policy weakest_doc \
  --min-exact-margin 0.25 \
  --output-prediction-json "$MAXSIM_BOUNDARY_DIR/mmdocir_exact_maxsim_boundary_gated_weakestdoc_m025_sample300.prediction.json" \
  --output-summary-json "$MAXSIM_BOUNDARY_DIR/mmdocir_exact_maxsim_boundary_gated_weakestdoc_m025_sample300.summary.json" \
  --output-case-json "$MAXSIM_BOUNDARY_DIR/mmdocir_exact_maxsim_boundary_gated_weakestdoc_m025_sample300.cases.json"
```

This gate is still label-free. It only permits a swap when rank 5 beats the weakest top-4 page by
exact MaxSim and both pages are in the same document. If too few swaps are accepted, relax to
`--boundary-doc-policy topk_doc --min-boundary-doc-topk-count 1`; if losses remain, add
`--max-base-margin-ratio-4-5 0.02` to restrict swaps to uncertain Graph-PPR boundaries.

Observed MMDocIR 300-query subset results:

| policy | sample | accepted | base page hit@4 | candidate page hit@4 | recovered | lost | net | page recall@4 | doc recall@4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `weakest_doc`, `min_exact_margin=0.25` | 300 | 43 | 186 | 184 | 0 | 2 | -2 | 0.5884 | 0.7900 |
| `topk_doc`, `min_exact_margin=0.25` | 300 | 57 | 186 | 183 | 0 | 3 | -3 | 0.5834 | 0.7900 |

Do not scale either observed gated setting as-is. Relaxing the document policy accepted more swaps
but only increased losses, so this exact rank-5 MaxSim gate is diagnostic rather than a candidate
full-dev reranker unless a stricter uncertainty subset shows positive net recovery.

### Limitation Report / Failure Taxonomy Audit

After a full Graph-PPR result exists, categorize its remaining page-hit failures:

```bash
FAILURE_AUDIT_DIR=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/failure_taxonomy
mkdir -p "$FAILURE_AUDIT_DIR"

python scripts/audit_retrieval_failure_taxonomy.py \
  --run vidore "$VIDORE_DATA/MMQA_dev.jsonl" "$VIDORE_OUT/vidore-v3_dev_graph_ppr_base.prediction.json" \
  --run mmdocir "$MMDOCIR_DATA/MMQA_dev.jsonl" "$MMDOCIR_OUT/mmdocir_dev_graph_ppr_base.prediction.json" \
  --run opendocvqa "$OPENDOC_DATA/MMQA_dev.jsonl" "$OPENDOC_OUT/opendocvqa_denseheavy125_medium_both.prediction.json" \
  --hit-k 4 \
  --boundary-k 20 \
  --adjacent-window 2 \
  --topn 50 \
  --output-json "$FAILURE_AUDIT_DIR/graph_page_preserve_limitation_report_rich.json" \
  --output-md "$FAILURE_AUDIT_DIR/graph_page_preserve_limitation_report_rich.md" \
  --output-csv "$FAILURE_AUDIT_DIR/graph_page_preserve_limitation_report_rich_cases.csv"
```

This is an oracle audit, not a routing method. Use it to report dataset limitations and decide which
new evidence source is worth testing next.

The Markdown output is a limitation report. It includes document-retrieval gaps, rank-boundary
localization, same-document page confusion, deep/missing right-document pages, retrievability
ceilings, primary category-by-limitation matrices, exact gold-page rank histograms, rank-5 gold
counts, limitation-by-page-rank and limitation-by-doc-rank matrices, query-cue slices, gold-label
shape, top-k evidence tags, score-margin diagnostics, metadata hotspots by field, and example
failures by limitation group. The CSV is the best artifact for custom pivot tables.

Observed limitation report for frozen Graph-PPR outputs:

| Dataset | qids | page hit@4 | doc hit@4 | page failures | document retrieval gap | same-document page confusion | rank-boundary localization | right-doc deep/missing page |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ViDoRe V3 | 14514 | 9383 | 13206 | 5131 | 1308 / 5131 = 25.5% | 1340 / 5131 = 26.1% | 2228 / 5131 = 43.4% | 255 / 5131 = 5.0% |
| MMDocIR | 1658 | 1114 | 1353 | 544 | 305 / 544 = 56.1% | 74 / 544 = 13.6% | 147 / 544 = 27.0% | 18 / 544 = 3.3% |
| OpenDocVQA | 41017 | 26173 | 26901 | 14844 | 14116 / 14844 = 95.1% | 119 / 14844 = 0.8% | 597 / 14844 = 4.0% | 12 / 14844 = 0.1% |

Primary failure category split from the latest rich run:

| Dataset | doc_miss_topk | doc_missing_from_pool | right_doc_boundary_page | right_doc_adjacent_page | right_doc_same_doc_sibling | right_doc_late_page | right_doc_gold_page_missing_from_pool |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ViDoRe V3 | 1235 | 73 | 2228 | 335 | 1005 | 241 | 14 |
| MMDocIR | 248 | 57 | 147 | 15 | 59 | 15 | 3 |
| OpenDocVQA | 13552 | 564 | 597 | 5 | 114 | 11 | 1 |

Rank and document-position diagnostics:

| Dataset | rank-5 gold failures | right-doc failures | rank-5 gold within right-doc failures | dominant failed gold-doc bucket |
| --- | ---: | ---: | ---: | --- |
| ViDoRe V3 | 470 / 5131 = 9.2% | 3823 | 449 / 3823 = 11.7% | `top4` = 3823 |
| MMDocIR | 16 / 544 = 2.9% | 239 | 15 / 239 = 6.3% | `top4` = 239 |
| OpenDocVQA | 1184 / 14844 = 8.0% | 728 | 207 / 728 = 28.4% | `doc_5_10` = 5220 |

Findings:

1. ViDoRe is mostly a page-local failure problem after the right document is already present. With `--boundary-k 20`, rank-boundary localization is the largest bucket, followed by same-document confusion, so exact MaxSim boundary checks, page content evidence, OCR/layout regions, and adjacent-page traps are plausible targets.
2. MMDocIR is mixed, but document discovery is the largest limitation. Page-local rescue can only attack the 239 right-document failures; 305 failures need stronger document/support recall.
3. OpenDocVQA is not primarily a page-local boundary problem under the current packed-document setup. More than 95% of page failures are document-retrieval gaps, explaining why the full-dev no-support content posterior run lost many existing hits.
4. Rank-5 gold is useful but not enough. The direct right-document rank-5 target is only 449 ViDoRe, 15 MMDocIR, and 207 OpenDocVQA failures, so a top4-vs-rank5 page verifier cannot solve the dominant failure modes without wider boundary candidates or better document/pack selection.

### MMLongBench DocQA

```bash
cd /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
source mmlongbench/env_hpc.sh
```

Prepare:

```bash
"$REPO_ROOT/env/bin/python" mmlongbench/prepare_mmlongbench.py \
  --download \
  --snapshot-dir "$MMLONGBENCH_WORK_ROOT/hf_snapshot/MMLongBench" \
  --output-root "$LOCAL_DATA_DIR/mmlongbench-docqa"
```

Observed prepare:

```text
doc_count=708
page_count=30917
qa_count=14466
missing_gold_page_count=19
task_counts={'longdocurl': 5039, 'mmlongdoc': 4160, 'slidevqa': 5267}
```

Embed:

```bash
sbatch --time=12:00:00 --array=0-31 --export=ALL,NUM_SHARDS=32,BATCH_SIZE=2 \
  mmlongbench/sbatch_embed_mmlongbench_array.sh
```

After embedding, follow the standard page-labeled pipeline with:

```text
DATA_NAME=mmlongbench-docqa
DATA_ROOT="$LOCAL_DATA_DIR/mmlongbench-docqa"
EMBEDDING_DIR="$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_mmlongbench-docqa_dev"
INDEX_DIR="$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_mmlongbench-docqa_dev_pageindex_ivfflat"
```

### DUDE

```bash
cd /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
source dude/env_hpc.sh
```

Prepare with `Amazon_original`; do not use `Amazon_due` for SPLADE text:

```bash
"$REPO_ROOT/env/bin/python" dude/prepare_dude.py \
  --output-root "$LOCAL_DATA_DIR/dude" \
  --hf-config Amazon_original \
  --source-split val
```

Observed prepare:

```text
doc_count=732
page_count=4083
qa_count=2903
skipped_no_gold_page_count=3412
answer_page_base=0
answer_page_base_missing_counts={'0': 17, '1': 1884}
answer_type_counts_kept={'extractive': 2586, 'list/extractive': 317}
```

Observed OCR sanity after parser fix:

```text
pages=4086
nonempty_text_pages=4020
empty_text_pages=66
empty_fraction=0.01615
bad_succeeded_prefix_pages=1
missing_images=0
```

Observed dense baseline retrieval:

```text
n_qids=2903
page_recall@4=0.5354403490641865
page_recall@20=0.6543920082673097
doc_recall@4=0.6114364450568378
doc_recall@20=0.7247674819152601
page_hit@4=1565
doc_hit@4=1775
```

Observed `plain_top224` retrieval:

```text
n_qids=2903
page_recall@4=0.5719543001492708
page_recall@20=0.6866134649978949
doc_recall@4=0.6507061660351361
doc_recall@20=0.7561143644505683
page_hit@4=1672
doc_hit@4=1889
improved_doc_rank_count=781
```

Observed SPLADE/doc-RRF:

```text
n_qids=2903
page_recall@4=0.5311861292915375
page_recall@20=0.6180139319477934
doc_recall@4=0.6631071305545987
doc_recall@20=0.7747158112297623
page_hit@4=1558
doc_hit@4=1925
```

Run Graph-PPR next:

```bash
DATA_NAME=dude \
DATA_ROOT="$LOCAL_DATA_DIR/dude" \
DENSE_PRED="$LOCAL_OUTPUT_DIR/dude/plain_top224_ret1000_prediction.json" \
SPARSE_PRED="$LOCAL_OUTPUT_DIR/dude/doc_rrf_plain_top224_splade/dude_splade_ret1000.prediction.json" \
OUT_DIR="$LOCAL_OUTPUT_DIR/dude/graph_ppr_plain_top224_splade" \
GRAPH_PROFILE=page_rank_probe \
GRAPH_LABEL="dude_denseheavy125_medium_both" \
FINAL_TOP_PAGES=1000 \
PER_DOC_PAGE_LIMIT=0 \
DENSE_WEIGHT=1.25 \
SPARSE_WEIGHT=0.75 \
bash scripts/run_external_graph_ppr_pipeline.sh
```

Then follow the standard page-labeled pipeline with:

```text
DATA_NAME=dude
DATA_ROOT="$LOCAL_DATA_DIR/dude"
EMBEDDING_DIR="$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_dude_dev"
INDEX_DIR="$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_dude_dev_pageindex_ivfflat"
```

### M3DocVQA/MMQA

MMQA has unreliable gold page labels for full-dev; use document recall for retrieval and EM/F1 for end-to-end QA.

Clean apples-to-apples baseline check:

```bash
cd /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
export M3DOCVQA_LOCAL_DATA_DIR="$PWD/data"
export M3DOCVQA_LOCAL_MODEL_DIR="$PWD/model"
export M3DOCVQA_LOCAL_EMBEDDINGS_DIR="$PWD/embeddings"
export M3DOCVQA_LOCAL_OUTPUT_DIR="$PWD/output"
OUTROOT=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_apples_ret4_vs_ret1000_current
mkdir -p "$OUTROOT"

SPLIT=dev \
FAISS_INDEX_TYPE=ivfflat \
FAISS_NPROBE=4 \
N_RETRIEVAL_PAGES=4 \
BASELINE_RETR_OUT_DIR="$OUTROOT/retrieval_ret4" \
BASELINE_LABEL="mmqa_dev_baseline_current_ret4_nprobe4" \
bash scripts/run_m3docvqa_baseline_retrieval.sh

SPLIT=dev \
FAISS_INDEX_TYPE=ivfflat \
FAISS_NPROBE=4 \
N_RETRIEVAL_PAGES=1000 \
BASELINE_RETR_OUT_DIR="$OUTROOT/retrieval_ret1000" \
BASELINE_LABEL="mmqa_dev_baseline_current_ret1000_nprobe4" \
bash scripts/run_m3docvqa_baseline_retrieval.sh
```

Then run the same QA adapter on top-4 pages:

```bash
GOLD="$PWD/data/m3-docvqa/multimodalqa/MMQA_dev.jsonl"
QAOUT="$OUTROOT/qa_top4"
mkdir -p "$QAOUT"

"$PWD/env/bin/python" scripts/run_m3docvqa_external_retrieval_qa.py \
  --prediction-json "$OUTROOT/retrieval_ret4/mmqa_dev_baseline_current_ret4_nprobe4.prediction.json" \
  --gold "$GOLD" \
  --data-name m3-docvqa \
  --split dev \
  --model-name-or-path Qwen2-VL-7B-Instruct \
  --bits 16 \
  --qa-top-pages 4 \
  --doc-image-cache-size 16 \
  --save-every 25 \
  --resume \
  --run-eval \
  --output-prediction-json "$QAOUT/mmqa_dev_baseline_current_ret4_qwen2vl_top4.prediction.json" \
  --output-eval-json "$QAOUT/mmqa_dev_baseline_current_ret4_qwen2vl_top4.eval.json"

"$PWD/env/bin/python" scripts/run_m3docvqa_external_retrieval_qa.py \
  --prediction-json "$OUTROOT/retrieval_ret1000/mmqa_dev_baseline_current_ret1000_nprobe4.prediction.json" \
  --gold "$GOLD" \
  --data-name m3-docvqa \
  --split dev \
  --model-name-or-path Qwen2-VL-7B-Instruct \
  --bits 16 \
  --qa-top-pages 4 \
  --doc-image-cache-size 16 \
  --save-every 25 \
  --resume \
  --run-eval \
  --output-prediction-json "$QAOUT/mmqa_dev_baseline_current_ret1000_qwen2vl_top4.prediction.json" \
  --output-eval-json "$QAOUT/mmqa_dev_baseline_current_ret1000_qwen2vl_top4.eval.json"
```

## Scoreboards And Handoffs

- MMQA scoreboard: `notes/mmqa_retrieval_scoreboard.md`
- Cross-dataset retrieval scoreboard: `notes/cross_dataset_retrieval_scoreboard.md`
- Graph-PPR external handoff: `notes/graph_ppr_external_datasets_handoff_2026-05-21.md`
- Visual/Graph historical handoff: `notes/visual_reranker_handoff_2026-04-27.md`
- Dataset-specific READMEs:
  - `mmdocir/README.md`
  - `sciegqa/README.md`
  - `vidore/README.md`
  - `vidoseek/README.md`
  - `opendocvqa/README.md`
  - `mmlongbench/README.md`
  - `dude/README.md`
