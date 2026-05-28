# MMLongBench DocQA Workflow

This workflow prepares the page-labeled DocQA part of MMLongBench for the same retrieval stack used by MMDocIR, ViDoRe, SciEGQA, and ViDoSeek.

It currently targets:

- `longdocurl`
- `mmlongdoc`
- `slidevqa`

across `K8`, `K16`, `K32`, `K64`, and `K128`. These subsets expose `ans_page_list`, so they support exact page retrieval evaluation.

## Prepare

```bash
cd /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
source mmlongbench/env_hpc.sh

"$REPO_ROOT/env/bin/python" mmlongbench/prepare_mmlongbench.py \
  --download \
  --snapshot-dir "$MMLONGBENCH_WORK_ROOT/hf_snapshot/MMLongBench" \
  --output-root "$LOCAL_DATA_DIR/mmlongbench-docqa"
```

For a quick smoke test:

```bash
"$REPO_ROOT/env/bin/python" mmlongbench/prepare_mmlongbench.py \
  --download \
  --snapshot-dir "$MMLONGBENCH_WORK_ROOT/hf_snapshot/MMLongBench" \
  --output-root "$LOCAL_DATA_DIR/mmlongbench-docqa-smoke" \
  --length K8 \
  --max-examples-per-file 5
```

Expected outputs:

```text
$LOCAL_DATA_DIR/mmlongbench-docqa/MMQA_dev.jsonl
$LOCAL_DATA_DIR/mmlongbench-docqa/qids_dev.jsonl
$LOCAL_DATA_DIR/mmlongbench-docqa/gold_pages_dev.jsonl
$LOCAL_DATA_DIR/mmlongbench-docqa/doc_pages_dev.jsonl
$LOCAL_DATA_DIR/mmlongbench-docqa/prepare_dev_summary.json
```

## Embed Pages

```bash
sbatch --time=12:00:00 --array=0-31 --export=ALL,NUM_SHARDS=32,BATCH_SIZE=2 \
  mmlongbench/sbatch_embed_mmlongbench_array.sh
```

Check completion:

```bash
find "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_mmlongbench-docqa_dev" -name "*.safetensors" | wc -l
"$REPO_ROOT/env/bin/python" - <<'PY'
import json, os
p=os.environ["LOCAL_DATA_DIR"] + "/mmlongbench-docqa/dev_doc_ids.json"
print(len(json.load(open(p))))
PY
```

## Build Dense Index

```bash
"$REPO_ROOT/env/bin/python" mmdocir/run_indexing_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/mmlongbench-docqa" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_mmlongbench-docqa_dev" \
  --output-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_mmlongbench-docqa_dev_pageindex_ivfflat" \
  --faiss-index-type ivfflat
```

## Dense Retrieval

```bash
mkdir -p "$LOCAL_OUTPUT_DIR/mmlongbench-docqa"

"$REPO_ROOT/env/bin/python" mmdocir/run_retrieval_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/mmlongbench-docqa" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_mmlongbench-docqa_dev" \
  --index-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_mmlongbench-docqa_dev_pageindex_ivfflat" \
  --output-json "$LOCAL_OUTPUT_DIR/mmlongbench-docqa/baseline_ret1000.json" \
  --n-retrieval-pages 1000 \
  --faiss-nprobe 4

"$REPO_ROOT/env/bin/python" mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/mmlongbench-docqa/baseline_ret1000.json" \
  --gold "$LOCAL_DATA_DIR/mmlongbench-docqa/MMQA_dev.jsonl" \
  --recall-k 1 2 4 5 10 20 50 100
```

## Plain Top-224

```bash
bash mmlongbench/run_plain_top224_mmlongbench.sh

"$REPO_ROOT/env/bin/python" mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/mmlongbench-docqa/plain_top224_ret1000_prediction.json" \
  --gold "$LOCAL_DATA_DIR/mmlongbench-docqa/MMQA_dev.jsonl" \
  --recall-k 1 2 4 5 10 20 50 100
```

Batch version:

```bash
sbatch mmlongbench/sbatch_plain_top224_mmlongbench.sh
```

## SPLADE + Graph-PPR

The converter stores `page_text_list` into the page manifest as `text` and `ocr_text`, so the SPLADE pipeline can use manifest text directly.

```bash
DATA_NAME=mmlongbench-docqa \
DATA_ROOT="$LOCAL_DATA_DIR/mmlongbench-docqa" \
DENSE_PRED="$LOCAL_OUTPUT_DIR/mmlongbench-docqa/plain_top224_ret1000_prediction.json" \
OUT_DIR="$LOCAL_OUTPUT_DIR/mmlongbench-docqa/doc_rrf_plain_top224_splade" \
DENSE_WEIGHT=1.25 \
SPARSE_WEIGHT=0.75 \
RRF_K=10 \
SPLADE_DEVICE=auto \
bash scripts/run_external_doc_rrf_pipeline.sh
```

Then run the page-labeled Graph-PPR default:

```bash
DATA_NAME=mmlongbench-docqa \
DATA_ROOT="$LOCAL_DATA_DIR/mmlongbench-docqa" \
DENSE_PRED="$LOCAL_OUTPUT_DIR/mmlongbench-docqa/plain_top224_ret1000_prediction.json" \
SPARSE_PRED="$LOCAL_OUTPUT_DIR/mmlongbench-docqa/doc_rrf_plain_top224_splade/mmlongbench-docqa_splade_ret1000.prediction.json" \
OUT_DIR="$LOCAL_OUTPUT_DIR/mmlongbench-docqa/graph_ppr_plain_top224_splade" \
GRAPH_PROFILE=page_rank_probe \
GRAPH_LABEL="mmlongbench_docqa_denseheavy125_medium_both" \
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

Batch version:

```bash
sbatch mmlongbench/sbatch_splade_graph_ppr_mmlongbench.sh
```
