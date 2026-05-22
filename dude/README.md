# DUDE Workflow

This workflow prepares DUDE for the same page-retrieval stack used by MMDocIR, ViDoRe, SciEGQA, ViDoSeek, and MMLongBench.

DUDE is a multi-page document QA benchmark. The public loader exposes PDF paths, OCR paths, question/answer annotations, and answer page bounding boxes. The converter below builds an exact page-labeled retrieval split by default and skips rows without answer page boxes, because those rows do not have an exact page retrieval target.

## Prepare

```bash
cd /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
source dude/env_hpc.sh

"$REPO_ROOT/env/bin/python" dude/prepare_dude.py \
  --output-root "$LOCAL_DATA_DIR/dude" \
  --hf-config Amazon_original \
  --source-split val
```

`prepare_dude.py` does not use `datasets.load_dataset` anymore. It downloads the public annotation JSON and DUDE binary tarball directly into `$LOCAL_DATA_DIR/dude/downloads`, then extracts PDFs/OCR under `$LOCAL_DATA_DIR/dude/raw`. This avoids the newer Hugging Face `trust_remote_code` block and avoids filling the default HF cache.

If you already have the extracted `DUDE_train-val-test_binaries` directory:

```bash
"$REPO_ROOT/env/bin/python" dude/prepare_dude.py \
  --data-dir /path/to/DUDE_train-val-test_binaries \
  --output-root "$LOCAL_DATA_DIR/dude" \
  --hf-config Amazon_original \
  --source-split val
```

If you already downloaded the public annotations too:

```bash
"$REPO_ROOT/env/bin/python" dude/prepare_dude.py \
  --data-dir /path/to/DUDE_train-val-test_binaries \
  --annotations-json /path/to/2023-03-23_DUDE_gt_test_PUBLIC.json \
  --output-root "$LOCAL_DATA_DIR/dude" \
  --hf-config Amazon_original \
  --source-split val
```

For a quick smoke test:

```bash
"$REPO_ROOT/env/bin/python" dude/prepare_dude.py \
  --output-root "$LOCAL_DATA_DIR/dude-smoke" \
  --hf-config Amazon_original \
  --source-split val \
  --max-docs 5
```

Expected outputs:

```text
$LOCAL_DATA_DIR/dude/MMQA_dev.jsonl
$LOCAL_DATA_DIR/dude/qids_dev.jsonl
$LOCAL_DATA_DIR/dude/gold_pages_dev.jsonl
$LOCAL_DATA_DIR/dude/doc_pages_dev.jsonl
$LOCAL_DATA_DIR/dude/prepare_dev_summary.json
```

Check the page-base decision after prepare:

```bash
"$REPO_ROOT/env/bin/python" - <<'PY'
import json, os
p=os.environ["LOCAL_DATA_DIR"] + "/dude/prepare_dev_summary.json"
s=json.load(open(p))
for k in ["source_row_count","qa_count","doc_count","page_count","answer_page_base","answer_page_base_missing_counts","skipped_no_gold_page_count","missing_gold_page_count","answer_type_counts_kept"]:
    print(k, s.get(k))
PY
```

## Embed Pages

```bash
sbatch --time=12:00:00 --array=0-31 --export=ALL,NUM_SHARDS=32,BATCH_SIZE=2 \
  dude/sbatch_embed_dude_array.sh
```

Check completion:

```bash
find "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_dude_dev" -name "*.safetensors" | wc -l
"$REPO_ROOT/env/bin/python" - <<'PY'
import json, os
p=os.environ["LOCAL_DATA_DIR"] + "/dude/dev_doc_ids.json"
print(len(json.load(open(p))))
PY
```

## Build Dense Index

```bash
"$REPO_ROOT/env/bin/python" mmdocir/run_indexing_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/dude" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_dude_dev" \
  --output-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_dude_dev_pageindex_ivfflat" \
  --faiss-index-type ivfflat
```

## Dense Retrieval

```bash
mkdir -p "$LOCAL_OUTPUT_DIR/dude"

"$REPO_ROOT/env/bin/python" mmdocir/run_retrieval_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/dude" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_dude_dev" \
  --index-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_dude_dev_pageindex_ivfflat" \
  --output-json "$LOCAL_OUTPUT_DIR/dude/baseline_ret1000.json" \
  --n-retrieval-pages 1000 \
  --faiss-nprobe 4

"$REPO_ROOT/env/bin/python" mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/dude/baseline_ret1000.json" \
  --gold "$LOCAL_DATA_DIR/dude/MMQA_dev.jsonl" \
  --recall-k 1 2 4 5 10 20 50 100
```

## Plain Top-224

```bash
bash dude/run_plain_top224_dude.sh

"$REPO_ROOT/env/bin/python" mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/dude/plain_top224_ret1000_prediction.json" \
  --gold "$LOCAL_DATA_DIR/dude/MMQA_dev.jsonl" \
  --recall-k 1 2 4 5 10 20 50 100
```

## SPLADE + Graph-PPR

Use `Amazon_original` for prepare because the current converter extracts text from Amazon Textract `Blocks`. The converter stores OCR text in the page manifest as `text` and `ocr_text`, so the SPLADE pipeline can use manifest text directly.

```bash
DATA_NAME=dude \
DATA_ROOT="$LOCAL_DATA_DIR/dude" \
DENSE_PRED="$LOCAL_OUTPUT_DIR/dude/plain_top224_ret1000_prediction.json" \
OUT_DIR="$LOCAL_OUTPUT_DIR/dude/doc_rrf_plain_top224_splade" \
DENSE_WEIGHT=1.25 \
SPARSE_WEIGHT=0.75 \
RRF_K=10 \
SPLADE_DEVICE=auto \
bash scripts/run_external_doc_rrf_pipeline.sh
```

Then run the page-labeled Graph-PPR default:

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
