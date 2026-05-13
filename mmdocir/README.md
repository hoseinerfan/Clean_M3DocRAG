# MMDocIR for M3DocRAG and `plain_top224`

This folder is the MMDocIR side workspace. It keeps MMDocIR data, embeddings, outputs, and run helpers separate from the existing M3DocVQA setup.

Default HPC work root:

```bash
/mmfs1/scratch/jacks.local/aerfanshekooh/custom/MMDocIR_M3DocRAG
```

## 1. Environment

```bash
cd /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
source mmdocir/env_hpc.sh
```

## 2. Download and convert MMDocIR

This writes M3DocVQA-like gold files plus page JPEGs under `$LOCAL_DATA_DIR/mm-docir`.

```bash
python mmdocir/prepare_mmdocir.py \
  --download \
  --snapshot-dir "$MMDocIR_WORK_ROOT/hf_snapshot/MMDocIR_Evaluation_Dataset" \
  --output-root "$LOCAL_DATA_DIR/mm-docir"
```

Expected outputs:

- `$LOCAL_DATA_DIR/mm-docir/MMQA_dev.jsonl`
- `$LOCAL_DATA_DIR/mm-docir/dev_doc_ids.json`
- `$LOCAL_DATA_DIR/mm-docir/qids_dev.jsonl`
- `$LOCAL_DATA_DIR/mm-docir/gold_pages_dev.jsonl`
- `$LOCAL_DATA_DIR/mm-docir/doc_pages_dev.jsonl`
- `$LOCAL_DATA_DIR/mm-docir/pages_dev/<doc_id>/<page_idx>.jpg`

## 3. Embed pages with ColPali

```bash
python mmdocir/run_page_embedding_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/mm-docir" \
  --output-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_mm-docir_dev" \
  --retrieval-model-name-or-path colpaligemma-3b-pt-448-base \
  --retrieval-adapter-model-name-or-path colpali-v1.2 \
  --per-device-eval-batch-size 4 \
  --resume
```

## 4. Build FAISS index

```bash
python mmdocir/run_indexing_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/mm-docir" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_mm-docir_dev" \
  --output-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_mm-docir_dev_pageindex_ivfflat" \
  --faiss-index-type ivfflat
```

## 5. Run baseline M3DocRAG-style retrieval

```bash
python mmdocir/run_retrieval_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/mm-docir" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_mm-docir_dev" \
  --index-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_mm-docir_dev_pageindex_ivfflat" \
  --output-json "$LOCAL_OUTPUT_DIR/mmdocir/baseline_ret1000.json" \
  --n-retrieval-pages 1000 \
  --faiss-nprobe 4
```

Evaluate exact gold-page retrieval:

```bash
python mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/mmdocir/baseline_ret1000.json" \
  --gold "$LOCAL_DATA_DIR/mm-docir/MMQA_dev.jsonl"
```

## 6. Run `plain_top224`

```bash
bash mmdocir/run_plain_top224_mmdocir.sh
```

The script writes:

- `$LOCAL_OUTPUT_DIR/mmdocir/plain_top224_ret1000.jsonl`
- `$LOCAL_OUTPUT_DIR/mmdocir/plain_top224_ret1000_summary.json`
- `$LOCAL_OUTPUT_DIR/mmdocir/plain_top224_ret1000_prediction.json`

Then evaluate:

```bash
python mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/mmdocir/plain_top224_ret1000_prediction.json" \
  --gold "$LOCAL_DATA_DIR/mm-docir/MMQA_dev.jsonl"
```

