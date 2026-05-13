# SciEGQA-Bench for M3DocRAG and `plain_top224`

This folder prepares SciEGQA-Bench in a separate HPC workspace and converts its 1-based evidence pages into M3DocRAG-style zero-based page labels.

Default HPC work root:

```bash
/mmfs1/scratch/jacks.local/aerfanshekooh/custom/SciEGQA_M3DocRAG
```

## 1. Environment

```bash
cd /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
source sciegqa/env_hpc.sh
```

## 2. Download and convert SciEGQA-Bench

```bash
python sciegqa/prepare_sciegqa_bench.py \
  --download \
  --snapshot-dir "$SciEGQA_WORK_ROOT/hf_snapshot/SciEGQA-Bench" \
  --output-root "$LOCAL_DATA_DIR/sci-egqa-bench"
```

Expected outputs:

- `$LOCAL_DATA_DIR/sci-egqa-bench/MMQA_dev.jsonl`
- `$LOCAL_DATA_DIR/sci-egqa-bench/dev_doc_ids.json`
- `$LOCAL_DATA_DIR/sci-egqa-bench/qids_dev.jsonl`
- `$LOCAL_DATA_DIR/sci-egqa-bench/gold_pages_dev.jsonl`
- `$LOCAL_DATA_DIR/sci-egqa-bench/doc_pages_dev.jsonl`
- `$LOCAL_DATA_DIR/sci-egqa-bench/images_raw/`

## 3. Embed pages

SciEGQA-Bench is much smaller than MMDocIR, but use a GPU job anyway:

```bash
sbatch sciegqa/sbatch_embed_sciegqa_array.sh
```

Or run a small smoke test on an allocated GPU:

```bash
python mmdocir/run_page_embedding_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/sci-egqa-bench" \
  --output-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_sci-egqa-bench_dev" \
  --retrieval-model-name-or-path colpaligemma-3b-pt-448-base \
  --retrieval-adapter-model-name-or-path colpali-v1.2 \
  --per-device-eval-batch-size 2 \
  --max-docs 2 \
  --resume
```

## 4. Build FAISS index

```bash
python mmdocir/run_indexing_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/sci-egqa-bench" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_sci-egqa-bench_dev" \
  --output-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_sci-egqa-bench_dev_pageindex_ivfflat" \
  --faiss-index-type ivfflat
```

## 5. Run baseline retrieval

```bash
python mmdocir/run_retrieval_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/sci-egqa-bench" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_sci-egqa-bench_dev" \
  --index-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_sci-egqa-bench_dev_pageindex_ivfflat" \
  --output-json "$LOCAL_OUTPUT_DIR/sciegqa/baseline_ret1000.json" \
  --n-retrieval-pages 1000 \
  --faiss-nprobe 4
```

Evaluate exact evidence-page retrieval:

```bash
python mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/sciegqa/baseline_ret1000.json" \
  --gold "$LOCAL_DATA_DIR/sci-egqa-bench/MMQA_dev.jsonl"
```

## 6. Run `plain_top224`

```bash
bash sciegqa/run_plain_top224_sciegqa.sh
```

Then evaluate:

```bash
python mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/sciegqa/plain_top224_ret1000_prediction.json" \
  --gold "$LOCAL_DATA_DIR/sci-egqa-bench/MMQA_dev.jsonl"
```

