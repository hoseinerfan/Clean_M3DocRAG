# MP-DocVQA for M3DocRAG/CAPP Transfer

This folder prepares MP-DocVQA for the same page-retrieval workflow used by
MMDocIR, MMLongBench DocQA, OpenDocVQA, DUDE, SciEGQA, ViDoSeek, and ViDoRe.

MP-DocVQA is useful here because it is a closed-domain multi-page DocVQA
benchmark with answer-page supervision. That makes it a cleaner transfer test
for CAPP than datasets whose page labels or document structure differ strongly
from M3DocVQA.

The converter emits:

```text
$LOCAL_DATA_DIR/mpdocvqa/MMQA_dev.jsonl
$LOCAL_DATA_DIR/mpdocvqa/qids_dev.jsonl
$LOCAL_DATA_DIR/mpdocvqa/gold_pages_dev.jsonl
$LOCAL_DATA_DIR/mpdocvqa/doc_pages_dev.jsonl
$LOCAL_DATA_DIR/mpdocvqa/dev_doc_ids.json
$LOCAL_DATA_DIR/mpdocvqa/prepare_dev_summary.json
```

## 1. Environment

```bash
cd /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
source mpdocvqa/env_hpc.sh
```

## 2. Put the Original Dataset on HPC

Create a source directory such as:

```text
/mmfs1/scratch/jacks.local/aerfanshekooh/custom/MPDocVQA_raw
```

Place the official MP-DocVQA annotation file and page images there. The
converter supports JSON, JSONL, and NumPy `.npy` annotation files. If your
annotation filename is not one of the common names, pass it explicitly with
`--annotation-file`.

## 3. Convert MP-DocVQA

For the normal validation split converted to our `dev` split:

```bash
RAW=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/MPDocVQA_raw

"$REPO_ROOT/env/bin/python" mpdocvqa/prepare_mpdocvqa.py \
  --input-root "$RAW" \
  --image-root "$RAW" \
  --source-split val \
  --split dev \
  --output-root "$LOCAL_DATA_DIR/mpdocvqa"
```

If the annotation file has a specific name:

```bash
"$REPO_ROOT/env/bin/python" mpdocvqa/prepare_mpdocvqa.py \
  --input-root "$RAW" \
  --image-root "$RAW/images" \
  --annotation-file path/relative/to/raw/imdb_val.npy \
  --source-split val \
  --split dev \
  --output-root "$LOCAL_DATA_DIR/mpdocvqa"
```

Smoke test first if the full conversion is slow:

```bash
"$REPO_ROOT/env/bin/python" mpdocvqa/prepare_mpdocvqa.py \
  --input-root "$RAW" \
  --image-root "$RAW" \
  --source-split val \
  --split dev \
  --output-root "$LOCAL_DATA_DIR/mpdocvqa-smoke" \
  --max-examples 20
```

## 4. Sanity Check Conversion

```bash
cat "$LOCAL_DATA_DIR/mpdocvqa/prepare_dev_summary.json"

"$REPO_ROOT/env/bin/python" - <<'PY'
import json, os
from pathlib import Path

root = Path(os.environ["LOCAL_DATA_DIR"]) / "mpdocvqa"
summary = json.load(open(root / "prepare_dev_summary.json"))
print("doc_count", summary["doc_count"])
print("page_count", summary["page_count"])
print("qa_count", summary["qa_count"])
print("labeled_qids", summary["labeled_qids"])
print("missing_image_count", summary["missing_image_count"])
print("missing_gold_page_count", summary["missing_gold_page_count"])
print("source_schema_counts", summary["source_schema_counts"])
print("first_gold")
with open(root / "gold_pages_dev.jsonl", encoding="utf-8") as f:
    print(next(f).strip())
print("first_page")
with open(root / "doc_pages_dev.jsonl", encoding="utf-8") as f:
    print(next(f).strip())
PY
```

You want `missing_image_count=0`, `missing_gold_page_count=0`, and most or all
questions to have at least one gold page.

## 5. Embed Pages

```bash
sbatch --time=12:00:00 --array=0-31 --export=ALL,NUM_SHARDS=32,BATCH_SIZE=2 \
  mpdocvqa/sbatch_embed_mpdocvqa_array.sh
```

Check completion:

```bash
find "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_mpdocvqa_dev" -name "*.safetensors" | wc -l
"$REPO_ROOT/env/bin/python" - <<'PY'
import json, os
p=os.environ["LOCAL_DATA_DIR"] + "/mpdocvqa/dev_doc_ids.json"
print(len(json.load(open(p))))
PY
```

## 6. Build Dense Index and Retrieve

```bash
sbatch mpdocvqa/sbatch_index_retrieve_mpdocvqa.sh
```

Foreground version:

```bash
"$REPO_ROOT/env/bin/python" mmdocir/run_indexing_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/mpdocvqa" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_mpdocvqa_dev" \
  --output-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_mpdocvqa_dev_pageindex_ivfflat" \
  --faiss-index-type ivfflat

"$REPO_ROOT/env/bin/python" mmdocir/run_retrieval_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/mpdocvqa" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_mpdocvqa_dev" \
  --index-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_mpdocvqa_dev_pageindex_ivfflat" \
  --output-json "$LOCAL_OUTPUT_DIR/mpdocvqa/baseline_ret1000.json" \
  --n-retrieval-pages 1000 \
  --faiss-nprobe 4 \
  --resume \
  --save-every 100
```

Evaluate dense retrieval:

```bash
"$REPO_ROOT/env/bin/python" mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/mpdocvqa/baseline_ret1000.json" \
  --gold "$LOCAL_DATA_DIR/mpdocvqa/MMQA_dev.jsonl" \
  --recall-k 1 2 4 5 10 20 50 100 1000
```

## 7. Plain Top-224 MaxSim Approximation

```bash
bash mpdocvqa/run_plain_top224_mpdocvqa.sh

"$REPO_ROOT/env/bin/python" mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/mpdocvqa/plain_top224_ret1000_prediction.json" \
  --gold "$LOCAL_DATA_DIR/mpdocvqa/MMQA_dev.jsonl" \
  --recall-k 1 2 4 5 10 20 50 100 1000
```

## 8. Apply the Trained M3DocVQA CAPP Model

Use this after `plain_top224_ret1000_prediction.json` exists. The model is not
trained on MP-DocVQA; gold pages are used only for evaluation.

```bash
MODEL_JSON="$REPO_ROOT/output/m3docvqa_content_aware_exact_maxsim_direct_exactonly_adaptive_norm05/mmqa_train_to_dev_content_aware_fixed_alpha_0p40_base_exact_maxsim_gpp_direct_exactonly_adaptive_norm05.model.json"
OUT="$LOCAL_OUTPUT_DIR/mpdocvqa/trained_capp_transfer"
mkdir -p "$OUT"

"$REPO_ROOT/env/bin/python" scripts/apply_trained_content_aware_page_reranker.py \
  --model-json "$MODEL_JSON" \
  --base-pred "$LOCAL_OUTPUT_DIR/mpdocvqa/plain_top224_ret1000_prediction.json" \
  --page-text-jsonl "$LOCAL_DATA_DIR/mpdocvqa/doc_pages_dev.jsonl" \
  --gold "$LOCAL_DATA_DIR/mpdocvqa/MMQA_dev.jsonl" \
  --candidate-top-k 1000 \
  --inference-mode blend_rerank \
  --blend-alpha 0.20 \
  --recall-k 1 2 4 5 10 20 50 100 1000 \
  --output-prediction-json "$OUT/mpdocvqa_capp_transfer_a0p20.prediction.json" \
  --output-summary-json "$OUT/mpdocvqa_capp_transfer_a0p20.summary.json" \
  --output-table-md "$OUT/mpdocvqa_capp_transfer_a0p20.table.md" \
  --output-prior-jsonl "$OUT/mpdocvqa_capp_transfer_a0p20.prior.jsonl"

cat "$OUT/mpdocvqa_capp_transfer_a0p20.table.md"
```

If alpha 0.20 helps, also test 0.40:

```bash
"$REPO_ROOT/env/bin/python" scripts/apply_trained_content_aware_page_reranker.py \
  --model-json "$MODEL_JSON" \
  --base-pred "$LOCAL_OUTPUT_DIR/mpdocvqa/plain_top224_ret1000_prediction.json" \
  --page-text-jsonl "$LOCAL_DATA_DIR/mpdocvqa/doc_pages_dev.jsonl" \
  --gold "$LOCAL_DATA_DIR/mpdocvqa/MMQA_dev.jsonl" \
  --candidate-top-k 1000 \
  --inference-mode blend_rerank \
  --blend-alpha 0.40 \
  --recall-k 1 2 4 5 10 20 50 100 1000 \
  --output-prediction-json "$OUT/mpdocvqa_capp_transfer_a0p40.prediction.json" \
  --output-summary-json "$OUT/mpdocvqa_capp_transfer_a0p40.summary.json" \
  --output-table-md "$OUT/mpdocvqa_capp_transfer_a0p40.table.md" \
  --output-prior-jsonl "$OUT/mpdocvqa_capp_transfer_a0p40.prior.jsonl"

cat "$OUT/mpdocvqa_capp_transfer_a0p40.table.md"
```
