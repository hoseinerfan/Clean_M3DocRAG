# OpenDocVQA for M3DocRAG and `plain_top224`

This folder prepares OpenDocVQA with the separate OpenDocVQA-Corpus in a dedicated HPC workspace.

Important: OpenDocVQA-Corpus is large and gated on Hugging Face. Accept the dataset terms in the browser and make sure the HPC environment has access, either with `huggingface-cli login` or by exporting `HF_TOKEN`.

Default HPC work root:

```bash
/mmfs1/scratch/jacks.local/aerfanshekooh/custom/OpenDocVQA_M3DocRAG
```

The corpus stores individual document images, not multi-page PDFs. To avoid creating about 170k tiny embedding files, the converter groups images into artificial packs of 64 pages by default. Page-level gold labels remain exact, but doc-level recall is only an artifact of the packing; use page recall as the main metric.

## 1. Environment

Clear any previous dataset env first, then source OpenDocVQA:

```bash
cd /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
unset HF_HOME HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE HF_HUB_CACHE TRANSFORMERS_CACHE XDG_CACHE_HOME
source opendocvqa/env_hpc.sh
```

The env file forces Hugging Face Hub, datasets, transformers, and XDG caches under `$OPENDOCVQA_WORK_ROOT` to avoid home-directory quota failures.

## 2. Download and convert OpenDocVQA

Full benchmark:

```bash
python opendocvqa/prepare_opendocvqa.py \
  --cache-dir "$OPENDOCVQA_WORK_ROOT/hf_cache" \
  --output-root "$LOCAL_DATA_DIR/opendocvqa"
```

This is heavy because OpenDocVQA-Corpus is about 63 GB before rendered-copy output. For a useful first smoke test:

```bash
python opendocvqa/prepare_opendocvqa.py \
  --cache-dir "$OPENDOCVQA_WORK_ROOT/hf_cache" \
  --output-root "$LOCAL_DATA_DIR/opendocvqa_smoke_infovqa" \
  --qa-config infovqa \
  --dataset-name infovqa \
  --corpus-scope relevant_only \
  --max-queries 50
```

Expected full outputs:

- `$LOCAL_DATA_DIR/opendocvqa/MMQA_dev.jsonl`
- `$LOCAL_DATA_DIR/opendocvqa/dev_doc_ids.json`
- `$LOCAL_DATA_DIR/opendocvqa/qids_dev.jsonl`
- `$LOCAL_DATA_DIR/opendocvqa/gold_pages_dev.jsonl`
- `$LOCAL_DATA_DIR/opendocvqa/doc_pages_dev.jsonl`
- `$LOCAL_DATA_DIR/opendocvqa/source_doc_id_map.json`
- `$LOCAL_DATA_DIR/opendocvqa/pages_dev/`

## 3. Sanity check conversion

```bash
python - <<'PY'
import json, os
from pathlib import Path
from PIL import Image

root = Path(os.environ["LOCAL_DATA_DIR"]) / "opendocvqa"
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
sample = pages[0]
path = root / sample["image_path"]
img = Image.open(path)
print("sample_page", sample)
print("sample_exists", path.exists(), path)
print("sample_size", img.size, img.mode)
PY
```

## 4. Embed pages

OpenDocVQA is much larger than the other prepared datasets. The default sbatch uses 16 shards and 24 hours:

```bash
sbatch opendocvqa/sbatch_embed_opendocvqa_array.sh
```

If you need fewer concurrent GPUs, use 8 shards and resubmit with `--resume` if it times out:

```bash
sbatch --time=24:00:00 --array=0-7 --export=ALL,NUM_SHARDS=8,BATCH_SIZE=2 \
  opendocvqa/sbatch_embed_opendocvqa_array.sh
```

Check completion:

```bash
find "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_opendocvqa_dev" -name "*.safetensors" | wc -l
jq length "$LOCAL_DATA_DIR/opendocvqa/dev_doc_ids.json"
```

The two counts should match.

## 5. Build FAISS index

```bash
python mmdocir/run_indexing_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/opendocvqa" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_opendocvqa_dev" \
  --output-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_opendocvqa_dev_pageindex_ivfflat" \
  --faiss-index-type ivfflat
```

## 6. Run baseline retrieval

```bash
mkdir -p "$LOCAL_OUTPUT_DIR/opendocvqa"

python mmdocir/run_retrieval_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/opendocvqa" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_opendocvqa_dev" \
  --index-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_opendocvqa_dev_pageindex_ivfflat" \
  --output-json "$LOCAL_OUTPUT_DIR/opendocvqa/baseline_ret1000.json" \
  --n-retrieval-pages 1000 \
  --faiss-nprobe 4
```

Evaluate exact relevant-image page retrieval:

```bash
python mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/opendocvqa/baseline_ret1000.json" \
  --gold "$LOCAL_DATA_DIR/opendocvqa/MMQA_dev.jsonl"
```

## 7. Run `plain_top224`

```bash
bash opendocvqa/run_plain_top224_opendocvqa.sh
```

Then evaluate:

```bash
python mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/opendocvqa/plain_top224_ret1000_prediction.json" \
  --gold "$LOCAL_DATA_DIR/opendocvqa/MMQA_dev.jsonl"
```
