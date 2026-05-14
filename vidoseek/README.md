# ViDoSeek for M3DocRAG and `plain_top224`

This folder prepares ViDoSeek in a separate HPC workspace and converts its 1-based `reference_page` labels into M3DocRAG-style zero-based page labels.

Default HPC work root:

```bash
/mmfs1/scratch/jacks.local/aerfanshekooh/custom/ViDoSeek_M3DocRAG
```

## 1. Environment

Clear any previous dataset env first, then source ViDoSeek:

```bash
cd /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
source vidoseek/env_hpc.sh
```

## 2. Download and convert ViDoSeek

```bash
python vidoseek/prepare_vidoseek.py \
  --download \
  --snapshot-dir "$VIDOSEEK_WORK_ROOT/hf_snapshot/ViDoSeek" \
  --output-root "$LOCAL_DATA_DIR/vidoseek"
```

Expected outputs:

- `$LOCAL_DATA_DIR/vidoseek/MMQA_dev.jsonl`
- `$LOCAL_DATA_DIR/vidoseek/dev_doc_ids.json`
- `$LOCAL_DATA_DIR/vidoseek/qids_dev.jsonl`
- `$LOCAL_DATA_DIR/vidoseek/gold_pages_dev.jsonl`
- `$LOCAL_DATA_DIR/vidoseek/doc_pages_dev.jsonl`
- `$LOCAL_DATA_DIR/vidoseek/pdfs_raw/`
- `$LOCAL_DATA_DIR/vidoseek/pages_dev/`

## 3. Sanity check conversion

```bash
python - <<'PY'
import json, os
from pathlib import Path
from PIL import Image

root = Path(os.environ["LOCAL_DATA_DIR"]) / "vidoseek"
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

Use a GPU array job:

```bash
sbatch vidoseek/sbatch_embed_vidoseek_array.sh
```

Or choose the shard count explicitly:

```bash
sbatch --array=0-7 --export=ALL,NUM_SHARDS=8,BATCH_SIZE=2 \
  vidoseek/sbatch_embed_vidoseek_array.sh
```

Check completion:

```bash
find "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_vidoseek_dev" -name "*.safetensors" | wc -l
```

Expected count should match `jq length "$LOCAL_DATA_DIR/vidoseek/dev_doc_ids.json"`.

## 5. Build FAISS index

```bash
python mmdocir/run_indexing_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/vidoseek" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_vidoseek_dev" \
  --output-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_vidoseek_dev_pageindex_ivfflat" \
  --faiss-index-type ivfflat
```

## 6. Run baseline retrieval

```bash
mkdir -p "$LOCAL_OUTPUT_DIR/vidoseek"

python mmdocir/run_retrieval_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/vidoseek" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_vidoseek_dev" \
  --index-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_vidoseek_dev_pageindex_ivfflat" \
  --output-json "$LOCAL_OUTPUT_DIR/vidoseek/baseline_ret1000.json" \
  --n-retrieval-pages 1000 \
  --faiss-nprobe 4
```

Evaluate exact reference-page retrieval:

```bash
python mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/vidoseek/baseline_ret1000.json" \
  --gold "$LOCAL_DATA_DIR/vidoseek/MMQA_dev.jsonl"
```

## 7. Run `plain_top224`

```bash
bash vidoseek/run_plain_top224_vidoseek.sh
```

Then evaluate:

```bash
python mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/vidoseek/plain_top224_ret1000_prediction.json" \
  --gold "$LOCAL_DATA_DIR/vidoseek/MMQA_dev.jsonl"
```
