# ViDoRe V3 for M3DocRAG and `plain_top224`

This folder prepares public ViDoRe V3 datasets in a separate HPC workspace and converts each domain's `corpus`/`queries`/`qrels` subsets into M3DocRAG-style page retrieval files.

Default HPC work root:

```bash
/mmfs1/scratch/jacks.local/aerfanshekooh/custom/ViDoRe_M3DocRAG
```

By default the converter combines these public V3 repos:

- `vidore/vidore_v3_computer_science`
- `vidore/vidore_v3_energy`
- `vidore/vidore_v3_finance_en`
- `vidore/vidore_v3_finance_fr`
- `vidore/vidore_v3_hr`
- `vidore/vidore_v3_industrial`
- `vidore/vidore_v3_pharmaceuticals`
- `vidore/vidore_v3_physics`

## 1. Environment

Clear any previous dataset env first, then source ViDoRe:

```bash
cd /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
source vidore/env_hpc.sh
```

The ViDoRe env file forces Hugging Face Hub, datasets, transformers, and XDG caches under `$VIDORE_WORK_ROOT` to avoid home-directory quota failures.

## 2. Download and convert ViDoRe V3

Prepare all public V3 domains:

```bash
python vidore/prepare_vidore_v3.py \
  --download \
  --cache-dir "$VIDORE_WORK_ROOT/hf_cache" \
  --output-root "$LOCAL_DATA_DIR/vidore-v3"
```

To restrict to one query language, add for example:

```bash
  --language english
```

For a smaller smoke test on one domain:

```bash
python vidore/prepare_vidore_v3.py \
  --dataset-repo vidore/vidore_v3_computer_science \
  --cache-dir "$VIDORE_WORK_ROOT/hf_cache" \
  --output-root "$LOCAL_DATA_DIR/vidore-v3-computer-science"
```

Expected outputs:

- `$LOCAL_DATA_DIR/vidore-v3/MMQA_dev.jsonl`
- `$LOCAL_DATA_DIR/vidore-v3/dev_doc_ids.json`
- `$LOCAL_DATA_DIR/vidore-v3/qids_dev.jsonl`
- `$LOCAL_DATA_DIR/vidore-v3/gold_pages_dev.jsonl`
- `$LOCAL_DATA_DIR/vidore-v3/doc_pages_dev.jsonl`
- `$LOCAL_DATA_DIR/vidore-v3/pages_test/`

The converter includes qrels with `score >= 1` by default, so both "critically relevant" and "fully relevant" pages are treated as gold pages.

## 3. Sanity check conversion

```bash
python - <<'PY'
import json, os
from pathlib import Path
from PIL import Image

root = Path(os.environ["LOCAL_DATA_DIR"]) / "vidore-v3"
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

ViDoRe V3 is larger than ViDoSeek and MMDocIR by page count. The default sbatch uses 16 shards and 12 hours:

```bash
sbatch vidore/sbatch_embed_vidore_v3_array.sh
```

If you need fewer concurrent GPUs, use 8 shards and resubmit with `--resume` if it times out:

```bash
sbatch --time=12:00:00 --array=0-7 --export=ALL,NUM_SHARDS=8,BATCH_SIZE=2 \
  vidore/sbatch_embed_vidore_v3_array.sh
```

Check completion:

```bash
find "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_vidore-v3_dev" -name "*.safetensors" | wc -l
jq length "$LOCAL_DATA_DIR/vidore-v3/dev_doc_ids.json"
```

The two counts should match.

## 5. Build FAISS index

```bash
python mmdocir/run_indexing_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/vidore-v3" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_vidore-v3_dev" \
  --output-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_vidore-v3_dev_pageindex_ivfflat" \
  --faiss-index-type ivfflat
```

## 6. Run baseline retrieval

```bash
mkdir -p "$LOCAL_OUTPUT_DIR/vidore-v3"

python mmdocir/run_retrieval_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/vidore-v3" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_vidore-v3_dev" \
  --index-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_vidore-v3_dev_pageindex_ivfflat" \
  --output-json "$LOCAL_OUTPUT_DIR/vidore-v3/baseline_ret1000.json" \
  --n-retrieval-pages 1000 \
  --faiss-nprobe 4
```

Evaluate exact qrel page retrieval:

```bash
python mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/vidore-v3/baseline_ret1000.json" \
  --gold "$LOCAL_DATA_DIR/vidore-v3/MMQA_dev.jsonl"
```

## 7. Run `plain_top224`

```bash
bash vidore/run_plain_top224_vidore_v3.sh
```

Then evaluate:

```bash
python mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/vidore-v3/plain_top224_ret1000_prediction.json" \
  --gold "$LOCAL_DATA_DIR/vidore-v3/MMQA_dev.jsonl"
```
