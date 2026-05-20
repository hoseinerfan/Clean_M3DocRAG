# OpenDocVQA for M3DocRAG and `plain_top224`

This folder prepares OpenDocVQA with the separate OpenDocVQA-Corpus in a dedicated HPC workspace.

Important: OpenDocVQA-Corpus is large and gated on Hugging Face. Accept the dataset terms in the browser and make sure the HPC environment has access, either with `huggingface-cli login` or by exporting `HF_TOKEN`.

Default HPC work root:

```bash
/mmfs1/scratch/jacks.local/aerfanshekooh/custom/OpenDocVQA_M3DocRAG
```

The corpus stores individual document images, not multi-page PDFs. To avoid creating one embedding file per image, the converter groups images into artificial packs of 64 pages by default. Page-level gold labels remain exact, but doc-level recall is only an artifact of the packing; use page recall as the main metric.

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
export HF_TOKEN="$(cat "$OPENDOCVQA_WORK_ROOT/hf_home/token")"
export ODVQA_NODE_CACHE="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/$USER/opendocvqa_hf_cache"
mkdir -p "$ODVQA_NODE_CACHE"

export HF_HOME="$ODVQA_NODE_CACHE/hf_home"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export XDG_CACHE_HOME="$ODVQA_NODE_CACHE/xdg_cache"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME"

"$REPO_ROOT/env/bin/python" opendocvqa/prepare_opendocvqa.py \
  --cache-dir "$ODVQA_NODE_CACHE/load_dataset_cache" \
  --output-root "$LOCAL_DATA_DIR/opendocvqa" \
  --corpus-config all \
  --corpus-split test \
  --streaming-corpus
```

The node-local cache avoids shared-filesystem lock failures from Hugging Face `datasets`. This is heavy because OpenDocVQA-Corpus is about 63 GB before rendered-copy output. For a useful first smoke test:

```bash
"$REPO_ROOT/env/bin/python" opendocvqa/prepare_opendocvqa.py \
  --cache-dir "$OPENDOCVQA_WORK_ROOT/hf_cache" \
  --output-root "$LOCAL_DATA_DIR/opendocvqa_smoke_infovqa_v6" \
  --qa-config infovqa \
  --corpus-config infovqa \
  --qa-split test \
  --corpus-split test \
  --dataset-name infovqa \
  --corpus-scope relevant_only \
  --max-queries 50 \
  --streaming-corpus
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

Observed full conversion sanity:

```text
docs 3223
pages 206267
qas 41017
missing_final_gold_page_uids 0
```

Observed InfoVQA smoke sanity:

```text
docs 1
pages 15
qas 50
missing_gold_pages 0
```

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

OpenDocVQA is much larger than the other prepared datasets. The full prepared dataset has 3223 artificial packed docs and 206267 pages. Use more shards and limit concurrency:

```bash
sbatch --time=24:00:00 --array=0-63%8 \
  --export=ALL,NUM_SHARDS=64,BATCH_SIZE=2,DATA_ROOT="$LOCAL_DATA_DIR/opendocvqa",EMBEDDING_NAME=colpali-v1.2_opendocvqa_dev \
  opendocvqa/sbatch_embed_opendocvqa_array.sh
```

If it times out, resubmit the same command. The embedding script uses `--resume`.

Check completion:

```bash
find "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_opendocvqa_dev" -name "*.safetensors" | wc -l
jq length "$LOCAL_DATA_DIR/opendocvqa/dev_doc_ids.json"
```

The two counts should match. Expected full count: `3223`.

## 5. Build FAISS index

```bash
"$REPO_ROOT/env/bin/python" mmdocir/run_indexing_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/opendocvqa" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_opendocvqa_dev" \
  --output-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_opendocvqa_dev_pageindex_ivfflat" \
  --faiss-index-type ivfflat
```

## 6. Run baseline retrieval

Full OpenDocVQA retrieval is too slow as a single foreground process. Use the sharded GPU array:

```bash
sbatch --time=24:00:00 --array=0-63%4 \
  --export=ALL,NUM_SHARDS=64,TOP_PAGES=1000,FAISS_NPROBE=4,SAVE_EVERY=25 \
  opendocvqa/sbatch_retrieval_opendocvqa_array.sh
```

Each shard writes to:

```text
$LOCAL_OUTPUT_DIR/opendocvqa/baseline_ret1000_shards/shard_<idx>_of_64.json
```

After all shards finish, merge:

```bash
"$REPO_ROOT/env/bin/python" mmdocir/merge_retrieval_predictions.py \
  --input-glob "$LOCAL_OUTPUT_DIR/opendocvqa/baseline_ret1000_shards/shard_*_of_64.json" \
  --output-json "$LOCAL_OUTPUT_DIR/opendocvqa/baseline_ret1000.json" \
  --gold "$LOCAL_DATA_DIR/opendocvqa/MMQA_dev.jsonl"
```

Single-process command, useful only for small sanity runs:

```bash
mkdir -p "$LOCAL_OUTPUT_DIR/opendocvqa"

"$REPO_ROOT/env/bin/python" mmdocir/run_retrieval_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/opendocvqa" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_opendocvqa_dev" \
  --index-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_opendocvqa_dev_pageindex_ivfflat" \
  --output-json "$LOCAL_OUTPUT_DIR/opendocvqa/baseline_ret1000.json" \
  --n-retrieval-pages 1000 \
  --faiss-nprobe 4
```

Evaluate exact relevant-image page retrieval:

```bash
"$REPO_ROOT/env/bin/python" mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/opendocvqa/baseline_ret1000.json" \
  --gold "$LOCAL_DATA_DIR/opendocvqa/MMQA_dev.jsonl"
```

Observed full baseline after merging 64 retrieval shards:

```text
n_qids 41017
page_recall@1 0.4251
page_recall@4 0.5803
page_recall@20 0.6968
page_recall@100 0.7949
page_recall@1000 0.9130
doc_recall@1 0.4334
doc_recall@4 0.5944
doc_recall@20 0.7241
doc_recall@100 0.8401
doc_recall@1000 0.9668
page_hit@4 26161/41017
doc_hit@4 26756/41017
```

## 7. Run `plain_top224`

For the full dataset, run `plain_top224` as a sharded GPU array after the merged baseline prediction exists:

```bash
sbatch --time=24:00:00 --array=0-63%4 \
  --export=ALL,NUM_SHARDS=64,TOP_PAGES=1000,BASE_ONLY_PAGE_BATCH_SIZE=64 \
  opendocvqa/sbatch_plain_top224_opendocvqa_array.sh
```

Each shard writes to:

```text
$LOCAL_OUTPUT_DIR/opendocvqa/plain_top224_ret1000_shards/shard_<idx>_of_64_prediction.json
```

After all shards finish, merge:

```bash
"$REPO_ROOT/env/bin/python" mmdocir/merge_retrieval_predictions.py \
  --input-glob "$LOCAL_OUTPUT_DIR/opendocvqa/plain_top224_ret1000_shards/shard_*_of_64_prediction.json" \
  --output-json "$LOCAL_OUTPUT_DIR/opendocvqa/plain_top224_ret1000_prediction.json" \
  --gold "$LOCAL_DATA_DIR/opendocvqa/MMQA_dev.jsonl"
```

Then evaluate:

```bash
"$REPO_ROOT/env/bin/python" mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/opendocvqa/plain_top224_ret1000_prediction.json" \
  --gold "$LOCAL_DATA_DIR/opendocvqa/MMQA_dev.jsonl"
```

The single-process wrapper is useful only for smoke runs or small subsets:

```bash
bash opendocvqa/run_plain_top224_opendocvqa.sh
```

Observed smoke end-to-end results:

```text
baseline page_recall@1 0.92
baseline page_recall@4 0.98
baseline page_recall@20 1.0
plain_top224 page_recall@1 0.94
plain_top224 page_recall@4 0.98
plain_top224 page_recall@10 1.0
```
