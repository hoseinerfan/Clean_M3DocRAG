# External Dataset Workflows

This file is the central HPC handoff for the external datasets prepared for M3DocRAG retrieval and `plain_top224` evaluation.

Repo root on the UNC HPC:

```bash
/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
```

Current working branch:

```bash
codex/mmdocir-hpc-workflow
```

Update the HPC checkout before using newly added workflows:

```bash
cd /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
git pull --rebase origin codex/mmdocir-hpc-workflow
```

## Environment Rule

Each dataset has its own scratch workspace and `env_hpc.sh`. Always clear the shared path variables before switching datasets:

```bash
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
```

For Hugging Face-heavy or gated datasets, also clear cache variables before sourcing the env:

```bash
unset HF_HOME HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE HF_HUB_CACHE TRANSFORMERS_CACHE XDG_CACHE_HOME
```

Then source the target env:

```bash
source <dataset-folder>/env_hpc.sh
```

The env files set:

- `REPO_ROOT`
- dataset-specific work root, such as `MMDocIR_WORK_ROOT` or `VIDORE_WORK_ROOT`
- `LOCAL_DATA_DIR`
- `LOCAL_EMBEDDINGS_DIR`
- `LOCAL_OUTPUT_DIR`
- `LOCAL_MODEL_DIR`
- `PYTHONPATH`

## Dataset Summary

| Dataset | Env script | Work root | Data folder | Embedding name | Output subdir | Current/expected scale |
|---|---|---|---|---|---|---|
| MMDocIR | `mmdocir/env_hpc.sh` | `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/MMDocIR_M3DocRAG` | `$LOCAL_DATA_DIR/mm-docir` | `colpali-v1.2_mm-docir_dev` | `$LOCAL_OUTPUT_DIR/mmdocir` | 313 docs, 20395 pages, 1658 QAs |
| SciEGQA-Bench | `sciegqa/env_hpc.sh` | `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/SciEGQA_M3DocRAG` | `$LOCAL_DATA_DIR/sci-egqa-bench` | `colpali-v1.2_sci-egqa-bench_dev` | `$LOCAL_OUTPUT_DIR/sciegqa` | 80 docs, 1823 pages, 1623 QAs |
| ViDoSeek | `vidoseek/env_hpc.sh` | `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/ViDoSeek_M3DocRAG` | `$LOCAL_DATA_DIR/vidoseek` | `colpali-v1.2_vidoseek_dev` | `$LOCAL_OUTPUT_DIR/vidoseek` | 290 docs, 5349 pages, 1142 QAs |
| ViDoRe V3 | `vidore/env_hpc.sh` | `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/ViDoRe_M3DocRAG` | `$LOCAL_DATA_DIR/vidore-v3` | `colpali-v1.2_vidore-v3_dev` | `$LOCAL_OUTPUT_DIR/vidore-v3` | 189 docs, 19252 pages, 14514 QAs, all languages |
| OpenDocVQA | `opendocvqa/env_hpc.sh` | `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/OpenDocVQA_M3DocRAG` | `$LOCAL_DATA_DIR/opendocvqa` | `colpali-v1.2_opendocvqa_dev` | `$LOCAL_OUTPUT_DIR/opendocvqa` | gated, large; about 43k QAs and about 170k images/pages in full corpus |

Notes:

- OpenDocVQA groups individual corpus images into artificial 64-page packs. Page recall is meaningful; doc recall is only a packing artifact.
- ViDoRe V3 uses input HF split `test` but writes local M3DocRAG split files named `*_dev.*`.
- ViDoRe and OpenDocVQA env files force Hugging Face caches under their scratch work roots to avoid home quota failures.
- All embedding sbatch files use `--resume`, so resubmitting after timeout is safe.

## MMDocIR

Source env:

```bash
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
source mmdocir/env_hpc.sh
```

Prepare:

```bash
python mmdocir/prepare_mmdocir.py \
  --download \
  --snapshot-dir "$MMDocIR_WORK_ROOT/hf_snapshot/MMDocIR_Evaluation_Dataset" \
  --output-root "$LOCAL_DATA_DIR/mm-docir"
```

Sanity values already observed after doc-id normalization:

```text
docs 313
pages 20395
qas 1658
missing_gold_pages 0
```

Embedding:

```bash
sbatch --time=12:00:00 --array=0-7 --export=ALL,NUM_SHARDS=8,BATCH_SIZE=2 \
  mmdocir/sbatch_embed_mmdocir_array.sh
```

Expected embedding count:

```bash
find "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_mm-docir_dev" -name "*.safetensors" | wc -l
# expected: 313
```

Index:

```bash
python mmdocir/run_indexing_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/mm-docir" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_mm-docir_dev" \
  --output-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_mm-docir_dev_pageindex_ivfflat" \
  --faiss-index-type ivfflat
```

Baseline retrieval:

```bash
mkdir -p "$LOCAL_OUTPUT_DIR/mmdocir"

python mmdocir/run_retrieval_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/mm-docir" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_mm-docir_dev" \
  --index-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_mm-docir_dev_pageindex_ivfflat" \
  --output-json "$LOCAL_OUTPUT_DIR/mmdocir/baseline_ret1000.json" \
  --n-retrieval-pages 1000 \
  --faiss-nprobe 4
```

Evaluate:

```bash
python mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/mmdocir/baseline_ret1000.json" \
  --gold "$LOCAL_DATA_DIR/mm-docir/MMQA_dev.jsonl"
```

`plain_top224`:

```bash
bash mmdocir/run_plain_top224_mmdocir.sh

python mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/mmdocir/plain_top224_ret1000_prediction.json" \
  --gold "$LOCAL_DATA_DIR/mm-docir/MMQA_dev.jsonl"
```

## SciEGQA-Bench

Source env:

```bash
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
source sciegqa/env_hpc.sh
```

Prepare:

```bash
python sciegqa/prepare_sciegqa_bench.py \
  --download \
  --snapshot-dir "$SciEGQA_WORK_ROOT/hf_snapshot/SciEGQA-Bench" \
  --output-root "$LOCAL_DATA_DIR/sci-egqa-bench"
```

Sanity values already observed:

```text
docs 80
pages 1823
qas 1623
missing_gold_pages 0
```

Embedding:

```bash
sbatch --time=06:00:00 --array=0-3 --export=ALL,NUM_SHARDS=4,BATCH_SIZE=2 \
  sciegqa/sbatch_embed_sciegqa_array.sh
```

Expected embedding count:

```bash
find "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_sci-egqa-bench_dev" -name "*.safetensors" | wc -l
# expected: 80
```

Index:

```bash
python mmdocir/run_indexing_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/sci-egqa-bench" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_sci-egqa-bench_dev" \
  --output-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_sci-egqa-bench_dev_pageindex_ivfflat" \
  --faiss-index-type ivfflat
```

Baseline retrieval:

```bash
mkdir -p "$LOCAL_OUTPUT_DIR/sciegqa"

python mmdocir/run_retrieval_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/sci-egqa-bench" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_sci-egqa-bench_dev" \
  --index-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_sci-egqa-bench_dev_pageindex_ivfflat" \
  --output-json "$LOCAL_OUTPUT_DIR/sciegqa/baseline_ret1000.json" \
  --n-retrieval-pages 1000 \
  --faiss-nprobe 4
```

Observed full baseline:

```text
page_recall@4 0.6898
page_recall@100 0.9227
doc_recall@4 0.8848
doc_recall@100 1.0
```

`plain_top224`:

```bash
bash sciegqa/run_plain_top224_sciegqa.sh

python mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/sciegqa/plain_top224_ret1000_prediction.json" \
  --gold "$LOCAL_DATA_DIR/sci-egqa-bench/MMQA_dev.jsonl"
```

Observed full `plain_top224`:

```text
page_recall@4 0.7394
page_recall@100 0.9393
doc_recall@4 0.9070
doc_recall@100 1.0
```

## ViDoSeek

Source env:

```bash
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
source vidoseek/env_hpc.sh
```

Prepare:

```bash
python vidoseek/prepare_vidoseek.py \
  --download \
  --snapshot-dir "$VIDOSEEK_WORK_ROOT/hf_snapshot/ViDoSeek" \
  --output-root "$LOCAL_DATA_DIR/vidoseek"
```

Sanity values already observed:

```text
docs 290
pages 5349
qas 1142
missing_gold_pages 0
```

Embedding:

```bash
sbatch --time=12:00:00 --array=0-7 --export=ALL,NUM_SHARDS=8,BATCH_SIZE=2 \
  vidoseek/sbatch_embed_vidoseek_array.sh
```

Expected embedding count:

```bash
find "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_vidoseek_dev" -name "*.safetensors" | wc -l
# expected: 290
```

Index:

```bash
python mmdocir/run_indexing_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/vidoseek" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_vidoseek_dev" \
  --output-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_vidoseek_dev_pageindex_ivfflat" \
  --faiss-index-type ivfflat
```

Baseline retrieval:

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

`plain_top224`:

```bash
bash vidoseek/run_plain_top224_vidoseek.sh

python mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/vidoseek/plain_top224_ret1000_prediction.json" \
  --gold "$LOCAL_DATA_DIR/vidoseek/MMQA_dev.jsonl"
```

## ViDoRe V3

Source env:

```bash
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
unset HF_HOME HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE HF_HUB_CACHE TRANSFORMERS_CACHE XDG_CACHE_HOME
source vidore/env_hpc.sh
```

Prepare all public V3 domains:

```bash
python vidore/prepare_vidore_v3.py \
  --download \
  --cache-dir "$VIDORE_WORK_ROOT/hf_cache" \
  --output-root "$LOCAL_DATA_DIR/vidore-v3"
```

Optional English-only folder:

```bash
python vidore/prepare_vidore_v3.py \
  --download \
  --cache-dir "$VIDORE_WORK_ROOT/hf_cache" \
  --output-root "$LOCAL_DATA_DIR/vidore-v3-english" \
  --language english
```

Sanity values already observed for all languages:

```text
docs 189
pages 19252
qas 14514
missing_gold_pages 0
missing_qrels 0
```

Embedding:

```bash
sbatch --time=12:00:00 --array=0-15 --export=ALL,NUM_SHARDS=16,BATCH_SIZE=2 \
  vidore/sbatch_embed_vidore_v3_array.sh
```

Expected embedding count:

```bash
find "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_vidore-v3_dev" -name "*.safetensors" | wc -l
# expected: 189
```

If you prepared `vidore-v3-english`, override the data root and output names manually or add a separate wrapper before running the standard pipeline.

Index:

```bash
python mmdocir/run_indexing_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/vidore-v3" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_vidore-v3_dev" \
  --output-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_vidore-v3_dev_pageindex_ivfflat" \
  --faiss-index-type ivfflat
```

Baseline retrieval:

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

`plain_top224`:

```bash
bash vidore/run_plain_top224_vidore_v3.sh

python mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/vidore-v3/plain_top224_ret1000_prediction.json" \
  --gold "$LOCAL_DATA_DIR/vidore-v3/MMQA_dev.jsonl"
```

## OpenDocVQA

Source env:

```bash
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
unset HF_HOME HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE HF_HUB_CACHE TRANSFORMERS_CACHE XDG_CACHE_HOME
source opendocvqa/env_hpc.sh
```

Authentication is required because `NTT-hil-insight/OpenDocVQA-Corpus` is gated:

```bash
huggingface-cli login
huggingface-cli whoami
```

The login token should be saved under:

```text
/mmfs1/scratch/jacks.local/aerfanshekooh/custom/OpenDocVQA_M3DocRAG/hf_home
```

Quick gated-corpus access test:

```bash
python - <<'PY'
from datasets import load_dataset

ds = load_dataset(
    "NTT-hil-insight/OpenDocVQA-Corpus",
    split="train",
    streaming=True,
)
row = next(iter(ds))
print(row.keys())
print("doc_id:", row.get("doc_id"))
print("dataset_name:", row.get("dataset_name"))
PY
```

If the row prints, access is working. A Python shutdown abort after printing the row is an environment cleanup issue; the prep script has a streaming-corpus clean-exit workaround.

Smoke prep:

```bash
python opendocvqa/prepare_opendocvqa.py \
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

Observed smoke sanity:

```text
docs 1
pages 15
qas 50
missing_gold_pages 0
```

Full prep:

```bash
python opendocvqa/prepare_opendocvqa.py \
  --cache-dir "$OPENDOCVQA_WORK_ROOT/hf_cache" \
  --output-root "$LOCAL_DATA_DIR/opendocvqa" \
  --corpus-config all \
  --corpus-split test \
  --streaming-corpus
```

Embedding:

```bash
sbatch --time=24:00:00 --array=0-15 --export=ALL,NUM_SHARDS=16,BATCH_SIZE=2 \
  opendocvqa/sbatch_embed_opendocvqa_array.sh
```

Smoke embedding batch:

```bash
sbatch --time=01:00:00 --array=0-0 \
  --export=ALL,NUM_SHARDS=1,BATCH_SIZE=2,DATA_ROOT="$LOCAL_DATA_DIR/opendocvqa_smoke_infovqa_v6",EMBEDDING_NAME=colpali-v1.2_opendocvqa_smoke_infovqa_v6_dev \
  opendocvqa/sbatch_embed_opendocvqa_array.sh
```

Expected embedding count:

```bash
find "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_opendocvqa_dev" -name "*.safetensors" | wc -l
jq length "$LOCAL_DATA_DIR/opendocvqa/dev_doc_ids.json"
# counts should match
```

Index:

```bash
python mmdocir/run_indexing_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/opendocvqa" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_opendocvqa_dev" \
  --output-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_opendocvqa_dev_pageindex_ivfflat" \
  --faiss-index-type ivfflat
```

Baseline retrieval:

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

`plain_top224`:

```bash
bash opendocvqa/run_plain_top224_opendocvqa.sh

python mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/opendocvqa/plain_top224_ret1000_prediction.json" \
  --gold "$LOCAL_DATA_DIR/opendocvqa/MMQA_dev.jsonl"
```

## Common Sanity Check

Use this after any prepare step, replacing `DATASET_DIR` with the local dataset folder name:

```bash
export DATASET_DIR=sci-egqa-bench

python - <<'PY'
import json, os
from pathlib import Path
from PIL import Image

dataset_dir = os.environ["DATASET_DIR"]
root = Path(os.environ["LOCAL_DATA_DIR"]) / dataset_dir
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

## Common Job Checks

Inspect Slurm status:

```bash
export JOB_ID=10879877

squeue -j "$JOB_ID"
sacct -j "$JOB_ID" --format=JobID,JobName,Partition,State,ExitCode,Elapsed,NodeList%30
```

Check embedding logs:

```bash
export WORK_ROOT="$VIDORE_WORK_ROOT"

grep -iE "traceback|error|exception|killed|oom|out of memory|cuda error|time limit" \
  "$WORK_ROOT/logs"/embed_"$JOB_ID"_*.err
```

Examples:

```bash
grep -iE "traceback|error|exception|killed|oom|out of memory|cuda error|time limit" \
  "$SciEGQA_WORK_ROOT/logs"/embed_"$JOB_ID"_*.err

grep -iE "traceback|error|exception|killed|oom|out of memory|cuda error|time limit" \
  "$VIDORE_WORK_ROOT/logs"/embed_"$JOB_ID"_*.err
```

If a job timed out but produced some `.safetensors`, resubmit the same sbatch command. The embedding scripts use `--resume`.

## File-Specific READMEs

Per-dataset implementation details live in:

- `mmdocir/README.md`
- `sciegqa/README.md`
- `vidoseek/README.md`
- `vidore/README.md`
- `opendocvqa/README.md`
