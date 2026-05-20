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
| OpenDocVQA | `opendocvqa/env_hpc.sh` | `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/OpenDocVQA_M3DocRAG` | `$LOCAL_DATA_DIR/opendocvqa` | `colpali-v1.2_opendocvqa_dev` | `$LOCAL_OUTPUT_DIR/opendocvqa` | gated; 3223 packed docs, 206267 pages, 41017 QAs |

Notes:

- OpenDocVQA groups individual corpus images into artificial 64-page packs. Page recall is meaningful; doc recall is only a packing artifact.
- ViDoRe V3 uses input HF split `test` but writes local M3DocRAG split files named `*_dev.*`.
- ViDoRe and OpenDocVQA env files force Hugging Face caches under their scratch work roots to avoid home quota failures.
- All embedding sbatch files use `--resume`, so resubmitting after timeout is safe.
- On compute nodes where plain `python` points to base and misses `torch`/`faiss`, use `"$REPO_ROOT/env/bin/python"` for direct commands. The dataset `run_plain_top224_*.sh` wrappers now default to that interpreter through `PYTHON_BIN`.

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

Observed full `plain_top224`:

```text
n_qids 1658
page_recall@1 0.4136
page_recall@4 0.6075
page_recall@10 0.6913
page_recall@20 0.7480
page_recall@100 0.8305
page_recall@1000 0.9192
doc_recall@1 0.6852
doc_recall@4 0.8058
doc_recall@20 0.8890
doc_recall@100 0.9692
doc_recall@1000 0.9867
page_hit@4 1068/1658
doc_hit@4 1336/1658
improved_doc_rank_count 326
reranked_top4_doc_count 1336
```

MMDocIR full baseline metrics are not recorded here yet. Re-run the baseline eval command above if the comparison table is needed.

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
"$REPO_ROOT/env/bin/python" mmdocir/run_indexing_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/vidoseek" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_vidoseek_dev" \
  --output-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_vidoseek_dev_pageindex_ivfflat" \
  --faiss-index-type ivfflat
```

Baseline retrieval:

```bash
mkdir -p "$LOCAL_OUTPUT_DIR/vidoseek"

"$REPO_ROOT/env/bin/python" mmdocir/run_retrieval_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/vidoseek" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_vidoseek_dev" \
  --index-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_vidoseek_dev_pageindex_ivfflat" \
  --output-json "$LOCAL_OUTPUT_DIR/vidoseek/baseline_ret1000.json" \
  --n-retrieval-pages 1000 \
  --faiss-nprobe 4
```

Evaluate:

```bash
"$REPO_ROOT/env/bin/python" mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/vidoseek/baseline_ret1000.json" \
  --gold "$LOCAL_DATA_DIR/vidoseek/MMQA_dev.jsonl"
```

Observed full baseline:

```text
n_qids 1142
page_recall@1 0.6979
page_recall@4 0.8958
page_recall@10 0.9545
page_recall@20 0.9746
doc_recall@1 0.9921
doc_recall@4 0.9982
doc_recall@10 1.0
page_hit@4 1023/1142
doc_hit@4 1140/1142
```

`plain_top224`:

```bash
bash vidoseek/run_plain_top224_vidoseek.sh

"$REPO_ROOT/env/bin/python" mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/vidoseek/plain_top224_ret1000_prediction.json" \
  --gold "$LOCAL_DATA_DIR/vidoseek/MMQA_dev.jsonl"
```

Observed full `plain_top224`:

```text
n_qids 1142
page_recall@1 0.6830
page_recall@4 0.8958
page_recall@10 0.9623
page_recall@20 0.9842
doc_recall@1 0.9939
doc_recall@4 0.9982
doc_recall@20 1.0
page_hit@4 1023/1142
doc_hit@4 1140/1142
improved_doc_rank_count 6
reranked_top4_doc_count 1140
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

"$REPO_ROOT/env/bin/python" mmdocir/run_retrieval_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/vidore-v3" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_vidore-v3_dev" \
  --index-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_vidore-v3_dev_pageindex_ivfflat" \
  --output-json "$LOCAL_OUTPUT_DIR/vidore-v3/baseline_ret1000.json" \
  --n-retrieval-pages 1000 \
  --faiss-nprobe 4
```

ViDoRe has 14514 queries, so the batch array is safer:

```bash
sbatch --time=24:00:00 --array=0-15%4 \
  --export=ALL,NUM_SHARDS=16,TOP_PAGES=1000,FAISS_NPROBE=4,SAVE_EVERY=25 \
  vidore/sbatch_retrieval_vidore_v3_array.sh
```

After all shards finish, merge them:

```bash
"$REPO_ROOT/env/bin/python" mmdocir/merge_retrieval_predictions.py \
  --input-glob "$LOCAL_OUTPUT_DIR/vidore-v3/baseline_ret1000_shards/shard_*_of_16.json" \
  --output-json "$LOCAL_OUTPUT_DIR/vidore-v3/baseline_ret1000.json" \
  --gold "$LOCAL_DATA_DIR/vidore-v3/MMQA_dev.jsonl"
```

`plain_top224`:

```bash
bash vidore/run_plain_top224_vidore_v3.sh

"$REPO_ROOT/env/bin/python" mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/vidore-v3/plain_top224_ret1000_prediction.json" \
  --gold "$LOCAL_DATA_DIR/vidore-v3/MMQA_dev.jsonl"
```

For the full ViDoRe V3 set, run `plain_top224` as an array after the merged baseline prediction exists:

```bash
sbatch --time=24:00:00 --array=0-15%4 \
  --export=ALL,NUM_SHARDS=16,TOP_PAGES=1000,BASE_ONLY_PAGE_BATCH_SIZE=64 \
  vidore/sbatch_plain_top224_vidore_v3_array.sh
```

Merge sharded `plain_top224` predictions:

```bash
"$REPO_ROOT/env/bin/python" mmdocir/merge_retrieval_predictions.py \
  --input-glob "$LOCAL_OUTPUT_DIR/vidore-v3/plain_top224_ret1000_shards/shard_*_of_16_prediction.json" \
  --output-json "$LOCAL_OUTPUT_DIR/vidore-v3/plain_top224_ret1000_prediction.json" \
  --gold "$LOCAL_DATA_DIR/vidore-v3/MMQA_dev.jsonl"
```

Then evaluate:

```bash
"$REPO_ROOT/env/bin/python" mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/vidore-v3/plain_top224_ret1000_prediction.json" \
  --gold "$LOCAL_DATA_DIR/vidore-v3/MMQA_dev.jsonl"
```

Observed full baseline:

```text
n_qids 14514
page_recall@1 0.1555
page_recall@4 0.3001
page_recall@20 0.4925
page_recall@100 0.6689
page_recall@1000 0.8922
doc_recall@1 0.6344
doc_recall@4 0.8700
doc_recall@20 0.9729
doc_recall@100 0.9985
doc_recall@1000 0.9989
page_hit@4 8417/14514
doc_hit@4 12833/14514
```

Observed full `plain_top224`:

```text
n_qids 14514
page_recall@1 0.1730
page_recall@4 0.3312
page_recall@20 0.5431
page_recall@100 0.7246
page_recall@1000 0.8922
doc_recall@1 0.6586
doc_recall@4 0.8854
doc_recall@20 0.9809
doc_recall@100 0.9986
doc_recall@1000 0.9989
page_hit@4 9092/14514
doc_hit@4 13039/14514
```

ViDoRe V3 embeddings, FAISS index, full baseline, and full `plain_top224` are complete.

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
    "all",
    split="test",
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

Observed smoke sanity:

```text
docs 1
pages 15
qas 50
missing_gold_pages 0
```

Observed smoke retrieval results:

```text
baseline page_recall@1 0.92
baseline page_recall@4 0.98
baseline page_recall@20 1.0
plain_top224 page_recall@1 0.94
plain_top224 page_recall@4 0.98
plain_top224 page_recall@10 1.0
doc_recall@1 1.0 for both, because all smoke pages are in one artificial pack
```

Full prep uses a node-local Hugging Face cache to avoid shared-filesystem lock errors such as `OSError: [Errno 37] No locks available`:

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

Observed full prep sanity:

```text
docs 3223
pages 206267
qas 41017
missing_final_gold_page_uids 0
dataset_counts {'chartqa': 20882, 'coyo': 65294, 'docvqa': 12767, 'dude': 27955, 'infovqa': 5485, 'mpmqa': 10018, 'openwikitable': 1257, 'slidevqa': 52380, 'visualmrc': 10229}
```

Embedding:

```bash
sbatch --time=24:00:00 --array=0-63%8 \
  --export=ALL,NUM_SHARDS=64,BATCH_SIZE=2,DATA_ROOT="$LOCAL_DATA_DIR/opendocvqa",EMBEDDING_NAME=colpali-v1.2_opendocvqa_dev \
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
# counts should match; expected full count: 3223
```

Observed full embedding/index status:

```text
embedding_count 3223/3223
index_path /mmfs1/scratch/jacks.local/aerfanshekooh/custom/OpenDocVQA_M3DocRAG/embeddings/colpali-v1.2_opendocvqa_dev_pageindex_ivfflat/index.bin
index_size 103G
index_meta_size 95K
```

Index:

```bash
"$REPO_ROOT/env/bin/python" mmdocir/run_indexing_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/opendocvqa" \
  --embedding-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_opendocvqa_dev" \
  --output-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_opendocvqa_dev_pageindex_ivfflat" \
  --faiss-index-type ivfflat
```

Baseline retrieval:

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

Full OpenDocVQA retrieval is very slow as one foreground process. Run it as a sharded GPU array instead:

```bash
sbatch --time=24:00:00 --array=0-63%4 \
  --export=ALL,NUM_SHARDS=64,TOP_PAGES=1000,FAISS_NPROBE=4,SAVE_EVERY=25 \
  opendocvqa/sbatch_retrieval_opendocvqa_array.sh
```

Each shard writes:

```text
$LOCAL_OUTPUT_DIR/opendocvqa/baseline_ret1000_shards/shard_<idx>_of_64.json
```

After all shards finish, merge them:

```bash
"$REPO_ROOT/env/bin/python" mmdocir/merge_retrieval_predictions.py \
  --input-glob "$LOCAL_OUTPUT_DIR/opendocvqa/baseline_ret1000_shards/shard_*_of_64.json" \
  --output-json "$LOCAL_OUTPUT_DIR/opendocvqa/baseline_ret1000.json" \
  --gold "$LOCAL_DATA_DIR/opendocvqa/MMQA_dev.jsonl"
```

Evaluate:

```bash
"$REPO_ROOT/env/bin/python" mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/opendocvqa/baseline_ret1000.json" \
  --gold "$LOCAL_DATA_DIR/opendocvqa/MMQA_dev.jsonl"
```

Observed full-index first-100 baseline sanity:

```text
n_qids 100
page_recall@1 0.46
page_recall@4 0.635
page_recall@20 0.735
page_recall@100 0.845
page_recall@1000 0.96
doc_recall@1 0.47
doc_recall@4 0.66
doc_recall@100 0.9
doc_recall@1000 0.995
page_hit@4 68/100
doc_hit@4 71/100
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

`plain_top224`:

```bash
bash opendocvqa/run_plain_top224_opendocvqa.sh

"$REPO_ROOT/env/bin/python" mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$LOCAL_OUTPUT_DIR/opendocvqa/plain_top224_ret1000_prediction.json" \
  --gold "$LOCAL_DATA_DIR/opendocvqa/MMQA_dev.jsonl"
```

For the full OpenDocVQA set, run `plain_top224` as a sharded GPU array after the merged baseline prediction exists:

```bash
sbatch --time=24:00:00 --array=0-63%4 \
  --export=ALL,NUM_SHARDS=64,TOP_PAGES=1000,BASE_ONLY_PAGE_BATCH_SIZE=64 \
  opendocvqa/sbatch_plain_top224_opendocvqa_array.sh
```

Merge sharded `plain_top224` predictions:

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

OpenDocVQA full `plain_top224` metrics are pending the sharded rerank merge/evaluation.

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
