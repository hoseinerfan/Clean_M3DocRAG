#!/usr/bin/env bash
#SBATCH --job-name=mmlongbench-index-retrieve
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/MMLongBench_M3DocRAG/logs/index_retrieve_%j.out
#SBATCH --error=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/MMLongBench_M3DocRAG/logs/index_retrieve_%j.err

set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG}"
cd "$REPO_ROOT"

source mmlongbench/env_hpc.sh

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/env/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

DATA_ROOT="${DATA_ROOT:-$LOCAL_DATA_DIR/mmlongbench-docqa}"
EMBEDDING_DIR="${EMBEDDING_DIR:-$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_mmlongbench-docqa_dev}"
INDEX_DIR="${INDEX_DIR:-$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_mmlongbench-docqa_dev_pageindex_ivfflat}"
OUT_DIR="${OUT_DIR:-$LOCAL_OUTPUT_DIR/mmlongbench-docqa}"
OUTPUT_JSON="${OUTPUT_JSON:-$OUT_DIR/baseline_ret1000.json}"
FAISS_INDEX_TYPE="${FAISS_INDEX_TYPE:-ivfflat}"
FAISS_NPROBE="${FAISS_NPROBE:-4}"
N_RETRIEVAL_PAGES="${N_RETRIEVAL_PAGES:-1000}"
SAVE_EVERY="${SAVE_EVERY:-100}"
FORCE_REBUILD_INDEX="${FORCE_REBUILD_INDEX:-0}"

mkdir -p "$INDEX_DIR" "$OUT_DIR"

if [[ "$FORCE_REBUILD_INDEX" == "1" || ! -f "$INDEX_DIR/index.bin" ]]; then
  "$PYTHON_BIN" mmdocir/run_indexing_mmdocir.py \
    --data-root "$DATA_ROOT" \
    --embedding-dir "$EMBEDDING_DIR" \
    --output-dir "$INDEX_DIR" \
    --faiss-index-type "$FAISS_INDEX_TYPE"
else
  echo "reusing_existing_index: $INDEX_DIR/index.bin"
fi

"$PYTHON_BIN" mmdocir/run_retrieval_mmdocir.py \
  --data-root "$DATA_ROOT" \
  --embedding-dir "$EMBEDDING_DIR" \
  --index-dir "$INDEX_DIR" \
  --output-json "$OUTPUT_JSON" \
  --n-retrieval-pages "$N_RETRIEVAL_PAGES" \
  --faiss-nprobe "$FAISS_NPROBE" \
  --resume \
  --save-every "$SAVE_EVERY"

echo "saved_dense_prediction=$OUTPUT_JSON"
