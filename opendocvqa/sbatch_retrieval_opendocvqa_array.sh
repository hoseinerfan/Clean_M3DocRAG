#!/usr/bin/env bash
#SBATCH --job-name=opendocvqa-ret
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=24:00:00
#SBATCH --array=0-63%4
#SBATCH --output=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/OpenDocVQA_M3DocRAG/logs/retrieval_%A_%a.out
#SBATCH --error=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/OpenDocVQA_M3DocRAG/logs/retrieval_%A_%a.err

set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG}"
cd "$REPO_ROOT"

source opendocvqa/env_hpc.sh

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/env/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-64}}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
TOP_PAGES="${TOP_PAGES:-1000}"
FAISS_NPROBE="${FAISS_NPROBE:-4}"
SAVE_EVERY="${SAVE_EVERY:-25}"

DATA_ROOT="${DATA_ROOT:-$LOCAL_DATA_DIR/opendocvqa}"
EMBEDDING_NAME="${EMBEDDING_NAME:-colpali-v1.2_opendocvqa_dev}"
EMBEDDING_DIR="${EMBEDDING_DIR:-$LOCAL_EMBEDDINGS_DIR/$EMBEDDING_NAME}"
INDEX_DIR="${INDEX_DIR:-$LOCAL_EMBEDDINGS_DIR/${EMBEDDING_NAME}_pageindex_ivfflat}"
OUT_DIR="${OUT_DIR:-$LOCAL_OUTPUT_DIR/opendocvqa/baseline_ret${TOP_PAGES}_shards}"
OUTPUT_JSON="$OUT_DIR/shard_${SHARD_INDEX}_of_${NUM_SHARDS}.json"

mkdir -p "$OUT_DIR"

echo "retrieval_shard index=$SHARD_INDEX num_shards=$NUM_SHARDS data_root=$DATA_ROOT output_json=$OUTPUT_JSON"

"$PYTHON_BIN" mmdocir/run_retrieval_mmdocir.py \
  --data-root "$DATA_ROOT" \
  --embedding-dir "$EMBEDDING_DIR" \
  --index-dir "$INDEX_DIR" \
  --output-json "$OUTPUT_JSON" \
  --n-retrieval-pages "$TOP_PAGES" \
  --faiss-nprobe "$FAISS_NPROBE" \
  --num-shards "$NUM_SHARDS" \
  --shard-index "$SHARD_INDEX" \
  --resume \
  --save-every "$SAVE_EVERY"
