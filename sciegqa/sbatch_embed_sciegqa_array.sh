#!/usr/bin/env bash
#SBATCH --job-name=sciegqa-embed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/SciEGQA_M3DocRAG/logs/embed_%A_%a.out
#SBATCH --error=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/SciEGQA_M3DocRAG/logs/embed_%A_%a.err

set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG}"
cd "$REPO_ROOT"

source sciegqa/env_hpc.sh

if command -v conda >/dev/null 2>&1; then
  conda activate "$REPO_ROOT/env"
fi

NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-4}}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-2}"

python mmdocir/run_page_embedding_mmdocir.py \
  --data-root "$LOCAL_DATA_DIR/sci-egqa-bench" \
  --output-dir "$LOCAL_EMBEDDINGS_DIR/colpali-v1.2_sci-egqa-bench_dev" \
  --retrieval-model-name-or-path colpaligemma-3b-pt-448-base \
  --retrieval-adapter-model-name-or-path colpali-v1.2 \
  --per-device-eval-batch-size "$BATCH_SIZE" \
  --num-shards "$NUM_SHARDS" \
  --shard-index "$SHARD_INDEX" \
  --resume

