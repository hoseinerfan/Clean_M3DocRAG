#!/usr/bin/env bash
#SBATCH --job-name=vidore-top224
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --array=0-15%4
#SBATCH --output=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/ViDoRe_M3DocRAG/logs/plain_top224_%A_%a.out
#SBATCH --error=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/ViDoRe_M3DocRAG/logs/plain_top224_%A_%a.err

set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG}"
cd "$REPO_ROOT"

unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
unset HF_HOME HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE HF_HUB_CACHE TRANSFORMERS_CACHE XDG_CACHE_HOME
source vidore/env_hpc.sh

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/env/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-16}}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
TOP_PAGES="${TOP_PAGES:-1000}"
BASE_ONLY_PAGE_BATCH_SIZE="${BASE_ONLY_PAGE_BATCH_SIZE:-64}"

DATA_ROOT="${DATA_ROOT:-$LOCAL_DATA_DIR/vidore-v3}"
EMBEDDING_NAME="${EMBEDDING_NAME:-colpali-v1.2_vidore-v3_dev}"
BASELINE_PRED="${BASELINE_PRED:-$LOCAL_OUTPUT_DIR/vidore-v3/baseline_ret${TOP_PAGES}.json}"
OUT_DIR="${OUT_DIR:-$LOCAL_OUTPUT_DIR/vidore-v3/plain_top224_ret${TOP_PAGES}_shards}"
SHARD_QIDS_DIR="${SHARD_QIDS_DIR:-$OUT_DIR/qids}"
SHARD_QIDS_JSONL="$SHARD_QIDS_DIR/qids_${SHARD_INDEX}_of_${NUM_SHARDS}.jsonl"

mkdir -p "$OUT_DIR" "$SHARD_QIDS_DIR"

awk -v n="$NUM_SHARDS" -v s="$SHARD_INDEX" '((NR - 1) % n) == s {print}' \
  "$DATA_ROOT/qids_dev.jsonl" > "$SHARD_QIDS_JSONL"

EMPTY_QUERY_LABELS="$OUT_DIR/empty_query_token_labels.json"
EMPTY_PATCH_LABELS="$OUT_DIR/empty_patch_labels.jsonl"
printf '{}\n' > "$EMPTY_QUERY_LABELS"
: > "$EMPTY_PATCH_LABELS"

echo "plain_top224_shard index=$SHARD_INDEX num_shards=$NUM_SHARDS qids=$SHARD_QIDS_JSONL baseline_pred=$BASELINE_PRED"

"$PYTHON_BIN" "$REPO_ROOT/scripts/run_visual_rerank_batch.py" \
  --qid-jsonl "$SHARD_QIDS_JSONL" \
  --gold "$DATA_ROOT/MMQA_dev.jsonl" \
  --baseline-pred "$BASELINE_PRED" \
  --data-name vidore-v3 \
  --split dev \
  --embedding_name "$EMBEDDING_NAME" \
  --from-baseline-top-pages "$TOP_PAGES" \
  --base-score-source approx_page_maxsim_topk \
  --approx-base-page-token-topk 224 \
  --approx-base-page-token-scorer query_mean \
  --approx-base-page-token-selector global_topk \
  --approx-base-page-token-coarse-dtype fp32 \
  --weight-base 1.0 \
  --weight-visual 0.0 \
  --weight-non-visual 0.0 \
  --weight-balance 0.0 \
  --splice-query-token-labels "$EMPTY_QUERY_LABELS" \
  --splice-patch-labels-jsonl "$EMPTY_PATCH_LABELS" \
  --base-only-page-batch-size "$BASE_ONLY_PAGE_BATCH_SIZE" \
  --output-jsonl "$OUT_DIR/shard_${SHARD_INDEX}_of_${NUM_SHARDS}.jsonl" \
  --output-summary-json "$OUT_DIR/shard_${SHARD_INDEX}_of_${NUM_SHARDS}_summary.json" \
  --output-prediction-json "$OUT_DIR/shard_${SHARD_INDEX}_of_${NUM_SHARDS}_prediction.json"
