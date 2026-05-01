#!/bin/bash
# Submit with:
#   sbatch /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/slurm/mmqa_dev_top256_array.sh
#
# Before submitting, split MMQA_dev.jsonl into chunk files such as:
#   /mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/mmqa_dev_chunks_500/mmqa_dev_chunk_000.jsonl
#   /mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/mmqa_dev_chunks_500/mmqa_dev_chunk_001.jsonl
#
# The default array range 0-4 matches 2441 dev qids split into chunks of 500.

#SBATCH --job-name=mmqa-top256
#SBATCH --output=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/slurm_mmqa_top256_%A_%a.out
#SBATCH --error=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/slurm_mmqa_top256_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-4

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG}"
ENV_PREFIX="${ENV_PREFIX:-$REPO_ROOT/env}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.1.1}"

CHUNK_DIR="${CHUNK_DIR:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/mmqa_dev_chunks_500}"
CHUNK_PREFIX="${CHUNK_PREFIX:-mmqa_dev_chunk_}"
OUTPUT_DIR="${OUTPUT_DIR:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs}"
RUN_TAG="${RUN_TAG:-mmqa_dev_top256}"

CHUNK_ID=$(printf "%03d" "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}")
QID_JSONL="${QID_JSONL:-$CHUNK_DIR/${CHUNK_PREFIX}${CHUNK_ID}.jsonl}"

GOLD_JSONL="${GOLD_JSONL:-$REPO_ROOT/data/m3-docvqa/multimodalqa/MMQA_dev.jsonl}"
BASELINE_PRED="${BASELINE_PRED:-$REPO_ROOT/output/retrieval_only_dev_ret1000full/colpali-v1.2_ivfflat_ret1000_2026-04-10_18-10-48.json}"
QUERY_LABELS="${QUERY_LABELS:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/visual_needed_binary/deberta_v3_large_seed42/export/dev_query_visual_binary_labels_union_relaxed_v6_fulltrainlex_v2.jsonl}"
PATCH_LABELS="${PATCH_LABELS:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/layout_patch_assignments_done_so_far_plus_new_3class_full.jsonl}"

RETRIEVAL_MODEL="${RETRIEVAL_MODEL:-$REPO_ROOT/model/colpaligemma-3b-pt-448-base}"
RETRIEVAL_ADAPTER="${RETRIEVAL_ADAPTER:-$REPO_ROOT/model/colpali-v1.2}"
EMBEDDING_NAME="${EMBEDDING_NAME:-colpali-v1.2_m3-docvqa_dev}"

OUTPUT_JSONL="$OUTPUT_DIR/${RUN_TAG}_chunk_${CHUNK_ID}.jsonl"
OUTPUT_SUMMARY_JSON="$OUTPUT_DIR/${RUN_TAG}_chunk_${CHUNK_ID}.summary.json"

mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$QID_JSONL" ]]; then
  echo "Missing chunk file: $QID_JSONL" >&2
  exit 1
fi

if [[ -f "$ENV_PREFIX/bin/activate" ]]; then
  # Python venv workflow used in current HPC setup.
  source "$ENV_PREFIX/bin/activate"
else
  echo "Missing virtualenv activate script: $ENV_PREFIX/bin/activate" >&2
  exit 1
fi

if command -v module >/dev/null 2>&1; then
  module load "$CUDA_MODULE"
fi

cd "$REPO_ROOT"

echo "REPO_ROOT=$REPO_ROOT"
echo "QID_JSONL=$QID_JSONL"
echo "OUTPUT_JSONL=$OUTPUT_JSONL"
echo "OUTPUT_SUMMARY_JSON=$OUTPUT_SUMMARY_JSON"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

python --version
which python
nvidia-smi

python scripts/run_visual_rerank_batch.py \
  --qid-jsonl "$QID_JSONL" \
  --gold "$GOLD_JSONL" \
  --embedding_name "$EMBEDDING_NAME" \
  --query_token_filter full \
  --base-score-source approx_page_maxsim_topk \
  --approx-base-page-token-topk 256 \
  --approx-base-page-token-scorer query_mean \
  --approx-base-page-token-selector global_topk \
  --approx-base-page-token-coarse-dtype fp32 \
  --retrieval_model_name_or_path "$RETRIEVAL_MODEL" \
  --retrieval_adapter_model_name_or_path "$RETRIEVAL_ADAPTER" \
  --splice-query-token-labels "$QUERY_LABELS" \
  --splice-patch-labels-jsonl "$PATCH_LABELS" \
  --baseline-pred "$BASELINE_PRED" \
  --from-baseline-top-pages 1000 \
  --weight-base 1.0 \
  --weight-visual 0.0 \
  --weight-non-visual 0.0 \
  --output-jsonl "$OUTPUT_JSONL" \
  --output-summary-json "$OUTPUT_SUMMARY_JSON"
