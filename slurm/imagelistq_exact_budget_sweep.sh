#!/bin/bash
# Submit with:
#   sbatch /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/slurm/imagelistq_exact_budget_sweep.sh
#
# Default array layout covers 12 runs:
#   0  baseline   top5
#   1  nonvisual  top5
#   2  gate005    top5
#   3  baseline   top10
#   4  nonvisual  top10
#   5  gate005    top10
#   6  baseline   top20
#   7  nonvisual  top20
#   8  gate005    top20
#   9  baseline   top50
#   10 nonvisual  top50
#   11 gate005    top50
#
# Completed runs are skipped automatically if their summary JSON already exists.
# Incomplete runs resume if a partial output JSONL already exists.

#SBATCH --job-name=imgl-exact-budget
#SBATCH --partition=gpu
#SBATCH --output=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/slurm_imagelistq_exact_budget_%A_%a.out
#SBATCH --error=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/slurm_imagelistq_exact_budget_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-11

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG}"
ENV_PREFIX="${ENV_PREFIX:-$REPO_ROOT/env}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.1.1}"
CONDA_SH="${CONDA_SH:-/mmfs1/cm/shared/apps_local/ondemand/anaconda/etc/profile.d/conda.sh}"

QIDS="${QIDS:-$REPO_ROOT/output/rag_dev_ret4/imagelistq_all_dev_qids.jsonl}"
GOLD="${GOLD:-$REPO_ROOT/data/m3-docvqa/multimodalqa/MMQA_dev.jsonl}"
BASELINE_PRED="${BASELINE_PRED:-$REPO_ROOT/output/retrieval_only_dev_ret1000full_nprobe4/colpali-v1.2_ivfflat_nprobe4_ret1000_2026-05-10_10-28-25.json}"
QUERY_LABELS="${QUERY_LABELS:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/visual_needed_binary/deberta_v3_large_seed42/export/dev_query_visual_binary_labels_union_relaxed_v6_fulltrainlex_v2.jsonl}"
PATCH_LABELS="${PATCH_LABELS:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/layout_patch_assignments_done_so_far_plus_new_3class_full.jsonl}"
RETRIEVAL_MODEL="${RETRIEVAL_MODEL:-$REPO_ROOT/model/colpaligemma-3b-pt-448-base}"
RETRIEVAL_ADAPTER="${RETRIEVAL_ADAPTER:-$REPO_ROOT/model/colpali-v1.2}"
EMBEDDING_NAME="${EMBEDDING_NAME:-colpali-v1.2_m3-docvqa_dev}"
OUTPUT_DIR="${OUTPUT_DIR:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs}"

mkdir -p "$OUTPUT_DIR"

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

METHODS=(baseline nonvisual gate005)
PS=(5 10 20 50)

METHOD_INDEX=$(( TASK_ID % 3 ))
P_INDEX=$(( TASK_ID / 3 ))

if (( P_INDEX < 0 || P_INDEX >= ${#PS[@]} )); then
  echo "Invalid array mapping for TASK_ID=$TASK_ID" >&2
  exit 1
fi

METHOD="${METHODS[$METHOD_INDEX]}"
P="${PS[$P_INDEX]}"

OUTPUT_JSONL="$OUTPUT_DIR/imagelistq_${METHOD}_exact_top${P}.jsonl"
OUTPUT_SUMMARY_JSON="$OUTPUT_DIR/imagelistq_${METHOD}_exact_top${P}.summary.json"

if [[ -f "$OUTPUT_SUMMARY_JSON" ]]; then
  echo "Skipping completed run: method=$METHOD top${P}"
  echo "Existing summary: $OUTPUT_SUMMARY_JSON"
  exit 0
fi

if [[ ! -f "$QIDS" ]]; then
  echo "Missing qid jsonl: $QIDS" >&2
  exit 1
fi

if [[ ! -f "$BASELINE_PRED" ]]; then
  echo "Missing baseline retrieval file: $BASELINE_PRED" >&2
  exit 1
fi

if [[ -f "$ENV_PREFIX/bin/activate" ]]; then
  source "$ENV_PREFIX/bin/activate"
elif [[ -f "$CONDA_SH" ]]; then
  source "$CONDA_SH"
  conda activate "$ENV_PREFIX"
else
  echo "Could not activate environment." >&2
  echo "Checked virtualenv activate: $ENV_PREFIX/bin/activate" >&2
  echo "Checked conda init script: $CONDA_SH" >&2
  exit 1
fi

if command -v module >/dev/null 2>&1; then
  module load "$CUDA_MODULE"
fi

cd "$REPO_ROOT"

echo "REPO_ROOT=$REPO_ROOT"
echo "TASK_ID=$TASK_ID"
echo "METHOD=$METHOD"
echo "P=$P"
echo "QIDS=$QIDS"
echo "BASELINE_PRED=$BASELINE_PRED"
echo "OUTPUT_JSONL=$OUTPUT_JSONL"
echo "OUTPUT_SUMMARY_JSON=$OUTPUT_SUMMARY_JSON"

python --version
which python
nvidia-smi

COMMON_ARGS=(
  --qid-jsonl "$QIDS"
  --gold "$GOLD"
  --embedding_name "$EMBEDDING_NAME"
  --query_token_filter full
  --retrieval_model_name_or_path "$RETRIEVAL_MODEL"
  --retrieval_adapter_model_name_or_path "$RETRIEVAL_ADAPTER"
  --splice-query-token-labels "$QUERY_LABELS"
  --splice-patch-labels-jsonl "$PATCH_LABELS"
  --baseline-pred "$BASELINE_PRED"
  --from-baseline-top-pages 1000
  --weight-base 1
  --weight-visual 0
  --weight-non-visual 0
  --weight-balance 0
  --resume-output-jsonl
  --output-jsonl "$OUTPUT_JSONL"
  --output-summary-json "$OUTPUT_SUMMARY_JSON"
)

case "$METHOD" in
  baseline)
    python scripts/run_visual_rerank_batch.py \
      "${COMMON_ARGS[@]}" \
      --base-score-source baseline_pred_two_stage_page_maxsim \
      --approx-base-page-token-topk 0 \
      --two-stage-exact-top-pages "$P"
    ;;
  nonvisual)
    python scripts/run_visual_rerank_batch.py \
      "${COMMON_ARGS[@]}" \
      --base-score-source visual_prefilter_exact_page_maxsim \
      --approx-base-page-token-topk 256 \
      --approx-base-page-token-scorer query_mean \
      --approx-base-page-token-selector global_topk \
      --approx-base-page-token-coarse-dtype fp32 \
      --two-stage-exact-top-pages "$P" \
      --visual-rerank-top-pages 1000 \
      --visual-prefilter-sort-key non_visual_only \
      --balance-score-mode visual_x_nonvisual_avg \
      --visual-score-query-mode visual_query_only \
      --non-visual-page-mode labeled_only \
      --grounded-context-radius 2
    ;;
  gate005)
    python scripts/run_visual_rerank_batch.py \
      "${COMMON_ARGS[@]}" \
      --base-score-source visual_prefilter_exact_page_maxsim \
      --approx-base-page-token-topk 256 \
      --approx-base-page-token-scorer query_mean \
      --approx-base-page-token-selector global_topk \
      --approx-base-page-token-coarse-dtype fp32 \
      --two-stage-exact-top-pages "$P" \
      --visual-rerank-top-pages 1000 \
      --visual-prefilter-sort-key non_visual_with_confirmed_visual_gate \
      --confirmed-visual-gate-threshold 0.05 \
      --balance-score-mode visual_x_nonvisual_avg \
      --visual-score-query-mode visual_query_only \
      --non-visual-page-mode labeled_only \
      --grounded-context-radius 2
    ;;
  *)
    echo "Unknown method: $METHOD" >&2
    exit 1
    ;;
esac
