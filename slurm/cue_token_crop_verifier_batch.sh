#!/bin/bash
# Submit with:
#   sbatch /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/slurm/cue_token_crop_verifier_batch.sh
#
# Defaults target the reviewed 9-qid ImageListQ subset built during the
# cue-token verifier experiment. Override any path or hyperparameter via env vars.

#SBATCH --job-name=cue-crop-verify
#SBATCH --partition=gpu
#SBATCH --output=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/slurm_cue_token_crop_verifier_%j.out
#SBATCH --error=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/slurm_cue_token_crop_verifier_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG}"
ENV_PREFIX="${ENV_PREFIX:-$REPO_ROOT/env}"
CONDA_SH="${CONDA_SH:-/mmfs1/cm/shared/apps_local/ondemand/anaconda/etc/profile.d/conda.sh}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.1.1}"

export LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-$REPO_ROOT/data}"

OUT_DIR="${OUT_DIR:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/cue_verifier_subset_top20_notop4_all}"
QIDS="${QIDS:-$OUT_DIR/reviewed_qids.jsonl}"
GOLD="${GOLD:-$REPO_ROOT/data/m3-docvqa/multimodalqa/MMQA_dev.jsonl}"
PREDICTION_JSON="${PREDICTION_JSON:-$OUT_DIR/plain_top224_reviewed9.prediction.json}"
MANUAL_OVERRIDES="${MANUAL_OVERRIDES:-$OUT_DIR/manual_overrides.reviewed.jsonl}"
PATCH_LABELS="${PATCH_LABELS:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/layout_patch_assignments_done_so_far_plus_new_3class_full.jsonl}"
RETRIEVAL_MODEL="${RETRIEVAL_MODEL:-$REPO_ROOT/model/colpaligemma-3b-pt-448-base}"
RETRIEVAL_ADAPTER="${RETRIEVAL_ADAPTER:-$REPO_ROOT/model/colpali-v1.2}"

DATA_NAME="${DATA_NAME:-m3-docvqa}"
SPLIT="${SPLIT:-dev}"
EMBEDDING_NAME="${EMBEDDING_NAME:-colpali-v1.2_m3-docvqa_dev}"
QUERY_TOKEN_FILTER="${QUERY_TOKEN_FILTER:-semantic_only}"
TOP_DOCS="${TOP_DOCS:-5}"
MAX_PAGES_PER_DOC="${MAX_PAGES_PER_DOC:-2}"
WINDOW_FRAC="${WINDOW_FRAC:-0.33}"
STRIDE_FRAC="${STRIDE_FRAC:-1.0}"
CROP_REGION_SOURCE="${CROP_REGION_SOURCE:-visual_patch_centers}"
VISUAL_REGION_FALLBACK="${VISUAL_REGION_FALLBACK:-error}"
TOP_CROP_COUNT="${TOP_CROP_COUNT:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_QIDS="${MAX_QIDS:-0}"

OUTPUT_STEM="${OUTPUT_STEM:-cue_verifier_reviewed9_top5docs_page2}"
OUTPUT_JSONL="${OUTPUT_JSONL:-$OUT_DIR/${OUTPUT_STEM}.jsonl}"
OUTPUT_SUMMARY_JSON="${OUTPUT_SUMMARY_JSON:-$OUT_DIR/${OUTPUT_STEM}.summary.json}"

mkdir -p "$OUT_DIR"

if [[ ! -f "$QIDS" ]]; then
  echo "Missing qid jsonl: $QIDS" >&2
  exit 1
fi
if [[ ! -f "$PREDICTION_JSON" ]]; then
  echo "Missing prediction json: $PREDICTION_JSON" >&2
  exit 1
fi
if [[ ! -f "$MANUAL_OVERRIDES" ]]; then
  echo "Missing manual override jsonl: $MANUAL_OVERRIDES" >&2
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
echo "ENV_PREFIX=$ENV_PREFIX"
echo "LOCAL_DATA_DIR=$LOCAL_DATA_DIR"
echo "QIDS=$QIDS"
echo "PREDICTION_JSON=$PREDICTION_JSON"
echo "MANUAL_OVERRIDES=$MANUAL_OVERRIDES"
echo "PATCH_LABELS=$PATCH_LABELS"
echo "TOP_DOCS=$TOP_DOCS"
echo "MAX_PAGES_PER_DOC=$MAX_PAGES_PER_DOC"
echo "OUTPUT_JSONL=$OUTPUT_JSONL"
echo "OUTPUT_SUMMARY_JSON=$OUTPUT_SUMMARY_JSON"

python --version
which python
nvidia-smi

CMD=(
  python scripts/run_cue_token_crop_verifier_batch.py
  --qid-jsonl "$QIDS"
  --gold "$GOLD"
  --prediction-json "$PREDICTION_JSON"
  --manual-override-jsonl "$MANUAL_OVERRIDES"
  --data-name "$DATA_NAME"
  --split "$SPLIT"
  --embedding-name "$EMBEDDING_NAME"
  --query-token-filter "$QUERY_TOKEN_FILTER"
  --retrieval-model-name-or-path "$RETRIEVAL_MODEL"
  --retrieval-adapter-model-name-or-path "$RETRIEVAL_ADAPTER"
  --splice-patch-labels-jsonl "$PATCH_LABELS"
  --top-docs "$TOP_DOCS"
  --max-pages-per-doc "$MAX_PAGES_PER_DOC"
  --window-frac "$WINDOW_FRAC"
  --stride-frac "$STRIDE_FRAC"
  --crop-region-source "$CROP_REGION_SOURCE"
  --visual-region-fallback "$VISUAL_REGION_FALLBACK"
  --top-crop-count "$TOP_CROP_COUNT"
  --batch-size "$BATCH_SIZE"
  --output-jsonl "$OUTPUT_JSONL"
  --output-summary-json "$OUTPUT_SUMMARY_JSON"
)

if [[ "$MAX_QIDS" != "0" ]]; then
  CMD+=(--max-qids "$MAX_QIDS")
fi

"${CMD[@]}"
