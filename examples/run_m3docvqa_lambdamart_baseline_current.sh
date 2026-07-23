#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/env/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

VITAL_PATHS_ENV="${VITAL_PATHS_ENV:-$REPO_ROOT/hpc_vital_paths.generated.env}"
if [[ -f "$VITAL_PATHS_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$VITAL_PATHS_ENV"
fi

CUSTOM_ROOT="${CUSTOM_ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/output/m3docvqa_standard_reranker_baselines}"
LABEL="${LABEL:-mmqa_train_to_dev_lambdamart_base_exact_maxsim_gpp_direct_exactonly_adaptive_norm05}"

first_existing_path() {
  local path
  for path in "$@"; do
    if [[ -n "$path" && -f "$path" ]]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
  return 1
}

require_file() {
  local name="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    echo "missing_${name}: $path" >&2
    exit 1
  fi
}

TRAIN_GOLD="${TRAIN_GOLD:-$REPO_ROOT/output/m3docvqa_mmqa_direct_evidence_pseudo_page_labels/mmqa_train_pseudo_page_labels_direct_exactonly_adaptive_norm05.augmented_gold.jsonl}"
EVAL_GOLD="${EVAL_GOLD:-$REPO_ROOT/output/m3docvqa_mmqa_direct_evidence_pseudo_page_labels/mmqa_dev_pseudo_page_labels_direct_exactonly_adaptive_norm05.augmented_gold.jsonl}"
TRAIN_PAGE_TEXT_JSONL="${TRAIN_PAGE_TEXT_JSONL:-$CUSTOM_ROOT/outputs/m3docvqa_page_text/m3docvqa_train_page_text.jsonl}"
EVAL_PAGE_TEXT_JSONL="${EVAL_PAGE_TEXT_JSONL:-$CUSTOM_ROOT/outputs/m3docvqa_page_text/m3docvqa_dev_page_text.jsonl}"

BASE_TRAIN_PRED="${BASE_TRAIN_PRED:-$(first_existing_path \
  "$REPO_ROOT/output/m3docvqa_gpp_hyperlink_node_ablation_exact_maxsim_train/mmqa_train_exact_maxsim_gpp_hyperlink_node_no_hyperlink.prediction.json" \
  "$REPO_ROOT/output/m3docvqa_gpp_hyperlink_node_ablation_exact_maxsim_train_aligned_20260709/mmqa_train_exact_maxsim_gpp_hyperlink_node_aligned_20260709_no_hyperlink.prediction.json" \
  "$REPO_ROOT/output/m3docvqa_gpp_hyperlink_node_ablation_exact_maxsim_train_aligned_20260709/mmqa_train_exact_maxsim_gpp_hyperlink_node_aligned_20260709.prediction.json" \
  || true)}"
BASE_EVAL_PRED="${BASE_EVAL_PRED:-$REPO_ROOT/output/m3docvqa_gpp_hyperlink_node_ablation_exact_maxsim/mmqa_dev_exact_maxsim_gpp_hyperlink_node_no_hyperlink.prediction.json}"

TRAIN_SPLADE_PRED="${TRAIN_SPLADE_PRED:-$(first_existing_path \
  "$CUSTOM_ROOT/outputs/m3docvqa_splade_mmqa_train/mmqa_train_splade.prediction.json" \
  "$REPO_ROOT/output/m3docvqa_splade_mmqa_train/mmqa_train_splade.prediction.json" \
  "$CUSTOM_ROOT/outputs/m3docvqa_splade_mmqa_train_aligned_20260709/mmqa_train_splade_aligned_20260709.prediction.json" \
  || true)}"
EVAL_SPLADE_PRED="${EVAL_SPLADE_PRED:-$(first_existing_path \
  "$CUSTOM_ROOT/outputs/m3docvqa_splade_mmqa_dev/mmqa_dev_splade.prediction.json" \
  "$REPO_ROOT/output/m3docvqa_splade_mmqa_dev/mmqa_dev_splade.prediction.json" \
  || true)}"

mkdir -p "$OUT_DIR"
require_file train_gold "$TRAIN_GOLD"
require_file eval_gold "$EVAL_GOLD"
require_file train_page_text_jsonl "$TRAIN_PAGE_TEXT_JSONL"
require_file eval_page_text_jsonl "$EVAL_PAGE_TEXT_JSONL"
require_file base_train_pred "$BASE_TRAIN_PRED"
require_file base_eval_pred "$BASE_EVAL_PRED"

train_sources=()
eval_sources=()
if [[ -f "$TRAIN_SPLADE_PRED" && -f "$EVAL_SPLADE_PRED" ]]; then
  train_sources+=(--train-source "splade=$TRAIN_SPLADE_PRED")
  eval_sources+=(--eval-source "splade=$EVAL_SPLADE_PRED")
else
  echo "skip_splade_source=train:$TRAIN_SPLADE_PRED eval:$EVAL_SPLADE_PRED" >&2
fi

echo "using_train_gold=$TRAIN_GOLD"
echo "using_eval_gold=$EVAL_GOLD"
echo "using_train_page_text_jsonl=$TRAIN_PAGE_TEXT_JSONL"
echo "using_eval_page_text_jsonl=$EVAL_PAGE_TEXT_JSONL"
echo "using_base_train_pred=$BASE_TRAIN_PRED"
echo "using_base_eval_pred=$BASE_EVAL_PRED"
echo "using_train_splade_pred=$TRAIN_SPLADE_PRED"
echo "using_eval_splade_pred=$EVAL_SPLADE_PRED"
echo "using_out_dir=$OUT_DIR"
echo "using_label=$LABEL"
echo "using_ltr_backend=${LTR_BACKEND:-lightgbm}"

"$PYTHON_BIN" "$REPO_ROOT/scripts/train_m3docvqa_ltr_page_reranker.py" \
  --train-gold "$TRAIN_GOLD" \
  --eval-gold "$EVAL_GOLD" \
  --train-base-pred "$BASE_TRAIN_PRED" \
  --eval-base-pred "$BASE_EVAL_PRED" \
  --train-page-text-jsonl "$TRAIN_PAGE_TEXT_JSONL" \
  --eval-page-text-jsonl "$EVAL_PAGE_TEXT_JSONL" \
  "${train_sources[@]}" \
  "${eval_sources[@]}" \
  --backend "${LTR_BACKEND:-lightgbm}" \
  --candidate-top-k "${CANDIDATE_TOP_K:-1000}" \
  --negatives-per-band "${NEGATIVES_PER_BAND:-24}" \
  --max-negatives-per-qid "${MAX_NEGATIVES_PER_QID:-160}" \
  --max-same-doc-pages-per-qid "${MAX_SAME_DOC_PAGES_PER_QID:-24}" \
  --n-estimators "${N_ESTIMATORS:-300}" \
  --learning-rate "${LEARNING_RATE:-0.05}" \
  --num-leaves "${NUM_LEAVES:-31}" \
  --min-data-in-leaf "${MIN_DATA_IN_LEAF:-30}" \
  --subsample "${SUBSAMPLE:-0.90}" \
  --colsample-bytree "${COLSAMPLE_BYTREE:-0.90}" \
  --seed "${SEED:-13}" \
  --n-jobs "${N_JOBS:-8}" \
  --inference-mode "${INFERENCE_MODE:-blend_rerank}" \
  --blend-alpha "${BLEND_ALPHA:-0.30}" \
  --auto-tune-blend-alpha \
  --tune-fraction "${TUNE_FRACTION:-0.20}" \
  --tune-blend-alpha-grid "${TUNE_BLEND_ALPHA_GRID:-0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50}" \
  --tune-hit-k "${TUNE_HIT_K:-4}" \
  --recall-k 1 2 4 5 10 20 50 100 1000 \
  --output-model-json "$OUT_DIR/${LABEL}.model.json" \
  --output-prediction-json "$OUT_DIR/${LABEL}.dev.prediction.json" \
  --output-summary-json "$OUT_DIR/${LABEL}.summary.json" \
  --output-table-md "$OUT_DIR/${LABEL}.table.md" \
  --output-eval-prior-jsonl "$OUT_DIR/${LABEL}.dev.learned_prior.jsonl"
