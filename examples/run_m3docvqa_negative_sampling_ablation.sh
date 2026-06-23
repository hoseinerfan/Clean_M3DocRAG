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

OUT_DIR="${OUT_DIR:-$REPO_ROOT/output/m3docvqa_content_aware_exact_maxsim_negative_sampling_ablation}"
SEED_REFERENCE_DIR="${SEED_REFERENCE_DIR:-$REPO_ROOT/output/m3docvqa_content_aware_exact_maxsim_seed_stability}"
TRAIN_GOLD="${TRAIN_GOLD:-$REPO_ROOT/output/m3docvqa_mmqa_pseudo_page_labels/mmqa_train_pseudo_page_labels_strict.augmented_gold.jsonl}"
EVAL_GOLD="${EVAL_GOLD:-$REPO_ROOT/output/m3docvqa_mmqa_pseudo_page_labels/mmqa_dev_pseudo_page_labels_strict.augmented_gold.jsonl}"

GPP_TRAIN_OUT_DIR="${GPP_TRAIN_OUT_DIR:-$REPO_ROOT/output/m3docvqa_gpp_hyperlink_node_ablation_exact_maxsim_train}"
GPP_EVAL_OUT_DIR="${GPP_EVAL_OUT_DIR:-$REPO_ROOT/output/m3docvqa_gpp_hyperlink_node_ablation_exact_maxsim}"
GPP_TRAIN_LABEL_PREFIX="${GPP_TRAIN_LABEL_PREFIX:-mmqa_train_exact_maxsim_gpp_hyperlink_node}"
GPP_EVAL_LABEL_PREFIX="${GPP_EVAL_LABEL_PREFIX:-mmqa_dev_exact_maxsim_gpp_hyperlink_node}"
BASE_LABEL="${BASE_LABEL:-exact_maxsim_gpp_no_hyperlink}"
BASE_TRAIN_PRED="${BASE_TRAIN_PRED:-$GPP_TRAIN_OUT_DIR/${GPP_TRAIN_LABEL_PREFIX}_no_hyperlink.prediction.json}"
BASE_EVAL_PRED="${BASE_EVAL_PRED:-$GPP_EVAL_OUT_DIR/${GPP_EVAL_LABEL_PREFIX}_no_hyperlink.prediction.json}"

STRATEGIES="${STRATEGIES:-rank_stratified uniform hard_top}"
SEEDS="${SEEDS:-13 42 73}"
NEGATIVE_BUDGET="${NEGATIVE_BUDGET:-50}"
POSITIVE_WEIGHT_CAP="${POSITIVE_WEIGHT_CAP:-20}"
TUNE_BLEND_ALPHA_GRID="${TUNE_BLEND_ALPHA_GRID:-0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50}"
TUNE_FRACTION="${TUNE_FRACTION:-0.20}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"
REUSE_SEED_STABILITY="${REUSE_SEED_STABILITY:-1}"

require_file() {
  local name="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    echo "missing_${name}: $path" >&2
    exit 1
  fi
}

label_for_run() {
  printf 'mmqa_train_to_dev_content_aware_neg_%s_seed_%s_poscap20_auto_page4_base_%s' "$1" "$2" "$BASE_LABEL"
}

seed_reference_label() {
  printf 'mmqa_train_to_dev_content_aware_seed_%s_poscap20_auto_page4_base_%s' "$1" "$BASE_LABEL"
}

prediction_for_run() {
  local strategy="$1"
  local seed="$2"
  local reference="$SEED_REFERENCE_DIR/$(seed_reference_label "$seed").dev.prediction.json"
  if [[ "$REUSE_SEED_STABILITY" == "1" && "$strategy" == "rank_stratified" && -f "$reference" ]]; then
    printf '%s' "$reference"
  else
    printf '%s/%s.dev.prediction.json' "$OUT_DIR" "$(label_for_run "$strategy" "$seed")"
  fi
}

model_for_run() {
  local strategy="$1"
  local seed="$2"
  local reference="$SEED_REFERENCE_DIR/$(seed_reference_label "$seed").model.json"
  if [[ "$REUSE_SEED_STABILITY" == "1" && "$strategy" == "rank_stratified" && -f "$reference" ]]; then
    printf '%s' "$reference"
  else
    printf '%s/%s.model.json' "$OUT_DIR" "$(label_for_run "$strategy" "$seed")"
  fi
}

run_one() {
  local strategy="$1"
  local seed="$2"
  local label pred model
  label="$(label_for_run "$strategy" "$seed")"
  pred="$(prediction_for_run "$strategy" "$seed")"
  model="$(model_for_run "$strategy" "$seed")"
  if [[ "$FORCE_RERUN" != "1" && -f "$pred" && -f "$model" ]]; then
    echo "reuse_${strategy}_seed_${seed}=$pred"
    return 0
  fi

  echo
  echo "== CAPP negative sampling: $strategy, seed $seed =="
  TRAIN_GOLD="$TRAIN_GOLD" \
  EVAL_GOLD="$EVAL_GOLD" \
  TRAIN_DENSE_PRED="$BASE_TRAIN_PRED" \
  EVAL_DENSE_PRED="$BASE_EVAL_PRED" \
  GPP_TRAIN_OUT_DIR="$GPP_TRAIN_OUT_DIR" \
  GPP_EVAL_OUT_DIR="$GPP_EVAL_OUT_DIR" \
  GPP_TRAIN_LABEL_PREFIX="$GPP_TRAIN_LABEL_PREFIX" \
  GPP_EVAL_LABEL_PREFIX="$GPP_EVAL_LABEL_PREFIX" \
  FEATURE_SET=all \
  SOURCE_SET=all \
  MODEL_TYPE=logistic \
  NEGATIVE_SAMPLING_STRATEGY="$strategy" \
  NEGATIVES_PER_BAND=10 \
  MAX_NEGATIVES_PER_QID="$NEGATIVE_BUDGET" \
  POSITIVE_WEIGHT_CAP="$POSITIVE_WEIGHT_CAP" \
  AUTO_TUNE_BLEND_ALPHA=1 \
  TUNE_HIT_K=4 \
  TUNE_FRACTION="$TUNE_FRACTION" \
  TUNE_BLEND_ALPHA_GRID="$TUNE_BLEND_ALPHA_GRID" \
  SEED="$seed" \
  OUT_DIR="$OUT_DIR" \
  LABEL="$label" \
  bash "$REPO_ROOT/examples/run_m3docvqa_content_aware_pseudo_page_reranker.sh"
}

mkdir -p "$OUT_DIR"
if [[ "$NEGATIVE_BUDGET" != "50" || "$POSITIVE_WEIGHT_CAP" != "20" ]]; then
  echo "Controlled ablation requires NEGATIVE_BUDGET=50 and POSITIVE_WEIGHT_CAP=20" >&2
  exit 1
fi
require_file train_gold "$TRAIN_GOLD"
require_file eval_gold "$EVAL_GOLD"
require_file base_train_pred "$BASE_TRAIN_PRED"
require_file base_eval_pred "$BASE_EVAL_PRED"

if [[ "$RUN_TRAIN" == "1" ]]; then
  for strategy in $STRATEGIES; do
    for seed in $SEEDS; do
      run_one "$strategy" "$seed"
    done
  done
fi

if [[ "$RUN_EVAL" == "1" ]]; then
  eval_args=(--run "GPP=$BASE_EVAL_PRED")
  model_args=()
  for strategy in $STRATEGIES; do
    for seed in $SEEDS; do
      pred="$(prediction_for_run "$strategy" "$seed")"
      model="$(model_for_run "$strategy" "$seed")"
      require_file "${strategy}_seed_${seed}_prediction" "$pred"
      require_file "${strategy}_seed_${seed}_model" "$model"
      eval_args+=(--run "${strategy}_seed_${seed}=$pred")
      model_args+=(--model "${strategy}:${seed}=$model")
    done
  done

  "$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_pseudo_page_retrieval.py" \
    --gold "$EVAL_GOLD" \
    --recall-k 1 2 4 10 20 50 100 \
    "${eval_args[@]}" \
    --format markdown \
    --output "$OUT_DIR/negative_sampling_runs.md"

  "$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_pseudo_page_retrieval.py" \
    --gold "$EVAL_GOLD" \
    --recall-k 1 2 4 10 20 50 100 \
    "${eval_args[@]}" \
    --format json \
    --output "$OUT_DIR/negative_sampling_runs.json"

  "$PYTHON_BIN" "$REPO_ROOT/scripts/summarize_m3docvqa_negative_sampling_ablation.py" \
    --eval-json "$OUT_DIR/negative_sampling_runs.json" \
    "${model_args[@]}" \
    --output-md "$OUT_DIR/negative_sampling_summary.md" \
    --output-json "$OUT_DIR/negative_sampling_summary.json"
fi
