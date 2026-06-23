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

OUT_DIR="${OUT_DIR:-$REPO_ROOT/output/m3docvqa_content_aware_exact_maxsim_seed_stability}"
TRAIN_GOLD="${TRAIN_GOLD:-$REPO_ROOT/output/m3docvqa_mmqa_pseudo_page_labels/mmqa_train_pseudo_page_labels_strict.augmented_gold.jsonl}"
EVAL_GOLD="${EVAL_GOLD:-$REPO_ROOT/output/m3docvqa_mmqa_pseudo_page_labels/mmqa_dev_pseudo_page_labels_strict.augmented_gold.jsonl}"

GPP_TRAIN_OUT_DIR="${GPP_TRAIN_OUT_DIR:-$REPO_ROOT/output/m3docvqa_gpp_hyperlink_node_ablation_exact_maxsim_train}"
GPP_EVAL_OUT_DIR="${GPP_EVAL_OUT_DIR:-$REPO_ROOT/output/m3docvqa_gpp_hyperlink_node_ablation_exact_maxsim}"
GPP_TRAIN_LABEL_PREFIX="${GPP_TRAIN_LABEL_PREFIX:-mmqa_train_exact_maxsim_gpp_hyperlink_node}"
GPP_EVAL_LABEL_PREFIX="${GPP_EVAL_LABEL_PREFIX:-mmqa_dev_exact_maxsim_gpp_hyperlink_node}"
BASE_LABEL="${BASE_LABEL:-exact_maxsim_gpp_no_hyperlink}"
BASE_TRAIN_PRED="${BASE_TRAIN_PRED:-$GPP_TRAIN_OUT_DIR/${GPP_TRAIN_LABEL_PREFIX}_no_hyperlink.prediction.json}"
BASE_EVAL_PRED="${BASE_EVAL_PRED:-$GPP_EVAL_OUT_DIR/${GPP_EVAL_LABEL_PREFIX}_no_hyperlink.prediction.json}"

SEEDS="${SEEDS:-13 42 73}"
POSITIVE_WEIGHT_CAP="${POSITIVE_WEIGHT_CAP:-20}"
TUNE_BLEND_ALPHA_GRID="${TUNE_BLEND_ALPHA_GRID:-0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50}"
TUNE_FRACTION="${TUNE_FRACTION:-0.20}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"
REUSE_REFERENCE_SEED13="${REUSE_REFERENCE_SEED13:-0}"

REFERENCE_SEED13_DIR="${REFERENCE_SEED13_DIR:-$REPO_ROOT/output/m3docvqa_content_aware_exact_maxsim_adaptive_topk_ablation}"
REFERENCE_SEED13_LABEL="${REFERENCE_SEED13_LABEL:-mmqa_train_to_dev_content_aware_auto_page4_base_exact_maxsim_gpp_no_hyperlink}"
REFERENCE_SEED13_PRED="${REFERENCE_SEED13_PRED:-$REFERENCE_SEED13_DIR/${REFERENCE_SEED13_LABEL}.dev.prediction.json}"
REFERENCE_SEED13_MODEL="${REFERENCE_SEED13_MODEL:-$REFERENCE_SEED13_DIR/${REFERENCE_SEED13_LABEL}.model.json}"

require_file() {
  local name="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    echo "missing_${name}: $path" >&2
    exit 1
  fi
}

label_for_seed() {
  printf 'mmqa_train_to_dev_content_aware_seed_%s_poscap20_auto_page4_base_%s' "$1" "$BASE_LABEL"
}

prediction_for_seed() {
  local seed="$1"
  if [[ "$REUSE_REFERENCE_SEED13" == "1" && "$seed" == "13" && -f "$REFERENCE_SEED13_PRED" ]]; then
    printf '%s' "$REFERENCE_SEED13_PRED"
  else
    printf '%s/%s.dev.prediction.json' "$OUT_DIR" "$(label_for_seed "$seed")"
  fi
}

model_for_seed() {
  local seed="$1"
  if [[ "$REUSE_REFERENCE_SEED13" == "1" && "$seed" == "13" && -f "$REFERENCE_SEED13_MODEL" ]]; then
    printf '%s' "$REFERENCE_SEED13_MODEL"
  else
    printf '%s/%s.model.json' "$OUT_DIR" "$(label_for_seed "$seed")"
  fi
}

run_seed() {
  local seed="$1"
  local label pred model
  label="$(label_for_seed "$seed")"
  pred="$(prediction_for_seed "$seed")"
  model="$(model_for_seed "$seed")"

  if [[ "$FORCE_RERUN" != "1" && -f "$pred" && -f "$model" ]]; then
    echo "reuse_seed_${seed}_prediction=$pred"
    echo "reuse_seed_${seed}_model=$model"
    return 0
  fi

  echo
  echo "== M3DocVQA CAPP seed stability: seed $seed =="
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
  AUTO_TUNE_BLEND_ALPHA=1 \
  TUNE_HIT_K=4 \
  TUNE_FRACTION="$TUNE_FRACTION" \
  TUNE_BLEND_ALPHA_GRID="$TUNE_BLEND_ALPHA_GRID" \
  POSITIVE_WEIGHT_CAP="$POSITIVE_WEIGHT_CAP" \
  SEED="$seed" \
  OUT_DIR="$OUT_DIR" \
  LABEL="$label" \
  bash "$REPO_ROOT/examples/run_m3docvqa_content_aware_pseudo_page_reranker.sh"
}

mkdir -p "$OUT_DIR"
if [[ "$POSITIVE_WEIGHT_CAP" != "20" ]]; then
  echo "This controlled seed audit requires POSITIVE_WEIGHT_CAP=20, got $POSITIVE_WEIGHT_CAP" >&2
  exit 1
fi
require_file train_gold "$TRAIN_GOLD"
require_file eval_gold "$EVAL_GOLD"
require_file base_train_pred "$BASE_TRAIN_PRED"
require_file base_eval_pred "$BASE_EVAL_PRED"

if [[ "$RUN_TRAIN" == "1" ]]; then
  for seed in $SEEDS; do
    run_seed "$seed"
  done
fi

if [[ "$RUN_EVAL" == "1" ]]; then
  eval_args=(--run "GPP=$BASE_EVAL_PRED")
  model_args=()
  for seed in $SEEDS; do
    pred="$(prediction_for_seed "$seed")"
    model="$(model_for_seed "$seed")"
    require_file "seed_${seed}_prediction" "$pred"
    require_file "seed_${seed}_model" "$model"
    eval_args+=(--run "seed_${seed}=$pred")
    model_args+=(--model "${seed}=$model")
  done

  "$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_pseudo_page_retrieval.py" \
    --gold "$EVAL_GOLD" \
    --recall-k 1 2 4 10 20 50 100 \
    "${eval_args[@]}" \
    --format markdown \
    --output "$OUT_DIR/seed_stability_runs.md"

  "$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_pseudo_page_retrieval.py" \
    --gold "$EVAL_GOLD" \
    --recall-k 1 2 4 10 20 50 100 \
    "${eval_args[@]}" \
    --format json \
    --output "$OUT_DIR/seed_stability_runs.json"

  "$PYTHON_BIN" "$REPO_ROOT/scripts/summarize_m3docvqa_capp_seed_stability.py" \
    --eval-json "$OUT_DIR/seed_stability_runs.json" \
    "${model_args[@]}" \
    --output-md "$OUT_DIR/seed_stability_summary.md" \
    --output-json "$OUT_DIR/seed_stability_summary.json"
fi
