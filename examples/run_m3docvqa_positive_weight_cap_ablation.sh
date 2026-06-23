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

OUT_DIR="${OUT_DIR:-$REPO_ROOT/output/m3docvqa_content_aware_exact_maxsim_positive_weight_cap_ablation}"
TRAIN_GOLD="${TRAIN_GOLD:-$REPO_ROOT/output/m3docvqa_mmqa_pseudo_page_labels/mmqa_train_pseudo_page_labels_strict.augmented_gold.jsonl}"
EVAL_GOLD="${EVAL_GOLD:-$REPO_ROOT/output/m3docvqa_mmqa_pseudo_page_labels/mmqa_dev_pseudo_page_labels_strict.augmented_gold.jsonl}"

GPP_TRAIN_OUT_DIR="${GPP_TRAIN_OUT_DIR:-$REPO_ROOT/output/m3docvqa_gpp_hyperlink_node_ablation_exact_maxsim_train}"
GPP_EVAL_OUT_DIR="${GPP_EVAL_OUT_DIR:-$REPO_ROOT/output/m3docvqa_gpp_hyperlink_node_ablation_exact_maxsim}"
GPP_TRAIN_LABEL_PREFIX="${GPP_TRAIN_LABEL_PREFIX:-mmqa_train_exact_maxsim_gpp_hyperlink_node}"
GPP_EVAL_LABEL_PREFIX="${GPP_EVAL_LABEL_PREFIX:-mmqa_dev_exact_maxsim_gpp_hyperlink_node}"
BASE_LABEL="${BASE_LABEL:-exact_maxsim_gpp_no_hyperlink}"
BASE_TRAIN_PRED="${BASE_TRAIN_PRED:-$GPP_TRAIN_OUT_DIR/${GPP_TRAIN_LABEL_PREFIX}_no_hyperlink.prediction.json}"
BASE_EVAL_PRED="${BASE_EVAL_PRED:-$GPP_EVAL_OUT_DIR/${GPP_EVAL_LABEL_PREFIX}_no_hyperlink.prediction.json}"

# Cap 20 is the existing final run. The other defaults test underweighting,
# near-balancing, and exact class balancing for the observed 34.132 ratio.
CAPS="${CAPS:-10 30 34.132143}"
REFERENCE_CAP20_PRED="${REFERENCE_CAP20_PRED:-$REPO_ROOT/output/m3docvqa_content_aware_exact_maxsim_adaptive_topk_ablation/mmqa_train_to_dev_content_aware_auto_page4_base_exact_maxsim_gpp_no_hyperlink.dev.prediction.json}"
TUNE_BLEND_ALPHA_GRID="${TUNE_BLEND_ALPHA_GRID:-0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50}"
TUNE_FRACTION="${TUNE_FRACTION:-0.20}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"

tag_value() {
  printf '%s' "$1" | tr '.-' 'pm'
}

require_file() {
  local name="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    echo "missing_${name}: $path" >&2
    exit 1
  fi
}

prediction_for_cap() {
  local cap_tag
  cap_tag="$(tag_value "$1")"
  printf '%s/mmqa_train_to_dev_content_aware_poscap_%s_auto_page4_base_%s.dev.prediction.json' \
    "$OUT_DIR" "$cap_tag" "$BASE_LABEL"
}

run_cap() {
  local cap="$1"
  local cap_tag label pred
  cap_tag="$(tag_value "$cap")"
  label="mmqa_train_to_dev_content_aware_poscap_${cap_tag}_auto_page4_base_${BASE_LABEL}"
  pred="$OUT_DIR/${label}.dev.prediction.json"

  if [[ "$FORCE_RERUN" != "1" && -f "$pred" ]]; then
    echo "reuse_positive_weight_cap_${cap_tag}=$pred"
    return 0
  fi

  echo
  echo "== M3DocVQA positive-weight cap: $cap =="
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
  POSITIVE_WEIGHT_CAP="$cap" \
  OUT_DIR="$OUT_DIR" \
  LABEL="$label" \
  bash "$REPO_ROOT/examples/run_m3docvqa_content_aware_pseudo_page_reranker.sh"
}

mkdir -p "$OUT_DIR"
require_file train_gold "$TRAIN_GOLD"
require_file eval_gold "$EVAL_GOLD"
require_file base_train_pred "$BASE_TRAIN_PRED"
require_file base_eval_pred "$BASE_EVAL_PRED"

if [[ "$RUN_TRAIN" == "1" ]]; then
  for cap in $CAPS; do
    run_cap "$cap"
  done
fi

if [[ "$RUN_EVAL" == "1" ]]; then
  eval_args=(--run "GPP=$BASE_EVAL_PRED")
  if [[ -f "$REFERENCE_CAP20_PRED" ]]; then
    eval_args+=(--run "cap_20=$REFERENCE_CAP20_PRED")
  else
    echo "eval_skip_missing_cap_20=$REFERENCE_CAP20_PRED" >&2
  fi
  for cap in $CAPS; do
    cap_tag="$(tag_value "$cap")"
    pred="$(prediction_for_cap "$cap")"
    if [[ -f "$pred" ]]; then
      eval_args+=(--run "cap_${cap_tag}=$pred")
    else
      echo "eval_skip_missing_cap_${cap_tag}=$pred" >&2
    fi
  done

  "$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_pseudo_page_retrieval.py" \
    --gold "$EVAL_GOLD" \
    --recall-k 1 2 4 10 20 50 100 \
    "${eval_args[@]}" \
    --output "$OUT_DIR/positive_weight_cap_ablation.md"

  echo "saved_eval=$OUT_DIR/positive_weight_cap_ablation.md"
fi
