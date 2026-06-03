#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

VITAL_PATHS_ENV="${VITAL_PATHS_ENV:-$REPO_ROOT/hpc_vital_paths.generated.env}"
if [[ -f "$VITAL_PATHS_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$VITAL_PATHS_ENV"
fi

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/m3docvqa_internal_env.sh"

CUSTOM_ROOT="${CUSTOM_ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen2-VL-7B-Instruct}"
BITS="${BITS:-16}"
QA_TOP_PAGES="${QA_TOP_PAGES:-4}"
DOC_IMAGE_CACHE_SIZE="${DOC_IMAGE_CACHE_SIZE:-16}"
SAVE_EVERY="${SAVE_EVERY:-25}"
RESUME="${RESUME:-1}"
LIMIT="${LIMIT:-}"
SHARD_ID="${SHARD_ID:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
MERGE_QA_SHARDS="${MERGE_QA_SHARDS:-0}"
EXPECTED_QID_COUNT="${EXPECTED_QID_COUNT:-0}"

if [[ "$NUM_SHARDS" -gt 1 || -n "$LIMIT" ]]; then
  DEFAULT_RUN_EVAL=0
else
  DEFAULT_RUN_EVAL=1
fi
RUN_EVAL="${RUN_EVAL:-$DEFAULT_RUN_EVAL}"

QA_OUT_DIR="${QA_OUT_DIR:-$REPO_ROOT/output/m3docvqa_final_qa_comparison}"

DENSE_PRED="${DENSE_PRED:-$CUSTOM_ROOT/outputs/mmqa_dev_plain_top224_nprobe4_effdiag_all.prediction.json}"
GPP_NO_HYPERLINK_PRED="${GPP_NO_HYPERLINK_PRED:-$REPO_ROOT/output/m3docvqa_gpp_hyperlink_node_ablation_mmr_target1/mmqa_dev_gpp_hyperlink_node_no_hyperlink.prediction.json}"
CONTENT_AWARE_PRED="${CONTENT_AWARE_PRED:-$REPO_ROOT/output/m3docvqa_content_aware_auto_blend_sweep_page5/mmqa_train_to_dev_content_aware_base_gpp_no_hyperlink.dev.prediction.json}"
OOF_HYBRID_INSERT4_PRED="${OOF_HYBRID_INSERT4_PRED:-$REPO_ROOT/output/m3docvqa_content_aware_counterfactual_oof_hybrid/insert_rank4/mmqa_train_to_dev_content_aware_counterfactual_oof_insert4.dev.prediction.json}"

RUN_DENSE_QA="${RUN_DENSE_QA:-1}"
RUN_GPP_NO_HYPERLINK_QA="${RUN_GPP_NO_HYPERLINK_QA:-1}"
RUN_CONTENT_AWARE_QA="${RUN_CONTENT_AWARE_QA:-1}"
RUN_OOF_HYBRID_INSERT4_QA="${RUN_OOF_HYBRID_INSERT4_QA:-1}"

require_file() {
  local name="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    echo "missing_${name}: $path" >&2
    exit 1
  fi
}

qa_label() {
  local label="$1"
  if [[ "$NUM_SHARDS" -gt 1 ]]; then
    printf 'mmqa_dev_%s_qwen2vl_top%s_shard%s_of_%s' "$label" "$QA_TOP_PAGES" "$SHARD_ID" "$NUM_SHARDS"
  else
    printf 'mmqa_dev_%s_qwen2vl_top%s' "$label" "$QA_TOP_PAGES"
  fi
}

run_qa() {
  local label="$1"
  local prediction_json="$2"
  local enabled="$3"
  if [[ "$enabled" != "1" ]]; then
    echo "skip_${label}_qa=disabled"
    return 0
  fi

  require_file "${label}_prediction" "$prediction_json"
  local output_label
  output_label="$(qa_label "$label")"
  local output_pred="$QA_OUT_DIR/${output_label}.prediction.json"
  local output_eval="$QA_OUT_DIR/${output_label}.eval.json"

  echo
  echo "== M3DocVQA QA: $label =="
  echo "using_prediction_json=$prediction_json"
  echo "using_output_pred=$output_pred"
  echo "using_output_eval=$output_eval"
  echo "using_qa_top_pages=$QA_TOP_PAGES"
  echo "using_shard_id=$SHARD_ID"
  echo "using_num_shards=$NUM_SHARDS"

  local args=(
    --prediction-json "$prediction_json"
    --gold "$GOLD"
    --data-name "$DATA_NAME"
    --split "$SPLIT"
    --model-name-or-path "$MODEL_NAME_OR_PATH"
    --bits "$BITS"
    --qa-top-pages "$QA_TOP_PAGES"
    --eval-shard-id "$SHARD_ID"
    --eval-num-shards "$NUM_SHARDS"
    --doc-image-cache-size "$DOC_IMAGE_CACHE_SIZE"
    --save-every "$SAVE_EVERY"
    --output-prediction-json "$output_pred"
    --output-eval-json "$output_eval"
  )
  if [[ -n "$QUESTION_TYPE_FILTER" ]]; then
    args+=(--question-type-filter "$QUESTION_TYPE_FILTER")
  fi
  if [[ -n "$LIMIT" ]]; then
    args+=(--limit "$LIMIT")
  fi
  if [[ "$RUN_EVAL" == "1" ]]; then
    args+=(--run-eval)
  fi
  if [[ "$RESUME" == "1" ]]; then
    args+=(--resume)
  fi

  "$PYTHON_BIN" "$REPO_ROOT/scripts/run_m3docvqa_external_retrieval_qa.py" "${args[@]}"
}

merge_qa() {
  local label="$1"
  local enabled="$2"
  if [[ "$enabled" != "1" ]]; then
    return 0
  fi
  if [[ "$NUM_SHARDS" -le 1 ]]; then
    return 0
  fi
  local merged_label="mmqa_dev_${label}_qwen2vl_top${QA_TOP_PAGES}"
  local input_glob="$QA_OUT_DIR/mmqa_dev_${label}_qwen2vl_top${QA_TOP_PAGES}_shard*_of_${NUM_SHARDS}.prediction.json"
  local output_pred="$QA_OUT_DIR/${merged_label}.prediction.json"
  local output_eval="$QA_OUT_DIR/${merged_label}.eval.json"

  echo
  echo "== Merge M3DocVQA QA shards: $label =="
  "$PYTHON_BIN" "$REPO_ROOT/scripts/merge_m3docvqa_prediction_shards.py" \
    --input-glob "$input_glob" \
    --output-pred "$output_pred" \
    --gold "$GOLD" \
    --output-eval "$output_eval" \
    --expected-count "$EXPECTED_QID_COUNT"
}

mkdir -p "$QA_OUT_DIR"

if [[ "$MERGE_QA_SHARDS" != "1" ]]; then
  run_qa dense "$DENSE_PRED" "$RUN_DENSE_QA"
  run_qa gpp_no_hyperlink "$GPP_NO_HYPERLINK_PRED" "$RUN_GPP_NO_HYPERLINK_QA"
  run_qa content_aware "$CONTENT_AWARE_PRED" "$RUN_CONTENT_AWARE_QA"
  run_qa oof_hybrid_insert4 "$OOF_HYBRID_INSERT4_PRED" "$RUN_OOF_HYBRID_INSERT4_QA"
else
  merge_qa dense "$RUN_DENSE_QA"
  merge_qa gpp_no_hyperlink "$RUN_GPP_NO_HYPERLINK_QA"
  merge_qa content_aware "$RUN_CONTENT_AWARE_QA"
  merge_qa oof_hybrid_insert4 "$RUN_OOF_HYBRID_INSERT4_QA"
fi
