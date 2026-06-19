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
EXPECTED_QID_COUNT="${EXPECTED_QID_COUNT:-2285}"

if [[ "$NUM_SHARDS" -gt 1 || -n "$LIMIT" ]]; then
  DEFAULT_RUN_EVAL=0
else
  DEFAULT_RUN_EVAL=1
fi
RUN_EVAL="${RUN_EVAL:-$DEFAULT_RUN_EVAL}"

AUGMENTED_GOLD="${AUGMENTED_GOLD:-$REPO_ROOT/output/m3docvqa_mmqa_pseudo_page_labels/mmqa_dev_pseudo_page_labels_strict.augmented_gold.jsonl}"
ORIGINAL_GOLD="${ORIGINAL_GOLD:-$GOLD}"
BASE_PRED="${BASE_PRED:-$REPO_ROOT/output/m3docvqa_gpp_hyperlink_node_ablation_exact_maxsim/mmqa_dev_exact_maxsim_gpp_hyperlink_node_no_hyperlink.prediction.json}"
ORACLE_OUT_DIR="${ORACLE_OUT_DIR:-$REPO_ROOT/output/m3docvqa_pseudo_gold_reader_oracle}"
QA_OUT_DIR="${QA_OUT_DIR:-$ORACLE_OUT_DIR/qa}"

RUN_GOLD_ONLY="${RUN_GOLD_ONLY:-1}"
RUN_GOLD_PLUS_BASE_FILL="${RUN_GOLD_PLUS_BASE_FILL:-1}"
BUILD_INPUTS="${BUILD_INPUTS:-1}"
RUN_QA="${RUN_QA:-1}"
REQUIRE_ALL_SUPPORT_DOCS_COVERED="${REQUIRE_ALL_SUPPORT_DOCS_COVERED:-0}"
REQUIRE_PSEUDO_PAGES_MATCH_SUPPORT_DOCS="${REQUIRE_PSEUDO_PAGES_MATCH_SUPPORT_DOCS:-0}"

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

build_input() {
  local label="$1"
  local fill_from_base="$2"
  local output_pred="$ORACLE_OUT_DIR/${label}.prediction.json"
  local output_gold="$ORACLE_OUT_DIR/${label}.gold.jsonl"
  local output_summary="$ORACLE_OUT_DIR/${label}.summary.json"

  mkdir -p "$ORACLE_OUT_DIR"
  local args=(
    --augmented-gold "$AUGMENTED_GOLD"
    --original-gold "$ORIGINAL_GOLD"
    --top-pages "$QA_TOP_PAGES"
    --output-prediction-json "$output_pred"
    --output-filtered-gold "$output_gold"
    --output-summary "$output_summary"
  )
  if [[ "$REQUIRE_ALL_SUPPORT_DOCS_COVERED" == "1" ]]; then
    args+=(--require-all-support-docs-covered)
  fi
  if [[ "$REQUIRE_PSEUDO_PAGES_MATCH_SUPPORT_DOCS" == "1" ]]; then
    args+=(--require-pseudo-pages-match-support-docs)
  fi
  if [[ "$fill_from_base" == "1" ]]; then
    require_file base_prediction "$BASE_PRED"
    args+=(--base-prediction "$BASE_PRED" --fill-from-base)
  fi

  "$PYTHON_BIN" "$REPO_ROOT/scripts/build_m3docvqa_pseudo_gold_reader_input.py" "${args[@]}"
}

run_qa() {
  local label="$1"
  local enabled="$2"
  if [[ "$enabled" != "1" ]]; then
    echo "skip_${label}_qa=disabled"
    return 0
  fi

  local prediction_json="$ORACLE_OUT_DIR/${label}.prediction.json"
  local filtered_gold="$ORACLE_OUT_DIR/${label}.gold.jsonl"
  require_file "${label}_prediction" "$prediction_json"
  require_file "${label}_filtered_gold" "$filtered_gold"

  local output_label
  output_label="$(qa_label "$label")"
  local output_pred="$QA_OUT_DIR/${output_label}.prediction.json"
  local output_eval="$QA_OUT_DIR/${output_label}.eval.json"

  echo
  echo "== M3DocVQA pseudo-gold reader oracle QA: $label =="
  echo "using_prediction_json=$prediction_json"
  echo "using_filtered_gold=$filtered_gold"
  echo "using_output_pred=$output_pred"
  echo "using_output_eval=$output_eval"
  echo "using_qa_top_pages=$QA_TOP_PAGES"
  echo "using_shard_id=$SHARD_ID"
  echo "using_num_shards=$NUM_SHARDS"

  local args=(
    --prediction-json "$prediction_json"
    --gold "$filtered_gold"
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
  if [[ "$enabled" != "1" || "$NUM_SHARDS" -le 1 ]]; then
    return 0
  fi

  local filtered_gold="$ORACLE_OUT_DIR/${label}.gold.jsonl"
  local merged_label="mmqa_dev_${label}_qwen2vl_top${QA_TOP_PAGES}"
  local input_glob="$QA_OUT_DIR/mmqa_dev_${label}_qwen2vl_top${QA_TOP_PAGES}_shard*_of_${NUM_SHARDS}.prediction.json"
  local output_pred="$QA_OUT_DIR/${merged_label}.prediction.json"
  local output_eval="$QA_OUT_DIR/${merged_label}.eval.json"

  echo
  echo "== Merge pseudo-gold reader oracle QA shards: $label =="
  "$PYTHON_BIN" "$REPO_ROOT/scripts/merge_m3docvqa_prediction_shards.py" \
    --input-glob "$input_glob" \
    --output-pred "$output_pred" \
    --gold "$filtered_gold" \
    --output-eval "$output_eval" \
    --expected-count "$EXPECTED_QID_COUNT"
}

require_file augmented_gold "$AUGMENTED_GOLD"
mkdir -p "$QA_OUT_DIR"

if [[ "$MERGE_QA_SHARDS" != "1" ]]; then
  if [[ "$BUILD_INPUTS" == "1" && "$RUN_GOLD_ONLY" == "1" ]]; then
    build_input pseudo_gold_only 0
  fi
  if [[ "$BUILD_INPUTS" == "1" && "$RUN_GOLD_PLUS_BASE_FILL" == "1" ]]; then
    build_input pseudo_gold_plus_gpp_fill 1
  fi

  if [[ "$RUN_QA" == "1" ]]; then
    run_qa pseudo_gold_only "$RUN_GOLD_ONLY"
    run_qa pseudo_gold_plus_gpp_fill "$RUN_GOLD_PLUS_BASE_FILL"
  else
    echo "skip_qa=disabled"
  fi
else
  merge_qa pseudo_gold_only "$RUN_GOLD_ONLY"
  merge_qa pseudo_gold_plus_gpp_fill "$RUN_GOLD_PLUS_BASE_FILL"
fi
