#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/m3docvqa_internal_env.sh"

SPLADE_INDEX_OUT_DIR="${SPLADE_INDEX_OUT_DIR:-$LOCAL_OUTPUT_DIR/m3docvqa_splade}"
PAGE_TEXT_JSONL="${PAGE_TEXT_JSONL:-$LOCAL_OUTPUT_DIR/m3docvqa_page_text/m3docvqa_${SPLIT}_page_text.jsonl}"
SPLADE_MODEL_NAME_OR_PATH="${SPLADE_MODEL_NAME_OR_PATH:-naver/splade-cocondenser-ensembledistil}"
SPLADE_BATCH_SIZE="${SPLADE_BATCH_SIZE:-8}"
SPLADE_MAX_LENGTH="${SPLADE_MAX_LENGTH:-512}"
SPLADE_TOPK_TERMS="${SPLADE_TOPK_TERMS:-128}"
SPLADE_MIN_WEIGHT="${SPLADE_MIN_WEIGHT:-0.0}"
SPLADE_DEVICE="${SPLADE_DEVICE:-auto}"
SPLADE_MAX_PAGES="${SPLADE_MAX_PAGES:-0}"
REQUIRE_NONEMPTY_TEXT="${REQUIRE_NONEMPTY_TEXT:-1}"
SPLADE_INDEX_LABEL="${SPLADE_INDEX_LABEL:-m3docvqa_${SPLIT}_splade}"
OUTPUT_INDEX_PT="${OUTPUT_INDEX_PT:-$SPLADE_INDEX_OUT_DIR/${SPLADE_INDEX_LABEL}.pt}"
OUTPUT_SUMMARY_JSON="${OUTPUT_SUMMARY_JSON:-$SPLADE_INDEX_OUT_DIR/${SPLADE_INDEX_LABEL}.summary.json}"

mkdir -p "$SPLADE_INDEX_OUT_DIR"

echo "using_page_text_jsonl=$PAGE_TEXT_JSONL"
echo "using_splade_model_name_or_path=$SPLADE_MODEL_NAME_OR_PATH"
echo "using_output_index_pt=$OUTPUT_INDEX_PT"
echo "using_output_summary_json=$OUTPUT_SUMMARY_JSON"

ARGS=(
  --page-text-jsonl "$PAGE_TEXT_JSONL"
  --model-name-or-path "$SPLADE_MODEL_NAME_OR_PATH"
  --batch-size "$SPLADE_BATCH_SIZE"
  --max-length "$SPLADE_MAX_LENGTH"
  --topk-terms "$SPLADE_TOPK_TERMS"
  --min-weight "$SPLADE_MIN_WEIGHT"
  --device "$SPLADE_DEVICE"
  --max-pages "$SPLADE_MAX_PAGES"
  --output-index-pt "$OUTPUT_INDEX_PT"
  --output-summary-json "$OUTPUT_SUMMARY_JSON"
)
if [[ "$REQUIRE_NONEMPTY_TEXT" == "1" ]]; then
  ARGS+=(--require-nonempty-text)
fi

"$PYTHON_BIN" "$REPO_ROOT/scripts/build_splade_page_index.py" "${ARGS[@]}"
