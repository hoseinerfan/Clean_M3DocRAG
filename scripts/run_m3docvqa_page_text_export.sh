#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/m3docvqa_internal_env.sh"

PAGE_TEXT_OUT_DIR="${PAGE_TEXT_OUT_DIR:-$LOCAL_OUTPUT_DIR/m3docvqa_page_text}"
PAGE_TEXT_LABEL="${PAGE_TEXT_LABEL:-m3docvqa_${SPLIT}_page_text}"
OUTPUT_JSONL="${OUTPUT_JSONL:-$PAGE_TEXT_OUT_DIR/${PAGE_TEXT_LABEL}.jsonl}"
OUTPUT_SUMMARY_JSON="${OUTPUT_SUMMARY_JSON:-$PAGE_TEXT_OUT_DIR/${PAGE_TEXT_LABEL}.summary.json}"
DOC_ID_JSON="${DOC_ID_JSON:-}"
MAX_DOCS="${MAX_DOCS:-0}"
MAX_PAGES_PER_DOC="${MAX_PAGES_PER_DOC:-0}"
PDFTOTEXT_BIN="${PDFTOTEXT_BIN:-pdftotext}"
PDFINFO_BIN="${PDFINFO_BIN:-pdfinfo}"

mkdir -p "$PAGE_TEXT_OUT_DIR"

echo "using_local_data_dir=$LOCAL_DATA_DIR"
echo "using_data_name=$DATA_NAME"
echo "using_split=$SPLIT"
echo "using_output_jsonl=$OUTPUT_JSONL"
echo "using_output_summary_json=$OUTPUT_SUMMARY_JSON"

ARGS=(
  --data-name "$DATA_NAME"
  --split "$SPLIT"
  --max-docs "$MAX_DOCS"
  --max-pages-per-doc "$MAX_PAGES_PER_DOC"
  --pdftotext-bin "$PDFTOTEXT_BIN"
  --pdfinfo-bin "$PDFINFO_BIN"
  --output-jsonl "$OUTPUT_JSONL"
  --output-summary-json "$OUTPUT_SUMMARY_JSON"
)
if [[ -n "$DOC_ID_JSON" ]]; then
  ARGS+=(--doc-id-json "$DOC_ID_JSON")
fi

"$PYTHON_BIN" "$REPO_ROOT/scripts/export_m3docvqa_page_text.py" "${ARGS[@]}"
