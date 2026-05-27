#!/usr/bin/env bash
set -euo pipefail

# Summarize completed native-versus-PyMuPDF4LLM Markdown/gate experiments.
#
# Usage on HPC:
#   DATASETS="vidoseek sciegqa mmdocir" bash examples/report_pdf_markdown_backend_comparison.sh
#   REPORT_ROOT=/path/to/reports DATASETS="dude" bash examples/report_pdf_markdown_backend_comparison.sh
#   VIDOSEEK_NATIVE_SAFE_SUMMARY=/path/to/historical_no_page0.summary.json DATASETS="vidoseek" bash examples/report_pdf_markdown_backend_comparison.sh
#
# This script reads completed artifacts. Run the PyMuPDF4LLM gate pipeline first
# for any dataset that has no `_pymupdf4llm_source_ablation` output directory yet.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/env/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

DATASETS="${DATASETS:-vidoseek sciegqa mmdocir}"
REPORT_ROOT="${REPORT_ROOT:-$REPO_ROOT/output/pdf_markdown_backend_comparison}"
mkdir -p "$REPORT_ROOT"

compare_pair() {
  local dataset="$1"
  local native_jsonl="$2"
  local alternate_jsonl="$3"
  local native_summary="$4"
  local alternate_summary="$5"
  local native_variant="$6"
  local alternate_variant="$7"
  local native_full="$8"
  local alternate_full="$9"
  local native_strict="${10}"
  local alternate_strict="${11}"
  local native_safe="${12}"
  local alternate_safe="${13}"

  if [[ ! -f "$native_jsonl" ]]; then
    echo "skip_${dataset}_missing_native_jsonl: $native_jsonl" >&2
    return 0
  fi
  if [[ ! -f "$alternate_jsonl" ]]; then
    echo "skip_${dataset}_missing_pymupdf4llm_jsonl: $alternate_jsonl" >&2
    return 0
  fi

  "$PYTHON_BIN" "$REPO_ROOT/scripts/compare_pdf_markdown_backends.py" \
    --dataset "$dataset" \
    --native-jsonl "$native_jsonl" \
    --alternate-jsonl "$alternate_jsonl" \
    --native-summary "$native_summary" \
    --alternate-summary "$alternate_summary" \
    --native-variant-summary "$native_variant" \
    --alternate-variant-summary "$alternate_variant" \
    --native-full-summary "$native_full" \
    --alternate-full-summary "$alternate_full" \
    --native-strict-summary "$native_strict" \
    --alternate-strict-summary "$alternate_strict" \
    --native-safe-summary "$native_safe" \
    --alternate-safe-summary "$alternate_safe" \
    --sample-limit "${SAMPLE_LIMIT:-5}" \
    --excerpt-chars "${EXCERPT_CHARS:-900}" \
    --output-json "$REPORT_ROOT/${dataset}_pdf_markdown_backend_comparison.json" \
    --output-md "$REPORT_ROOT/${dataset}_pdf_markdown_backend_comparison.md"
}

report_mmdocir() {
  unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR LOCAL_MODEL_DIR
  # shellcheck disable=SC1091
  source "$REPO_ROOT/mmdocir/env_hpc.sh"
  local native_jsonl="${MMDOCIR_PDF_MD_JSONL:-$LOCAL_OUTPUT_DIR/mmdocir/pdf_markdown/doc_pages_dev_with_pdf_markdown.jsonl}"
  local native_out="$LOCAL_OUTPUT_DIR/mmdocir/heading_breadcrumb_pdf_markdown_source_ablation"
  local alternate_out="$LOCAL_OUTPUT_DIR/mmdocir/heading_breadcrumb_pdf_markdown_pymupdf4llm_source_ablation"
  compare_pair \
    mmdocir \
    "$native_jsonl" \
    "$alternate_out/doc_pages_dev_with_pdf_markdown.jsonl" \
    "$(dirname "$native_jsonl")/pdf_markdown_summary.json" \
    "$alternate_out/pdf_markdown_summary.json" \
    "${MMDOCIR_VARIANT_SUMMARY:-$LOCAL_OUTPUT_DIR/mmdocir/pdf_markdown_variants/pdf_markdown_variants.summary.json}" \
    "$alternate_out/pdf_markdown_variants/pdf_markdown_variants.summary.json" \
    "$native_out/mmdocir_heading_full_wide_edgeonly_transfer.summary.json" \
    "$alternate_out/mmdocir_heading_full_wide_edgeonly_transfer.summary.json" \
    "$native_out/mmdocir_heading_strict_heading_wide_edgeonly_transfer.summary.json" \
    "$alternate_out/mmdocir_heading_strict_heading_wide_edgeonly_transfer.summary.json" \
    "$native_out/mmdocir_heuristic_strict_safe_gate_bodyguard.summary.json" \
    "$alternate_out/mmdocir_safe_gate_bodyguard.summary.json"
}

report_sciegqa() {
  unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR LOCAL_MODEL_DIR
  # shellcheck disable=SC1091
  source "$REPO_ROOT/sciegqa/env_hpc.sh"
  local native_jsonl="${SCIEGQA_PDF_MD_JSONL:-$LOCAL_OUTPUT_DIR/sciegqa/pdf_markdown/doc_pages_dev_with_pdf_markdown.jsonl}"
  local native_out="$LOCAL_OUTPUT_DIR/sciegqa/heading_breadcrumb_pdf_markdown_source_ablation"
  local alternate_out="$LOCAL_OUTPUT_DIR/sciegqa/heading_breadcrumb_pdf_markdown_pymupdf4llm_source_ablation"
  compare_pair \
    sciegqa \
    "$native_jsonl" \
    "$alternate_out/doc_pages_dev_with_pdf_markdown.jsonl" \
    "$(dirname "$native_jsonl")/pdf_markdown_summary.json" \
    "$alternate_out/pdf_markdown_summary.json" \
    "${SCIEGQA_VARIANT_SUMMARY:-$LOCAL_OUTPUT_DIR/sciegqa/pdf_markdown_variants/pdf_markdown_variants.summary.json}" \
    "$alternate_out/pdf_markdown_variants/pdf_markdown_variants.summary.json" \
    "$native_out/sciegqa_heading_full_wide_edgeonly_transfer.summary.json" \
    "$alternate_out/sciegqa_heading_full_wide_edgeonly_transfer.summary.json" \
    "$native_out/sciegqa_heading_strict_heading_wide_edgeonly_transfer.summary.json" \
    "$alternate_out/sciegqa_heading_strict_heading_wide_edgeonly_transfer.summary.json" \
    "$native_out/sciegqa_safe_gate_bodyguard.summary.json" \
    "$alternate_out/sciegqa_safe_gate_bodyguard.summary.json"
}

report_vidoseek() {
  unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR LOCAL_MODEL_DIR
  # shellcheck disable=SC1091
  source "$REPO_ROOT/vidoseek/env_hpc.sh"
  local native_jsonl="${VIDOSEEK_PDF_MD_JSONL:-$LOCAL_OUTPUT_DIR/vidoseek/pdf_markdown/doc_pages_dev_with_pdf_markdown.jsonl}"
  local native_out="$LOCAL_OUTPUT_DIR/vidoseek/heading_breadcrumb_pdf_markdown_source_ablation"
  local alternate_out="$LOCAL_OUTPUT_DIR/vidoseek/heading_breadcrumb_pdf_markdown_pymupdf4llm_source_ablation"
  local native_safe="${VIDOSEEK_NATIVE_SAFE_SUMMARY:-$native_out/vidoseek_safe_gate_bodyguard.summary.json}"
  local alternate_safe="${VIDOSEEK_ALTERNATE_SAFE_SUMMARY:-$alternate_out/vidoseek_safe_gate_bodyguard.summary.json}"
  compare_pair \
    vidoseek \
    "$native_jsonl" \
    "$alternate_out/doc_pages_dev_with_pdf_markdown.jsonl" \
    "$(dirname "$native_jsonl")/pdf_markdown_summary.json" \
    "$alternate_out/pdf_markdown_summary.json" \
    "${VIDOSEEK_VARIANT_SUMMARY:-$LOCAL_OUTPUT_DIR/vidoseek/pdf_markdown_variants/pdf_markdown_variants.summary.json}" \
    "$alternate_out/pdf_markdown_variants/pdf_markdown_variants.summary.json" \
    "$native_out/vidoseek_heading_full_wide_edgeonly_transfer.summary.json" \
    "$alternate_out/vidoseek_heading_full_wide_edgeonly_transfer.summary.json" \
    "$native_out/vidoseek_heading_strict_heading_wide_edgeonly_transfer.summary.json" \
    "$alternate_out/vidoseek_heading_strict_heading_wide_edgeonly_transfer.summary.json" \
    "$native_safe" \
    "$alternate_safe"
}

report_dude() {
  unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR LOCAL_MODEL_DIR
  # shellcheck disable=SC1091
  source "$REPO_ROOT/dude/env_hpc.sh"
  local native_out="$LOCAL_OUTPUT_DIR/dude/heading_breadcrumb_pdf_markdown_source_ablation"
  local alternate_out="$LOCAL_OUTPUT_DIR/dude/heading_breadcrumb_pdf_markdown_pymupdf4llm_source_ablation"
  local native_safe="${DUDE_NATIVE_SAFE_SUMMARY:-$native_out/dude_safe_gate_bodyguard.summary.json}"
  local alternate_safe="${DUDE_ALTERNATE_SAFE_SUMMARY:-$alternate_out/dude_safe_gate_bodyguard.summary.json}"
  compare_pair \
    dude \
    "$native_out/doc_pages_dev_with_pdf_markdown.jsonl" \
    "$alternate_out/doc_pages_dev_with_pdf_markdown.jsonl" \
    "$native_out/pdf_markdown_summary.json" \
    "$alternate_out/pdf_markdown_summary.json" \
    "$native_out/pdf_markdown_variants/pdf_markdown_variants.summary.json" \
    "$alternate_out/pdf_markdown_variants/pdf_markdown_variants.summary.json" \
    "$native_out/dude_heading_full_wide_edgeonly_transfer.summary.json" \
    "$alternate_out/dude_heading_full_wide_edgeonly_transfer.summary.json" \
    "$native_out/dude_heading_strict_heading_wide_edgeonly_transfer.summary.json" \
    "$alternate_out/dude_heading_strict_heading_wide_edgeonly_transfer.summary.json" \
    "$native_safe" \
    "$alternate_safe"
}

for dataset in $DATASETS; do
  case "$dataset" in
    mmdocir|mm-docir)
      report_mmdocir
      ;;
    sciegqa|sci-egqa)
      report_sciegqa
      ;;
    vidoseek)
      report_vidoseek
      ;;
    dude)
      report_dude
      ;;
    *)
      echo "unknown_dataset: $dataset" >&2
      exit 2
      ;;
  esac
done

echo "report_root: $REPORT_ROOT"
