#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

VITAL_PATHS_ENV="${VITAL_PATHS_ENV:-$REPO_ROOT/hpc_vital_paths.generated.env}"
if [[ -f "$VITAL_PATHS_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$VITAL_PATHS_ENV"
fi

unset M3DOCVQA_INTERNAL_ENV_LOADED
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/m3docvqa_internal_env.sh"

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    echo "missing_${label}: $path" >&2
    exit 1
  fi
}

first_existing_path() {
  local path
  for path in "$@"; do
    if [[ -n "$path" && -f "$path" ]]; then
      echo "$path"
      return 0
    fi
  done
  return 1
}

SPLIT="${SPLIT:-dev}"
GOLD="${M3DOCVQA_GOLD:-$GOLD}"
DENSE_PRED="${M3DOCVQA_DENSE_PRED:-$LOCAL_OUTPUT_DIR/m3docvqa_plain_top224_mmqa_${SPLIT}/mmqa_${SPLIT}_plain_top224_nprobe${FAISS_NPROBE}_effdiag_all.prediction.json}"
SPARSE_PRED="${M3DOCVQA_SPARSE_PRED:-$LOCAL_OUTPUT_DIR/m3docvqa_splade_mmqa_${SPLIT}/mmqa_${SPLIT}_splade.prediction.json}"
DOC_PAGES_JSONL="${M3DOCVQA_PAGE_TEXT_JSONL:-$LOCAL_OUTPUT_DIR/m3docvqa_page_text/m3docvqa_${SPLIT}_page_text.jsonl}"
SPLADE_INDEX_PT="${SPLADE_INDEX_PT:-$LOCAL_OUTPUT_DIR/m3docvqa_splade/m3docvqa_${SPLIT}_splade.pt}"
GRAPH_OUT_DIR="${GRAPH_OUT_DIR:-$LOCAL_OUTPUT_DIR/m3docvqa_doc_doc_edge_ablation}"
LABEL_PREFIX="${LABEL_PREFIX:-mmqa_${SPLIT}_docdoc_ablation}"
GRAPH_PROFILE="${GRAPH_PROFILE:-denseheavy125_medium_both}"
DOC_DOC_EDGE_WEIGHT="${DOC_DOC_EDGE_WEIGHT:-0.10}"
DOC_DOC_TOP_DOCS="${DOC_DOC_TOP_DOCS:-20}"
DOC_DOC_MAX_EDGES_PER_DOC="${DOC_DOC_MAX_EDGES_PER_DOC:-8}"
RECALL_K_VALUES="${RECALL_K_VALUES:-1 2 4 5 10 20 50 100 500 1000}"

CUSTOM_ROOT="${CUSTOM_ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom}"
PDF_HYPERLINK_EDGES_JSONL="$(
  first_existing_path \
    "${M3DOCVQA_PDF_HYPERLINK_EDGES_JSONL:-}" \
    "${PDF_HYPERLINK_EDGES_JSONL:-}" \
    "${MMDOCIR_PDF_HYPERLINK_EDGES_JSONL:-}" \
    "$LOCAL_OUTPUT_DIR/m3docvqa_hyperlink_audit_mapped_full.edges.jsonl" \
    "$REPO_ROOT/output/m3docvqa_hyperlink_audit_mapped_full.edges.jsonl" \
    "$CUSTOM_ROOT/MMDocIR_M3DocRAG/output/m3docvqa_hyperlink_audit_mapped_full.edges.jsonl" \
    || true
)"

require_file gold "$GOLD"
require_file dense_pred "$DENSE_PRED"
require_file sparse_pred "$SPARSE_PRED"
if [[ -n "$DOC_PAGES_JSONL" && ! -f "$DOC_PAGES_JSONL" ]]; then
  echo "warning_missing_doc_pages_jsonl: $DOC_PAGES_JSONL" >&2
fi

mkdir -p "$GRAPH_OUT_DIR"

run_variant() {
  local variant="$1"
  local edge_mode="$2"
  local edge_weight="$3"
  local label="${LABEL_PREFIX}_${variant}"

  echo
  echo "== m3docvqa: $variant =="
  DATA_NAME="m3-docvqa" \
  SPLIT="$SPLIT" \
  GOLD="$GOLD" \
  DENSE_PRED="$DENSE_PRED" \
  SPARSE_PRED="$SPARSE_PRED" \
  DOC_PAGES_JSONL="$DOC_PAGES_JSONL" \
  SPLADE_INDEX_PT="$SPLADE_INDEX_PT" \
  GRAPH_OUT_DIR="$GRAPH_OUT_DIR" \
  GRAPH_PROFILE="$GRAPH_PROFILE" \
  GRAPH_LABEL="$label" \
  RECALL_K_VALUES="$RECALL_K_VALUES" \
  DOC_DOC_EDGE_MODE="$edge_mode" \
  DOC_DOC_EDGE_WEIGHT="$edge_weight" \
  DOC_DOC_TOP_DOCS="$DOC_DOC_TOP_DOCS" \
  DOC_DOC_MAX_EDGES_PER_DOC="$DOC_DOC_MAX_EDGES_PER_DOC" \
  PDF_HYPERLINK_EDGES_JSONL="$PDF_HYPERLINK_EDGES_JSONL" \
  bash "$REPO_ROOT/scripts/run_m3docvqa_page_preserving_graph_pipeline.sh"
}

run_variant no_doc_doc none 0.0

if [[ -n "$PDF_HYPERLINK_EDGES_JSONL" && -f "$PDF_HYPERLINK_EDGES_JSONL" ]]; then
  echo "using_pdf_hyperlink_edges=$PDF_HYPERLINK_EDGES_JSONL"
  run_variant hyperlink_citation hyperlink_citation "$DOC_DOC_EDGE_WEIGHT"

  "$PYTHON_BIN" "$REPO_ROOT/scripts/compare_m3docvqa_retrieval_runs.py" \
    --baseline "$GRAPH_OUT_DIR/${LABEL_PREFIX}_no_doc_doc.prediction.json" \
    --candidate "$GRAPH_OUT_DIR/${LABEL_PREFIX}_hyperlink_citation.prediction.json" \
    --gold "$GOLD" \
    --recall-k $RECALL_K_VALUES \
    --json > "$GRAPH_OUT_DIR/${LABEL_PREFIX}_hyperlink_citation.vs_no_doc_doc.json"

  "$PYTHON_BIN" "$REPO_ROOT/scripts/audit_m3docvqa_hyperlink_effects.py" \
    --baseline "$GRAPH_OUT_DIR/${LABEL_PREFIX}_no_doc_doc.prediction.json" \
    --candidate "$GRAPH_OUT_DIR/${LABEL_PREFIX}_hyperlink_citation.prediction.json" \
    --gold "$GOLD" \
    --hyperlink-edges-jsonl "$PDF_HYPERLINK_EDGES_JSONL" \
    --source-top-pages 1000 \
    --hit-k 4 \
    --output-json "$GRAPH_OUT_DIR/${LABEL_PREFIX}_hyperlink_citation.audit.json" \
    --output-md "$GRAPH_OUT_DIR/${LABEL_PREFIX}_hyperlink_citation.audit.md"
else
  echo "skip_hyperlink_citation_missing_pdf_hyperlink_edges: set M3DOCVQA_PDF_HYPERLINK_EDGES_JSONL" >&2
fi

analysis_paths=( "$GRAPH_OUT_DIR/${LABEL_PREFIX}_"*.retrieval_analysis.json )
if [[ -e "${analysis_paths[0]}" ]]; then
  "$PYTHON_BIN" "$REPO_ROOT/scripts/summarize_m3docvqa_retrieval_analyses.py" \
    --format markdown \
    "${analysis_paths[@]}" \
    | tee "$GRAPH_OUT_DIR/${LABEL_PREFIX}_recall_table.md"

  "$PYTHON_BIN" "$REPO_ROOT/scripts/summarize_m3docvqa_retrieval_analyses.py" \
    --format csv \
    "${analysis_paths[@]}" \
    > "$GRAPH_OUT_DIR/${LABEL_PREFIX}_recall_table.csv"

  echo "saved_recall_table_md=$GRAPH_OUT_DIR/${LABEL_PREFIX}_recall_table.md"
  echo "saved_recall_table_csv=$GRAPH_OUT_DIR/${LABEL_PREFIX}_recall_table.csv"
fi
