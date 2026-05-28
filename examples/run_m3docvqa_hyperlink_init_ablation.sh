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
GRAPH_OUT_DIR="${GRAPH_OUT_DIR:-$LOCAL_OUTPUT_DIR/m3docvqa_hyperlink_init_ablation}"
LABEL_PREFIX="${LABEL_PREFIX:-mmqa_${SPLIT}_hyperlink_init}"
GRAPH_PROFILE="${GRAPH_PROFILE:-denseheavy125_medium_both}"
DOC_DOC_EDGE_WEIGHT="${DOC_DOC_EDGE_WEIGHT:-2.25}"
DOC_DOC_TOP_DOCS="${DOC_DOC_TOP_DOCS:-20}"
DOC_DOC_MAX_EDGES_PER_DOC="${DOC_DOC_MAX_EDGES_PER_DOC:-8}"
DOC_DOC_HYPERLINK_WEIGHT_MODE="${DOC_DOC_HYPERLINK_WEIGHT_MODE:-log_count}"
FINAL_SELECTION_MODE="${FINAL_SELECTION_MODE:-mmr_doc_diverse}"
FINAL_SELECTION_TOP_K="${FINAL_SELECTION_TOP_K:-4}"
FINAL_SELECTION_CANDIDATE_POOL="${FINAL_SELECTION_CANDIDATE_POOL:-20}"
FINAL_SELECTION_NEW_DOC_BONUS="${FINAL_SELECTION_NEW_DOC_BONUS:-0.10}"
FINAL_SELECTION_SAME_DOC_PENALTY="${FINAL_SELECTION_SAME_DOC_PENALTY:-0.05}"
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
require_file pdf_hyperlink_edges "$PDF_HYPERLINK_EDGES_JSONL"
if [[ -n "$DOC_PAGES_JSONL" && ! -f "$DOC_PAGES_JSONL" ]]; then
  echo "warning_missing_doc_pages_jsonl: $DOC_PAGES_JSONL" >&2
fi

mkdir -p "$GRAPH_OUT_DIR"

run_variant() {
  local variant="$1"
  local edge_mode="$2"
  local edge_weight="$3"
  local init_mode="$4"
  local label="${LABEL_PREFIX}_${variant}"

  echo
  echo "== m3docvqa hyperlink init: $variant =="
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
  DOC_DOC_HYPERLINK_WEIGHT_MODE="$DOC_DOC_HYPERLINK_WEIGHT_MODE" \
  DOC_DOC_HYPERLINK_INIT_MODE="$init_mode" \
  PDF_HYPERLINK_EDGES_JSONL="$PDF_HYPERLINK_EDGES_JSONL" \
  FINAL_SELECTION_MODE="$FINAL_SELECTION_MODE" \
  FINAL_SELECTION_TOP_K="$FINAL_SELECTION_TOP_K" \
  FINAL_SELECTION_CANDIDATE_POOL="$FINAL_SELECTION_CANDIDATE_POOL" \
  FINAL_SELECTION_NEW_DOC_BONUS="$FINAL_SELECTION_NEW_DOC_BONUS" \
  FINAL_SELECTION_SAME_DOC_PENALTY="$FINAL_SELECTION_SAME_DOC_PENALTY" \
  bash "$REPO_ROOT/scripts/run_m3docvqa_page_preserving_graph_pipeline.sh"
}

run_variant no_doc_doc none 0.0 log_count
run_variant hyperlink_init_uniform hyperlink_citation "$DOC_DOC_EDGE_WEIGHT" uniform
run_variant hyperlink_init_sqrt_count hyperlink_citation "$DOC_DOC_EDGE_WEIGHT" sqrt_count
run_variant hyperlink_init_log_count hyperlink_citation "$DOC_DOC_EDGE_WEIGHT" log_count
run_variant hyperlink_init_raw_count hyperlink_citation "$DOC_DOC_EDGE_WEIGHT" raw_count

baseline_pred="$GRAPH_OUT_DIR/${LABEL_PREFIX}_no_doc_doc.prediction.json"
for variant in \
  hyperlink_init_uniform \
  hyperlink_init_sqrt_count \
  hyperlink_init_log_count \
  hyperlink_init_raw_count
do
  candidate_pred="$GRAPH_OUT_DIR/${LABEL_PREFIX}_${variant}.prediction.json"
  "$PYTHON_BIN" "$REPO_ROOT/scripts/compare_m3docvqa_retrieval_runs.py" \
    --baseline "$baseline_pred" \
    --candidate "$candidate_pred" \
    --gold "$GOLD" \
    --recall-k $RECALL_K_VALUES \
    --json > "$GRAPH_OUT_DIR/${LABEL_PREFIX}_${variant}.vs_no_doc_doc.json"

  "$PYTHON_BIN" "$REPO_ROOT/scripts/audit_m3docvqa_hyperlink_effects.py" \
    --baseline "$baseline_pred" \
    --candidate "$candidate_pred" \
    --gold "$GOLD" \
    --hyperlink-edges-jsonl "$PDF_HYPERLINK_EDGES_JSONL" \
    --source-top-pages 1000 \
    --hit-k 4 \
    --output-json "$GRAPH_OUT_DIR/${LABEL_PREFIX}_${variant}.audit.json" \
    --output-md "$GRAPH_OUT_DIR/${LABEL_PREFIX}_${variant}.audit.md"
done

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
