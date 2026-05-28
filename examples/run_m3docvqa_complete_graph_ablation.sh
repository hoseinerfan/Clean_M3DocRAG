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
GRAPH_OUT_DIR="${GRAPH_OUT_DIR:-$LOCAL_OUTPUT_DIR/m3docvqa_complete_graph_ablation}"
LABEL_PREFIX="${LABEL_PREFIX:-mmqa_${SPLIT}_complete_graph_ablation}"
GRAPH_PROFILE="${GRAPH_PROFILE:-denseheavy125_medium_both}"
RECALL_K_VALUES="${RECALL_K_VALUES:-1 2 4 5 10 20 50 100 500 1000}"

DOC_DOC_EDGE_WEIGHT="${DOC_DOC_EDGE_WEIGHT:-0.10}"
DOC_DOC_TOP_DOCS="${DOC_DOC_TOP_DOCS:-20}"
DOC_DOC_MAX_EDGES_PER_DOC="${DOC_DOC_MAX_EDGES_PER_DOC:-8}"
DOC_DOC_MIN_SHARED_SIGNALS="${DOC_DOC_MIN_SHARED_SIGNALS:-1}"
DOC_DOC_MAX_SIGNAL_DOC_MATCHES="${DOC_DOC_MAX_SIGNAL_DOC_MATCHES:-8}"
DOC_DOC_MIN_SEMANTIC_SIMILARITY="${DOC_DOC_MIN_SEMANTIC_SIMILARITY:-0.35}"
DOC_DOC_SEMANTIC_TOP_TERMS="${DOC_DOC_SEMANTIC_TOP_TERMS:-64}"

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

RUN_DOC_DOC="${RUN_DOC_DOC:-1}"
RUN_DOC_SEED="${RUN_DOC_SEED:-1}"
RUN_CROSS_DOC_SELECT="${RUN_CROSS_DOC_SELECT:-1}"
RUN_DENSE_SPARSE_AUDIT="${RUN_DENSE_SPARSE_AUDIT:-1}"
RUN_FEATURE_AUDIT="${RUN_FEATURE_AUDIT:-1}"
RUN_HYPERLINK_AUDIT="${RUN_HYPERLINK_AUDIT:-1}"

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
  local max_edges_per_doc="$4"
  local seed_mode="$5"
  local seed_weight="$6"
  local selection_mode="$7"
  local selection_pool="$8"
  local selection_bonus="$9"
  local selection_penalty="${10}"
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
  DOC_DOC_MAX_EDGES_PER_DOC="$max_edges_per_doc" \
  DOC_DOC_MIN_SHARED_SIGNALS="$DOC_DOC_MIN_SHARED_SIGNALS" \
  DOC_DOC_MAX_SIGNAL_DOC_MATCHES="$DOC_DOC_MAX_SIGNAL_DOC_MATCHES" \
  DOC_DOC_MIN_SEMANTIC_SIMILARITY="$DOC_DOC_MIN_SEMANTIC_SIMILARITY" \
  DOC_DOC_SEMANTIC_TOP_TERMS="$DOC_DOC_SEMANTIC_TOP_TERMS" \
  DOC_SEED_MODE="$seed_mode" \
  DOC_SEED_WEIGHT="$seed_weight" \
  FINAL_SELECTION_MODE="$selection_mode" \
  FINAL_SELECTION_TOP_K=4 \
  FINAL_SELECTION_CANDIDATE_POOL="$selection_pool" \
  FINAL_SELECTION_NEW_DOC_BONUS="$selection_bonus" \
  FINAL_SELECTION_SAME_DOC_PENALTY="$selection_penalty" \
  PDF_HYPERLINK_EDGES_JSONL="$PDF_HYPERLINK_EDGES_JSONL" \
  bash "$REPO_ROOT/scripts/run_m3docvqa_page_preserving_graph_pipeline.sh"
}

if [[ "$RUN_DOC_DOC" == "1" ]]; then
  run_variant docdoc_no_doc_doc none 0.0 "$DOC_DOC_MAX_EDGES_PER_DOC" rrf 0.0 score 20 0.05 0.05
  run_variant docdoc_dense_sparse_agreement dense_sparse_agreement "$DOC_DOC_EDGE_WEIGHT" "$DOC_DOC_MAX_EDGES_PER_DOC" rrf 0.0 score 20 0.05 0.05
  run_variant docdoc_fully_connected_topdocs fully_connected "$DOC_DOC_EDGE_WEIGHT" 0 rrf 0.0 score 20 0.05 0.05
  run_variant docdoc_shared_entity_title_topic shared_entity_title_topic "$DOC_DOC_EDGE_WEIGHT" "$DOC_DOC_MAX_EDGES_PER_DOC" rrf 0.0 score 20 0.05 0.05
  if [[ -n "$SPLADE_INDEX_PT" && -f "$SPLADE_INDEX_PT" ]]; then
    run_variant docdoc_semantic_similarity semantic_similarity "$DOC_DOC_EDGE_WEIGHT" "$DOC_DOC_MAX_EDGES_PER_DOC" rrf 0.0 score 20 0.05 0.05
  else
    echo "skip_docdoc_semantic_similarity_missing_splade_index: $SPLADE_INDEX_PT" >&2
  fi
  if [[ -n "$PDF_HYPERLINK_EDGES_JSONL" && -f "$PDF_HYPERLINK_EDGES_JSONL" ]]; then
    echo "using_pdf_hyperlink_edges=$PDF_HYPERLINK_EDGES_JSONL"
    run_variant docdoc_hyperlink_citation hyperlink_citation "$DOC_DOC_EDGE_WEIGHT" "$DOC_DOC_MAX_EDGES_PER_DOC" rrf 0.0 score 20 0.05 0.05
  else
    echo "skip_docdoc_hyperlink_citation_missing_pdf_hyperlink_edges: set M3DOCVQA_PDF_HYPERLINK_EDGES_JSONL" >&2
  fi
  run_variant docdoc_all_doc_doc_features all "$DOC_DOC_EDGE_WEIGHT" "$DOC_DOC_MAX_EDGES_PER_DOC" rrf 0.0 score 20 0.05 0.05
fi

if [[ "$RUN_DOC_SEED" == "1" ]]; then
  run_variant docseed_none none 0.0 "$DOC_DOC_MAX_EDGES_PER_DOC" rrf 0.0 score 20 0.05 0.05
  run_variant docseed_rrf_0p25 none 0.0 "$DOC_DOC_MAX_EDGES_PER_DOC" rrf 0.25 score 20 0.05 0.05
  run_variant docseed_rrf_0p50 none 0.0 "$DOC_DOC_MAX_EDGES_PER_DOC" rrf 0.50 score 20 0.05 0.05
  run_variant docseed_rrf_1p00 none 0.0 "$DOC_DOC_MAX_EDGES_PER_DOC" rrf 1.00 score 20 0.05 0.05
  run_variant docseed_graphsize_0p50 none 0.0 "$DOC_DOC_MAX_EDGES_PER_DOC" graph_size_adaptive 0.50 score 20 0.05 0.05
  run_variant docseed_avgpage_0p50 none 0.0 "$DOC_DOC_MAX_EDGES_PER_DOC" avg_page_seed 0.50 score 20 0.05 0.05
  run_variant docseed_avgpage_graphsize_0p50 none 0.0 "$DOC_DOC_MAX_EDGES_PER_DOC" avg_page_seed_graph_size 0.50 score 20 0.05 0.05
fi

if [[ "$RUN_CROSS_DOC_SELECT" == "1" ]]; then
  run_variant select_score_baseline none 0.0 "$DOC_DOC_MAX_EDGES_PER_DOC" rrf 0.0 score 20 0.05 0.05
  run_variant select_max1doc_pool20 none 0.0 "$DOC_DOC_MAX_EDGES_PER_DOC" rrf 0.0 max1_per_doc_then_fill 20 0.05 0.05
  run_variant select_max1doc_pool50 none 0.0 "$DOC_DOC_MAX_EDGES_PER_DOC" rrf 0.0 max1_per_doc_then_fill 50 0.05 0.05
  run_variant select_mmr_docdiv_pool20_b0p02 none 0.0 "$DOC_DOC_MAX_EDGES_PER_DOC" rrf 0.0 mmr_doc_diverse 20 0.02 0.02
  run_variant select_mmr_docdiv_pool20_b0p05 none 0.0 "$DOC_DOC_MAX_EDGES_PER_DOC" rrf 0.0 mmr_doc_diverse 20 0.05 0.05
  run_variant select_mmr_docdiv_pool20_b0p10 none 0.0 "$DOC_DOC_MAX_EDGES_PER_DOC" rrf 0.0 mmr_doc_diverse 20 0.10 0.10
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

  cp "$GRAPH_OUT_DIR/${LABEL_PREFIX}_recall_table.md" "$REPO_ROOT/m3docvqa_complete_graph_ablation_results.md"
  cp "$GRAPH_OUT_DIR/${LABEL_PREFIX}_recall_table.csv" "$REPO_ROOT/m3docvqa_complete_graph_ablation_results.csv"
  echo "saved_recall_table_md=$GRAPH_OUT_DIR/${LABEL_PREFIX}_recall_table.md"
  echo "saved_recall_table_csv=$GRAPH_OUT_DIR/${LABEL_PREFIX}_recall_table.csv"
  echo "saved_report_md=$REPO_ROOT/m3docvqa_complete_graph_ablation_results.md"
  echo "saved_report_csv=$REPO_ROOT/m3docvqa_complete_graph_ablation_results.csv"
fi

if [[ "$RUN_HYPERLINK_AUDIT" == "1" && -n "$PDF_HYPERLINK_EDGES_JSONL" && -f "$PDF_HYPERLINK_EDGES_JSONL" ]]; then
  base_pred="$GRAPH_OUT_DIR/${LABEL_PREFIX}_docdoc_no_doc_doc.prediction.json"
  link_pred="$GRAPH_OUT_DIR/${LABEL_PREFIX}_docdoc_hyperlink_citation.prediction.json"
  if [[ -f "$base_pred" && -f "$link_pred" ]]; then
    "$PYTHON_BIN" "$REPO_ROOT/scripts/audit_m3docvqa_hyperlink_effects.py" \
      --baseline "$base_pred" \
      --candidate "$link_pred" \
      --gold "$GOLD" \
      --hyperlink-edges-jsonl "$PDF_HYPERLINK_EDGES_JSONL" \
      --source-top-pages 1000 \
      --hit-k 4 \
      --output-json "$GRAPH_OUT_DIR/${LABEL_PREFIX}_docdoc_hyperlink_citation.audit.json" \
      --output-md "$GRAPH_OUT_DIR/${LABEL_PREFIX}_docdoc_hyperlink_citation.audit.md"
  fi
fi

if [[ "$RUN_DENSE_SPARSE_AUDIT" == "1" ]]; then
  dense_sparse_summary="$GRAPH_OUT_DIR/${LABEL_PREFIX}_docdoc_dense_sparse_agreement.summary.json"
  if [[ -f "$dense_sparse_summary" ]]; then
    "$PYTHON_BIN" "$REPO_ROOT/scripts/audit_dense_sparse_agreement_ablation.py" \
      --strict \
      "$dense_sparse_summary"
  fi
fi

if [[ "$RUN_FEATURE_AUDIT" == "1" ]]; then
  for feature in shared_entity_title_topic semantic_similarity; do
    summary_path="$GRAPH_OUT_DIR/${LABEL_PREFIX}_docdoc_${feature}.summary.json"
    if [[ -f "$summary_path" ]]; then
      "$PYTHON_BIN" "$REPO_ROOT/scripts/audit_doc_doc_feature_ablation.py" \
        --strict \
        --feature "$feature" \
        "$summary_path"
    fi
  done
fi
