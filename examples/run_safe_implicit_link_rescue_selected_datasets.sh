#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

VITAL_PATHS_ENV="${VITAL_PATHS_ENV:-$REPO_ROOT/hpc_vital_paths.generated.env}"
if [[ -f "$VITAL_PATHS_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$VITAL_PATHS_ENV"
fi

DATASETS="${DATASETS:-dude mmdocir}"

SAFE_DOC_DOC_EDGE_WEIGHT="${SAFE_DOC_DOC_EDGE_WEIGHT:-0.05}"
SAFE_DOC_DOC_TOP_DOCS="${SAFE_DOC_DOC_TOP_DOCS:-20}"
SAFE_DOC_DOC_MAX_EDGES_PER_DOC="${SAFE_DOC_DOC_MAX_EDGES_PER_DOC:-3}"
SAFE_DOC_DOC_EMBEDDING_MUTUAL_TOP_K="${SAFE_DOC_DOC_EMBEDDING_MUTUAL_TOP_K:-3}"
SAFE_DOC_DOC_RESCUE_ANCHOR_TOP_K="${SAFE_DOC_DOC_RESCUE_ANCHOR_TOP_K:-4}"
SAFE_DOC_DOC_RESCUE_RANK_MIN="${SAFE_DOC_DOC_RESCUE_RANK_MIN:-5}"
SAFE_DOC_DOC_RESCUE_RANK_MAX="${SAFE_DOC_DOC_RESCUE_RANK_MAX:-20}"

SAFE_TOKEN_GRAPH_MAX_NEW_PAGES="${SAFE_TOKEN_GRAPH_MAX_NEW_PAGES:-20}"
SAFE_TOKEN_GRAPH_MIN_SCORE="${SAFE_TOKEN_GRAPH_MIN_SCORE:-0.50}"
SAFE_TOKEN_GRAPH_APPEND_AFTER_TOP_K="${SAFE_TOKEN_GRAPH_APPEND_AFTER_TOP_K:-20}"

SAFE_REPORT_DIR="${SAFE_REPORT_DIR:-$REPO_ROOT/output/safe_implicit_link_rescue_reports}"
RUN_SAFE_DOC_RESCUE="${RUN_SAFE_DOC_RESCUE:-1}"
RUN_SAFE_FAISS_EXPAND="${RUN_SAFE_FAISS_EXPAND:-1}"
RUN_SAFE_COMBINED="${RUN_SAFE_COMBINED:-1}"
mkdir -p "$SAFE_REPORT_DIR"

run_doc_rescue_original() {
  echo
  echo "== Safe implicit links: doc cosine rescue only =="
  DATASETS="$DATASETS" \
  REPORT_OUT="$SAFE_REPORT_DIR/doc_cosine_rescue_only.md" \
  REPORT_CSV_OUT="$SAFE_REPORT_DIR/doc_cosine_rescue_only.csv" \
  DOC_EMBED_OUTPUT_SUBDIR=implicit_link_safe_doc_rescue \
  DOC_EMBED_LABEL_SUFFIX=_safe_doc_rescue \
  DOC_EMBED_RUN_BASELINE=1 \
  DOC_EMBED_RUN_RAW_VARIANTS=0 \
  DOC_EMBED_RUN_DENSE_SPARSE_GATED=0 \
  DOC_EMBED_RUN_SEMANTIC_GATED=0 \
  DOC_EMBED_RUN_RESCUE_GATED=1 \
  DOC_DOC_EDGE_WEIGHT="$SAFE_DOC_DOC_EDGE_WEIGHT" \
  DOC_DOC_TOP_DOCS="$SAFE_DOC_DOC_TOP_DOCS" \
  DOC_DOC_MAX_EDGES_PER_DOC="$SAFE_DOC_DOC_MAX_EDGES_PER_DOC" \
  DOC_DOC_EMBEDDING_MUTUAL_TOP_K="$SAFE_DOC_DOC_EMBEDDING_MUTUAL_TOP_K" \
  DOC_DOC_RESCUE_ANCHOR_TOP_K="$SAFE_DOC_DOC_RESCUE_ANCHOR_TOP_K" \
  DOC_DOC_RESCUE_RANK_MIN="$SAFE_DOC_DOC_RESCUE_RANK_MIN" \
  DOC_DOC_RESCUE_RANK_MAX="$SAFE_DOC_DOC_RESCUE_RANK_MAX" \
  bash "$REPO_ROOT/examples/run_doc_embedding_cosine_ablation_selected_datasets.sh"
}

run_faiss_candidate_expansion() {
  echo
  echo "== Safe implicit links: FAISS token-neighbor candidate expansion only =="
  DATASETS="$DATASETS" \
  REPORT_OUT="$SAFE_REPORT_DIR/faiss_candidate_expansion_only.md" \
  REPORT_CSV_OUT="$SAFE_REPORT_DIR/faiss_candidate_expansion_only.csv" \
  TOKEN_GRAPH_OUTPUT_SUBDIR=implicit_link_safe_faiss_expand \
  TOKEN_GRAPH_GRAPH_SUBDIR=implicit_link_safe_faiss_token_graph \
  TOKEN_GRAPH_LABEL_SUFFIX=_safe_expand \
  TOKEN_GRAPH_RUN_EDGE_VARIANTS=0 \
  TOKEN_GRAPH_RUN_CANDIDATE_EXPANSION=1 \
  TOKEN_GRAPH_CANDIDATE_EXPANSION_MAX_NEW_PAGES="$SAFE_TOKEN_GRAPH_MAX_NEW_PAGES" \
  TOKEN_GRAPH_CANDIDATE_EXPANSION_MIN_SCORE="$SAFE_TOKEN_GRAPH_MIN_SCORE" \
  TOKEN_GRAPH_CANDIDATE_EXPANSION_APPEND_AFTER_TOP_K="$SAFE_TOKEN_GRAPH_APPEND_AFTER_TOP_K" \
  bash "$REPO_ROOT/examples/run_faiss_token_neighbor_page_graph_selected_datasets.sh"
}

expanded_dense_path_for_dataset() {
  local dataset="$1"
  local max_new_pages="$SAFE_TOKEN_GRAPH_MAX_NEW_PAGES"
  case "$dataset" in
    mmdocir)
      printf '%s\n' "$MMDocIR_WORK_ROOT/output/mmdocir/implicit_link_safe_faiss_token_graph/mmdocir_faiss_token_neighbor_safe_expand_faiss_token_neighbor_expanded_dense_top${max_new_pages}.prediction.json"
      ;;
    sciegqa)
      printf '%s\n' "$SciEGQA_WORK_ROOT/output/sciegqa/implicit_link_safe_faiss_token_graph/sciegqa_faiss_token_neighbor_safe_expand_faiss_token_neighbor_expanded_dense_top${max_new_pages}.prediction.json"
      ;;
    vidoseek)
      printf '%s\n' "$VIDOSEEK_WORK_ROOT/output/vidoseek/implicit_link_safe_faiss_token_graph/vidoseek_faiss_token_neighbor_safe_expand_faiss_token_neighbor_expanded_dense_top${max_new_pages}.prediction.json"
      ;;
    dude)
      printf '%s\n' "$DUDE_WORK_ROOT/output/dude/implicit_link_safe_faiss_token_graph/dude_faiss_token_neighbor_safe_expand_faiss_token_neighbor_expanded_dense_top${max_new_pages}.prediction.json"
      ;;
    vidore|vidore-v3)
      printf '%s\n' "$VIDORE_WORK_ROOT/output/vidore-v3/implicit_link_safe_faiss_token_graph/vidore_faiss_token_neighbor_safe_expand_faiss_token_neighbor_expanded_dense_top${max_new_pages}.prediction.json"
      ;;
    opendocvqa)
      printf '%s\n' "$OPENDOCVQA_WORK_ROOT/output/opendocvqa/implicit_link_safe_faiss_token_graph/opendocvqa_faiss_token_neighbor_safe_expand_faiss_token_neighbor_expanded_dense_top${max_new_pages}.prediction.json"
      ;;
    mmlongbench|mmlongbench-docqa)
      printf '%s\n' "$MMLONGBENCH_WORK_ROOT/output/mmlongbench-docqa/implicit_link_safe_faiss_token_graph/mmlongbench_faiss_token_neighbor_safe_expand_faiss_token_neighbor_expanded_dense_top${max_new_pages}.prediction.json"
      ;;
    *)
      echo "unknown_dataset: $dataset" >&2
      exit 1
      ;;
  esac
}

set_expanded_dense_overrides() {
  local dataset
  local expanded
  for dataset in $DATASETS; do
    expanded="$(expanded_dense_path_for_dataset "$dataset")"
    if [[ ! -f "$expanded" ]]; then
      echo "missing_expanded_dense_prediction_for_${dataset}: $expanded" >&2
      exit 1
    fi
    case "$dataset" in
      mmdocir) export MMDOCIR_DENSE_PRED="$expanded" ;;
      sciegqa) export SCIEGQA_DENSE_PRED="$expanded" ;;
      vidoseek) export VIDOSEEK_DENSE_PRED="$expanded" ;;
      dude) export DUDE_DENSE_PRED="$expanded" ;;
      vidore|vidore-v3) export VIDORE_DENSE_PRED="$expanded" ;;
      opendocvqa) export OPENDOCVQA_DENSE_PRED="$expanded" ;;
      mmlongbench|mmlongbench-docqa) export MMLONGBENCH_DENSE_PRED="$expanded" ;;
    esac
  done
}

run_combined_safe() {
  echo
  echo "== Safe implicit links: FAISS expansion + doc cosine rescue =="
  set_expanded_dense_overrides
  DATASETS="$DATASETS" \
  REPORT_OUT="$SAFE_REPORT_DIR/combined_faiss_expand_doc_rescue.md" \
  REPORT_CSV_OUT="$SAFE_REPORT_DIR/combined_faiss_expand_doc_rescue.csv" \
  DOC_EMBED_OUTPUT_SUBDIR=implicit_link_safe_combined \
  DOC_EMBED_LABEL_SUFFIX=_safe_combined \
  DOC_EMBED_RUN_BASELINE=1 \
  DOC_EMBED_RUN_RAW_VARIANTS=0 \
  DOC_EMBED_RUN_DENSE_SPARSE_GATED=0 \
  DOC_EMBED_RUN_SEMANTIC_GATED=0 \
  DOC_EMBED_RUN_RESCUE_GATED=1 \
  DOC_DOC_EDGE_WEIGHT="$SAFE_DOC_DOC_EDGE_WEIGHT" \
  DOC_DOC_TOP_DOCS="$SAFE_DOC_DOC_TOP_DOCS" \
  DOC_DOC_MAX_EDGES_PER_DOC="$SAFE_DOC_DOC_MAX_EDGES_PER_DOC" \
  DOC_DOC_EMBEDDING_MUTUAL_TOP_K="$SAFE_DOC_DOC_EMBEDDING_MUTUAL_TOP_K" \
  DOC_DOC_RESCUE_ANCHOR_TOP_K="$SAFE_DOC_DOC_RESCUE_ANCHOR_TOP_K" \
  DOC_DOC_RESCUE_RANK_MIN="$SAFE_DOC_DOC_RESCUE_RANK_MIN" \
  DOC_DOC_RESCUE_RANK_MAX="$SAFE_DOC_DOC_RESCUE_RANK_MAX" \
  bash "$REPO_ROOT/examples/run_doc_embedding_cosine_ablation_selected_datasets.sh"
}

if [[ "$RUN_SAFE_DOC_RESCUE" == "1" ]]; then
  run_doc_rescue_original
fi
if [[ "$RUN_SAFE_FAISS_EXPAND" == "1" ]]; then
  run_faiss_candidate_expansion
fi
if [[ "$RUN_SAFE_COMBINED" == "1" ]]; then
  run_combined_safe
fi

echo "saved_safe_report_dir=$SAFE_REPORT_DIR"
