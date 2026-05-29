#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/env/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

: "${DATA_NAME:?Set DATA_NAME, e.g. mmdocir, sciegqa, vidoseek, vidore-v3, or dude}"
: "${DATA_ROOT:?Set DATA_ROOT to the converted dataset root}"
: "${DENSE_PRED:?Set DENSE_PRED to the dense/plain_top224 prediction JSON}"
: "${SPARSE_PRED:?Set SPARSE_PRED to the SPLADE prediction JSON}"
: "${OUT_DIR:?Set OUT_DIR for doc-doc edge ablation outputs}"

SPLIT="${SPLIT:-dev}"
GOLD="${GOLD:-$DATA_ROOT/MMQA_${SPLIT}.jsonl}"
DOC_PAGES_JSONL="${DOC_PAGES_JSONL:-$DATA_ROOT/doc_pages_${SPLIT}.jsonl}"
LABEL_PREFIX="${LABEL_PREFIX:-${DATA_NAME}_pagepreserve_docdoc_ablation}"
RECALL_K_VALUES="${RECALL_K_VALUES:-1 2 4 5 10 20 50 100}"

DENSE_WEIGHT="${DENSE_WEIGHT:-1.25}"
SPARSE_WEIGHT="${SPARSE_WEIGHT:-0.75}"
RESTART_PROB="${RESTART_PROB:-0.15}"
PPR_ITERS="${PPR_ITERS:-30}"
PAGE_DOC_EDGE_WEIGHT="${PAGE_DOC_EDGE_WEIGHT:-1.0}"
SAME_DOC_WINDOW="${SAME_DOC_WINDOW:-1}"
ADJACENT_PAGE_EDGE_WEIGHT="${ADJACENT_PAGE_EDGE_WEIGHT:-0.25}"
FINAL_PAGE_SEED_WEIGHT="${FINAL_PAGE_SEED_WEIGHT:-1.0}"
FINAL_PPR_PAGE_WEIGHT="${FINAL_PPR_PAGE_WEIGHT:-0.5}"
FINAL_PPR_DOC_WEIGHT="${FINAL_PPR_DOC_WEIGHT:-0.25}"
DOC_DOC_EDGE_WEIGHT="${DOC_DOC_EDGE_WEIGHT:-0.10}"
DOC_DOC_TOP_DOCS="${DOC_DOC_TOP_DOCS:-20}"
DOC_DOC_MAX_EDGES_PER_DOC="${DOC_DOC_MAX_EDGES_PER_DOC:-8}"
DOC_DOC_MIN_SHARED_SIGNALS="${DOC_DOC_MIN_SHARED_SIGNALS:-1}"
DOC_DOC_MAX_SIGNAL_DOC_MATCHES="${DOC_DOC_MAX_SIGNAL_DOC_MATCHES:-8}"
DOC_DOC_MIN_SEMANTIC_SIMILARITY="${DOC_DOC_MIN_SEMANTIC_SIMILARITY:-0.35}"
DOC_DOC_SEMANTIC_TOP_TERMS="${DOC_DOC_SEMANTIC_TOP_TERMS:-64}"
DOC_DOC_PAGE_EMBEDDING_DIR="${DOC_DOC_PAGE_EMBEDDING_DIR:-}"
DOC_DOC_EMBEDDING_POOLING="${DOC_DOC_EMBEDDING_POOLING:-page_seed_weighted_mean}"
DOC_DOC_EMBEDDING_PAGE_TOP_K="${DOC_DOC_EMBEDDING_PAGE_TOP_K:-0}"
DOC_DOC_EMBEDDING_MIN_SIMILARITY="${DOC_DOC_EMBEDDING_MIN_SIMILARITY:-0.0}"
DOC_DOC_EMBEDDING_CACHE_DOCS="${DOC_DOC_EMBEDDING_CACHE_DOCS:-128}"
SPLADE_INDEX_PT="${SPLADE_INDEX_PT:-}"
PDF_HYPERLINK_EDGES_JSONL="${PDF_HYPERLINK_EDGES_JSONL:-}"

mkdir -p "$OUT_DIR"

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    echo "missing_${label}: $path" >&2
    exit 1
  fi
}

require_file gold "$GOLD"
require_file dense_pred "$DENSE_PRED"
require_file sparse_pred "$SPARSE_PRED"
require_file doc_pages_jsonl "$DOC_PAGES_JSONL"

run_variant() {
  local variant="$1"
  local edge_mode="$2"
  local edge_weight="$3"
  local max_edges_per_doc="${4:-$DOC_DOC_MAX_EDGES_PER_DOC}"
  local label="${LABEL_PREFIX}_${variant}"

  echo "== $DATA_NAME: $variant =="
  DATA_NAME="$DATA_NAME" \
  DATA_ROOT="$DATA_ROOT" \
  GOLD="$GOLD" \
  DOC_PAGES_JSONL="$DOC_PAGES_JSONL" \
  DENSE_PRED="$DENSE_PRED" \
  SPARSE_PRED="$SPARSE_PRED" \
  OUT_DIR="$OUT_DIR" \
  GRAPH_PROFILE=page_rank_probe \
  GRAPH_LABEL="$label" \
  FINAL_TOP_PAGES=1000 \
  PER_DOC_PAGE_LIMIT=0 \
  DENSE_WEIGHT="$DENSE_WEIGHT" \
  SPARSE_WEIGHT="$SPARSE_WEIGHT" \
  DOC_SEED_WEIGHT=0.0 \
  DOC_SEED_MODE=rrf \
  SCORE_SEED_WEIGHT=0.0 \
  RESTART_PROB="$RESTART_PROB" \
  PPR_ITERS="$PPR_ITERS" \
  PAGE_DOC_EDGE_WEIGHT="$PAGE_DOC_EDGE_WEIGHT" \
  PAGE_TO_DOC_EDGE_WEIGHT="$PAGE_DOC_EDGE_WEIGHT" \
  DOC_TO_PAGE_EDGE_WEIGHT="$PAGE_DOC_EDGE_WEIGHT" \
  SAME_DOC_WINDOW="$SAME_DOC_WINDOW" \
  ADJACENT_PAGE_EDGE_WEIGHT="$ADJACENT_PAGE_EDGE_WEIGHT" \
  FINAL_PAGE_SEED_WEIGHT="$FINAL_PAGE_SEED_WEIGHT" \
  FINAL_PPR_PAGE_WEIGHT="$FINAL_PPR_PAGE_WEIGHT" \
  FINAL_PPR_DOC_WEIGHT="$FINAL_PPR_DOC_WEIGHT" \
  ADAPTIVE_SOURCE_WEIGHT_MODE=none \
  ADAPTIVE_RESTART_MODE=none \
  ADAPTIVE_TRANSITION_MODE=none \
  ADAPTIVE_ADJACENT_MODE=none \
  EVIDENCE_COMMUNITY_MODE=none \
  POSITION_EVIDENCE_MODE=none \
  QUERY_ANCHOR_EVIDENCE_MODE=none \
  HEADING_BREADCRUMB_MODE=none \
  ENTITY_ALIAS_MODE=none \
  CONSTRAINT_COMPETITION_MODE=none \
  PDF_HYPERLINK_EDGES_JSONL="$PDF_HYPERLINK_EDGES_JSONL" \
  PDF_HYPERLINK_EDGE_WEIGHT=0.0 \
  EXTERNAL_PAGE_GRAPH_JSONL= \
  EXTERNAL_PAGE_GRAPH_EDGE_WEIGHT=0.0 \
  DOC_DOC_EDGE_MODE="$edge_mode" \
  DOC_DOC_EDGE_WEIGHT="$edge_weight" \
  DOC_DOC_TOP_DOCS="$DOC_DOC_TOP_DOCS" \
  DOC_DOC_MAX_EDGES_PER_DOC="$max_edges_per_doc" \
  DOC_DOC_MIN_SHARED_SIGNALS="$DOC_DOC_MIN_SHARED_SIGNALS" \
  DOC_DOC_MAX_SIGNAL_DOC_MATCHES="$DOC_DOC_MAX_SIGNAL_DOC_MATCHES" \
  DOC_DOC_MIN_SEMANTIC_SIMILARITY="$DOC_DOC_MIN_SEMANTIC_SIMILARITY" \
  DOC_DOC_SEMANTIC_TOP_TERMS="$DOC_DOC_SEMANTIC_TOP_TERMS" \
  DOC_DOC_PAGE_EMBEDDING_DIR="$DOC_DOC_PAGE_EMBEDDING_DIR" \
  DOC_DOC_EMBEDDING_POOLING="$DOC_DOC_EMBEDDING_POOLING" \
  DOC_DOC_EMBEDDING_PAGE_TOP_K="$DOC_DOC_EMBEDDING_PAGE_TOP_K" \
  DOC_DOC_EMBEDDING_MIN_SIMILARITY="$DOC_DOC_EMBEDDING_MIN_SIMILARITY" \
  DOC_DOC_EMBEDDING_CACHE_DOCS="$DOC_DOC_EMBEDDING_CACHE_DOCS" \
  SPLADE_INDEX_PT="$SPLADE_INDEX_PT" \
  EXPANSION_TOP_PAGES=0 \
  NEIGHBOR_EXPANSION_WINDOW=0 \
  RECALL_K_VALUES="$RECALL_K_VALUES" \
  bash "$REPO_ROOT/scripts/run_external_graph_ppr_pipeline.sh"
}

run_variant no_doc_doc none 0.0
run_variant dense_sparse_agreement dense_sparse_agreement "$DOC_DOC_EDGE_WEIGHT"
run_variant fully_connected_topdocs fully_connected "$DOC_DOC_EDGE_WEIGHT" 0
run_variant shared_entity_title_topic shared_entity_title_topic "$DOC_DOC_EDGE_WEIGHT"

if [[ -n "$SPLADE_INDEX_PT" && -f "$SPLADE_INDEX_PT" ]]; then
  run_variant semantic_similarity semantic_similarity "$DOC_DOC_EDGE_WEIGHT"
else
  echo "skip_semantic_similarity_missing_splade_index: $SPLADE_INDEX_PT" >&2
fi

if [[ -n "$DOC_DOC_PAGE_EMBEDDING_DIR" && -d "$DOC_DOC_PAGE_EMBEDDING_DIR" ]]; then
  run_variant page_embedding_cosine page_embedding_cosine "$DOC_DOC_EDGE_WEIGHT"
else
  echo "skip_page_embedding_cosine_missing_embedding_dir: $DOC_DOC_PAGE_EMBEDDING_DIR" >&2
fi

if [[ -n "$PDF_HYPERLINK_EDGES_JSONL" && -f "$PDF_HYPERLINK_EDGES_JSONL" ]]; then
  run_variant hyperlink_citation hyperlink_citation "$DOC_DOC_EDGE_WEIGHT"
else
  echo "skip_hyperlink_citation_missing_pdf_hyperlink_edges: $PDF_HYPERLINK_EDGES_JSONL" >&2
fi

run_variant all_doc_doc_features all "$DOC_DOC_EDGE_WEIGHT"

summary_paths=( "$OUT_DIR/${LABEL_PREFIX}_"*.summary.json )
if [[ -e "${summary_paths[0]}" ]]; then
  "$PYTHON_BIN" "$REPO_ROOT/scripts/summarize_graph_ppr_summaries.py" \
    --recall-table \
    --format markdown \
    "${summary_paths[@]}" \
    | tee "$OUT_DIR/${LABEL_PREFIX}_recall_table.md"

  "$PYTHON_BIN" "$REPO_ROOT/scripts/summarize_graph_ppr_summaries.py" \
    --recall-table \
    --format csv \
    "${summary_paths[@]}" \
    > "$OUT_DIR/${LABEL_PREFIX}_recall_table.csv"

  echo "saved_recall_table_md=$OUT_DIR/${LABEL_PREFIX}_recall_table.md"
  echo "saved_recall_table_csv=$OUT_DIR/${LABEL_PREFIX}_recall_table.csv"
fi
