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
: "${DENSE_PRED:?Set DENSE_PRED to the plain_top224 prediction JSON}"
: "${SPARSE_PRED:?Set SPARSE_PRED to the SPLADE prediction JSON}"
: "${OUT_DIR:?Set OUT_DIR for structural ablation outputs}"

SPLIT="${SPLIT:-dev}"
GOLD="${GOLD:-$DATA_ROOT/MMQA_${SPLIT}.jsonl}"
DOC_PAGES_JSONL="${DOC_PAGES_JSONL:-$DATA_ROOT/doc_pages_${SPLIT}.jsonl}"
LABEL_PREFIX="${LABEL_PREFIX:-${DATA_NAME}_graph_structure_ablation}"
RECALL_K_VALUES="${RECALL_K_VALUES:-1 2 4 5 10 20 50 100}"

# Keep the frozen page-labeled source fusion fixed while varying graph structure only.
DENSE_WEIGHT="${DENSE_WEIGHT:-1.25}"
SPARSE_WEIGHT="${SPARSE_WEIGHT:-0.75}"
RESTART_PROB="${RESTART_PROB:-0.15}"
FINAL_PAGE_SEED_WEIGHT="${FINAL_PAGE_SEED_WEIGHT:-1.0}"
FINAL_PPR_PAGE_WEIGHT="${FINAL_PPR_PAGE_WEIGHT:-0.5}"
FINAL_PPR_DOC_WEIGHT="${FINAL_PPR_DOC_WEIGHT:-0.25}"
PAGE_DOC_EDGE_WEIGHT="${PAGE_DOC_EDGE_WEIGHT:-1.0}"
SAME_DOC_WINDOW="${SAME_DOC_WINDOW:-1}"
ADJACENT_PAGE_EDGE_WEIGHT="${ADJACENT_PAGE_EDGE_WEIGHT:-0.25}"

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
  local ppr_iters="$2"
  local page_doc_weight="$3"
  local same_doc_window="$4"
  local adjacent_weight="$5"
  local ppr_page_weight="$6"
  local ppr_doc_weight="$7"
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
  SCORE_SEED_WEIGHT=0.0 \
  RESTART_PROB="$RESTART_PROB" \
  PPR_ITERS="$ppr_iters" \
  PAGE_DOC_EDGE_WEIGHT="$page_doc_weight" \
  PAGE_TO_DOC_EDGE_WEIGHT="$page_doc_weight" \
  DOC_TO_PAGE_EDGE_WEIGHT="$page_doc_weight" \
  SAME_DOC_WINDOW="$same_doc_window" \
  ADJACENT_PAGE_EDGE_WEIGHT="$adjacent_weight" \
  FINAL_PAGE_SEED_WEIGHT="$FINAL_PAGE_SEED_WEIGHT" \
  FINAL_PPR_PAGE_WEIGHT="$ppr_page_weight" \
  FINAL_PPR_DOC_WEIGHT="$ppr_doc_weight" \
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
  PDF_HYPERLINK_EDGES_JSONL= \
  PDF_HYPERLINK_EDGE_WEIGHT=0.0 \
  EXTERNAL_PAGE_GRAPH_JSONL= \
  EXTERNAL_PAGE_GRAPH_EDGE_WEIGHT=0.0 \
  EXPANSION_TOP_PAGES=0 \
  NEIGHBOR_EXPANSION_WINDOW=0 \
  RECALL_K_VALUES="$RECALL_K_VALUES" \
  bash "$REPO_ROOT/scripts/run_external_graph_ppr_pipeline.sh"
}

# No propagation: establishes the dense/SPLADE page-seed reference.
run_variant seed_only 0 0.0 0 0.0 0.0 0.0

# Page-local positional continuity only; no parent-document path exists.
run_variant adjacent_only 30 0.0 "$SAME_DOC_WINDOW" "$ADJACENT_PAGE_EDGE_WEIGHT" "$FINAL_PPR_PAGE_WEIGHT" 0.0

# Document paths can diffuse page mass, but are not scored explicitly at the output.
run_variant doc_edges_page_score_only 30 "$PAGE_DOC_EDGE_WEIGHT" 0 0.0 "$FINAL_PPR_PAGE_WEIGHT" 0.0

# Document propagation plus the explicit parent-document final score, without adjacency.
run_variant doc_prior_only 30 "$PAGE_DOC_EDGE_WEIGHT" 0 0.0 "$FINAL_PPR_PAGE_WEIGHT" "$FINAL_PPR_DOC_WEIGHT"

# Current transitions, without the explicit parent-document final-score component.
run_variant full_no_explicit_doc_score 30 "$PAGE_DOC_EDGE_WEIGHT" "$SAME_DOC_WINDOW" "$ADJACENT_PAGE_EDGE_WEIGHT" "$FINAL_PPR_PAGE_WEIGHT" 0.0

# Frozen current Graph-PPR backbone.
run_variant current_full_graph 30 "$PAGE_DOC_EDGE_WEIGHT" "$SAME_DOC_WINDOW" "$ADJACENT_PAGE_EDGE_WEIGHT" "$FINAL_PPR_PAGE_WEIGHT" "$FINAL_PPR_DOC_WEIGHT"

summary_paths=( "$OUT_DIR/${LABEL_PREFIX}_"*.summary.json )
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
