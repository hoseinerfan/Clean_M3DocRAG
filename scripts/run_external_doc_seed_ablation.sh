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
: "${OUT_DIR:?Set OUT_DIR for doc-seed ablation outputs}"

SPLIT="${SPLIT:-dev}"
GOLD="${GOLD:-$DATA_ROOT/MMQA_${SPLIT}.jsonl}"
DOC_PAGES_JSONL="${DOC_PAGES_JSONL:-$DATA_ROOT/doc_pages_${SPLIT}.jsonl}"
LABEL_PREFIX="${LABEL_PREFIX:-${DATA_NAME}_pagepreserve_docseed_ablation}"
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
DOC_SEED_GRAPH_SIZE_REFERENCE="${DOC_SEED_GRAPH_SIZE_REFERENCE:-20.0}"
DOC_SEED_GRAPH_SIZE_MIN_MULT="${DOC_SEED_GRAPH_SIZE_MIN_MULT:-0.25}"
DOC_SEED_GRAPH_SIZE_MAX_MULT="${DOC_SEED_GRAPH_SIZE_MAX_MULT:-2.0}"

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
  local seed_mode="$2"
  local seed_weight="$3"
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
  DOC_SEED_WEIGHT="$seed_weight" \
  DOC_SEED_MODE="$seed_mode" \
  DOC_SEED_GRAPH_SIZE_REFERENCE="$DOC_SEED_GRAPH_SIZE_REFERENCE" \
  DOC_SEED_GRAPH_SIZE_MIN_MULT="$DOC_SEED_GRAPH_SIZE_MIN_MULT" \
  DOC_SEED_GRAPH_SIZE_MAX_MULT="$DOC_SEED_GRAPH_SIZE_MAX_MULT" \
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
  PDF_HYPERLINK_EDGES_JSONL= \
  PDF_HYPERLINK_EDGE_WEIGHT=0.0 \
  EXTERNAL_PAGE_GRAPH_JSONL= \
  EXTERNAL_PAGE_GRAPH_EDGE_WEIGHT=0.0 \
  DOC_DOC_EDGE_MODE=none \
  DOC_DOC_EDGE_WEIGHT=0.0 \
  EXPANSION_TOP_PAGES=0 \
  NEIGHBOR_EXPANSION_WINDOW=0 \
  RECALL_K_VALUES="$RECALL_K_VALUES" \
  bash "$REPO_ROOT/scripts/run_external_graph_ppr_pipeline.sh"
}

run_variant docseed_none rrf 0.0
run_variant docseed_rrf_0p25 rrf 0.25
run_variant docseed_rrf_0p50 rrf 0.50
run_variant docseed_rrf_1p00 rrf 1.00
run_variant docseed_graphsize_0p50 graph_size_adaptive 0.50
run_variant docseed_avgpage_0p50 avg_page_seed 0.50
run_variant docseed_avgpage_graphsize_0p50 avg_page_seed_graph_size 0.50

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
