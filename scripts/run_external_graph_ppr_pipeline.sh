#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/env/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

: "${DATA_NAME:?Set DATA_NAME, e.g. mmdocir, sciegqa, vidoseek, or vidore-v3}"
: "${DATA_ROOT:?Set DATA_ROOT to the converted dataset root}"
: "${DENSE_PRED:?Set DENSE_PRED to the dense/plain_top224 prediction JSON}"
: "${SPARSE_PRED:?Set SPARSE_PRED to the SPLADE prediction JSON}"
: "${OUT_DIR:?Set OUT_DIR for graph-PPR outputs}"

SPLIT="${SPLIT:-dev}"
GOLD="${GOLD:-$DATA_ROOT/MMQA_${SPLIT}.jsonl}"
GRAPH_LABEL="${GRAPH_LABEL:-${DATA_NAME}_plain_top224_splade_graph_ppr_page1000_nodocseed}"
PRED_OUT="${PRED_OUT:-$OUT_DIR/${GRAPH_LABEL}.prediction.json}"
SUMMARY_OUT="${SUMMARY_OUT:-$OUT_DIR/${GRAPH_LABEL}.summary.json}"

DENSE_TOP_PAGES="${DENSE_TOP_PAGES:-1000}"
SPARSE_TOP_PAGES="${SPARSE_TOP_PAGES:-1000}"
FINAL_TOP_PAGES="${FINAL_TOP_PAGES:-1000}"
PER_DOC_PAGE_LIMIT="${PER_DOC_PAGE_LIMIT:-0}"
RRF_K="${RRF_K:-10}"
DENSE_WEIGHT="${DENSE_WEIGHT:-1.0}"
SPARSE_WEIGHT="${SPARSE_WEIGHT:-1.0}"
DOC_SEED_WEIGHT="${DOC_SEED_WEIGHT:-0.0}"
RESTART_PROB="${RESTART_PROB:-0.20}"
PPR_ITERS="${PPR_ITERS:-30}"
PAGE_DOC_EDGE_WEIGHT="${PAGE_DOC_EDGE_WEIGHT:-1.0}"
ADJACENT_PAGE_EDGE_WEIGHT="${ADJACENT_PAGE_EDGE_WEIGHT:-0.25}"
SAME_DOC_WINDOW="${SAME_DOC_WINDOW:-1}"
FINAL_PAGE_SEED_WEIGHT="${FINAL_PAGE_SEED_WEIGHT:-1.0}"
FINAL_PPR_PAGE_WEIGHT="${FINAL_PPR_PAGE_WEIGHT:-1.5}"
FINAL_PPR_DOC_WEIGHT="${FINAL_PPR_DOC_WEIGHT:-0.5}"

SPLADE_INDEX_PT="${SPLADE_INDEX_PT:-}"
EXPANSION_TOP_PAGES="${EXPANSION_TOP_PAGES:-0}"
EXPAND_FROM_TOP_DENSE_PAGES="${EXPAND_FROM_TOP_DENSE_PAGES:-50}"
EXPAND_FROM_TOP_SPARSE_PAGES="${EXPAND_FROM_TOP_SPARSE_PAGES:-50}"
EXPANSION_WEIGHT="${EXPANSION_WEIGHT:-1.0}"
EXPANSION_SOURCE_TERM_TOPK="${EXPANSION_SOURCE_TERM_TOPK:-32}"
EXPANSION_QUERY_TOPK_TERMS="${EXPANSION_QUERY_TOPK_TERMS:-256}"
EXPANSION_MIN_SCORE="${EXPANSION_MIN_SCORE:-0.0}"
SCORE_SEED_WEIGHT="${SCORE_SEED_WEIGHT:-0.0}"
QUESTION_TYPE="${QUESTION_TYPE:-}"
RECALL_K_VALUES="${RECALL_K_VALUES:-1 2 4 5 10 20 50 100}"

mkdir -p "$OUT_DIR"

GRAPH_ARGS=(
  --dense-prediction-json "$DENSE_PRED"
  --sparse-prediction-json "$SPARSE_PRED"
  --gold "$GOLD"
  --dense-top-pages "$DENSE_TOP_PAGES"
  --sparse-top-pages "$SPARSE_TOP_PAGES"
  --final-top-pages "$FINAL_TOP_PAGES"
  --per-doc-page-limit "$PER_DOC_PAGE_LIMIT"
  --rrf-k "$RRF_K"
  --dense-weight "$DENSE_WEIGHT"
  --sparse-weight "$SPARSE_WEIGHT"
  --doc-seed-weight "$DOC_SEED_WEIGHT"
  --restart-prob "$RESTART_PROB"
  --ppr-iters "$PPR_ITERS"
  --page-doc-edge-weight "$PAGE_DOC_EDGE_WEIGHT"
  --adjacent-page-edge-weight "$ADJACENT_PAGE_EDGE_WEIGHT"
  --same-doc-window "$SAME_DOC_WINDOW"
  --final-page-seed-weight "$FINAL_PAGE_SEED_WEIGHT"
  --final-ppr-page-weight "$FINAL_PPR_PAGE_WEIGHT"
  --final-ppr-doc-weight "$FINAL_PPR_DOC_WEIGHT"
  --score-seed-weight "$SCORE_SEED_WEIGHT"
  --expansion-top-pages "$EXPANSION_TOP_PAGES"
  --expand-from-top-dense-pages "$EXPAND_FROM_TOP_DENSE_PAGES"
  --expand-from-top-sparse-pages "$EXPAND_FROM_TOP_SPARSE_PAGES"
  --expansion-weight "$EXPANSION_WEIGHT"
  --expansion-source-term-topk "$EXPANSION_SOURCE_TERM_TOPK"
  --expansion-query-topk-terms "$EXPANSION_QUERY_TOPK_TERMS"
  --expansion-min-score "$EXPANSION_MIN_SCORE"
  --output-prediction-json "$PRED_OUT"
  --output-summary-json "$SUMMARY_OUT"
)
if [[ -n "$QUESTION_TYPE" ]]; then
  GRAPH_ARGS+=(--question-type "$QUESTION_TYPE")
fi
if [[ -n "$SPLADE_INDEX_PT" ]]; then
  GRAPH_ARGS+=(--splade-index-pt "$SPLADE_INDEX_PT")
fi

"$PYTHON_BIN" "$REPO_ROOT/scripts/graph_rerank_page_retrieval_predictions.py" "${GRAPH_ARGS[@]}"

read -r -a RECALL_K_ARRAY <<< "$RECALL_K_VALUES"
"$PYTHON_BIN" "$REPO_ROOT/mmdocir/evaluate_mmdocir_retrieval.py" \
  --pred "$PRED_OUT" \
  --gold "$GOLD" \
  --recall-k "${RECALL_K_ARRAY[@]}"
