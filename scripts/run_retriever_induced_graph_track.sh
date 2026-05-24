#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/env/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

: "${DATA_NAME:?Set DATA_NAME, e.g. vidore-v3, mmdocir, or opendocvqa}"
: "${DATA_ROOT:?Set DATA_ROOT to the converted dataset root}"
: "${DENSE_PRED:?Set DENSE_PRED to the dense/plain_top224 prediction JSON}"
: "${SPARSE_PRED:?Set SPARSE_PRED to the SPLADE prediction JSON}"
: "${OUT_DIR:?Set OUT_DIR for graph-PPR outputs}"

SPLIT="${SPLIT:-dev}"
SUBSET_LABEL="${SUBSET_LABEL:-dev}"
GOLD="${GOLD:-${SUBSET_GOLD:-$DATA_ROOT/MMQA_${SPLIT}.jsonl}}"
BASELINE_PRED="${BASELINE_PRED:-${BASE_PRED:-$DENSE_PRED}}"

SPARSE_PRED_DIR="$(cd "$(dirname "$SPARSE_PRED")" && pwd)"
SPLADE_INDEX_PT="${SPLADE_INDEX_PT:-$SPARSE_PRED_DIR/${DATA_NAME}_splade_page_index.pt}"
if [[ ! -f "$SPLADE_INDEX_PT" ]]; then
  echo "SPLADE index not found: $SPLADE_INDEX_PT" >&2
  echo "Set SPLADE_INDEX_PT explicitly, or run scripts/run_external_doc_rrf_pipeline.sh first." >&2
  exit 1
fi
if [[ ! -f "$GOLD" ]]; then
  echo "Gold JSONL not found: $GOLD" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

BASE_GRAPH_LABEL="${BASE_GRAPH_LABEL:-${DATA_NAME}_${SUBSET_LABEL}_graph_ppr_base}"
BASE_GRAPH_PRED="${BASE_GRAPH_PRED:-$OUT_DIR/${BASE_GRAPH_LABEL}.prediction.json}"
BASE_GRAPH_SUMMARY="${BASE_GRAPH_SUMMARY:-$OUT_DIR/${BASE_GRAPH_LABEL}.summary.json}"

SPLADE_KNN_TOP_K="${SPLADE_KNN_TOP_K:-8}"
SPLADE_KNN_SOURCE_TOP_PAGES="${SPLADE_KNN_SOURCE_TOP_PAGES:-1000}"
SPLADE_KNN_SOURCE_TOPK_TERMS="${SPLADE_KNN_SOURCE_TOPK_TERMS:-64}"
SPLADE_KNN_MIN_SCORE="${SPLADE_KNN_MIN_SCORE:-0.0}"
SPLADE_KNN_SCORE_MODE="${SPLADE_KNN_SCORE_MODE:-cosine}"
SPLADE_KNN_LABEL="${SPLADE_KNN_LABEL:-${DATA_NAME}_${SUBSET_LABEL}_splade_knn_top${SPLADE_KNN_TOP_K}}"
SPLADE_KNN_EDGES_JSONL="${SPLADE_KNN_EDGES_JSONL:-$OUT_DIR/${SPLADE_KNN_LABEL}.edges.jsonl}"
SPLADE_KNN_SUMMARY_JSON="${SPLADE_KNN_SUMMARY_JSON:-$OUT_DIR/${SPLADE_KNN_LABEL}.summary.json}"

SPLADE_GRAPH_LABEL="${SPLADE_GRAPH_LABEL:-${DATA_NAME}_${SUBSET_LABEL}_graph_ppr_splade_knn}"
SPLADE_GRAPH_PRED="${SPLADE_GRAPH_PRED:-$OUT_DIR/${SPLADE_GRAPH_LABEL}.prediction.json}"
SPLADE_GRAPH_SUMMARY="${SPLADE_GRAPH_SUMMARY:-$OUT_DIR/${SPLADE_GRAPH_LABEL}.summary.json}"

FUSION_LABEL="${FUSION_LABEL:-${DATA_NAME}_${SUBSET_LABEL}_graph_view_rrf}"
FUSION_PRED="${FUSION_PRED:-$OUT_DIR/${FUSION_LABEL}.prediction.json}"
FUSION_SUMMARY="${FUSION_SUMMARY:-$OUT_DIR/${FUSION_LABEL}.summary.json}"

COMMON_GRAPH_ENV=(
  GRAPH_PROFILE="${GRAPH_PROFILE:-page_rank_probe}"
  FINAL_TOP_PAGES="${FINAL_TOP_PAGES:-1000}"
  PER_DOC_PAGE_LIMIT="${PER_DOC_PAGE_LIMIT:-0}"
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
  QUERY_ANCHOR_EVIDENCE_MODE=none
  QUERY_ANCHOR_REASONING_MODE=none
  CONSTRAINT_COMPETITION_MODE=none
  CONSTRAINT_COMPETITION_STRENGTH=0.0
  EXPANSION_TOP_PAGES=0
)

if [[ "${OVERWRITE_BASE_GRAPH:-0}" == "1" || ! -f "$BASE_GRAPH_PRED" ]]; then
  env \
    GOLD="$GOLD" \
    GRAPH_LABEL="$BASE_GRAPH_LABEL" \
    PRED_OUT="$BASE_GRAPH_PRED" \
    SUMMARY_OUT="$BASE_GRAPH_SUMMARY" \
    "${COMMON_GRAPH_ENV[@]}" \
    bash "$REPO_ROOT/scripts/run_external_graph_ppr_pipeline.sh"
else
  echo "using_existing_base_graph_prediction: $BASE_GRAPH_PRED"
fi

if [[ "${OVERWRITE_KNN:-0}" == "1" || ! -f "$SPLADE_KNN_EDGES_JSONL" ]]; then
  "$PYTHON_BIN" "$REPO_ROOT/scripts/build_splade_page_knn_graph.py" \
    --splade-index-pt "$SPLADE_INDEX_PT" \
    --source-prediction-json "$DENSE_PRED" \
    --source-prediction-json "$SPARSE_PRED" \
    --qid-filter-jsonl "$GOLD" \
    --source-top-pages "$SPLADE_KNN_SOURCE_TOP_PAGES" \
    --top-k "$SPLADE_KNN_TOP_K" \
    --source-topk-terms "$SPLADE_KNN_SOURCE_TOPK_TERMS" \
    --min-score "$SPLADE_KNN_MIN_SCORE" \
    --score-mode "$SPLADE_KNN_SCORE_MODE" \
    --output-jsonl "$SPLADE_KNN_EDGES_JSONL" \
    --output-summary-json "$SPLADE_KNN_SUMMARY_JSON"
else
  echo "using_existing_splade_knn_edges: $SPLADE_KNN_EDGES_JSONL"
fi

if [[ "${OVERWRITE_SPLADE_GRAPH:-0}" == "1" || ! -f "$SPLADE_GRAPH_PRED" ]]; then
  env \
    GOLD="$GOLD" \
    GRAPH_LABEL="$SPLADE_GRAPH_LABEL" \
    PRED_OUT="$SPLADE_GRAPH_PRED" \
    SUMMARY_OUT="$SPLADE_GRAPH_SUMMARY" \
    EXTERNAL_PAGE_GRAPH_JSONL="$SPLADE_KNN_EDGES_JSONL" \
    EXTERNAL_PAGE_GRAPH_EDGE_WEIGHT="${EXTERNAL_PAGE_GRAPH_EDGE_WEIGHT:-0.25}" \
    EXTERNAL_PAGE_GRAPH_DIRECTION="${EXTERNAL_PAGE_GRAPH_DIRECTION:-bidirectional}" \
    EXTERNAL_PAGE_GRAPH_WEIGHT_MODE=score \
    EXTERNAL_PAGE_GRAPH_MAX_EDGES_PER_SOURCE="${EXTERNAL_PAGE_GRAPH_MAX_EDGES_PER_SOURCE:-8}" \
    EXTERNAL_PAGE_GRAPH_SOURCE_TOP_K="${EXTERNAL_PAGE_GRAPH_SOURCE_TOP_K:-1000}" \
    EXTERNAL_PAGE_GRAPH_TARGET_TOP_K="${EXTERNAL_PAGE_GRAPH_TARGET_TOP_K:-1000}" \
    "${COMMON_GRAPH_ENV[@]}" \
    bash "$REPO_ROOT/scripts/run_external_graph_ppr_pipeline.sh"
else
  echo "using_existing_splade_graph_prediction: $SPLADE_GRAPH_PRED"
fi

if [[ "${OVERWRITE_FUSION:-0}" == "1" || ! -f "$FUSION_PRED" ]]; then
  "$PYTHON_BIN" "$REPO_ROOT/scripts/fuse_graph_view_predictions.py" \
    --prediction "base_graph=$BASE_GRAPH_PRED" \
    --prediction "splade_knn_graph=$SPLADE_GRAPH_PRED" \
    --gold "$GOLD" \
    --baseline "$BASELINE_PRED" \
    --rrf-k "${GRAPH_VIEW_RRF_K:-60}" \
    --input-top-pages "${GRAPH_VIEW_RRF_INPUT_TOP_PAGES:-1000}" \
    --output-top-pages "${GRAPH_VIEW_RRF_OUTPUT_TOP_PAGES:-1000}" \
    --hit-k "${GRAPH_VIEW_RRF_HIT_K:-4}" \
    --output-prediction-json "$FUSION_PRED" \
    --output-summary-json "$FUSION_SUMMARY"
else
  echo "using_existing_fusion_prediction: $FUSION_PRED"
fi

if [[ "${RUN_CASE_COMPARE:-1}" == "1" && -n "$BASELINE_PRED" && -f "$BASELINE_PRED" ]]; then
  for label_pred in \
    "$BASE_GRAPH_LABEL=$BASE_GRAPH_PRED" \
    "$SPLADE_GRAPH_LABEL=$SPLADE_GRAPH_PRED" \
    "$FUSION_LABEL=$FUSION_PRED"
  do
    label="${label_pred%%=*}"
    pred="${label_pred#*=}"
    "$PYTHON_BIN" "$REPO_ROOT/scripts/compare_page_retrieval_cases.py" \
      --gold "$GOLD" \
      --baseline "$BASELINE_PRED" \
      --candidate "$pred" \
      --topk "${COMPARE_TOPK:-4}" \
      --topn "${COMPARE_TOPN:-30}" \
      --output-json "$OUT_DIR/${label}_top${COMPARE_TOPK:-4}_cases.json"
  done
fi

echo "base_graph_prediction: $BASE_GRAPH_PRED"
echo "splade_knn_graph_prediction: $SPLADE_GRAPH_PRED"
echo "fusion_prediction: $FUSION_PRED"
