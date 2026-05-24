#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/env/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

: "${DATA_NAME:?Set DATA_NAME, e.g. vidore-v3}"
: "${DATA_ROOT:?Set DATA_ROOT to the converted dataset root}"
: "${DENSE_PRED:?Set DENSE_PRED to the dense/plain_top224 prediction JSON}"
: "${SPARSE_PRED:?Set SPARSE_PRED to the SPLADE prediction JSON}"
: "${OUT_DIR:?Set OUT_DIR for graph-PPR outputs}"

SPLIT="${SPLIT:-dev}"
SUBSET_LABEL="${SUBSET_LABEL:-subset100}"
GOLD="${GOLD:-${SUBSET_GOLD:-$DATA_ROOT/MMQA_${SPLIT}.jsonl}}"
BASELINE_PRED="${BASELINE_PRED:-${BASE_PRED:-}}"
REFERENCE_PRED="${REFERENCE_PRED:-$OUT_DIR/${DATA_NAME}_${SUBSET_LABEL}_cb_comp_s100.prediction.json}"
PRIMARY_CANDIDATE="${PRIMARY_CANDIDATE:-$OUT_DIR/${DATA_NAME}_${SUBSET_LABEL}_qlsoft_bnd_c5_w6_f025_s100_m100.prediction.json}"

SPARSE_PRED_DIR="$(cd "$(dirname "$SPARSE_PRED")" && pwd)"
SPLADE_INDEX_PT="${SPLADE_INDEX_PT:-$SPARSE_PRED_DIR/${DATA_NAME}_splade_page_index.pt}"
if [[ ! -f "$SPLADE_INDEX_PT" ]]; then
  echo "SPLADE index not found: $SPLADE_INDEX_PT" >&2
  echo "Set SPLADE_INDEX_PT explicitly, or run scripts/run_external_doc_rrf_pipeline.sh to build it." >&2
  exit 1
fi

SPLADE_KNN_TOP_K="${SPLADE_KNN_TOP_K:-8}"
SPLADE_KNN_SOURCE_TOP_PAGES="${SPLADE_KNN_SOURCE_TOP_PAGES:-1000}"
SPLADE_KNN_SOURCE_TOPK_TERMS="${SPLADE_KNN_SOURCE_TOPK_TERMS:-64}"
SPLADE_KNN_MIN_SCORE="${SPLADE_KNN_MIN_SCORE:-0.0}"
SPLADE_KNN_SCORE_MODE="${SPLADE_KNN_SCORE_MODE:-cosine}"
SPLADE_KNN_LABEL="${SPLADE_KNN_LABEL:-${DATA_NAME}_${SUBSET_LABEL}_splade_knn_top${SPLADE_KNN_TOP_K}}"
SPLADE_KNN_EDGES_JSONL="${SPLADE_KNN_EDGES_JSONL:-$OUT_DIR/${SPLADE_KNN_LABEL}.edges.jsonl}"
SPLADE_KNN_SUMMARY_JSON="${SPLADE_KNN_SUMMARY_JSON:-$OUT_DIR/${SPLADE_KNN_LABEL}.summary.json}"

SUPPORT_LABEL="${SUPPORT_LABEL:-${DATA_NAME}_${SUBSET_LABEL}_support_splade_knn}"
SUPPORT_PRED="${SUPPORT_PRED:-$OUT_DIR/${SUPPORT_LABEL}.prediction.json}"
SUPPORT_SUMMARY="${SUPPORT_SUMMARY:-$OUT_DIR/${SUPPORT_LABEL}.summary.json}"

SELECTOR_LABEL="${SELECTOR_LABEL:-${DATA_NAME}_${SUBSET_LABEL}_selector_splade_knn_consensus}"
SELECTOR_PRED="${SELECTOR_PRED:-$OUT_DIR/${SELECTOR_LABEL}.prediction.json}"
SELECTOR_SUMMARY="${SELECTOR_SUMMARY:-$OUT_DIR/${SELECTOR_LABEL}.summary.json}"

if [[ ! -f "$GOLD" ]]; then
  echo "Gold JSONL not found: $GOLD" >&2
  exit 1
fi
if [[ ! -f "$REFERENCE_PRED" ]]; then
  echo "Reference prediction not found: $REFERENCE_PRED" >&2
  exit 1
fi
if [[ ! -f "$PRIMARY_CANDIDATE" ]]; then
  echo "Primary candidate prediction not found: $PRIMARY_CANDIDATE" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

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

if [[ "${OVERWRITE_SUPPORT:-0}" == "1" || ! -f "$SUPPORT_PRED" ]]; then
  GOLD="$GOLD" \
  GRAPH_PROFILE="${GRAPH_PROFILE:-page_rank_probe}" \
  GRAPH_LABEL="$SUPPORT_LABEL" \
  PRED_OUT="$SUPPORT_PRED" \
  SUMMARY_OUT="$SUPPORT_SUMMARY" \
  FINAL_TOP_PAGES="${FINAL_TOP_PAGES:-1000}" \
  PER_DOC_PAGE_LIMIT="${PER_DOC_PAGE_LIMIT:-0}" \
  DENSE_WEIGHT="${DENSE_WEIGHT:-1.25}" \
  SPARSE_WEIGHT="${SPARSE_WEIGHT:-0.75}" \
  RESTART_PROB="${RESTART_PROB:-0.15}" \
  PPR_ITERS="${PPR_ITERS:-30}" \
  PAGE_DOC_EDGE_WEIGHT="${PAGE_DOC_EDGE_WEIGHT:-1.0}" \
  SAME_DOC_WINDOW="${SAME_DOC_WINDOW:-1}" \
  ADJACENT_PAGE_EDGE_WEIGHT="${ADJACENT_PAGE_EDGE_WEIGHT:-0.25}" \
  FINAL_PAGE_SEED_WEIGHT="${FINAL_PAGE_SEED_WEIGHT:-1.0}" \
  FINAL_PPR_PAGE_WEIGHT="${FINAL_PPR_PAGE_WEIGHT:-0.5}" \
  FINAL_PPR_DOC_WEIGHT="${FINAL_PPR_DOC_WEIGHT:-0.25}" \
  QUERY_ANCHOR_EVIDENCE_MODE=none \
  QUERY_ANCHOR_REASONING_MODE=none \
  CONSTRAINT_COMPETITION_MODE=none \
  CONSTRAINT_COMPETITION_STRENGTH=0.0 \
  EXPANSION_TOP_PAGES=0 \
  EXTERNAL_PAGE_GRAPH_JSONL="$SPLADE_KNN_EDGES_JSONL" \
  EXTERNAL_PAGE_GRAPH_EDGE_WEIGHT="${EXTERNAL_PAGE_GRAPH_EDGE_WEIGHT:-0.25}" \
  EXTERNAL_PAGE_GRAPH_DIRECTION="${EXTERNAL_PAGE_GRAPH_DIRECTION:-bidirectional}" \
  EXTERNAL_PAGE_GRAPH_WEIGHT_MODE=score \
  EXTERNAL_PAGE_GRAPH_MAX_EDGES_PER_SOURCE="${EXTERNAL_PAGE_GRAPH_MAX_EDGES_PER_SOURCE:-8}" \
  EXTERNAL_PAGE_GRAPH_SOURCE_TOP_K="${EXTERNAL_PAGE_GRAPH_SOURCE_TOP_K:-1000}" \
  EXTERNAL_PAGE_GRAPH_TARGET_TOP_K="${EXTERNAL_PAGE_GRAPH_TARGET_TOP_K:-1000}" \
  bash "$REPO_ROOT/scripts/run_external_graph_ppr_pipeline.sh"
else
  echo "using_existing_support_prediction: $SUPPORT_PRED"
fi

SELECTOR_ARGS=(
  --reference "$REFERENCE_PRED"
  --primary-candidate "$PRIMARY_CANDIDATE"
  --support-prediction "splade_knn=$SUPPORT_PRED"
  --exclude-primary-from-support
  --min-variant-count 1
  --topk "${SELECTOR_TOPK:-4}"
  --support-topk "${SELECTOR_SUPPORT_TOPK:-4}"
  --gold "$GOLD"
  --output-prediction-json "$SELECTOR_PRED"
  --output-summary-json "$SELECTOR_SUMMARY"
)
if [[ -n "$BASELINE_PRED" ]]; then
  SELECTOR_ARGS+=(--baseline "$BASELINE_PRED")
fi

"$PYTHON_BIN" "$REPO_ROOT/scripts/select_page_predictions_by_stability.py" "${SELECTOR_ARGS[@]}"
