#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/env/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

: "${DATA_NAME:?Set DATA_NAME, e.g. vidore-v3 or mmdocir}"
: "${DATA_ROOT:?Set DATA_ROOT to the converted dataset root}"
: "${OUT_DIR:?Set OUT_DIR for outputs}"

SPLIT="${SPLIT:-dev}"
SUBSET_LABEL="${SUBSET_LABEL:-dev}"
GOLD="${GOLD:-${SUBSET_GOLD:-$DATA_ROOT/MMQA_${SPLIT}.jsonl}}"
DOC_PAGES_JSONL="${DOC_PAGES_JSONL:-$DATA_ROOT/doc_pages_${SPLIT}.jsonl}"

DEFAULT_BASE_GRAPH="$OUT_DIR/${DATA_NAME}_${SUBSET_LABEL}_graph_ppr_base.prediction.json"
if [[ -z "${PREDICTION:-}" ]]; then
  if [[ -f "$DEFAULT_BASE_GRAPH" ]]; then
    PREDICTION="$DEFAULT_BASE_GRAPH"
  elif [[ -n "${BASE_PRED:-}" && -f "$BASE_PRED" ]]; then
    PREDICTION="$BASE_PRED"
  elif [[ -n "${DENSE_PRED:-}" && -f "$DENSE_PRED" ]]; then
    PREDICTION="$DENSE_PRED"
  else
    echo "Set PREDICTION, BASE_PRED, or DENSE_PRED to an existing prediction JSON." >&2
    exit 1
  fi
fi

if [[ ! -f "$PREDICTION" ]]; then
  echo "Prediction JSON not found: $PREDICTION" >&2
  exit 1
fi
if [[ ! -f "$DOC_PAGES_JSONL" ]]; then
  echo "doc_pages JSONL not found: $DOC_PAGES_JSONL" >&2
  exit 1
fi
if [[ -n "$GOLD" && ! -f "$GOLD" ]]; then
  echo "Gold JSONL not found: $GOLD" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

LAYOUT_LABEL="${LAYOUT_LABEL:-${DATA_NAME}_${SUBSET_LABEL}_layout_evidence_graph}"
LAYOUT_PRED="${LAYOUT_PRED:-$OUT_DIR/${LAYOUT_LABEL}.prediction.json}"
LAYOUT_SUMMARY="${LAYOUT_SUMMARY:-$OUT_DIR/${LAYOUT_LABEL}.summary.json}"
LAYOUT_CASES="${LAYOUT_CASES:-$OUT_DIR/${LAYOUT_LABEL}_cases.json}"

if [[ "${OVERWRITE_LAYOUT_EVIDENCE:-0}" == "1" || ! -f "$LAYOUT_PRED" ]]; then
  LAYOUT_ARGS=(
    --prediction "$PREDICTION"
    --doc-pages-jsonl "$DOC_PAGES_JSONL"
    --gold "$GOLD"
    --candidate-scope "${LAYOUT_CANDIDATE_SCOPE:-top_docs_prediction_pages}"
    --top-docs "${LAYOUT_TOP_DOCS:-4}"
    --input-top-pages "${LAYOUT_INPUT_TOP_PAGES:-1000}"
    --candidate-top-pages "${LAYOUT_CANDIDATE_TOP_PAGES:-1000}"
    --output-top-pages "${LAYOUT_OUTPUT_TOP_PAGES:-1000}"
    --max-pages-per-doc "${LAYOUT_MAX_PAGES_PER_DOC:-250}"
    --max-regions-per-page "${LAYOUT_MAX_REGIONS_PER_PAGE:-32}"
    --max-region-tokens "${LAYOUT_MAX_REGION_TOKENS:-96}"
    --ppr-restart-prob "${LAYOUT_PPR_RESTART_PROB:-0.30}"
    --ppr-iters "${LAYOUT_PPR_ITERS:-20}"
    --region-adjacent-edge-weight "${LAYOUT_REGION_ADJACENT_EDGE_WEIGHT:-0.15}"
    --page-restart-weight "${LAYOUT_PAGE_RESTART_WEIGHT:-0.0}"
    --rrf-k "${LAYOUT_RRF_K:-60}"
    --hit-k "${LAYOUT_HIT_K:-4}"
    --output-prediction-json "$LAYOUT_PRED"
    --output-summary-json "$LAYOUT_SUMMARY"
    --output-case-json "$LAYOUT_CASES"
  )
  if [[ "${LAYOUT_DISABLE_FALLBACK_REGIONS:-0}" == "1" ]]; then
    LAYOUT_ARGS+=(--disable-fallback-regions)
  fi
  if [[ -n "${REGION_JSONL:-}" ]]; then
    IFS=: read -r -a REGION_JSONL_PARTS <<< "$REGION_JSONL"
    for region_jsonl in "${REGION_JSONL_PARTS[@]}"; do
      if [[ -n "$region_jsonl" ]]; then
        LAYOUT_ARGS+=(--region-jsonl "$region_jsonl")
      fi
    done
  fi
  "$PYTHON_BIN" "$REPO_ROOT/scripts/rerank_layout_evidence_graph.py" "${LAYOUT_ARGS[@]}"
else
  echo "using_existing_layout_evidence_prediction: $LAYOUT_PRED"
fi

if [[ "${RUN_CASE_COMPARE:-1}" == "1" ]]; then
  "$PYTHON_BIN" "$REPO_ROOT/scripts/compare_page_retrieval_cases.py" \
    --gold "$GOLD" \
    --baseline "$PREDICTION" \
    --candidate "$LAYOUT_PRED" \
    --topk "${COMPARE_TOPK:-4}" \
    --topn "${COMPARE_TOPN:-30}" \
    --output-json "$OUT_DIR/${LAYOUT_LABEL}_top${COMPARE_TOPK:-4}_cases.json"
fi

echo "layout_evidence_prediction: $LAYOUT_PRED"
