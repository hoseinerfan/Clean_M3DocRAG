#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

if [[ -f "$SCRIPT_DIR/env_hpc.sh" ]]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/env_hpc.sh"
fi

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/env/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

DATA_NAME="${DATA_NAME:-mpdocvqa}"
DATA_ROOT="${MPDOCVQA_DATA_ROOT:-$LOCAL_DATA_DIR/mpdocvqa}"
TOP_PAGES="${MPDOCVQA_TOP_PAGES:-1000}"
OUT_ROOT="${MPDOCVQA_OUT_ROOT:-$LOCAL_OUTPUT_DIR/mpdocvqa}"

EXACT_DENSE_PRED="${EXACT_DENSE_PRED:-$OUT_ROOT/exact_maxsim_ret${TOP_PAGES}_prediction.json}"
SPLADE_OUT_DIR="${SPLADE_OUT_DIR:-$OUT_ROOT/doc_rrf_exact_maxsim_splade}"
GRAPH_OUT_DIR="${GRAPH_OUT_DIR:-$OUT_ROOT/graph_ppr_exact_maxsim_splade}"
SPLADE_PRED="${SPLADE_PRED:-$SPLADE_OUT_DIR/${DATA_NAME}_splade_ret${TOP_PAGES}.prediction.json}"
SPLADE_INDEX_PT="${SPLADE_INDEX_PT:-$SPLADE_OUT_DIR/${DATA_NAME}_splade_page_index.pt}"
GRAPH_LABEL="${GRAPH_LABEL:-mpdocvqa_exactmaxsim_splade_gpp}"
GRAPH_PRED="$GRAPH_OUT_DIR/${GRAPH_LABEL}.prediction.json"

RUN_EXACT_MAXSIM="${RUN_EXACT_MAXSIM:-1}"
RUN_SPLADE="${RUN_SPLADE:-1}"
RUN_GPP="${RUN_GPP:-1}"
FORCE_SPLADE="${FORCE_SPLADE:-0}"

mkdir -p "$OUT_ROOT" "$SPLADE_OUT_DIR" "$GRAPH_OUT_DIR"

if [[ "$RUN_EXACT_MAXSIM" == "1" ]]; then
  MPDOCVQA_DATA_ROOT="$DATA_ROOT" \
  MPDOCVQA_OUT_DIR="$OUT_ROOT" \
  MPDOCVQA_TOP_PAGES="$TOP_PAGES" \
  bash "$REPO_ROOT/mpdocvqa/run_exact_maxsim_mpdocvqa.sh"
fi

if [[ ! -f "$EXACT_DENSE_PRED" ]]; then
  echo "missing_exact_dense_prediction: $EXACT_DENSE_PRED" >&2
  echo "hint: run bash mpdocvqa/run_exact_maxsim_mpdocvqa.sh first" >&2
  exit 1
fi

if [[ "$RUN_SPLADE" == "1" ]]; then
  if [[ "$FORCE_SPLADE" != "1" && -f "$SPLADE_PRED" && -f "$SPLADE_INDEX_PT" ]]; then
    export SKIP_EXPORT=1
    export SKIP_BUILD=1
    export SKIP_SPLADE_RETRIEVAL=1
    echo "reusing_existing_splade_prediction=$SPLADE_PRED"
    echo "reusing_existing_splade_index=$SPLADE_INDEX_PT"
  fi

  DATA_NAME="$DATA_NAME" \
  DATA_ROOT="$DATA_ROOT" \
  DENSE_PRED="$EXACT_DENSE_PRED" \
  OUT_DIR="$SPLADE_OUT_DIR" \
  TOP_PAGES="$TOP_PAGES" \
  DENSE_WEIGHT="${DENSE_WEIGHT:-1.25}" \
  SPARSE_WEIGHT="${SPARSE_WEIGHT:-0.75}" \
  RRF_K="${RRF_K:-10}" \
  SPLADE_DEVICE="${SPLADE_DEVICE:-cuda}" \
  SPLADE_INDEX_PT="$SPLADE_INDEX_PT" \
  SPLADE_PRED="$SPLADE_PRED" \
  bash "$REPO_ROOT/scripts/run_external_doc_rrf_pipeline.sh"
fi

if [[ "$RUN_GPP" == "1" ]]; then
  if [[ ! -f "$SPLADE_PRED" ]]; then
    echo "missing_splade_prediction: $SPLADE_PRED" >&2
    echo "hint: run this script with RUN_SPLADE=1 first" >&2
    exit 1
  fi

  DATA_NAME="$DATA_NAME" \
  DATA_ROOT="$DATA_ROOT" \
  DENSE_PRED="$EXACT_DENSE_PRED" \
  SPARSE_PRED="$SPLADE_PRED" \
  OUT_DIR="$GRAPH_OUT_DIR" \
  GRAPH_PROFILE="${GRAPH_PROFILE:-page_rank_probe}" \
  GRAPH_LABEL="$GRAPH_LABEL" \
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
  SPLADE_INDEX_PT="$SPLADE_INDEX_PT" \
  RECALL_K_VALUES="${RECALL_K_VALUES:-1 2 4 5 10 20 50 100 1000}" \
  bash "$REPO_ROOT/scripts/run_external_graph_ppr_pipeline.sh"
fi

echo "exact_dense_prediction=$EXACT_DENSE_PRED"
echo "splade_prediction=$SPLADE_PRED"
echo "gpp_prediction=$GRAPH_PRED"
