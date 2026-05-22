#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/env/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

OUT_DIR="${OUT_DIR:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_graph_pagepreserve_mmqa_dev}"
DATA_NAME="${DATA_NAME:-m3docvqa-mmqa}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data/m3-docvqa/multimodalqa}"
SPLIT="${SPLIT:-dev}"
GOLD="${GOLD:-$REPO_ROOT/data/m3-docvqa/multimodalqa/MMQA_dev.jsonl}"

DENSE_PRED="${DENSE_PRED:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/mmqa_dev_plain_top224_nprobe4_effdiag_all.prediction.json}"
SPARSE_PRED="${SPARSE_PRED:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_splade_mmqa_dev/mmqa_dev_splade.prediction.json}"
QUESTION_TYPE="${QUESTION_TYPE:-}"

GRAPH_LABEL="${GRAPH_LABEL:-mmqa_dev_plain_top224_splade_graph_pagepreserve_denseheavy_lightboth}"
PRED_OUT="${PRED_OUT:-$OUT_DIR/${GRAPH_LABEL}.prediction.json}"
SUMMARY_OUT="${SUMMARY_OUT:-$OUT_DIR/${GRAPH_LABEL}.summary.json}"
ANALYSIS_OUT="${ANALYSIS_OUT:-$OUT_DIR/${GRAPH_LABEL}.retrieval_analysis.json}"
VS_DENSE_OUT="${VS_DENSE_OUT:-$OUT_DIR/${GRAPH_LABEL}.vs_dense.json}"
VS_SPLADE_OUT="${VS_SPLADE_OUT:-$OUT_DIR/${GRAPH_LABEL}.vs_splade.json}"

RECALL_K_VALUES="${RECALL_K_VALUES:-1 2 4 5 10 20 50 100 500 1000}"

mkdir -p "$OUT_DIR"

# Best current page-preserving transfer config from the handoff:
# denseheavy_lightboth
GRAPH_ARGS=(
  --dense-prediction-json "$DENSE_PRED"
  --sparse-prediction-json "$SPARSE_PRED"
  --gold "$GOLD"
  --dense-top-pages 1000
  --sparse-top-pages 1000
  --final-top-pages 1000
  --per-doc-page-limit 0
  --rrf-k 10
  --dense-weight 1.25
  --sparse-weight 0.75
  --doc-seed-weight 0.0
  --restart-prob 0.15
  --ppr-iters 30
  --page-doc-edge-weight 1.0
  --same-doc-window 1
  --adjacent-page-edge-weight 0.25
  --final-page-seed-weight 1.0
  --final-ppr-page-weight 0.25
  --final-ppr-doc-weight 0.25
  --output-prediction-json "$PRED_OUT"
  --output-summary-json "$SUMMARY_OUT"
)
if [[ -n "$QUESTION_TYPE" ]]; then
  GRAPH_ARGS+=(--question-type "$QUESTION_TYPE")
fi
"$PYTHON_BIN" "$REPO_ROOT/scripts/graph_rerank_page_retrieval_predictions.py" "${GRAPH_ARGS[@]}"

"$PYTHON_BIN" "$REPO_ROOT/scripts/analyze_m3docvqa_retrieval.py" \
  --pred "$PRED_OUT" \
  --gold "$GOLD" \
  --summary-only \
  --recall-k $RECALL_K_VALUES \
  --json > "$ANALYSIS_OUT"

"$PYTHON_BIN" "$REPO_ROOT/scripts/compare_m3docvqa_retrieval_runs.py" \
  --baseline "$DENSE_PRED" \
  --candidate "$PRED_OUT" \
  --gold "$GOLD" \
  --recall-k $RECALL_K_VALUES \
  --json > "$VS_DENSE_OUT"

"$PYTHON_BIN" "$REPO_ROOT/scripts/compare_m3docvqa_retrieval_runs.py" \
  --baseline "$SPARSE_PRED" \
  --candidate "$PRED_OUT" \
  --gold "$GOLD" \
  --recall-k $RECALL_K_VALUES \
  --json > "$VS_SPLADE_OUT"

echo "saved_prediction=$PRED_OUT"
echo "saved_summary=$SUMMARY_OUT"
echo "saved_retrieval_analysis=$ANALYSIS_OUT"
echo "saved_vs_dense=$VS_DENSE_OUT"
echo "saved_vs_splade=$VS_SPLADE_OUT"
