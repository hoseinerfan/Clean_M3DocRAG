#!/usr/bin/env bash
#SBATCH --job-name=mmlongbench-splade-ppr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --output=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/MMLongBench_M3DocRAG/logs/splade_graph_ppr_%j.out
#SBATCH --error=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/MMLongBench_M3DocRAG/logs/splade_graph_ppr_%j.err

set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG}"
cd "$REPO_ROOT"

source mmlongbench/env_hpc.sh

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/env/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

DATA_NAME="${DATA_NAME:-mmlongbench-docqa}"
DATA_ROOT="${DATA_ROOT:-$LOCAL_DATA_DIR/mmlongbench-docqa}"
TOP_PAGES="${TOP_PAGES:-1000}"
DENSE_PRED="${DENSE_PRED:-$LOCAL_OUTPUT_DIR/mmlongbench-docqa/plain_top224_ret${TOP_PAGES}_prediction.json}"
SPLADE_OUT_DIR="${SPLADE_OUT_DIR:-$LOCAL_OUTPUT_DIR/mmlongbench-docqa/doc_rrf_plain_top224_splade}"
GRAPH_OUT_DIR="${GRAPH_OUT_DIR:-$LOCAL_OUTPUT_DIR/mmlongbench-docqa/graph_ppr_plain_top224_splade}"
SPLADE_PRED="${SPLADE_PRED:-$SPLADE_OUT_DIR/${DATA_NAME}_splade_ret${TOP_PAGES}.prediction.json}"
SPLADE_INDEX_PT="${SPLADE_INDEX_PT:-$SPLADE_OUT_DIR/${DATA_NAME}_splade_page_index.pt}"

if [[ ! -f "$DENSE_PRED" ]]; then
  echo "Missing dense/plain_top224 prediction: $DENSE_PRED" >&2
  echo "Run: sbatch mmlongbench/sbatch_plain_top224_mmlongbench.sh" >&2
  exit 1
fi

mkdir -p "$SPLADE_OUT_DIR" "$GRAPH_OUT_DIR"

if [[ "${FORCE_SPLADE:-0}" != "1" && -f "$SPLADE_PRED" && -f "$SPLADE_INDEX_PT" ]]; then
  export SKIP_EXPORT=1
  export SKIP_BUILD=1
  export SKIP_SPLADE_RETRIEVAL=1
  echo "reusing_existing_splade_prediction=$SPLADE_PRED"
  echo "reusing_existing_splade_index=$SPLADE_INDEX_PT"
fi

DATA_NAME="$DATA_NAME" \
DATA_ROOT="$DATA_ROOT" \
DENSE_PRED="$DENSE_PRED" \
OUT_DIR="$SPLADE_OUT_DIR" \
TOP_PAGES="$TOP_PAGES" \
DENSE_WEIGHT="${DENSE_WEIGHT:-1.25}" \
SPARSE_WEIGHT="${SPARSE_WEIGHT:-0.75}" \
RRF_K="${RRF_K:-10}" \
SPLADE_DEVICE="${SPLADE_DEVICE:-cuda}" \
SPLADE_INDEX_PT="$SPLADE_INDEX_PT" \
SPLADE_PRED="$SPLADE_PRED" \
bash scripts/run_external_doc_rrf_pipeline.sh

DATA_NAME="$DATA_NAME" \
DATA_ROOT="$DATA_ROOT" \
DENSE_PRED="$DENSE_PRED" \
SPARSE_PRED="$SPLADE_PRED" \
OUT_DIR="$GRAPH_OUT_DIR" \
GRAPH_PROFILE="${GRAPH_PROFILE:-page_rank_probe}" \
GRAPH_LABEL="${GRAPH_LABEL:-mmlongbench_docqa_denseheavy125_medium_both}" \
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
bash scripts/run_external_graph_ppr_pipeline.sh

echo "saved_splade_prediction=$SPLADE_PRED"
echo "saved_graph_prediction=$GRAPH_OUT_DIR/${GRAPH_LABEL:-mmlongbench_docqa_denseheavy125_medium_both}.prediction.json"
