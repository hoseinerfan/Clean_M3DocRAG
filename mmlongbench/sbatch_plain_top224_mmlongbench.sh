#!/usr/bin/env bash
#SBATCH --job-name=mmlongbench-plain224
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/MMLongBench_M3DocRAG/logs/plain_top224_%j.out
#SBATCH --error=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/MMLongBench_M3DocRAG/logs/plain_top224_%j.err

set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG}"
cd "$REPO_ROOT"

source mmlongbench/env_hpc.sh

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/env/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

DATA_ROOT="${DATA_ROOT:-$LOCAL_DATA_DIR/mmlongbench-docqa}"
OUT_DIR="${OUT_DIR:-$LOCAL_OUTPUT_DIR/mmlongbench-docqa}"
TOP_PAGES="${MMLONGBENCH_TOP_PAGES:-1000}"
BASELINE_PRED="${MMLONGBENCH_BASELINE_PRED:-$OUT_DIR/baseline_ret1000.json}"
PLAIN_PRED="${MMLONGBENCH_PLAIN_PRED:-$OUT_DIR/plain_top224_ret${TOP_PAGES}_prediction.json}"

export MMLONGBENCH_DATA_ROOT="$DATA_ROOT"
export MMLONGBENCH_BASELINE_PRED="$BASELINE_PRED"
export MMLONGBENCH_OUT_DIR="$OUT_DIR"
export MMLONGBENCH_TOP_PAGES="$TOP_PAGES"
export BASE_ONLY_PAGE_BATCH_SIZE="${BASE_ONLY_PAGE_BATCH_SIZE:-64}"

bash mmlongbench/run_plain_top224_mmlongbench.sh

"$PYTHON_BIN" mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$BASELINE_PRED" \
  --gold "$DATA_ROOT/MMQA_dev.jsonl" \
  --recall-k 1 2 4 5 10 20 50 100

"$PYTHON_BIN" mmdocir/evaluate_mmdocir_retrieval.py \
  --pred "$PLAIN_PRED" \
  --gold "$DATA_ROOT/MMQA_dev.jsonl" \
  --recall-k 1 2 4 5 10 20 50 100

echo "saved_plain_top224_prediction=$PLAIN_PRED"
