#!/usr/bin/env bash
#SBATCH --job-name=racs-train-holdout
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=output/racs_controlled_training_%j.out
#SBATCH --error=output/racs_controlled_training_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the Clean_M3DocRAG repository root}"
test -x env/bin/python
RACS_CODE_DIR="$PWD/scripts"
test -f "$RACS_CODE_DIR/run_racs_controlled_training.py"
test -f "$RACS_CODE_DIR/train_content_aware_pseudo_page_reranker.py"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=""

# Repository scripts; the runner records code snapshots in the new output directory.
env/bin/python -B "$RACS_CODE_DIR/run_racs_controlled_training.py" \
  --repo-root "$PWD" \
  --output-dir "$PWD/output/racs_controlled_training_${SLURM_JOB_ID:?}"
