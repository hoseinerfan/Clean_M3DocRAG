#!/bin/bash -l
#SBATCH --job-name=racs-ltr30-train
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=output/racs_lambdamart30_train_%j.out
#SBATCH --error=output/racs_lambdamart30_train_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from Clean_M3DocRAG root}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
env/bin/python -B scripts/run_racs_lambdamart30.py train --repo-root "$PWD" \
  --run-dir "$PWD/output/racs_lambdamart30_${SLURM_JOB_ID:?}"
