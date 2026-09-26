#!/bin/bash -l
#SBATCH --job-name=racs-table9-audit
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=output/racs_table9_audit_%j.out
#SBATCH --error=output/racs_table9_audit_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from Clean_M3DocRAG root}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
env/bin/python -B scripts/audit_racs_table9_qa.py \
  --hpc-root "$PWD" --report "$PWD/output/racs_table9_qa_audit_${SLURM_JOB_ID:?}.json"
