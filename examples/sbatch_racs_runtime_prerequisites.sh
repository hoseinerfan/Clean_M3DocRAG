#!/usr/bin/env bash
#SBATCH --job-name=racs-runtime-audit
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=output/racs_runtime_audit_%j.out
#SBATCH --error=output/racs_runtime_audit_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
test -x env/bin/python
RACS_AUDIT_DIR="output/racs_runtime_audit_${SLURM_JOB_ID:?}"
mkdir "$RACS_AUDIT_DIR"
# Read-only inventory: no GPU, retraining, generation, or historical-output writes.
env/bin/python scripts/audit_racs_runtime_prerequisites.py \
  --repo-root "$PWD" --output-json "$RACS_AUDIT_DIR/prerequisites.json"
