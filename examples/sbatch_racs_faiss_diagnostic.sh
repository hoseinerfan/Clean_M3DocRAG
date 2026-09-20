#!/bin/bash -l
#SBATCH --job-name=racs-faiss-diag
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=output/racs_faiss_diagnostic_%j.out
#SBATCH --error=output/racs_faiss_diagnostic_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
module load cuda/12.1.1
export REPO_ROOT="$PWD"
if [[ -f hpc_vital_paths.generated.env ]]; then
  source hpc_vital_paths.generated.env
fi
source scripts/m3docvqa_internal_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
RACS_DIAGNOSTIC_BUNDLE="${RACS_DIAGNOSTIC_BUNDLE:-output/racs_online_runtime_15911874/capp.online.bundle.json}"
env/bin/python -B scripts/diagnose_racs_faiss_replay.py \
  --bundle "$RACS_DIAGNOSTIC_BUNDLE" \
  --output-dir "output/racs_faiss_diagnostic_${SLURM_JOB_ID:?}"
