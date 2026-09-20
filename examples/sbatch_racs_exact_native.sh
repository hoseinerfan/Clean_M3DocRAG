#!/bin/bash -l
#SBATCH --job-name=racs-exact-native
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=output/racs_exact_native_%j.out
#SBATCH --error=output/racs_exact_native_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
module load cuda/12.1.1
export REPO_ROOT="$PWD"
if [[ -f hpc_vital_paths.generated.env ]]; then
  source hpc_vital_paths.generated.env
fi
source scripts/m3docvqa_internal_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
# Full-node CPU allocation prevents native PyTorch defaults oversubscribing a
# smaller allocation. This tests unforced defaults, not a proposed timing setup.
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
unset VECLIB_MAXIMUM_THREADS NUMEXPR_NUM_THREADS BLIS_NUM_THREADS
env/bin/python -B scripts/diagnose_racs_exact_native.py \
  --inputs output/racs_exact_diagnostic_15915206/inputs.json \
  --output-dir "output/racs_exact_native_${SLURM_JOB_ID:?}"
