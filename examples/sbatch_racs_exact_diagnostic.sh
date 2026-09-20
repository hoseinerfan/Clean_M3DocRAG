#!/bin/bash -l
#SBATCH --job-name=racs-exact-diag
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=output/racs_exact_diagnostic_%j.out
#SBATCH --error=output/racs_exact_diagnostic_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
module load cuda/12.1.1
export REPO_ROOT="$PWD"
if [[ -f hpc_vital_paths.generated.env ]]; then
  source hpc_vital_paths.generated.env
fi
source scripts/m3docvqa_internal_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
env/bin/python -B scripts/diagnose_racs_exact_replay.py \
  --bundle output/racs_online_runtime_15915128/capp.online.bundle.json \
  --exact-summary /mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_exact_maxsim_mmqa_dev/mmqa_dev_exact_maxsim_nprobe4_ret1000.summary.json \
  --prior-query-dir output/racs_faiss_diagnostic_15915127 \
  --output-dir "output/racs_exact_diagnostic_${SLURM_JOB_ID:?}"
