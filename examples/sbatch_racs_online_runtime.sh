#!/bin/bash -l
#SBATCH --job-name=racs-online-time
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=output/racs_online_runtime_%j.out
#SBATCH --error=output/racs_online_runtime_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
module load cuda/12.1.1
export REPO_ROOT="$PWD"
if [[ -f hpc_vital_paths.generated.env ]]; then
  source hpc_vital_paths.generated.env
fi
source scripts/m3docvqa_internal_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
: "${RACS_SPLADE_MODEL_DIR:?Set RACS_SPLADE_MODEL_DIR to the existing local checkpoint directory before sbatch}"
RACS_SPLADE_TOKENIZER_DIR="${RACS_SPLADE_TOKENIZER_DIR:-$RACS_SPLADE_MODEL_DIR}"
env/bin/python -B scripts/benchmark_racs_online.py \
  --splade-model-dir "$RACS_SPLADE_MODEL_DIR" \
  --splade-tokenizer-dir "$RACS_SPLADE_TOKENIZER_DIR" \
  --audit-json output/racs_runtime_audit_15911623/prerequisites.json \
  --replay-json output/racs_graph_replay_15911730/replay.json \
  --run-dir "output/racs_online_runtime_${SLURM_JOB_ID:?}" \
  --questions 128 --warmup-questions 4 --repeats 4
