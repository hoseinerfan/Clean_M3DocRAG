#!/bin/bash -l
#SBATCH --job-name=racs-graph-qa-time
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=output/racs_graph_reader_runtime_%j.out
#SBATCH --error=output/racs_graph_reader_runtime_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
module load cuda/12.1.1
export REPO_ROOT="$PWD"
if [[ -f hpc_vital_paths.generated.env ]]; then
  source hpc_vital_paths.generated.env
fi
source scripts/m3docvqa_internal_env.sh
# Use existing local checkpoints only; this job must not download replacements.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
RACS_RUN_DIR="output/racs_graph_reader_runtime_${SLURM_JOB_ID:?}"
env/bin/python -B scripts/benchmark_racs_graph_reader.py \
  --audit-json output/racs_runtime_audit_15911623/prerequisites.json \
  --replay-json output/racs_graph_replay_15911730/replay.json \
  --run-dir "$RACS_RUN_DIR" --questions 128 --warmup-questions 4 --repeats 4
