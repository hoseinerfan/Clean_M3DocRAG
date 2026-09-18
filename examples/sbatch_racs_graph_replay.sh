#!/usr/bin/env bash
#SBATCH --job-name=racs-graph-replay
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=output/racs_graph_replay_%j.out
#SBATCH --error=output/racs_graph_replay_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
RACS_REPORT_DIR="output/racs_graph_replay_${SLURM_JOB_ID:?}"
mkdir "$RACS_REPORT_DIR"
env/bin/python -B scripts/validate_racs_graph_replay.py \
  --audit-json output/racs_runtime_audit_15911623/prerequisites.json \
  --sample-size 16 --output-json "$RACS_REPORT_DIR/replay.json"
