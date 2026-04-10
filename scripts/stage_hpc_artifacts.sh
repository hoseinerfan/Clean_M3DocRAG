#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

required_files=(
  "requirements.lock.txt"
  "HPC_SETUP.md"
  "slurm/m3docrag_dev_pipeline.sbatch"
  ".gitignore"
)

for path in "${required_files[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done

git add requirements.lock.txt HPC_SETUP.md slurm/m3docrag_dev_pipeline.sbatch .gitignore

echo "Staged HPC artifacts:"
git status --short
echo
echo 'Next: git commit -m "Add HPC lockfile and Slurm pipeline notes" && git push'
