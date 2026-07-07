#!/usr/bin/env bash
#SBATCH --job-name=mpdocvqa-exact-gpp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --output=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/MPDocVQA_M3DocRAG/logs/exactmaxsim_gpp_%j.out
#SBATCH --error=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/MPDocVQA_M3DocRAG/logs/exactmaxsim_gpp_%j.err

set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG}"
cd "$REPO_ROOT"

source mpdocvqa/env_hpc.sh

bash mpdocvqa/run_exactmaxsim_gpp_mpdocvqa.sh
