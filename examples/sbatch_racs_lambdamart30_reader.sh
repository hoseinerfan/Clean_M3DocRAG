#!/bin/bash -l
#SBATCH --job-name=racs-ltr30-reader
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=output/racs_lambdamart30_reader_%j.out
#SBATCH --error=output/racs_lambdamart30_reader_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from Clean_M3DocRAG root}"
RACS_TRAINING_DIR="${1:?Pass the completed training output directory}"
module load cuda/12.1.1
export REPO_ROOT="$PWD"
if [[ -f hpc_vital_paths.generated.env ]]; then
  source hpc_vital_paths.generated.env
fi
source scripts/m3docvqa_internal_env.sh
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RACS_READER_DIR="$PWD/output/racs_lambdamart30_reader_${SLURM_JOB_ID:?}"
RACS_GOLD="$PWD/output/m3docvqa_mmqa_direct_evidence_pseudo_page_labels/mmqa_dev_pseudo_page_labels_direct_exactonly_adaptive_norm05.augmented_gold.jsonl"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || true
fi
env/bin/python -B scripts/run_racs_lambdamart30.py prepare-reader \
  --training-dir "$RACS_TRAINING_DIR" --run-dir "$RACS_READER_DIR"
env/bin/python -B scripts/run_m3docvqa_external_retrieval_qa.py \
  --prediction-json "$RACS_READER_DIR/reader_input.json" --gold "$RACS_GOLD" \
  --data-name m3-docvqa --split dev --model-name-or-path Qwen2-VL-7B-Instruct \
  --bits 16 --qa-top-pages 4 --eval-num-shards 1 --eval-shard-id 0 \
  --doc-image-cache-size 16 --save-every 25 \
  --output-prediction-json "$RACS_READER_DIR/qa.prediction.json" \
  --output-eval-json "$RACS_READER_DIR/qa.eval.json" --run-eval
env/bin/python -B scripts/run_racs_lambdamart30.py check-reader --run-dir "$RACS_READER_DIR"
