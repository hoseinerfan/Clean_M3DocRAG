#!/bin/bash -l
#SBATCH --job-name=racs-bge-reader
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=output/racs_bge_reader_%j.out
#SBATCH --error=output/racs_bge_reader_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
module load cuda/12.1.1
export REPO_ROOT="$PWD"
if [[ -f hpc_vital_paths.generated.env ]]; then
  source hpc_vital_paths.generated.env
fi
source scripts/m3docvqa_internal_env.sh

# Pin the retained top-1000 / alpha-0.20 baseline, not the top-100 trial.
RACS_PYTHON="$PWD/env/bin/python"
test -x "$RACS_PYTHON"
RACS_GOLD=output/m3docvqa_mmqa_direct_evidence_pseudo_page_labels/mmqa_dev_pseudo_page_labels_direct_exactonly_adaptive_norm05.augmented_gold.jsonl
RACS_BASE=output/m3docvqa_gpp_hyperlink_node_ablation_exact_maxsim/mmqa_dev_exact_maxsim_gpp_hyperlink_node_no_hyperlink.prediction.json
RACS_BGE_STEM=output/m3docvqa_standard_reranker_baselines/mmqa_dev_bge_reranker_base_blend0p20_cross_encoder_gpp_top1000
RACS_RUN_DIR="output/racs_bge_reader_top4_${SLURM_JOB_ID:?}"
# Fail on an existing directory. Do not silently resume or replace any results.
mkdir "$RACS_RUN_DIR"
RACS_QA_STEM="$RACS_RUN_DIR/mmqa_dev_bge_qwen2vl_top4"

printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || true
fi

"$RACS_PYTHON" scripts/validate_racs_bge_reader.py prepare \
  --gold "$RACS_GOLD" --base "$RACS_BASE" \
  --prediction "$RACS_BGE_STEM.prediction.json" \
  --summary "$RACS_BGE_STEM.summary.json" --run-dir "$RACS_RUN_DIR"

# Only frozen-reader inference is new. No BGE/CAPP training or reranking.
"$RACS_PYTHON" scripts/run_m3docvqa_external_retrieval_qa.py \
  --prediction-json "$RACS_RUN_DIR/bge_top4.reader_input.json" --gold "$RACS_GOLD" \
  --data-name m3-docvqa --split dev \
  --model-name-or-path Qwen2-VL-7B-Instruct --bits 16 --qa-top-pages 4 \
  --eval-num-shards 1 --eval-shard-id 0 --doc-image-cache-size 16 --save-every 25 \
  --output-prediction-json "$RACS_QA_STEM.prediction.json" \
  --output-eval-json "$RACS_QA_STEM.eval.json" --run-eval

"$RACS_PYTHON" scripts/validate_racs_bge_reader.py check --run-dir "$RACS_RUN_DIR"
printf 'saved_bge_reader_directory=%s\n' "$RACS_RUN_DIR"
