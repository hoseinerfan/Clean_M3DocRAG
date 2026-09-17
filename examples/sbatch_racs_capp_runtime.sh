#!/usr/bin/env bash
#SBATCH --job-name=racs-capp-runtime
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=output/racs_capp_runtime_%j.out
#SBATCH --error=output/racs_capp_runtime_%j.err

set -euo pipefail
# Slurm copies the submitted script, so do not locate the repository via $0.
cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch from the repository root}"
test -f scripts/benchmark_capp_runtime.py
test -x env/bin/python

RACS_ARTIFACT_DIR=output/m3docvqa_content_aware_exact_maxsim_direct_exactonly_adaptive_norm05
RACS_LABEL=mmqa_train_to_dev_content_aware_fixed_alpha_0p40_base_exact_maxsim_gpp_direct_exactonly_adaptive_norm05
RACS_LEGACY=output/m3docvqa_gpp_hyperlink_node_ablation_mmr_target1
RACS_REPORT_DIR="output/racs_capp_runtime_${SLURM_JOB_ID:?}"

printf 'benchmark_git_commit='
# Git may exist on the login node but not on compute nodes. Provenance logging
# must not prevent inference; the Python report also records code SHA-256s.
if command -v git >/dev/null 2>&1; then
  git rev-parse HEAD || printf 'unavailable\n'
else
  printf 'unavailable (git not on compute-node PATH)\n'
fi

# No GPU, training, source search, or change to original predictions.
env/bin/python scripts/benchmark_capp_runtime.py \
  --model-json "$RACS_ARTIFACT_DIR/$RACS_LABEL.model.json" \
  --base-pred output/m3docvqa_gpp_hyperlink_node_ablation_exact_maxsim/mmqa_dev_exact_maxsim_gpp_hyperlink_node_no_hyperlink.prediction.json \
  --question-jsonl output/m3docvqa_mmqa_direct_evidence_pseudo_page_labels/mmqa_dev_pseudo_page_labels_direct_exactonly_adaptive_norm05.augmented_gold.jsonl \
  --page-text-jsonl /mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_page_text/m3docvqa_dev_page_text.jsonl \
  --saved-pred "$RACS_ARTIFACT_DIR/$RACS_LABEL.dev.prediction.json" \
  --source "gpp_no_hyperlink=$RACS_LEGACY/mmqa_dev_gpp_hyperlink_node_no_hyperlink.prediction.json" \
  --source "gpp_doc_hyperlink=$RACS_LEGACY/mmqa_dev_gpp_hyperlink_node_docnode_to_hyperlink_docs.prediction.json" \
  --source "gpp_page_hyperlink=$RACS_LEGACY/mmqa_dev_gpp_hyperlink_node_pagenode_to_hyperlink_pages.prediction.json" \
  --expected-qids 2441 --expected-candidates 1000 \
  --warmup-passes 1 --repeats 3 \
  --output-json "$RACS_REPORT_DIR/runtime.json"
