#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/env/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

VITAL_PATHS_ENV="${VITAL_PATHS_ENV:-$REPO_ROOT/hpc_vital_paths.generated.env}"
if [[ -f "$VITAL_PATHS_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$VITAL_PATHS_ENV"
fi

OUT_DIR="${OUT_DIR:-$REPO_ROOT/output/m3docvqa_content_aware_learned_query_alpha_page5}"
LABEL="${LABEL:-mmqa_train_to_dev_content_aware_learned_query_alpha_page5}"
TUNE_HIT_K="${TUNE_HIT_K:-5}"
TUNE_BLEND_ALPHA_GRID="${TUNE_BLEND_ALPHA_GRID:-0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70,0.80,1.00}"
QUERY_ALPHA_FEATURE_TOP_K="${QUERY_ALPHA_FEATURE_TOP_K:-20}"
QUERY_ALPHA_RIDGE="${QUERY_ALPHA_RIDGE:-1.0}"

TRAIN_BASE_PRED="${TRAIN_BASE_PRED:-$REPO_ROOT/output/m3docvqa_gpp_hyperlink_node_ablation_mmr_target1_train_real/mmqa_train_gpp_hyperlink_node_no_hyperlink.prediction.json}"
EVAL_BASE_PRED="${EVAL_BASE_PRED:-$REPO_ROOT/output/m3docvqa_gpp_hyperlink_node_ablation_mmr_target1/mmqa_dev_gpp_hyperlink_node_no_hyperlink.prediction.json}"
EVAL_GOLD="${EVAL_GOLD:-$REPO_ROOT/output/m3docvqa_mmqa_pseudo_page_labels/mmqa_dev_pseudo_page_labels_strict.augmented_gold.jsonl}"

mkdir -p "$OUT_DIR"

echo
echo "== M3DocVQA content-aware learned query alpha =="
TRAIN_DENSE_PRED="$TRAIN_BASE_PRED" \
EVAL_DENSE_PRED="$EVAL_BASE_PRED" \
OUT_DIR="$OUT_DIR" \
LABEL="$LABEL" \
AUTO_TUNE_BLEND_ALPHA=1 \
QUERY_ADAPTIVE_ALPHA=0 \
LEARNED_QUERY_ALPHA=1 \
TUNE_HIT_K="$TUNE_HIT_K" \
TUNE_BLEND_ALPHA_GRID="$TUNE_BLEND_ALPHA_GRID" \
QUERY_ALPHA_FEATURE_TOP_K="$QUERY_ALPHA_FEATURE_TOP_K" \
QUERY_ALPHA_RIDGE="$QUERY_ALPHA_RIDGE" \
bash "$REPO_ROOT/examples/run_m3docvqa_content_aware_pseudo_page_reranker.sh"

PRED_PATH="$OUT_DIR/${LABEL}.dev.prediction.json"
SUMMARY_PATH="$OUT_DIR/${LABEL}.summary.json"
RECALL_MD="${RECALL_MD:-$OUT_DIR/${LABEL}.recall.md}"

echo
echo "== Learned query alpha recall table =="
run_args=(
  --run "gpp_no_hyperlink=$EVAL_BASE_PRED"
  --run "learned_query_alpha=$PRED_PATH"
)
if [[ -f "$REPO_ROOT/output/m3docvqa_content_aware_auto_blend_sweep_page5/mmqa_train_to_dev_content_aware_base_gpp_no_hyperlink.dev.prediction.json" ]]; then
  run_args+=(
    --run "content_aware_page5=$REPO_ROOT/output/m3docvqa_content_aware_auto_blend_sweep_page5/mmqa_train_to_dev_content_aware_base_gpp_no_hyperlink.dev.prediction.json"
  )
fi
if [[ -f "$REPO_ROOT/output/m3docvqa_content_aware_counterfactual_oof_hybrid/insert_rank4/mmqa_train_to_dev_content_aware_counterfactual_oof_insert4.dev.prediction.json" ]]; then
  run_args+=(
    --run "oof_hybrid_insert4=$REPO_ROOT/output/m3docvqa_content_aware_counterfactual_oof_hybrid/insert_rank4/mmqa_train_to_dev_content_aware_counterfactual_oof_insert4.dev.prediction.json"
  )
fi

"$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_pseudo_page_retrieval.py" \
  --gold "$EVAL_GOLD" \
  "${run_args[@]}" \
  --output "$RECALL_MD"

echo "saved_prediction=$PRED_PATH"
echo "saved_summary=$SUMMARY_PATH"
echo "saved_recall_md=$RECALL_MD"
