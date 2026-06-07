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

OUT_DIR="${OUT_DIR:-$REPO_ROOT/output/m3docvqa_content_aware_base_aware_alpha_gate_page5}"
LABEL="${LABEL:-mmqa_train_to_dev_content_aware_base_aware_alpha_gate_page5}"
EVAL_GOLD="${EVAL_GOLD:-$REPO_ROOT/output/m3docvqa_mmqa_pseudo_page_labels/mmqa_dev_pseudo_page_labels_strict.augmented_gold.jsonl}"
GPP_EVAL_OUT_DIR="${GPP_EVAL_OUT_DIR:-$REPO_ROOT/output/m3docvqa_gpp_hyperlink_node_ablation_mmr_target1}"
GPP_TRAIN_OUT_DIR="${GPP_TRAIN_OUT_DIR:-$REPO_ROOT/output/m3docvqa_gpp_hyperlink_node_ablation_mmr_target1_train_real}"
GPP_EVAL_LABEL_PREFIX="${GPP_EVAL_LABEL_PREFIX:-mmqa_dev_gpp_hyperlink_node}"
GPP_TRAIN_LABEL_PREFIX="${GPP_TRAIN_LABEL_PREFIX:-mmqa_train_gpp_hyperlink_node}"
EVAL_BASE_PRED="${EVAL_BASE_PRED:-$GPP_EVAL_OUT_DIR/${GPP_EVAL_LABEL_PREFIX}_no_hyperlink.prediction.json}"
TRAIN_BASE_PRED="${TRAIN_BASE_PRED:-$GPP_TRAIN_OUT_DIR/${GPP_TRAIN_LABEL_PREFIX}_no_hyperlink.prediction.json}"

TUNE_HIT_K="${TUNE_HIT_K:-5}"
TUNE_BLEND_ALPHA_GRID="${TUNE_BLEND_ALPHA_GRID:-0.00,0.03,0.05,0.08,0.10,0.15,0.20,0.25,0.30,0.40}"
ALPHA_UTILITY_THRESHOLD_GRID="${ALPHA_UTILITY_THRESHOLD_GRID:-0.00,0.01,0.02,0.05,0.10}"
ALPHA_UTILITY_RISK_PENALTY_GRID="${ALPHA_UTILITY_RISK_PENALTY_GRID:-1.00,1.50,2.00,3.00}"
QUERY_ALPHA_FEATURE_TOP_K="${QUERY_ALPHA_FEATURE_TOP_K:-20}"
ALPHA_UTILITY_RIDGE="${ALPHA_UTILITY_RIDGE:-1.0}"

mkdir -p "$OUT_DIR"

echo
echo "== M3DocVQA content-aware base-aware alpha gate =="
TRAIN_DENSE_PRED="$TRAIN_BASE_PRED" \
EVAL_DENSE_PRED="$EVAL_BASE_PRED" \
OUT_DIR="$OUT_DIR" \
LABEL="$LABEL" \
AUTO_TUNE_BLEND_ALPHA=1 \
QUERY_ADAPTIVE_ALPHA=0 \
LEARNED_QUERY_ALPHA=0 \
LEARNED_ALPHA_ACTION=0 \
LEARNED_ALPHA_UTILITY_GATE=0 \
BASE_AWARE_ALPHA_UTILITY_GATE=1 \
TUNE_HIT_K="$TUNE_HIT_K" \
TUNE_BLEND_ALPHA_GRID="$TUNE_BLEND_ALPHA_GRID" \
QUERY_ALPHA_FEATURE_TOP_K="$QUERY_ALPHA_FEATURE_TOP_K" \
ALPHA_UTILITY_RIDGE="$ALPHA_UTILITY_RIDGE" \
ALPHA_UTILITY_THRESHOLD_GRID="$ALPHA_UTILITY_THRESHOLD_GRID" \
ALPHA_UTILITY_RISK_PENALTY_GRID="$ALPHA_UTILITY_RISK_PENALTY_GRID" \
bash "$REPO_ROOT/examples/run_m3docvqa_content_aware_pseudo_page_reranker.sh"

PRED_PATH="$OUT_DIR/${LABEL}.dev.prediction.json"
MODEL_PATH="$OUT_DIR/${LABEL}.model.json"
SUMMARY_PATH="$OUT_DIR/${LABEL}.summary.json"
RECALL_MD="${RECALL_MD:-$OUT_DIR/${LABEL}.recall.md}"

echo
echo "== Base-aware alpha gate recall table =="
run_args=(
  --run "gpp_no_hyperlink=$EVAL_BASE_PRED"
  --run "base_aware_alpha_gate=$PRED_PATH"
)
if [[ -f "$REPO_ROOT/output/m3docvqa_content_aware_auto_blend_sweep_page5/mmqa_train_to_dev_content_aware_base_gpp_no_hyperlink.dev.prediction.json" ]]; then
  run_args+=(
    --run "fixed_content_aware_page5=$REPO_ROOT/output/m3docvqa_content_aware_auto_blend_sweep_page5/mmqa_train_to_dev_content_aware_base_gpp_no_hyperlink.dev.prediction.json"
  )
fi
if [[ -f "$REPO_ROOT/output/m3docvqa_content_aware_alpha_utility_gate_page5/mmqa_train_to_dev_content_aware_alpha_utility_gate_page5.dev.prediction.json" ]]; then
  run_args+=(
    --run "alpha_utility_gate=$REPO_ROOT/output/m3docvqa_content_aware_alpha_utility_gate_page5/mmqa_train_to_dev_content_aware_alpha_utility_gate_page5.dev.prediction.json"
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

if [[ "${RUN_TRANSFER:-1}" == "1" ]]; then
  echo
  echo "== Zero-shot transfer: base-aware alpha gate =="
  DATASETS="${DATASETS:-dude mmdocir}" \
  MODEL_JSON="$MODEL_PATH" \
  RUN_DENSE_BASE="${RUN_DENSE_BASE:-1}" \
  RUN_GPP_BASE="${RUN_GPP_BASE:-1}" \
  RUN_DOCSEED_BASE="${RUN_DOCSEED_BASE:-0}" \
  OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-trained_content_aware_transfer_base_aware_alpha_gate}" \
  REPORT_OUT="${TRANSFER_REPORT_OUT:-$OUT_DIR/${LABEL}.transfer.md}" \
  REPORT_CSV_OUT="${TRANSFER_REPORT_CSV_OUT:-$OUT_DIR/${LABEL}.transfer.csv}" \
  bash "$REPO_ROOT/examples/run_trained_content_aware_transfer_selected_datasets.sh"
fi

echo "saved_model=$MODEL_PATH"
echo "saved_prediction=$PRED_PATH"
echo "saved_summary=$SUMMARY_PATH"
echo "saved_recall_md=$RECALL_MD"
