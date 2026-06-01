#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

VITAL_PATHS_ENV="${VITAL_PATHS_ENV:-$REPO_ROOT/hpc_vital_paths.generated.env}"
if [[ -f "$VITAL_PATHS_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$VITAL_PATHS_ENV"
fi

unset M3DOCVQA_INTERNAL_ENV_LOADED
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/m3docvqa_internal_env.sh"

CUSTOM_ROOT="${CUSTOM_ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom}"
CUSTOM_OUTPUT_DIR="${CUSTOM_OUTPUT_DIR:-$CUSTOM_ROOT/outputs}"
REPO_OUTPUT_DIR="${REPO_OUTPUT_DIR:-$REPO_ROOT/output}"

OUT_DIR="${OUT_DIR:-$REPO_OUTPUT_DIR/m3docvqa_content_aware_oracle_router}"
LABEL="${LABEL:-mmqa_dev_content_aware_oracle_single_page_multi_nohyperlink}"
PSEUDO_GOLD="${PSEUDO_GOLD:-$REPO_OUTPUT_DIR/m3docvqa_mmqa_pseudo_page_labels/mmqa_dev_pseudo_page_labels_strict.augmented_gold.jsonl}"
SINGLE_QIDS="${SINGLE_QIDS:-$REPO_OUTPUT_DIR/m3docvqa_qid_groups/single_gold_doc.qids.txt}"
MULTI_QIDS="${MULTI_QIDS:-$REPO_OUTPUT_DIR/m3docvqa_qid_groups/multi_gold_doc.qids.txt}"

DENSE_PRED="${DENSE_PRED:-$CUSTOM_OUTPUT_DIR/mmqa_dev_plain_top224_nprobe4_effdiag_all.prediction.json}"
CONTENT_GPP_NO_HYPERLINK_PRED="${CONTENT_GPP_NO_HYPERLINK_PRED:-$REPO_OUTPUT_DIR/m3docvqa_content_aware_base_sweep/mmqa_train_to_dev_content_aware_base_gpp_no_hyperlink.dev.prediction.json}"
CONTENT_GPP_DOC_HYPERLINK_PRED="${CONTENT_GPP_DOC_HYPERLINK_PRED:-$REPO_OUTPUT_DIR/m3docvqa_content_aware_base_sweep/mmqa_train_to_dev_content_aware_base_gpp_doc_hyperlink.dev.prediction.json}"
CONTENT_GPP_PAGE_HYPERLINK_PRED="${CONTENT_GPP_PAGE_HYPERLINK_PRED:-$REPO_OUTPUT_DIR/m3docvqa_content_aware_base_sweep/mmqa_train_to_dev_content_aware_base_gpp_page_hyperlink.dev.prediction.json}"

mkdir -p "$OUT_DIR"

for required in \
  "$PSEUDO_GOLD" \
  "$SINGLE_QIDS" \
  "$MULTI_QIDS" \
  "$DENSE_PRED" \
  "$CONTENT_GPP_NO_HYPERLINK_PRED" \
  "$CONTENT_GPP_DOC_HYPERLINK_PRED" \
  "$CONTENT_GPP_PAGE_HYPERLINK_PRED"; do
  if [[ ! -f "$required" ]]; then
    echo "missing_required_file: $required" >&2
    exit 1
  fi
done

ROUTED_PRED="$OUT_DIR/$LABEL.prediction.json"
ROUTED_SUMMARY="$OUT_DIR/$LABEL.summary.json"
EVAL_MD="$OUT_DIR/$LABEL.pseudo_page_eval.md"
EVAL_CSV="$OUT_DIR/$LABEL.pseudo_page_eval.csv"
EVAL_JSON="$OUT_DIR/$LABEL.pseudo_page_eval.json"

"$PYTHON_BIN" "$REPO_ROOT/scripts/route_prediction_by_qid_groups.py" \
  --prediction "content_gpp_no_hyperlink=$CONTENT_GPP_NO_HYPERLINK_PRED" \
  --prediction "content_gpp_doc_hyperlink=$CONTENT_GPP_DOC_HYPERLINK_PRED" \
  --prediction "content_gpp_page_hyperlink=$CONTENT_GPP_PAGE_HYPERLINK_PRED" \
  --default-label content_gpp_no_hyperlink \
  --route "single_gold_doc=$SINGLE_QIDS:content_gpp_page_hyperlink" \
  --route "multi_gold_doc=$MULTI_QIDS:content_gpp_no_hyperlink" \
  --output-prediction-json "$ROUTED_PRED" \
  --output-summary-json "$ROUTED_SUMMARY"

eval_common=(
  --gold "$PSEUDO_GOLD"
  --run "dense=$DENSE_PRED"
  --run "content_gpp_no_hyperlink=$CONTENT_GPP_NO_HYPERLINK_PRED"
  --run "content_gpp_page_hyperlink=$CONTENT_GPP_PAGE_HYPERLINK_PRED"
  --run "oracle_single_page_multi_nohyperlink=$ROUTED_PRED"
  --group "single_gold_doc=$SINGLE_QIDS"
  --group "multi_gold_doc=$MULTI_QIDS"
)

"$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_pseudo_page_retrieval.py" \
  "${eval_common[@]}" \
  --format markdown \
  --output "$EVAL_MD"
"$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_pseudo_page_retrieval.py" \
  "${eval_common[@]}" \
  --format csv \
  --output "$EVAL_CSV"
"$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_pseudo_page_retrieval.py" \
  "${eval_common[@]}" \
  --format json \
  --output "$EVAL_JSON"

echo "saved_routed_prediction=$ROUTED_PRED"
echo "saved_routed_summary=$ROUTED_SUMMARY"
echo "saved_eval_md=$EVAL_MD"
echo "saved_eval_csv=$EVAL_CSV"
echo "saved_eval_json=$EVAL_JSON"
