#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

VITAL_PATHS_ENV="${VITAL_PATHS_ENV:-$REPO_ROOT/hpc_vital_paths.generated.env}"
if [[ -f "$VITAL_PATHS_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$VITAL_PATHS_ENV"
fi

CUSTOM_ROOT="${CUSTOM_ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/env/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

OUT_DIR="${OUT_DIR:-$REPO_ROOT/output/m3docvqa_content_aware_counterfactual_hybrid}"
CONTENT_AWARE_DIR="${CONTENT_AWARE_DIR:-$REPO_ROOT/output/m3docvqa_content_aware_auto_blend_sweep_page5}"
CONTENT_AWARE_LABEL="${CONTENT_AWARE_LABEL:-mmqa_train_to_dev_content_aware_base_gpp_no_hyperlink}"
CONTENT_MODEL_JSON="${CONTENT_MODEL_JSON:-$CONTENT_AWARE_DIR/${CONTENT_AWARE_LABEL}.model.json}"
CONTENT_EVAL_PRED="${CONTENT_EVAL_PRED:-$CONTENT_AWARE_DIR/${CONTENT_AWARE_LABEL}.dev.prediction.json}"
CONTENT_TRAIN_PRED="${CONTENT_TRAIN_PRED:-$OUT_DIR/${CONTENT_AWARE_LABEL}.train.prediction.json}"

TRAIN_GOLD="${TRAIN_GOLD:-$REPO_ROOT/output/m3docvqa_mmqa_pseudo_page_labels/mmqa_train_pseudo_page_labels_strict.augmented_gold.jsonl}"
EVAL_GOLD="${EVAL_GOLD:-$REPO_ROOT/output/m3docvqa_mmqa_pseudo_page_labels/mmqa_dev_pseudo_page_labels_strict.augmented_gold.jsonl}"
TRAIN_PAGE_TEXT_JSONL="${TRAIN_PAGE_TEXT_JSONL:-${M3DOCVQA_TRAIN_PAGE_TEXT_JSONL:-$CUSTOM_ROOT/outputs/m3docvqa_page_text/m3docvqa_train_page_text.jsonl}}"
EVAL_PAGE_TEXT_JSONL="${EVAL_PAGE_TEXT_JSONL:-${M3DOCVQA_DEV_PAGE_TEXT_JSONL:-${M3DOCVQA_PAGE_TEXT_JSONL:-$CUSTOM_ROOT/outputs/m3docvqa_page_text/m3docvqa_dev_page_text.jsonl}}}"

GPP_TRAIN_OUT_DIR="${GPP_TRAIN_OUT_DIR:-$REPO_ROOT/output/m3docvqa_gpp_hyperlink_node_ablation_mmr_target1_train_real}"
GPP_EVAL_OUT_DIR="${GPP_EVAL_OUT_DIR:-$REPO_ROOT/output/m3docvqa_gpp_hyperlink_node_ablation_mmr_target1}"
GPP_TRAIN_LABEL_PREFIX="${GPP_TRAIN_LABEL_PREFIX:-mmqa_train_gpp_hyperlink_node}"
GPP_EVAL_LABEL_PREFIX="${GPP_EVAL_LABEL_PREFIX:-mmqa_dev_gpp_hyperlink_node}"

TRAIN_GPP_NO_HYPERLINK_PRED="${TRAIN_GPP_NO_HYPERLINK_PRED:-$GPP_TRAIN_OUT_DIR/${GPP_TRAIN_LABEL_PREFIX}_no_hyperlink.prediction.json}"
TRAIN_GPP_DOC_HYPERLINK_PRED="${TRAIN_GPP_DOC_HYPERLINK_PRED:-$GPP_TRAIN_OUT_DIR/${GPP_TRAIN_LABEL_PREFIX}_docnode_to_hyperlink_docs.prediction.json}"
TRAIN_GPP_PAGE_HYPERLINK_PRED="${TRAIN_GPP_PAGE_HYPERLINK_PRED:-$GPP_TRAIN_OUT_DIR/${GPP_TRAIN_LABEL_PREFIX}_pagenode_to_hyperlink_pages.prediction.json}"
EVAL_GPP_NO_HYPERLINK_PRED="${EVAL_GPP_NO_HYPERLINK_PRED:-$GPP_EVAL_OUT_DIR/${GPP_EVAL_LABEL_PREFIX}_no_hyperlink.prediction.json}"
EVAL_GPP_DOC_HYPERLINK_PRED="${EVAL_GPP_DOC_HYPERLINK_PRED:-$GPP_EVAL_OUT_DIR/${GPP_EVAL_LABEL_PREFIX}_docnode_to_hyperlink_docs.prediction.json}"
EVAL_GPP_PAGE_HYPERLINK_PRED="${EVAL_GPP_PAGE_HYPERLINK_PRED:-$GPP_EVAL_OUT_DIR/${GPP_EVAL_LABEL_PREFIX}_pagenode_to_hyperlink_pages.prediction.json}"

TRAIN_SPLADE_PRED="${TRAIN_SPLADE_PRED:-${M3DOCVQA_TRAIN_SPLADE_PRED:-$CUSTOM_ROOT/outputs/m3docvqa_splade_mmqa_train/mmqa_train_splade.prediction.json}}"
EVAL_SPLADE_PRED="${EVAL_SPLADE_PRED:-${M3DOCVQA_DEV_SPLADE_PRED:-${M3DOCVQA_SPLADE_PRED:-${SPARSE_PRED:-$CUSTOM_ROOT/outputs/m3docvqa_splade_mmqa_dev/mmqa_dev_splade.prediction.json}}}}"

RUN_INSERT_RANKS="${RUN_INSERT_RANKS:-4 5}"
FORCE_REBUILD_CONTENT_TRAIN="${FORCE_REBUILD_CONTENT_TRAIN:-0}"
FORCE_REBUILD_CONTENT_EVAL="${FORCE_REBUILD_CONTENT_EVAL:-0}"

require_file() {
  local name="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    echo "missing_${name}: $path" >&2
    exit 1
  fi
}

add_source_if_exists() {
  local label="$1"
  local path="$2"
  local array_name="$3"
  if [[ -f "$path" ]]; then
    if [[ "$array_name" == "train" ]]; then
      train_source_args+=(--source "$label=$path")
    else
      eval_source_args+=(--source "$label=$path")
    fi
  else
    echo "skip_missing_${array_name}_source_${label}=$path" >&2
  fi
}

mkdir -p "$OUT_DIR"
require_file content_model_json "$CONTENT_MODEL_JSON"
require_file train_gold "$TRAIN_GOLD"
require_file eval_gold "$EVAL_GOLD"
require_file train_page_text_jsonl "$TRAIN_PAGE_TEXT_JSONL"
require_file eval_page_text_jsonl "$EVAL_PAGE_TEXT_JSONL"
require_file train_gpp_no_hyperlink_pred "$TRAIN_GPP_NO_HYPERLINK_PRED"
require_file eval_gpp_no_hyperlink_pred "$EVAL_GPP_NO_HYPERLINK_PRED"

train_source_args=()
eval_source_args=()
add_source_if_exists splade "$TRAIN_SPLADE_PRED" train
add_source_if_exists gpp_no_hyperlink "$TRAIN_GPP_NO_HYPERLINK_PRED" train
add_source_if_exists gpp_doc_hyperlink "$TRAIN_GPP_DOC_HYPERLINK_PRED" train
add_source_if_exists gpp_page_hyperlink "$TRAIN_GPP_PAGE_HYPERLINK_PRED" train
add_source_if_exists splade "$EVAL_SPLADE_PRED" eval
add_source_if_exists gpp_no_hyperlink "$EVAL_GPP_NO_HYPERLINK_PRED" eval
add_source_if_exists gpp_doc_hyperlink "$EVAL_GPP_DOC_HYPERLINK_PRED" eval
add_source_if_exists gpp_page_hyperlink "$EVAL_GPP_PAGE_HYPERLINK_PRED" eval

if [[ "$FORCE_REBUILD_CONTENT_TRAIN" == "1" || ! -f "$CONTENT_TRAIN_PRED" ]]; then
  echo
  echo "== Build train-side content-aware base =="
  "$PYTHON_BIN" "$REPO_ROOT/scripts/apply_trained_content_aware_page_reranker.py" \
    --model-json "$CONTENT_MODEL_JSON" \
    --base-pred "$TRAIN_GPP_NO_HYPERLINK_PRED" \
    --page-text-jsonl "$TRAIN_PAGE_TEXT_JSONL" \
    --gold "$TRAIN_GOLD" \
    "${train_source_args[@]}" \
    --candidate-top-k "${CONTENT_CANDIDATE_TOP_K:-1000}" \
    --output-prediction-json "$CONTENT_TRAIN_PRED" \
    --output-summary-json "$OUT_DIR/${CONTENT_AWARE_LABEL}.train.summary.json" \
    --output-table-md "$OUT_DIR/${CONTENT_AWARE_LABEL}.train.table.md" \
    --output-prior-jsonl "$OUT_DIR/${CONTENT_AWARE_LABEL}.train.prior.jsonl"
else
  echo "reuse_content_train_pred=$CONTENT_TRAIN_PRED"
fi

if [[ "$FORCE_REBUILD_CONTENT_EVAL" == "1" || ! -f "$CONTENT_EVAL_PRED" ]]; then
  CONTENT_EVAL_PRED="$OUT_DIR/${CONTENT_AWARE_LABEL}.dev.prediction.json"
  echo
  echo "== Build eval-side content-aware base =="
  "$PYTHON_BIN" "$REPO_ROOT/scripts/apply_trained_content_aware_page_reranker.py" \
    --model-json "$CONTENT_MODEL_JSON" \
    --base-pred "$EVAL_GPP_NO_HYPERLINK_PRED" \
    --page-text-jsonl "$EVAL_PAGE_TEXT_JSONL" \
    --gold "$EVAL_GOLD" \
    "${eval_source_args[@]}" \
    --candidate-top-k "${CONTENT_CANDIDATE_TOP_K:-1000}" \
    --output-prediction-json "$CONTENT_EVAL_PRED" \
    --output-summary-json "$OUT_DIR/${CONTENT_AWARE_LABEL}.dev.summary.json" \
    --output-table-md "$OUT_DIR/${CONTENT_AWARE_LABEL}.dev.table.md" \
    --output-prior-jsonl "$OUT_DIR/${CONTENT_AWARE_LABEL}.dev.prior.jsonl"
else
  echo "reuse_content_eval_pred=$CONTENT_EVAL_PRED"
fi

require_file content_train_pred "$CONTENT_TRAIN_PRED"
require_file content_eval_pred "$CONTENT_EVAL_PRED"

eval_args=(
  --run "gpp_no_hyperlink=$EVAL_GPP_NO_HYPERLINK_PRED"
  --run "content_aware=$CONTENT_EVAL_PRED"
)

for insert_rank in $RUN_INSERT_RANKS; do
  repair_hit_k="${REPAIR_HIT_K:-$insert_rank}"
  promotion_rank_min="${PROMOTION_RANK_MIN:-$((insert_rank + 1))}"
  label="mmqa_train_to_dev_content_aware_counterfactual_insert${insert_rank}"
  run_out_dir="$OUT_DIR/insert_rank${insert_rank}"

  echo
  echo "== Counterfactual repair on content-aware base: insert rank $insert_rank =="
  TRAIN_GOLD="$TRAIN_GOLD" \
  EVAL_GOLD="$EVAL_GOLD" \
  TRAIN_PAGE_TEXT_JSONL="$TRAIN_PAGE_TEXT_JSONL" \
  EVAL_PAGE_TEXT_JSONL="$EVAL_PAGE_TEXT_JSONL" \
  TRAIN_BASE_PRED="$CONTENT_TRAIN_PRED" \
  EVAL_BASE_PRED="$CONTENT_EVAL_PRED" \
  TRAIN_SPLADE_PRED="$TRAIN_SPLADE_PRED" \
  EVAL_SPLADE_PRED="$EVAL_SPLADE_PRED" \
  TRAIN_GPP_DOC_HYPERLINK_PRED="$TRAIN_GPP_DOC_HYPERLINK_PRED" \
  TRAIN_GPP_PAGE_HYPERLINK_PRED="$TRAIN_GPP_PAGE_HYPERLINK_PRED" \
  EVAL_GPP_DOC_HYPERLINK_PRED="$EVAL_GPP_DOC_HYPERLINK_PRED" \
  EVAL_GPP_PAGE_HYPERLINK_PRED="$EVAL_GPP_PAGE_HYPERLINK_PRED" \
  REPAIR_HIT_K="$repair_hit_k" \
  INSERT_RANK="$insert_rank" \
  PROMOTION_RANK_MIN="$promotion_rank_min" \
  AUTO_TUNE_THRESHOLD="${AUTO_TUNE_THRESHOLD:-1}" \
  LOST_PENALTY="${LOST_PENALTY:-2.0}" \
  OUT_DIR="$run_out_dir" \
  LABEL="$label" \
  bash "$REPO_ROOT/examples/run_m3docvqa_counterfactual_page_promotion.sh"

  pred="$run_out_dir/${label}.dev.prediction.json"
  if [[ -f "$pred" ]]; then
    eval_args+=(--run "content_counterfactual_insert${insert_rank}=$pred")
  fi
done

"$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_pseudo_page_retrieval.py" \
  --gold "$EVAL_GOLD" \
  "${eval_args[@]}" \
  --format markdown \
  --output "$OUT_DIR/content_aware_counterfactual_hybrid_eval.md"

"$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_pseudo_page_retrieval.py" \
  --gold "$EVAL_GOLD" \
  "${eval_args[@]}" \
  --format csv \
  --output "$OUT_DIR/content_aware_counterfactual_hybrid_eval.csv"

echo "saved_eval_md=$OUT_DIR/content_aware_counterfactual_hybrid_eval.md"
echo "saved_eval_csv=$OUT_DIR/content_aware_counterfactual_hybrid_eval.csv"
