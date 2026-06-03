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

OUT_DIR="${OUT_DIR:-$REPO_ROOT/output/m3docvqa_content_aware_counterfactual_oof_hybrid}"
FOLD_DIR="${FOLD_DIR:-$OUT_DIR/folds}"
CONTENT_FOLD_DIR="${CONTENT_FOLD_DIR:-$OUT_DIR/content_aware_folds}"
OOF_TRAIN_PRED="${OOF_TRAIN_PRED:-$OUT_DIR/mmqa_train_content_aware_oof.prediction.json}"

TRAIN_GOLD="${TRAIN_GOLD:-$REPO_ROOT/output/m3docvqa_mmqa_pseudo_page_labels/mmqa_train_pseudo_page_labels_strict.augmented_gold.jsonl}"
EVAL_GOLD="${EVAL_GOLD:-$REPO_ROOT/output/m3docvqa_mmqa_pseudo_page_labels/mmqa_dev_pseudo_page_labels_strict.augmented_gold.jsonl}"
TRAIN_PAGE_TEXT_JSONL="${TRAIN_PAGE_TEXT_JSONL:-${M3DOCVQA_TRAIN_PAGE_TEXT_JSONL:-$CUSTOM_ROOT/outputs/m3docvqa_page_text/m3docvqa_train_page_text.jsonl}}"
EVAL_PAGE_TEXT_JSONL="${EVAL_PAGE_TEXT_JSONL:-${M3DOCVQA_DEV_PAGE_TEXT_JSONL:-${M3DOCVQA_PAGE_TEXT_JSONL:-$CUSTOM_ROOT/outputs/m3docvqa_page_text/m3docvqa_dev_page_text.jsonl}}}"

CONTENT_AWARE_DIR="${CONTENT_AWARE_DIR:-$REPO_ROOT/output/m3docvqa_content_aware_auto_blend_sweep_page5}"
CONTENT_AWARE_LABEL="${CONTENT_AWARE_LABEL:-mmqa_train_to_dev_content_aware_base_gpp_no_hyperlink}"
CONTENT_MODEL_JSON="${CONTENT_MODEL_JSON:-$CONTENT_AWARE_DIR/${CONTENT_AWARE_LABEL}.model.json}"
CONTENT_EVAL_PRED="${CONTENT_EVAL_PRED:-$CONTENT_AWARE_DIR/${CONTENT_AWARE_LABEL}.dev.prediction.json}"

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

OOF_FOLD_COUNT="${OOF_FOLD_COUNT:-5}"
OOF_SEED="${OOF_SEED:-13}"
RUN_INSERT_RANKS="${RUN_INSERT_RANKS:-4 5}"
FORCE_REBUILD_FOLDS="${FORCE_REBUILD_FOLDS:-0}"
FORCE_REBUILD_OOF_CONTENT="${FORCE_REBUILD_OOF_CONTENT:-0}"
FORCE_REBUILD_CONTENT_EVAL="${FORCE_REBUILD_CONTENT_EVAL:-0}"

CONTENT_CANDIDATE_TOP_K="${CONTENT_CANDIDATE_TOP_K:-1000}"
CONTENT_NEGATIVES_PER_BAND="${CONTENT_NEGATIVES_PER_BAND:-10}"
CONTENT_MAX_NEGATIVES_PER_QID="${CONTENT_MAX_NEGATIVES_PER_QID:-64}"
CONTENT_EPOCHS="${CONTENT_EPOCHS:-80}"
CONTENT_LEARNING_RATE="${CONTENT_LEARNING_RATE:-0.01}"
CONTENT_WEIGHT_DECAY="${CONTENT_WEIGHT_DECAY:-1e-4}"
CONTENT_BATCH_SIZE="${CONTENT_BATCH_SIZE:-65536}"
CONTENT_POSITIVE_WEIGHT_CAP="${CONTENT_POSITIVE_WEIGHT_CAP:-20}"
CONTENT_INFERENCE_MODE="${CONTENT_INFERENCE_MODE:-blend_rerank}"
CONTENT_AUTO_TUNE_BLEND_ALPHA="${CONTENT_AUTO_TUNE_BLEND_ALPHA:-1}"
CONTENT_BLEND_ALPHA="${CONTENT_BLEND_ALPHA:-0.30}"
CONTENT_TUNE_FRACTION="${CONTENT_TUNE_FRACTION:-0.20}"
CONTENT_TUNE_HIT_K="${CONTENT_TUNE_HIT_K:-5}"
CONTENT_TUNE_BLEND_ALPHA_GRID="${CONTENT_TUNE_BLEND_ALPHA_GRID:-0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50}"

require_file() {
  local name="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    echo "missing_${name}: $path" >&2
    exit 1
  fi
}

add_source_pair_if_exists() {
  local label="$1"
  local train_path="$2"
  local eval_path="$3"
  if [[ -f "$train_path" && -f "$eval_path" ]]; then
    content_train_source_args+=(--train-source "$label=$train_path")
    content_eval_source_args+=(--eval-source "$label=$eval_path")
  else
    [[ -f "$train_path" ]] || echo "skip_missing_train_source_${label}=$train_path" >&2
    [[ -f "$eval_path" ]] || echo "skip_missing_eval_source_${label}=$eval_path" >&2
  fi
}

add_apply_source_if_exists() {
  local label="$1"
  local path="$2"
  if [[ -f "$path" ]]; then
    apply_eval_source_args+=(--source "$label=$path")
  else
    echo "skip_missing_apply_eval_source_${label}=$path" >&2
  fi
}

mkdir -p "$OUT_DIR" "$FOLD_DIR" "$CONTENT_FOLD_DIR"
require_file train_gold "$TRAIN_GOLD"
require_file eval_gold "$EVAL_GOLD"
require_file train_page_text_jsonl "$TRAIN_PAGE_TEXT_JSONL"
require_file eval_page_text_jsonl "$EVAL_PAGE_TEXT_JSONL"
require_file train_gpp_no_hyperlink_pred "$TRAIN_GPP_NO_HYPERLINK_PRED"
require_file eval_gpp_no_hyperlink_pred "$EVAL_GPP_NO_HYPERLINK_PRED"

content_train_source_args=()
content_eval_source_args=()
add_source_pair_if_exists splade "$TRAIN_SPLADE_PRED" "$TRAIN_SPLADE_PRED"
add_source_pair_if_exists gpp_no_hyperlink "$TRAIN_GPP_NO_HYPERLINK_PRED" "$TRAIN_GPP_NO_HYPERLINK_PRED"
add_source_pair_if_exists gpp_doc_hyperlink "$TRAIN_GPP_DOC_HYPERLINK_PRED" "$TRAIN_GPP_DOC_HYPERLINK_PRED"
add_source_pair_if_exists gpp_page_hyperlink "$TRAIN_GPP_PAGE_HYPERLINK_PRED" "$TRAIN_GPP_PAGE_HYPERLINK_PRED"

apply_eval_source_args=()
add_apply_source_if_exists splade "$EVAL_SPLADE_PRED"
add_apply_source_if_exists gpp_no_hyperlink "$EVAL_GPP_NO_HYPERLINK_PRED"
add_apply_source_if_exists gpp_doc_hyperlink "$EVAL_GPP_DOC_HYPERLINK_PRED"
add_apply_source_if_exists gpp_page_hyperlink "$EVAL_GPP_PAGE_HYPERLINK_PRED"

if [[ "$FORCE_REBUILD_FOLDS" == "1" || ! -f "$FOLD_DIR/mmqa_train_oof_folds.summary.json" ]]; then
  echo
  echo "== Create deterministic train folds =="
  "$PYTHON_BIN" "$REPO_ROOT/scripts/create_jsonl_folds.py" \
    --input-jsonl "$TRAIN_GOLD" \
    --out-dir "$FOLD_DIR" \
    --label mmqa_train_oof \
    --fold-count "$OOF_FOLD_COUNT" \
    --seed "$OOF_SEED" \
    --summary-json "$FOLD_DIR/mmqa_train_oof_folds.summary.json"
else
  echo "reuse_folds_summary=$FOLD_DIR/mmqa_train_oof_folds.summary.json"
fi

fold_specs=()
auto_tune_args=()
if [[ "$CONTENT_AUTO_TUNE_BLEND_ALPHA" == "1" ]]; then
  auto_tune_args+=(--auto-tune-blend-alpha)
fi

for ((fold_idx=0; fold_idx<OOF_FOLD_COUNT; fold_idx++)); do
  fold_train="$FOLD_DIR/mmqa_train_oof_fold${fold_idx}_train.jsonl"
  fold_heldout="$FOLD_DIR/mmqa_train_oof_fold${fold_idx}_heldout.jsonl"
  fold_label="mmqa_train_content_aware_oof_fold${fold_idx}"
  fold_pred="$CONTENT_FOLD_DIR/${fold_label}.heldout.prediction.json"
  require_file "fold${fold_idx}_train" "$fold_train"
  require_file "fold${fold_idx}_heldout" "$fold_heldout"

  if [[ "$FORCE_REBUILD_OOF_CONTENT" == "1" || ! -f "$fold_pred" ]]; then
    echo
    echo "== Content-aware OOF fold $fold_idx/$((OOF_FOLD_COUNT - 1)) =="
    "$PYTHON_BIN" "$REPO_ROOT/scripts/train_content_aware_pseudo_page_reranker.py" \
      --train-gold "$fold_train" \
      --eval-gold "$fold_heldout" \
      --train-base-pred "$TRAIN_GPP_NO_HYPERLINK_PRED" \
      --eval-base-pred "$TRAIN_GPP_NO_HYPERLINK_PRED" \
      --train-page-text-jsonl "$TRAIN_PAGE_TEXT_JSONL" \
      --eval-page-text-jsonl "$TRAIN_PAGE_TEXT_JSONL" \
      "${content_train_source_args[@]}" \
      "${content_eval_source_args[@]}" \
      --candidate-top-k "$CONTENT_CANDIDATE_TOP_K" \
      --negatives-per-band "$CONTENT_NEGATIVES_PER_BAND" \
      --max-negatives-per-qid "$CONTENT_MAX_NEGATIVES_PER_QID" \
      --epochs "$CONTENT_EPOCHS" \
      --learning-rate "$CONTENT_LEARNING_RATE" \
      --weight-decay "$CONTENT_WEIGHT_DECAY" \
      --batch-size "$CONTENT_BATCH_SIZE" \
      --positive-weight-cap "$CONTENT_POSITIVE_WEIGHT_CAP" \
      --seed "$((OOF_SEED + fold_idx))" \
      --inference-mode "$CONTENT_INFERENCE_MODE" \
      --blend-alpha "$CONTENT_BLEND_ALPHA" \
      "${auto_tune_args[@]}" \
      --tune-fraction "$CONTENT_TUNE_FRACTION" \
      --tune-hit-k "$CONTENT_TUNE_HIT_K" \
      --tune-blend-alpha-grid "$CONTENT_TUNE_BLEND_ALPHA_GRID" \
      --restrict-eval-to-gold-qids \
      --output-model-json "$CONTENT_FOLD_DIR/${fold_label}.model.json" \
      --output-prediction-json "$fold_pred" \
      --output-summary-json "$CONTENT_FOLD_DIR/${fold_label}.summary.json" \
      --output-table-md "$CONTENT_FOLD_DIR/${fold_label}.table.md" \
      --output-eval-prior-jsonl "$CONTENT_FOLD_DIR/${fold_label}.heldout.prior.jsonl"
  else
    echo "reuse_content_oof_fold${fold_idx}=$fold_pred"
  fi
  fold_specs+=(--fold "$fold_heldout=$fold_pred")
done

if [[ "$FORCE_REBUILD_OOF_CONTENT" == "1" || ! -f "$OOF_TRAIN_PRED" ]]; then
  echo
  echo "== Merge OOF content-aware train predictions =="
  "$PYTHON_BIN" "$REPO_ROOT/scripts/merge_oof_predictions.py" \
    "${fold_specs[@]}" \
    --expected-jsonl "$TRAIN_GOLD" \
    --output-prediction-json "$OOF_TRAIN_PRED" \
    --output-summary-json "$OUT_DIR/mmqa_train_content_aware_oof.merge_summary.json"
else
  echo "reuse_oof_train_pred=$OOF_TRAIN_PRED"
fi
require_file oof_train_pred "$OOF_TRAIN_PRED"

if [[ "$FORCE_REBUILD_CONTENT_EVAL" == "1" || ! -f "$CONTENT_EVAL_PRED" ]]; then
  require_file content_model_json "$CONTENT_MODEL_JSON"
  CONTENT_EVAL_PRED="$OUT_DIR/${CONTENT_AWARE_LABEL}.dev.prediction.json"
  echo
  echo "== Build dev content-aware base from full-train model =="
  "$PYTHON_BIN" "$REPO_ROOT/scripts/apply_trained_content_aware_page_reranker.py" \
    --model-json "$CONTENT_MODEL_JSON" \
    --base-pred "$EVAL_GPP_NO_HYPERLINK_PRED" \
    --page-text-jsonl "$EVAL_PAGE_TEXT_JSONL" \
    --gold "$EVAL_GOLD" \
    "${apply_eval_source_args[@]}" \
    --candidate-top-k "$CONTENT_CANDIDATE_TOP_K" \
    --output-prediction-json "$CONTENT_EVAL_PRED" \
    --output-summary-json "$OUT_DIR/${CONTENT_AWARE_LABEL}.dev.summary.json" \
    --output-table-md "$OUT_DIR/${CONTENT_AWARE_LABEL}.dev.table.md" \
    --output-prior-jsonl "$OUT_DIR/${CONTENT_AWARE_LABEL}.dev.prior.jsonl"
else
  echo "reuse_content_eval_pred=$CONTENT_EVAL_PRED"
fi
require_file content_eval_pred "$CONTENT_EVAL_PRED"

eval_args=(
  --run "gpp_no_hyperlink=$EVAL_GPP_NO_HYPERLINK_PRED"
  --run "content_aware=$CONTENT_EVAL_PRED"
)

for insert_rank in $RUN_INSERT_RANKS; do
  repair_hit_k="${REPAIR_HIT_K:-$insert_rank}"
  promotion_rank_min="${PROMOTION_RANK_MIN:-$((insert_rank + 1))}"
  label="mmqa_train_to_dev_content_aware_counterfactual_oof_insert${insert_rank}"
  run_out_dir="$OUT_DIR/insert_rank${insert_rank}"

  echo
  echo "== OOF counterfactual repair: insert rank $insert_rank =="
  TRAIN_GOLD="$TRAIN_GOLD" \
  EVAL_GOLD="$EVAL_GOLD" \
  TRAIN_PAGE_TEXT_JSONL="$TRAIN_PAGE_TEXT_JSONL" \
  EVAL_PAGE_TEXT_JSONL="$EVAL_PAGE_TEXT_JSONL" \
  TRAIN_BASE_PRED="$OOF_TRAIN_PRED" \
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
  THRESHOLD_GRID="${THRESHOLD_GRID:-0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70,0.80}" \
  OUT_DIR="$run_out_dir" \
  LABEL="$label" \
  bash "$REPO_ROOT/examples/run_m3docvqa_counterfactual_page_promotion.sh"

  pred="$run_out_dir/${label}.dev.prediction.json"
  if [[ -f "$pred" ]]; then
    eval_args+=(--run "oof_hybrid_insert${insert_rank}=$pred")
  fi
done

"$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_pseudo_page_retrieval.py" \
  --gold "$EVAL_GOLD" \
  "${eval_args[@]}" \
  --format markdown \
  --output "$OUT_DIR/oof_hybrid_eval.md"
"$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_pseudo_page_retrieval.py" \
  --gold "$EVAL_GOLD" \
  "${eval_args[@]}" \
  --format csv \
  --output "$OUT_DIR/oof_hybrid_eval.csv"

single_group="$REPO_ROOT/output/m3docvqa_qid_groups/single_gold_doc.qids.txt"
multi_group="$REPO_ROOT/output/m3docvqa_qid_groups/multi_gold_doc.qids.txt"
if [[ -f "$single_group" && -f "$multi_group" ]]; then
  "$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_pseudo_page_retrieval.py" \
    --gold "$EVAL_GOLD" \
    --group "single_gold_doc=$single_group" \
    --group "multi_gold_doc=$multi_group" \
    "${eval_args[@]}" \
    --format markdown \
    --output "$OUT_DIR/oof_hybrid_group_audit.md"
  echo "saved_group_audit=$OUT_DIR/oof_hybrid_group_audit.md"
fi

echo "saved_oof_train_pred=$OOF_TRAIN_PRED"
echo "saved_eval_md=$OUT_DIR/oof_hybrid_eval.md"
echo "saved_eval_csv=$OUT_DIR/oof_hybrid_eval.csv"
