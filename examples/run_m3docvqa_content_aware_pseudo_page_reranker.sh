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

CUSTOM_ROOT="${CUSTOM_ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/output/m3docvqa_content_aware_pseudo_page_reranker}"
LABEL="${LABEL:-mmqa_train_to_dev_content_aware_pseudo_page}"

first_existing_path() {
  local path
  for path in "$@"; do
    if [[ -n "$path" && -f "$path" ]]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
  return 1
}

require_file() {
  local name="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    echo "missing_${name}: $path" >&2
    exit 1
  fi
}

TRAIN_GOLD="${TRAIN_GOLD:-$REPO_ROOT/output/m3docvqa_mmqa_pseudo_page_labels/mmqa_train_pseudo_page_labels_strict.augmented_gold.jsonl}"
EVAL_GOLD="${EVAL_GOLD:-$REPO_ROOT/output/m3docvqa_mmqa_pseudo_page_labels/mmqa_dev_pseudo_page_labels_strict.augmented_gold.jsonl}"
TRAIN_PAGE_TEXT_JSONL="${TRAIN_PAGE_TEXT_JSONL:-${M3DOCVQA_TRAIN_PAGE_TEXT_JSONL:-$CUSTOM_ROOT/outputs/m3docvqa_page_text/m3docvqa_train_page_text.jsonl}}"
EVAL_PAGE_TEXT_JSONL="${EVAL_PAGE_TEXT_JSONL:-${M3DOCVQA_DEV_PAGE_TEXT_JSONL:-${M3DOCVQA_PAGE_TEXT_JSONL:-$CUSTOM_ROOT/outputs/m3docvqa_page_text/m3docvqa_dev_page_text.jsonl}}}"

TRAIN_DENSE_PRED="${TRAIN_DENSE_PRED:-${M3DOCVQA_TRAIN_DENSE_PRED:-$(first_existing_path \
  "$CUSTOM_ROOT/outputs/m3docvqa_baseline_mmqa_train/mmqa_train_baseline_ret1000_ivfflat_nprobe4.prediction.json" \
  "$REPO_ROOT/output/m3docvqa_baseline_mmqa_train/mmqa_train_baseline_ret1000_ivfflat_nprobe4.prediction.json" \
  || true)}}"
EVAL_DENSE_PRED="${EVAL_DENSE_PRED:-${M3DOCVQA_DEV_DENSE_PRED:-${M3DOCVQA_DENSE_PRED:-${DENSE_PRED:-$(first_existing_path \
  "$CUSTOM_ROOT/outputs/mmqa_dev_plain_top224_nprobe4_effdiag_all.prediction.json" \
  "$REPO_ROOT/output/m3docvqa_plain_top224_mmqa_dev/mmqa_dev_plain_top224_nprobe4_effdiag_all.prediction.json" \
  || true)}}}}"

TRAIN_SPLADE_PRED="${TRAIN_SPLADE_PRED:-${M3DOCVQA_TRAIN_SPLADE_PRED:-$(first_existing_path \
  "$CUSTOM_ROOT/outputs/m3docvqa_splade_mmqa_train/mmqa_train_splade.prediction.json" \
  "$REPO_ROOT/output/m3docvqa_splade_mmqa_train/mmqa_train_splade.prediction.json" \
  || true)}}"
EVAL_SPLADE_PRED="${EVAL_SPLADE_PRED:-${M3DOCVQA_DEV_SPLADE_PRED:-${M3DOCVQA_SPLADE_PRED:-${SPARSE_PRED:-$(first_existing_path \
  "$CUSTOM_ROOT/outputs/m3docvqa_splade_mmqa_dev/mmqa_dev_splade.prediction.json" \
  "$REPO_ROOT/output/m3docvqa_splade_mmqa_dev/mmqa_dev_splade.prediction.json" \
  || true)}}}}"

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

mkdir -p "$OUT_DIR"
require_file train_gold "$TRAIN_GOLD"
require_file eval_gold "$EVAL_GOLD"
require_file train_page_text_jsonl "$TRAIN_PAGE_TEXT_JSONL"
require_file eval_page_text_jsonl "$EVAL_PAGE_TEXT_JSONL"
require_file train_dense_pred "$TRAIN_DENSE_PRED"
require_file eval_dense_pred "$EVAL_DENSE_PRED"

train_sources=()
eval_sources=()
add_source_pair_if_exists() {
  local label="$1"
  local train_path="$2"
  local eval_path="$3"
  if [[ -f "$train_path" && -f "$eval_path" ]]; then
    train_sources+=(--train-source "$label=$train_path")
    eval_sources+=(--eval-source "$label=$eval_path")
  else
    [[ -f "$train_path" ]] || echo "skip_missing_train_source_${label}=$train_path" >&2
    [[ -f "$eval_path" ]] || echo "skip_missing_eval_source_${label}=$eval_path" >&2
  fi
}

add_source_pair_if_exists splade "$TRAIN_SPLADE_PRED" "$EVAL_SPLADE_PRED"
add_source_pair_if_exists gpp_no_hyperlink "$TRAIN_GPP_NO_HYPERLINK_PRED" "$EVAL_GPP_NO_HYPERLINK_PRED"
add_source_pair_if_exists gpp_doc_hyperlink "$TRAIN_GPP_DOC_HYPERLINK_PRED" "$EVAL_GPP_DOC_HYPERLINK_PRED"
add_source_pair_if_exists gpp_page_hyperlink "$TRAIN_GPP_PAGE_HYPERLINK_PRED" "$EVAL_GPP_PAGE_HYPERLINK_PRED"

auto_tune_args=()
if [[ "${AUTO_TUNE_BLEND_ALPHA:-0}" == "1" ]]; then
  auto_tune_args+=(--auto-tune-blend-alpha)
fi

echo "using_train_gold=$TRAIN_GOLD"
echo "using_eval_gold=$EVAL_GOLD"
echo "using_train_page_text_jsonl=$TRAIN_PAGE_TEXT_JSONL"
echo "using_eval_page_text_jsonl=$EVAL_PAGE_TEXT_JSONL"
echo "using_train_dense_pred=$TRAIN_DENSE_PRED"
echo "using_eval_dense_pred=$EVAL_DENSE_PRED"
echo "using_out_dir=$OUT_DIR"
echo "using_label=$LABEL"

"$PYTHON_BIN" "$REPO_ROOT/scripts/train_content_aware_pseudo_page_reranker.py" \
  --train-gold "$TRAIN_GOLD" \
  --eval-gold "$EVAL_GOLD" \
  --train-base-pred "$TRAIN_DENSE_PRED" \
  --eval-base-pred "$EVAL_DENSE_PRED" \
  --train-page-text-jsonl "$TRAIN_PAGE_TEXT_JSONL" \
  --eval-page-text-jsonl "$EVAL_PAGE_TEXT_JSONL" \
  "${train_sources[@]}" \
  "${eval_sources[@]}" \
  --candidate-top-k "${CANDIDATE_TOP_K:-1000}" \
  --negatives-per-band "${NEGATIVES_PER_BAND:-10}" \
  --max-negatives-per-qid "${MAX_NEGATIVES_PER_QID:-64}" \
  --epochs "${EPOCHS:-80}" \
  --learning-rate "${LEARNING_RATE:-0.01}" \
  --weight-decay "${WEIGHT_DECAY:-1e-4}" \
  --batch-size "${BATCH_SIZE:-65536}" \
  --positive-weight-cap "${POSITIVE_WEIGHT_CAP:-20}" \
  --seed "${SEED:-13}" \
  --inference-mode "${INFERENCE_MODE:-blend_rerank}" \
  --blend-alpha "${BLEND_ALPHA:-0.30}" \
  "${auto_tune_args[@]}" \
  --tune-fraction "${TUNE_FRACTION:-0.20}" \
  --tune-blend-alpha-grid "${TUNE_BLEND_ALPHA_GRID:-0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50}" \
  --tune-hit-k "${TUNE_HIT_K:-4}" \
  --anchor-top-k "${ANCHOR_TOP_K:-4}" \
  --promotion-rank-min "${PROMOTION_RANK_MIN:-5}" \
  --promotion-rank-max "${PROMOTION_RANK_MAX:-200}" \
  --max-promotions-per-qid "${MAX_PROMOTIONS_PER_QID:-2}" \
  --promotion-margin "${PROMOTION_MARGIN:-0.05}" \
  --output-model-json "$OUT_DIR/${LABEL}.model.json" \
  --output-prediction-json "$OUT_DIR/${LABEL}.dev.prediction.json" \
  --output-summary-json "$OUT_DIR/${LABEL}.summary.json" \
  --output-table-md "$OUT_DIR/${LABEL}.table.md" \
  --output-eval-prior-jsonl "$OUT_DIR/${LABEL}.dev.learned_prior.jsonl"
