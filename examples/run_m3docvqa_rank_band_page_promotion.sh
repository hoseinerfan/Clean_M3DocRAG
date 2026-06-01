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
TRAIN_GOLD="${TRAIN_GOLD:-$REPO_ROOT/data/m3-docvqa/multimodalqa/MMQA_train.jsonl}"
EVAL_GOLD="${EVAL_GOLD:-$REPO_ROOT/data/m3-docvqa/multimodalqa/MMQA_dev.jsonl}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/output/m3docvqa_rank_band_page_promotion}"
LABEL="${LABEL:-mmqa_train_to_dev_rank_band_page_promotion}"

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

TRAIN_DENSE_PRED="${TRAIN_DENSE_PRED:-${M3DOCVQA_TRAIN_DENSE_PRED:-$(first_existing_path \
  "$CUSTOM_ROOT/outputs/mmqa_train_plain_top224_nprobe4_effdiag_all.prediction.json" \
  "$CUSTOM_ROOT/outputs/mmqa_train_plain_top224_nprobe4_effdiag_train.prediction.json" \
  "$REPO_ROOT/output/m3docvqa_plain_top224_mmqa_train/mmqa_train_plain_top224_nprobe4_effdiag_all.prediction.json" \
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

GPP_TRAIN_OUT_DIR="${GPP_TRAIN_OUT_DIR:-$REPO_ROOT/output/m3docvqa_gpp_hyperlink_node_ablation_mmr_target1_train}"
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

echo "using_train_gold=$TRAIN_GOLD"
echo "using_eval_gold=$EVAL_GOLD"
echo "using_train_dense_pred=$TRAIN_DENSE_PRED"
echo "using_eval_dense_pred=$EVAL_DENSE_PRED"
echo "using_out_dir=$OUT_DIR"

"$PYTHON_BIN" "$REPO_ROOT/scripts/train_rank_band_page_promotion_reranker.py" \
  --train-gold "$TRAIN_GOLD" \
  --eval-gold "$EVAL_GOLD" \
  --train-base-pred "$TRAIN_DENSE_PRED" \
  --eval-base-pred "$EVAL_DENSE_PRED" \
  "${train_sources[@]}" \
  "${eval_sources[@]}" \
  --candidate-top-k "${CANDIDATE_TOP_K:-1000}" \
  --anchor-top-k "${ANCHOR_TOP_K:-4}" \
  --promotion-rank-min "${PROMOTION_RANK_MIN:-5}" \
  --promotion-rank-max "${PROMOTION_RANK_MAX:-500}" \
  --positive-scope "${POSITIVE_SCOPE:-doc}" \
  --negatives-per-band "${NEGATIVES_PER_BAND:-8}" \
  --max-negatives-per-qid "${MAX_NEGATIVES_PER_QID:-48}" \
  --epochs "${EPOCHS:-120}" \
  --learning-rate "${LEARNING_RATE:-0.03}" \
  --weight-decay "${WEIGHT_DECAY:-1e-4}" \
  --pair-batch-size "${PAIR_BATCH_SIZE:-65536}" \
  --seed "${SEED:-13}" \
  --inference-mode "${INFERENCE_MODE:-safe_promote}" \
  --max-promotions-per-qid "${MAX_PROMOTIONS_PER_QID:-2}" \
  --promotion-margin "${PROMOTION_MARGIN:-0.0}" \
  --output-model-json "$OUT_DIR/${LABEL}.model.json" \
  --output-prediction-json "$OUT_DIR/${LABEL}.dev.prediction.json" \
  --output-summary-json "$OUT_DIR/${LABEL}.summary.json" \
  --output-table-md "$OUT_DIR/${LABEL}.table.md" \
  --output-eval-prior-jsonl "$OUT_DIR/${LABEL}.dev.learned_prior.jsonl"
