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
OUT_DIR="${OUT_DIR:-$REPO_ROOT/output/m3docvqa_doc_fusion_train_to_dev}"
LABEL="${LABEL:-mmqa_train_to_dev_doc_fusion}"

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

TRAIN_DENSE_PRED="${TRAIN_DENSE_PRED:-${M3DOCVQA_TRAIN_DENSE_PRED:-$(first_existing_path \
  "$CUSTOM_ROOT/outputs/mmqa_train_plain_top224_nprobe4_effdiag_all.prediction.json" \
  "$CUSTOM_ROOT/outputs/mmqa_train_plain_top224_nprobe4_effdiag_train.prediction.json" \
  "$REPO_ROOT/output/m3docvqa_plain_top224_mmqa_train/mmqa_train_plain_top224_nprobe4_effdiag_all.prediction.json" \
  || true)}}"
EVAL_DENSE_PRED="${EVAL_DENSE_PRED:-${M3DOCVQA_DEV_DENSE_PRED:-${M3DOCVQA_DENSE_PRED:-$(first_existing_path \
  "$CUSTOM_ROOT/outputs/mmqa_dev_plain_top224_nprobe4_effdiag_all.prediction.json" \
  "$REPO_ROOT/output/m3docvqa_plain_top224_mmqa_dev/mmqa_dev_plain_top224_nprobe4_effdiag_all.prediction.json" \
  || true)}}}"

TRAIN_SPLADE_PRED="${TRAIN_SPLADE_PRED:-${M3DOCVQA_TRAIN_SPLADE_PRED:-$(first_existing_path \
  "$CUSTOM_ROOT/outputs/m3docvqa_splade_mmqa_train/mmqa_train_splade.prediction.json" \
  "$REPO_ROOT/output/m3docvqa_splade_mmqa_train/mmqa_train_splade.prediction.json" \
  || true)}}"
EVAL_SPLADE_PRED="${EVAL_SPLADE_PRED:-${M3DOCVQA_DEV_SPLADE_PRED:-${M3DOCVQA_SPLADE_PRED:-$(first_existing_path \
  "$CUSTOM_ROOT/outputs/m3docvqa_splade_mmqa_dev/mmqa_dev_splade.prediction.json" \
  "$REPO_ROOT/output/m3docvqa_splade_mmqa_dev/mmqa_dev_splade.prediction.json" \
  || true)}}}"

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

add_source_pair_if_exists dense "$TRAIN_DENSE_PRED" "$EVAL_DENSE_PRED"
add_source_pair_if_exists splade "$TRAIN_SPLADE_PRED" "$EVAL_SPLADE_PRED"
add_source_pair_if_exists gpp_no_hyperlink "$TRAIN_GPP_NO_HYPERLINK_PRED" "$EVAL_GPP_NO_HYPERLINK_PRED"
add_source_pair_if_exists gpp_doc_hyperlink "$TRAIN_GPP_DOC_HYPERLINK_PRED" "$EVAL_GPP_DOC_HYPERLINK_PRED"
add_source_pair_if_exists gpp_page_hyperlink "$TRAIN_GPP_PAGE_HYPERLINK_PRED" "$EVAL_GPP_PAGE_HYPERLINK_PRED"

source_count=$(( ${#train_sources[@]} / 2 ))
if [[ "$source_count" -lt 4 ]]; then
  echo "warning_source_count=$source_count ; expected dense, splade, and GPP hyperlink sources" >&2
fi
if [[ "$source_count" -lt 2 ]]; then
  echo "not_enough_source_pairs=$source_count" >&2
  exit 1
fi

"$PYTHON_BIN" "$REPO_ROOT/scripts/tune_m3docvqa_doc_fusion_train_dev.py" \
  --train-gold "$TRAIN_GOLD" \
  --eval-gold "$EVAL_GOLD" \
  "${train_sources[@]}" \
  "${eval_sources[@]}" \
  --top-rows "${TOP_ROWS:-1000}" \
  --source-score-mode "${SOURCE_SCORE_MODE:-rank_score}" \
  --weight-grid "${WEIGHT_GRID:-0,0.25,0.5,1,2,4}" \
  --objective-k "${OBJECTIVE_K:-4}" \
  --output-summary-json "$OUT_DIR/${LABEL}.summary.json" \
  --output-table-md "$OUT_DIR/${LABEL}.table.md" \
  --output-prediction-json "$OUT_DIR/${LABEL}.dev.prediction.json"
