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
OUT_DIR="${OUT_DIR:-$REPO_ROOT/output/m3docvqa_content_aware_feature_ablation}"
TRAIN_GOLD="${TRAIN_GOLD:-$REPO_ROOT/output/m3docvqa_mmqa_pseudo_page_labels/mmqa_train_pseudo_page_labels_strict.augmented_gold.jsonl}"
EVAL_GOLD="${EVAL_GOLD:-$REPO_ROOT/output/m3docvqa_mmqa_pseudo_page_labels/mmqa_dev_pseudo_page_labels_strict.augmented_gold.jsonl}"
TRAIN_PAGE_TEXT_JSONL="${TRAIN_PAGE_TEXT_JSONL:-${M3DOCVQA_TRAIN_PAGE_TEXT_JSONL:-$CUSTOM_ROOT/outputs/m3docvqa_page_text/m3docvqa_train_page_text.jsonl}}"
EVAL_PAGE_TEXT_JSONL="${EVAL_PAGE_TEXT_JSONL:-${M3DOCVQA_DEV_PAGE_TEXT_JSONL:-${M3DOCVQA_PAGE_TEXT_JSONL:-$CUSTOM_ROOT/outputs/m3docvqa_page_text/m3docvqa_dev_page_text.jsonl}}}"

GPP_TRAIN_OUT_DIR="${GPP_TRAIN_OUT_DIR:-$REPO_ROOT/output/m3docvqa_gpp_hyperlink_node_ablation_mmr_target1_train_real}"
GPP_EVAL_OUT_DIR="${GPP_EVAL_OUT_DIR:-$REPO_ROOT/output/m3docvqa_gpp_hyperlink_node_ablation_mmr_target1}"
BASE_LABEL="${BASE_LABEL:-gpp_no_hyperlink}"
BASE_TRAIN_PRED="${BASE_TRAIN_PRED:-$GPP_TRAIN_OUT_DIR/mmqa_train_gpp_hyperlink_node_no_hyperlink.prediction.json}"
BASE_EVAL_PRED="${BASE_EVAL_PRED:-$GPP_EVAL_OUT_DIR/mmqa_dev_gpp_hyperlink_node_no_hyperlink.prediction.json}"

FEATURE_SETS="${FEATURE_SETS:-rank_only rank_source rank_structure rank_source_structure content_only no_content no_source no_structure all}"
SOURCE_SET="${SOURCE_SET:-all}"
AUTO_TUNE_BLEND_ALPHA="${AUTO_TUNE_BLEND_ALPHA:-1}"
TUNE_HIT_K="${TUNE_HIT_K:-5}"
TUNE_FRACTION="${TUNE_FRACTION:-0.20}"
TUNE_BLEND_ALPHA_GRID="${TUNE_BLEND_ALPHA_GRID:-0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50}"
EPOCHS="${EPOCHS:-80}"
FORCE_RERUN="${FORCE_RERUN:-0}"

require_file() {
  local name="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    echo "missing_${name}: $path" >&2
    exit 1
  fi
}

add_eval_run() {
  local label="$1"
  local path="$2"
  if [[ -f "$path" ]]; then
    eval_args+=(--run "$label=$path")
  else
    echo "eval_skip_missing_${label}: $path" >&2
  fi
}

mkdir -p "$OUT_DIR"
require_file train_gold "$TRAIN_GOLD"
require_file eval_gold "$EVAL_GOLD"
require_file train_page_text_jsonl "$TRAIN_PAGE_TEXT_JSONL"
require_file eval_page_text_jsonl "$EVAL_PAGE_TEXT_JSONL"
require_file base_train_pred "$BASE_TRAIN_PRED"
require_file base_eval_pred "$BASE_EVAL_PRED"

eval_args=()
add_eval_run "$BASE_LABEL" "$BASE_EVAL_PRED"

for feature_set in $FEATURE_SETS; do
  label="mmqa_train_to_dev_content_aware_feature_${feature_set}_base_${BASE_LABEL}"
  pred="$OUT_DIR/${label}.dev.prediction.json"
  if [[ "$FORCE_RERUN" != "1" && -f "$pred" ]]; then
    echo "reuse_${feature_set}=$pred"
  else
    echo
    echo "== M3DocVQA content-aware feature ablation: $feature_set =="
    TRAIN_GOLD="$TRAIN_GOLD" \
    EVAL_GOLD="$EVAL_GOLD" \
    TRAIN_PAGE_TEXT_JSONL="$TRAIN_PAGE_TEXT_JSONL" \
    EVAL_PAGE_TEXT_JSONL="$EVAL_PAGE_TEXT_JSONL" \
    TRAIN_DENSE_PRED="$BASE_TRAIN_PRED" \
    EVAL_DENSE_PRED="$BASE_EVAL_PRED" \
    FEATURE_SET="$feature_set" \
    SOURCE_SET="$SOURCE_SET" \
    AUTO_TUNE_BLEND_ALPHA="$AUTO_TUNE_BLEND_ALPHA" \
    TUNE_HIT_K="$TUNE_HIT_K" \
    TUNE_FRACTION="$TUNE_FRACTION" \
    TUNE_BLEND_ALPHA_GRID="$TUNE_BLEND_ALPHA_GRID" \
    EPOCHS="$EPOCHS" \
    OUT_DIR="$OUT_DIR" \
    LABEL="$label" \
    bash "$REPO_ROOT/examples/run_m3docvqa_content_aware_pseudo_page_reranker.sh"
  fi
  add_eval_run "feature_${feature_set}" "$pred"
done

"$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_pseudo_page_retrieval.py" \
  --gold "$EVAL_GOLD" \
  "${eval_args[@]}" \
  --format markdown \
  --output "$OUT_DIR/feature_ablation_eval.md"
"$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_pseudo_page_retrieval.py" \
  --gold "$EVAL_GOLD" \
  "${eval_args[@]}" \
  --format csv \
  --output "$OUT_DIR/feature_ablation_eval.csv"

single_group="$REPO_ROOT/output/m3docvqa_qid_groups/single_gold_doc.qids.txt"
multi_group="$REPO_ROOT/output/m3docvqa_qid_groups/multi_gold_doc.qids.txt"
if [[ -f "$single_group" && -f "$multi_group" ]]; then
  "$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_pseudo_page_retrieval.py" \
    --gold "$EVAL_GOLD" \
    --group "single_gold_doc=$single_group" \
    --group "multi_gold_doc=$multi_group" \
    "${eval_args[@]}" \
    --format markdown \
    --output "$OUT_DIR/feature_ablation_group_eval.md"
  echo "saved_group_eval=$OUT_DIR/feature_ablation_group_eval.md"
fi

echo "saved_eval_md=$OUT_DIR/feature_ablation_eval.md"
echo "saved_eval_csv=$OUT_DIR/feature_ablation_eval.csv"
