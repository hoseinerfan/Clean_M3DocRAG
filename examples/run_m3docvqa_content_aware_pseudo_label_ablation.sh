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
LABEL_DIR="${LABEL_DIR:-$REPO_ROOT/output/m3docvqa_mmqa_pseudo_page_labels}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/output/m3docvqa_content_aware_pseudo_label_ablation}"
TRAIN_PAGE_TEXT_JSONL="${TRAIN_PAGE_TEXT_JSONL:-${M3DOCVQA_TRAIN_PAGE_TEXT_JSONL:-$CUSTOM_ROOT/outputs/m3docvqa_page_text/m3docvqa_train_page_text.jsonl}}"
EVAL_PAGE_TEXT_JSONL="${EVAL_PAGE_TEXT_JSONL:-${M3DOCVQA_DEV_PAGE_TEXT_JSONL:-${M3DOCVQA_PAGE_TEXT_JSONL:-$CUSTOM_ROOT/outputs/m3docvqa_page_text/m3docvqa_dev_page_text.jsonl}}}"

STRICT_TRAIN_GOLD="${STRICT_TRAIN_GOLD:-$LABEL_DIR/mmqa_train_pseudo_page_labels_strict.augmented_gold.jsonl}"
STRICT_EVAL_GOLD="${STRICT_EVAL_GOLD:-$LABEL_DIR/mmqa_dev_pseudo_page_labels_strict.augmented_gold.jsonl}"
LOOSE_TRAIN_GOLD="${LOOSE_TRAIN_GOLD:-$LABEL_DIR/mmqa_train_pseudo_page_labels.augmented_gold.jsonl}"
LOOSE_EVAL_GOLD="${LOOSE_EVAL_GOLD:-$LABEL_DIR/mmqa_dev_pseudo_page_labels.augmented_gold.jsonl}"

GPP_TRAIN_OUT_DIR="${GPP_TRAIN_OUT_DIR:-$REPO_ROOT/output/m3docvqa_gpp_hyperlink_node_ablation_mmr_target1_train_real}"
GPP_EVAL_OUT_DIR="${GPP_EVAL_OUT_DIR:-$REPO_ROOT/output/m3docvqa_gpp_hyperlink_node_ablation_mmr_target1}"
BASE_LABEL="${BASE_LABEL:-gpp_no_hyperlink}"
BASE_TRAIN_PRED="${BASE_TRAIN_PRED:-$GPP_TRAIN_OUT_DIR/mmqa_train_gpp_hyperlink_node_no_hyperlink.prediction.json}"
BASE_EVAL_PRED="${BASE_EVAL_PRED:-$GPP_EVAL_OUT_DIR/mmqa_dev_gpp_hyperlink_node_no_hyperlink.prediction.json}"

BUILD_MISSING_LABELS="${BUILD_MISSING_LABELS:-1}"
PSEUDO_LABEL_VARIANTS="${PSEUDO_LABEL_VARIANTS:-strict loose}"
FEATURE_SET="${FEATURE_SET:-all}"
SOURCE_SET="${SOURCE_SET:-all}"
AUTO_TUNE_BLEND_ALPHA="${AUTO_TUNE_BLEND_ALPHA:-1}"
TUNE_HIT_K="${TUNE_HIT_K:-5}"
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

build_labels_if_missing() {
  local split="$1"
  local label="$2"
  local doc_pages="$3"
  local strict="$4"
  local augmented="$LABEL_DIR/${label}.augmented_gold.jsonl"
  if [[ -f "$augmented" || "$BUILD_MISSING_LABELS" != "1" ]]; then
    return 0
  fi

  echo
  echo "== Build missing pseudo-page labels: $label =="
  if [[ "$strict" == "1" ]]; then
    SPLIT="$split" \
    DOC_PAGES_JSONL="$doc_pages" \
    MIN_SCORE=8 \
    TOP_PAGES_PER_DOC=1 \
    TOP_PAGES_PER_QID=4 \
    LABEL="$label" \
    bash "$REPO_ROOT/examples/run_m3docvqa_mmqa_pseudo_page_labels.sh"
  else
    SPLIT="$split" \
    DOC_PAGES_JSONL="$doc_pages" \
    LABEL="$label" \
    bash "$REPO_ROOT/examples/run_m3docvqa_mmqa_pseudo_page_labels.sh"
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
require_file train_page_text_jsonl "$TRAIN_PAGE_TEXT_JSONL"
require_file eval_page_text_jsonl "$EVAL_PAGE_TEXT_JSONL"
require_file base_train_pred "$BASE_TRAIN_PRED"
require_file base_eval_pred "$BASE_EVAL_PRED"

build_labels_if_missing train mmqa_train_pseudo_page_labels "$TRAIN_PAGE_TEXT_JSONL" 0
build_labels_if_missing dev mmqa_dev_pseudo_page_labels "$EVAL_PAGE_TEXT_JSONL" 0
build_labels_if_missing train mmqa_train_pseudo_page_labels_strict "$TRAIN_PAGE_TEXT_JSONL" 1
build_labels_if_missing dev mmqa_dev_pseudo_page_labels_strict "$EVAL_PAGE_TEXT_JSONL" 1

require_file strict_train_gold "$STRICT_TRAIN_GOLD"
require_file strict_eval_gold "$STRICT_EVAL_GOLD"

eval_args=()
add_eval_run "$BASE_LABEL" "$BASE_EVAL_PRED"

for variant in $PSEUDO_LABEL_VARIANTS; do
  case "$variant" in
    strict)
      train_gold="$STRICT_TRAIN_GOLD"
      eval_gold="$STRICT_EVAL_GOLD"
      ;;
    loose)
      train_gold="$LOOSE_TRAIN_GOLD"
      eval_gold="$LOOSE_EVAL_GOLD"
      if [[ ! -f "$train_gold" || ! -f "$eval_gold" ]]; then
        echo "skip_loose_missing_labels train=$train_gold eval=$eval_gold" >&2
        continue
      fi
      ;;
    *)
      echo "unknown_pseudo_label_variant: $variant" >&2
      exit 1
      ;;
  esac

  label="mmqa_train_to_dev_content_aware_pseudolabel_${variant}_base_${BASE_LABEL}"
  pred="$OUT_DIR/${label}.dev.prediction.json"
  if [[ "$FORCE_RERUN" != "1" && -f "$pred" ]]; then
    echo "reuse_${variant}=$pred"
  else
    echo
    echo "== M3DocVQA content-aware pseudo-label ablation: $variant =="
    TRAIN_GOLD="$train_gold" \
    EVAL_GOLD="$eval_gold" \
    TRAIN_PAGE_TEXT_JSONL="$TRAIN_PAGE_TEXT_JSONL" \
    EVAL_PAGE_TEXT_JSONL="$EVAL_PAGE_TEXT_JSONL" \
    TRAIN_DENSE_PRED="$BASE_TRAIN_PRED" \
    EVAL_DENSE_PRED="$BASE_EVAL_PRED" \
    FEATURE_SET="$FEATURE_SET" \
    SOURCE_SET="$SOURCE_SET" \
    AUTO_TUNE_BLEND_ALPHA="$AUTO_TUNE_BLEND_ALPHA" \
    TUNE_HIT_K="$TUNE_HIT_K" \
    EPOCHS="$EPOCHS" \
    OUT_DIR="$OUT_DIR" \
    LABEL="$label" \
    bash "$REPO_ROOT/examples/run_m3docvqa_content_aware_pseudo_page_reranker.sh"
  fi
  add_eval_run "pseudolabel_${variant}" "$pred"
done

"$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_pseudo_page_retrieval.py" \
  --gold "$STRICT_EVAL_GOLD" \
  "${eval_args[@]}" \
  --format markdown \
  --output "$OUT_DIR/pseudo_label_ablation_eval_on_strict.md"
"$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_pseudo_page_retrieval.py" \
  --gold "$STRICT_EVAL_GOLD" \
  "${eval_args[@]}" \
  --format csv \
  --output "$OUT_DIR/pseudo_label_ablation_eval_on_strict.csv"

echo "saved_eval_md=$OUT_DIR/pseudo_label_ablation_eval_on_strict.md"
echo "saved_eval_csv=$OUT_DIR/pseudo_label_ablation_eval_on_strict.csv"
