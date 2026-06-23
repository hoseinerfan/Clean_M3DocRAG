#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/m3docvqa_internal_env.sh"

MODEL_NAME="${MODEL_NAME:-naver/splade-v3}"
ENCODER_BACKEND="${ENCODER_BACKEND:-sentence-transformers}"
INDEX_OUT_DIR="${INDEX_OUT_DIR:-$LOCAL_OUTPUT_DIR/m3docvqa_splade_v3}"
RETR_OUT_DIR="${RETR_OUT_DIR:-$LOCAL_OUTPUT_DIR/m3docvqa_splade_v3_mmqa_dev}"
REPORT_DIR="${REPORT_DIR:-$REPO_ROOT/output/m3docvqa_splade_v3_ablation}"
INDEX_PT="${INDEX_PT:-$INDEX_OUT_DIR/m3docvqa_dev_splade_v3.pt}"
INDEX_SUMMARY="${INDEX_SUMMARY:-$INDEX_OUT_DIR/m3docvqa_dev_splade_v3.summary.json}"
V3_PRED="${V3_PRED:-$RETR_OUT_DIR/mmqa_dev_splade_v3.prediction.json}"
V2_PRED="${V2_PRED:-$LOCAL_OUTPUT_DIR/m3docvqa_splade_mmqa_dev/mmqa_dev_splade.prediction.json}"
STRICT_GOLD="${STRICT_GOLD:-$REPO_ROOT/output/m3docvqa_mmqa_pseudo_page_labels/mmqa_dev_pseudo_page_labels_strict.augmented_gold.jsonl}"

BUILD_INDEX="${BUILD_INDEX:-1}"
RUN_RETRIEVAL="${RUN_RETRIEVAL:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"

require_file() {
  local name="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    echo "missing_${name}: $path" >&2
    exit 1
  fi
}

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
from sentence_transformers import SparseEncoder
PY
then
  echo "missing_sentence_transformers_sparse_encoder" >&2
  echo "install_with: $PYTHON_BIN -m pip install -r $REPO_ROOT/requirements-splade-v3.txt" >&2
  exit 1
fi

if [[ "$SPLIT" != "dev" ]]; then
  echo "This controlled ablation expects SPLIT=dev, got SPLIT=$SPLIT" >&2
  exit 1
fi

require_file page_text "$LOCAL_OUTPUT_DIR/m3docvqa_page_text/m3docvqa_dev_page_text.jsonl"
require_file gold "$GOLD"
require_file qids "$QIDS_JSONL"
mkdir -p "$INDEX_OUT_DIR" "$RETR_OUT_DIR" "$REPORT_DIR"

if [[ "$BUILD_INDEX" == "1" ]]; then
  if [[ "$FORCE_RERUN" != "1" && -f "$INDEX_PT" ]]; then
    echo "reuse_index=$INDEX_PT"
  else
    SPLIT=dev \
    PAGE_TEXT_JSONL="$LOCAL_OUTPUT_DIR/m3docvqa_page_text/m3docvqa_dev_page_text.jsonl" \
    SPLADE_INDEX_OUT_DIR="$INDEX_OUT_DIR" \
    SPLADE_INDEX_LABEL=m3docvqa_dev_splade_v3 \
    SPLADE_MODEL_NAME_OR_PATH="$MODEL_NAME" \
    SPLADE_ENCODER_BACKEND="$ENCODER_BACKEND" \
    SPLADE_MAX_LENGTH=512 \
    SPLADE_TOPK_TERMS=128 \
    SPLADE_MIN_WEIGHT=0.0 \
    OUTPUT_INDEX_PT="$INDEX_PT" \
    OUTPUT_SUMMARY_JSON="$INDEX_SUMMARY" \
    bash "$REPO_ROOT/scripts/run_m3docvqa_build_splade_index.sh"
  fi
fi

require_file v3_index "$INDEX_PT"

if [[ "$RUN_RETRIEVAL" == "1" ]]; then
  if [[ "$FORCE_RERUN" != "1" && -f "$V3_PRED" ]]; then
    echo "reuse_prediction=$V3_PRED"
  else
    SPLIT=dev \
    SPLADE_RETR_OUT_DIR="$RETR_OUT_DIR" \
    SPLADE_RETR_LABEL=mmqa_dev_splade_v3 \
    SPLADE_INDEX_PT="$INDEX_PT" \
    SPLADE_MODEL_NAME_OR_PATH="$MODEL_NAME" \
    SPLADE_ENCODER_BACKEND="$ENCODER_BACKEND" \
    SPLADE_QUERY_MAX_LENGTH=64 \
    SPLADE_QUERY_TOPK_TERMS=32 \
    SPLADE_QUERY_MIN_WEIGHT=0.0 \
    TOP_PAGES=1000 \
    bash "$REPO_ROOT/scripts/run_m3docvqa_splade_retrieval.sh"
  fi
fi

if [[ "$RUN_EVAL" == "1" ]]; then
  require_file strict_gold "$STRICT_GOLD"
  require_file splade_v2_prediction "$V2_PRED"
  require_file splade_v3_prediction "$V3_PRED"
  "$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_pseudo_page_retrieval.py" \
    --gold "$STRICT_GOLD" \
    --recall-k 4 10 100 \
    --run "splade_pp=$V2_PRED" \
    --run "splade_v3=$V3_PRED" \
    --output "$REPORT_DIR/splade_pp_vs_v3_on_strict_gold.md"
  echo "saved_report=$REPORT_DIR/splade_pp_vs_v3_on_strict_gold.md"
fi
