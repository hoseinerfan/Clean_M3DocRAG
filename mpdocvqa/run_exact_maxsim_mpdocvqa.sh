#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

if [[ -f "$SCRIPT_DIR/env_hpc.sh" ]]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/env_hpc.sh"
fi

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/env/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

DATA_ROOT="${MPDOCVQA_DATA_ROOT:-$LOCAL_DATA_DIR/mpdocvqa}"
EMBEDDING_NAME="${MPDOCVQA_EMBEDDING_NAME:-colpali-v1.2_mpdocvqa_dev}"
BASELINE_PRED="${MPDOCVQA_BASELINE_PRED:-$LOCAL_OUTPUT_DIR/mpdocvqa/baseline_ret1000.json}"
OUT_DIR="${MPDOCVQA_OUT_DIR:-$LOCAL_OUTPUT_DIR/mpdocvqa}"
TOP_PAGES="${MPDOCVQA_TOP_PAGES:-1000}"
BASE_ONLY_PAGE_BATCH_SIZE="${BASE_ONLY_PAGE_BATCH_SIZE:-64}"

if [[ ! -f "$BASELINE_PRED" ]]; then
  echo "missing_baseline_pred: $BASELINE_PRED" >&2
  echo "hint: run mpdocvqa/sbatch_index_retrieve_mpdocvqa.sh first" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

EMPTY_QUERY_LABELS="$OUT_DIR/empty_query_token_labels.json"
EMPTY_PATCH_LABELS="$OUT_DIR/empty_patch_labels.jsonl"
printf '{}\n' > "$EMPTY_QUERY_LABELS"
: > "$EMPTY_PATCH_LABELS"

OUTPUT_JSONL="$OUT_DIR/exact_maxsim_ret${TOP_PAGES}.jsonl"
OUTPUT_SUMMARY_JSON="$OUT_DIR/exact_maxsim_ret${TOP_PAGES}_summary.json"
OUTPUT_PREDICTION_JSON="$OUT_DIR/exact_maxsim_ret${TOP_PAGES}_prediction.json"

echo "using_data_root=$DATA_ROOT"
echo "using_embedding_name=$EMBEDDING_NAME"
echo "using_baseline_pred=$BASELINE_PRED"
echo "using_top_pages=$TOP_PAGES"
echo "using_base_only_page_batch_size=$BASE_ONLY_PAGE_BATCH_SIZE"
echo "using_output_prediction_json=$OUTPUT_PREDICTION_JSON"
echo "using_output_summary_json=$OUTPUT_SUMMARY_JSON"

"$PYTHON_BIN" "$REPO_ROOT/scripts/run_visual_rerank_batch.py" \
  --qid-jsonl "$DATA_ROOT/qids_dev.jsonl" \
  --gold "$DATA_ROOT/MMQA_dev.jsonl" \
  --baseline-pred "$BASELINE_PRED" \
  --data-name mpdocvqa \
  --split dev \
  --embedding_name "$EMBEDDING_NAME" \
  --from-baseline-top-pages "$TOP_PAGES" \
  --base-score-source exact_page_maxsim \
  --approx-base-page-token-topk 0 \
  --weight-base 1.0 \
  --weight-visual 0.0 \
  --weight-non-visual 0.0 \
  --weight-balance 0.0 \
  --splice-query-token-labels "$EMPTY_QUERY_LABELS" \
  --splice-patch-labels-jsonl "$EMPTY_PATCH_LABELS" \
  --base-only-page-batch-size "$BASE_ONLY_PAGE_BATCH_SIZE" \
  --output-jsonl "$OUTPUT_JSONL" \
  --output-summary-json "$OUTPUT_SUMMARY_JSON" \
  --output-prediction-json "$OUTPUT_PREDICTION_JSON"

echo "saved_prediction=$OUTPUT_PREDICTION_JSON"
echo "saved_summary=$OUTPUT_SUMMARY_JSON"
