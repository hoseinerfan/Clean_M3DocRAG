#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/m3docvqa_internal_env.sh"

SPLADE_RETR_OUT_DIR="${SPLADE_RETR_OUT_DIR:-$LOCAL_OUTPUT_DIR/m3docvqa_splade_mmqa_${SPLIT}}"
SPLADE_INDEX_PT="${SPLADE_INDEX_PT:-$LOCAL_OUTPUT_DIR/m3docvqa_splade/m3docvqa_${SPLIT}_splade.pt}"
SPLADE_MODEL_NAME_OR_PATH="${SPLADE_MODEL_NAME_OR_PATH:-naver/splade-cocondenser-ensembledistil}"
SPLADE_ENCODER_BACKEND="${SPLADE_ENCODER_BACKEND:-auto}"
SPLADE_BATCH_SIZE="${SPLADE_BATCH_SIZE:-8}"
SPLADE_QUERY_MAX_LENGTH="${SPLADE_QUERY_MAX_LENGTH:-64}"
SPLADE_QUERY_TOPK_TERMS="${SPLADE_QUERY_TOPK_TERMS:-32}"
SPLADE_QUERY_MIN_WEIGHT="${SPLADE_QUERY_MIN_WEIGHT:-0.0}"
SPLADE_DEVICE="${SPLADE_DEVICE:-auto}"
TOP_PAGES="${TOP_PAGES:-1000}"
SPLADE_RETR_LABEL="${SPLADE_RETR_LABEL:-mmqa_${SPLIT}_splade}"
OUTPUT_JSONL="${OUTPUT_JSONL:-$SPLADE_RETR_OUT_DIR/${SPLADE_RETR_LABEL}.jsonl}"
OUTPUT_PREDICTION_JSON="${OUTPUT_PREDICTION_JSON:-$SPLADE_RETR_OUT_DIR/${SPLADE_RETR_LABEL}.prediction.json}"
OUTPUT_SUMMARY_JSON="${OUTPUT_SUMMARY_JSON:-$SPLADE_RETR_OUT_DIR/${SPLADE_RETR_LABEL}.summary.json}"

mkdir -p "$SPLADE_RETR_OUT_DIR"

echo "using_qids_jsonl=$QIDS_JSONL"
echo "using_gold=$GOLD"
echo "using_splade_index_pt=$SPLADE_INDEX_PT"
echo "using_splade_model_name_or_path=$SPLADE_MODEL_NAME_OR_PATH"
echo "using_splade_encoder_backend=$SPLADE_ENCODER_BACKEND"
echo "using_output_prediction_json=$OUTPUT_PREDICTION_JSON"
echo "using_output_summary_json=$OUTPUT_SUMMARY_JSON"

"$PYTHON_BIN" "$REPO_ROOT/scripts/run_splade_page_retrieval.py" \
  --qid-jsonl "$QIDS_JSONL" \
  --gold "$GOLD" \
  --index-pt "$SPLADE_INDEX_PT" \
  --model-name-or-path "$SPLADE_MODEL_NAME_OR_PATH" \
  --encoder-backend "$SPLADE_ENCODER_BACKEND" \
  --batch-size "$SPLADE_BATCH_SIZE" \
  --max-length "$SPLADE_QUERY_MAX_LENGTH" \
  --query-topk-terms "$SPLADE_QUERY_TOPK_TERMS" \
  --query-min-weight "$SPLADE_QUERY_MIN_WEIGHT" \
  --device "$SPLADE_DEVICE" \
  --top-pages "$TOP_PAGES" \
  --output-jsonl "$OUTPUT_JSONL" \
  --output-prediction-json "$OUTPUT_PREDICTION_JSON" \
  --output-summary-json "$OUTPUT_SUMMARY_JSON"
