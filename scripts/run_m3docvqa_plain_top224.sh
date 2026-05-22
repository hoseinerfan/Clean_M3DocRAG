#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/m3docvqa_internal_env.sh"

TOP_PAGES="${TOP_PAGES:-1000}"
BASE_ONLY_PAGE_BATCH_SIZE="${BASE_ONLY_PAGE_BATCH_SIZE:-64}"
PLAIN_TOP224_OUT_DIR="${PLAIN_TOP224_OUT_DIR:-$LOCAL_OUTPUT_DIR/m3docvqa_plain_top224_mmqa_${SPLIT}}"
PLAIN_TOP224_LABEL="${PLAIN_TOP224_LABEL:-mmqa_${SPLIT}_plain_top224_nprobe${FAISS_NPROBE}_effdiag_all}"
BASELINE_PRED="${BASELINE_PRED:-$LOCAL_OUTPUT_DIR/m3docvqa_baseline_mmqa_${SPLIT}/mmqa_${SPLIT}_baseline_ret${TOP_PAGES}_${FAISS_INDEX_TYPE}_nprobe${FAISS_NPROBE}.prediction.json}"
OUTPUT_JSONL="${OUTPUT_JSONL:-$PLAIN_TOP224_OUT_DIR/${PLAIN_TOP224_LABEL}.jsonl}"
OUTPUT_SUMMARY_JSON="${OUTPUT_SUMMARY_JSON:-$PLAIN_TOP224_OUT_DIR/${PLAIN_TOP224_LABEL}.summary.json}"
OUTPUT_PREDICTION_JSON="${OUTPUT_PREDICTION_JSON:-$PLAIN_TOP224_OUT_DIR/${PLAIN_TOP224_LABEL}.prediction.json}"

mkdir -p "$PLAIN_TOP224_OUT_DIR"

EMPTY_QUERY_LABELS="$PLAIN_TOP224_OUT_DIR/empty_query_token_labels.json"
EMPTY_PATCH_LABELS="$PLAIN_TOP224_OUT_DIR/empty_patch_labels.jsonl"
printf '{}\n' > "$EMPTY_QUERY_LABELS"
: > "$EMPTY_PATCH_LABELS"

echo "using_local_data_dir=$LOCAL_DATA_DIR"
echo "using_local_embeddings_dir=$LOCAL_EMBEDDINGS_DIR"
echo "using_qids_jsonl=$QIDS_JSONL"
echo "using_gold=$GOLD"
echo "using_baseline_pred=$BASELINE_PRED"
echo "using_embedding_name=$EMBEDDING_NAME"
echo "using_output_prediction_json=$OUTPUT_PREDICTION_JSON"
echo "using_output_summary_json=$OUTPUT_SUMMARY_JSON"

"$PYTHON_BIN" "$REPO_ROOT/scripts/run_visual_rerank_batch.py" \
  --qid-jsonl "$QIDS_JSONL" \
  --gold "$GOLD" \
  --baseline-pred "$BASELINE_PRED" \
  --data-name "$DATA_NAME" \
  --split "$SPLIT" \
  --embedding_name "$EMBEDDING_NAME" \
  --from-baseline-top-pages "$TOP_PAGES" \
  --base-score-source approx_page_maxsim_topk \
  --approx-base-page-token-topk 224 \
  --approx-base-page-token-scorer query_mean \
  --approx-base-page-token-selector global_topk \
  --approx-base-page-token-coarse-dtype fp32 \
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
