#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/m3docvqa_internal_env.sh"

TOP_PAGES="${TOP_PAGES:-1000}"
BASE_ONLY_PAGE_BATCH_SIZE="${BASE_ONLY_PAGE_BATCH_SIZE:-64}"
EXACT_MAXSIM_OUT_DIR="${EXACT_MAXSIM_OUT_DIR:-$LOCAL_OUTPUT_DIR/m3docvqa_exact_maxsim_mmqa_${SPLIT}}"
EXACT_MAXSIM_LABEL="${EXACT_MAXSIM_LABEL:-mmqa_${SPLIT}_exact_maxsim_nprobe${FAISS_NPROBE}_ret${TOP_PAGES}}"
BASELINE_PRED="${BASELINE_PRED:-$LOCAL_OUTPUT_DIR/m3docvqa_baseline_mmqa_${SPLIT}/mmqa_${SPLIT}_baseline_ret${TOP_PAGES}_${FAISS_INDEX_TYPE}_nprobe${FAISS_NPROBE}.prediction.json}"
OUTPUT_JSONL="${OUTPUT_JSONL:-$EXACT_MAXSIM_OUT_DIR/${EXACT_MAXSIM_LABEL}.jsonl}"
OUTPUT_SUMMARY_JSON="${OUTPUT_SUMMARY_JSON:-$EXACT_MAXSIM_OUT_DIR/${EXACT_MAXSIM_LABEL}.summary.json}"
OUTPUT_PREDICTION_JSON="${OUTPUT_PREDICTION_JSON:-$EXACT_MAXSIM_OUT_DIR/${EXACT_MAXSIM_LABEL}.prediction.json}"

if [[ ! -f "$BASELINE_PRED" ]]; then
  echo "missing_baseline_pred: $BASELINE_PRED" >&2
  echo "hint: run SPLIT=$SPLIT N_RETRIEVAL_PAGES=$TOP_PAGES FAISS_NPROBE=$FAISS_NPROBE bash scripts/run_m3docvqa_baseline_retrieval.sh first" >&2
  exit 1
fi

mkdir -p "$EXACT_MAXSIM_OUT_DIR"

EMPTY_QUERY_LABELS="$EXACT_MAXSIM_OUT_DIR/empty_query_token_labels.json"
EMPTY_PATCH_LABELS="$EXACT_MAXSIM_OUT_DIR/empty_patch_labels.jsonl"
printf '{}\n' > "$EMPTY_QUERY_LABELS"
: > "$EMPTY_PATCH_LABELS"

echo "using_local_data_dir=$LOCAL_DATA_DIR"
echo "using_local_embeddings_dir=$LOCAL_EMBEDDINGS_DIR"
echo "using_qids_jsonl=$QIDS_JSONL"
echo "using_gold=$GOLD"
echo "using_baseline_pred=$BASELINE_PRED"
echo "using_embedding_name=$EMBEDDING_NAME"
echo "using_top_pages=$TOP_PAGES"
echo "using_base_only_page_batch_size=$BASE_ONLY_PAGE_BATCH_SIZE"
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
