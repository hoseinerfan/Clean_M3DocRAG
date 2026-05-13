#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

if [[ -f "$SCRIPT_DIR/env_hpc.sh" ]]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/env_hpc.sh"
fi

DATA_ROOT="${DATA_ROOT:-$LOCAL_DATA_DIR/sci-egqa-bench}"
EMBEDDING_NAME="${EMBEDDING_NAME:-colpali-v1.2_sci-egqa-bench_dev}"
BASELINE_PRED="${BASELINE_PRED:-$LOCAL_OUTPUT_DIR/sciegqa/baseline_ret1000.json}"
OUT_DIR="${OUT_DIR:-$LOCAL_OUTPUT_DIR/sciegqa}"
TOP_PAGES="${TOP_PAGES:-1000}"

mkdir -p "$OUT_DIR"

EMPTY_QUERY_LABELS="$OUT_DIR/empty_query_token_labels.json"
EMPTY_PATCH_LABELS="$OUT_DIR/empty_patch_labels.jsonl"
printf '{}\n' > "$EMPTY_QUERY_LABELS"
: > "$EMPTY_PATCH_LABELS"

python "$REPO_ROOT/scripts/run_visual_rerank_batch.py" \
  --qid-jsonl "$DATA_ROOT/qids_dev.jsonl" \
  --gold "$DATA_ROOT/MMQA_dev.jsonl" \
  --baseline-pred "$BASELINE_PRED" \
  --data-name sci-egqa-bench \
  --split dev \
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
  --base-only-page-batch-size "${BASE_ONLY_PAGE_BATCH_SIZE:-64}" \
  --output-jsonl "$OUT_DIR/plain_top224_ret${TOP_PAGES}.jsonl" \
  --output-summary-json "$OUT_DIR/plain_top224_ret${TOP_PAGES}_summary.json" \
  --output-prediction-json "$OUT_DIR/plain_top224_ret${TOP_PAGES}_prediction.json"

