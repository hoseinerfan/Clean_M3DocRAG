#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/env/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

: "${DATA_NAME:?Set DATA_NAME, e.g. mmdocir or vidore-v3}"
: "${DATA_ROOT:?Set DATA_ROOT to the converted dataset root}"
: "${DENSE_PRED:?Set DENSE_PRED to the exact/full dense prediction JSON}"
: "${OUT_DIR:?Set OUT_DIR for doc-RRF outputs}"

SPLIT="${SPLIT:-dev}"
GOLD="${GOLD:-$DATA_ROOT/MMQA_${SPLIT}.jsonl}"
QIDS_JSONL="${QIDS_JSONL:-$DATA_ROOT/qids_${SPLIT}.jsonl}"
DOC_PAGES_JSONL="${DOC_PAGES_JSONL:-$DATA_ROOT/doc_pages_${SPLIT}.jsonl}"

SPLADE_MODEL="${SPLADE_MODEL:-naver/splade-cocondenser-ensembledistil}"
SPLADE_PAGE_BATCH_SIZE="${SPLADE_PAGE_BATCH_SIZE:-8}"
SPLADE_QUERY_BATCH_SIZE="${SPLADE_QUERY_BATCH_SIZE:-16}"
SPLADE_DEVICE="${SPLADE_DEVICE:-auto}"
PAGE_MAX_LENGTH="${PAGE_MAX_LENGTH:-512}"
QUERY_MAX_LENGTH="${QUERY_MAX_LENGTH:-64}"
PAGE_TOPK_TERMS="${PAGE_TOPK_TERMS:-128}"
QUERY_TOPK_TERMS="${QUERY_TOPK_TERMS:-32}"
TOP_PAGES="${TOP_PAGES:-1000}"
REQUIRE_NONEMPTY_PAGE_TEXT="${REQUIRE_NONEMPTY_PAGE_TEXT:-1}"

RRF_K="${RRF_K:-10}"
DENSE_WEIGHT="${DENSE_WEIGHT:-0.75}"
SPARSE_WEIGHT="${SPARSE_WEIGHT:-1.25}"
DENSE_TOP_DOCS="${DENSE_TOP_DOCS:-20}"
SPARSE_TOP_DOCS="${SPARSE_TOP_DOCS:-20}"
FINAL_TOP_DOCS="${FINAL_TOP_DOCS:-20}"

mkdir -p "$OUT_DIR"

echo "doc_rrf_config DATA_NAME=$DATA_NAME"
echo "doc_rrf_config DATA_ROOT=$DATA_ROOT"
echo "doc_rrf_config GOLD=$GOLD"
echo "doc_rrf_config QIDS_JSONL=$QIDS_JSONL"
echo "doc_rrf_config DOC_PAGES_JSONL=$DOC_PAGES_JSONL"
echo "doc_rrf_config OUT_DIR=$OUT_DIR"

PAGE_TEXT_JSONL="${PAGE_TEXT_JSONL:-$OUT_DIR/${DATA_NAME}_page_text_${SPLIT}.jsonl}"
PAGE_TEXT_SUMMARY="${PAGE_TEXT_SUMMARY:-$OUT_DIR/${DATA_NAME}_page_text_${SPLIT}_summary.json}"
SPLADE_INDEX_PT="${SPLADE_INDEX_PT:-$OUT_DIR/${DATA_NAME}_splade_page_index.pt}"
SPLADE_INDEX_SUMMARY="${SPLADE_INDEX_SUMMARY:-$OUT_DIR/${DATA_NAME}_splade_page_index_summary.json}"
SPLADE_JSONL="${SPLADE_JSONL:-$OUT_DIR/${DATA_NAME}_splade_ret${TOP_PAGES}.jsonl}"
SPLADE_PRED="${SPLADE_PRED:-$OUT_DIR/${DATA_NAME}_splade_ret${TOP_PAGES}.prediction.json}"
SPLADE_SUMMARY="${SPLADE_SUMMARY:-$OUT_DIR/${DATA_NAME}_splade_ret${TOP_PAGES}.summary.json}"
RRF_PRED="${RRF_PRED:-$OUT_DIR/${DATA_NAME}_exact_dense_splade_doc_rrf.prediction.json}"
RRF_SUMMARY="${RRF_SUMMARY:-$OUT_DIR/${DATA_NAME}_exact_dense_splade_doc_rrf.summary.json}"

EXPORT_ARGS=(
  --doc-pages-jsonl "$DOC_PAGES_JSONL"
  --output-jsonl "$PAGE_TEXT_JSONL"
  --output-summary-json "$PAGE_TEXT_SUMMARY"
  --require-nonempty
)
if [[ -n "${PDF_ROOT:-}" ]]; then
  EXPORT_ARGS+=(--pdf-root "$PDF_ROOT")
fi
if [[ -n "${IMAGE_ROOT:-}" ]]; then
  EXPORT_ARGS+=(--image-root "$IMAGE_ROOT")
fi
if [[ "${OCR_IMAGE:-0}" == "1" ]]; then
  EXPORT_ARGS+=(--ocr-image)
  if [[ -n "${OCR_ENGINE:-}" ]]; then
    EXPORT_ARGS+=(--ocr-engine "$OCR_ENGINE")
  fi
  if [[ -n "${OCR_BIN:-}" ]]; then
    EXPORT_ARGS+=(--ocr-bin "$OCR_BIN")
  fi
  if [[ -n "${OCR_LANG:-}" ]]; then
    EXPORT_ARGS+=(--ocr-lang "$OCR_LANG")
  fi
  if [[ -n "${OCR_PSM:-}" ]]; then
    EXPORT_ARGS+=(--ocr-psm "$OCR_PSM")
  fi
  if [[ -n "${OCR_TIMEOUT:-}" ]]; then
    EXPORT_ARGS+=(--ocr-timeout "$OCR_TIMEOUT")
  fi
  if [[ "${OCR_CONTINUE_ON_ERROR:-0}" == "1" ]]; then
    EXPORT_ARGS+=(--ocr-continue-on-error)
  fi
  if [[ "${EASYOCR_GPU:-0}" == "1" ]]; then
    EXPORT_ARGS+=(--easyocr-gpu)
  fi
  if [[ -n "${EASYOCR_MODEL_DIR:-}" ]]; then
    EXPORT_ARGS+=(--easyocr-model-dir "$EASYOCR_MODEL_DIR")
  fi
  if [[ "${NO_EASYOCR_DOWNLOAD:-0}" == "1" ]]; then
    EXPORT_ARGS+=(--no-easyocr-download)
  fi
fi
if [[ -n "${PAGE_TEXT_FIELDS:-}" ]]; then
  for field in $PAGE_TEXT_FIELDS; do
    EXPORT_ARGS+=(--text-field "$field")
  done
fi
if [[ -n "${PAGE_TEXT_EXTRA_FIELDS:-}" ]]; then
  for field in $PAGE_TEXT_EXTRA_FIELDS; do
    EXPORT_ARGS+=(--extra-field "$field")
  done
fi

if [[ "${SKIP_EXPORT:-0}" != "1" ]]; then
  "$PYTHON_BIN" "$REPO_ROOT/scripts/export_converted_page_text.py" "${EXPORT_ARGS[@]}"
fi

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  BUILD_ARGS=(
    --page-text-jsonl "$PAGE_TEXT_JSONL" \
    --model-name-or-path "$SPLADE_MODEL" \
    --batch-size "$SPLADE_PAGE_BATCH_SIZE" \
    --max-length "$PAGE_MAX_LENGTH" \
    --topk-terms "$PAGE_TOPK_TERMS" \
    --device "$SPLADE_DEVICE" \
    --output-index-pt "$SPLADE_INDEX_PT" \
    --output-summary-json "$SPLADE_INDEX_SUMMARY"
  )
  if [[ "$REQUIRE_NONEMPTY_PAGE_TEXT" != "0" ]]; then
    BUILD_ARGS+=(--require-nonempty-text)
  fi
  "$PYTHON_BIN" "$REPO_ROOT/scripts/build_splade_page_index.py" "${BUILD_ARGS[@]}"
fi

if [[ "${SKIP_SPLADE_RETRIEVAL:-0}" != "1" ]]; then
  "$PYTHON_BIN" "$REPO_ROOT/scripts/run_splade_page_retrieval.py" \
    --qid-jsonl "$QIDS_JSONL" \
    --gold "$GOLD" \
    --index-pt "$SPLADE_INDEX_PT" \
    --model-name-or-path "$SPLADE_MODEL" \
    --batch-size "$SPLADE_QUERY_BATCH_SIZE" \
    --max-length "$QUERY_MAX_LENGTH" \
    --query-topk-terms "$QUERY_TOPK_TERMS" \
    --device "$SPLADE_DEVICE" \
    --top-pages "$TOP_PAGES" \
    --output-jsonl "$SPLADE_JSONL" \
    --output-prediction-json "$SPLADE_PRED" \
    --output-summary-json "$SPLADE_SUMMARY"
fi

"$PYTHON_BIN" "$REPO_ROOT/scripts/fuse_page_retrieval_predictions.py" \
  --dense-prediction-json "$DENSE_PRED" \
  --sparse-prediction-json "$SPLADE_PRED" \
  --gold "$GOLD" \
  --fusion-mode doc_rrf \
  --dense-top-docs "$DENSE_TOP_DOCS" \
  --sparse-top-docs "$SPARSE_TOP_DOCS" \
  --final-top-docs "$FINAL_TOP_DOCS" \
  --rrf-k "$RRF_K" \
  --dense-weight "$DENSE_WEIGHT" \
  --sparse-weight "$SPARSE_WEIGHT" \
  --output-prediction-json "$RRF_PRED" \
  --output-summary-json "$RRF_SUMMARY"

"$PYTHON_BIN" "$REPO_ROOT/mmdocir/evaluate_mmdocir_retrieval.py" \
  --pred "$RRF_PRED" \
  --gold "$GOLD" \
  --recall-k 1 2 4 5 10 20
