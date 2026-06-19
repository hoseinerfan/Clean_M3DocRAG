#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$REPO_ROOT/hpc_vital_paths.generated.env" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/hpc_vital_paths.generated.env"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
SPLIT="${SPLIT:-dev}"
DATA_ROOT="${M3DOCVQA_DATA_ROOT:-$REPO_ROOT/data/m3-docvqa}"
GOLD="${GOLD:-$DATA_ROOT/multimodalqa/MMQA_${SPLIT}.jsonl}"
DOC_PAGES_JSONL="${DOC_PAGES_JSONL:-${M3DOCVQA_PAGE_TEXT_JSONL:-${LOCAL_OUTPUT_DIR:-$REPO_ROOT/output}/m3docvqa_page_text/m3docvqa_${SPLIT}_page_text.jsonl}}"
MMQA_TEXTS_JSONL="${MMQA_TEXTS_JSONL:-$DATA_ROOT/multimodalqa/MMQA_texts.jsonl}"
MMQA_TABLES_JSONL="${MMQA_TABLES_JSONL:-$DATA_ROOT/multimodalqa/MMQA_tables.jsonl}"
MMQA_IMAGES_JSONL="${MMQA_IMAGES_JSONL:-$DATA_ROOT/multimodalqa/MMQA_images.jsonl}"
ID_URL_MAPPING_JSONL="${ID_URL_MAPPING_JSONL:-$DATA_ROOT/id_url_mapping.jsonl}"

OUT_DIR="${OUT_DIR:-$REPO_ROOT/output/m3docvqa_mmqa_evidence_unit_aware_pseudo_page_labels}"
LABEL="${LABEL:-mmqa_${SPLIT}_evidence_unit_aware_strict}"
MIN_SCORE="${MIN_SCORE:-8.0}"
HIGH_CONFIDENCE_SCORE="${HIGH_CONFIDENCE_SCORE:-14.0}"
TOP_PAGES_PER_QID="${TOP_PAGES_PER_QID:-4}"
MAX_PAGES_PER_DOC="${MAX_PAGES_PER_DOC:-3}"
MIN_TOKEN_OVERLAP="${MIN_TOKEN_OVERLAP:-0.72}"
INCLUDE_QUESTION_CONTEXT_SIGNALS="${INCLUDE_QUESTION_CONTEXT_SIGNALS:-0}"
REQUIRE_ALL_UNITS_MAPPED="${REQUIRE_ALL_UNITS_MAPPED:-0}"
REQUIRE_ALL_SUPPORT_DOCS_COVERED="${REQUIRE_ALL_SUPPORT_DOCS_COVERED:-0}"
ALLOW_FUZZY_ONLY_MEDIUM="${ALLOW_FUZZY_ONLY_MEDIUM:-0}"
DEDUPLICATE_NORMALIZED_PHRASES="${DEDUPLICATE_NORMALIZED_PHRASES:-0}"
REQUIRE_DIRECT_EVIDENCE_GATE="${REQUIRE_DIRECT_EVIDENCE_GATE:-0}"
DIRECT_FUZZY_OVERLAP="${DIRECT_FUZZY_OVERLAP:-0.90}"
IMAGE_EVIDENCE_MODE="${IMAGE_EVIDENCE_MODE:-legacy_direct}"
IMAGE_TITLE_PROXY_MIN_SCORE="${IMAGE_TITLE_PROXY_MIN_SCORE:-7.0}"

mkdir -p "$OUT_DIR"

echo "using_gold=$GOLD"
echo "using_doc_pages_jsonl=$DOC_PAGES_JSONL"
echo "using_mmqa_texts_jsonl=$MMQA_TEXTS_JSONL"
echo "using_mmqa_tables_jsonl=$MMQA_TABLES_JSONL"
echo "using_mmqa_images_jsonl=$MMQA_IMAGES_JSONL"
echo "using_id_url_mapping_jsonl=$ID_URL_MAPPING_JSONL"
echo "using_out_dir=$OUT_DIR"
echo "using_label=$LABEL"
echo "using_min_score=$MIN_SCORE"
echo "using_high_confidence_score=$HIGH_CONFIDENCE_SCORE"
echo "using_top_pages_per_qid=$TOP_PAGES_PER_QID"
echo "using_max_pages_per_doc=$MAX_PAGES_PER_DOC"
echo "using_image_evidence_mode=$IMAGE_EVIDENCE_MODE"
echo "using_image_title_proxy_min_score=$IMAGE_TITLE_PROXY_MIN_SCORE"

args=(
  --gold "$GOLD"
  --doc-pages-jsonl "$DOC_PAGES_JSONL"
  --mmqa-texts-jsonl "$MMQA_TEXTS_JSONL"
  --mmqa-tables-jsonl "$MMQA_TABLES_JSONL"
  --mmqa-images-jsonl "$MMQA_IMAGES_JSONL"
  --id-url-mapping-jsonl "$ID_URL_MAPPING_JSONL"
  --min-score "$MIN_SCORE"
  --high-confidence-score "$HIGH_CONFIDENCE_SCORE"
  --top-pages-per-qid "$TOP_PAGES_PER_QID"
  --max-pages-per-doc "$MAX_PAGES_PER_DOC"
  --min-token-overlap "$MIN_TOKEN_OVERLAP"
  --direct-fuzzy-overlap "$DIRECT_FUZZY_OVERLAP"
  --image-evidence-mode "$IMAGE_EVIDENCE_MODE"
  --image-title-proxy-min-score "$IMAGE_TITLE_PROXY_MIN_SCORE"
  --output-jsonl "$OUT_DIR/${LABEL}.jsonl"
  --output-summary-json "$OUT_DIR/${LABEL}.summary.json"
  --output-augmented-gold-jsonl "$OUT_DIR/${LABEL}.augmented_gold.jsonl"
)

if [[ "$INCLUDE_QUESTION_CONTEXT_SIGNALS" == "1" ]]; then
  args+=(--include-question-context-signals)
fi
if [[ "$REQUIRE_ALL_UNITS_MAPPED" == "1" ]]; then
  args+=(--require-all-units-mapped)
fi
if [[ "$REQUIRE_ALL_SUPPORT_DOCS_COVERED" == "1" ]]; then
  args+=(--require-all-support-docs-covered)
fi
if [[ "$ALLOW_FUZZY_ONLY_MEDIUM" == "1" ]]; then
  args+=(--allow-fuzzy-only-medium)
fi
if [[ "$DEDUPLICATE_NORMALIZED_PHRASES" == "1" ]]; then
  args+=(--deduplicate-normalized-phrases)
fi
if [[ "$REQUIRE_DIRECT_EVIDENCE_GATE" == "1" ]]; then
  args+=(--require-direct-evidence-gate)
fi

"$PYTHON_BIN" "$REPO_ROOT/scripts/build_mmqa_evidence_unit_aware_pseudo_page_labels.py" "${args[@]}"
