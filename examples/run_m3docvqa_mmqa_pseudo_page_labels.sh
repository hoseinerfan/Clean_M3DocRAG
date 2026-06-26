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

OUT_DIR="${OUT_DIR:-$REPO_ROOT/output/m3docvqa_mmqa_pseudo_page_labels}"
LABEL="${LABEL:-mmqa_${SPLIT}_pseudo_page_labels}"
MIN_SCORE="${MIN_SCORE:-4.0}"
TOP_PAGES_PER_DOC="${TOP_PAGES_PER_DOC:-2}"
TOP_PAGES_PER_QID="${TOP_PAGES_PER_QID:-8}"
ADAPTIVE_PAGE_CAPS="${ADAPTIVE_PAGE_CAPS:-0}"
MIN_TOKEN_OVERLAP="${MIN_TOKEN_OVERLAP:-0.72}"
SELECTION_POLICY="${SELECTION_POLICY:-score}"
COVERAGE_MIN_MATCH_WEIGHT="${COVERAGE_MIN_MATCH_WEIGHT:-3.0}"
EVIDENCE_WEIGHT_OVERRIDES="${EVIDENCE_WEIGHT_OVERRIDES:-}"
TEXT_INSTANCE_CONTEXT_WINDOW_CHARS="${TEXT_INSTANCE_CONTEXT_WINDOW_CHARS:-0}"
TEXT_INSTANCE_CONTEXT_MAX_PHRASES="${TEXT_INSTANCE_CONTEXT_MAX_PHRASES:-6}"
TEXT_INSTANCE_CONTEXT_PHRASE_TOKEN_COUNT="${TEXT_INSTANCE_CONTEXT_PHRASE_TOKEN_COUNT:-4}"
TEXT_INSTANCE_CONTEXT_MIN_TOKEN_LEN="${TEXT_INSTANCE_CONTEXT_MIN_TOKEN_LEN:-4}"
TEXT_INSTANCE_CONTEXT_VERIFICATION_BONUS="${TEXT_INSTANCE_CONTEXT_VERIFICATION_BONUS:-0}"

mkdir -p "$OUT_DIR"

echo "using_gold=$GOLD"
echo "using_doc_pages_jsonl=$DOC_PAGES_JSONL"
echo "using_mmqa_texts_jsonl=$MMQA_TEXTS_JSONL"
echo "using_mmqa_tables_jsonl=$MMQA_TABLES_JSONL"
echo "using_mmqa_images_jsonl=$MMQA_IMAGES_JSONL"
echo "using_id_url_mapping_jsonl=$ID_URL_MAPPING_JSONL"
echo "using_out_dir=$OUT_DIR"
echo "using_label=$LABEL"
echo "using_selection_policy=$SELECTION_POLICY"
echo "using_coverage_min_match_weight=$COVERAGE_MIN_MATCH_WEIGHT"
echo "using_adaptive_page_caps=$ADAPTIVE_PAGE_CAPS"
echo "using_evidence_weight_overrides=$EVIDENCE_WEIGHT_OVERRIDES"
echo "using_text_instance_context_window_chars=$TEXT_INSTANCE_CONTEXT_WINDOW_CHARS"
echo "using_text_instance_context_max_phrases=$TEXT_INSTANCE_CONTEXT_MAX_PHRASES"
echo "using_text_instance_context_phrase_token_count=$TEXT_INSTANCE_CONTEXT_PHRASE_TOKEN_COUNT"
echo "using_text_instance_context_min_token_len=$TEXT_INSTANCE_CONTEXT_MIN_TOKEN_LEN"
echo "using_text_instance_context_verification_bonus=$TEXT_INSTANCE_CONTEXT_VERIFICATION_BONUS"

ADAPTIVE_ARGS=()
if [[ "$ADAPTIVE_PAGE_CAPS" == "1" || "$ADAPTIVE_PAGE_CAPS" == "true" || "$ADAPTIVE_PAGE_CAPS" == "TRUE" ]]; then
  ADAPTIVE_ARGS+=(--adaptive-page-caps)
fi

"$PYTHON_BIN" "$REPO_ROOT/scripts/build_mmqa_pseudo_page_labels.py" \
  --gold "$GOLD" \
  --doc-pages-jsonl "$DOC_PAGES_JSONL" \
  --mmqa-texts-jsonl "$MMQA_TEXTS_JSONL" \
  --mmqa-tables-jsonl "$MMQA_TABLES_JSONL" \
  --mmqa-images-jsonl "$MMQA_IMAGES_JSONL" \
  --id-url-mapping-jsonl "$ID_URL_MAPPING_JSONL" \
  --min-score "$MIN_SCORE" \
  --top-pages-per-doc "$TOP_PAGES_PER_DOC" \
  --top-pages-per-qid "$TOP_PAGES_PER_QID" \
  "${ADAPTIVE_ARGS[@]}" \
  --min-token-overlap "$MIN_TOKEN_OVERLAP" \
  --selection-policy "$SELECTION_POLICY" \
  --coverage-min-match-weight "$COVERAGE_MIN_MATCH_WEIGHT" \
  --evidence-weight-overrides "$EVIDENCE_WEIGHT_OVERRIDES" \
  --text-instance-context-window-chars "$TEXT_INSTANCE_CONTEXT_WINDOW_CHARS" \
  --text-instance-context-max-phrases "$TEXT_INSTANCE_CONTEXT_MAX_PHRASES" \
  --text-instance-context-phrase-token-count "$TEXT_INSTANCE_CONTEXT_PHRASE_TOKEN_COUNT" \
  --text-instance-context-min-token-len "$TEXT_INSTANCE_CONTEXT_MIN_TOKEN_LEN" \
  --text-instance-context-verification-bonus "$TEXT_INSTANCE_CONTEXT_VERIFICATION_BONUS" \
  --output-jsonl "$OUT_DIR/${LABEL}.jsonl" \
  --output-summary-json "$OUT_DIR/${LABEL}.summary.json" \
  --output-augmented-gold-jsonl "$OUT_DIR/${LABEL}.augmented_gold.jsonl"
