#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$REPO_ROOT/hpc_vital_paths.generated.env" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/hpc_vital_paths.generated.env"
fi

SPLITS="${SPLITS:-dev}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/output/m3docvqa_mmqa_direct_evidence_pseudo_page_labels}"
LABEL_SUFFIX="${LABEL_SUFFIX:-direct_strict}"
MIN_SCORE="${MIN_SCORE:-8}"
TOP_PAGES_PER_DOC="${TOP_PAGES_PER_DOC:-1}"
TOP_PAGES_PER_QID="${TOP_PAGES_PER_QID:-4}"
ADAPTIVE_PAGE_CAPS="${ADAPTIVE_PAGE_CAPS:-0}"
MIN_TOKEN_OVERLAP="${MIN_TOKEN_OVERLAP:-0.72}"
SELECTION_POLICY="${SELECTION_POLICY:-score}"
COVERAGE_MIN_MATCH_WEIGHT="${COVERAGE_MIN_MATCH_WEIGHT:-3.0}"
EVIDENCE_COVERAGE_TIE_BREAKER="${EVIDENCE_COVERAGE_TIE_BREAKER:-page_order}"
TEXT_INSTANCE_CONTEXT_WINDOW_CHARS="${TEXT_INSTANCE_CONTEXT_WINDOW_CHARS:-240}"
TEXT_INSTANCE_CONTEXT_MAX_PHRASES="${TEXT_INSTANCE_CONTEXT_MAX_PHRASES:-6}"
TEXT_INSTANCE_CONTEXT_PHRASE_TOKEN_COUNT="${TEXT_INSTANCE_CONTEXT_PHRASE_TOKEN_COUNT:-4}"
TEXT_INSTANCE_CONTEXT_MIN_TOKEN_LEN="${TEXT_INSTANCE_CONTEXT_MIN_TOKEN_LEN:-4}"
TEXT_INSTANCE_CONTEXT_VERIFICATION_BONUS="${TEXT_INSTANCE_CONTEXT_VERIFICATION_BONUS:-5}"

# Direct-evidence-only strict policy:
# - Only MMQA direct evidence channels can create pseudo-page labels.
# - Topic, title, entity, support-document, row-link, and answer-string signals are disabled.
# - With MIN_SCORE=8, an exact match of text_instance/table_answer_cell/image_title
#   passes the threshold; fuzzy matches remain below threshold unless this policy is
#   explicitly changed.
# - start_byte-derived text_instance_context has zero standalone weight. It adds only
#   a verification bonus when the same page also matches text_instance, so context-only
#   pages cannot become pseudo-gold labels.
DIRECT_EVIDENCE_WEIGHT_OVERRIDES="${DIRECT_EVIDENCE_WEIGHT_OVERRIDES:-answer_text=0,text_instance=10,text_instance_context=0,image_title=10,image_doc_title=0,table_title=0,table_answer_cell=10,table_row_cell=0,table_row_link_text=0,table_row_link_title=0,supporting_doc_title=0,answer_entity=0,question_entity=0,pseudo_question_slot=0}"

for split in $SPLITS; do
  doc_pages_jsonl="${DOC_PAGES_JSONL:-${M3DOCVQA_PAGE_TEXT_DIR:-${LOCAL_OUTPUT_DIR:-$REPO_ROOT/output}/m3docvqa_page_text}/m3docvqa_${split}_page_text.jsonl}"
  label="mmqa_${split}_pseudo_page_labels_${LABEL_SUFFIX}"

  echo
  echo "== Build direct-evidence strict pseudo-page labels: split=$split =="
  echo "using_doc_pages_jsonl=$doc_pages_jsonl"
  echo "using_label=$label"
  echo "using_selection_policy=$SELECTION_POLICY"
  echo "using_coverage_min_match_weight=$COVERAGE_MIN_MATCH_WEIGHT"
  echo "using_evidence_coverage_tie_breaker=$EVIDENCE_COVERAGE_TIE_BREAKER"
  echo "using_adaptive_page_caps=$ADAPTIVE_PAGE_CAPS"
  echo "using_direct_evidence_weight_overrides=$DIRECT_EVIDENCE_WEIGHT_OVERRIDES"
  echo "using_text_instance_context_window_chars=$TEXT_INSTANCE_CONTEXT_WINDOW_CHARS"
  echo "using_text_instance_context_verification_bonus=$TEXT_INSTANCE_CONTEXT_VERIFICATION_BONUS"

  SPLIT="$split" \
    DOC_PAGES_JSONL="$doc_pages_jsonl" \
    OUT_DIR="$OUT_DIR" \
    LABEL="$label" \
    MIN_SCORE="$MIN_SCORE" \
    TOP_PAGES_PER_DOC="$TOP_PAGES_PER_DOC" \
    TOP_PAGES_PER_QID="$TOP_PAGES_PER_QID" \
    ADAPTIVE_PAGE_CAPS="$ADAPTIVE_PAGE_CAPS" \
    MIN_TOKEN_OVERLAP="$MIN_TOKEN_OVERLAP" \
    SELECTION_POLICY="$SELECTION_POLICY" \
    COVERAGE_MIN_MATCH_WEIGHT="$COVERAGE_MIN_MATCH_WEIGHT" \
    EVIDENCE_COVERAGE_TIE_BREAKER="$EVIDENCE_COVERAGE_TIE_BREAKER" \
    TEXT_INSTANCE_CONTEXT_WINDOW_CHARS="$TEXT_INSTANCE_CONTEXT_WINDOW_CHARS" \
    TEXT_INSTANCE_CONTEXT_MAX_PHRASES="$TEXT_INSTANCE_CONTEXT_MAX_PHRASES" \
    TEXT_INSTANCE_CONTEXT_PHRASE_TOKEN_COUNT="$TEXT_INSTANCE_CONTEXT_PHRASE_TOKEN_COUNT" \
    TEXT_INSTANCE_CONTEXT_MIN_TOKEN_LEN="$TEXT_INSTANCE_CONTEXT_MIN_TOKEN_LEN" \
    TEXT_INSTANCE_CONTEXT_VERIFICATION_BONUS="$TEXT_INSTANCE_CONTEXT_VERIFICATION_BONUS" \
    EVIDENCE_WEIGHT_OVERRIDES="$DIRECT_EVIDENCE_WEIGHT_OVERRIDES" \
    bash "$REPO_ROOT/examples/run_m3docvqa_mmqa_pseudo_page_labels.sh"
done
