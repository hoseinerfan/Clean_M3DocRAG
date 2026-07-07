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

DATA_NAME="${DATA_NAME:-mpdocvqa}"
DATA_ROOT="${MPDOCVQA_DATA_ROOT:-$LOCAL_DATA_DIR/mpdocvqa}"
TOP_PAGES="${MPDOCVQA_TOP_PAGES:-1000}"
OUT_ROOT="${MPDOCVQA_OUT_ROOT:-$LOCAL_OUTPUT_DIR/mpdocvqa}"
GRAPH_LABEL="${GRAPH_LABEL:-mpdocvqa_exactmaxsim_splade_gpp}"

MODEL_JSON="${MODEL_JSON:-$REPO_ROOT/output/m3docvqa_content_aware_exact_maxsim_direct_exactonly_adaptive_norm05/mmqa_train_to_dev_content_aware_fixed_alpha_0p40_base_exact_maxsim_gpp_direct_exactonly_adaptive_norm05.model.json}"
BASE_PRED="${BASE_PRED:-$OUT_ROOT/graph_ppr_exact_maxsim_splade/${GRAPH_LABEL}.prediction.json}"
SPARSE_PRED="${SPARSE_PRED:-$OUT_ROOT/doc_rrf_exact_maxsim_splade/${DATA_NAME}_splade_ret${TOP_PAGES}.prediction.json}"
OUT_DIR="${CAPP_OUT_DIR:-$OUT_ROOT/trained_capp_transfer_exactmaxsim_gpp}"
BLEND_ALPHA="${BLEND_ALPHA:-0.20}"
ALPHA_TAG="${BLEND_ALPHA//./p}"
LABEL="${LABEL:-${DATA_NAME}_trained_capp_on_exactmaxsim_gpp_a${ALPHA_TAG}}"

PRED_OUT="$OUT_DIR/${LABEL}.prediction.json"
SUMMARY_OUT="$OUT_DIR/${LABEL}.summary.json"
TABLE_OUT="$OUT_DIR/${LABEL}.table.md"
PRIOR_OUT="$OUT_DIR/${LABEL}.prior.jsonl"

if [[ ! -f "$MODEL_JSON" ]]; then
  echo "missing_model_json: $MODEL_JSON" >&2
  exit 1
fi
if [[ ! -f "$BASE_PRED" ]]; then
  echo "missing_base_gpp_prediction: $BASE_PRED" >&2
  echo "hint: run bash mpdocvqa/run_exactmaxsim_gpp_mpdocvqa.sh first" >&2
  exit 1
fi
if [[ ! -f "$DATA_ROOT/doc_pages_dev.jsonl" ]]; then
  echo "missing_doc_pages_jsonl: $DATA_ROOT/doc_pages_dev.jsonl" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

source_args=()
if [[ "${INCLUDE_BASE_AS_SOURCE:-1}" == "1" ]]; then
  source_args+=(--source "gpp=$BASE_PRED")
fi
if [[ -f "$SPARSE_PRED" ]]; then
  source_args+=(--source "sparse=$SPARSE_PRED")
else
  echo "warning_missing_sparse_source=$SPARSE_PRED" >&2
fi

"$PYTHON_BIN" "$REPO_ROOT/scripts/apply_trained_content_aware_page_reranker.py" \
  --model-json "$MODEL_JSON" \
  --base-pred "$BASE_PRED" \
  --page-text-jsonl "$DATA_ROOT/doc_pages_dev.jsonl" \
  --gold "$DATA_ROOT/MMQA_dev.jsonl" \
  "${source_args[@]}" \
  --candidate-top-k "${CANDIDATE_TOP_K:-1000}" \
  --inference-mode "${INFERENCE_MODE:-blend_rerank}" \
  --blend-alpha "$BLEND_ALPHA" \
  --recall-k ${RECALL_K_VALUES:-1 2 4 5 10 20 50 100 1000} \
  --output-prediction-json "$PRED_OUT" \
  --output-summary-json "$SUMMARY_OUT" \
  --output-table-md "$TABLE_OUT" \
  --output-prior-jsonl "$PRIOR_OUT"

echo "saved_capp_prediction=$PRED_OUT"
echo "saved_capp_summary=$SUMMARY_OUT"
echo "saved_capp_table=$TABLE_OUT"
