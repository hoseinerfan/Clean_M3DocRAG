#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/env/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

VITAL_PATHS_ENV="${VITAL_PATHS_ENV:-$REPO_ROOT/hpc_vital_paths.generated.env}"
if [[ -f "$VITAL_PATHS_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$VITAL_PATHS_ENV"
fi

CUSTOM_ROOT="${CUSTOM_ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/output/m3docvqa_standard_reranker_baselines}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-castorini/monot5-base-msmarco-10k}"
MODEL_TAG="${MODEL_TAG:-monot5_base_msmarco_10k}"
RERANK_TOP_K="${RERANK_TOP_K:-100}"
LABEL="${LABEL:-mmqa_dev_${MODEL_TAG}_seq2seq_gpp_top${RERANK_TOP_K}}"
DEFAULT_PROMPT_TEMPLATE="Query: {question} Document: {document} Relevant:"
PROMPT_TEMPLATE="${PROMPT_TEMPLATE:-$DEFAULT_PROMPT_TEMPLATE}"

require_file() {
  local name="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    echo "missing_${name}: $path" >&2
    exit 1
  fi
}

EVAL_GOLD="${EVAL_GOLD:-$REPO_ROOT/output/m3docvqa_mmqa_direct_evidence_pseudo_page_labels/mmqa_dev_pseudo_page_labels_direct_exactonly_adaptive_norm05.augmented_gold.jsonl}"
BASE_EVAL_PRED="${BASE_EVAL_PRED:-$REPO_ROOT/output/m3docvqa_gpp_hyperlink_node_ablation_exact_maxsim/mmqa_dev_exact_maxsim_gpp_hyperlink_node_no_hyperlink.prediction.json}"
EVAL_PAGE_TEXT_JSONL="${EVAL_PAGE_TEXT_JSONL:-$CUSTOM_ROOT/outputs/m3docvqa_page_text/m3docvqa_dev_page_text.jsonl}"

mkdir -p "$OUT_DIR"
require_file eval_gold "$EVAL_GOLD"
require_file base_eval_pred "$BASE_EVAL_PRED"
require_file eval_page_text_jsonl "$EVAL_PAGE_TEXT_JSONL"

extra_model_args=()
if [[ "${USE_FP16:-1}" == "1" ]]; then
  extra_model_args+=(--use-fp16)
fi
if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  extra_model_args+=(--trust-remote-code)
fi
if [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  extra_model_args+=(--local-files-only)
fi
if [[ "${RESUME:-1}" == "1" ]]; then
  extra_model_args+=(--resume)
fi

echo "using_eval_gold=$EVAL_GOLD"
echo "using_base_eval_pred=$BASE_EVAL_PRED"
echo "using_eval_page_text_jsonl=$EVAL_PAGE_TEXT_JSONL"
echo "using_model_name_or_path=$MODEL_NAME_OR_PATH"
echo "using_model_tag=$MODEL_TAG"
echo "using_rerank_top_k=$RERANK_TOP_K"
echo "using_out_dir=$OUT_DIR"
echo "using_label=$LABEL"

"$PYTHON_BIN" "$REPO_ROOT/scripts/rerank_m3docvqa_seq2seq_reranker_pages.py" \
  --gold "$EVAL_GOLD" \
  --base-pred "$BASE_EVAL_PRED" \
  --page-text-jsonl "$EVAL_PAGE_TEXT_JSONL" \
  --model-name-or-path "$MODEL_NAME_OR_PATH" \
  --batch-size "${BATCH_SIZE:-8}" \
  --max-length "${MAX_LENGTH:-512}" \
  --max-page-chars "${MAX_PAGE_CHARS:-4000}" \
  --candidate-top-k "${CANDIDATE_TOP_K:-1000}" \
  --rerank-top-k "$RERANK_TOP_K" \
  --blend-alpha "${BLEND_ALPHA:-1.0}" \
  --prompt-template "$PROMPT_TEMPLATE" \
  --positive-token "${POSITIVE_TOKEN:-true}" \
  --negative-token "${NEGATIVE_TOKEN:-false}" \
  --save-every "${SAVE_EVERY:-25}" \
  --qid-limit "${QID_LIMIT:-0}" \
  --recall-k 1 2 4 5 10 20 50 100 1000 \
  --output-prediction-json "$OUT_DIR/${LABEL}.prediction.json" \
  --output-summary-json "$OUT_DIR/${LABEL}.summary.json" \
  --output-table-md "$OUT_DIR/${LABEL}.table.md" \
  --output-score-jsonl "$OUT_DIR/${LABEL}.scores.jsonl" \
  "${extra_model_args[@]}"
