#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/env/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

GRAPH_OUT_DIR="${GRAPH_OUT_DIR:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_graph_pagepreserve_mmqa_dev}"
GRAPH_PRED="${GRAPH_PRED:-$GRAPH_OUT_DIR/mmqa_dev_plain_top224_splade_graph_pagepreserve_denseheavy_lightboth.prediction.json}"
GOLD="${GOLD:-$REPO_ROOT/data/m3-docvqa/multimodalqa/MMQA_dev.jsonl}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen2-VL-7B-Instruct}"
BITS="${BITS:-16}"
QA_TOP_PAGES="${QA_TOP_PAGES:-4}"
QUESTION_TYPE_FILTER="${QUESTION_TYPE_FILTER:-}"
QA_OUT_DIR="${QA_OUT_DIR:-$GRAPH_OUT_DIR}"
OUTPUT_LABEL="${OUTPUT_LABEL:-mmqa_dev_plain_top224_splade_graph_pagepreserve_denseheavy_lightboth_qwen2vl_top${QA_TOP_PAGES}}"
OUTPUT_PRED="${OUTPUT_PRED:-$QA_OUT_DIR/${OUTPUT_LABEL}.prediction.json}"
OUTPUT_EVAL="${OUTPUT_EVAL:-$QA_OUT_DIR/${OUTPUT_LABEL}.eval.json}"
SAVE_EVERY="${SAVE_EVERY:-25}"
DOC_IMAGE_CACHE_SIZE="${DOC_IMAGE_CACHE_SIZE:-16}"
RUN_EVAL="${RUN_EVAL:-1}"
RESUME="${RESUME:-1}"

mkdir -p "$QA_OUT_DIR"

echo "using_graph_pred=$GRAPH_PRED"
echo "using_gold=$GOLD"
echo "using_model_name_or_path=$MODEL_NAME_OR_PATH"
echo "using_qa_top_pages=$QA_TOP_PAGES"
echo "using_qa_out_dir=$QA_OUT_DIR"
echo "using_output_pred=$OUTPUT_PRED"
echo "using_output_eval=$OUTPUT_EVAL"
echo "using_question_type_filter=${QUESTION_TYPE_FILTER:-ALL}"

ARGS=(
  --prediction-json "$GRAPH_PRED"
  --gold "$GOLD"
  --data-name m3-docvqa
  --split dev
  --model-name-or-path "$MODEL_NAME_OR_PATH"
  --bits "$BITS"
  --qa-top-pages "$QA_TOP_PAGES"
  --doc-image-cache-size "$DOC_IMAGE_CACHE_SIZE"
  --save-every "$SAVE_EVERY"
  --output-prediction-json "$OUTPUT_PRED"
  --output-eval-json "$OUTPUT_EVAL"
)

if [[ -n "$QUESTION_TYPE_FILTER" ]]; then
  ARGS+=(--question-type-filter "$QUESTION_TYPE_FILTER")
fi
if [[ "$RUN_EVAL" == "1" ]]; then
  ARGS+=(--run-eval)
fi
if [[ "$RESUME" == "1" ]]; then
  ARGS+=(--resume)
fi

"$PYTHON_BIN" "$REPO_ROOT/scripts/run_m3docvqa_external_retrieval_qa.py" "${ARGS[@]}"
