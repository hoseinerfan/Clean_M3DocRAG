#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/m3docvqa_internal_env.sh"

BITS="${BITS:-16}"
N_RETRIEVAL_PAGES="${N_RETRIEVAL_PAGES:-1000}"
IGNORE_PAD_SCORES_IN_FINAL_RANKING="${IGNORE_PAD_SCORES_IN_FINAL_RANKING:-0}"
FAISS_TOKEN_TABLE_MODE="${FAISS_TOKEN_TABLE_MODE:-load_embeddings}"
FAISS_INDEX_SCORE_SOURCE="${FAISS_INDEX_SCORE_SOURCE:-embedding}"
BASELINE_RETR_OUT_DIR="${BASELINE_RETR_OUT_DIR:-$LOCAL_OUTPUT_DIR/m3docvqa_baseline_mmqa_${SPLIT}}"
BASELINE_RUN_DIR="${BASELINE_RUN_DIR:-$BASELINE_RETR_OUT_DIR/raw_run_outputs}"
BASELINE_LABEL="${BASELINE_LABEL:-mmqa_${SPLIT}_baseline_ret${N_RETRIEVAL_PAGES}_${FAISS_INDEX_TYPE}_nprobe${FAISS_NPROBE}}"
OUTPUT_PRED="${OUTPUT_PRED:-$BASELINE_RETR_OUT_DIR/${BASELINE_LABEL}.prediction.json}"
OUTPUT_EVAL="${OUTPUT_EVAL:-$BASELINE_RETR_OUT_DIR/${BASELINE_LABEL}.eval.json}"

mkdir -p "$BASELINE_RETR_OUT_DIR" "$BASELINE_RUN_DIR"

echo "using_local_data_dir=$LOCAL_DATA_DIR"
echo "using_local_model_dir=$LOCAL_MODEL_DIR"
echo "using_local_embeddings_dir=$LOCAL_EMBEDDINGS_DIR"
echo "using_data_name=$DATA_NAME"
echo "using_split=$SPLIT"
echo "using_embedding_name=$EMBEDDING_NAME"
echo "using_faiss_index_type=$FAISS_INDEX_TYPE"
echo "using_faiss_nprobe=$FAISS_NPROBE"
echo "using_n_retrieval_pages=$N_RETRIEVAL_PAGES"
echo "using_ignore_pad_scores=${IGNORE_PAD_SCORES_IN_FINAL_RANKING}"
echo "using_faiss_token_table_mode=$FAISS_TOKEN_TABLE_MODE"
echo "using_faiss_index_score_source=$FAISS_INDEX_SCORE_SOURCE"
echo "using_baseline_run_dir=$BASELINE_RUN_DIR"
echo "using_output_pred=$OUTPUT_PRED"
echo "using_output_eval=$OUTPUT_EVAL"

ARGS=(
  --use_retrieval
  --retrieval_only=True
  --retrieval_model_type="$RETRIEVAL_MODEL_TYPE"
  --load_embedding=True
  --split="$SPLIT"
  --bits="$BITS"
  --n_retrieval_pages="$N_RETRIEVAL_PAGES"
  --data_name="$DATA_NAME"
  --embedding_name="$EMBEDDING_NAME"
  --retrieval_model_name_or_path="$RETRIEVAL_MODEL_NAME"
  --retrieval_adapter_model_name_or_path="$RETRIEVAL_ADAPTER_MODEL_NAME"
  --faiss_index_type="$FAISS_INDEX_TYPE"
  --faiss_nprobe="$FAISS_NPROBE"
  --faiss_token_table_mode="$FAISS_TOKEN_TABLE_MODE"
  --faiss_index_score_source="$FAISS_INDEX_SCORE_SOURCE"
  --output_dir="$BASELINE_RUN_DIR"
)
if [[ "$IGNORE_PAD_SCORES_IN_FINAL_RANKING" == "1" ]]; then
  ARGS+=(--ignore_pad_scores_in_final_ranking=True)
fi

"$PYTHON_BIN" "$REPO_ROOT/examples/run_rag_m3docvqa.py" "${ARGS[@]}"

RET_NAME="$RETRIEVAL_MODEL_NAME"
if [[ "$RETRIEVAL_MODEL_TYPE" == "colpali" ]]; then
  RET_NAME="$RETRIEVAL_ADAPTER_MODEL_NAME"
fi

mapfile -t LATEST_OUTPUTS < <(
  "$PYTHON_BIN" - "$BASELINE_RUN_DIR" "$RET_NAME" "$FAISS_INDEX_TYPE" "$FAISS_NPROBE" "$N_RETRIEVAL_PAGES" "$IGNORE_PAD_SCORES_IN_FINAL_RANKING" <<'PY'
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
ret_name = sys.argv[2]
faiss_index_type = sys.argv[3]
faiss_nprobe = sys.argv[4]
n_retrieval_pages = sys.argv[5]
ignore_pad = sys.argv[6] == "1"

nprobe_suffix = "" if faiss_index_type == "flatip" else f"_nprobe{faiss_nprobe}"
rerank_suffix = "_ignorepadscore" if ignore_pad else ""

pred_candidates = sorted(
    (
        path
        for path in run_dir.glob(
            f"{ret_name}_{faiss_index_type}{nprobe_suffix}_ret{n_retrieval_pages}{rerank_suffix}_*.json"
        )
        if not path.name.endswith("_eval_results.json")
    ),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)
eval_candidates = sorted(
    run_dir.glob(
        f"{ret_name}_{faiss_index_type}_ret{n_retrieval_pages}{rerank_suffix}_*_eval_results.json"
    ),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)

if not pred_candidates:
    raise SystemExit("Could not find baseline prediction JSON in the run directory.")
if not eval_candidates:
    raise SystemExit("Could not find baseline eval JSON in the run directory.")

print(pred_candidates[0])
print(eval_candidates[0])
PY
)

LATEST_PRED="${LATEST_OUTPUTS[0]}"
LATEST_EVAL="${LATEST_OUTPUTS[1]}"
cp "$LATEST_PRED" "$OUTPUT_PRED"
cp "$LATEST_EVAL" "$OUTPUT_EVAL"

echo "saved_prediction=$OUTPUT_PRED"
echo "saved_eval=$OUTPUT_EVAL"
echo "source_prediction=$LATEST_PRED"
echo "source_eval=$LATEST_EVAL"
