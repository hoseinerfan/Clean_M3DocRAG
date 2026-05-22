#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/m3docvqa_internal_env.sh"

NUM_PROCESSES="${NUM_PROCESSES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
EMBEDDING_DIR="${EMBEDDING_DIR:-$LOCAL_EMBEDDINGS_DIR/$EMBEDDING_NAME}"

mkdir -p "$LOCAL_EMBEDDINGS_DIR"

echo "using_local_data_dir=$LOCAL_DATA_DIR"
echo "using_local_model_dir=$LOCAL_MODEL_DIR"
echo "using_local_embeddings_dir=$LOCAL_EMBEDDINGS_DIR"
echo "using_data_name=$DATA_NAME"
echo "using_split=$SPLIT"
echo "using_retrieval_model_type=$RETRIEVAL_MODEL_TYPE"
echo "using_retrieval_model_name=$RETRIEVAL_MODEL_NAME"
echo "using_retrieval_adapter_model_name=$RETRIEVAL_ADAPTER_MODEL_NAME"
echo "using_embedding_name=$EMBEDDING_NAME"
echo "using_embedding_dir=$EMBEDDING_DIR"

"$ACCELERATE_BIN" launch --num_processes="$NUM_PROCESSES" --mixed_precision="$MIXED_PRECISION" \
  "$REPO_ROOT/examples/run_page_embedding.py" \
  --use_retrieval \
  --retrieval_model_type="$RETRIEVAL_MODEL_TYPE" \
  --data_name="$DATA_NAME" \
  --split="$SPLIT" \
  --loop_unique_doc_ids=True \
  --per_device_eval_batch_size="$PER_DEVICE_EVAL_BATCH_SIZE" \
  --dataloader_num_workers="$DATALOADER_NUM_WORKERS" \
  --output_dir="$EMBEDDING_DIR" \
  --retrieval_model_name_or_path="$RETRIEVAL_MODEL_NAME" \
  --retrieval_adapter_model_name_or_path="$RETRIEVAL_ADAPTER_MODEL_NAME"

echo "saved_embedding_dir=$EMBEDDING_DIR"
