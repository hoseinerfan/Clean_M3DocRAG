#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/m3docvqa_internal_env.sh"

INDEX_DIR="${INDEX_DIR:-$LOCAL_EMBEDDINGS_DIR/$INDEX_NAME}"

mkdir -p "$LOCAL_EMBEDDINGS_DIR"

echo "using_local_data_dir=$LOCAL_DATA_DIR"
echo "using_local_embeddings_dir=$LOCAL_EMBEDDINGS_DIR"
echo "using_data_name=$DATA_NAME"
echo "using_split=$SPLIT"
echo "using_retrieval_model_type=$RETRIEVAL_MODEL_TYPE"
echo "using_embedding_name=$EMBEDDING_NAME"
echo "using_faiss_index_type=$FAISS_INDEX_TYPE"
echo "using_index_name=$INDEX_NAME"
echo "using_index_dir=$INDEX_DIR"

"$PYTHON_BIN" "$REPO_ROOT/examples/run_indexing_m3docvqa.py" \
  --use_retrieval \
  --retrieval_model_type="$RETRIEVAL_MODEL_TYPE" \
  --data_name="$DATA_NAME" \
  --split="$SPLIT" \
  --loop_unique_doc_ids=False \
  --embedding_name="$EMBEDDING_NAME" \
  --faiss_index_type="$FAISS_INDEX_TYPE" \
  --output_dir="$INDEX_DIR"

echo "saved_index_dir=$INDEX_DIR"
echo "saved_index_bin=$INDEX_DIR/index.bin"
