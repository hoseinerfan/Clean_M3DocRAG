#!/bin/bash
#SBATCH --job-name=m3docrag-dev
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG}"
CONDA_SH="${CONDA_SH:-/mmfs1/cm/shared/apps_local/ondemand/anaconda/etc/profile.d/conda.sh}"
ENV_PREFIX="${ENV_PREFIX:-$REPO_ROOT/env}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.1.1}"

STAGE="${STAGE:-rag}"                    # embed | index | rag | all
DATASET_NAME="${DATASET_NAME:-m3-docvqa}"
SPLIT="${SPLIT:-dev}"
BITS="${BITS:-16}"
N_RETRIEVAL_PAGES="${N_RETRIEVAL_PAGES:-1}"
RETRIEVAL_ONLY="${RETRIEVAL_ONLY:-False}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
DATA_LEN="${DATA_LEN:-}"
OUTPUT_TAG="${OUTPUT_TAG:-}"
QUERY_TOKEN_FILTER="${QUERY_TOKEN_FILTER:-full}"

RETRIEVAL_MODEL_NAME="${RETRIEVAL_MODEL_NAME:-colpaligemma-3b-pt-448-base}"
RETRIEVAL_ADAPTER_MODEL_NAME="${RETRIEVAL_ADAPTER_MODEL_NAME:-colpali-v1.2}"
VQA_MODEL_NAME="${VQA_MODEL_NAME:-Qwen2-VL-7B-Instruct}"
FAISS_INDEX_TYPE="${FAISS_INDEX_TYPE:-ivfflat}"

EMBEDDING_NAME="${RETRIEVAL_ADAPTER_MODEL_NAME}_${DATASET_NAME}_${SPLIT}"
INDEX_NAME="${EMBEDDING_NAME}_pageindex_${FAISS_INDEX_TYPE}"

EMBEDDING_DIR="${REPO_ROOT}/embeddings/${EMBEDDING_NAME}"
INDEX_DIR="${REPO_ROOT}/embeddings/${INDEX_NAME}"

if [[ "${RETRIEVAL_ONLY,,}" == "true" ]]; then
  OUTPUT_PREFIX="retrieval_only"
else
  OUTPUT_PREFIX="rag"
fi

RAG_OUTPUT_DIR="${REPO_ROOT}/output/${OUTPUT_PREFIX}_${SPLIT}${OUTPUT_TAG:+_${OUTPUT_TAG}}"

source "$CONDA_SH"
conda activate "$ENV_PREFIX"
module load "$CUDA_MODULE"

export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$REPO_ROOT/playwright-browsers}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$REPO_ROOT/.conda/pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$REPO_ROOT/.pip-cache}"
export TMPDIR="${TMPDIR:-$REPO_ROOT/.tmp}"

mkdir -p "$TMPDIR"
cd "$REPO_ROOT"

echo "REPO_ROOT=$REPO_ROOT"
echo "ENV_PREFIX=$ENV_PREFIX"
echo "STAGE=$STAGE"
echo "DATASET_NAME=$DATASET_NAME"
echo "SPLIT=$SPLIT"
echo "EMBEDDING_NAME=$EMBEDDING_NAME"
echo "INDEX_NAME=$INDEX_NAME"
echo "RETRIEVAL_ONLY=$RETRIEVAL_ONLY"
echo "QUERY_TOKEN_FILTER=$QUERY_TOKEN_FILTER"
echo "RAG_OUTPUT_DIR=$RAG_OUTPUT_DIR"

python --version
which python
nvidia-smi

run_embed() {
  if [[ -n "$DATA_LEN" ]]; then
    echo "Warning: DATA_LEN is ignored by the upstream embedding code path when loop_unique_doc_ids=True."
  fi

  accelerate launch --num_processes="$NUM_PROCESSES" --mixed_precision="$MIXED_PRECISION" examples/run_page_embedding.py \
    --use_retrieval \
    --retrieval_model_type=colpali \
    --data_name="$DATASET_NAME" \
    --split="$SPLIT" \
    --loop_unique_doc_ids=True \
    --output_dir="$EMBEDDING_DIR" \
    --retrieval_model_name_or_path="$RETRIEVAL_MODEL_NAME" \
    --retrieval_adapter_model_name_or_path="$RETRIEVAL_ADAPTER_MODEL_NAME"
}

run_index() {
  python examples/run_indexing_m3docvqa.py \
    --use_retrieval \
    --retrieval_model_type=colpali \
    --data_name="$DATASET_NAME" \
    --split="$SPLIT" \
    --loop_unique_doc_ids=False \
    --embedding_name="$EMBEDDING_NAME" \
    --faiss_index_type="$FAISS_INDEX_TYPE" \
    --output_dir="$INDEX_DIR"
}

run_rag() {
  rag_cmd=(
    python examples/run_rag_m3docvqa.py
    --use_retrieval
    --retrieval_model_type=colpali
    --load_embedding=True
    --split="$SPLIT"
    --bits="$BITS"
    --n_retrieval_pages="$N_RETRIEVAL_PAGES"
    --retrieval_only="$RETRIEVAL_ONLY"
    --data_name="$DATASET_NAME"
    --model_name_or_path="$VQA_MODEL_NAME"
    --embedding_name="$EMBEDDING_NAME"
    --faiss_index_type="$FAISS_INDEX_TYPE"
    --retrieval_model_name_or_path="$RETRIEVAL_MODEL_NAME"
    --retrieval_adapter_model_name_or_path="$RETRIEVAL_ADAPTER_MODEL_NAME"
    --output_dir="$RAG_OUTPUT_DIR"
    --query_token_filter="$QUERY_TOKEN_FILTER"
  )

  if [[ -n "$DATA_LEN" ]]; then
    rag_cmd+=(--data_len="$DATA_LEN")
  fi

  "${rag_cmd[@]}"
}

case "$STAGE" in
  embed)
    run_embed
    ;;
  index)
    run_index
    ;;
  rag)
    run_rag
    ;;
  all)
    run_embed
    run_index
    run_rag
    ;;
  *)
    echo "Unsupported STAGE: $STAGE"
    echo "Expected one of: embed, index, rag, all"
    exit 1
    ;;
esac
