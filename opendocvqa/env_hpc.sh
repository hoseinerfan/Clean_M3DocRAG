#!/usr/bin/env bash

# Source this file on the HPC before running the OpenDocVQA helpers:
#   source opendocvqa/env_hpc.sh

export REPO_ROOT="${REPO_ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG}"
export OPENDOCVQA_WORK_ROOT="${OPENDOCVQA_WORK_ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/OpenDocVQA_M3DocRAG}"

export LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-$OPENDOCVQA_WORK_ROOT/data}"
export LOCAL_EMBEDDINGS_DIR="${LOCAL_EMBEDDINGS_DIR:-$OPENDOCVQA_WORK_ROOT/embeddings}"
export LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_DIR:-$OPENDOCVQA_WORK_ROOT/output}"
export LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-$REPO_ROOT/model}"

export HF_HOME="${HF_HOME:-$OPENDOCVQA_WORK_ROOT/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HUGGINGFACE_HUB_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$OPENDOCVQA_WORK_ROOT/xdg_cache}"

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p \
  "$LOCAL_DATA_DIR" \
  "$LOCAL_EMBEDDINGS_DIR" \
  "$LOCAL_OUTPUT_DIR" \
  "$OPENDOCVQA_WORK_ROOT/hf_cache" \
  "$OPENDOCVQA_WORK_ROOT/logs" \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$HUGGINGFACE_HUB_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$XDG_CACHE_HOME"
