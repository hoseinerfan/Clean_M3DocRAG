#!/usr/bin/env bash

# Source this file on the HPC before running the DUDE helpers:
#   source dude/env_hpc.sh

export REPO_ROOT="${REPO_ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG}"
export DUDE_WORK_ROOT="${DUDE_WORK_ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/DUDE_M3DocRAG}"

export LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-$DUDE_WORK_ROOT/data}"
export LOCAL_EMBEDDINGS_DIR="${LOCAL_EMBEDDINGS_DIR:-$DUDE_WORK_ROOT/embeddings}"
export LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_DIR:-$DUDE_WORK_ROOT/output}"
export LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-$REPO_ROOT/model}"

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

export HF_HOME="${HF_HOME:-$DUDE_WORK_ROOT/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HUGGINGFACE_HUB_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DUDE_WORK_ROOT/xdg_cache}"

mkdir -p \
  "$LOCAL_DATA_DIR" \
  "$LOCAL_EMBEDDINGS_DIR" \
  "$LOCAL_OUTPUT_DIR" \
  "$HF_DATASETS_CACHE" \
  "$HUGGINGFACE_HUB_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$XDG_CACHE_HOME" \
  "$DUDE_WORK_ROOT/logs"
