#!/usr/bin/env bash

# Source this file on the HPC before running the MMDocIR helpers:
#   source mmdocir/env_hpc.sh

export REPO_ROOT="${REPO_ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG}"
export MMDocIR_WORK_ROOT="${MMDocIR_WORK_ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/MMDocIR_M3DocRAG}"

export LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-$MMDocIR_WORK_ROOT/data}"
export LOCAL_EMBEDDINGS_DIR="${LOCAL_EMBEDDINGS_DIR:-$MMDocIR_WORK_ROOT/embeddings}"
export LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_DIR:-$MMDocIR_WORK_ROOT/output}"
export LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-$REPO_ROOT/model}"

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$LOCAL_DATA_DIR" "$LOCAL_EMBEDDINGS_DIR" "$LOCAL_OUTPUT_DIR" "$MMDocIR_WORK_ROOT/hf_snapshot" "$MMDocIR_WORK_ROOT/logs"
