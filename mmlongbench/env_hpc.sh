#!/usr/bin/env bash

# Source this file on the HPC before running the MMLongBench helpers:
#   source mmlongbench/env_hpc.sh

export REPO_ROOT="${REPO_ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG}"
export MMLONGBENCH_WORK_ROOT="${MMLONGBENCH_WORK_ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/MMLongBench_M3DocRAG}"

export LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-$MMLONGBENCH_WORK_ROOT/data}"
export LOCAL_EMBEDDINGS_DIR="${LOCAL_EMBEDDINGS_DIR:-$MMLONGBENCH_WORK_ROOT/embeddings}"
export LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_DIR:-$MMLONGBENCH_WORK_ROOT/output}"
export LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-$REPO_ROOT/model}"

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$LOCAL_DATA_DIR" "$LOCAL_EMBEDDINGS_DIR" "$LOCAL_OUTPUT_DIR" "$MMLONGBENCH_WORK_ROOT/hf_snapshot" "$MMLONGBENCH_WORK_ROOT/logs"
