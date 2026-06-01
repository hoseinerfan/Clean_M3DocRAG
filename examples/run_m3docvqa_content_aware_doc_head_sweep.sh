#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

INFERENCE_MODE="${INFERENCE_MODE:-doc_head_blend}"
BLEND_ALPHA="${BLEND_ALPHA:-0.30}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/output/m3docvqa_content_aware_doc_head_sweep}"
FORCE_RERUN="${FORCE_RERUN:-1}"

export INFERENCE_MODE BLEND_ALPHA OUT_DIR FORCE_RERUN

bash "$REPO_ROOT/examples/run_m3docvqa_content_aware_base_sweep.sh"
