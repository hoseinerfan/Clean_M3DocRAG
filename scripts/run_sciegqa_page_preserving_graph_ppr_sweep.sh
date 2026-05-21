#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

if [[ "${SOURCE_SCIEGQA_ENV:-1}" == "1" ]]; then
  unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
  # shellcheck source=../sciegqa/env_hpc.sh
  source "$REPO_ROOT/sciegqa/env_hpc.sh"
fi

: "${LOCAL_DATA_DIR:?Set LOCAL_DATA_DIR or source sciegqa/env_hpc.sh}"
: "${LOCAL_OUTPUT_DIR:?Set LOCAL_OUTPUT_DIR or source sciegqa/env_hpc.sh}"

DATA_NAME="${DATA_NAME:-sciegqa}" \
DATA_ROOT="${DATA_ROOT:-$LOCAL_DATA_DIR/sci-egqa-bench}" \
DENSE_PRED="${DENSE_PRED:-$LOCAL_OUTPUT_DIR/sciegqa/plain_top224_ret1000_prediction.json}" \
SPARSE_PRED="${SPARSE_PRED:-$LOCAL_OUTPUT_DIR/sciegqa/doc_rrf_plain_top224_splade/sciegqa_splade_ret1000.prediction.json}" \
OUT_DIR="${OUT_DIR:-$LOCAL_OUTPUT_DIR/sciegqa/graph_ppr_plain_top224_splade}" \
LABEL_PREFIX="${LABEL_PREFIX:-sciegqa_pagepreserve_sweep}" \
bash "$REPO_ROOT/scripts/run_external_page_preserving_graph_ppr_sweep.sh"
