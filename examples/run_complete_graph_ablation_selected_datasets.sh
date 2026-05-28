#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

VITAL_PATHS_ENV="${VITAL_PATHS_ENV:-$REPO_ROOT/hpc_vital_paths.generated.env}"
if [[ -f "$VITAL_PATHS_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$VITAL_PATHS_ENV"
fi

DATASETS="${DATASETS:-vidore opendocvqa}"
RUN_DOC_DOC="${RUN_DOC_DOC:-1}"
RUN_DOC_SEED="${RUN_DOC_SEED:-1}"
RUN_CROSS_DOC_SELECT="${RUN_CROSS_DOC_SELECT:-1}"
RUN_SANITY="${RUN_SANITY:-1}"
RUN_DENSE_SPARSE_AUDIT="${RUN_DENSE_SPARSE_AUDIT:-1}"
RUN_FEATURE_AUDIT="${RUN_FEATURE_AUDIT:-1}"

if [[ "$RUN_DOC_DOC" == "1" ]]; then
  DATASETS="$DATASETS" bash "$REPO_ROOT/examples/run_doc_doc_edge_ablation_selected_datasets.sh"
fi

if [[ "$RUN_DOC_SEED" == "1" ]]; then
  DATASETS="$DATASETS" bash "$REPO_ROOT/examples/run_doc_seed_ablation_selected_datasets.sh"
fi

if [[ "$RUN_CROSS_DOC_SELECT" == "1" ]]; then
  DATASETS="$DATASETS" bash "$REPO_ROOT/examples/run_cross_doc_page_selection_selected_datasets.sh"
fi

if [[ "$RUN_SANITY" == "1" ]]; then
  DATASETS="$DATASETS" bash "$REPO_ROOT/examples/sanity_check_graph_ablation_selected_datasets.sh"
fi

if [[ "$RUN_DENSE_SPARSE_AUDIT" == "1" ]]; then
  DATASETS="$DATASETS" bash "$REPO_ROOT/examples/audit_dense_sparse_agreement_selected_datasets.sh"
fi

if [[ "$RUN_FEATURE_AUDIT" == "1" ]]; then
  DATASETS="$DATASETS" bash "$REPO_ROOT/examples/audit_doc_doc_features_selected_datasets.sh"
fi
