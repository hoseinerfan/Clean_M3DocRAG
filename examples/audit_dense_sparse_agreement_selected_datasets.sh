#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/env/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

VITAL_PATHS_ENV="${VITAL_PATHS_ENV:-$REPO_ROOT/hpc_vital_paths.generated.env}"
if [[ -f "$VITAL_PATHS_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$VITAL_PATHS_ENV"
fi

DATASETS="${DATASETS:-mmdocir sciegqa vidoseek}"

require_value() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "missing_env: $name. Source hpc_vital_paths.generated.env first." >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "missing_summary: $path" >&2
    exit 1
  fi
}

run_audit() {
  local display_name="$1"
  local summary_path="$2"
  require_file "$summary_path"
  echo
  echo "== $display_name dense/sparse agreement audit =="
  "$PYTHON_BIN" "$REPO_ROOT/scripts/audit_dense_sparse_agreement_ablation.py" \
    --strict \
    "$summary_path"
}

for dataset in $DATASETS; do
  case "$dataset" in
    mmdocir)
      require_value MMDocIR_WORK_ROOT
      run_audit "MMDocIR" \
        "$MMDocIR_WORK_ROOT/output/mmdocir/doc_doc_edge_ablation/mmdocir_docdoc_ablation_dense_sparse_agreement.summary.json"
      ;;
    sciegqa)
      require_value SciEGQA_WORK_ROOT
      run_audit "SciEGQA" \
        "$SciEGQA_WORK_ROOT/output/sciegqa/doc_doc_edge_ablation/sciegqa_docdoc_ablation_dense_sparse_agreement.summary.json"
      ;;
    vidoseek)
      require_value VIDOSEEK_WORK_ROOT
      run_audit "ViDoSeek" \
        "$VIDOSEEK_WORK_ROOT/output/vidoseek/doc_doc_edge_ablation/vidoseek_docdoc_ablation_dense_sparse_agreement.summary.json"
      ;;
    dude)
      require_value DUDE_WORK_ROOT
      run_audit "DUDE" \
        "$DUDE_WORK_ROOT/output/dude/doc_doc_edge_ablation/dude_docdoc_ablation_dense_sparse_agreement.summary.json"
      ;;
    *)
      echo "unknown_dataset: $dataset" >&2
      exit 1
      ;;
  esac
done
