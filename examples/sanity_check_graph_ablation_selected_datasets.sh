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

args=()

for dataset in $DATASETS; do
  case "$dataset" in
    mmdocir)
      require_value MMDocIR_WORK_ROOT
      args+=(--doc-doc-summary-glob "$MMDocIR_WORK_ROOT/output/mmdocir/doc_doc_edge_ablation/*.summary.json")
      args+=(--doc-seed-summary-glob "$MMDocIR_WORK_ROOT/output/mmdocir/doc_seed_ablation/*.summary.json")
      ;;
    sciegqa)
      require_value SciEGQA_WORK_ROOT
      args+=(--doc-doc-summary-glob "$SciEGQA_WORK_ROOT/output/sciegqa/doc_doc_edge_ablation/*.summary.json")
      args+=(--doc-seed-summary-glob "$SciEGQA_WORK_ROOT/output/sciegqa/doc_seed_ablation/*.summary.json")
      ;;
    vidoseek)
      require_value VIDOSEEK_WORK_ROOT
      args+=(--doc-doc-summary-glob "$VIDOSEEK_WORK_ROOT/output/vidoseek/doc_doc_edge_ablation/*.summary.json")
      args+=(--doc-seed-summary-glob "$VIDOSEEK_WORK_ROOT/output/vidoseek/doc_seed_ablation/*.summary.json")
      ;;
    dude)
      require_value DUDE_WORK_ROOT
      args+=(--doc-doc-summary-glob "$DUDE_WORK_ROOT/output/dude/doc_doc_edge_ablation/*.summary.json")
      args+=(--doc-seed-summary-glob "$DUDE_WORK_ROOT/output/dude/doc_seed_ablation/*.summary.json")
      ;;
    *)
      echo "unknown_dataset: $dataset" >&2
      exit 1
      ;;
  esac
done

"$PYTHON_BIN" "$REPO_ROOT/scripts/sanity_check_graph_ablation_outputs.py" --strict "${args[@]}"
