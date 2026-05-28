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
FEATURES="${FEATURES:-shared_entity_title_topic semantic_similarity}"

require_value() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "missing_env: $name. Source hpc_vital_paths.generated.env first." >&2
    exit 1
  fi
}

run_audit() {
  local display_name="$1"
  local feature="$2"
  local summary_path="$3"
  if [[ ! -f "$summary_path" ]]; then
    echo "missing_summary: $summary_path" >&2
    exit 1
  fi
  echo
  echo "== $display_name $feature audit =="
  "$PYTHON_BIN" "$REPO_ROOT/scripts/audit_doc_doc_feature_ablation.py" \
    --strict \
    --feature "$feature" \
    "$summary_path"
}

dataset_summary_root() {
  local dataset="$1"
  case "$dataset" in
    mmdocir)
      require_value MMDocIR_WORK_ROOT
      echo "$MMDocIR_WORK_ROOT/output/mmdocir/doc_doc_edge_ablation/mmdocir_docdoc_ablation"
      ;;
    sciegqa)
      require_value SciEGQA_WORK_ROOT
      echo "$SciEGQA_WORK_ROOT/output/sciegqa/doc_doc_edge_ablation/sciegqa_docdoc_ablation"
      ;;
    vidoseek)
      require_value VIDOSEEK_WORK_ROOT
      echo "$VIDOSEEK_WORK_ROOT/output/vidoseek/doc_doc_edge_ablation/vidoseek_docdoc_ablation"
      ;;
    dude)
      require_value DUDE_WORK_ROOT
      echo "$DUDE_WORK_ROOT/output/dude/doc_doc_edge_ablation/dude_docdoc_ablation"
      ;;
    vidore|vidore-v3)
      require_value VIDORE_WORK_ROOT
      echo "$VIDORE_WORK_ROOT/output/vidore-v3/doc_doc_edge_ablation/vidore_docdoc_ablation"
      ;;
    opendocvqa)
      require_value OPENDOCVQA_WORK_ROOT
      echo "$OPENDOCVQA_WORK_ROOT/output/opendocvqa/doc_doc_edge_ablation/opendocvqa_docdoc_ablation"
      ;;
    *)
      echo "unknown_dataset: $dataset" >&2
      exit 1
      ;;
  esac
}

for dataset in $DATASETS; do
  case "$dataset" in
    mmdocir) display_name="MMDocIR" ;;
    sciegqa) display_name="SciEGQA" ;;
    vidoseek) display_name="ViDoSeek" ;;
    dude) display_name="DUDE" ;;
    vidore|vidore-v3) display_name="ViDoRe-V3" ;;
    opendocvqa) display_name="OpenDocVQA" ;;
    *) display_name="$dataset" ;;
  esac
  prefix="$(dataset_summary_root "$dataset")"
  for feature in $FEATURES; do
    run_audit "$display_name" "$feature" "$prefix"_"$feature.summary.json"
  done
done
