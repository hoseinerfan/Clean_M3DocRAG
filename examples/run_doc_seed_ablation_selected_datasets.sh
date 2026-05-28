#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

VITAL_PATHS_ENV="${VITAL_PATHS_ENV:-$REPO_ROOT/hpc_vital_paths.generated.env}"
if [[ -f "$VITAL_PATHS_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$VITAL_PATHS_ENV"
fi

DATASETS="${DATASETS:-mmdocir sciegqa vidoseek}"
REPORT_OUT="${REPORT_OUT:-$REPO_ROOT/doc_seed_ablation_results.md}"
REPORT_CSV_OUT="${REPORT_CSV_OUT:-$REPO_ROOT/doc_seed_ablation_results.csv}"

require_value() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "missing_env: $name. Source hpc_vital_paths.generated.env first." >&2
    exit 1
  fi
}

require_file() {
  local name="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    echo "missing_${name}: $path" >&2
    exit 1
  fi
}

init_reports() {
  : > "$REPORT_OUT"
  : > "$REPORT_CSV_OUT"
}

append_reports() {
  local display_name="$1"
  local recall_md="$2"
  local recall_csv="$3"

  if [[ -f "$recall_md" ]]; then
    {
      echo "## $display_name"
      echo
      cat "$recall_md"
      echo
    } >> "$REPORT_OUT"
  fi

  if [[ -f "$recall_csv" ]]; then
    if [[ ! -s "$REPORT_CSV_OUT" ]]; then
      head -n 1 "$recall_csv" | awk '{print "dataset," $0}' >> "$REPORT_CSV_OUT"
    fi
    tail -n +2 "$recall_csv" | awk -v dataset="$display_name" '{print dataset "," $0}' >> "$REPORT_CSV_OUT"
  fi
}

run_dataset() {
  local display_name="$1"
  local data_name="$2"
  local label_prefix="$3"
  local work_root="$4"
  local output_slug="$5"
  local gold="$6"
  local doc_pages="$7"
  local dense_pred="$8"
  local sparse_pred="$9"
  local data_root
  local out_dir

  data_root="$(dirname "$gold")"
  out_dir="$work_root/output/$output_slug/doc_seed_ablation"

  require_file gold "$gold"
  require_file doc_pages "$doc_pages"
  require_file dense_pred "$dense_pred"
  require_file sparse_pred "$sparse_pred"
  mkdir -p "$out_dir"

  echo
  echo "== $display_name doc-seed ablation =="
  DATA_NAME="$data_name" \
  DATA_ROOT="$data_root" \
  GOLD="$gold" \
  DOC_PAGES_JSONL="$doc_pages" \
  DENSE_PRED="$dense_pred" \
  SPARSE_PRED="$sparse_pred" \
  OUT_DIR="$out_dir" \
  LABEL_PREFIX="$label_prefix" \
  bash "$REPO_ROOT/scripts/run_external_doc_seed_ablation.sh"

  append_reports "$display_name" \
    "$out_dir/${label_prefix}_recall_table.md" \
    "$out_dir/${label_prefix}_recall_table.csv"
}

init_reports

for dataset in $DATASETS; do
  case "$dataset" in
    mmdocir)
      require_value MMDocIR_WORK_ROOT
      require_value MMDOCIR_GOLD
      require_value MMDOCIR_DOC_PAGES
      require_value MMDOCIR_DENSE_PRED
      require_value MMDOCIR_SPARSE_PRED
      run_dataset "MMDocIR" mmdocir mmdocir_docseed_ablation \
        "$MMDocIR_WORK_ROOT" mmdocir "$MMDOCIR_GOLD" "$MMDOCIR_DOC_PAGES" \
        "$MMDOCIR_DENSE_PRED" "$MMDOCIR_SPARSE_PRED"
      ;;
    sciegqa)
      require_value SciEGQA_WORK_ROOT
      require_value SCIEGQA_GOLD
      require_value SCIEGQA_DOC_PAGES
      require_value SCIEGQA_DENSE_PRED
      require_value SCIEGQA_SPARSE_PRED
      run_dataset "SciEGQA" sciegqa sciegqa_docseed_ablation \
        "$SciEGQA_WORK_ROOT" sciegqa "$SCIEGQA_GOLD" "$SCIEGQA_DOC_PAGES" \
        "$SCIEGQA_DENSE_PRED" "$SCIEGQA_SPARSE_PRED"
      ;;
    vidoseek)
      require_value VIDOSEEK_WORK_ROOT
      require_value VIDOSEEK_GOLD
      require_value VIDOSEEK_DOC_PAGES
      require_value VIDOSEEK_DENSE_PRED
      require_value VIDOSEEK_SPARSE_PRED
      run_dataset "ViDoSeek" vidoseek vidoseek_docseed_ablation \
        "$VIDOSEEK_WORK_ROOT" vidoseek "$VIDOSEEK_GOLD" "$VIDOSEEK_DOC_PAGES" \
        "$VIDOSEEK_DENSE_PRED" "$VIDOSEEK_SPARSE_PRED"
      ;;
    dude)
      require_value DUDE_WORK_ROOT
      require_value DUDE_GOLD
      require_value DUDE_DOC_PAGES
      require_value DUDE_DENSE_PRED
      require_value DUDE_SPARSE_PRED
      run_dataset "DUDE" dude dude_docseed_ablation \
        "$DUDE_WORK_ROOT" dude "$DUDE_GOLD" "$DUDE_DOC_PAGES" \
        "$DUDE_DENSE_PRED" "$DUDE_SPARSE_PRED"
      ;;
    *)
      echo "unknown_dataset: $dataset" >&2
      exit 1
      ;;
  esac
done

echo "saved_report_md=$REPORT_OUT"
echo "saved_report_csv=$REPORT_CSV_OUT"
