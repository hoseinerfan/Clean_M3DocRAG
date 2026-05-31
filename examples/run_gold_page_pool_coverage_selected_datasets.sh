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

DATASETS="${DATASETS:-dude mmdocir sciegqa vidoseek vidore opendocvqa}"
AUDIT_PRED_SOURCE="${AUDIT_PRED_SOURCE:-dense}"
HIT_K="${HIT_K:-4}"
POOL_K="${POOL_K:-1000}"
COVERAGE_KS="${COVERAGE_KS:-4 20 50 100 200 500 1000}"
REPORT_DIR="${REPORT_DIR:-$REPO_ROOT/output/gold_page_pool_coverage}"
REPORT_OUT="${REPORT_OUT:-$REPORT_DIR/gold_page_pool_coverage_report.md}"
REPORT_CSV_OUT="${REPORT_CSV_OUT:-$REPORT_DIR/gold_page_pool_coverage_report.csv}"
mkdir -p "$REPORT_DIR"

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

choose_prediction() {
  local dense_pred="$1"
  local sparse_pred="$2"
  local custom_pred="$3"
  case "$AUDIT_PRED_SOURCE" in
    dense)
      printf '%s\n' "$dense_pred"
      ;;
    sparse)
      printf '%s\n' "$sparse_pred"
      ;;
    custom)
      printf '%s\n' "$custom_pred"
      ;;
    *)
      echo "unknown_AUDIT_PRED_SOURCE: $AUDIT_PRED_SOURCE" >&2
      exit 1
      ;;
  esac
}

run_dataset() {
  local display_name="$1"
  local label_slug="$2"
  local gold="$3"
  local dense_pred="$4"
  local sparse_pred="$5"
  local custom_pred="${6:-}"
  local prediction
  local audit_json
  local audit_md
  local audit_jsonl

  prediction="$(choose_prediction "$dense_pred" "$sparse_pred" "$custom_pred")"
  require_file gold "$gold"
  require_file prediction "$prediction"
  require_file dense_pred "$dense_pred"
  require_file sparse_pred "$sparse_pred"

  audit_json="$REPORT_DIR/${label_slug}_${AUDIT_PRED_SOURCE}_pool_coverage.json"
  audit_md="$REPORT_DIR/${label_slug}_${AUDIT_PRED_SOURCE}_pool_coverage.md"
  audit_jsonl="$REPORT_DIR/${label_slug}_${AUDIT_PRED_SOURCE}_pool_coverage.failures.jsonl"

  echo
  echo "== $display_name gold page pool coverage =="
  echo "using_gold=$gold"
  echo "using_prediction=$prediction"
  echo "using_dense_pred=$dense_pred"
  echo "using_sparse_pred=$sparse_pred"

  "$PYTHON_BIN" "$REPO_ROOT/scripts/audit_gold_page_candidate_coverage.py" \
    --gold "$gold" \
    --prediction "$prediction" \
    --prediction-label "${label_slug}_${AUDIT_PRED_SOURCE}" \
    --dense-pred "$dense_pred" \
    --sparse-pred "$sparse_pred" \
    --hit-k "$HIT_K" \
    --rank-bins 5 10 20 50 100 200 500 "$POOL_K" \
    --output-md "$audit_md" \
    --output-json "$audit_json" \
    --output-jsonl "$audit_jsonl"

  AUDIT_JSONS+=( "$audit_json" )
}

AUDIT_JSONS=()

for dataset in $DATASETS; do
  case "$dataset" in
    m3docvqa|m3-docvqa|mmqa)
      gold="${M3DOCVQA_GOLD:-$REPO_ROOT/data/m3-docvqa/multimodalqa/MMQA_dev.jsonl}"
      dense="${M3DOCVQA_DENSE_PRED:-${DENSE_PRED:-}}"
      sparse="${M3DOCVQA_SPARSE_PRED:-${SPARSE_PRED:-}}"
      custom="${M3DOCVQA_AUDIT_PRED:-}"
      run_dataset "M3DocVQA/MMQA" m3docvqa "$gold" "$dense" "$sparse" "$custom"
      ;;
    mmdocir)
      require_value MMDOCIR_GOLD
      require_value MMDOCIR_DENSE_PRED
      require_value MMDOCIR_SPARSE_PRED
      run_dataset "MMDocIR" mmdocir "$MMDOCIR_GOLD" "$MMDOCIR_DENSE_PRED" "$MMDOCIR_SPARSE_PRED" "${MMDOCIR_AUDIT_PRED:-}"
      ;;
    sciegqa)
      require_value SCIEGQA_GOLD
      require_value SCIEGQA_DENSE_PRED
      require_value SCIEGQA_SPARSE_PRED
      run_dataset "SciEGQA" sciegqa "$SCIEGQA_GOLD" "$SCIEGQA_DENSE_PRED" "$SCIEGQA_SPARSE_PRED" "${SCIEGQA_AUDIT_PRED:-}"
      ;;
    vidoseek)
      require_value VIDOSEEK_GOLD
      require_value VIDOSEEK_DENSE_PRED
      require_value VIDOSEEK_SPARSE_PRED
      run_dataset "ViDoSeek" vidoseek "$VIDOSEEK_GOLD" "$VIDOSEEK_DENSE_PRED" "$VIDOSEEK_SPARSE_PRED" "${VIDOSEEK_AUDIT_PRED:-}"
      ;;
    dude)
      require_value DUDE_GOLD
      require_value DUDE_DENSE_PRED
      require_value DUDE_SPARSE_PRED
      run_dataset "DUDE" dude "$DUDE_GOLD" "$DUDE_DENSE_PRED" "$DUDE_SPARSE_PRED" "${DUDE_AUDIT_PRED:-}"
      ;;
    vidore|vidore-v3)
      require_value VIDORE_GOLD
      require_value VIDORE_DENSE_PRED
      require_value VIDORE_SPARSE_PRED
      run_dataset "ViDoRe-V3" vidore "$VIDORE_GOLD" "$VIDORE_DENSE_PRED" "$VIDORE_SPARSE_PRED" "${VIDORE_AUDIT_PRED:-}"
      ;;
    opendocvqa)
      require_value OPENDOCVQA_GOLD
      require_value OPENDOCVQA_DENSE_PRED
      require_value OPENDOCVQA_SPARSE_PRED
      run_dataset "OpenDocVQA" opendocvqa "$OPENDOCVQA_GOLD" "$OPENDOCVQA_DENSE_PRED" "$OPENDOCVQA_SPARSE_PRED" "${OPENDOCVQA_AUDIT_PRED:-}"
      ;;
    mmlongbench|mmlongbench-docqa)
      require_value MMLONGBENCH_GOLD
      require_value MMLONGBENCH_DENSE_PRED
      require_value MMLONGBENCH_SPARSE_PRED
      run_dataset "MMLongBench DocQA" mmlongbench "$MMLONGBENCH_GOLD" "$MMLONGBENCH_DENSE_PRED" "$MMLONGBENCH_SPARSE_PRED" "${MMLONGBENCH_AUDIT_PRED:-}"
      ;;
    *)
      echo "unknown_dataset: $dataset" >&2
      exit 1
      ;;
  esac
done

COLLECTOR_INPUT_ARGS=()
for audit_json in "${AUDIT_JSONS[@]}"; do
  COLLECTOR_INPUT_ARGS+=(--input-json "$audit_json")
done

"$PYTHON_BIN" "$REPO_ROOT/scripts/collect_gold_page_candidate_coverage_reports.py" \
  "${COLLECTOR_INPUT_ARGS[@]}" \
  --pool-k "$POOL_K" \
  --hit-k "$HIT_K" \
  --coverage-ks $COVERAGE_KS \
  --output-md "$REPORT_OUT" \
  --output-csv "$REPORT_CSV_OUT"

echo "saved_report_md=$REPORT_OUT"
echo "saved_report_csv=$REPORT_CSV_OUT"
