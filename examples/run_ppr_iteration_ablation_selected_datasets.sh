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

DATASETS="${DATASETS:-dude sciegqa opendocvqa}"
REPORT_OUT="${REPORT_OUT:-$REPO_ROOT/ppr_iteration_ablation_results.md}"
REPORT_CSV_OUT="${REPORT_CSV_OUT:-$REPO_ROOT/ppr_iteration_ablation_results.csv}"

DENSE_WEIGHT="${DENSE_WEIGHT:-1.25}"
SPARSE_WEIGHT="${SPARSE_WEIGHT:-0.75}"
RESTART_PROB="${RESTART_PROB:-0.15}"
PAGE_DOC_EDGE_WEIGHT="${PAGE_DOC_EDGE_WEIGHT:-1.0}"
SAME_DOC_WINDOW="${SAME_DOC_WINDOW:-1}"
ADJACENT_PAGE_EDGE_WEIGHT="${ADJACENT_PAGE_EDGE_WEIGHT:-0.25}"
FINAL_PAGE_SEED_WEIGHT="${FINAL_PAGE_SEED_WEIGHT:-1.0}"
FINAL_PPR_PAGE_WEIGHT="${FINAL_PPR_PAGE_WEIGHT:-0.5}"
FINAL_PPR_DOC_WEIGHT="${FINAL_PPR_DOC_WEIGHT:-0.25}"
PPR_GRAPH_SIZE_METRIC="${PPR_GRAPH_SIZE_METRIC:-nodes}"
PPR_GRAPH_SIZE_REFERENCE="${PPR_GRAPH_SIZE_REFERENCE:-1000}"
PPR_GRAPH_SIZE_MIN_ITERS="${PPR_GRAPH_SIZE_MIN_ITERS:-5}"
PPR_GRAPH_SIZE_MAX_ITERS="${PPR_GRAPH_SIZE_MAX_ITERS:-80}"
PPR_CONVERGENCE_TOL="${PPR_CONVERGENCE_TOL:-1e-7}"
PPR_CONVERGENCE_MIN_ITERS="${PPR_CONVERGENCE_MIN_ITERS:-5}"
RECALL_K_VALUES="${RECALL_K_VALUES:-1 2 4 5 10 20 50 100}"

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

run_variant() {
  local data_name="$1"
  local data_root="$2"
  local gold="$3"
  local doc_pages="$4"
  local dense_pred="$5"
  local sparse_pred="$6"
  local out_dir="$7"
  local label_prefix="$8"
  local variant="$9"
  local iteration_mode="${10}"
  local ppr_iters="${11}"

  local label="${label_prefix}_${variant}"
  echo "== $data_name: $variant =="
  DATA_NAME="$data_name" \
  DATA_ROOT="$data_root" \
  GOLD="$gold" \
  DOC_PAGES_JSONL="$doc_pages" \
  DENSE_PRED="$dense_pred" \
  SPARSE_PRED="$sparse_pred" \
  OUT_DIR="$out_dir" \
  GRAPH_PROFILE=page_rank_probe \
  GRAPH_LABEL="$label" \
  FINAL_TOP_PAGES=1000 \
  PER_DOC_PAGE_LIMIT=0 \
  DENSE_WEIGHT="$DENSE_WEIGHT" \
  SPARSE_WEIGHT="$SPARSE_WEIGHT" \
  DOC_SEED_WEIGHT=0.0 \
  DOC_SEED_MODE=rrf \
  SCORE_SEED_WEIGHT=0.0 \
  RESTART_PROB="$RESTART_PROB" \
  PPR_ITERS="$ppr_iters" \
  PPR_ITERATION_MODE="$iteration_mode" \
  PPR_GRAPH_SIZE_METRIC="$PPR_GRAPH_SIZE_METRIC" \
  PPR_GRAPH_SIZE_REFERENCE="$PPR_GRAPH_SIZE_REFERENCE" \
  PPR_GRAPH_SIZE_MIN_ITERS="$PPR_GRAPH_SIZE_MIN_ITERS" \
  PPR_GRAPH_SIZE_MAX_ITERS="$PPR_GRAPH_SIZE_MAX_ITERS" \
  PPR_CONVERGENCE_TOL="$PPR_CONVERGENCE_TOL" \
  PPR_CONVERGENCE_MIN_ITERS="$PPR_CONVERGENCE_MIN_ITERS" \
  PAGE_DOC_EDGE_WEIGHT="$PAGE_DOC_EDGE_WEIGHT" \
  PAGE_TO_DOC_EDGE_WEIGHT="$PAGE_DOC_EDGE_WEIGHT" \
  DOC_TO_PAGE_EDGE_WEIGHT="$PAGE_DOC_EDGE_WEIGHT" \
  SAME_DOC_WINDOW="$SAME_DOC_WINDOW" \
  ADJACENT_PAGE_EDGE_WEIGHT="$ADJACENT_PAGE_EDGE_WEIGHT" \
  FINAL_PAGE_SEED_WEIGHT="$FINAL_PAGE_SEED_WEIGHT" \
  FINAL_PPR_PAGE_WEIGHT="$FINAL_PPR_PAGE_WEIGHT" \
  FINAL_PPR_DOC_WEIGHT="$FINAL_PPR_DOC_WEIGHT" \
  ADAPTIVE_SOURCE_WEIGHT_MODE=none \
  ADAPTIVE_RESTART_MODE=none \
  ADAPTIVE_TRANSITION_MODE=none \
  ADAPTIVE_ADJACENT_MODE=none \
  EVIDENCE_COMMUNITY_MODE=none \
  POSITION_EVIDENCE_MODE=none \
  QUERY_ANCHOR_EVIDENCE_MODE=none \
  HEADING_BREADCRUMB_MODE=none \
  ENTITY_ALIAS_MODE=none \
  CONSTRAINT_COMPETITION_MODE=none \
  PDF_HYPERLINK_EDGES_JSONL= \
  PDF_HYPERLINK_EDGE_WEIGHT=0.0 \
  EXTERNAL_PAGE_GRAPH_JSONL= \
  EXTERNAL_PAGE_GRAPH_EDGE_WEIGHT=0.0 \
  DOC_DOC_EDGE_MODE=none \
  DOC_DOC_EDGE_WEIGHT=0.0 \
  EXPANSION_TOP_PAGES=0 \
  NEIGHBOR_EXPANSION_WINDOW=0 \
  RECALL_K_VALUES="$RECALL_K_VALUES" \
  bash "$REPO_ROOT/scripts/run_external_graph_ppr_pipeline.sh"
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
  out_dir="$work_root/output/$output_slug/ppr_iteration_ablation"

  require_file gold "$gold"
  require_file doc_pages "$doc_pages"
  require_file dense_pred "$dense_pred"
  require_file sparse_pred "$sparse_pred"
  mkdir -p "$out_dir"

  echo
  echo "== $display_name PPR iteration ablation =="
  run_variant "$data_name" "$data_root" "$gold" "$doc_pages" "$dense_pred" "$sparse_pred" \
    "$out_dir" "$label_prefix" fixed30 fixed 30
  run_variant "$data_name" "$data_root" "$gold" "$doc_pages" "$dense_pred" "$sparse_pred" \
    "$out_dir" "$label_prefix" graph_size_nodes graph_size 30
  run_variant "$data_name" "$data_root" "$gold" "$doc_pages" "$dense_pred" "$sparse_pred" \
    "$out_dir" "$label_prefix" convergence_tol1e7 convergence 80

  summary_paths=( "$out_dir/${label_prefix}_"*.summary.json )
  if [[ -e "${summary_paths[0]}" ]]; then
    "$PYTHON_BIN" "$REPO_ROOT/scripts/summarize_graph_ppr_summaries.py" \
      --recall-table \
      --format markdown \
      "${summary_paths[@]}" \
      | tee "$out_dir/${label_prefix}_recall_table.md"

    "$PYTHON_BIN" "$REPO_ROOT/scripts/summarize_graph_ppr_summaries.py" \
      --recall-table \
      --format csv \
      "${summary_paths[@]}" \
      > "$out_dir/${label_prefix}_recall_table.csv"
  fi

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
      run_dataset "MMDocIR" mmdocir mmdocir_ppr_iter_ablation \
        "$MMDocIR_WORK_ROOT" mmdocir "$MMDOCIR_GOLD" "$MMDOCIR_DOC_PAGES" \
        "$MMDOCIR_DENSE_PRED" "$MMDOCIR_SPARSE_PRED"
      ;;
    sciegqa)
      require_value SciEGQA_WORK_ROOT
      require_value SCIEGQA_GOLD
      require_value SCIEGQA_DOC_PAGES
      require_value SCIEGQA_DENSE_PRED
      require_value SCIEGQA_SPARSE_PRED
      run_dataset "SciEGQA" sciegqa sciegqa_ppr_iter_ablation \
        "$SciEGQA_WORK_ROOT" sciegqa "$SCIEGQA_GOLD" "$SCIEGQA_DOC_PAGES" \
        "$SCIEGQA_DENSE_PRED" "$SCIEGQA_SPARSE_PRED"
      ;;
    vidoseek)
      require_value VIDOSEEK_WORK_ROOT
      require_value VIDOSEEK_GOLD
      require_value VIDOSEEK_DOC_PAGES
      require_value VIDOSEEK_DENSE_PRED
      require_value VIDOSEEK_SPARSE_PRED
      run_dataset "ViDoSeek" vidoseek vidoseek_ppr_iter_ablation \
        "$VIDOSEEK_WORK_ROOT" vidoseek "$VIDOSEEK_GOLD" "$VIDOSEEK_DOC_PAGES" \
        "$VIDOSEEK_DENSE_PRED" "$VIDOSEEK_SPARSE_PRED"
      ;;
    dude)
      require_value DUDE_WORK_ROOT
      require_value DUDE_GOLD
      require_value DUDE_DOC_PAGES
      require_value DUDE_DENSE_PRED
      require_value DUDE_SPARSE_PRED
      run_dataset "DUDE" dude dude_ppr_iter_ablation \
        "$DUDE_WORK_ROOT" dude "$DUDE_GOLD" "$DUDE_DOC_PAGES" \
        "$DUDE_DENSE_PRED" "$DUDE_SPARSE_PRED"
      ;;
    vidore|vidore-v3)
      require_value VIDORE_WORK_ROOT
      require_value VIDORE_GOLD
      require_value VIDORE_DOC_PAGES
      require_value VIDORE_DENSE_PRED
      require_value VIDORE_SPARSE_PRED
      run_dataset "ViDoRe-V3" vidore-v3 vidore_ppr_iter_ablation \
        "$VIDORE_WORK_ROOT" vidore-v3 "$VIDORE_GOLD" "$VIDORE_DOC_PAGES" \
        "$VIDORE_DENSE_PRED" "$VIDORE_SPARSE_PRED"
      ;;
    opendocvqa)
      require_value OPENDOCVQA_WORK_ROOT
      require_value OPENDOCVQA_GOLD
      require_value OPENDOCVQA_DOC_PAGES
      require_value OPENDOCVQA_DENSE_PRED
      require_value OPENDOCVQA_SPARSE_PRED
      run_dataset "OpenDocVQA" opendocvqa opendocvqa_ppr_iter_ablation \
        "$OPENDOCVQA_WORK_ROOT" opendocvqa "$OPENDOCVQA_GOLD" "$OPENDOCVQA_DOC_PAGES" \
        "$OPENDOCVQA_DENSE_PRED" "$OPENDOCVQA_SPARSE_PRED"
      ;;
    *)
      echo "unknown_dataset: $dataset" >&2
      exit 1
      ;;
  esac
done

echo "saved_report_md=$REPORT_OUT"
echo "saved_report_csv=$REPORT_CSV_OUT"
