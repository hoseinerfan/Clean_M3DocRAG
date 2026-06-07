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

DATASETS="${DATASETS:-dude mmdocir sciegqa vidoseek}"
MODEL_JSON="${MODEL_JSON:-$REPO_ROOT/output/m3docvqa_content_aware_auto_blend_sweep_page5/mmqa_train_to_dev_content_aware_base_gpp_no_hyperlink.model.json}"
REPORT_OUT="${REPORT_OUT:-$REPO_ROOT/trained_content_aware_transfer_results.md}"
REPORT_CSV_OUT="${REPORT_CSV_OUT:-$REPO_ROOT/trained_content_aware_transfer_results.csv}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-trained_content_aware_transfer}"

CANDIDATE_TOP_K="${CANDIDATE_TOP_K:-1000}"
INFERENCE_MODE="${INFERENCE_MODE:-}"
BLEND_ALPHA="${BLEND_ALPHA:--1}"
PROMOTION_RANK_MIN="${PROMOTION_RANK_MIN:-5}"
PROMOTION_RANK_MAX="${PROMOTION_RANK_MAX:-200}"
MAX_PROMOTIONS_PER_QID="${MAX_PROMOTIONS_PER_QID:-2}"
PROMOTION_MARGIN="${PROMOTION_MARGIN:-0.05}"
RECALL_K_VALUES="${RECALL_K_VALUES:-1 2 4 5 10 20 50 100}"
RUN_DENSE_BASE="${RUN_DENSE_BASE:-1}"
RUN_GPP_BASE="${RUN_GPP_BASE:-0}"
RUN_DOCSEED_BASE="${RUN_DOCSEED_BASE:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"

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

tag_value() {
  printf '%s' "$1" | tr '.-' 'pm'
}

init_reports() {
  : > "$REPORT_OUT"
  : > "$REPORT_CSV_OUT"
}

append_reports() {
  local display_name="$1"
  local table_md="$2"
  local summary_json="$3"
  if [[ -f "$table_md" ]]; then
    {
      echo "## $display_name"
      echo
      cat "$table_md"
      echo
    } >> "$REPORT_OUT"
  fi
  if [[ -f "$summary_json" ]]; then
    "$PYTHON_BIN" - "$display_name" "$summary_json" "$REPORT_CSV_OUT" <<'PY'
import csv, json, sys
dataset, summary_path, out_path = sys.argv[1:4]
s = json.load(open(summary_path))
rows = s.get("metrics", [])
if not rows:
    raise SystemExit(0)
cols = ["dataset", "label", "qid_count", "doc_eval_count", "page_eval_count", "doc_mrr", "page_mrr"]
for k in [1, 2, 4, 5, 10, 20, 50, 100]:
    cols.extend([f"doc@{k}", f"page@{k}"])
exists = False
try:
    exists = open(out_path).read(1) != ""
except FileNotFoundError:
    exists = False
with open(out_path, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    if not exists:
        w.writeheader()
    for row in rows:
        out = dict(row)
        out["dataset"] = dataset
        w.writerow(out)
PY
  fi
}

run_variant() {
  local display_name="$1"
  local data_name="$2"
  local gold="$3"
  local doc_pages="$4"
  local base_pred="$5"
  local sparse_pred="$6"
  local out_dir="$7"
  local label="$8"
  local pred_out="$out_dir/${label}.prediction.json"
  local summary_out="$out_dir/${label}.summary.json"
  local table_out="$out_dir/${label}.table.md"
  local prior_out="$out_dir/${label}.prior.jsonl"

  if [[ ! -f "$base_pred" ]]; then
    echo "skip_${label}_missing_base_pred=$base_pred" >&2
    return 0
  fi
  if [[ "$FORCE_RERUN" != "1" && -f "$pred_out" && -f "$summary_out" ]]; then
    echo "reuse_${label}=$pred_out"
    append_reports "$display_name $label" "$table_out" "$summary_out"
    return 0
  fi

  echo
  echo "== $display_name trained content-aware transfer: $label =="
  source_args=()
  if [[ -f "$sparse_pred" ]]; then
    source_args+=(--source "sparse=$sparse_pred")
  else
    echo "warning_missing_sparse_source=$sparse_pred" >&2
  fi

  "$PYTHON_BIN" "$REPO_ROOT/scripts/apply_trained_content_aware_page_reranker.py" \
    --model-json "$MODEL_JSON" \
    --base-pred "$base_pred" \
    --page-text-jsonl "$doc_pages" \
    --gold "$gold" \
    "${source_args[@]}" \
    --candidate-top-k "$CANDIDATE_TOP_K" \
    --inference-mode "$INFERENCE_MODE" \
    --blend-alpha "$BLEND_ALPHA" \
    --promotion-rank-min "$PROMOTION_RANK_MIN" \
    --promotion-rank-max "$PROMOTION_RANK_MAX" \
    --max-promotions-per-qid "$MAX_PROMOTIONS_PER_QID" \
    --promotion-margin "$PROMOTION_MARGIN" \
    --recall-k $RECALL_K_VALUES \
    --output-prediction-json "$pred_out" \
    --output-summary-json "$summary_out" \
    --output-table-md "$table_out" \
    --output-prior-jsonl "$prior_out"

  append_reports "$display_name $label" "$table_out" "$summary_out"
}

run_dataset() {
  local display_name="$1"
  local data_name="$2"
  local work_root="$3"
  local output_slug="$4"
  local gold="$5"
  local doc_pages="$6"
  local dense_pred="$7"
  local sparse_pred="$8"
  local gpp_pred="$9"
  local mode_tag
  local alpha_tag
  local out_dir
  local docseed_pred

  mode_tag="${INFERENCE_MODE:-modelmode}"
  alpha_tag="$(tag_value "$BLEND_ALPHA")"
  out_dir="$work_root/output/$output_slug/$OUTPUT_SUBDIR"
  mkdir -p "$out_dir"
  require_file gold "$gold"
  require_file doc_pages "$doc_pages"
  require_file dense_pred "$dense_pred"

  if [[ "$RUN_DENSE_BASE" == "1" ]]; then
    run_variant "$display_name" "$data_name" "$gold" "$doc_pages" "$dense_pred" "$sparse_pred" \
      "$out_dir" "${data_name}_trained_content_transfer_dense_${mode_tag}_a${alpha_tag}"
  fi

  if [[ "$RUN_GPP_BASE" == "1" ]]; then
    run_variant "$display_name" "$data_name" "$gold" "$doc_pages" "$gpp_pred" "$sparse_pred" \
      "$out_dir" "${data_name}_trained_content_transfer_gpp_${mode_tag}_a${alpha_tag}"
  fi

  if [[ "$RUN_DOCSEED_BASE" == "1" ]]; then
    docseed_pred="$work_root/output/$output_slug/doc_seed_ablation/${data_name}_docseed_ablation_docseed_rrf_1p00.prediction.json"
    run_variant "$display_name" "$data_name" "$gold" "$doc_pages" "$docseed_pred" "$sparse_pred" \
      "$out_dir" "${data_name}_trained_content_transfer_docseed_rrf_1p00_${mode_tag}_a${alpha_tag}"
  fi
}

require_file model_json "$MODEL_JSON"
init_reports

for dataset in $DATASETS; do
  case "$dataset" in
    mmdocir)
      require_value MMDocIR_WORK_ROOT
      require_value MMDOCIR_GOLD
      require_value MMDOCIR_DOC_PAGES
      require_value MMDOCIR_DENSE_PRED
      require_value MMDOCIR_SPARSE_PRED
      MMDOCIR_GPP_PRED="${MMDOCIR_GPP_PRED:-$MMDocIR_WORK_ROOT/output/mmdocir/graph_ppr_plain_top224_splade/mmdocir_denseheavy125_medium_both.prediction.json}"
      run_dataset "MMDocIR" mmdocir "$MMDocIR_WORK_ROOT" mmdocir \
        "$MMDOCIR_GOLD" "$MMDOCIR_DOC_PAGES" "$MMDOCIR_DENSE_PRED" "$MMDOCIR_SPARSE_PRED" "$MMDOCIR_GPP_PRED"
      ;;
    sciegqa)
      require_value SciEGQA_WORK_ROOT
      require_value SCIEGQA_GOLD
      require_value SCIEGQA_DOC_PAGES
      require_value SCIEGQA_DENSE_PRED
      require_value SCIEGQA_SPARSE_PRED
      SCIEGQA_GPP_PRED="${SCIEGQA_GPP_PRED:-$SciEGQA_WORK_ROOT/output/sciegqa/graph_ppr_plain_top224_splade/sciegqa_denseheavy125_medium_both.prediction.json}"
      run_dataset "SciEGQA" sciegqa "$SciEGQA_WORK_ROOT" sciegqa \
        "$SCIEGQA_GOLD" "$SCIEGQA_DOC_PAGES" "$SCIEGQA_DENSE_PRED" "$SCIEGQA_SPARSE_PRED" "$SCIEGQA_GPP_PRED"
      ;;
    vidoseek)
      require_value VIDOSEEK_WORK_ROOT
      require_value VIDOSEEK_GOLD
      require_value VIDOSEEK_DOC_PAGES
      require_value VIDOSEEK_DENSE_PRED
      require_value VIDOSEEK_SPARSE_PRED
      VIDOSEEK_GPP_PRED="${VIDOSEEK_GPP_PRED:-$VIDOSEEK_WORK_ROOT/output/vidoseek/graph_ppr_plain_top224_splade/vidoseek_denseheavy125_medium_both.prediction.json}"
      run_dataset "ViDoSeek" vidoseek "$VIDOSEEK_WORK_ROOT" vidoseek \
        "$VIDOSEEK_GOLD" "$VIDOSEEK_DOC_PAGES" "$VIDOSEEK_DENSE_PRED" "$VIDOSEEK_SPARSE_PRED" "$VIDOSEEK_GPP_PRED"
      ;;
    dude)
      require_value DUDE_WORK_ROOT
      require_value DUDE_GOLD
      require_value DUDE_DOC_PAGES
      require_value DUDE_DENSE_PRED
      require_value DUDE_SPARSE_PRED
      DUDE_GPP_PRED="${DUDE_GPP_PRED:-$DUDE_WORK_ROOT/output/dude/graph_ppr_plain_top224_splade/dude_denseheavy125_medium_both.prediction.json}"
      run_dataset "DUDE" dude "$DUDE_WORK_ROOT" dude \
        "$DUDE_GOLD" "$DUDE_DOC_PAGES" "$DUDE_DENSE_PRED" "$DUDE_SPARSE_PRED" "$DUDE_GPP_PRED"
      ;;
    vidore|vidore-v3)
      require_value VIDORE_WORK_ROOT
      require_value VIDORE_GOLD
      require_value VIDORE_DOC_PAGES
      require_value VIDORE_DENSE_PRED
      require_value VIDORE_SPARSE_PRED
      VIDORE_GPP_PRED="${VIDORE_GPP_PRED:-$VIDORE_WORK_ROOT/output/vidore-v3/graph_ppr_plain_top224_splade/vidore-v3_denseheavy125_medium_both.prediction.json}"
      run_dataset "ViDoRe-V3" vidore-v3 "$VIDORE_WORK_ROOT" vidore-v3 \
        "$VIDORE_GOLD" "$VIDORE_DOC_PAGES" "$VIDORE_DENSE_PRED" "$VIDORE_SPARSE_PRED" "$VIDORE_GPP_PRED"
      ;;
    opendocvqa)
      require_value OPENDOCVQA_WORK_ROOT
      require_value OPENDOCVQA_GOLD
      require_value OPENDOCVQA_DOC_PAGES
      require_value OPENDOCVQA_DENSE_PRED
      require_value OPENDOCVQA_SPARSE_PRED
      OPENDOCVQA_GPP_PRED="${OPENDOCVQA_GPP_PRED:-$OPENDOCVQA_WORK_ROOT/output/opendocvqa/graph_ppr_plain_top224_splade/opendocvqa_denseheavy125_medium_both.prediction.json}"
      run_dataset "OpenDocVQA" opendocvqa "$OPENDOCVQA_WORK_ROOT" opendocvqa \
        "$OPENDOCVQA_GOLD" "$OPENDOCVQA_DOC_PAGES" "$OPENDOCVQA_DENSE_PRED" "$OPENDOCVQA_SPARSE_PRED" "$OPENDOCVQA_GPP_PRED"
      ;;
    *)
      echo "unknown_dataset: $dataset" >&2
      exit 1
      ;;
  esac
done

echo "saved_report_md=$REPORT_OUT"
echo "saved_report_csv=$REPORT_CSV_OUT"
