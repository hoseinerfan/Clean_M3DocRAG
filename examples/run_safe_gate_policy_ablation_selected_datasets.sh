#!/usr/bin/env bash
set -euo pipefail

# Run gate-only policy ablations using completed heading graph predictions.
#
# Run `examples/run_safe_heading_gate_selected_datasets.sh` first. This wrapper
# does not rebuild Markdown or rerun Graph-PPR, and writes uniquely named outputs.
#
# Usage on HPC after a completed native top-8 run:
#   HIT_K=8 DATASETS="mmdocir sciegqa vidoseek dude" \
#     bash examples/run_safe_gate_policy_ablation_selected_datasets.sh

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/env/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

VITAL_PATHS_ENV="${VITAL_PATHS_ENV:-$REPO_ROOT/hpc_vital_paths.generated.env}"
if [[ -f "$VITAL_PATHS_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$VITAL_PATHS_ENV"
fi

DATASETS="${DATASETS:-mmdocir sciegqa vidoseek dude}"
POLICY_VARIANTS="${POLICY_VARIANTS:-control no_doc_rank_cap require_topk_doc relax_overlap relax_support combined_relaxed}"
PDF_MARKDOWN_BACKEND="${PDF_MARKDOWN_BACKEND:-native}"
PDF_MARKDOWN_RUN_SUFFIX="${PDF_MARKDOWN_RUN_SUFFIX:-}"
HIT_K="${HIT_K:-8}"
RUN_GOLD_RANK_AUDIT="${RUN_GOLD_RANK_AUDIT:-0}"
LAYOUT_QUERY_BLOCK="${LAYOUT_QUERY_BLOCK:-(?i)\b(row|column)\b|\b(immediately\s+)?(to\s+the\s+)?(right|left)\s+of\b}"
HEADING_MIN_SCORE_ADVANTAGE="${HEADING_MIN_SCORE_ADVANTAGE:-0.01}"
BODY_MIN_SCORE_ADVANTAGE="${BODY_MIN_SCORE_ADVANTAGE:-0.0}"
SUPPORT_PREDICTION_LABELS=(heuristic strict)
SUPPORT_PREDICTION_COUNT="${#SUPPORT_PREDICTION_LABELS[@]}"

case "$PDF_MARKDOWN_BACKEND" in
  native)
    ;;
  pymupdf4llm)
    PDF_MARKDOWN_RUN_SUFFIX="${PDF_MARKDOWN_RUN_SUFFIX:-_pymupdf4llm}"
    ;;
  *)
    echo "unknown_PDF_MARKDOWN_BACKEND: $PDF_MARKDOWN_BACKEND" >&2
    exit 2
    ;;
esac

if ! [[ "$HIT_K" =~ ^[1-9][0-9]*$ ]]; then
  echo "invalid_HIT_K: $HIT_K (expected a positive integer)" >&2
  exit 2
fi

BOUNDARY_RANK=$((HIT_K + 1))
CONTROL_OVERLAP=$((HIT_K - 1))
RELAXED_OVERLAP=$((CONTROL_OVERLAP > 0 ? CONTROL_OVERLAP - 1 : 0))
CONTROL_SUPPORT_PAGE_VOTES="${CONTROL_SUPPORT_PAGE_VOTES:-$SUPPORT_PREDICTION_COUNT}"
RELAXED_SUPPORT_PAGE_VOTES="${RELAXED_SUPPORT_PAGE_VOTES:-1}"
RUN_LABEL="${SAFE_GATE_POLICY_RUN_LABEL:-${PDF_MARKDOWN_BACKEND}_boundary_top${HIT_K}}"
REPORT_ROOT="${REPORT_ROOT:-$REPO_ROOT/output/safe_gate_policy_ablation}"
REPORT_MD="${REPORT_MD:-$REPORT_ROOT/${RUN_LABEL}_policy_ablation.md}"
REPORT_CSV="${REPORT_CSV:-$REPORT_ROOT/${RUN_LABEL}_policy_ablation.csv}"
mkdir -p "$REPORT_ROOT"

require_value() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "missing_env: $name. Source hpc_vital_paths.generated.env first." >&2
    exit 1
  fi
}

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    echo "missing_${label}: $path" >&2
    echo "Finish the matching full safe-gate run before running gate-only policy ablations." >&2
    exit 1
  fi
}

summary_args=()

run_gate_variant() {
  local dataset="$1"
  local tag="$2"
  local variant="$3"
  local gold="$4"
  local out_dir="$5"
  local base_prediction="$6"
  local candidate_prediction="$7"
  local heuristic_prediction="$8"
  local strict_prediction="$9"
  local markdown_jsonl="${10}"
  local min_overlap="${11}"
  local min_support_votes="${12}"
  local promoted_doc_max_rank="${13}"
  local require_promoted_doc_in_base_topk="${14}"
  local output_stem="${tag}_safe_gate_policy_${RUN_LABEL}_${variant}"
  local output_prediction="$out_dir/${output_stem}.prediction.json"
  local output_summary="$out_dir/${output_stem}.summary.json"
  local output_cases="$out_dir/${output_stem}.cases.json"

  echo
  echo "== $dataset: $variant =="
  local extra_args=()
  if [[ "$require_promoted_doc_in_base_topk" == "1" ]]; then
    extra_args+=(--require-promoted-doc-in-base-topk)
  fi

  echo "min_page_overlap=$min_overlap min_support_page_votes=$min_support_votes promoted_doc_max_base_rank=$promoted_doc_max_rank require_promoted_doc_in_base_topk=$require_promoted_doc_in_base_topk"
  "$PYTHON_BIN" "$REPO_ROOT/scripts/apply_page_rescue_gate.py" \
    --base-prediction "$base_prediction" \
    --candidate-prediction "$candidate_prediction" \
    --support-prediction "heuristic=$heuristic_prediction" \
    --support-prediction "strict=$strict_prediction" \
    --gold "$gold" \
    --hit-k "$HIT_K" \
    --candidate-rank-max "$HIT_K" \
    --rescue-rank-min "$BOUNDARY_RANK" \
    --rescue-rank-max "$BOUNDARY_RANK" \
    --min-page-overlap "$min_overlap" \
    --promoted-doc-max-base-rank "$promoted_doc_max_rank" \
    "${extra_args[@]}" \
    --support-page-rank-max "$HIT_K" \
    --min-support-page-votes "$min_support_votes" \
    --heading-doc-pages-jsonl "$markdown_jsonl" \
    --heading-min-score-advantage "$HEADING_MIN_SCORE_ADVANTAGE" \
    --heading-compare-base-top-k "$HIT_K" \
    --heading-compare-mode displaced_boundary \
    --body-doc-pages-jsonl "$markdown_jsonl" \
    --body-field markdown \
    --body-field text \
    --body-field ocr_text \
    --body-field page_text \
    --body-field content \
    --body-min-score-advantage "$BODY_MIN_SCORE_ADVANTAGE" \
    --body-compare-base-top-k "$HIT_K" \
    --body-compare-mode displaced_boundary \
    --query-block-regex "$LAYOUT_QUERY_BLOCK" \
    --mode swap_promoted \
    --insert-position "$HIT_K" \
    --max-promotions 1 \
    --output-prediction-json "$output_prediction" \
    --output-summary-json "$output_summary" \
    --output-cases-json "$output_cases"

  if [[ "$RUN_GOLD_RANK_AUDIT" == "1" ]]; then
    "$PYTHON_BIN" "$REPO_ROOT/scripts/audit_gold_rank_positions.py" \
      --prediction "$output_prediction" \
      --gold "$gold" \
      --top-k "$HIT_K" \
      --boundary-rank "$BOUNDARY_RANK" \
      --output-json "$out_dir/${output_stem}.gold_rank_positions.json" \
      --output-md "$out_dir/${output_stem}.gold_rank_positions.md"
  fi

  summary_args+=(--summary "$dataset" "$variant" "$output_summary")
}

run_policy_set() {
  local dataset="$1"
  local tag="$2"
  local gold="$3"
  local out_dir="$4"
  local base_prediction="$5"
  local candidate_prediction="$6"
  local heuristic_prediction="$7"
  local strict_prediction="$8"
  local markdown_jsonl="$9"
  local variant
  local overlap
  local support_votes
  local doc_max_rank
  local require_topk_doc

  require_file "${dataset}_gold" "$gold"
  require_file "${dataset}_base_prediction" "$base_prediction"
  require_file "${dataset}_candidate_prediction" "$candidate_prediction"
  require_file "${dataset}_heuristic_prediction" "$heuristic_prediction"
  require_file "${dataset}_strict_prediction" "$strict_prediction"
  require_file "${dataset}_markdown_jsonl" "$markdown_jsonl"

  for variant in $POLICY_VARIANTS; do
    case "$variant" in
      control)
        overlap="$CONTROL_OVERLAP"
        support_votes="$CONTROL_SUPPORT_PAGE_VOTES"
        doc_max_rank="$HIT_K"
        require_topk_doc=0
        ;;
      no_doc_rank_cap)
        overlap="$CONTROL_OVERLAP"
        support_votes="$CONTROL_SUPPORT_PAGE_VOTES"
        doc_max_rank=0
        require_topk_doc=0
        ;;
      require_topk_doc)
        overlap="$CONTROL_OVERLAP"
        support_votes="$CONTROL_SUPPORT_PAGE_VOTES"
        doc_max_rank=0
        require_topk_doc=1
        ;;
      relax_overlap)
        overlap="$RELAXED_OVERLAP"
        support_votes="$CONTROL_SUPPORT_PAGE_VOTES"
        doc_max_rank="$HIT_K"
        require_topk_doc=0
        ;;
      relax_support)
        overlap="$CONTROL_OVERLAP"
        support_votes="$RELAXED_SUPPORT_PAGE_VOTES"
        doc_max_rank="$HIT_K"
        require_topk_doc=0
        ;;
      combined_relaxed)
        overlap="$RELAXED_OVERLAP"
        support_votes="$RELAXED_SUPPORT_PAGE_VOTES"
        doc_max_rank=0
        require_topk_doc=0
        ;;
      *)
        echo "unknown_POLICY_VARIANT: $variant" >&2
        exit 2
        ;;
    esac
    run_gate_variant "$dataset" "$tag" "$variant" "$gold" "$out_dir" \
      "$base_prediction" "$candidate_prediction" "$heuristic_prediction" \
      "$strict_prediction" "$markdown_jsonl" "$overlap" "$support_votes" "$doc_max_rank" \
      "$require_topk_doc"
  done
}

echo "safe_gate_policy_ablation backend=$PDF_MARKDOWN_BACKEND hit_k=$HIT_K boundary_rank=$BOUNDARY_RANK run_label=$RUN_LABEL"
echo "reuses_graph_predictions=1 datasets=[$DATASETS] policies=[$POLICY_VARIANTS] support_views=${SUPPORT_PREDICTION_LABELS[*]} control_support_page_votes=$CONTROL_SUPPORT_PAGE_VOTES relaxed_support_page_votes=$RELAXED_SUPPORT_PAGE_VOTES"

for dataset in $DATASETS; do
  case "$dataset" in
    mmdocir|mm-docir)
      require_value MMDocIR_WORK_ROOT
      require_value MMDOCIR_GOLD
      out_dir="$MMDocIR_WORK_ROOT/output/mmdocir/heading_breadcrumb_pdf_markdown${PDF_MARKDOWN_RUN_SUFFIX}_source_ablation"
      run_policy_set MMDocIR mmdocir "$MMDOCIR_GOLD" "$out_dir" \
        "$out_dir/mmdocir_heading_control_no_heading.prediction.json" \
        "$out_dir/mmdocir_heading_full_wide_edgeonly_transfer.prediction.json" \
        "$out_dir/mmdocir_heading_heuristic_only_wide_edgeonly_transfer.prediction.json" \
        "$out_dir/mmdocir_heading_strict_heading_wide_edgeonly_transfer.prediction.json" \
        "$out_dir/doc_pages_dev_with_pdf_markdown.jsonl"
      ;;
    sciegqa|sci-egqa)
      require_value SciEGQA_WORK_ROOT
      require_value SCIEGQA_GOLD
      out_dir="$SciEGQA_WORK_ROOT/output/sciegqa/heading_breadcrumb_pdf_markdown${PDF_MARKDOWN_RUN_SUFFIX}_source_ablation"
      run_policy_set SciEGQA sciegqa "$SCIEGQA_GOLD" "$out_dir" \
        "$out_dir/sciegqa_heading_control_no_heading.prediction.json" \
        "$out_dir/sciegqa_heading_full_wide_edgeonly_transfer.prediction.json" \
        "$out_dir/sciegqa_heading_heuristic_only_wide_edgeonly_transfer.prediction.json" \
        "$out_dir/sciegqa_heading_strict_heading_wide_edgeonly_transfer.prediction.json" \
        "$out_dir/doc_pages_dev_with_pdf_markdown.jsonl"
      ;;
    vidoseek)
      require_value VIDOSEEK_WORK_ROOT
      require_value VIDOSEEK_GOLD
      require_value VIDOSEEK_DENSE_PRED
      out_dir="$VIDOSEEK_WORK_ROOT/output/vidoseek/heading_breadcrumb_pdf_markdown${PDF_MARKDOWN_RUN_SUFFIX}_source_ablation"
      run_policy_set ViDoSeek vidoseek "$VIDOSEEK_GOLD" "$out_dir" \
        "$VIDOSEEK_DENSE_PRED" \
        "$out_dir/vidoseek_heading_full_wide_edgeonly_transfer.prediction.json" \
        "$out_dir/vidoseek_heading_heuristic_only_wide_edgeonly_transfer.prediction.json" \
        "$out_dir/vidoseek_heading_strict_heading_wide_edgeonly_transfer.prediction.json" \
        "$out_dir/doc_pages_dev_with_pdf_markdown.jsonl"
      ;;
    dude)
      require_value DUDE_WORK_ROOT
      require_value DUDE_GOLD
      out_dir="$DUDE_WORK_ROOT/output/dude/heading_breadcrumb_pdf_markdown${PDF_MARKDOWN_RUN_SUFFIX}_source_ablation"
      run_policy_set DUDE dude "$DUDE_GOLD" "$out_dir" \
        "$out_dir/dude_heading_control_no_heading.prediction.json" \
        "$out_dir/dude_heading_full_wide_edgeonly_transfer.prediction.json" \
        "$out_dir/dude_heading_heuristic_only_wide_edgeonly_transfer.prediction.json" \
        "$out_dir/dude_heading_strict_heading_wide_edgeonly_transfer.prediction.json" \
        "$out_dir/doc_pages_dev_with_pdf_markdown.jsonl"
      ;;
    *)
      echo "unknown_dataset: $dataset" >&2
      exit 2
      ;;
  esac
done

"$PYTHON_BIN" "$REPO_ROOT/scripts/collect_safe_gate_policy_ablation_results.py" \
  "${summary_args[@]}" \
  --output-md "$REPORT_MD" \
  --output-csv "$REPORT_CSV"
