#!/usr/bin/env bash
set -euo pipefail

# Audit completed top-k EvidenceGuard/boundary safe-gate outputs and matching policy variants.
#
# This reads existing *.summary.json / *.cases.json files only. It does not run
# graph reranking, PDF Markdown extraction, or the gate itself.
#
# Typical HPC use:
#   HIT_K=8 DATASETS="mmdocir sciegqa vidoseek dude" \
#     bash examples/audit_safe_gate_topk_boundary_cases.sh

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

HIT_K="${HIT_K:-8}"
DATASETS="${DATASETS:-mmdocir sciegqa vidoseek dude}"
PDF_MARKDOWN_BACKEND="${PDF_MARKDOWN_BACKEND:-native}"
POLICY_RUN_LABEL="${POLICY_RUN_LABEL:-${PDF_MARKDOWN_BACKEND}_boundary_top${HIT_K}}"
POLICY_VARIANTS="${POLICY_VARIANTS:-control no_doc_rank_cap require_topk_doc relax_overlap relax_support combined_relaxed}"
SIDE_BY_SIDE_VARIANTS="${SIDE_BY_SIDE_VARIANTS:-evidenceguard_ppr boundary policy_require_topk_doc}"
CASE_LIMIT="${CASE_LIMIT:-8}"
MARKDOWN_CHARS="${MARKDOWN_CHARS:-1400}"
REPORT_ROOT="${REPORT_ROOT:-$REPO_ROOT/output/safe_gate_top${HIT_K}_case_audit}"
mkdir -p "$REPORT_ROOT"

if ! [[ "$HIT_K" =~ ^[1-9][0-9]*$ ]]; then
  echo "invalid_HIT_K: $HIT_K" >&2
  exit 2
fi

if [[ "$HIT_K" == "4" ]]; then
  SAFE_GATE_SUFFIX=""
else
  SAFE_GATE_SUFFIX="_top${HIT_K}"
fi

require_value() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "missing_env: $name. Source hpc_vital_paths.generated.env first." >&2
    exit 1
  fi
}

contains_word() {
  local needle="$1"
  local item
  for item in $SIDE_BY_SIDE_VARIANTS; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

summary_args=()
side_by_side_specs=()
baseline_args=()

add_baseline_prediction() {
  local dataset="$1"
  local prediction="$2"
  if [[ -z "$prediction" ]]; then
    return 0
  fi
  if [[ ! -f "$prediction" ]]; then
    echo "skip_${dataset}_missing_baseline_prediction: $prediction" >&2
    return 0
  fi
  baseline_args+=(--baseline-prediction "$dataset" "$prediction")
}

add_entry() {
  local dataset="$1"
  local variant="$2"
  local gold="$3"
  local out_dir="$4"
  local stem="$5"
  local markdown_jsonl="$out_dir/doc_pages_dev_with_pdf_markdown.jsonl"
  local strict_jsonl="$out_dir/pdf_markdown_variants/doc_pages_dev_pdf_markdown.strict_heading.jsonl"
  local summary="$out_dir/${stem}.summary.json"
  local cases="$out_dir/${stem}.cases.json"

  if [[ ! -f "$summary" || ! -f "$cases" ]]; then
    echo "skip_${dataset}_${variant}_missing_cases_or_summary: $summary" >&2
    return 0
  fi
  summary_args+=(--entry "$dataset" "$variant" "$summary" "$cases")

  if contains_word "$variant"; then
    if [[ ! -f "$markdown_jsonl" ]]; then
      echo "skip_${dataset}_${variant}_missing_markdown_jsonl: $markdown_jsonl" >&2
      return 0
    fi
    side_by_side_specs+=("$dataset|$variant|$gold|$cases|$markdown_jsonl|$strict_jsonl")
  fi
}

add_dataset() {
  local dataset="$1"
  local tag="$2"
  local gold="$3"
  local out_dir="$4"

  add_entry "$dataset" evidenceguard_ppr "$gold" "$out_dir" "${tag}_evidenceguard_ppr_top${HIT_K}"
  add_entry "$dataset" boundary "$gold" "$out_dir" "${tag}_safe_gate_bodyguard${SAFE_GATE_SUFFIX}"

  local variant
  for variant in $POLICY_VARIANTS; do
    add_entry "$dataset" "policy_${variant}" "$gold" "$out_dir" \
      "${tag}_safe_gate_policy_${POLICY_RUN_LABEL}_${variant}"
  done
}

for dataset in $DATASETS; do
  case "$dataset" in
    mmdocir|mm-docir)
      require_value MMDocIR_WORK_ROOT
      require_value MMDOCIR_GOLD
      add_baseline_prediction MMDocIR "${MMDOCIR_DENSE_PRED:-}"
      add_dataset MMDocIR mmdocir "$MMDOCIR_GOLD" \
        "$MMDocIR_WORK_ROOT/output/mmdocir/heading_breadcrumb_pdf_markdown_source_ablation"
      ;;
    sciegqa|sci-egqa)
      require_value SciEGQA_WORK_ROOT
      require_value SCIEGQA_GOLD
      add_baseline_prediction SciEGQA "${SCIEGQA_DENSE_PRED:-}"
      add_dataset SciEGQA sciegqa "$SCIEGQA_GOLD" \
        "$SciEGQA_WORK_ROOT/output/sciegqa/heading_breadcrumb_pdf_markdown_source_ablation"
      ;;
    vidoseek)
      require_value VIDOSEEK_WORK_ROOT
      require_value VIDOSEEK_GOLD
      add_baseline_prediction ViDoSeek "${VIDOSEEK_DENSE_PRED:-}"
      add_dataset ViDoSeek vidoseek "$VIDOSEEK_GOLD" \
        "$VIDOSEEK_WORK_ROOT/output/vidoseek/heading_breadcrumb_pdf_markdown_source_ablation"
      ;;
    dude)
      require_value DUDE_WORK_ROOT
      require_value DUDE_GOLD
      add_baseline_prediction DUDE "${DUDE_DENSE_PRED:-}"
      add_dataset DUDE dude "$DUDE_GOLD" \
        "$DUDE_WORK_ROOT/output/dude/heading_breadcrumb_pdf_markdown_source_ablation"
      ;;
    *)
      echo "unknown_dataset: $dataset" >&2
      exit 2
      ;;
  esac
done

if [[ "${#summary_args[@]}" -eq 0 ]]; then
  echo "no_completed_safe_gate_case_files_found" >&2
  exit 1
fi

"$PYTHON_BIN" "$REPO_ROOT/scripts/summarize_safe_gate_case_diagnostics.py" \
  "${summary_args[@]}" \
  "${baseline_args[@]}" \
  --topn "$CASE_LIMIT" \
  --output-md "$REPORT_ROOT/safe_gate_top${HIT_K}_case_diagnostics.md" \
  --output-json "$REPORT_ROOT/safe_gate_top${HIT_K}_case_diagnostics.json"

for spec in "${side_by_side_specs[@]}"; do
  IFS='|' read -r dataset variant gold cases markdown_jsonl strict_jsonl <<< "$spec"
  doc_page_args=(--doc-pages-jsonl "native=$markdown_jsonl")
  if [[ -f "$strict_jsonl" ]]; then
    doc_page_args+=(--doc-pages-jsonl "strict=$strict_jsonl")
  fi
  for movement in lost recovered; do
    "$PYTHON_BIN" "$REPO_ROOT/scripts/audit_heading_rescue_cases.py" \
      --cases-json "$cases" \
      "${doc_page_args[@]}" \
      --gold "$gold" \
      --movement "$movement" \
      --accepted-only \
      --limit "$CASE_LIMIT" \
      --base-top-n "$HIT_K" \
      --candidate-top-n "$HIT_K" \
      --markdown-chars "$MARKDOWN_CHARS" \
      --output-md "$REPORT_ROOT/${dataset}_${variant}_${movement}_cases.md"
  done
done

echo "report_root: $REPORT_ROOT"
echo "main_report: $REPORT_ROOT/safe_gate_top${HIT_K}_case_diagnostics.md"
