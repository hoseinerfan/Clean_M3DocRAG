#!/usr/bin/env bash
set -euo pipefail

# Run the heading/bodyguard rescue gate on selected datasets.
#
# Usage examples on HPC:
#   bash examples/run_safe_heading_gate_selected_datasets.sh
#   DATASETS="m3docvqa dude vidore" bash examples/run_safe_heading_gate_selected_datasets.sh
#   DATASETS="dude" BODY_MIN_SCORE_ADVANTAGE=0.0 bash examples/run_safe_heading_gate_selected_datasets.sh
#   SAFE_GATE_PROFILE=window20 RUN_GOLD_RANK_AUDIT=1 DATASETS="dude vidore" bash examples/run_safe_heading_gate_selected_datasets.sh
#   PDF_MARKDOWN_BACKEND=pymupdf4llm SAFE_GATE_PROFILE=window20 DATASETS="m3docvqa" bash examples/run_safe_heading_gate_selected_datasets.sh
#   PDF_MARKDOWN_BACKEND=pymupdf4llm SAFE_GATE_PROFILE=boundary RUN_GOLD_RANK_AUDIT=1 DATASETS="mmdocir vidoseek sciegqa dude" bash examples/run_safe_heading_gate_selected_datasets.sh
#   NATIVE_CODEGUARD_ABLATION=1 SAFE_GATE_PROFILE=boundary RUN_GOLD_RANK_AUDIT=1 DATASETS="mmdocir" bash examples/run_safe_heading_gate_selected_datasets.sh
#   HIT_K=8 SAFE_GATE_PROFILE=boundary RUN_GOLD_RANK_AUDIT=1 DATASETS="mmdocir" bash examples/run_safe_heading_gate_selected_datasets.sh
#   VIDOSEEK_REJECT_PROMOTED_PAGE_IDX=0 DATASETS="vidoseek" bash examples/run_safe_heading_gate_selected_datasets.sh
#   DUDE_PROMOTED_DOC_MAX_BASE_RANK=1 DATASETS="dude" bash examples/run_safe_heading_gate_selected_datasets.sh
#
# The script assumes the expensive dense/plain_top224 and SPLADE predictions already
# exist. If any are missing it prints the expected path and exits before reranking.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/env/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

DATASETS="${DATASETS:-m3docvqa dude vidore}"

LAYOUT_QUERY_BLOCK="${LAYOUT_QUERY_BLOCK:-(?i)\b(row|column)\b|\b(immediately\s+)?(to\s+the\s+)?(right|left)\s+of\b}"
HEADING_MIN_SCORE_ADVANTAGE="${HEADING_MIN_SCORE_ADVANTAGE:-0.01}"
BODY_MIN_SCORE_ADVANTAGE="${BODY_MIN_SCORE_ADVANTAGE:-0.0}"
SAFE_GATE_PROFILE="${SAFE_GATE_PROFILE:-boundary}"
HIT_K="${HIT_K:-4}"
INSERT_POSITION="${INSERT_POSITION:-$HIT_K}"
MAX_PROMOTIONS="${MAX_PROMOTIONS:-1}"
RUN_GOLD_RANK_AUDIT="${RUN_GOLD_RANK_AUDIT:-0}"
REJECT_PROMOTED_PAGE_IDX="${REJECT_PROMOTED_PAGE_IDX:-}"
HPC_PATH_ENV="${HPC_PATH_ENV:-}"
PDF_MARKDOWN_BACKEND="${PDF_MARKDOWN_BACKEND:-native}"
PDF_MARKDOWN_RUN_SUFFIX="${PDF_MARKDOWN_RUN_SUFFIX:-}"
PDF_MARKDOWN_FORCE_REBUILD="${PDF_MARKDOWN_FORCE_REBUILD:-0}"
NATIVE_CODEGUARD_ABLATION="${NATIVE_CODEGUARD_ABLATION:-0}"
SUPPORT_PREDICTION_LABELS=(heuristic strict)
SUPPORT_PREDICTION_COUNT="${#SUPPORT_PREDICTION_LABELS[@]}"

if [[ -n "$HPC_PATH_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$HPC_PATH_ENV"
elif [[ -f "$REPO_ROOT/hpc_vital_paths.generated.env" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/hpc_vital_paths.generated.env"
fi

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

if [[ "$NATIVE_CODEGUARD_ABLATION" != "0" && "$NATIVE_CODEGUARD_ABLATION" != "1" ]]; then
  echo "unknown_NATIVE_CODEGUARD_ABLATION: $NATIVE_CODEGUARD_ABLATION" >&2
  exit 2
fi
if [[ "$NATIVE_CODEGUARD_ABLATION" == "1" && "$PDF_MARKDOWN_BACKEND" != "native" ]]; then
  echo "NATIVE_CODEGUARD_ABLATION=1 requires PDF_MARKDOWN_BACKEND=native" >&2
  exit 2
fi

if ! [[ "$HIT_K" =~ ^[1-9][0-9]*$ ]]; then
  echo "invalid_HIT_K: $HIT_K (expected a positive integer)" >&2
  exit 2
fi

BOUNDARY_RANK=$((HIT_K + 1))
MIN_CONSERVATIVE_OVERLAP=$((HIT_K - 1))
HIT_K_OUTPUT_SUFFIX=""
if [[ "$HIT_K" != "4" ]]; then
  HIT_K_OUTPUT_SUFFIX="_top${HIT_K}"
fi

case "$SAFE_GATE_PROFILE" in
  boundary)
    SAFE_GATE_OUTPUT_SUFFIX="${SAFE_GATE_OUTPUT_SUFFIX:-safe_gate_bodyguard${HIT_K_OUTPUT_SUFFIX}}"
    CANDIDATE_RANK_MAX="${CANDIDATE_RANK_MAX:-$HIT_K}"
    RESCUE_RANK_MIN="${RESCUE_RANK_MIN:-$BOUNDARY_RANK}"
    RESCUE_RANK_MAX="${RESCUE_RANK_MAX:-$BOUNDARY_RANK}"
    MIN_PAGE_OVERLAP="${MIN_PAGE_OVERLAP:-$MIN_CONSERVATIVE_OVERLAP}"
    SUPPORT_PAGE_RANK_MAX="${SUPPORT_PAGE_RANK_MAX:-$HIT_K}"
    MIN_SUPPORT_PAGE_VOTES="${MIN_SUPPORT_PAGE_VOTES:-$SUPPORT_PREDICTION_COUNT}"
    ;;
  window20)
    SAFE_GATE_OUTPUT_SUFFIX="${SAFE_GATE_OUTPUT_SUFFIX:-safe_window20_gate_bodyguard${HIT_K_OUTPUT_SUFFIX}}"
    CANDIDATE_RANK_MAX="${CANDIDATE_RANK_MAX:-20}"
    RESCUE_RANK_MIN="${RESCUE_RANK_MIN:-5}"
    RESCUE_RANK_MAX="${RESCUE_RANK_MAX:-20}"
    MIN_PAGE_OVERLAP="${MIN_PAGE_OVERLAP:-3}"
    SUPPORT_PAGE_RANK_MAX="${SUPPORT_PAGE_RANK_MAX:-20}"
    MIN_SUPPORT_PAGE_VOTES="${MIN_SUPPORT_PAGE_VOTES:-$SUPPORT_PREDICTION_COUNT}"
    ;;
  *)
    echo "unknown_SAFE_GATE_PROFILE: $SAFE_GATE_PROFILE" >&2
    exit 2
    ;;
esac

HEADING_COMPARE_BASE_TOP_K="${HEADING_COMPARE_BASE_TOP_K:-$HIT_K}"
BODY_COMPARE_BASE_TOP_K="${BODY_COMPARE_BASE_TOP_K:-$HIT_K}"
GOLD_RANK_AUDIT_BOUNDARY_RANK="${GOLD_RANK_AUDIT_BOUNDARY_RANK:-$BOUNDARY_RANK}"

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    echo "missing_${label}: $path" >&2
    return 1
  fi
}

first_existing_file() {
  local path
  for path in "$@"; do
    if [[ -f "$path" ]]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
  return 1
}

recorded_pdf_root() {
  local summary_json="$1"
  local markdown_jsonl="$2"
  "$PYTHON_BIN" - "$summary_json" "$markdown_jsonl" <<'PY'
import json
import os
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
markdown_path = Path(sys.argv[2])
if summary_path.is_file():
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    for root in summary.get("pdf_roots") or []:
        if str(root).strip():
            print(str(root).strip())
            raise SystemExit(0)

paths = []
if markdown_path.is_file():
    with markdown_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            pdf_path = str(row.get("pdf_path") or "").strip()
            if pdf_path:
                paths.append(pdf_path)
if paths:
    root = Path(os.path.commonpath(paths))
    print(str(root.parent if root.is_file() else root))
PY
}

print_summary_row() {
  local dataset="$1"
  local summary_json="$2"
  "$PYTHON_BIN" - "$dataset" "$summary_json" <<'PY'
import json
import sys

dataset, path = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as handle:
    s = json.load(handle)
page_metrics_available = s.get("page_metrics_available", True)
def report(value):
    return "NA" if value is None else value
print(
    "| {dataset} | {accepted} | {base_page} | {candidate_page} | {page} | "
    "{recovered} | {lost} | {net} | {body_rejects} |".format(
        dataset=dataset,
        accepted=s.get("accepted_count"),
        base_page=report(s.get("base_page_hit_at_k_count") if page_metrics_available else None),
        candidate_page=report(s.get("candidate_page_hit_at_k_count") if page_metrics_available else None),
        page=report(s.get("page_hit_at_k_count") if page_metrics_available else None),
        recovered=report(s.get("recovered") if page_metrics_available else None),
        lost=report(s.get("lost") if page_metrics_available else None),
        net=report(s.get("net_recovered") if page_metrics_available else None),
        body_rejects=(s.get("rejected_promotion_reason_counts") or {}).get(
            "promoted_body_score_not_above_base", 0
        ),
    )
)
if not page_metrics_available and s.get("doc_metrics_available"):
    print(
        "# {dataset}: page gold unavailable; doc@{k} base={base} candidate={candidate} "
        "gated={output} doc_net={net}".format(
            dataset=dataset,
            k=(s.get("config") or {}).get("hit_k", 4),
            base=s.get("base_doc_hit_at_k_count"),
            candidate=s.get("candidate_doc_hit_at_k_count"),
            output=s.get("doc_hit_at_k_count"),
            net=s.get("doc_net_recovered"),
        )
    )
PY
}

run_graph_view() {
  local data_name="$1"
  local data_root="$2"
  local gold="$3"
  local dense_pred="$4"
  local sparse_pred="$5"
  local out_dir="$6"
  local doc_pages_jsonl="$7"
  local graph_label="$8"
  local heading_mode="$9"

  mkdir -p "$out_dir"
  DATA_NAME="$data_name" \
  DATA_ROOT="$data_root" \
  GOLD="$gold" \
  DENSE_PRED="$dense_pred" \
  SPARSE_PRED="$sparse_pred" \
  OUT_DIR="$out_dir" \
  DOC_PAGES_JSONL="$doc_pages_jsonl" \
  GRAPH_PROFILE=page_rank_probe \
  GRAPH_LABEL="$graph_label" \
  FINAL_TOP_PAGES=1000 \
  PER_DOC_PAGE_LIMIT=0 \
  DENSE_WEIGHT=1.25 \
  SPARSE_WEIGHT=0.75 \
  RESTART_PROB=0.15 \
  PPR_ITERS=30 \
  PAGE_DOC_EDGE_WEIGHT=1.0 \
  SAME_DOC_WINDOW=1 \
  ADJACENT_PAGE_EDGE_WEIGHT=0.25 \
  FINAL_PAGE_SEED_WEIGHT=1.0 \
  FINAL_PPR_PAGE_WEIGHT=0.5 \
  FINAL_PPR_DOC_WEIGHT=0.25 \
  HEADING_BREADCRUMB_MODE="$heading_mode" \
  HEADING_BREADCRUMB_FIELD=markdown \
  HEADING_BREADCRUMB_EDGE_WEIGHT=0.10 \
  HEADING_BREADCRUMB_RESTART_WEIGHT=0.0 \
  HEADING_BREADCRUMB_MAX_PAGE_MATCHES=0 \
  HEADING_BREADCRUMB_MAX_DOC_MATCHES=0 \
  bash "$REPO_ROOT/scripts/run_external_graph_ppr_pipeline.sh"
}

prepare_pdf_markdown() {
  local doc_pages_jsonl="$1"
  local pdf_root="$2"
  local output_jsonl="$3"
  local summary_json="$4"
  local variant_dir="$5"

  if [[ "$PDF_MARKDOWN_FORCE_REBUILD" == "1" || ! -f "$output_jsonl" || ! -f "$summary_json" ]]; then
    "$PYTHON_BIN" "$REPO_ROOT/scripts/export_pdf_page_markdown.py" \
      --doc-pages-jsonl "$doc_pages_jsonl" \
      --pdf-root "$pdf_root" \
      --backend "$PDF_MARKDOWN_BACKEND" \
      --output-jsonl "$output_jsonl" \
      --output-summary-json "$summary_json" \
      --progress-every 1000 \
      --body-char-limit 6000
  fi

  "$PYTHON_BIN" "$REPO_ROOT/scripts/prepare_pdf_markdown_variants.py" \
    --input-jsonl "$output_jsonl" \
    --output-dir "$variant_dir"
}

run_safe_gate() {
  local dataset_label="$1"
  local gold="$2"
  local out_dir="$3"
  local base_pred="$4"
  local candidate_pred="$5"
  local heuristic_pred="$6"
  local strict_pred="$7"
  local heading_doc_pages="$8"
  local body_doc_pages="$9"
  local output_stem="${10}"
  local promoted_doc_max_base_rank="${11:-$HIT_K}"
  local reject_promoted_page_idx="${12:-$REJECT_PROMOTED_PAGE_IDX}"
  local output_prediction="$out_dir/${output_stem}.prediction.json"
  local output_summary="$out_dir/${output_stem}.summary.json"
  local output_cases="$out_dir/${output_stem}.cases.json"
  local extra_args=()

  for page_idx in $reject_promoted_page_idx; do
    extra_args+=(--reject-promoted-page-idx "$page_idx")
  done

  "$PYTHON_BIN" "$REPO_ROOT/scripts/apply_page_rescue_gate.py" \
    --base-prediction "$base_pred" \
    --candidate-prediction "$candidate_pred" \
    --support-prediction "heuristic=$heuristic_pred" \
    --support-prediction "strict=$strict_pred" \
    --gold "$gold" \
    --hit-k "$HIT_K" \
    --candidate-rank-max "$CANDIDATE_RANK_MAX" \
    --rescue-rank-min "$RESCUE_RANK_MIN" \
    --rescue-rank-max "$RESCUE_RANK_MAX" \
    --min-page-overlap "$MIN_PAGE_OVERLAP" \
    --promoted-doc-max-base-rank "$promoted_doc_max_base_rank" \
    --support-page-rank-max "$SUPPORT_PAGE_RANK_MAX" \
    --min-support-page-votes "$MIN_SUPPORT_PAGE_VOTES" \
    --heading-doc-pages-jsonl "$heading_doc_pages" \
    --heading-min-score-advantage "$HEADING_MIN_SCORE_ADVANTAGE" \
    --heading-compare-base-top-k "$HEADING_COMPARE_BASE_TOP_K" \
    --heading-compare-mode displaced_boundary \
    --body-doc-pages-jsonl "$body_doc_pages" \
    --body-field markdown \
    --body-field text \
    --body-field ocr_text \
    --body-field page_text \
    --body-field content \
    --body-min-score-advantage "$BODY_MIN_SCORE_ADVANTAGE" \
    --body-compare-base-top-k "$BODY_COMPARE_BASE_TOP_K" \
    --body-compare-mode displaced_boundary \
    --query-block-regex "$LAYOUT_QUERY_BLOCK" \
    "${extra_args[@]}" \
    --mode swap_promoted \
    --insert-position "$INSERT_POSITION" \
    --max-promotions "$MAX_PROMOTIONS" \
    --output-prediction-json "$output_prediction" \
    --output-summary-json "$output_summary" \
    --output-cases-json "$output_cases"

  print_summary_row "$dataset_label" "$output_summary"

  if [[ "$RUN_GOLD_RANK_AUDIT" == "1" && "$dataset_label" == "m3docvqa" ]]; then
    echo "# m3docvqa: skipping page-position audit because MMQA gold has document labels only"
    "$PYTHON_BIN" "$REPO_ROOT/scripts/analyze_m3docvqa_retrieval.py" \
      --pred "$output_prediction" \
      --gold "$gold" \
      --recall-k 1 2 "$HIT_K" "$BOUNDARY_RANK" 10 20 \
      --summary-only
  elif [[ "$RUN_GOLD_RANK_AUDIT" == "1" ]]; then
    "$PYTHON_BIN" "$REPO_ROOT/scripts/audit_gold_rank_positions.py" \
      --prediction "$output_prediction" \
      --gold "$gold" \
      --top-k "$HIT_K" \
      --boundary-rank "$GOLD_RANK_AUDIT_BOUNDARY_RANK" \
      --output-json "$out_dir/${output_stem}.gold_rank_positions.json" \
      --output-md "$out_dir/${output_stem}.gold_rank_positions.md"
  fi
}

run_m3docvqa() {
  echo
  echo "== m3docvqa =="
  unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR LOCAL_MODEL_DIR
  unset M3DOCVQA_INTERNAL_ENV_LOADED
  # shellcheck disable=SC1091
  source "$REPO_ROOT/scripts/m3docvqa_internal_env.sh"

  local data_root="$DATASET_ROOT"
  local gold="${M3DOCVQA_GOLD:-$GOLD}"
  local out_dir="$LOCAL_OUTPUT_DIR/m3docvqa_heading_breadcrumb_pdf_markdown${PDF_MARKDOWN_RUN_SUFFIX}_source_ablation"
  local page_text_dir="$LOCAL_OUTPUT_DIR/m3docvqa_page_text"
  local page_text_jsonl="${M3DOCVQA_PAGE_TEXT_JSONL:-$page_text_dir/m3docvqa_dev_page_text.jsonl}"
  local pdf_markdown_jsonl="$out_dir/doc_pages_dev_with_pdf_markdown.jsonl"
  local pdf_markdown_summary="$out_dir/pdf_markdown_summary.json"
  local variant_dir="$out_dir/pdf_markdown_variants"
  local dense_pred
  dense_pred="$(
    first_existing_file \
      "${M3DOCVQA_DENSE_PRED:-}" \
      "$LOCAL_OUTPUT_DIR/m3docvqa_plain_top224_mmqa_dev/mmqa_dev_plain_top224_nprobe4_effdiag_all.prediction.json" \
      "$REPO_ROOT/output/m3docvqa_plain_top224_mmqa_dev/mmqa_dev_plain_top224_nprobe4_effdiag_all.prediction.json" \
      "/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/mmqa_dev_plain_top224_nprobe4_effdiag_all.prediction.json" \
      || true
  )"
  local sparse_pred
  sparse_pred="$(
    first_existing_file \
      "${M3DOCVQA_SPARSE_PRED:-}" \
      "$LOCAL_OUTPUT_DIR/m3docvqa_splade_mmqa_dev/mmqa_dev_splade.prediction.json" \
      "$REPO_ROOT/output/m3docvqa_splade_mmqa_dev/mmqa_dev_splade.prediction.json" \
      "/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_splade_mmqa_dev/mmqa_dev_splade.prediction.json" \
      || true
  )"

  require_file gold "$gold"
  if [[ -z "$dense_pred" ]]; then
    echo "missing_dense_pred. Set M3DOCVQA_DENSE_PRED=/path/to/mmqa_dev_plain_top224*.prediction.json" >&2
    echo "hint: find /mmfs1/scratch/jacks.local/aerfanshekooh/custom -type f -name '*plain_top224*.prediction.json' | grep -E 'mmqa|m3docvqa' | sort" >&2
    return 1
  fi
  if [[ -z "$sparse_pred" ]]; then
    echo "missing_sparse_pred. Set M3DOCVQA_SPARSE_PRED=/path/to/mmqa_dev_splade.prediction.json" >&2
    echo "hint: find /mmfs1/scratch/jacks.local/aerfanshekooh/custom -type f -name '*splade*.prediction.json' | grep -E 'mmqa|m3docvqa' | sort" >&2
    return 1
  fi
  mkdir -p "$out_dir" "$page_text_dir"

  if [[ ! -f "$page_text_jsonl" ]]; then
    OUTPUT_JSONL="$page_text_jsonl" \
    OUTPUT_SUMMARY_JSON="$page_text_dir/m3docvqa_dev_page_text.summary.json" \
    bash "$REPO_ROOT/scripts/run_m3docvqa_page_text_export.sh"
  fi

  prepare_pdf_markdown "$page_text_jsonl" "$data_root" "$pdf_markdown_jsonl" "$pdf_markdown_summary" "$variant_dir"

  local tag="m3docvqa"
  run_graph_view "m3-docvqa" "$data_root" "$gold" "$dense_pred" "$sparse_pred" "$out_dir" "$pdf_markdown_jsonl" "${tag}_heading_control_no_heading" none
  run_graph_view "m3-docvqa" "$data_root" "$gold" "$dense_pred" "$sparse_pred" "$out_dir" "$pdf_markdown_jsonl" "${tag}_heading_full_wide_edgeonly_transfer" query_gated_shared
  run_graph_view "m3-docvqa" "$data_root" "$gold" "$dense_pred" "$sparse_pred" "$out_dir" "$variant_dir/doc_pages_dev_pdf_markdown.heuristic_only.jsonl" "${tag}_heading_heuristic_only_wide_edgeonly_transfer" query_gated_shared
  run_graph_view "m3-docvqa" "$data_root" "$gold" "$dense_pred" "$sparse_pred" "$out_dir" "$variant_dir/doc_pages_dev_pdf_markdown.strict_heading.jsonl" "${tag}_heading_strict_heading_wide_edgeonly_transfer" query_gated_shared

  local strict_support_pred="$out_dir/${tag}_heading_strict_heading_wide_edgeonly_transfer.prediction.json"
  local heading_evidence_jsonl="$pdf_markdown_jsonl"
  local gate_suffix="$SAFE_GATE_OUTPUT_SUFFIX"
  if [[ "$NATIVE_CODEGUARD_ABLATION" == "1" ]]; then
    run_graph_view "m3-docvqa" "$data_root" "$gold" "$dense_pred" "$sparse_pred" "$out_dir" "$variant_dir/doc_pages_dev_pdf_markdown.strict_heading_codeguard.jsonl" "${tag}_heading_strict_heading_codeguard_wide_edgeonly_transfer" query_gated_shared
    strict_support_pred="$out_dir/${tag}_heading_strict_heading_codeguard_wide_edgeonly_transfer.prediction.json"
    heading_evidence_jsonl="$variant_dir/doc_pages_dev_pdf_markdown.strict_heading_codeguard.jsonl"
    gate_suffix="${SAFE_GATE_OUTPUT_SUFFIX}_codeguard"
  fi

  run_safe_gate \
    m3docvqa \
    "$gold" \
    "$out_dir" \
    "$out_dir/${tag}_heading_control_no_heading.prediction.json" \
    "$out_dir/${tag}_heading_full_wide_edgeonly_transfer.prediction.json" \
    "$out_dir/${tag}_heading_heuristic_only_wide_edgeonly_transfer.prediction.json" \
    "$strict_support_pred" \
    "$heading_evidence_jsonl" \
    "$pdf_markdown_jsonl" \
    "${tag}_${gate_suffix}" \
    "$HIT_K"
}

run_dude() {
  echo
  echo "== dude =="
  unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR LOCAL_MODEL_DIR
  # shellcheck disable=SC1091
  source "$REPO_ROOT/dude/env_hpc.sh"

  local data_root="$LOCAL_DATA_DIR/dude"
  local gold="${DUDE_GOLD:-$data_root/MMQA_dev.jsonl}"
  local doc_pages_jsonl="${DUDE_DOC_PAGES:-$data_root/doc_pages_dev.jsonl}"
  local out_dir="$LOCAL_OUTPUT_DIR/dude/heading_breadcrumb_pdf_markdown${PDF_MARKDOWN_RUN_SUFFIX}_source_ablation"
  local pdf_root="${DUDE_PDF_ROOT:-$data_root/raw/DUDE_train-val-test_binaries/PDF}"
  local pdf_markdown_jsonl="$out_dir/doc_pages_dev_with_pdf_markdown.jsonl"
  local pdf_markdown_summary="$out_dir/pdf_markdown_summary.json"
  local variant_dir="$out_dir/pdf_markdown_variants"
  local dense_pred="$LOCAL_OUTPUT_DIR/dude/plain_top224_ret1000_prediction.json"
  dense_pred="${DUDE_DENSE_PRED:-$dense_pred}"
  local sparse_pred="$LOCAL_OUTPUT_DIR/dude/doc_rrf_plain_top224_splade/dude_splade_ret1000.prediction.json"
  sparse_pred="${DUDE_SPARSE_PRED:-$sparse_pred}"

  require_file gold "$gold"
  require_file doc_pages "$doc_pages_jsonl"
  require_file dense_pred "$dense_pred"
  require_file sparse_pred "$sparse_pred"
  mkdir -p "$out_dir"

  prepare_pdf_markdown "$doc_pages_jsonl" "$pdf_root" "$pdf_markdown_jsonl" "$pdf_markdown_summary" "$variant_dir"

  local tag="dude"
  run_graph_view dude "$data_root" "$gold" "$dense_pred" "$sparse_pred" "$out_dir" "$pdf_markdown_jsonl" "${tag}_heading_control_no_heading" none
  run_graph_view dude "$data_root" "$gold" "$dense_pred" "$sparse_pred" "$out_dir" "$pdf_markdown_jsonl" "${tag}_heading_full_wide_edgeonly_transfer" query_gated_shared
  run_graph_view dude "$data_root" "$gold" "$dense_pred" "$sparse_pred" "$out_dir" "$variant_dir/doc_pages_dev_pdf_markdown.heuristic_only.jsonl" "${tag}_heading_heuristic_only_wide_edgeonly_transfer" query_gated_shared
  run_graph_view dude "$data_root" "$gold" "$dense_pred" "$sparse_pred" "$out_dir" "$variant_dir/doc_pages_dev_pdf_markdown.strict_heading.jsonl" "${tag}_heading_strict_heading_wide_edgeonly_transfer" query_gated_shared

  local strict_support_pred="$out_dir/${tag}_heading_strict_heading_wide_edgeonly_transfer.prediction.json"
  local heading_evidence_jsonl="$pdf_markdown_jsonl"
  local gate_suffix="$SAFE_GATE_OUTPUT_SUFFIX"
  if [[ "$NATIVE_CODEGUARD_ABLATION" == "1" ]]; then
    run_graph_view dude "$data_root" "$gold" "$dense_pred" "$sparse_pred" "$out_dir" "$variant_dir/doc_pages_dev_pdf_markdown.strict_heading_codeguard.jsonl" "${tag}_heading_strict_heading_codeguard_wide_edgeonly_transfer" query_gated_shared
    strict_support_pred="$out_dir/${tag}_heading_strict_heading_codeguard_wide_edgeonly_transfer.prediction.json"
    heading_evidence_jsonl="$variant_dir/doc_pages_dev_pdf_markdown.strict_heading_codeguard.jsonl"
    gate_suffix="${SAFE_GATE_OUTPUT_SUFFIX}_codeguard"
  fi

  local promoted_doc_max_base_rank="${DUDE_PROMOTED_DOC_MAX_BASE_RANK:-$HIT_K}"
  local default_dude_suffix="$gate_suffix"
  if [[ -n "${DUDE_PROMOTED_DOC_MAX_BASE_RANK+x}" ]]; then
    default_dude_suffix="${gate_suffix}_docrank${promoted_doc_max_base_rank}"
  fi
  local dude_suffix="${DUDE_SAFE_GATE_OUTPUT_SUFFIX:-$default_dude_suffix}"
  run_safe_gate \
    dude \
    "$gold" \
    "$out_dir" \
    "$out_dir/${tag}_heading_control_no_heading.prediction.json" \
    "$out_dir/${tag}_heading_full_wide_edgeonly_transfer.prediction.json" \
    "$out_dir/${tag}_heading_heuristic_only_wide_edgeonly_transfer.prediction.json" \
    "$strict_support_pred" \
    "$heading_evidence_jsonl" \
    "$pdf_markdown_jsonl" \
    "${tag}_${dude_suffix}" \
    "$promoted_doc_max_base_rank"
}

run_mmdocir() {
  echo
  echo "== mmdocir =="
  unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR LOCAL_MODEL_DIR
  # shellcheck disable=SC1091
  source "$REPO_ROOT/mmdocir/env_hpc.sh"

  local data_root="$LOCAL_DATA_DIR/mm-docir"
  local gold="${MMDOCIR_GOLD:-$data_root/MMQA_dev.jsonl}"
  local doc_pages_jsonl="${MMDOCIR_DOC_PAGES:-$data_root/doc_pages_dev.jsonl}"
  local native_pdf_markdown_jsonl="${MMDOCIR_PDF_MD_JSONL:-$LOCAL_OUTPUT_DIR/mmdocir/pdf_markdown/doc_pages_dev_with_pdf_markdown.jsonl}"
  local native_summary="${MMDOCIR_NATIVE_PDF_MD_SUMMARY:-$(dirname "$native_pdf_markdown_jsonl")/pdf_markdown_summary.json}"
  local out_dir="$LOCAL_OUTPUT_DIR/mmdocir/heading_breadcrumb_pdf_markdown${PDF_MARKDOWN_RUN_SUFFIX}_source_ablation"
  local pdf_root="${MMDOCIR_PDF_ROOT:-}"
  local pdf_markdown_jsonl="$out_dir/doc_pages_dev_with_pdf_markdown.jsonl"
  local pdf_markdown_summary="$out_dir/pdf_markdown_summary.json"
  local variant_dir="$out_dir/pdf_markdown_variants"
  local dense_pred="${MMDOCIR_DENSE_PRED:-$LOCAL_OUTPUT_DIR/mmdocir/plain_top224_ret1000_prediction.json}"
  local sparse_pred="${MMDOCIR_SPARSE_PRED:-$LOCAL_OUTPUT_DIR/mmdocir/doc_rrf_exact_dense_splade/mmdocir_splade_ret1000.prediction.json}"

  require_file gold "$gold"
  require_file doc_pages "$doc_pages_jsonl"
  require_file dense_pred "$dense_pred"
  require_file sparse_pred "$sparse_pred"

  if [[ -z "$pdf_root" ]]; then
    pdf_root="$(recorded_pdf_root "$native_summary" "$native_pdf_markdown_jsonl")"
    if [[ -n "$pdf_root" ]]; then
      echo "mmdocir_pdf_root_from_native_provenance: $pdf_root"
    fi
  fi
  if [[ -z "$pdf_root" ]]; then
    echo "missing_mmdocir_pdf_root: set MMDOCIR_PDF_ROOT to the directory containing the source PDFs." >&2
    echo "hint: inspect native provenance at $native_summary or $native_pdf_markdown_jsonl." >&2
    return 1
  fi
  if [[ ! -d "$pdf_root" ]]; then
    echo "missing_mmdocir_pdf_root_directory: $pdf_root" >&2
    echo "Set MMDOCIR_PDF_ROOT to the available source-PDF directory before rerunning." >&2
    return 1
  fi
  mkdir -p "$out_dir"

  prepare_pdf_markdown "$doc_pages_jsonl" "$pdf_root" "$pdf_markdown_jsonl" "$pdf_markdown_summary" "$variant_dir"

  local tag="mmdocir"
  run_graph_view mmdocir "$data_root" "$gold" "$dense_pred" "$sparse_pred" "$out_dir" "$pdf_markdown_jsonl" "${tag}_heading_control_no_heading" none
  run_graph_view mmdocir "$data_root" "$gold" "$dense_pred" "$sparse_pred" "$out_dir" "$pdf_markdown_jsonl" "${tag}_heading_full_wide_edgeonly_transfer" query_gated_shared
  run_graph_view mmdocir "$data_root" "$gold" "$dense_pred" "$sparse_pred" "$out_dir" "$variant_dir/doc_pages_dev_pdf_markdown.heuristic_only.jsonl" "${tag}_heading_heuristic_only_wide_edgeonly_transfer" query_gated_shared
  run_graph_view mmdocir "$data_root" "$gold" "$dense_pred" "$sparse_pred" "$out_dir" "$variant_dir/doc_pages_dev_pdf_markdown.strict_heading.jsonl" "${tag}_heading_strict_heading_wide_edgeonly_transfer" query_gated_shared

  local strict_support_pred="$out_dir/${tag}_heading_strict_heading_wide_edgeonly_transfer.prediction.json"
  local heading_evidence_jsonl="$pdf_markdown_jsonl"
  local gate_suffix="$SAFE_GATE_OUTPUT_SUFFIX"
  if [[ "$NATIVE_CODEGUARD_ABLATION" == "1" ]]; then
    run_graph_view mmdocir "$data_root" "$gold" "$dense_pred" "$sparse_pred" "$out_dir" "$variant_dir/doc_pages_dev_pdf_markdown.strict_heading_codeguard.jsonl" "${tag}_heading_strict_heading_codeguard_wide_edgeonly_transfer" query_gated_shared
    strict_support_pred="$out_dir/${tag}_heading_strict_heading_codeguard_wide_edgeonly_transfer.prediction.json"
    heading_evidence_jsonl="$variant_dir/doc_pages_dev_pdf_markdown.strict_heading_codeguard.jsonl"
    gate_suffix="${SAFE_GATE_OUTPUT_SUFFIX}_codeguard"
  fi

  run_safe_gate \
    mmdocir \
    "$gold" \
    "$out_dir" \
    "$out_dir/${tag}_heading_control_no_heading.prediction.json" \
    "$out_dir/${tag}_heading_full_wide_edgeonly_transfer.prediction.json" \
    "$out_dir/${tag}_heading_heuristic_only_wide_edgeonly_transfer.prediction.json" \
    "$strict_support_pred" \
    "$heading_evidence_jsonl" \
    "$pdf_markdown_jsonl" \
    "${tag}_${gate_suffix}" \
    "$HIT_K"
}

run_sciegqa() {
  echo
  echo "== sciegqa =="
  unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR LOCAL_MODEL_DIR
  # shellcheck disable=SC1091
  source "$REPO_ROOT/sciegqa/env_hpc.sh"

  local data_root="$LOCAL_DATA_DIR/sci-egqa-bench"
  local gold="${SCIEGQA_GOLD:-$data_root/MMQA_dev.jsonl}"
  local doc_pages_jsonl="${SCIEGQA_DOC_PAGES:-$data_root/doc_pages_dev.jsonl}"
  local out_dir="$LOCAL_OUTPUT_DIR/sciegqa/heading_breadcrumb_pdf_markdown${PDF_MARKDOWN_RUN_SUFFIX}_source_ablation"
  local pdf_root="${SCIEGQA_PDF_ROOT:-$data_root/images_raw}"
  local pdf_markdown_jsonl="$out_dir/doc_pages_dev_with_pdf_markdown.jsonl"
  local pdf_markdown_summary="$out_dir/pdf_markdown_summary.json"
  local variant_dir="$out_dir/pdf_markdown_variants"
  local dense_pred="${SCIEGQA_DENSE_PRED:-$LOCAL_OUTPUT_DIR/sciegqa/plain_top224_ret1000_prediction.json}"
  local sparse_pred="${SCIEGQA_SPARSE_PRED:-$LOCAL_OUTPUT_DIR/sciegqa/doc_rrf_plain_top224_splade/sciegqa_splade_ret1000.prediction.json}"

  require_file gold "$gold"
  require_file doc_pages "$doc_pages_jsonl"
  require_file dense_pred "$dense_pred"
  require_file sparse_pred "$sparse_pred"
  mkdir -p "$out_dir"

  prepare_pdf_markdown "$doc_pages_jsonl" "$pdf_root" "$pdf_markdown_jsonl" "$pdf_markdown_summary" "$variant_dir"

  local tag="sciegqa"
  run_graph_view sciegqa "$data_root" "$gold" "$dense_pred" "$sparse_pred" "$out_dir" "$pdf_markdown_jsonl" "${tag}_heading_control_no_heading" none
  run_graph_view sciegqa "$data_root" "$gold" "$dense_pred" "$sparse_pred" "$out_dir" "$pdf_markdown_jsonl" "${tag}_heading_full_wide_edgeonly_transfer" query_gated_shared
  run_graph_view sciegqa "$data_root" "$gold" "$dense_pred" "$sparse_pred" "$out_dir" "$variant_dir/doc_pages_dev_pdf_markdown.heuristic_only.jsonl" "${tag}_heading_heuristic_only_wide_edgeonly_transfer" query_gated_shared
  run_graph_view sciegqa "$data_root" "$gold" "$dense_pred" "$sparse_pred" "$out_dir" "$variant_dir/doc_pages_dev_pdf_markdown.strict_heading.jsonl" "${tag}_heading_strict_heading_wide_edgeonly_transfer" query_gated_shared

  local strict_support_pred="$out_dir/${tag}_heading_strict_heading_wide_edgeonly_transfer.prediction.json"
  local heading_evidence_jsonl="$pdf_markdown_jsonl"
  local gate_suffix="$SAFE_GATE_OUTPUT_SUFFIX"
  if [[ "$NATIVE_CODEGUARD_ABLATION" == "1" ]]; then
    run_graph_view sciegqa "$data_root" "$gold" "$dense_pred" "$sparse_pred" "$out_dir" "$variant_dir/doc_pages_dev_pdf_markdown.strict_heading_codeguard.jsonl" "${tag}_heading_strict_heading_codeguard_wide_edgeonly_transfer" query_gated_shared
    strict_support_pred="$out_dir/${tag}_heading_strict_heading_codeguard_wide_edgeonly_transfer.prediction.json"
    heading_evidence_jsonl="$variant_dir/doc_pages_dev_pdf_markdown.strict_heading_codeguard.jsonl"
    gate_suffix="${SAFE_GATE_OUTPUT_SUFFIX}_codeguard"
  fi

  run_safe_gate \
    sciegqa \
    "$gold" \
    "$out_dir" \
    "$out_dir/${tag}_heading_control_no_heading.prediction.json" \
    "$out_dir/${tag}_heading_full_wide_edgeonly_transfer.prediction.json" \
    "$out_dir/${tag}_heading_heuristic_only_wide_edgeonly_transfer.prediction.json" \
    "$strict_support_pred" \
    "$heading_evidence_jsonl" \
    "$pdf_markdown_jsonl" \
    "${tag}_${gate_suffix}" \
    "$HIT_K"
}

run_vidoseek() {
  echo
  echo "== vidoseek =="
  unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR LOCAL_MODEL_DIR
  # shellcheck disable=SC1091
  source "$REPO_ROOT/vidoseek/env_hpc.sh"

  local data_root="$LOCAL_DATA_DIR/vidoseek"
  local gold="${VIDOSEEK_GOLD:-$data_root/MMQA_dev.jsonl}"
  local doc_pages_jsonl="${VIDOSEEK_DOC_PAGES:-$data_root/doc_pages_dev.jsonl}"
  local out_dir="$LOCAL_OUTPUT_DIR/vidoseek/heading_breadcrumb_pdf_markdown${PDF_MARKDOWN_RUN_SUFFIX}_source_ablation"
  local pdf_root="${VIDOSEEK_PDF_ROOT:-$data_root/pdfs_raw}"
  local pdf_markdown_jsonl="$out_dir/doc_pages_dev_with_pdf_markdown.jsonl"
  local pdf_markdown_summary="$out_dir/pdf_markdown_summary.json"
  local variant_dir="$out_dir/pdf_markdown_variants"
  local dense_pred="${VIDOSEEK_DENSE_PRED:-$LOCAL_OUTPUT_DIR/vidoseek/plain_top224_ret1000_prediction.json}"
  local sparse_pred="${VIDOSEEK_SPARSE_PRED:-$LOCAL_OUTPUT_DIR/vidoseek/doc_rrf_plain_top224_splade/vidoseek_splade_ret1000.prediction.json}"

  require_file gold "$gold"
  require_file doc_pages "$doc_pages_jsonl"
  require_file dense_pred "$dense_pred"
  require_file sparse_pred "$sparse_pred"
  mkdir -p "$out_dir"

  prepare_pdf_markdown "$doc_pages_jsonl" "$pdf_root" "$pdf_markdown_jsonl" "$pdf_markdown_summary" "$variant_dir"

  local tag="vidoseek"
  run_graph_view vidoseek "$data_root" "$gold" "$dense_pred" "$sparse_pred" "$out_dir" "$pdf_markdown_jsonl" "${tag}_heading_full_wide_edgeonly_transfer" query_gated_shared
  run_graph_view vidoseek "$data_root" "$gold" "$dense_pred" "$sparse_pred" "$out_dir" "$variant_dir/doc_pages_dev_pdf_markdown.heuristic_only.jsonl" "${tag}_heading_heuristic_only_wide_edgeonly_transfer" query_gated_shared
  run_graph_view vidoseek "$data_root" "$gold" "$dense_pred" "$sparse_pred" "$out_dir" "$variant_dir/doc_pages_dev_pdf_markdown.strict_heading.jsonl" "${tag}_heading_strict_heading_wide_edgeonly_transfer" query_gated_shared

  local strict_support_pred="$out_dir/${tag}_heading_strict_heading_wide_edgeonly_transfer.prediction.json"
  local heading_evidence_jsonl="$pdf_markdown_jsonl"
  local gate_base_suffix="$SAFE_GATE_OUTPUT_SUFFIX"
  if [[ "$NATIVE_CODEGUARD_ABLATION" == "1" ]]; then
    run_graph_view vidoseek "$data_root" "$gold" "$dense_pred" "$sparse_pred" "$out_dir" "$variant_dir/doc_pages_dev_pdf_markdown.strict_heading_codeguard.jsonl" "${tag}_heading_strict_heading_codeguard_wide_edgeonly_transfer" query_gated_shared
    strict_support_pred="$out_dir/${tag}_heading_strict_heading_codeguard_wide_edgeonly_transfer.prediction.json"
    heading_evidence_jsonl="$variant_dir/doc_pages_dev_pdf_markdown.strict_heading_codeguard.jsonl"
    gate_base_suffix="${SAFE_GATE_OUTPUT_SUFFIX}_codeguard"
  fi
  local rejected_promoted_page_idx="${VIDOSEEK_REJECT_PROMOTED_PAGE_IDX:-}"
  local default_gate_suffix="$gate_base_suffix"
  if [[ -n "$rejected_promoted_page_idx" ]]; then
    if [[ "$rejected_promoted_page_idx" == "0" ]]; then
      default_gate_suffix="${gate_base_suffix}_no_page0"
    else
      default_gate_suffix="${gate_base_suffix}_reject_pageidx"
    fi
  fi
  local gate_suffix="${VIDOSEEK_SAFE_GATE_OUTPUT_SUFFIX:-$default_gate_suffix}"

  run_safe_gate \
    vidoseek \
    "$gold" \
    "$out_dir" \
    "$dense_pred" \
    "$out_dir/${tag}_heading_full_wide_edgeonly_transfer.prediction.json" \
    "$out_dir/${tag}_heading_heuristic_only_wide_edgeonly_transfer.prediction.json" \
    "$strict_support_pred" \
    "$heading_evidence_jsonl" \
    "$pdf_markdown_jsonl" \
    "${tag}_${gate_suffix}" \
    "$HIT_K" \
    "$rejected_promoted_page_idx"
}

run_vidore() {
  echo
  echo "== vidore =="
  unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR LOCAL_MODEL_DIR
  # shellcheck disable=SC1091
  source "$REPO_ROOT/vidore/env_hpc.sh"

  local data_root="$LOCAL_DATA_DIR/vidore-v3"
  local gold="${VIDORE_GOLD:-$data_root/MMQA_dev.jsonl}"
  local doc_pages_jsonl="${VIDORE_DOC_PAGES:-$data_root/doc_pages_dev.jsonl}"
  local out_dir="$LOCAL_OUTPUT_DIR/vidore-v3/heading_breadcrumb_text_source_ablation"
  local variant_dir="$out_dir/markdown_variants"
  local dense_pred="$LOCAL_OUTPUT_DIR/vidore-v3/plain_top224_ret1000_prediction.json"
  dense_pred="${VIDORE_DENSE_PRED:-$dense_pred}"
  local sparse_pred="$LOCAL_OUTPUT_DIR/vidore-v3/doc_rrf_plain_top224_splade/vidore-v3_splade_ret1000.prediction.json"
  sparse_pred="${VIDORE_SPARSE_PRED:-$sparse_pred}"

  require_file gold "$gold"
  require_file doc_pages "$doc_pages_jsonl"
  require_file dense_pred "$dense_pred"
  require_file sparse_pred "$sparse_pred"
  mkdir -p "$out_dir"

  "$PYTHON_BIN" "$REPO_ROOT/scripts/prepare_pdf_markdown_variants.py" \
    --input-jsonl "$doc_pages_jsonl" \
    --output-dir "$variant_dir"

  local tag="vidore"
  run_graph_view vidore-v3 "$data_root" "$gold" "$dense_pred" "$sparse_pred" "$out_dir" "$doc_pages_jsonl" "${tag}_heading_control_no_heading" none
  run_graph_view vidore-v3 "$data_root" "$gold" "$dense_pred" "$sparse_pred" "$out_dir" "$doc_pages_jsonl" "${tag}_heading_full_wide_edgeonly_transfer" query_gated_shared
  run_graph_view vidore-v3 "$data_root" "$gold" "$dense_pred" "$sparse_pred" "$out_dir" "$variant_dir/doc_pages_dev_pdf_markdown.heuristic_only.jsonl" "${tag}_heading_heuristic_only_wide_edgeonly_transfer" query_gated_shared
  run_graph_view vidore-v3 "$data_root" "$gold" "$dense_pred" "$sparse_pred" "$out_dir" "$variant_dir/doc_pages_dev_pdf_markdown.strict_heading.jsonl" "${tag}_heading_strict_heading_wide_edgeonly_transfer" query_gated_shared

  local strict_support_pred="$out_dir/${tag}_heading_strict_heading_wide_edgeonly_transfer.prediction.json"
  local heading_evidence_jsonl="$doc_pages_jsonl"
  local gate_suffix="$SAFE_GATE_OUTPUT_SUFFIX"
  if [[ "$NATIVE_CODEGUARD_ABLATION" == "1" ]]; then
    run_graph_view vidore-v3 "$data_root" "$gold" "$dense_pred" "$sparse_pred" "$out_dir" "$variant_dir/doc_pages_dev_pdf_markdown.strict_heading_codeguard.jsonl" "${tag}_heading_strict_heading_codeguard_wide_edgeonly_transfer" query_gated_shared
    strict_support_pred="$out_dir/${tag}_heading_strict_heading_codeguard_wide_edgeonly_transfer.prediction.json"
    heading_evidence_jsonl="$variant_dir/doc_pages_dev_pdf_markdown.strict_heading_codeguard.jsonl"
    gate_suffix="${SAFE_GATE_OUTPUT_SUFFIX}_codeguard"
  fi

  run_safe_gate \
    vidore \
    "$gold" \
    "$out_dir" \
    "$out_dir/${tag}_heading_control_no_heading.prediction.json" \
    "$out_dir/${tag}_heading_full_wide_edgeonly_transfer.prediction.json" \
    "$out_dir/${tag}_heading_heuristic_only_wide_edgeonly_transfer.prediction.json" \
    "$strict_support_pred" \
    "$heading_evidence_jsonl" \
    "$doc_pages_jsonl" \
    "${tag}_${gate_suffix}" \
    "$HIT_K"
}

echo "safe_gate_profile=$SAFE_GATE_PROFILE pdf_markdown_backend=$PDF_MARKDOWN_BACKEND native_codeguard_ablation=$NATIVE_CODEGUARD_ABLATION output_suffix=$SAFE_GATE_OUTPUT_SUFFIX hit_k=$HIT_K candidate_rank_max=$CANDIDATE_RANK_MAX rescue_rank=${RESCUE_RANK_MIN}-${RESCUE_RANK_MAX} min_page_overlap=$MIN_PAGE_OVERLAP support_page_rank_max=$SUPPORT_PAGE_RANK_MAX min_support_page_votes=$MIN_SUPPORT_PAGE_VOTES support_views=${SUPPORT_PREDICTION_LABELS[*]}"
echo "| dataset | accepted | base page@$HIT_K | candidate page@$HIT_K | gated page@$HIT_K | recovered | lost | net | body rejects |"
echo "|---|---:|---:|---:|---:|---:|---:|---:|---:|"

for dataset in $DATASETS; do
  case "$dataset" in
    m3docvqa|m3docqa|m3dovqa)
      run_m3docvqa
      ;;
    dude)
      run_dude
      ;;
    mmdocir|mm-docir)
      run_mmdocir
      ;;
    sciegqa|sci-egqa)
      run_sciegqa
      ;;
    vidoseek)
      run_vidoseek
      ;;
    vidore|vidore-v3)
      run_vidore
      ;;
    *)
      echo "unknown_dataset: $dataset" >&2
      exit 2
      ;;
  esac
done
