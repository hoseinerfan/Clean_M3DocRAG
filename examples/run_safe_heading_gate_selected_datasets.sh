#!/usr/bin/env bash
set -euo pipefail

# Run the frozen heading/bodyguard rescue gate on selected datasets.
#
# Usage examples on HPC:
#   bash examples/run_safe_heading_gate_selected_datasets.sh
#   DATASETS="m3docvqa dude vidore" bash examples/run_safe_heading_gate_selected_datasets.sh
#   DATASETS="dude" BODY_MIN_SCORE_ADVANTAGE=0.0 bash examples/run_safe_heading_gate_selected_datasets.sh
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

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    echo "missing_${label}: $path" >&2
    return 1
  fi
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
print(
    "| {dataset} | {accepted} | {base_page} | {candidate_page} | {page} | "
    "{recovered} | {lost} | {net} | {body_rejects} |".format(
        dataset=dataset,
        accepted=s.get("accepted_count"),
        base_page=s.get("base_page_hit_at_k_count") or s.get("base_page_hit_at_4_count"),
        candidate_page=s.get("candidate_page_hit_at_k_count") or s.get("candidate_page_hit_at_4_count"),
        page=s.get("page_hit_at_k_count") or s.get("page_hit_at_4_count"),
        recovered=s.get("recovered"),
        lost=s.get("lost"),
        net=s.get("net_recovered"),
        body_rejects=(s.get("rejected_promotion_reason_counts") or {}).get(
            "promoted_body_score_not_above_base", 0
        ),
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

  if [[ ! -f "$output_jsonl" ]]; then
    "$PYTHON_BIN" "$REPO_ROOT/scripts/export_pdf_page_markdown.py" \
      --doc-pages-jsonl "$doc_pages_jsonl" \
      --pdf-root "$pdf_root" \
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

  "$PYTHON_BIN" "$REPO_ROOT/scripts/apply_page_rescue_gate.py" \
    --base-prediction "$base_pred" \
    --candidate-prediction "$candidate_pred" \
    --support-prediction "heuristic=$heuristic_pred" \
    --support-prediction "strict=$strict_pred" \
    --gold "$gold" \
    --hit-k 4 \
    --candidate-rank-max 4 \
    --rescue-rank-min 5 \
    --rescue-rank-max 5 \
    --min-page-overlap 3 \
    --promoted-doc-max-base-rank 4 \
    --support-page-rank-max 4 \
    --min-support-page-votes 2 \
    --heading-doc-pages-jsonl "$heading_doc_pages" \
    --heading-min-score-advantage "$HEADING_MIN_SCORE_ADVANTAGE" \
    --heading-compare-base-top-k 4 \
    --heading-compare-mode displaced_boundary \
    --body-doc-pages-jsonl "$body_doc_pages" \
    --body-field markdown \
    --body-field text \
    --body-min-score-advantage "$BODY_MIN_SCORE_ADVANTAGE" \
    --body-compare-base-top-k 4 \
    --body-compare-mode displaced_boundary \
    --query-block-regex "$LAYOUT_QUERY_BLOCK" \
    --mode swap_promoted \
    --insert-position 4 \
    --output-prediction-json "$out_dir/${output_stem}.prediction.json" \
    --output-summary-json "$out_dir/${output_stem}.summary.json" \
    --output-cases-json "$out_dir/${output_stem}.cases.json"

  print_summary_row "$dataset_label" "$out_dir/${output_stem}.summary.json"
}

run_m3docvqa() {
  echo
  echo "== m3docvqa =="
  # shellcheck disable=SC1091
  source "$REPO_ROOT/scripts/m3docvqa_internal_env.sh"

  local data_root="$DATASET_ROOT"
  local gold="$GOLD"
  local out_dir="$LOCAL_OUTPUT_DIR/m3docvqa_heading_breadcrumb_pdf_markdown_source_ablation"
  local page_text_dir="$LOCAL_OUTPUT_DIR/m3docvqa_page_text"
  local page_text_jsonl="$page_text_dir/m3docvqa_dev_page_text.jsonl"
  local pdf_markdown_jsonl="$out_dir/doc_pages_dev_with_pdf_markdown.jsonl"
  local pdf_markdown_summary="$out_dir/pdf_markdown_summary.json"
  local variant_dir="$out_dir/pdf_markdown_variants"
  local dense_pred="$LOCAL_OUTPUT_DIR/m3docvqa_plain_top224_mmqa_dev/mmqa_dev_plain_top224_nprobe4_effdiag_all.prediction.json"
  local sparse_pred="$LOCAL_OUTPUT_DIR/m3docvqa_splade_mmqa_dev/mmqa_dev_splade.prediction.json"

  require_file gold "$gold"
  require_file dense_pred "$dense_pred"
  require_file sparse_pred "$sparse_pred"
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

  run_safe_gate \
    m3docvqa \
    "$gold" \
    "$out_dir" \
    "$out_dir/${tag}_heading_control_no_heading.prediction.json" \
    "$out_dir/${tag}_heading_full_wide_edgeonly_transfer.prediction.json" \
    "$out_dir/${tag}_heading_heuristic_only_wide_edgeonly_transfer.prediction.json" \
    "$out_dir/${tag}_heading_strict_heading_wide_edgeonly_transfer.prediction.json" \
    "$pdf_markdown_jsonl" \
    "$pdf_markdown_jsonl" \
    "${tag}_safe_gate_bodyguard"
}

run_dude() {
  echo
  echo "== dude =="
  # shellcheck disable=SC1091
  source "$REPO_ROOT/dude/env_hpc.sh"

  local data_root="$LOCAL_DATA_DIR/dude"
  local gold="$data_root/MMQA_dev.jsonl"
  local out_dir="$LOCAL_OUTPUT_DIR/dude/heading_breadcrumb_pdf_markdown_source_ablation"
  local pdf_root="${DUDE_PDF_ROOT:-$data_root/raw/DUDE_train-val-test_binaries/PDF}"
  local pdf_markdown_jsonl="$out_dir/doc_pages_dev_with_pdf_markdown.jsonl"
  local pdf_markdown_summary="$out_dir/pdf_markdown_summary.json"
  local variant_dir="$out_dir/pdf_markdown_variants"
  local dense_pred="$LOCAL_OUTPUT_DIR/dude/plain_top224_ret1000_prediction.json"
  local sparse_pred="$LOCAL_OUTPUT_DIR/dude/doc_rrf_plain_top224_splade/dude_splade_ret1000.prediction.json"

  require_file gold "$gold"
  require_file doc_pages "$data_root/doc_pages_dev.jsonl"
  require_file dense_pred "$dense_pred"
  require_file sparse_pred "$sparse_pred"
  mkdir -p "$out_dir"

  prepare_pdf_markdown "$data_root/doc_pages_dev.jsonl" "$pdf_root" "$pdf_markdown_jsonl" "$pdf_markdown_summary" "$variant_dir"

  local tag="dude"
  run_graph_view dude "$data_root" "$gold" "$dense_pred" "$sparse_pred" "$out_dir" "$pdf_markdown_jsonl" "${tag}_heading_control_no_heading" none
  run_graph_view dude "$data_root" "$gold" "$dense_pred" "$sparse_pred" "$out_dir" "$pdf_markdown_jsonl" "${tag}_heading_full_wide_edgeonly_transfer" query_gated_shared
  run_graph_view dude "$data_root" "$gold" "$dense_pred" "$sparse_pred" "$out_dir" "$variant_dir/doc_pages_dev_pdf_markdown.heuristic_only.jsonl" "${tag}_heading_heuristic_only_wide_edgeonly_transfer" query_gated_shared
  run_graph_view dude "$data_root" "$gold" "$dense_pred" "$sparse_pred" "$out_dir" "$variant_dir/doc_pages_dev_pdf_markdown.strict_heading.jsonl" "${tag}_heading_strict_heading_wide_edgeonly_transfer" query_gated_shared

  run_safe_gate \
    dude \
    "$gold" \
    "$out_dir" \
    "$out_dir/${tag}_heading_control_no_heading.prediction.json" \
    "$out_dir/${tag}_heading_full_wide_edgeonly_transfer.prediction.json" \
    "$out_dir/${tag}_heading_heuristic_only_wide_edgeonly_transfer.prediction.json" \
    "$out_dir/${tag}_heading_strict_heading_wide_edgeonly_transfer.prediction.json" \
    "$pdf_markdown_jsonl" \
    "$pdf_markdown_jsonl" \
    "${tag}_safe_gate_bodyguard"
}

run_vidore() {
  echo
  echo "== vidore =="
  # shellcheck disable=SC1091
  source "$REPO_ROOT/vidore/env_hpc.sh"

  local data_root="$LOCAL_DATA_DIR/vidore-v3"
  local gold="$data_root/MMQA_dev.jsonl"
  local out_dir="$LOCAL_OUTPUT_DIR/vidore-v3/heading_breadcrumb_text_source_ablation"
  local variant_dir="$out_dir/markdown_variants"
  local dense_pred="$LOCAL_OUTPUT_DIR/vidore-v3/plain_top224_ret1000_prediction.json"
  local sparse_pred="$LOCAL_OUTPUT_DIR/vidore-v3/doc_rrf_plain_top224_splade/vidore-v3_splade_ret1000.prediction.json"

  require_file gold "$gold"
  require_file doc_pages "$data_root/doc_pages_dev.jsonl"
  require_file dense_pred "$dense_pred"
  require_file sparse_pred "$sparse_pred"
  mkdir -p "$out_dir"

  "$PYTHON_BIN" "$REPO_ROOT/scripts/prepare_pdf_markdown_variants.py" \
    --input-jsonl "$data_root/doc_pages_dev.jsonl" \
    --output-dir "$variant_dir"

  local tag="vidore"
  run_graph_view vidore-v3 "$data_root" "$gold" "$dense_pred" "$sparse_pred" "$out_dir" "$data_root/doc_pages_dev.jsonl" "${tag}_heading_control_no_heading" none
  run_graph_view vidore-v3 "$data_root" "$gold" "$dense_pred" "$sparse_pred" "$out_dir" "$data_root/doc_pages_dev.jsonl" "${tag}_heading_full_wide_edgeonly_transfer" query_gated_shared
  run_graph_view vidore-v3 "$data_root" "$gold" "$dense_pred" "$sparse_pred" "$out_dir" "$variant_dir/doc_pages_dev_pdf_markdown.heuristic_only.jsonl" "${tag}_heading_heuristic_only_wide_edgeonly_transfer" query_gated_shared
  run_graph_view vidore-v3 "$data_root" "$gold" "$dense_pred" "$sparse_pred" "$out_dir" "$variant_dir/doc_pages_dev_pdf_markdown.strict_heading.jsonl" "${tag}_heading_strict_heading_wide_edgeonly_transfer" query_gated_shared

  run_safe_gate \
    vidore \
    "$gold" \
    "$out_dir" \
    "$out_dir/${tag}_heading_control_no_heading.prediction.json" \
    "$out_dir/${tag}_heading_full_wide_edgeonly_transfer.prediction.json" \
    "$out_dir/${tag}_heading_heuristic_only_wide_edgeonly_transfer.prediction.json" \
    "$out_dir/${tag}_heading_strict_heading_wide_edgeonly_transfer.prediction.json" \
    "$data_root/doc_pages_dev.jsonl" \
    "$data_root/doc_pages_dev.jsonl" \
    "${tag}_safe_gate_bodyguard"
}

echo "| dataset | accepted | base page@4 | candidate page@4 | gated page@4 | recovered | lost | net | body rejects |"
echo "|---|---:|---:|---:|---:|---:|---:|---:|---:|"

for dataset in $DATASETS; do
  case "$dataset" in
    m3docvqa|m3docqa|m3dovqa)
      run_m3docvqa
      ;;
    dude)
      run_dude
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
