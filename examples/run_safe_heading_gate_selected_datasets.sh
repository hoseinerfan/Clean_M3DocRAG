#!/usr/bin/env bash
set -euo pipefail

# Run the frozen heading/bodyguard rescue gate on selected datasets.
#
# Usage examples on HPC:
#   bash examples/run_safe_heading_gate_selected_datasets.sh
#   DATASETS="m3docvqa dude vidore" bash examples/run_safe_heading_gate_selected_datasets.sh
#   DATASETS="dude" BODY_MIN_SCORE_ADVANTAGE=0.0 bash examples/run_safe_heading_gate_selected_datasets.sh
#   SAFE_GATE_PROFILE=window20 RUN_GOLD_RANK_AUDIT=1 DATASETS="dude vidore" bash examples/run_safe_heading_gate_selected_datasets.sh
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

if [[ -n "$HPC_PATH_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$HPC_PATH_ENV"
elif [[ -f "$REPO_ROOT/hpc_vital_paths.generated.env" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/hpc_vital_paths.generated.env"
fi

case "$SAFE_GATE_PROFILE" in
  boundary)
    SAFE_GATE_OUTPUT_SUFFIX="${SAFE_GATE_OUTPUT_SUFFIX:-safe_gate_bodyguard}"
    CANDIDATE_RANK_MAX="${CANDIDATE_RANK_MAX:-4}"
    RESCUE_RANK_MIN="${RESCUE_RANK_MIN:-5}"
    RESCUE_RANK_MAX="${RESCUE_RANK_MAX:-5}"
    MIN_PAGE_OVERLAP="${MIN_PAGE_OVERLAP:-3}"
    SUPPORT_PAGE_RANK_MAX="${SUPPORT_PAGE_RANK_MAX:-4}"
    MIN_SUPPORT_PAGE_VOTES="${MIN_SUPPORT_PAGE_VOTES:-2}"
    ;;
  window20)
    SAFE_GATE_OUTPUT_SUFFIX="${SAFE_GATE_OUTPUT_SUFFIX:-safe_window20_gate_bodyguard}"
    CANDIDATE_RANK_MAX="${CANDIDATE_RANK_MAX:-20}"
    RESCUE_RANK_MIN="${RESCUE_RANK_MIN:-5}"
    RESCUE_RANK_MAX="${RESCUE_RANK_MAX:-20}"
    MIN_PAGE_OVERLAP="${MIN_PAGE_OVERLAP:-3}"
    SUPPORT_PAGE_RANK_MAX="${SUPPORT_PAGE_RANK_MAX:-20}"
    MIN_SUPPORT_PAGE_VOTES="${MIN_SUPPORT_PAGE_VOTES:-2}"
    ;;
  *)
    echo "unknown_SAFE_GATE_PROFILE: $SAFE_GATE_PROFILE" >&2
    exit 2
    ;;
esac

HEADING_COMPARE_BASE_TOP_K="${HEADING_COMPARE_BASE_TOP_K:-$HIT_K}"
BODY_COMPARE_BASE_TOP_K="${BODY_COMPARE_BASE_TOP_K:-$HIT_K}"
GOLD_RANK_AUDIT_BOUNDARY_RANK="${GOLD_RANK_AUDIT_BOUNDARY_RANK:-$((HIT_K + 1))}"

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
  local promoted_doc_max_base_rank="${11:-4}"
  local output_prediction="$out_dir/${output_stem}.prediction.json"
  local output_summary="$out_dir/${output_stem}.summary.json"
  local output_cases="$out_dir/${output_stem}.cases.json"
  local extra_args=()

  for page_idx in $REJECT_PROMOTED_PAGE_IDX; do
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

  if [[ "$RUN_GOLD_RANK_AUDIT" == "1" ]]; then
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
  local out_dir="$LOCAL_OUTPUT_DIR/m3docvqa_heading_breadcrumb_pdf_markdown_source_ablation"
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
    "${tag}_${SAFE_GATE_OUTPUT_SUFFIX}" \
    4
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
  local out_dir="$LOCAL_OUTPUT_DIR/dude/heading_breadcrumb_pdf_markdown_source_ablation"
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

  local dude_suffix="${DUDE_SAFE_GATE_OUTPUT_SUFFIX:-${SAFE_GATE_OUTPUT_SUFFIX}_docrank${DUDE_PROMOTED_DOC_MAX_BASE_RANK:-1}}"
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
    "${tag}_${dude_suffix}" \
    "${DUDE_PROMOTED_DOC_MAX_BASE_RANK:-1}"
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

  run_safe_gate \
    vidore \
    "$gold" \
    "$out_dir" \
    "$out_dir/${tag}_heading_control_no_heading.prediction.json" \
    "$out_dir/${tag}_heading_full_wide_edgeonly_transfer.prediction.json" \
    "$out_dir/${tag}_heading_heuristic_only_wide_edgeonly_transfer.prediction.json" \
    "$out_dir/${tag}_heading_strict_heading_wide_edgeonly_transfer.prediction.json" \
    "$doc_pages_jsonl" \
    "$doc_pages_jsonl" \
    "${tag}_${SAFE_GATE_OUTPUT_SUFFIX}" \
    4
}

echo "safe_gate_profile=$SAFE_GATE_PROFILE hit_k=$HIT_K candidate_rank_max=$CANDIDATE_RANK_MAX rescue_rank=${RESCUE_RANK_MIN}-${RESCUE_RANK_MAX} support_page_rank_max=$SUPPORT_PAGE_RANK_MAX"
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
    vidore|vidore-v3)
      run_vidore
      ;;
    *)
      echo "unknown_dataset: $dataset" >&2
      exit 2
      ;;
  esac
done
