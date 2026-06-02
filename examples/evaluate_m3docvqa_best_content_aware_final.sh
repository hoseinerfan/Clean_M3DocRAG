#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

VITAL_PATHS_ENV="${VITAL_PATHS_ENV:-$REPO_ROOT/hpc_vital_paths.generated.env}"
if [[ -f "$VITAL_PATHS_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$VITAL_PATHS_ENV"
fi

unset M3DOCVQA_INTERNAL_ENV_LOADED
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/m3docvqa_internal_env.sh"

CUSTOM_ROOT="${CUSTOM_ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom}"
CUSTOM_OUTPUT_DIR="${CUSTOM_OUTPUT_DIR:-$CUSTOM_ROOT/outputs}"
REPO_OUTPUT_DIR="${REPO_OUTPUT_DIR:-$REPO_ROOT/output}"
OUT_DIR="${OUT_DIR:-$REPO_OUTPUT_DIR/m3docvqa_best_content_aware_final_eval}"
ORIGINAL_GOLD="${ORIGINAL_GOLD:-$REPO_ROOT/data/m3-docvqa/multimodalqa/MMQA_dev.jsonl}"
PSEUDO_GOLD="${PSEUDO_GOLD:-$REPO_OUTPUT_DIR/m3docvqa_mmqa_pseudo_page_labels/mmqa_dev_pseudo_page_labels_strict.augmented_gold.jsonl}"
HIT_K="${HIT_K:-4}"
RECALL_K_VALUES="${RECALL_K_VALUES:-1 2 4 5 10 20 50 100}"
RUN_GROUPS="${RUN_GROUPS:-1}"

first_existing_path() {
  local path
  for path in "$@"; do
    if [[ -n "$path" && -f "$path" ]]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
  return 1
}

require_file() {
  local name="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    echo "missing_${name}: $path" >&2
    exit 1
  fi
}

add_pseudo_run_if_exists() {
  local label="$1"
  local path="$2"
  if [[ -f "$path" ]]; then
    pseudo_run_args+=(--run "$label=$path")
    echo "pseudo_add_${label}=$path"
  else
    echo "pseudo_skip_missing_${label}: $path" >&2
  fi
}

add_original_candidate_if_exists() {
  local label="$1"
  local path="$2"
  if [[ -f "$path" ]]; then
    original_candidate_args+=(--candidate "$label=$path")
    echo "original_add_${label}=$path"
  else
    echo "original_skip_missing_${label}: $path" >&2
  fi
}

run_pseudo_eval() {
  local suffix="$1"
  shift
  local out_md="$OUT_DIR/final_pseudo_page_recall${suffix}.md"
  local out_csv="$OUT_DIR/final_pseudo_page_recall${suffix}.csv"
  local out_json="$OUT_DIR/final_pseudo_page_recall${suffix}.json"

  "$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_pseudo_page_retrieval.py" \
    --gold "$PSEUDO_GOLD" \
    "${pseudo_run_args[@]}" \
    "$@" \
    --hit-k "$HIT_K" \
    --recall-k $RECALL_K_VALUES \
    --format markdown \
    --output "$out_md"

  "$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_pseudo_page_retrieval.py" \
    --gold "$PSEUDO_GOLD" \
    "${pseudo_run_args[@]}" \
    "$@" \
    --hit-k "$HIT_K" \
    --recall-k $RECALL_K_VALUES \
    --format csv \
    --output "$out_csv"

  "$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_pseudo_page_retrieval.py" \
    --gold "$PSEUDO_GOLD" \
    "${pseudo_run_args[@]}" \
    "$@" \
    --hit-k "$HIT_K" \
    --recall-k $RECALL_K_VALUES \
    --format json \
    --output "$out_json"

  echo "saved_pseudo_md=$out_md"
  echo "saved_pseudo_csv=$out_csv"
  echo "saved_pseudo_json=$out_json"
}

run_original_eval() {
  local suffix="$1"
  shift
  local out_md="$OUT_DIR/final_original_gold_doc_recall${suffix}.md"
  local out_csv="$OUT_DIR/final_original_gold_doc_recall${suffix}.csv"
  local out_json="$OUT_DIR/final_original_gold_doc_recall${suffix}.json"

  "$PYTHON_BIN" "$REPO_ROOT/scripts/summarize_m3docvqa_grouped_runs.py" \
    --gold "$ORIGINAL_GOLD" \
    --baseline "$DENSE_PRED" \
    --baseline-label dense \
    "${original_candidate_args[@]}" \
    "$@" \
    --recall-k $RECALL_K_VALUES \
    --format markdown > "$out_md"

  "$PYTHON_BIN" "$REPO_ROOT/scripts/summarize_m3docvqa_grouped_runs.py" \
    --gold "$ORIGINAL_GOLD" \
    --baseline "$DENSE_PRED" \
    --baseline-label dense \
    "${original_candidate_args[@]}" \
    "$@" \
    --recall-k $RECALL_K_VALUES \
    --format csv > "$out_csv"

  "$PYTHON_BIN" "$REPO_ROOT/scripts/summarize_m3docvqa_grouped_runs.py" \
    --gold "$ORIGINAL_GOLD" \
    --baseline "$DENSE_PRED" \
    --baseline-label dense \
    "${original_candidate_args[@]}" \
    "$@" \
    --recall-k $RECALL_K_VALUES \
    --format json > "$out_json"

  echo "saved_original_md=$out_md"
  echo "saved_original_csv=$out_csv"
  echo "saved_original_json=$out_json"
}

mkdir -p "$OUT_DIR"
require_file original_gold "$ORIGINAL_GOLD"
require_file pseudo_gold "$PSEUDO_GOLD"

DENSE_PRED="${DENSE_PRED:-$(first_existing_path \
  "$CUSTOM_OUTPUT_DIR/mmqa_dev_plain_top224_nprobe4_effdiag_all.prediction.json" \
  "$CUSTOM_OUTPUT_DIR/m3docvqa_plain_top224_mmqa_dev/mmqa_dev_plain_top224_nprobe4_effdiag_all.prediction.json" \
  "$REPO_OUTPUT_DIR/m3docvqa_plain_top224_mmqa_dev/mmqa_dev_plain_top224_nprobe4_effdiag_all.prediction.json" \
  || true)}"
require_file dense_pred "$DENSE_PRED"

GPP_BASE_PRED="${GPP_BASE_PRED:-$(first_existing_path \
  "$REPO_OUTPUT_DIR/m3docvqa_graph_pagepreserve_mmqa_dev/mmqa_dev_plain_top224_splade_graph_pagepreserve_denseheavy125_medium_both.prediction.json" \
  "$CUSTOM_OUTPUT_DIR/m3docvqa_graph_pagepreserve_mmqa_dev/mmqa_dev_plain_top224_splade_graph_pagepreserve_denseheavy125_medium_both.prediction.json" \
  || true)}"
SPARSE_PRED="${SPARSE_PRED:-$(first_existing_path \
  "$CUSTOM_OUTPUT_DIR/m3docvqa_splade_mmqa_dev/mmqa_dev_splade.prediction.json" \
  "$REPO_OUTPUT_DIR/m3docvqa_splade_mmqa_dev/mmqa_dev_splade.prediction.json" \
  || true)}"
GPP_NO_HYPERLINK_PRED="${GPP_NO_HYPERLINK_PRED:-$REPO_OUTPUT_DIR/m3docvqa_gpp_hyperlink_node_ablation_mmr_target1/mmqa_dev_gpp_hyperlink_node_no_hyperlink.prediction.json}"
GPP_DOC_HYPERLINK_PRED="${GPP_DOC_HYPERLINK_PRED:-$REPO_OUTPUT_DIR/m3docvqa_gpp_hyperlink_node_ablation_mmr_target1/mmqa_dev_gpp_hyperlink_node_docnode_to_hyperlink_docs.prediction.json}"
GPP_PAGE_HYPERLINK_PRED="${GPP_PAGE_HYPERLINK_PRED:-$REPO_OUTPUT_DIR/m3docvqa_gpp_hyperlink_node_ablation_mmr_target1/mmqa_dev_gpp_hyperlink_node_pagenode_to_hyperlink_pages.prediction.json}"

CONTENT_AWARE_FIXED_PRED="${CONTENT_AWARE_FIXED_PRED:-$REPO_OUTPUT_DIR/m3docvqa_content_aware_base_sweep/mmqa_train_to_dev_content_aware_base_gpp_no_hyperlink.dev.prediction.json}"
CONTENT_AWARE_PAGE4_PRED="${CONTENT_AWARE_PAGE4_PRED:-$REPO_OUTPUT_DIR/m3docvqa_content_aware_auto_blend_sweep/mmqa_train_to_dev_content_aware_base_gpp_no_hyperlink.dev.prediction.json}"
CONTENT_AWARE_PAGE5_PRED="${CONTENT_AWARE_PAGE5_PRED:-$REPO_OUTPUT_DIR/m3docvqa_content_aware_auto_blend_sweep_page5/mmqa_train_to_dev_content_aware_base_gpp_no_hyperlink.dev.prediction.json}"
CONTENT_AWARE_PAGE10_PRED="${CONTENT_AWARE_PAGE10_PRED:-$REPO_OUTPUT_DIR/m3docvqa_content_aware_auto_blend_sweep_page10/mmqa_train_to_dev_content_aware_base_gpp_no_hyperlink.dev.prediction.json}"
LTR_PAGE_RERANKER_PRED="${LTR_PAGE_RERANKER_PRED:-$REPO_OUTPUT_DIR/m3docvqa_ltr_page_reranker/mmqa_train_to_dev_graph_aware_ltr_gpp_no_hyperlink.dev.prediction.json}"
require_file content_aware_page5_pred "$CONTENT_AWARE_PAGE5_PRED"

pseudo_run_args=(--run "dense=$DENSE_PRED")
original_candidate_args=()

add_pseudo_run_if_exists splade "$SPARSE_PRED"
add_original_candidate_if_exists splade "$SPARSE_PRED"
add_pseudo_run_if_exists gpp_base "$GPP_BASE_PRED"
add_original_candidate_if_exists gpp_base "$GPP_BASE_PRED"
add_pseudo_run_if_exists gpp_no_hyperlink "$GPP_NO_HYPERLINK_PRED"
add_original_candidate_if_exists gpp_no_hyperlink "$GPP_NO_HYPERLINK_PRED"
add_pseudo_run_if_exists gpp_doc_hyperlink "$GPP_DOC_HYPERLINK_PRED"
add_original_candidate_if_exists gpp_doc_hyperlink "$GPP_DOC_HYPERLINK_PRED"
add_pseudo_run_if_exists gpp_page_hyperlink "$GPP_PAGE_HYPERLINK_PRED"
add_original_candidate_if_exists gpp_page_hyperlink "$GPP_PAGE_HYPERLINK_PRED"

add_pseudo_run_if_exists content_aware_fixed_a0p30 "$CONTENT_AWARE_FIXED_PRED"
add_original_candidate_if_exists content_aware_fixed_a0p30 "$CONTENT_AWARE_FIXED_PRED"
add_pseudo_run_if_exists content_aware_adaptive_page4 "$CONTENT_AWARE_PAGE4_PRED"
add_original_candidate_if_exists content_aware_adaptive_page4 "$CONTENT_AWARE_PAGE4_PRED"
add_pseudo_run_if_exists content_aware_adaptive_page5_final "$CONTENT_AWARE_PAGE5_PRED"
add_original_candidate_if_exists content_aware_adaptive_page5_final "$CONTENT_AWARE_PAGE5_PRED"
add_pseudo_run_if_exists content_aware_adaptive_page10 "$CONTENT_AWARE_PAGE10_PRED"
add_original_candidate_if_exists content_aware_adaptive_page10 "$CONTENT_AWARE_PAGE10_PRED"
add_pseudo_run_if_exists graph_aware_ltr "$LTR_PAGE_RERANKER_PRED"
add_original_candidate_if_exists graph_aware_ltr "$LTR_PAGE_RERANKER_PRED"

echo "original_gold=$ORIGINAL_GOLD"
echo "pseudo_gold=$PSEUDO_GOLD"
echo "dense_pred=$DENSE_PRED"
echo "out_dir=$OUT_DIR"

run_pseudo_eval ""
run_original_eval ""

if [[ "$RUN_GROUPS" == "1" ]]; then
  group_args=()
  if [[ -f "$REPO_OUTPUT_DIR/m3docvqa_qid_groups/single_gold_doc.qids.txt" ]]; then
    group_args+=(--group "single_gold_doc=$REPO_OUTPUT_DIR/m3docvqa_qid_groups/single_gold_doc.qids.txt")
  fi
  if [[ -f "$REPO_OUTPUT_DIR/m3docvqa_qid_groups/multi_gold_doc.qids.txt" ]]; then
    group_args+=(--group "multi_gold_doc=$REPO_OUTPUT_DIR/m3docvqa_qid_groups/multi_gold_doc.qids.txt")
  fi
  if [[ "${#group_args[@]}" -gt 0 ]]; then
    run_pseudo_eval "_by_gold_doc_count" "${group_args[@]}"
    run_original_eval "_by_gold_doc_count" "${group_args[@]}"
  else
    echo "skip_group_eval_missing_qid_groups=$REPO_OUTPUT_DIR/m3docvqa_qid_groups" >&2
  fi
fi
