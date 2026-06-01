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
OUT_DIR="${OUT_DIR:-$REPO_OUTPUT_DIR/m3docvqa_pseudo_page_ablation_eval}"
LABEL="${LABEL:-dev_strict_pseudo_page_ablation_suite}"
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

sanitize_label() {
  printf '%s' "$1" | sed -e 's/\.prediction$//' -e 's/[^A-Za-z0-9_.-]/_/g'
}

label_exists() {
  local needle="$1"
  local existing
  for existing in "${run_labels[@]}"; do
    if [[ "$existing" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

add_run() {
  local label="$1"
  local path="$2"
  label="$(sanitize_label "$label")"
  if [[ -f "$path" ]]; then
    if label_exists "$label"; then
      echo "skip_duplicate_run_${label}: $path" >&2
      return
    fi
    run_labels+=( "$label" )
    run_args+=( --run "$label=$path" )
    echo "add_run_${label}=$path"
  else
    echo "skip_missing_run_${label}: $path" >&2
  fi
}

add_first_existing_run() {
  local label="$1"
  shift
  local path
  path="$(first_existing_path "$@" || true)"
  if [[ -n "$path" ]]; then
    add_run "$label" "$path"
  else
    echo "skip_missing_run_$(sanitize_label "$label"): $*" >&2
  fi
}

add_glob_runs() {
  local prefix="$1"
  shift
  local pattern path base label
  shopt -s nullglob
  for pattern in "$@"; do
    for path in $pattern; do
      base="$(basename "$path" .json)"
      base="${base%.prediction}"
      base="${base#mmqa_dev_}"
      label="${prefix}${base}"
      add_run "$label" "$path"
    done
  done
  shopt -u nullglob
}

run_eval() {
  local suffix="$1"
  shift
  local out_md="$OUT_DIR/${LABEL}${suffix}.md"
  local out_csv="$OUT_DIR/${LABEL}${suffix}.csv"
  local out_json="$OUT_DIR/${LABEL}${suffix}.json"

  "$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_pseudo_page_retrieval.py" \
    --gold "$PSEUDO_GOLD" \
    "${run_args[@]}" \
    "$@" \
    --hit-k "$HIT_K" \
    --recall-k $RECALL_K_VALUES \
    --format markdown \
    --output "$out_md"

  "$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_pseudo_page_retrieval.py" \
    --gold "$PSEUDO_GOLD" \
    "${run_args[@]}" \
    "$@" \
    --hit-k "$HIT_K" \
    --recall-k $RECALL_K_VALUES \
    --format csv \
    --output "$out_csv"

  "$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_pseudo_page_retrieval.py" \
    --gold "$PSEUDO_GOLD" \
    "${run_args[@]}" \
    "$@" \
    --hit-k "$HIT_K" \
    --recall-k $RECALL_K_VALUES \
    --format json \
    --output "$out_json"

  echo "saved_md=$out_md"
  echo "saved_csv=$out_csv"
  echo "saved_json=$out_json"
}

mkdir -p "$OUT_DIR"
require_file pseudo_gold "$PSEUDO_GOLD"

run_labels=()
run_args=()

dense_pred="${DENSE_PRED:-$(first_existing_path \
  "$CUSTOM_OUTPUT_DIR/mmqa_dev_plain_top224_nprobe4_effdiag_all.prediction.json" \
  "$CUSTOM_OUTPUT_DIR/m3docvqa_plain_top224_mmqa_dev/mmqa_dev_plain_top224_nprobe4_effdiag_all.prediction.json" \
  "$REPO_OUTPUT_DIR/m3docvqa_plain_top224_mmqa_dev/mmqa_dev_plain_top224_nprobe4_effdiag_all.prediction.json" \
  || true)}"
require_file dense_pred "$dense_pred"
add_run dense "$dense_pred"

add_first_existing_run splade \
  "${SPARSE_PRED:-}" \
  "$CUSTOM_OUTPUT_DIR/m3docvqa_splade_mmqa_dev/mmqa_dev_splade.prediction.json" \
  "$REPO_OUTPUT_DIR/m3docvqa_splade_mmqa_dev/mmqa_dev_splade.prediction.json"

add_first_existing_run gpp_base \
  "$REPO_OUTPUT_DIR/m3docvqa_graph_pagepreserve_mmqa_dev/mmqa_dev_plain_top224_splade_graph_pagepreserve_denseheavy125_medium_both.prediction.json" \
  "$CUSTOM_OUTPUT_DIR/m3docvqa_graph_pagepreserve_mmqa_dev/mmqa_dev_plain_top224_splade_graph_pagepreserve_denseheavy125_medium_both.prediction.json"

add_first_existing_run gpp_no_hyperlink \
  "$REPO_OUTPUT_DIR/m3docvqa_gpp_hyperlink_node_ablation_mmr_target1/mmqa_dev_gpp_hyperlink_node_no_hyperlink.prediction.json"
add_first_existing_run gpp_doc_hyperlink \
  "$REPO_OUTPUT_DIR/m3docvqa_gpp_hyperlink_node_ablation_mmr_target1/mmqa_dev_gpp_hyperlink_node_docnode_to_hyperlink_docs.prediction.json"
add_first_existing_run gpp_page_hyperlink \
  "$REPO_OUTPUT_DIR/m3docvqa_gpp_hyperlink_node_ablation_mmr_target1/mmqa_dev_gpp_hyperlink_node_pagenode_to_hyperlink_pages.prediction.json"

add_glob_runs complete_ \
  "$CUSTOM_OUTPUT_DIR/m3docvqa_complete_graph_ablation/"*.prediction.json \
  "$REPO_OUTPUT_DIR/m3docvqa_complete_graph_ablation/"*.prediction.json
add_glob_runs docseed_page_ \
  "$CUSTOM_OUTPUT_DIR/m3docvqa_doc_seed_page_score_ablation/"*.prediction.json \
  "$REPO_OUTPUT_DIR/m3docvqa_doc_seed_page_score_ablation/"*.prediction.json
add_glob_runs docdoc_ \
  "$CUSTOM_OUTPUT_DIR/m3docvqa_doc_doc_edge_ablation/"*.prediction.json \
  "$REPO_OUTPUT_DIR/m3docvqa_doc_doc_edge_ablation/"*.prediction.json
add_glob_runs hyperlink_init_ \
  "$CUSTOM_OUTPUT_DIR/m3docvqa_hyperlink_init_ablation/"*.prediction.json \
  "$REPO_OUTPUT_DIR/m3docvqa_hyperlink_init_ablation/"*.prediction.json
add_glob_runs adaptive_hyperlink_ \
  "$CUSTOM_OUTPUT_DIR/m3docvqa_adaptive_hyperlink_ablation/"*.prediction.json \
  "$REPO_OUTPUT_DIR/m3docvqa_adaptive_hyperlink_ablation/"*.prediction.json

add_first_existing_run ppr_convergence_tol1e7 \
  "$REPO_OUTPUT_DIR/m3docvqa_gpp_ppr_convergence_tol1e7/mmqa_dev_gpp_ppr_convergence_tol1e7.prediction.json"
add_first_existing_run ppr_graph_size_nodes \
  "$REPO_OUTPUT_DIR/m3docvqa_gpp_ppr_graph_size_nodes/mmqa_dev_gpp_ppr_graph_size_nodes.prediction.json"

add_glob_runs idea01_doc_embed_ \
  "$REPO_OUTPUT_DIR/m3docvqa_gpp_doc_embed_cosine"*/*.prediction.json \
  "$CUSTOM_OUTPUT_DIR/m3docvqa_gpp_doc_embed_cosine"*/*.prediction.json

add_glob_runs idea02_faiss_token_ \
  "$REPO_OUTPUT_DIR/m3docvqa_gpp_faiss_token_neighbor"*/*.prediction.json \
  "$CUSTOM_OUTPUT_DIR/m3docvqa_gpp_faiss_token_neighbor"*/*.prediction.json

add_first_existing_run rank_band_direct \
  "$REPO_OUTPUT_DIR/m3docvqa_rank_band_page_promotion_pseudo_page/mmqa_train_to_dev_rank_band_pseudo_page.dev.prediction.json" \
  "$REPO_OUTPUT_DIR/m3docvqa_rank_band_page_promotion/mmqa_train_to_dev_rank_band_page_promotion.dev.prediction.json"
add_glob_runs rank_band_graph_prior_ \
  "$REPO_OUTPUT_DIR/m3docvqa_rank_band_graph_prior"*/*.prediction.json
add_first_existing_run content_aware_pseudo_page \
  "$REPO_OUTPUT_DIR/m3docvqa_content_aware_pseudo_page_reranker/mmqa_train_to_dev_content_aware_pseudo_page.dev.prediction.json"

if [[ "${#run_args[@]}" -lt 4 ]]; then
  echo "too_few_runs=${#run_args[@]}: expected dense plus at least a few candidate outputs" >&2
  exit 1
fi

echo "pseudo_gold=$PSEUDO_GOLD"
echo "run_count=${#run_labels[@]}"
run_eval ""

if [[ "$RUN_GROUPS" == "1" ]]; then
  group_args=()
  if [[ -f "$REPO_OUTPUT_DIR/m3docvqa_qid_groups/single_gold_doc.qids.txt" ]]; then
    group_args+=( --group "single_gold_doc=$REPO_OUTPUT_DIR/m3docvqa_qid_groups/single_gold_doc.qids.txt" )
  fi
  if [[ -f "$REPO_OUTPUT_DIR/m3docvqa_qid_groups/multi_gold_doc.qids.txt" ]]; then
    group_args+=( --group "multi_gold_doc=$REPO_OUTPUT_DIR/m3docvqa_qid_groups/multi_gold_doc.qids.txt" )
  fi
  if [[ "${#group_args[@]}" -gt 0 ]]; then
    run_eval "_by_gold_doc_count" "${group_args[@]}"
  else
    echo "skip_group_eval_missing_qid_groups=$REPO_OUTPUT_DIR/m3docvqa_qid_groups" >&2
  fi
fi
