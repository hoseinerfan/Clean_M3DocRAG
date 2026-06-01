#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

VITAL_PATHS_ENV="${VITAL_PATHS_ENV:-$REPO_ROOT/hpc_vital_paths.generated.env}"
if [[ -f "$VITAL_PATHS_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$VITAL_PATHS_ENV"
fi

CUSTOM_ROOT="${CUSTOM_ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/env/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

TRAIN_GOLD="${TRAIN_GOLD:-$REPO_ROOT/output/m3docvqa_mmqa_pseudo_page_labels/mmqa_train_pseudo_page_labels_strict.augmented_gold.jsonl}"
EVAL_GOLD="${EVAL_GOLD:-$REPO_ROOT/output/m3docvqa_mmqa_pseudo_page_labels/mmqa_dev_pseudo_page_labels_strict.augmented_gold.jsonl}"
TRAIN_PAGE_TEXT_JSONL="${TRAIN_PAGE_TEXT_JSONL:-$CUSTOM_ROOT/outputs/m3docvqa_page_text/m3docvqa_train_page_text.jsonl}"
EVAL_PAGE_TEXT_JSONL="${EVAL_PAGE_TEXT_JSONL:-$CUSTOM_ROOT/outputs/m3docvqa_page_text/m3docvqa_dev_page_text.jsonl}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/output/m3docvqa_content_aware_base_sweep}"

TRAIN_DENSE_PRED="${TRAIN_DENSE_PRED:-$CUSTOM_ROOT/outputs/m3docvqa_baseline_mmqa_train/mmqa_train_baseline_ret1000_ivfflat_nprobe4.prediction.json}"
EVAL_DENSE_PRED="${EVAL_DENSE_PRED:-$CUSTOM_ROOT/outputs/mmqa_dev_plain_top224_nprobe4_effdiag_all.prediction.json}"
EVAL_SPLADE_PRED="${EVAL_SPLADE_PRED:-$CUSTOM_ROOT/outputs/m3docvqa_splade_mmqa_dev/mmqa_dev_splade.prediction.json}"

TRAIN_GPP_BASE_PRED="${TRAIN_GPP_BASE_PRED:-$REPO_ROOT/output/m3docvqa_graph_pagepreserve_mmqa_train/mmqa_train_plain_top224_splade_graph_pagepreserve_denseheavy125_medium_both.prediction.json}"
EVAL_GPP_BASE_PRED="${EVAL_GPP_BASE_PRED:-$REPO_ROOT/output/m3docvqa_graph_pagepreserve_mmqa_dev/mmqa_dev_plain_top224_splade_graph_pagepreserve_denseheavy125_medium_both.prediction.json}"

GPP_TRAIN_OUT_DIR="${GPP_TRAIN_OUT_DIR:-$REPO_ROOT/output/m3docvqa_gpp_hyperlink_node_ablation_mmr_target1_train_real}"
GPP_EVAL_OUT_DIR="${GPP_EVAL_OUT_DIR:-$REPO_ROOT/output/m3docvqa_gpp_hyperlink_node_ablation_mmr_target1}"
TRAIN_GPP_NO_HYPERLINK_PRED="${TRAIN_GPP_NO_HYPERLINK_PRED:-$GPP_TRAIN_OUT_DIR/mmqa_train_gpp_hyperlink_node_no_hyperlink.prediction.json}"
TRAIN_GPP_DOC_HYPERLINK_PRED="${TRAIN_GPP_DOC_HYPERLINK_PRED:-$GPP_TRAIN_OUT_DIR/mmqa_train_gpp_hyperlink_node_docnode_to_hyperlink_docs.prediction.json}"
TRAIN_GPP_PAGE_HYPERLINK_PRED="${TRAIN_GPP_PAGE_HYPERLINK_PRED:-$GPP_TRAIN_OUT_DIR/mmqa_train_gpp_hyperlink_node_pagenode_to_hyperlink_pages.prediction.json}"
EVAL_GPP_NO_HYPERLINK_PRED="${EVAL_GPP_NO_HYPERLINK_PRED:-$GPP_EVAL_OUT_DIR/mmqa_dev_gpp_hyperlink_node_no_hyperlink.prediction.json}"
EVAL_GPP_DOC_HYPERLINK_PRED="${EVAL_GPP_DOC_HYPERLINK_PRED:-$GPP_EVAL_OUT_DIR/mmqa_dev_gpp_hyperlink_node_docnode_to_hyperlink_docs.prediction.json}"
EVAL_GPP_PAGE_HYPERLINK_PRED="${EVAL_GPP_PAGE_HYPERLINK_PRED:-$GPP_EVAL_OUT_DIR/mmqa_dev_gpp_hyperlink_node_pagenode_to_hyperlink_pages.prediction.json}"

DOCSEED_TRAIN_OUT_DIR="${DOCSEED_TRAIN_OUT_DIR:-$REPO_ROOT/output/m3docvqa_doc_seed_page_score_ablation_train}"
DOCSEED_EVAL_OUT_DIR="${DOCSEED_EVAL_OUT_DIR:-$REPO_ROOT/output/m3docvqa_doc_seed_page_score_ablation}"
TRAIN_DOCSEED_TOP3MEAN_1P00_PRED="${TRAIN_DOCSEED_TOP3MEAN_1P00_PRED:-$DOCSEED_TRAIN_OUT_DIR/mmqa_train_docseed_page_score_docseed_page_top3mean_1p00.prediction.json}"
EVAL_DOCSEED_TOP3MEAN_1P00_PRED="${EVAL_DOCSEED_TOP3MEAN_1P00_PRED:-$DOCSEED_EVAL_OUT_DIR/mmqa_dev_docseed_page_score_docseed_page_top3mean_1p00.prediction.json}"
TRAIN_DOCSEED_AVGPAGE_0P50_PRED="${TRAIN_DOCSEED_AVGPAGE_0P50_PRED:-$REPO_ROOT/output/m3docvqa_complete_graph_ablation_train/mmqa_train_complete_graph_ablation_docseed_avgpage_0p50.prediction.json}"
EVAL_DOCSEED_AVGPAGE_0P50_PRED="${EVAL_DOCSEED_AVGPAGE_0P50_PRED:-$REPO_ROOT/output/m3docvqa_complete_graph_ablation/mmqa_dev_complete_graph_ablation_docseed_avgpage_0p50.prediction.json}"

BLEND_ALPHA="${BLEND_ALPHA:-0.30}"
EPOCHS="${EPOCHS:-80}"
INFERENCE_MODE="${INFERENCE_MODE:-blend_rerank}"
FORCE_RERUN="${FORCE_RERUN:-0}"
RUN_DENSE_BASE="${RUN_DENSE_BASE:-0}"

require_file() {
  local name="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    echo "missing_${name}: $path" >&2
    exit 1
  fi
}

add_eval_run() {
  local label="$1"
  local path="$2"
  if [[ -f "$path" ]]; then
    eval_args+=( --run "$label=$path" )
    echo "eval_add_${label}=$path"
  else
    echo "eval_skip_missing_${label}: $path" >&2
  fi
}

run_base_pair() {
  local label="$1"
  local train_base="$2"
  local eval_base="$3"
  local out_pred="$OUT_DIR/mmqa_train_to_dev_content_aware_${label}.dev.prediction.json"

  if [[ ! -f "$train_base" || ! -f "$eval_base" ]]; then
    [[ -f "$train_base" ]] || echo "skip_${label}_missing_train_base=$train_base" >&2
    [[ -f "$eval_base" ]] || echo "skip_${label}_missing_eval_base=$eval_base" >&2
    return 0
  fi

  if [[ "$FORCE_RERUN" != "1" && -f "$out_pred" ]]; then
    echo "reuse_${label}=$out_pred"
    return 0
  fi

  echo
  echo "== content-aware base sweep: $label =="
  TRAIN_GOLD="$TRAIN_GOLD" \
  EVAL_GOLD="$EVAL_GOLD" \
  TRAIN_DENSE_PRED="$train_base" \
  EVAL_DENSE_PRED="$eval_base" \
  TRAIN_PAGE_TEXT_JSONL="$TRAIN_PAGE_TEXT_JSONL" \
  EVAL_PAGE_TEXT_JSONL="$EVAL_PAGE_TEXT_JSONL" \
  INFERENCE_MODE="$INFERENCE_MODE" \
  BLEND_ALPHA="$BLEND_ALPHA" \
  EPOCHS="$EPOCHS" \
  OUT_DIR="$OUT_DIR" \
  LABEL="mmqa_train_to_dev_content_aware_${label}" \
  bash "$REPO_ROOT/examples/run_m3docvqa_content_aware_pseudo_page_reranker.sh"
}

mkdir -p "$OUT_DIR"
require_file train_gold "$TRAIN_GOLD"
require_file eval_gold "$EVAL_GOLD"
require_file train_page_text_jsonl "$TRAIN_PAGE_TEXT_JSONL"
require_file eval_page_text_jsonl "$EVAL_PAGE_TEXT_JSONL"
require_file eval_dense_pred "$EVAL_DENSE_PRED"

if [[ "$RUN_DENSE_BASE" == "1" ]]; then
  run_base_pair base_dense "$TRAIN_DENSE_PRED" "$EVAL_DENSE_PRED"
fi
run_base_pair base_gpp_pagepreserve "$TRAIN_GPP_BASE_PRED" "$EVAL_GPP_BASE_PRED"
run_base_pair base_gpp_no_hyperlink "$TRAIN_GPP_NO_HYPERLINK_PRED" "$EVAL_GPP_NO_HYPERLINK_PRED"
run_base_pair base_gpp_doc_hyperlink "$TRAIN_GPP_DOC_HYPERLINK_PRED" "$EVAL_GPP_DOC_HYPERLINK_PRED"
run_base_pair base_gpp_page_hyperlink "$TRAIN_GPP_PAGE_HYPERLINK_PRED" "$EVAL_GPP_PAGE_HYPERLINK_PRED"
run_base_pair base_docseed_top3mean_1p00 "$TRAIN_DOCSEED_TOP3MEAN_1P00_PRED" "$EVAL_DOCSEED_TOP3MEAN_1P00_PRED"
run_base_pair base_docseed_avgpage_0p50 "$TRAIN_DOCSEED_AVGPAGE_0P50_PRED" "$EVAL_DOCSEED_AVGPAGE_0P50_PRED"

eval_args=()
add_eval_run dense "$EVAL_DENSE_PRED"
add_eval_run splade "$EVAL_SPLADE_PRED"
add_eval_run gpp_base "$EVAL_GPP_BASE_PRED"
add_eval_run gpp_no_hyperlink "$EVAL_GPP_NO_HYPERLINK_PRED"
add_eval_run gpp_doc_hyperlink "$EVAL_GPP_DOC_HYPERLINK_PRED"
add_eval_run gpp_page_hyperlink "$EVAL_GPP_PAGE_HYPERLINK_PRED"
add_eval_run docseed_top3mean_1p00 "$EVAL_DOCSEED_TOP3MEAN_1P00_PRED"
add_eval_run docseed_avgpage_0p50 "$EVAL_DOCSEED_AVGPAGE_0P50_PRED"
for pred in "$OUT_DIR"/mmqa_train_to_dev_content_aware_*.dev.prediction.json; do
  [[ -f "$pred" ]] || continue
  label="$(basename "$pred" .dev.prediction.json)"
  label="${label#mmqa_train_to_dev_content_aware_}"
  add_eval_run "content_aware_${label}" "$pred"
done

if [[ "${#eval_args[@]}" -gt 2 ]]; then
  "$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_pseudo_page_retrieval.py" \
    --gold "$EVAL_GOLD" \
    "${eval_args[@]}" \
    --format markdown \
    --output "$OUT_DIR/content_aware_base_sweep_eval.md"
  "$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_pseudo_page_retrieval.py" \
    --gold "$EVAL_GOLD" \
    "${eval_args[@]}" \
    --format csv \
    --output "$OUT_DIR/content_aware_base_sweep_eval.csv"
  "$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_pseudo_page_retrieval.py" \
    --gold "$EVAL_GOLD" \
    "${eval_args[@]}" \
    --format json \
    --output "$OUT_DIR/content_aware_base_sweep_eval.json"
  echo "saved_eval_md=$OUT_DIR/content_aware_base_sweep_eval.md"
  echo "saved_eval_csv=$OUT_DIR/content_aware_base_sweep_eval.csv"
  echo "saved_eval_json=$OUT_DIR/content_aware_base_sweep_eval.json"
fi
