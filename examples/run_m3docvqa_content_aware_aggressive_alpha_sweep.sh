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

alpha_tag() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s\n' "$value"
}

CUSTOM_ROOT="${CUSTOM_ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom}"
ALPHAS="${ALPHAS:-0.35 0.40 0.50 0.60 0.70 0.80 1.00}"
HIT_KS="${HIT_KS:-4 5}"
FORCE_RERUN="${FORCE_RERUN:-0}"
REFERENCE_RUNS="${REFERENCE_RUNS:-}"

TRAIN_GOLD="${TRAIN_GOLD:-$REPO_ROOT/output/m3docvqa_mmqa_pseudo_page_labels/mmqa_train_pseudo_page_labels_strict.augmented_gold.jsonl}"
EVAL_GOLD="${EVAL_GOLD:-$REPO_ROOT/output/m3docvqa_mmqa_pseudo_page_labels/mmqa_dev_pseudo_page_labels_strict.augmented_gold.jsonl}"
BASELINE_PRED="${BASELINE_PRED:-$REPO_ROOT/output/m3docvqa_gpp_hyperlink_node_ablation_mmr_target1/mmqa_dev_gpp_hyperlink_node_no_hyperlink.prediction.json}"
TRAIN_BASE_PRED="${TRAIN_BASE_PRED:-$REPO_ROOT/output/m3docvqa_gpp_hyperlink_node_ablation_mmr_target1_train_real/mmqa_train_gpp_hyperlink_node_no_hyperlink.prediction.json}"

OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/output/m3docvqa_content_aware_aggressive_alpha_sweep}"
RECALL_MD="${RECALL_MD:-$OUT_ROOT/aggressive_alpha_recall.md}"
RESCUE_OUT_DIR="${RESCUE_OUT_DIR:-$REPO_ROOT/output/m3docvqa_content_aware_alpha_rescue_audit}"

mkdir -p "$OUT_ROOT" "$RESCUE_OUT_DIR"

run_args=()
if [[ -n "$REFERENCE_RUNS" ]]; then
  for labeled_path in $REFERENCE_RUNS; do
    label="${labeled_path%%=*}"
    path="${labeled_path#*=}"
    if [[ "$label" == "$labeled_path" || -z "$label" || -z "$path" ]]; then
      echo "bad_reference_run: $labeled_path" >&2
      echo "expected_reference_run_format: label=/path/to/prediction.json" >&2
      exit 1
    fi
    if [[ ! -f "$path" ]]; then
      echo "missing_reference_run_${label}: $path" >&2
      exit 1
    fi
    echo "using_reference_run_${label}=$path"
    run_args+=(--run "$labeled_path")
  done
fi

for alpha in $ALPHAS; do
  tag="$(alpha_tag "$alpha")"
  alpha_out_dir="$OUT_ROOT/alpha_${tag}"
  label="content_aware_alpha_${tag}"
  pred_path="$alpha_out_dir/${label}.dev.prediction.json"

  if [[ "$FORCE_RERUN" != "1" && -f "$pred_path" ]]; then
    echo "skip_existing_alpha_${tag}=$pred_path"
  else
    echo
    echo "== M3DocVQA content-aware fixed alpha: $alpha =="
    TRAIN_GOLD="$TRAIN_GOLD" \
    EVAL_GOLD="$EVAL_GOLD" \
    TRAIN_DENSE_PRED="$TRAIN_BASE_PRED" \
    EVAL_DENSE_PRED="$BASELINE_PRED" \
    AUTO_TUNE_BLEND_ALPHA=0 \
    QUERY_ADAPTIVE_ALPHA=0 \
    BLEND_ALPHA="$alpha" \
    OUT_DIR="$alpha_out_dir" \
    LABEL="$label" \
    bash "$REPO_ROOT/examples/run_m3docvqa_content_aware_pseudo_page_reranker.sh"
  fi

  if [[ ! -f "$pred_path" ]]; then
    echo "missing_alpha_prediction_${tag}: $pred_path" >&2
    exit 1
  fi
  run_args+=(--run "alpha${tag}=$pred_path")
done

echo
echo "== Aggressive alpha recall table =="
"$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_pseudo_page_retrieval.py" \
  --gold "$EVAL_GOLD" \
  --run "gpp_no_hyperlink=$BASELINE_PRED" \
  "${run_args[@]}" \
  --output "$RECALL_MD"
echo "saved_recall_md=$RECALL_MD"

for hit_k in $HIT_KS; do
  output_md="$RESCUE_OUT_DIR/aggressive_alpha_hit${hit_k}.md"
  output_json="$RESCUE_OUT_DIR/aggressive_alpha_hit${hit_k}.json"
  echo
  echo "== Aggressive alpha rescue audit: hit@$hit_k =="
  "$PYTHON_BIN" "$REPO_ROOT/scripts/compare_pseudo_page_alpha_rescues.py" \
    --gold "$EVAL_GOLD" \
    --baseline "$BASELINE_PRED" \
    "${run_args[@]}" \
    --hit-k "$hit_k" \
    --output-md "$output_md" \
    --output-json "$output_json"
  echo "saved_rescue_md=$output_md"
  echo "saved_rescue_json=$output_json"
done
