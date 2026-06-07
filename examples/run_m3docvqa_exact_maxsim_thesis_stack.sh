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

RUN_EXACT_DENSE_DEV="${RUN_EXACT_DENSE_DEV:-1}"
RUN_EXACT_DENSE_TRAIN="${RUN_EXACT_DENSE_TRAIN:-1}"
RUN_GPP_DEV="${RUN_GPP_DEV:-1}"
RUN_GPP_TRAIN="${RUN_GPP_TRAIN:-1}"
RUN_CONTENT_AWARE_TOPK="${RUN_CONTENT_AWARE_TOPK:-1}"
RUN_FEATURE_ABLATION="${RUN_FEATURE_ABLATION:-1}"
RUN_PSEUDO_LABEL_ABLATION="${RUN_PSEUDO_LABEL_ABLATION:-1}"
RUN_OOF_HYBRID="${RUN_OOF_HYBRID:-1}"
RUN_FINAL_EVAL="${RUN_FINAL_EVAL:-1}"

TOP_PAGES="${TOP_PAGES:-1000}"
FAISS_NPROBE="${FAISS_NPROBE:-4}"
GRAPH_PROFILE="${GRAPH_PROFILE:-denseheavy125_medium_both}"
EXACT_TAG="${EXACT_TAG:-exact_maxsim}"

EXACT_DENSE_DEV_OUT_DIR="${EXACT_DENSE_DEV_OUT_DIR:-$LOCAL_OUTPUT_DIR/m3docvqa_${EXACT_TAG}_mmqa_dev}"
EXACT_DENSE_TRAIN_OUT_DIR="${EXACT_DENSE_TRAIN_OUT_DIR:-$LOCAL_OUTPUT_DIR/m3docvqa_${EXACT_TAG}_mmqa_train}"
EXACT_DENSE_DEV_LABEL="${EXACT_DENSE_DEV_LABEL:-mmqa_dev_${EXACT_TAG}_nprobe${FAISS_NPROBE}_ret${TOP_PAGES}}"
EXACT_DENSE_TRAIN_LABEL="${EXACT_DENSE_TRAIN_LABEL:-mmqa_train_${EXACT_TAG}_nprobe${FAISS_NPROBE}_ret${TOP_PAGES}}"
EXACT_DENSE_DEV_PRED="${EXACT_DENSE_DEV_PRED:-$EXACT_DENSE_DEV_OUT_DIR/${EXACT_DENSE_DEV_LABEL}.prediction.json}"
EXACT_DENSE_TRAIN_PRED="${EXACT_DENSE_TRAIN_PRED:-$EXACT_DENSE_TRAIN_OUT_DIR/${EXACT_DENSE_TRAIN_LABEL}.prediction.json}"

DEV_SPARSE_PRED="${DEV_SPARSE_PRED:-$LOCAL_OUTPUT_DIR/m3docvqa_splade_mmqa_dev/mmqa_dev_splade.prediction.json}"
TRAIN_SPARSE_PRED="${TRAIN_SPARSE_PRED:-$LOCAL_OUTPUT_DIR/m3docvqa_splade_mmqa_train/mmqa_train_splade.prediction.json}"
if [[ ! -f "$TRAIN_SPARSE_PRED" ]]; then
  TRAIN_SPARSE_PRED="$EXACT_DENSE_TRAIN_PRED"
fi

GPP_DEV_OUT_DIR="${GPP_DEV_OUT_DIR:-$REPO_ROOT/output/m3docvqa_gpp_hyperlink_node_ablation_${EXACT_TAG}}"
GPP_TRAIN_OUT_DIR="${GPP_TRAIN_OUT_DIR:-$REPO_ROOT/output/m3docvqa_gpp_hyperlink_node_ablation_${EXACT_TAG}_train}"
GPP_DEV_LABEL_PREFIX="${GPP_DEV_LABEL_PREFIX:-mmqa_dev_${EXACT_TAG}_gpp_hyperlink_node}"
GPP_TRAIN_LABEL_PREFIX="${GPP_TRAIN_LABEL_PREFIX:-mmqa_train_${EXACT_TAG}_gpp_hyperlink_node}"
GPP_DEV_NO_HYPERLINK="$GPP_DEV_OUT_DIR/${GPP_DEV_LABEL_PREFIX}_no_hyperlink.prediction.json"
GPP_TRAIN_NO_HYPERLINK="$GPP_TRAIN_OUT_DIR/${GPP_TRAIN_LABEL_PREFIX}_no_hyperlink.prediction.json"

CONTENT_TOPK_OUT_DIR="${CONTENT_TOPK_OUT_DIR:-$REPO_ROOT/output/m3docvqa_content_aware_${EXACT_TAG}_adaptive_topk_ablation}"
FEATURE_OUT_DIR="${FEATURE_OUT_DIR:-$REPO_ROOT/output/m3docvqa_content_aware_${EXACT_TAG}_feature_ablation}"
PSEUDO_LABEL_OUT_DIR="${PSEUDO_LABEL_OUT_DIR:-$REPO_ROOT/output/m3docvqa_content_aware_${EXACT_TAG}_pseudo_label_ablation}"
OOF_OUT_DIR="${OOF_OUT_DIR:-$REPO_ROOT/output/m3docvqa_content_aware_counterfactual_oof_hybrid_${EXACT_TAG}}"
FINAL_EVAL_OUT_DIR="${FINAL_EVAL_OUT_DIR:-$REPO_ROOT/output/m3docvqa_best_content_aware_final_eval_${EXACT_TAG}}"

CONTENT_BASE_LABEL="${CONTENT_BASE_LABEL:-${EXACT_TAG}_gpp_no_hyperlink}"
CONTENT_PAGE5_LABEL="${CONTENT_PAGE5_LABEL:-mmqa_train_to_dev_content_aware_auto_page5_base_${CONTENT_BASE_LABEL}}"
CONTENT_PAGE5_PRED="$CONTENT_TOPK_OUT_DIR/${CONTENT_PAGE5_LABEL}.dev.prediction.json"
OOF_INSERT4_PRED="$OOF_OUT_DIR/insert_rank4/mmqa_train_to_dev_content_aware_counterfactual_oof_insert4.dev.prediction.json"

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    echo "missing_${label}: $path" >&2
    exit 1
  fi
}

if [[ "$RUN_EXACT_DENSE_DEV" == "1" ]]; then
  echo
  echo "== Exact MaxSim dense rerank: dev =="
  SPLIT=dev \
  TOP_PAGES="$TOP_PAGES" \
  FAISS_NPROBE="$FAISS_NPROBE" \
  EXACT_MAXSIM_OUT_DIR="$EXACT_DENSE_DEV_OUT_DIR" \
  EXACT_MAXSIM_LABEL="$EXACT_DENSE_DEV_LABEL" \
  bash "$REPO_ROOT/scripts/run_m3docvqa_exact_maxsim.sh"
fi
require_file exact_dense_dev "$EXACT_DENSE_DEV_PRED"

if [[ "$RUN_EXACT_DENSE_TRAIN" == "1" ]]; then
  echo
  echo "== Exact MaxSim dense rerank: train =="
  SPLIT=train \
  TOP_PAGES="$TOP_PAGES" \
  FAISS_NPROBE="$FAISS_NPROBE" \
  EXACT_MAXSIM_OUT_DIR="$EXACT_DENSE_TRAIN_OUT_DIR" \
  EXACT_MAXSIM_LABEL="$EXACT_DENSE_TRAIN_LABEL" \
  bash "$REPO_ROOT/scripts/run_m3docvqa_exact_maxsim.sh"
fi
require_file exact_dense_train "$EXACT_DENSE_TRAIN_PRED"

if [[ "$RUN_GPP_DEV" == "1" ]]; then
  echo
  echo "== Exact MaxSim-derived GPP hyperlink ablation: dev =="
  SPLIT=dev \
  DENSE_PRED="$EXACT_DENSE_DEV_PRED" \
  SPARSE_PRED="$DEV_SPARSE_PRED" \
  GRAPH_OUT_DIR="$GPP_DEV_OUT_DIR" \
  LABEL_PREFIX="$GPP_DEV_LABEL_PREFIX" \
  GRAPH_PROFILE="$GRAPH_PROFILE" \
  bash "$REPO_ROOT/examples/run_m3docvqa_gpp_hyperlink_node_ablation.sh"
fi
require_file gpp_dev_no_hyperlink "$GPP_DEV_NO_HYPERLINK"

if [[ "$RUN_GPP_TRAIN" == "1" ]]; then
  echo
  echo "== Exact MaxSim-derived GPP hyperlink ablation: train =="
  SPLIT=train \
  DENSE_PRED="$EXACT_DENSE_TRAIN_PRED" \
  SPARSE_PRED="$TRAIN_SPARSE_PRED" \
  GRAPH_OUT_DIR="$GPP_TRAIN_OUT_DIR" \
  LABEL_PREFIX="$GPP_TRAIN_LABEL_PREFIX" \
  GRAPH_PROFILE="$GRAPH_PROFILE" \
  bash "$REPO_ROOT/examples/run_m3docvqa_gpp_hyperlink_node_ablation.sh"
fi
require_file gpp_train_no_hyperlink "$GPP_TRAIN_NO_HYPERLINK"

if [[ "$RUN_CONTENT_AWARE_TOPK" == "1" ]]; then
  echo
  echo "== Exact MaxSim content-aware adaptive top-k ablation =="
  GPP_TRAIN_OUT_DIR="$GPP_TRAIN_OUT_DIR" \
  GPP_EVAL_OUT_DIR="$GPP_DEV_OUT_DIR" \
  BASE_LABEL="$CONTENT_BASE_LABEL" \
  BASE_TRAIN_PRED="$GPP_TRAIN_NO_HYPERLINK" \
  BASE_EVAL_PRED="$GPP_DEV_NO_HYPERLINK" \
  OUT_DIR="$CONTENT_TOPK_OUT_DIR" \
  FORCE_RERUN="${FORCE_RERUN_CONTENT:-1}" \
  bash "$REPO_ROOT/examples/run_m3docvqa_content_aware_adaptive_topk_ablation.sh"
fi
require_file content_page5_pred "$CONTENT_PAGE5_PRED"

if [[ "$RUN_FEATURE_ABLATION" == "1" ]]; then
  echo
  echo "== Exact MaxSim content-aware feature ablation =="
  GPP_TRAIN_OUT_DIR="$GPP_TRAIN_OUT_DIR" \
  GPP_EVAL_OUT_DIR="$GPP_DEV_OUT_DIR" \
  BASE_LABEL="$CONTENT_BASE_LABEL" \
  BASE_TRAIN_PRED="$GPP_TRAIN_NO_HYPERLINK" \
  BASE_EVAL_PRED="$GPP_DEV_NO_HYPERLINK" \
  OUT_DIR="$FEATURE_OUT_DIR" \
  FORCE_RERUN="${FORCE_RERUN_FEATURE:-1}" \
  bash "$REPO_ROOT/examples/run_m3docvqa_content_aware_feature_ablation.sh"
fi

if [[ "$RUN_PSEUDO_LABEL_ABLATION" == "1" ]]; then
  echo
  echo "== Exact MaxSim content-aware pseudo-label ablation =="
  GPP_TRAIN_OUT_DIR="$GPP_TRAIN_OUT_DIR" \
  GPP_EVAL_OUT_DIR="$GPP_DEV_OUT_DIR" \
  BASE_LABEL="$CONTENT_BASE_LABEL" \
  BASE_TRAIN_PRED="$GPP_TRAIN_NO_HYPERLINK" \
  BASE_EVAL_PRED="$GPP_DEV_NO_HYPERLINK" \
  OUT_DIR="$PSEUDO_LABEL_OUT_DIR" \
  FORCE_RERUN="${FORCE_RERUN_PSEUDO_LABEL:-1}" \
  bash "$REPO_ROOT/examples/run_m3docvqa_content_aware_pseudo_label_ablation.sh"
fi

if [[ "$RUN_OOF_HYBRID" == "1" ]]; then
  echo
  echo "== Exact MaxSim OOF hybrid boundary repair =="
  GPP_TRAIN_OUT_DIR="$GPP_TRAIN_OUT_DIR" \
  GPP_EVAL_OUT_DIR="$GPP_DEV_OUT_DIR" \
  GPP_TRAIN_LABEL_PREFIX="$GPP_TRAIN_LABEL_PREFIX" \
  GPP_EVAL_LABEL_PREFIX="$GPP_DEV_LABEL_PREFIX" \
  CONTENT_AWARE_DIR="$CONTENT_TOPK_OUT_DIR" \
  CONTENT_AWARE_LABEL="$CONTENT_PAGE5_LABEL" \
  OUT_DIR="$OOF_OUT_DIR" \
  FORCE_REBUILD_CONTENT_EVAL=0 \
  FORCE_REBUILD_OOF_CONTENT="${FORCE_REBUILD_OOF_CONTENT:-1}" \
  bash "$REPO_ROOT/examples/run_m3docvqa_content_aware_counterfactual_oof_hybrid.sh"
fi

if [[ "$RUN_FINAL_EVAL" == "1" ]]; then
  echo
  echo "== Exact MaxSim final pseudo-page and original-gold evaluation tables =="
  DENSE_PRED="$EXACT_DENSE_DEV_PRED" \
  GPP_NO_HYPERLINK_PRED="$GPP_DEV_NO_HYPERLINK" \
  CONTENT_AWARE_PAGE5_PRED="$CONTENT_PAGE5_PRED" \
  OOF_HYBRID_INSERT4_PRED="$OOF_INSERT4_PRED" \
  OUT_DIR="$FINAL_EVAL_OUT_DIR" \
  bash "$REPO_ROOT/examples/evaluate_m3docvqa_best_content_aware_final.sh"
fi

cat <<EOF

exact_dense_dev=$EXACT_DENSE_DEV_PRED
exact_dense_train=$EXACT_DENSE_TRAIN_PRED
gpp_dev_no_hyperlink=$GPP_DEV_NO_HYPERLINK
gpp_train_no_hyperlink=$GPP_TRAIN_NO_HYPERLINK
content_aware_page5=$CONTENT_PAGE5_PRED
oof_hybrid_insert4=$OOF_INSERT4_PRED
final_eval_dir=$FINAL_EVAL_OUT_DIR
EOF
