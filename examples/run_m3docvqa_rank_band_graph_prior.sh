#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

RANK_BAND_OUT_DIR="${RANK_BAND_OUT_DIR:-$REPO_ROOT/output/m3docvqa_rank_band_page_promotion}"
RANK_BAND_LABEL="${RANK_BAND_LABEL:-mmqa_train_to_dev_rank_band_page_promotion}"
LEARNED_PAGE_PRIOR_JSONL="${LEARNED_PAGE_PRIOR_JSONL:-$RANK_BAND_OUT_DIR/${RANK_BAND_LABEL}.dev.learned_prior.jsonl}"

if [[ "${SKIP_RANK_BAND_TRAIN:-0}" != "1" ]]; then
  OUT_DIR="$RANK_BAND_OUT_DIR" \
  LABEL="$RANK_BAND_LABEL" \
  bash "$REPO_ROOT/examples/run_m3docvqa_rank_band_page_promotion.sh"
fi

if [[ ! -f "$LEARNED_PAGE_PRIOR_JSONL" ]]; then
  echo "missing_learned_page_prior_jsonl=$LEARNED_PAGE_PRIOR_JSONL" >&2
  exit 1
fi

export LEARNED_PAGE_PRIOR_JSONL
export LEARNED_PAGE_PRIOR_SEED_WEIGHT="${LEARNED_PAGE_PRIOR_SEED_WEIGHT:-0.25}"
export LEARNED_PAGE_PRIOR_MIN_BASE_RANK="${LEARNED_PAGE_PRIOR_MIN_BASE_RANK:-5}"
export LEARNED_PAGE_PRIOR_MAX_BASE_RANK="${LEARNED_PAGE_PRIOR_MAX_BASE_RANK:-1000}"
export LEARNED_PAGE_PRIOR_TOP_K="${LEARNED_PAGE_PRIOR_TOP_K:-200}"
export LEARNED_PAGE_PRIOR_SCORE_FIELD="${LEARNED_PAGE_PRIOR_SCORE_FIELD:-learned_score}"
export LEARNED_PAGE_PRIOR_NORMALIZE="${LEARNED_PAGE_PRIOR_NORMALIZE:-1}"
export GRAPH_OUT_DIR="${GRAPH_OUT_DIR:-$REPO_ROOT/output/m3docvqa_rank_band_graph_prior}"
export GRAPH_LABEL="${GRAPH_LABEL:-mmqa_dev_gpp_rank_band_graph_prior_w${LEARNED_PAGE_PRIOR_SEED_WEIGHT}_top${LEARNED_PAGE_PRIOR_TOP_K}}"
if [[ -z "${DENSE_PRED:-}" && -n "${EVAL_DENSE_PRED:-}" ]]; then
  export DENSE_PRED="$EVAL_DENSE_PRED"
fi
if [[ -z "${SPARSE_PRED:-}" && -n "${EVAL_SPLADE_PRED:-}" ]]; then
  export SPARSE_PRED="$EVAL_SPLADE_PRED"
fi

echo "using_learned_page_prior_jsonl=$LEARNED_PAGE_PRIOR_JSONL"
echo "using_learned_page_prior_seed_weight=$LEARNED_PAGE_PRIOR_SEED_WEIGHT"
echo "using_learned_page_prior_min_base_rank=$LEARNED_PAGE_PRIOR_MIN_BASE_RANK"
echo "using_learned_page_prior_max_base_rank=$LEARNED_PAGE_PRIOR_MAX_BASE_RANK"
echo "using_learned_page_prior_top_k=$LEARNED_PAGE_PRIOR_TOP_K"
echo "using_dense_pred=${DENSE_PRED:-}"
echo "using_sparse_pred=${SPARSE_PRED:-}"

bash "$REPO_ROOT/scripts/run_m3docvqa_page_preserving_graph_pipeline.sh"
