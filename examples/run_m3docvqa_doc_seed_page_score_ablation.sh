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

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    echo "missing_${label}: $path" >&2
    exit 1
  fi
}

SPLIT="${SPLIT:-dev}"
GOLD="${M3DOCVQA_GOLD:-$GOLD}"
DENSE_PRED="${M3DOCVQA_DENSE_PRED:-$LOCAL_OUTPUT_DIR/m3docvqa_plain_top224_mmqa_${SPLIT}/mmqa_${SPLIT}_plain_top224_nprobe${FAISS_NPROBE}_effdiag_all.prediction.json}"
SPARSE_PRED="${M3DOCVQA_SPARSE_PRED:-$LOCAL_OUTPUT_DIR/m3docvqa_splade_mmqa_${SPLIT}/mmqa_${SPLIT}_splade.prediction.json}"
DOC_PAGES_JSONL="${M3DOCVQA_PAGE_TEXT_JSONL:-$LOCAL_OUTPUT_DIR/m3docvqa_page_text/m3docvqa_${SPLIT}_page_text.jsonl}"
SPLADE_INDEX_PT="${SPLADE_INDEX_PT:-$LOCAL_OUTPUT_DIR/m3docvqa_splade/m3docvqa_${SPLIT}_splade.pt}"
GRAPH_OUT_DIR="${GRAPH_OUT_DIR:-$LOCAL_OUTPUT_DIR/m3docvqa_doc_seed_page_score_ablation}"
LABEL_PREFIX="${LABEL_PREFIX:-mmqa_${SPLIT}_docseed_page_score}"
GRAPH_PROFILE="${GRAPH_PROFILE:-denseheavy125_medium_both}"
DOC_SEED_GRAPH_SIZE_REFERENCE="${DOC_SEED_GRAPH_SIZE_REFERENCE:-20.0}"
DOC_SEED_GRAPH_SIZE_MIN_MULT="${DOC_SEED_GRAPH_SIZE_MIN_MULT:-0.25}"
DOC_SEED_GRAPH_SIZE_MAX_MULT="${DOC_SEED_GRAPH_SIZE_MAX_MULT:-2.0}"
FINAL_SELECTION_MODE="${FINAL_SELECTION_MODE:-score}"
FINAL_SELECTION_TOP_K="${FINAL_SELECTION_TOP_K:-4}"
FINAL_SELECTION_CANDIDATE_POOL="${FINAL_SELECTION_CANDIDATE_POOL:-20}"
FINAL_SELECTION_NEW_DOC_BONUS="${FINAL_SELECTION_NEW_DOC_BONUS:-0.05}"
FINAL_SELECTION_SAME_DOC_PENALTY="${FINAL_SELECTION_SAME_DOC_PENALTY:-0.05}"
RECALL_K_VALUES="${RECALL_K_VALUES:-1 2 4 5 10 20 50 100 500 1000}"

require_file gold "$GOLD"
require_file dense_pred "$DENSE_PRED"
require_file sparse_pred "$SPARSE_PRED"
if [[ -n "$DOC_PAGES_JSONL" && ! -f "$DOC_PAGES_JSONL" ]]; then
  echo "warning_missing_doc_pages_jsonl: $DOC_PAGES_JSONL" >&2
fi

mkdir -p "$GRAPH_OUT_DIR"

run_variant() {
  local variant="$1"
  local seed_mode="$2"
  local seed_weight="$3"
  local page_score_mode="$4"
  local page_top_k="$5"
  local label="${LABEL_PREFIX}_${variant}"

  echo
  echo "== m3docvqa doc seed page score: $variant =="
  DATA_NAME="m3-docvqa" \
  SPLIT="$SPLIT" \
  GOLD="$GOLD" \
  DENSE_PRED="$DENSE_PRED" \
  SPARSE_PRED="$SPARSE_PRED" \
  DOC_PAGES_JSONL="$DOC_PAGES_JSONL" \
  SPLADE_INDEX_PT="$SPLADE_INDEX_PT" \
  GRAPH_OUT_DIR="$GRAPH_OUT_DIR" \
  GRAPH_PROFILE="$GRAPH_PROFILE" \
  GRAPH_LABEL="$label" \
  RECALL_K_VALUES="$RECALL_K_VALUES" \
  DOC_SEED_MODE="$seed_mode" \
  DOC_SEED_WEIGHT="$seed_weight" \
  DOC_SEED_PAGE_SCORE_MODE="$page_score_mode" \
  DOC_SEED_PAGE_TOP_K="$page_top_k" \
  DOC_SEED_GRAPH_SIZE_REFERENCE="$DOC_SEED_GRAPH_SIZE_REFERENCE" \
  DOC_SEED_GRAPH_SIZE_MIN_MULT="$DOC_SEED_GRAPH_SIZE_MIN_MULT" \
  DOC_SEED_GRAPH_SIZE_MAX_MULT="$DOC_SEED_GRAPH_SIZE_MAX_MULT" \
  DOC_DOC_EDGE_MODE=none \
  DOC_DOC_EDGE_WEIGHT=0.0 \
  PDF_HYPERLINK_EDGES_JSONL= \
  PDF_HYPERLINK_EDGE_WEIGHT=0.0 \
  FINAL_SELECTION_MODE="$FINAL_SELECTION_MODE" \
  FINAL_SELECTION_TOP_K="$FINAL_SELECTION_TOP_K" \
  FINAL_SELECTION_CANDIDATE_POOL="$FINAL_SELECTION_CANDIDATE_POOL" \
  FINAL_SELECTION_NEW_DOC_BONUS="$FINAL_SELECTION_NEW_DOC_BONUS" \
  FINAL_SELECTION_SAME_DOC_PENALTY="$FINAL_SELECTION_SAME_DOC_PENALTY" \
  bash "$REPO_ROOT/scripts/run_m3docvqa_page_preserving_graph_pipeline.sh"
}

run_variant docseed_none rrf 0.0 mean 3
run_variant docseed_rrf_0p50 rrf 0.50 mean 3

for weight in 0.25 0.50 1.00; do
  suffix="${weight//./p}"
  run_variant "docseed_page_mean_${suffix}" page_seed "$weight" mean 3
  run_variant "docseed_page_max_${suffix}" page_seed "$weight" max 3
  run_variant "docseed_page_top3mean_${suffix}" page_seed "$weight" topk_mean 3
done

for weight in 0.10 0.25 0.50; do
  suffix="${weight//./p}"
  run_variant "docseed_page_sum_${suffix}" page_seed "$weight" sum 3
done

baseline_pred="$GRAPH_OUT_DIR/${LABEL_PREFIX}_docseed_none.prediction.json"
candidate_preds=( "$GRAPH_OUT_DIR/${LABEL_PREFIX}_"*.prediction.json )
for candidate_pred in "${candidate_preds[@]}"; do
  if [[ "$candidate_pred" == "$baseline_pred" ]]; then
    continue
  fi
  variant="$(basename "$candidate_pred" .prediction.json)"
  "$PYTHON_BIN" "$REPO_ROOT/scripts/compare_m3docvqa_retrieval_runs.py" \
    --baseline "$baseline_pred" \
    --candidate "$candidate_pred" \
    --gold "$GOLD" \
    --recall-k $RECALL_K_VALUES \
    --json > "$GRAPH_OUT_DIR/${variant}.vs_docseed_none.json"
done

analysis_paths=( "$GRAPH_OUT_DIR/${LABEL_PREFIX}_"*.retrieval_analysis.json )
if [[ -e "${analysis_paths[0]}" ]]; then
  "$PYTHON_BIN" "$REPO_ROOT/scripts/summarize_m3docvqa_retrieval_analyses.py" \
    --format markdown \
    "${analysis_paths[@]}" \
    | tee "$GRAPH_OUT_DIR/${LABEL_PREFIX}_recall_table.md"

  "$PYTHON_BIN" "$REPO_ROOT/scripts/summarize_m3docvqa_retrieval_analyses.py" \
    --format csv \
    "${analysis_paths[@]}" \
    > "$GRAPH_OUT_DIR/${LABEL_PREFIX}_recall_table.csv"

  echo "saved_recall_table_md=$GRAPH_OUT_DIR/${LABEL_PREFIX}_recall_table.md"
  echo "saved_recall_table_csv=$GRAPH_OUT_DIR/${LABEL_PREFIX}_recall_table.csv"
fi
