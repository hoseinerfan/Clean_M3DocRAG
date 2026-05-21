#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/env/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

: "${DATA_NAME:?Set DATA_NAME, e.g. mmdocir, sciegqa, vidoseek, or vidore-v3}"
: "${DATA_ROOT:?Set DATA_ROOT to the converted dataset root}"
: "${DENSE_PRED:?Set DENSE_PRED to the dense/plain_top224 prediction JSON}"
: "${SPARSE_PRED:?Set SPARSE_PRED to the SPLADE prediction JSON}"
: "${OUT_DIR:?Set OUT_DIR for graph-PPR outputs}"

SPLIT="${SPLIT:-dev}"
GOLD="${GOLD:-$DATA_ROOT/MMQA_${SPLIT}.jsonl}"
LABEL_PREFIX="${LABEL_PREFIX:-${DATA_NAME}_pagepreserve_sweep}"
RECALL_K_VALUES="${RECALL_K_VALUES:-1 2 4 5 10 20 50 100}"
RUN_PLAIN_BASELINE="${RUN_PLAIN_BASELINE:-1}"
RUN_DOC_SHORTLIST_CONTROL="${RUN_DOC_SHORTLIST_CONTROL:-1}"

mkdir -p "$OUT_DIR"

if [[ "$RUN_PLAIN_BASELINE" == "1" ]]; then
  echo "== plain_top224 baseline =="
  "$PYTHON_BIN" "$REPO_ROOT/mmdocir/evaluate_mmdocir_retrieval.py" \
    --pred "$DENSE_PRED" \
    --gold "$GOLD" \
    --recall-k $RECALL_K_VALUES \
    | tee "$OUT_DIR/${LABEL_PREFIX}_plain_top224.eval.txt"
fi

run_graph() {
  local label="$1"
  local dense_weight="$2"
  local sparse_weight="$3"
  local final_ppr_page_weight="$4"
  local final_ppr_doc_weight="$5"
  local ppr_iters="$6"
  local page_doc_edge_weight="$7"
  local same_doc_window="$8"
  local adjacent_page_edge_weight="$9"

  echo "== $label =="
  DATA_NAME="$DATA_NAME" \
  DATA_ROOT="$DATA_ROOT" \
  DENSE_PRED="$DENSE_PRED" \
  SPARSE_PRED="$SPARSE_PRED" \
  OUT_DIR="$OUT_DIR" \
  GOLD="$GOLD" \
  GRAPH_PROFILE=page_rank_probe \
  GRAPH_LABEL="$label" \
  FINAL_TOP_PAGES=1000 \
  PER_DOC_PAGE_LIMIT=0 \
  DENSE_WEIGHT="$dense_weight" \
  SPARSE_WEIGHT="$sparse_weight" \
  RESTART_PROB=0.15 \
  PPR_ITERS="$ppr_iters" \
  PAGE_DOC_EDGE_WEIGHT="$page_doc_edge_weight" \
  SAME_DOC_WINDOW="$same_doc_window" \
  ADJACENT_PAGE_EDGE_WEIGHT="$adjacent_page_edge_weight" \
  FINAL_PAGE_SEED_WEIGHT=1.0 \
  FINAL_PPR_PAGE_WEIGHT="$final_ppr_page_weight" \
  FINAL_PPR_DOC_WEIGHT="$final_ppr_doc_weight" \
  RECALL_K_VALUES="$RECALL_K_VALUES" \
  bash "$REPO_ROOT/scripts/run_external_graph_ppr_pipeline.sh"
}

source_labels=("equal" "denseheavy125" "denseheavy150")
dense_weights=("1.0" "1.25" "1.5")
sparse_weights=("1.0" "0.75" "0.5")

ppr_labels=(
  "page_rrf"
  "light_page"
  "light_doc"
  "light_both"
  "medium_both"
  "m3best_pagepreserve"
)
page_weights=("0.0" "0.25" "0.0" "0.25" "0.5" "1.5")
doc_weights=("0.0" "0.0" "0.25" "0.25" "0.25" "0.75")

for source_i in "${!source_labels[@]}"; do
  for ppr_i in "${!ppr_labels[@]}"; do
    label="${LABEL_PREFIX}_${source_labels[$source_i]}_${ppr_labels[$ppr_i]}"
    if [[ "${ppr_labels[$ppr_i]}" == "page_rrf" ]]; then
      run_graph "$label" \
        "${dense_weights[$source_i]}" \
        "${sparse_weights[$source_i]}" \
        "${page_weights[$ppr_i]}" \
        "${doc_weights[$ppr_i]}" \
        0 \
        0.0 \
        0 \
        0.0
    else
      run_graph "$label" \
        "${dense_weights[$source_i]}" \
        "${sparse_weights[$source_i]}" \
        "${page_weights[$ppr_i]}" \
        "${doc_weights[$ppr_i]}" \
        30 \
        1.0 \
        1 \
        0.25
    fi
  done
done

if [[ "$RUN_DOC_SHORTLIST_CONTROL" == "1" ]]; then
  echo "== ${LABEL_PREFIX}_doc_shortlist_control =="
  DATA_NAME="$DATA_NAME" \
  DATA_ROOT="$DATA_ROOT" \
  DENSE_PRED="$DENSE_PRED" \
  SPARSE_PRED="$SPARSE_PRED" \
  OUT_DIR="$OUT_DIR" \
  GOLD="$GOLD" \
  GRAPH_PROFILE=doc_shortlist_best \
  GRAPH_LABEL="${LABEL_PREFIX}_doc_shortlist_control" \
  RECALL_K_VALUES="$RECALL_K_VALUES" \
  bash "$REPO_ROOT/scripts/run_external_graph_ppr_pipeline.sh"
fi

summary_paths=( "$OUT_DIR/${LABEL_PREFIX}_"*.summary.json )
if [[ -e "${summary_paths[0]}" ]]; then
  "$PYTHON_BIN" "$REPO_ROOT/scripts/summarize_graph_ppr_summaries.py" \
    --recall-table \
    --format markdown \
    "${summary_paths[@]}" \
    | tee "$OUT_DIR/${LABEL_PREFIX}_recall_table.md"

  "$PYTHON_BIN" "$REPO_ROOT/scripts/summarize_graph_ppr_summaries.py" \
    --recall-table \
    --format csv \
    "${summary_paths[@]}" \
    > "$OUT_DIR/${LABEL_PREFIX}_recall_table.csv"

  echo "saved_recall_table_md=$OUT_DIR/${LABEL_PREFIX}_recall_table.md"
  echo "saved_recall_table_csv=$OUT_DIR/${LABEL_PREFIX}_recall_table.csv"
fi
