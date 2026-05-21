#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/env/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

OUTDIR="${OUTDIR:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_splade_mmqa_dev}"
GOLD="${GOLD:-$REPO_ROOT/data/m3-docvqa/multimodalqa/MMQA_dev.jsonl}"
DENSE_PRED="${DENSE_PRED:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/mmqa_dev_plain_top224_nprobe4_effdiag_all.prediction.json}"
RAW_DENSE_PRED="${RAW_DENSE_PRED:-$REPO_ROOT/output/retrieval_only_dev_ret1000full_nprobe4/colpali-v1.2_ivfflat_nprobe4_ret1000_2026-05-10_10-28-25.json}"
EXPAND_DENSE_PRED="${EXPAND_DENSE_PRED:-$DENSE_PRED}"
SPLADE_PRED="${SPLADE_PRED:-$OUTDIR/mmqa_dev_splade.prediction.json}"
SPLADE_INDEX_PT="${SPLADE_INDEX_PT:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_splade/m3docvqa_dev_splade.pt}"

QUESTION_TYPE="${QUESTION_TYPE-ImageListQ}"
LABEL_PREFIX="${LABEL_PREFIX:-imagelistq}"
RRF_K="${RRF_K:-10}"
DENSE_WEIGHT="${DENSE_WEIGHT:-1.0}"
SPARSE_WEIGHT="${SPARSE_WEIGHT:-1.0}"
BEST_RESTART_PROB="${BEST_RESTART_PROB:-0.20}"
BEST_PAGE_PPR_WEIGHT="${BEST_PAGE_PPR_WEIGHT:-1.5}"
BEST_DOC_PPR_WEIGHT="${BEST_DOC_PPR_WEIGHT:-0.5}"

mkdir -p "$OUTDIR"

run_graph() {
  local label="$1"
  shift
  "$PYTHON_BIN" "$REPO_ROOT/scripts/graph_rerank_page_retrieval_predictions.py" \
    --dense-prediction-json "$DENSE_PRED" \
    --sparse-prediction-json "$SPLADE_PRED" \
    --gold "$GOLD" \
    --question-type "$QUESTION_TYPE" \
    --rrf-k "$RRF_K" \
    --dense-weight "$DENSE_WEIGHT" \
    --sparse-weight "$SPARSE_WEIGHT" \
    "$@" \
    --output-prediction-json "$OUTDIR/${label}.prediction.json" \
    --output-summary-json "$OUTDIR/${label}.summary.json"
}

if [[ "${RUN_DOC_TOP20:-1}" == "1" ]]; then
  run_graph "${LABEL_PREFIX}_graph_ppr_eq_k10_top20" \
    --dense-top-pages 1000 \
    --sparse-top-pages 1000 \
    --final-top-pages 20 \
    --per-doc-page-limit 1
fi

if [[ "${RUN_PAGE_TOP500:-1}" == "1" ]]; then
  run_graph "${LABEL_PREFIX}_graph_ppr_eq_k10_top500pages" \
    --dense-top-pages 1000 \
    --sparse-top-pages 1000 \
    --final-top-pages 500 \
    --per-doc-page-limit 0
fi

if [[ "${RUN_EXPAND_TOP500:-0}" == "1" ]]; then
  run_graph "${LABEL_PREFIX}_graph_ppr_sparse_expand_top500pages" \
    --dense-prediction-json "$EXPAND_DENSE_PRED" \
    --dense-top-pages 200 \
    --sparse-top-pages 200 \
    --final-top-pages 500 \
    --per-doc-page-limit 0 \
    --splade-index-pt "$SPLADE_INDEX_PT" \
    --expansion-top-pages "${EXPANSION_TOP_PAGES:-1000}" \
    --expand-from-top-dense-pages "${EXPAND_FROM_TOP_DENSE_PAGES:-50}" \
    --expand-from-top-sparse-pages "${EXPAND_FROM_TOP_SPARSE_PAGES:-50}" \
    --expansion-weight "${EXPANSION_WEIGHT:-1.0}" \
    --expansion-source-term-topk "${EXPANSION_SOURCE_TERM_TOPK:-32}" \
    --expansion-query-topk-terms "${EXPANSION_QUERY_TOPK_TERMS:-256}"
fi

if [[ "${RUN_RAW_EXPAND_TOP500:-0}" == "1" ]]; then
  run_graph "${LABEL_PREFIX}_graph_ppr_rawdense_sparse_expand_top500pages" \
    --dense-prediction-json "$RAW_DENSE_PRED" \
    --dense-top-pages 200 \
    --sparse-top-pages 200 \
    --final-top-pages 500 \
    --per-doc-page-limit 0 \
    --splade-index-pt "$SPLADE_INDEX_PT" \
    --expansion-top-pages "${EXPANSION_TOP_PAGES:-1000}" \
    --expand-from-top-dense-pages "${EXPAND_FROM_TOP_DENSE_PAGES:-50}" \
    --expand-from-top-sparse-pages "${EXPAND_FROM_TOP_SPARSE_PAGES:-50}" \
    --expansion-weight "${EXPANSION_WEIGHT:-1.0}" \
    --expansion-source-term-topk "${EXPANSION_SOURCE_TERM_TOPK:-32}" \
    --expansion-query-topk-terms "${EXPANSION_QUERY_TOPK_TERMS:-256}"
fi

if [[ "${RUN_BEST_TOP20:-0}" == "1" ]]; then
  run_graph "${LABEL_PREFIX}_graph_ppr_eq_k10_top20_nodocseed_pagew1p5" \
    --dense-top-pages 1000 \
    --sparse-top-pages 1000 \
    --final-top-pages 20 \
    --per-doc-page-limit 1 \
    --doc-seed-weight 0.0 \
    --final-ppr-page-weight 1.5
fi

if [[ "${RUN_CUSTOM_BEST_TOP20:-0}" == "1" ]]; then
  label_restart="${BEST_RESTART_PROB/./p}"
  label_page_weight="${BEST_PAGE_PPR_WEIGHT/./p}"
  label_doc_weight="${BEST_DOC_PPR_WEIGHT/./p}"
  run_graph "${LABEL_PREFIX}_graph_ppr_eq_k10_top20_nodocseed_restart${label_restart}_pagew${label_page_weight}_docw${label_doc_weight}" \
    --dense-top-pages 1000 \
    --sparse-top-pages 1000 \
    --final-top-pages 20 \
    --per-doc-page-limit 1 \
    --doc-seed-weight 0.0 \
    --restart-prob "$BEST_RESTART_PROB" \
    --final-ppr-page-weight "$BEST_PAGE_PPR_WEIGHT" \
    --final-ppr-doc-weight "$BEST_DOC_PPR_WEIGHT"
fi

if [[ "${RUN_SOURCE_ABLATIONS:-0}" == "1" ]]; then
  run_graph "${LABEL_PREFIX}_graph_ppr_sourceablate_no_splade" \
    --dense-top-pages 1000 \
    --sparse-top-pages 0 \
    --sparse-weight 0.0 \
    --final-top-pages 20 \
    --per-doc-page-limit 1 \
    --doc-seed-weight 0.0 \
    --restart-prob "$BEST_RESTART_PROB" \
    --final-ppr-page-weight "$BEST_PAGE_PPR_WEIGHT" \
    --final-ppr-doc-weight "$BEST_DOC_PPR_WEIGHT"

  run_graph "${LABEL_PREFIX}_graph_ppr_sourceablate_splade_only" \
    --dense-top-pages 0 \
    --sparse-top-pages 1000 \
    --dense-weight 0.0 \
    --final-top-pages 20 \
    --per-doc-page-limit 1 \
    --doc-seed-weight 0.0 \
    --restart-prob "$BEST_RESTART_PROB" \
    --final-ppr-page-weight "$BEST_PAGE_PPR_WEIGHT" \
    --final-ppr-doc-weight "$BEST_DOC_PPR_WEIGHT"

  run_graph "${LABEL_PREFIX}_graph_ppr_sourceablate_rawdense_plus_splade" \
    --dense-prediction-json "$RAW_DENSE_PRED" \
    --dense-top-pages 1000 \
    --sparse-top-pages 1000 \
    --final-top-pages 20 \
    --per-doc-page-limit 1 \
    --doc-seed-weight 0.0 \
    --restart-prob "$BEST_RESTART_PROB" \
    --final-ppr-page-weight "$BEST_PAGE_PPR_WEIGHT" \
    --final-ppr-doc-weight "$BEST_DOC_PPR_WEIGHT"

  run_graph "${LABEL_PREFIX}_graph_ppr_sourceablate_rawdense_only" \
    --dense-prediction-json "$RAW_DENSE_PRED" \
    --dense-top-pages 1000 \
    --sparse-top-pages 0 \
    --sparse-weight 0.0 \
    --final-top-pages 20 \
    --per-doc-page-limit 1 \
    --doc-seed-weight 0.0 \
    --restart-prob "$BEST_RESTART_PROB" \
    --final-ppr-page-weight "$BEST_PAGE_PPR_WEIGHT" \
    --final-ppr-doc-weight "$BEST_DOC_PPR_WEIGHT"
fi

if [[ "${RUN_ABLATIONS:-0}" == "1" ]]; then
  run_graph "${LABEL_PREFIX}_graph_ppr_eq_k10_top20_seedonly" \
    --dense-top-pages 1000 \
    --sparse-top-pages 1000 \
    --final-top-pages 20 \
    --per-doc-page-limit 1 \
    --final-page-seed-weight 1.0 \
    --final-ppr-page-weight 0.0 \
    --final-ppr-doc-weight 0.0

  run_graph "${LABEL_PREFIX}_graph_ppr_eq_k10_top20_pageppr_only" \
    --dense-top-pages 1000 \
    --sparse-top-pages 1000 \
    --final-top-pages 20 \
    --per-doc-page-limit 1 \
    --final-page-seed-weight 0.0 \
    --final-ppr-page-weight 1.0 \
    --final-ppr-doc-weight 0.0

  run_graph "${LABEL_PREFIX}_graph_ppr_eq_k10_top20_docppr_only" \
    --dense-top-pages 1000 \
    --sparse-top-pages 1000 \
    --final-top-pages 20 \
    --per-doc-page-limit 1 \
    --final-page-seed-weight 0.0 \
    --final-ppr-page-weight 0.0 \
    --final-ppr-doc-weight 1.0

  run_graph "${LABEL_PREFIX}_graph_ppr_eq_k10_top20_no_adjacent" \
    --dense-top-pages 1000 \
    --sparse-top-pages 1000 \
    --final-top-pages 20 \
    --per-doc-page-limit 1 \
    --same-doc-window 0 \
    --adjacent-page-edge-weight 0.0

  run_graph "${LABEL_PREFIX}_graph_ppr_eq_k10_top20_no_doc_seed" \
    --dense-top-pages 1000 \
    --sparse-top-pages 1000 \
    --final-top-pages 20 \
    --per-doc-page-limit 1 \
    --doc-seed-weight 0.0

  DENSE_WEIGHT=0.75 SPARSE_WEIGHT=1.25 run_graph "${LABEL_PREFIX}_graph_ppr_sparseheavy_k10_top20" \
    --dense-top-pages 1000 \
    --sparse-top-pages 1000 \
    --final-top-pages 20 \
    --per-doc-page-limit 1

  DENSE_WEIGHT=1.25 SPARSE_WEIGHT=0.75 run_graph "${LABEL_PREFIX}_graph_ppr_denseheavy_k10_top20" \
    --dense-top-pages 1000 \
    --sparse-top-pages 1000 \
    --final-top-pages 20 \
    --per-doc-page-limit 1
fi

if [[ "${RUN_NO_DOC_SEED_SWEEP:-0}" == "1" ]]; then
  for doc_ppr_weight in 0.0 0.25 0.5 0.75 1.0 1.5; do
    label_doc_weight="${doc_ppr_weight/./p}"
    run_graph "${LABEL_PREFIX}_graph_ppr_eq_k10_top20_nodocseed_docw${label_doc_weight}" \
      --dense-top-pages 1000 \
      --sparse-top-pages 1000 \
      --final-top-pages 20 \
      --per-doc-page-limit 1 \
      --doc-seed-weight 0.0 \
      --final-ppr-doc-weight "$doc_ppr_weight"
  done

  for page_ppr_weight in 0.25 0.5 1.0 1.5 2.0; do
    label_page_weight="${page_ppr_weight/./p}"
    run_graph "${LABEL_PREFIX}_graph_ppr_eq_k10_top20_nodocseed_pagew${label_page_weight}" \
      --dense-top-pages 1000 \
      --sparse-top-pages 1000 \
      --final-top-pages 20 \
      --per-doc-page-limit 1 \
      --doc-seed-weight 0.0 \
      --final-ppr-page-weight "$page_ppr_weight"
  done

  for restart_prob in 0.10 0.15 0.20 0.25 0.35; do
    label_restart="${restart_prob/./p}"
    run_graph "${LABEL_PREFIX}_graph_ppr_eq_k10_top20_nodocseed_restart${label_restart}" \
      --dense-top-pages 1000 \
      --sparse-top-pages 1000 \
      --final-top-pages 20 \
      --per-doc-page-limit 1 \
      --doc-seed-weight 0.0 \
      --restart-prob "$restart_prob"
  done
fi

if [[ "${RUN_BEST_COMBO_SWEEP:-0}" == "1" ]]; then
  for restart_prob in 0.08 0.10 0.12 0.15 0.18; do
    for page_ppr_weight in 1.25 1.5 1.75; do
      label_restart="${restart_prob/./p}"
      label_page_weight="${page_ppr_weight/./p}"
      run_graph "${LABEL_PREFIX}_graph_ppr_eq_k10_top20_nodocseed_restart${label_restart}_pagew${label_page_weight}" \
        --dense-top-pages 1000 \
        --sparse-top-pages 1000 \
        --final-top-pages 20 \
        --per-doc-page-limit 1 \
        --doc-seed-weight 0.0 \
        --restart-prob "$restart_prob" \
        --final-ppr-page-weight "$page_ppr_weight"
    done
  done

  for restart_prob in 0.10 0.15; do
    for doc_ppr_weight in 0.25 0.5 0.75; do
      label_restart="${restart_prob/./p}"
      label_doc_weight="${doc_ppr_weight/./p}"
      run_graph "${LABEL_PREFIX}_graph_ppr_eq_k10_top20_nodocseed_restart${label_restart}_pagew1p5_docw${label_doc_weight}" \
        --dense-top-pages 1000 \
        --sparse-top-pages 1000 \
        --final-top-pages 20 \
        --per-doc-page-limit 1 \
        --doc-seed-weight 0.0 \
        --restart-prob "$restart_prob" \
        --final-ppr-page-weight 1.5 \
        --final-ppr-doc-weight "$doc_ppr_weight"
    done
  done
fi

for summary in "$OUTDIR"/"${LABEL_PREFIX}"_graph_ppr_*summary.json; do
  [[ -f "$summary" ]] || continue
  "$PYTHON_BIN" - "$summary" <<'PY'
import json
import sys

path = sys.argv[1]
summary = json.load(open(path))
print("=" * 100)
print(path)
for key in [
    "qid_count",
    "dense_top4_doc_count",
    "sparse_top4_doc_count",
    "reranked_top4_doc_count",
    "dense_top20_doc_count",
    "sparse_top20_doc_count",
    "reranked_top20_doc_count",
    "reranked_top4_page_count",
    "reranked_top20_page_count",
    "graph_recovers_top4_doc_vs_dense_count",
    "graph_loses_top4_doc_vs_dense_count",
    "mean_candidate_page_count",
    "mean_candidate_doc_count",
]:
    print(f"{key}: {summary.get(key)}")
PY
done
