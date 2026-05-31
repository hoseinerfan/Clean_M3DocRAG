#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/m3docvqa_internal_env.sh"

DEFAULT_GRAPH_OUT_DIR="$LOCAL_OUTPUT_DIR/m3docvqa_graph_pagepreserve_mmqa_${SPLIT}"
GRAPH_OUT_DIR="${GRAPH_OUT_DIR:-$DEFAULT_GRAPH_OUT_DIR}"

DENSE_PRED="${DENSE_PRED:-$LOCAL_OUTPUT_DIR/m3docvqa_plain_top224_mmqa_${SPLIT}/mmqa_${SPLIT}_plain_top224_nprobe${FAISS_NPROBE}_effdiag_all.prediction.json}"
SPARSE_PRED="${SPARSE_PRED:-$LOCAL_OUTPUT_DIR/m3docvqa_splade_mmqa_${SPLIT}/mmqa_${SPLIT}_splade.prediction.json}"
SPLADE_INDEX_PT="${SPLADE_INDEX_PT:-$LOCAL_OUTPUT_DIR/m3docvqa_splade/m3docvqa_${SPLIT}_splade.pt}"
DOC_PAGES_JSONL="${DOC_PAGES_JSONL:-${M3DOCVQA_PAGE_TEXT_JSONL:-$LOCAL_OUTPUT_DIR/m3docvqa_page_text/m3docvqa_${SPLIT}_page_text.jsonl}}"

GRAPH_PROFILE="${GRAPH_PROFILE:-denseheavy125_medium_both}"
GRAPH_LABEL="${GRAPH_LABEL:-mmqa_${SPLIT}_plain_top224_splade_graph_pagepreserve_${GRAPH_PROFILE}}"
PRED_OUT="${PRED_OUT:-$GRAPH_OUT_DIR/${GRAPH_LABEL}.prediction.json}"
SUMMARY_OUT="${SUMMARY_OUT:-$GRAPH_OUT_DIR/${GRAPH_LABEL}.summary.json}"
ANALYSIS_OUT="${ANALYSIS_OUT:-$GRAPH_OUT_DIR/${GRAPH_LABEL}.retrieval_analysis.json}"
VS_DENSE_OUT="${VS_DENSE_OUT:-$GRAPH_OUT_DIR/${GRAPH_LABEL}.vs_dense.json}"
VS_SPLADE_OUT="${VS_SPLADE_OUT:-$GRAPH_OUT_DIR/${GRAPH_LABEL}.vs_splade.json}"

RECALL_K_VALUES="${RECALL_K_VALUES:-1 2 4 5 10 20 50 100 500 1000}"
QUESTION_TYPE_FILTER="${QUESTION_TYPE_FILTER:-}"

mkdir -p "$GRAPH_OUT_DIR"

echo "using_dense_pred=$DENSE_PRED"
echo "using_sparse_pred=$SPARSE_PRED"
echo "using_gold=$GOLD"
echo "using_doc_pages_jsonl=$DOC_PAGES_JSONL"
echo "using_graph_out_dir=$GRAPH_OUT_DIR"
echo "using_graph_profile=$GRAPH_PROFILE"
if [[ -n "${OUT_DIR:-}" && "$GRAPH_OUT_DIR" == "$DEFAULT_GRAPH_OUT_DIR" ]]; then
  echo "ignoring_generic_out_dir=$OUT_DIR"
fi
echo "using_question_type_filter=${QUESTION_TYPE_FILTER:-ALL}"

"$PYTHON_BIN" - "$DENSE_PRED" "$SPARSE_PRED" "$GOLD" "$QUESTION_TYPE_FILTER" <<'PY'
import json
import sys
from pathlib import Path

dense_path = Path(sys.argv[1])
sparse_path = Path(sys.argv[2])
gold_path = Path(sys.argv[3])
question_type = str(sys.argv[4]).strip()

def load_prediction(path: Path) -> tuple[dict[str, dict], list[str], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "predictions" in payload and isinstance(
        payload["predictions"], (dict, list)
    ):
        payload = payload["predictions"]

    rows_by_qid = {}
    raw_keys = []
    row_qids = []
    if isinstance(payload, list):
        iterable = enumerate(payload)
    elif isinstance(payload, dict):
        iterable = payload.items()
    else:
        raise TypeError(f"Prediction JSON must be a list or object of prediction rows: {path}")

    for raw_key, row in iterable:
        if not isinstance(row, dict):
            raise TypeError(f"Prediction row must be an object: {path} key={raw_key!r}")
        raw_keys.append(str(raw_key))
        row_qid = str(row.get("qid", "")).strip()
        if row_qid:
            row_qids.append(row_qid)
        qid = row_qid or str(raw_key).strip()
        if not qid:
            raise ValueError(f"Prediction row is missing qid and key is empty: {path} key={raw_key!r}")
        if qid in rows_by_qid:
            raise ValueError(f"Duplicate qid after normalization: {qid} ({path})")
        rows_by_qid[qid] = row
    return rows_by_qid, raw_keys[:3], row_qids[:3]

dense, dense_raw_keys, dense_row_qids = load_prediction(dense_path)
sparse, sparse_raw_keys, sparse_row_qids = load_prediction(sparse_path)

gold_qids = set()
with gold_path.open("r", encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        if question_type:
            row_type = str(row.get("metadata", {}).get("type", "")).strip()
            if row_type != question_type:
                continue
        qid = str(row.get("qid", "")).strip()
        if qid:
            gold_qids.add(qid)

dense_qids = set(map(str, dense.keys()))
sparse_qids = set(map(str, sparse.keys()))
common_qids = dense_qids & sparse_qids
joint_gold_qids = common_qids & gold_qids

print(f"preflight_dense_qids={len(dense_qids)}")
print(f"preflight_sparse_qids={len(sparse_qids)}")
print(f"preflight_common_qids={len(common_qids)}")
print(f"preflight_gold_qids={len(gold_qids)}")
print(f"preflight_joint_gold_qids={len(joint_gold_qids)}")
print(f"preflight_question_type_filter={question_type or 'ALL'}")
print(f"preflight_dense_sample_raw_keys={dense_raw_keys}")
print(f"preflight_dense_sample_row_qids={dense_row_qids}")
print(f"preflight_sparse_sample_raw_keys={sparse_raw_keys}")
print(f"preflight_sparse_sample_row_qids={sparse_row_qids}")

if not common_qids:
    raise SystemExit("Dense and sparse predictions have no qids in common.")
if not joint_gold_qids:
    raise SystemExit(
        "No qids remain after intersecting dense, sparse, and gold qids. "
        "Check the gold path and question-type filter."
    )
PY

case "$GRAPH_PROFILE" in
  denseheavy125_medium_both)
    PROFILE_FINAL_TOP_PAGES=1000
    PROFILE_PER_DOC_PAGE_LIMIT=0
    PROFILE_DENSE_WEIGHT=1.25
    PROFILE_SPARSE_WEIGHT=0.75
    PROFILE_FINAL_PPR_PAGE_WEIGHT=0.5
    PROFILE_FINAL_PPR_DOC_WEIGHT=0.25
    PROFILE_NEIGHBOR_EXPANSION_WINDOW=0
    PROFILE_NEIGHBOR_SEED_WEIGHT=0.25
    PROFILE_EXPAND_NEIGHBORS_FROM_TOP_DENSE_PAGES=50
    PROFILE_EXPAND_NEIGHBORS_FROM_TOP_SPARSE_PAGES=50
    ;;
  denseheavy125_medium_both_neighbor1)
    PROFILE_FINAL_TOP_PAGES=1000
    PROFILE_PER_DOC_PAGE_LIMIT=0
    PROFILE_DENSE_WEIGHT=1.25
    PROFILE_SPARSE_WEIGHT=0.75
    PROFILE_FINAL_PPR_PAGE_WEIGHT=0.5
    PROFILE_FINAL_PPR_DOC_WEIGHT=0.25
    PROFILE_NEIGHBOR_EXPANSION_WINDOW=1
    PROFILE_NEIGHBOR_SEED_WEIGHT=0.25
    PROFILE_EXPAND_NEIGHBORS_FROM_TOP_DENSE_PAGES=50
    PROFILE_EXPAND_NEIGHBORS_FROM_TOP_SPARSE_PAGES=50
    ;;
  denseheavy_lightboth)
    PROFILE_FINAL_TOP_PAGES=1000
    PROFILE_PER_DOC_PAGE_LIMIT=0
    PROFILE_DENSE_WEIGHT=1.25
    PROFILE_SPARSE_WEIGHT=0.75
    PROFILE_FINAL_PPR_PAGE_WEIGHT=0.25
    PROFILE_FINAL_PPR_DOC_WEIGHT=0.25
    PROFILE_NEIGHBOR_EXPANSION_WINDOW=0
    PROFILE_NEIGHBOR_SEED_WEIGHT=0.25
    PROFILE_EXPAND_NEIGHBORS_FROM_TOP_DENSE_PAGES=50
    PROFILE_EXPAND_NEIGHBORS_FROM_TOP_SPARSE_PAGES=50
    ;;
  denseheavy150_m3best_pagepreserve)
    PROFILE_FINAL_TOP_PAGES=1000
    PROFILE_PER_DOC_PAGE_LIMIT=0
    PROFILE_DENSE_WEIGHT=1.5
    PROFILE_SPARSE_WEIGHT=0.5
    PROFILE_FINAL_PPR_PAGE_WEIGHT=1.5
    PROFILE_FINAL_PPR_DOC_WEIGHT=0.75
    PROFILE_NEIGHBOR_EXPANSION_WINDOW=0
    PROFILE_NEIGHBOR_SEED_WEIGHT=0.25
    PROFILE_EXPAND_NEIGHBORS_FROM_TOP_DENSE_PAGES=50
    PROFILE_EXPAND_NEIGHBORS_FROM_TOP_SPARSE_PAGES=50
    ;;
  doc_shortlist_best|graph1000)
    PROFILE_FINAL_TOP_PAGES=20
    PROFILE_PER_DOC_PAGE_LIMIT=1
    PROFILE_DENSE_WEIGHT=1.0
    PROFILE_SPARSE_WEIGHT=1.0
    PROFILE_FINAL_PPR_PAGE_WEIGHT=1.5
    PROFILE_FINAL_PPR_DOC_WEIGHT=0.75
    PROFILE_NEIGHBOR_EXPANSION_WINDOW=0
    PROFILE_NEIGHBOR_SEED_WEIGHT=0.25
    PROFILE_EXPAND_NEIGHBORS_FROM_TOP_DENSE_PAGES=50
    PROFILE_EXPAND_NEIGHBORS_FROM_TOP_SPARSE_PAGES=50
    ;;
  *)
    echo "Unsupported GRAPH_PROFILE=$GRAPH_PROFILE" >&2
    exit 1
    ;;
esac

DENSE_TOP_PAGES="${DENSE_TOP_PAGES:-1000}"
SPARSE_TOP_PAGES="${SPARSE_TOP_PAGES:-1000}"
FINAL_TOP_PAGES="${FINAL_TOP_PAGES:-$PROFILE_FINAL_TOP_PAGES}"
PER_DOC_PAGE_LIMIT="${PER_DOC_PAGE_LIMIT:-$PROFILE_PER_DOC_PAGE_LIMIT}"
RRF_K="${RRF_K:-10}"
DENSE_WEIGHT="${DENSE_WEIGHT:-$PROFILE_DENSE_WEIGHT}"
SPARSE_WEIGHT="${SPARSE_WEIGHT:-$PROFILE_SPARSE_WEIGHT}"
ADAPTIVE_SOURCE_WEIGHT_MODE="${ADAPTIVE_SOURCE_WEIGHT_MODE:-none}"
ADAPTIVE_SOURCE_AGREEMENT_TOP_PAGES="${ADAPTIVE_SOURCE_AGREEMENT_TOP_PAGES:-20}"
ADAPTIVE_SOURCE_TOP1_LOOKUP_PAGES="${ADAPTIVE_SOURCE_TOP1_LOOKUP_PAGES:-1000}"
ADAPTIVE_SOURCE_DOC_OVERLAP_WEIGHT="${ADAPTIVE_SOURCE_DOC_OVERLAP_WEIGHT:-0.7}"
ADAPTIVE_SOURCE_PAGE_OVERLAP_WEIGHT="${ADAPTIVE_SOURCE_PAGE_OVERLAP_WEIGHT:-0.2}"
ADAPTIVE_SOURCE_TOP1_WEIGHT="${ADAPTIVE_SOURCE_TOP1_WEIGHT:-0.1}"
ADAPTIVE_SOURCE_STRENGTH="${ADAPTIVE_SOURCE_STRENGTH:-0.5}"
ADAPTIVE_SOURCE_GAMMA="${ADAPTIVE_SOURCE_GAMMA:-1.0}"
ADAPTIVE_SOURCE_MIN_DENSE_MULT="${ADAPTIVE_SOURCE_MIN_DENSE_MULT:-1.0}"
ADAPTIVE_SOURCE_MAX_DENSE_MULT="${ADAPTIVE_SOURCE_MAX_DENSE_MULT:-1.5}"
ADAPTIVE_SOURCE_MIN_SPARSE_MULT="${ADAPTIVE_SOURCE_MIN_SPARSE_MULT:-0.5}"
ADAPTIVE_SOURCE_MAX_SPARSE_MULT="${ADAPTIVE_SOURCE_MAX_SPARSE_MULT:-1.0}"
ADAPTIVE_SOURCE_RECIPROCAL_TOP_PAGES="${ADAPTIVE_SOURCE_RECIPROCAL_TOP_PAGES:-20}"
ADAPTIVE_SOURCE_RECIPROCAL_LOOKUP_PAGES="${ADAPTIVE_SOURCE_RECIPROCAL_LOOKUP_PAGES:-1000}"
ADAPTIVE_SOURCE_RECIPROCAL_DOC_WEIGHT="${ADAPTIVE_SOURCE_RECIPROCAL_DOC_WEIGHT:-0.7}"
ADAPTIVE_SOURCE_RECIPROCAL_PAGE_WEIGHT="${ADAPTIVE_SOURCE_RECIPROCAL_PAGE_WEIGHT:-0.3}"
ADAPTIVE_SOURCE_RECIPROCAL_MIN_MULT="${ADAPTIVE_SOURCE_RECIPROCAL_MIN_MULT:-0.75}"
ADAPTIVE_SOURCE_RECIPROCAL_MAX_MULT="${ADAPTIVE_SOURCE_RECIPROCAL_MAX_MULT:-1.25}"
ADAPTIVE_SOURCE_RECIPROCAL_PRESERVE_TOTAL="${ADAPTIVE_SOURCE_RECIPROCAL_PRESERVE_TOTAL:-1}"
ADAPTIVE_RESTART_MODE="${ADAPTIVE_RESTART_MODE:-none}"
ADAPTIVE_RESTART_SOURCE_STRENGTH="${ADAPTIVE_RESTART_SOURCE_STRENGTH:-0.5}"
ADAPTIVE_RESTART_GAMMA="${ADAPTIVE_RESTART_GAMMA:-1.0}"
ADAPTIVE_RESTART_MIN_DENSE_MULT="${ADAPTIVE_RESTART_MIN_DENSE_MULT:-1.0}"
ADAPTIVE_RESTART_MAX_DENSE_MULT="${ADAPTIVE_RESTART_MAX_DENSE_MULT:-1.5}"
ADAPTIVE_RESTART_MIN_SPARSE_MULT="${ADAPTIVE_RESTART_MIN_SPARSE_MULT:-0.5}"
ADAPTIVE_RESTART_MAX_SPARSE_MULT="${ADAPTIVE_RESTART_MAX_SPARSE_MULT:-1.0}"
ADAPTIVE_RESTART_PAGE_SEED_WEIGHT="${ADAPTIVE_RESTART_PAGE_SEED_WEIGHT:-1.0}"
ADAPTIVE_RESTART_DOC_SEED_WEIGHT="${ADAPTIVE_RESTART_DOC_SEED_WEIGHT:-0.25}"
ADAPTIVE_RESTART_MIN_DOC_MULT="${ADAPTIVE_RESTART_MIN_DOC_MULT:-0.0}"
ADAPTIVE_RESTART_MAX_DOC_MULT="${ADAPTIVE_RESTART_MAX_DOC_MULT:-1.0}"
ADAPTIVE_RESTART_NEIGHBOR_SEED_WEIGHT="${ADAPTIVE_RESTART_NEIGHBOR_SEED_WEIGHT:-1.0}"
ADAPTIVE_RESTART_PRESERVE_SOURCE_TOTAL="${ADAPTIVE_RESTART_PRESERVE_SOURCE_TOTAL:-1}"
ADAPTIVE_TRANSITION_MODE="${ADAPTIVE_TRANSITION_MODE:-none}"
ADAPTIVE_TRANSITION_AGREEMENT_MIN_MULT="${ADAPTIVE_TRANSITION_AGREEMENT_MIN_MULT:-0.75}"
ADAPTIVE_TRANSITION_AGREEMENT_MAX_MULT="${ADAPTIVE_TRANSITION_AGREEMENT_MAX_MULT:-1.0}"
ADAPTIVE_TRANSITION_LOCAL_MIN_MULT="${ADAPTIVE_TRANSITION_LOCAL_MIN_MULT:-0.75}"
ADAPTIVE_TRANSITION_LOCAL_MAX_MULT="${ADAPTIVE_TRANSITION_LOCAL_MAX_MULT:-1.0}"
ADAPTIVE_TRANSITION_LOCAL_WEIGHT="${ADAPTIVE_TRANSITION_LOCAL_WEIGHT:-0.5}"
ADAPTIVE_TRANSITION_GATE_ADJACENT="${ADAPTIVE_TRANSITION_GATE_ADJACENT:-1}"
ADAPTIVE_TRANSITION_GATE_PAGE_DOC="${ADAPTIVE_TRANSITION_GATE_PAGE_DOC:-1}"
ADAPTIVE_TRANSITION_GATE_PAGE_TO_DOC="${ADAPTIVE_TRANSITION_GATE_PAGE_TO_DOC:-1}"
ADAPTIVE_TRANSITION_GATE_DOC_TO_PAGE="${ADAPTIVE_TRANSITION_GATE_DOC_TO_PAGE:-1}"
ADAPTIVE_ADJACENT_MODE="${ADAPTIVE_ADJACENT_MODE:-none}"
ADAPTIVE_ADJACENT_MIN_MULT="${ADAPTIVE_ADJACENT_MIN_MULT:-0.0}"
ADAPTIVE_ADJACENT_MAX_MULT="${ADAPTIVE_ADJACENT_MAX_MULT:-1.0}"
ADAPTIVE_ADJACENT_POWER="${ADAPTIVE_ADJACENT_POWER:-0.5}"
DOC_SEED_WEIGHT="${DOC_SEED_WEIGHT:-0.0}"
DOC_SEED_MODE="${DOC_SEED_MODE:-rrf}"
DOC_SEED_PAGE_SCORE_MODE="${DOC_SEED_PAGE_SCORE_MODE:-mean}"
DOC_SEED_PAGE_TOP_K="${DOC_SEED_PAGE_TOP_K:-3}"
DOC_SEED_GRAPH_SIZE_REFERENCE="${DOC_SEED_GRAPH_SIZE_REFERENCE:-20.0}"
DOC_SEED_GRAPH_SIZE_MIN_MULT="${DOC_SEED_GRAPH_SIZE_MIN_MULT:-0.25}"
DOC_SEED_GRAPH_SIZE_MAX_MULT="${DOC_SEED_GRAPH_SIZE_MAX_MULT:-2.0}"
LEARNED_PAGE_PRIOR_JSONL="${LEARNED_PAGE_PRIOR_JSONL:-}"
LEARNED_PAGE_PRIOR_SCORE_FIELD="${LEARNED_PAGE_PRIOR_SCORE_FIELD:-learned_score}"
LEARNED_PAGE_PRIOR_SEED_WEIGHT="${LEARNED_PAGE_PRIOR_SEED_WEIGHT:-0.0}"
LEARNED_PAGE_PRIOR_MIN_BASE_RANK="${LEARNED_PAGE_PRIOR_MIN_BASE_RANK:-1}"
LEARNED_PAGE_PRIOR_MAX_BASE_RANK="${LEARNED_PAGE_PRIOR_MAX_BASE_RANK:-1000}"
LEARNED_PAGE_PRIOR_TOP_K="${LEARNED_PAGE_PRIOR_TOP_K:-0}"
LEARNED_PAGE_PRIOR_NORMALIZE="${LEARNED_PAGE_PRIOR_NORMALIZE:-1}"
PDF_HYPERLINK_EDGES_JSONL="${PDF_HYPERLINK_EDGES_JSONL:-}"
PDF_HYPERLINK_EDGE_WEIGHT="${PDF_HYPERLINK_EDGE_WEIGHT:-0.0}"
PDF_HYPERLINK_DIRECTION="${PDF_HYPERLINK_DIRECTION:-source_to_target_doc}"
PDF_HYPERLINK_TARGET_MODE="${PDF_HYPERLINK_TARGET_MODE:-target_doc}"
PDF_HYPERLINK_TARGET_PAGES_PER_DOC="${PDF_HYPERLINK_TARGET_PAGES_PER_DOC:-1}"
PDF_HYPERLINK_TARGET_PAGE_WEIGHT_MODE="${PDF_HYPERLINK_TARGET_PAGE_WEIGHT_MODE:-split}"
PDF_HYPERLINK_WEIGHT_MODE="${PDF_HYPERLINK_WEIGHT_MODE:-uniform}"
PDF_HYPERLINK_MAX_EDGES_PER_SOURCE="${PDF_HYPERLINK_MAX_EDGES_PER_SOURCE:-0}"
PDF_HYPERLINK_SOURCE_TOP_K="${PDF_HYPERLINK_SOURCE_TOP_K:-0}"
PDF_HYPERLINK_TARGET_DOC_TOP_K="${PDF_HYPERLINK_TARGET_DOC_TOP_K:-0}"
PDF_HYPERLINK_QUERY_SUPPORT_WEIGHT_MODE="${PDF_HYPERLINK_QUERY_SUPPORT_WEIGHT_MODE:-none}"
EXTERNAL_PAGE_GRAPH_JSONL="${EXTERNAL_PAGE_GRAPH_JSONL:-}"
EXTERNAL_PAGE_GRAPH_EDGE_WEIGHT="${EXTERNAL_PAGE_GRAPH_EDGE_WEIGHT:-0.0}"
EXTERNAL_PAGE_GRAPH_DIRECTION="${EXTERNAL_PAGE_GRAPH_DIRECTION:-as_directed}"
EXTERNAL_PAGE_GRAPH_WEIGHT_MODE="${EXTERNAL_PAGE_GRAPH_WEIGHT_MODE:-score}"
EXTERNAL_PAGE_GRAPH_MAX_EDGES_PER_SOURCE="${EXTERNAL_PAGE_GRAPH_MAX_EDGES_PER_SOURCE:-0}"
EXTERNAL_PAGE_GRAPH_SOURCE_TOP_K="${EXTERNAL_PAGE_GRAPH_SOURCE_TOP_K:-0}"
EXTERNAL_PAGE_GRAPH_TARGET_TOP_K="${EXTERNAL_PAGE_GRAPH_TARGET_TOP_K:-0}"
DOC_DOC_EDGE_MODE="${DOC_DOC_EDGE_MODE:-none}"
DOC_DOC_EDGE_WEIGHT="${DOC_DOC_EDGE_WEIGHT:-0.0}"
DOC_DOC_TOP_DOCS="${DOC_DOC_TOP_DOCS:-20}"
DOC_DOC_MAX_EDGES_PER_DOC="${DOC_DOC_MAX_EDGES_PER_DOC:-8}"
DOC_DOC_MIN_SHARED_SIGNALS="${DOC_DOC_MIN_SHARED_SIGNALS:-1}"
DOC_DOC_MAX_SIGNAL_DOC_MATCHES="${DOC_DOC_MAX_SIGNAL_DOC_MATCHES:-8}"
DOC_DOC_MIN_SEMANTIC_SIMILARITY="${DOC_DOC_MIN_SEMANTIC_SIMILARITY:-0.35}"
DOC_DOC_SEMANTIC_TOP_TERMS="${DOC_DOC_SEMANTIC_TOP_TERMS:-64}"
DOC_DOC_PAGE_EMBEDDING_DIR="${DOC_DOC_PAGE_EMBEDDING_DIR:-}"
DOC_DOC_EMBEDDING_POOLING="${DOC_DOC_EMBEDDING_POOLING:-page_seed_weighted_mean}"
DOC_DOC_EMBEDDING_PAGE_TOP_K="${DOC_DOC_EMBEDDING_PAGE_TOP_K:-0}"
DOC_DOC_EMBEDDING_MIN_SIMILARITY="${DOC_DOC_EMBEDDING_MIN_SIMILARITY:-0.0}"
DOC_DOC_EMBEDDING_CACHE_DOCS="${DOC_DOC_EMBEDDING_CACHE_DOCS:-128}"
DOC_DOC_HYPERLINK_WEIGHT_MODE="${DOC_DOC_HYPERLINK_WEIGHT_MODE:-log_count}"
DOC_DOC_HYPERLINK_INIT_MODE="${DOC_DOC_HYPERLINK_INIT_MODE:-log_count}"
DOC_DOC_HYPERLINK_SOURCE_SEED_FLOOR="${DOC_DOC_HYPERLINK_SOURCE_SEED_FLOOR:-0.5}"
DOC_DOC_HYPERLINK_SOURCE_SEED_SCALE="${DOC_DOC_HYPERLINK_SOURCE_SEED_SCALE:-0.5}"
DOC_DOC_HYPERLINK_TARGET_SUPPORT_FLOOR="${DOC_DOC_HYPERLINK_TARGET_SUPPORT_FLOOR:-0.5}"
DOC_DOC_HYPERLINK_TARGET_SUPPORT_SCALE="${DOC_DOC_HYPERLINK_TARGET_SUPPORT_SCALE:-0.75}"
RESTART_PROB="${RESTART_PROB:-0.15}"
PPR_ITERS="${PPR_ITERS:-30}"
PPR_ITERATION_MODE="${PPR_ITERATION_MODE:-fixed}"
PPR_GRAPH_SIZE_METRIC="${PPR_GRAPH_SIZE_METRIC:-nodes}"
PPR_GRAPH_SIZE_REFERENCE="${PPR_GRAPH_SIZE_REFERENCE:-1000.0}"
PPR_GRAPH_SIZE_MIN_ITERS="${PPR_GRAPH_SIZE_MIN_ITERS:-5}"
PPR_GRAPH_SIZE_MAX_ITERS="${PPR_GRAPH_SIZE_MAX_ITERS:-80}"
PPR_CONVERGENCE_TOL="${PPR_CONVERGENCE_TOL:-1e-7}"
PPR_CONVERGENCE_MIN_ITERS="${PPR_CONVERGENCE_MIN_ITERS:-5}"
PAGE_DOC_EDGE_WEIGHT="${PAGE_DOC_EDGE_WEIGHT:-1.0}"
PAGE_TO_DOC_EDGE_WEIGHT="${PAGE_TO_DOC_EDGE_WEIGHT:-$PAGE_DOC_EDGE_WEIGHT}"
DOC_TO_PAGE_EDGE_WEIGHT="${DOC_TO_PAGE_EDGE_WEIGHT:-$PAGE_DOC_EDGE_WEIGHT}"
SAME_DOC_WINDOW="${SAME_DOC_WINDOW:-1}"
ADJACENT_PAGE_EDGE_WEIGHT="${ADJACENT_PAGE_EDGE_WEIGHT:-0.25}"
FINAL_PAGE_SEED_WEIGHT="${FINAL_PAGE_SEED_WEIGHT:-1.0}"
FINAL_PPR_PAGE_WEIGHT="${FINAL_PPR_PAGE_WEIGHT:-$PROFILE_FINAL_PPR_PAGE_WEIGHT}"
FINAL_PPR_DOC_WEIGHT="${FINAL_PPR_DOC_WEIGHT:-$PROFILE_FINAL_PPR_DOC_WEIGHT}"
FINAL_SELECTION_MODE="${FINAL_SELECTION_MODE:-score}"
FINAL_SELECTION_TOP_K="${FINAL_SELECTION_TOP_K:-4}"
FINAL_SELECTION_CANDIDATE_POOL="${FINAL_SELECTION_CANDIDATE_POOL:-20}"
FINAL_SELECTION_NEW_DOC_BONUS="${FINAL_SELECTION_NEW_DOC_BONUS:-0.05}"
FINAL_SELECTION_SAME_DOC_PENALTY="${FINAL_SELECTION_SAME_DOC_PENALTY:-0.05}"
NEIGHBOR_EXPANSION_WINDOW="${NEIGHBOR_EXPANSION_WINDOW:-$PROFILE_NEIGHBOR_EXPANSION_WINDOW}"
NEIGHBOR_SEED_WEIGHT="${NEIGHBOR_SEED_WEIGHT:-$PROFILE_NEIGHBOR_SEED_WEIGHT}"
EXPAND_NEIGHBORS_FROM_TOP_DENSE_PAGES="${EXPAND_NEIGHBORS_FROM_TOP_DENSE_PAGES:-$PROFILE_EXPAND_NEIGHBORS_FROM_TOP_DENSE_PAGES}"
EXPAND_NEIGHBORS_FROM_TOP_SPARSE_PAGES="${EXPAND_NEIGHBORS_FROM_TOP_SPARSE_PAGES:-$PROFILE_EXPAND_NEIGHBORS_FROM_TOP_SPARSE_PAGES}"

GRAPH_ARGS=(
  --dense-prediction-json "$DENSE_PRED"
  --sparse-prediction-json "$SPARSE_PRED"
  --gold "$GOLD"
  --dense-top-pages "$DENSE_TOP_PAGES"
  --sparse-top-pages "$SPARSE_TOP_PAGES"
  --final-top-pages "$FINAL_TOP_PAGES"
  --per-doc-page-limit "$PER_DOC_PAGE_LIMIT"
  --rrf-k "$RRF_K"
  --dense-weight "$DENSE_WEIGHT"
  --sparse-weight "$SPARSE_WEIGHT"
  --adaptive-source-weight-mode "$ADAPTIVE_SOURCE_WEIGHT_MODE"
  --adaptive-source-agreement-top-pages "$ADAPTIVE_SOURCE_AGREEMENT_TOP_PAGES"
  --adaptive-source-top1-lookup-pages "$ADAPTIVE_SOURCE_TOP1_LOOKUP_PAGES"
  --adaptive-source-doc-overlap-weight "$ADAPTIVE_SOURCE_DOC_OVERLAP_WEIGHT"
  --adaptive-source-page-overlap-weight "$ADAPTIVE_SOURCE_PAGE_OVERLAP_WEIGHT"
  --adaptive-source-top1-weight "$ADAPTIVE_SOURCE_TOP1_WEIGHT"
  --adaptive-source-strength "$ADAPTIVE_SOURCE_STRENGTH"
  --adaptive-source-gamma "$ADAPTIVE_SOURCE_GAMMA"
  --adaptive-source-min-dense-mult "$ADAPTIVE_SOURCE_MIN_DENSE_MULT"
  --adaptive-source-max-dense-mult "$ADAPTIVE_SOURCE_MAX_DENSE_MULT"
  --adaptive-source-min-sparse-mult "$ADAPTIVE_SOURCE_MIN_SPARSE_MULT"
  --adaptive-source-max-sparse-mult "$ADAPTIVE_SOURCE_MAX_SPARSE_MULT"
  --adaptive-source-reciprocal-top-pages "$ADAPTIVE_SOURCE_RECIPROCAL_TOP_PAGES"
  --adaptive-source-reciprocal-lookup-pages "$ADAPTIVE_SOURCE_RECIPROCAL_LOOKUP_PAGES"
  --adaptive-source-reciprocal-doc-weight "$ADAPTIVE_SOURCE_RECIPROCAL_DOC_WEIGHT"
  --adaptive-source-reciprocal-page-weight "$ADAPTIVE_SOURCE_RECIPROCAL_PAGE_WEIGHT"
  --adaptive-source-reciprocal-min-mult "$ADAPTIVE_SOURCE_RECIPROCAL_MIN_MULT"
  --adaptive-source-reciprocal-max-mult "$ADAPTIVE_SOURCE_RECIPROCAL_MAX_MULT"
  --adaptive-restart-mode "$ADAPTIVE_RESTART_MODE"
  --adaptive-restart-source-strength "$ADAPTIVE_RESTART_SOURCE_STRENGTH"
  --adaptive-restart-gamma "$ADAPTIVE_RESTART_GAMMA"
  --adaptive-restart-min-dense-mult "$ADAPTIVE_RESTART_MIN_DENSE_MULT"
  --adaptive-restart-max-dense-mult "$ADAPTIVE_RESTART_MAX_DENSE_MULT"
  --adaptive-restart-min-sparse-mult "$ADAPTIVE_RESTART_MIN_SPARSE_MULT"
  --adaptive-restart-max-sparse-mult "$ADAPTIVE_RESTART_MAX_SPARSE_MULT"
  --adaptive-restart-page-seed-weight "$ADAPTIVE_RESTART_PAGE_SEED_WEIGHT"
  --adaptive-restart-doc-seed-weight "$ADAPTIVE_RESTART_DOC_SEED_WEIGHT"
  --adaptive-restart-min-doc-mult "$ADAPTIVE_RESTART_MIN_DOC_MULT"
  --adaptive-restart-max-doc-mult "$ADAPTIVE_RESTART_MAX_DOC_MULT"
  --adaptive-restart-neighbor-seed-weight "$ADAPTIVE_RESTART_NEIGHBOR_SEED_WEIGHT"
  --adaptive-transition-mode "$ADAPTIVE_TRANSITION_MODE"
  --adaptive-transition-agreement-min-mult "$ADAPTIVE_TRANSITION_AGREEMENT_MIN_MULT"
  --adaptive-transition-agreement-max-mult "$ADAPTIVE_TRANSITION_AGREEMENT_MAX_MULT"
  --adaptive-transition-local-min-mult "$ADAPTIVE_TRANSITION_LOCAL_MIN_MULT"
  --adaptive-transition-local-max-mult "$ADAPTIVE_TRANSITION_LOCAL_MAX_MULT"
  --adaptive-transition-local-weight "$ADAPTIVE_TRANSITION_LOCAL_WEIGHT"
  --adaptive-adjacent-mode "$ADAPTIVE_ADJACENT_MODE"
  --adaptive-adjacent-min-mult "$ADAPTIVE_ADJACENT_MIN_MULT"
  --adaptive-adjacent-max-mult "$ADAPTIVE_ADJACENT_MAX_MULT"
  --adaptive-adjacent-power "$ADAPTIVE_ADJACENT_POWER"
  --doc-seed-weight "$DOC_SEED_WEIGHT"
  --doc-seed-mode "$DOC_SEED_MODE"
  --doc-seed-page-score-mode "$DOC_SEED_PAGE_SCORE_MODE"
  --doc-seed-page-top-k "$DOC_SEED_PAGE_TOP_K"
  --doc-seed-graph-size-reference "$DOC_SEED_GRAPH_SIZE_REFERENCE"
  --doc-seed-graph-size-min-mult "$DOC_SEED_GRAPH_SIZE_MIN_MULT"
  --doc-seed-graph-size-max-mult "$DOC_SEED_GRAPH_SIZE_MAX_MULT"
  --learned-page-prior-score-field "$LEARNED_PAGE_PRIOR_SCORE_FIELD"
  --learned-page-prior-seed-weight "$LEARNED_PAGE_PRIOR_SEED_WEIGHT"
  --learned-page-prior-min-base-rank "$LEARNED_PAGE_PRIOR_MIN_BASE_RANK"
  --learned-page-prior-max-base-rank "$LEARNED_PAGE_PRIOR_MAX_BASE_RANK"
  --learned-page-prior-top-k "$LEARNED_PAGE_PRIOR_TOP_K"
  --pdf-hyperlink-edge-weight "$PDF_HYPERLINK_EDGE_WEIGHT"
  --pdf-hyperlink-direction "$PDF_HYPERLINK_DIRECTION"
  --pdf-hyperlink-target-mode "$PDF_HYPERLINK_TARGET_MODE"
  --pdf-hyperlink-target-pages-per-doc "$PDF_HYPERLINK_TARGET_PAGES_PER_DOC"
  --pdf-hyperlink-target-page-weight-mode "$PDF_HYPERLINK_TARGET_PAGE_WEIGHT_MODE"
  --pdf-hyperlink-weight-mode "$PDF_HYPERLINK_WEIGHT_MODE"
  --pdf-hyperlink-max-edges-per-source "$PDF_HYPERLINK_MAX_EDGES_PER_SOURCE"
  --pdf-hyperlink-source-top-k "$PDF_HYPERLINK_SOURCE_TOP_K"
  --pdf-hyperlink-target-doc-top-k "$PDF_HYPERLINK_TARGET_DOC_TOP_K"
  --pdf-hyperlink-query-support-weight-mode "$PDF_HYPERLINK_QUERY_SUPPORT_WEIGHT_MODE"
  --external-page-graph-edge-weight "$EXTERNAL_PAGE_GRAPH_EDGE_WEIGHT"
  --external-page-graph-direction "$EXTERNAL_PAGE_GRAPH_DIRECTION"
  --external-page-graph-weight-mode "$EXTERNAL_PAGE_GRAPH_WEIGHT_MODE"
  --external-page-graph-max-edges-per-source "$EXTERNAL_PAGE_GRAPH_MAX_EDGES_PER_SOURCE"
  --external-page-graph-source-top-k "$EXTERNAL_PAGE_GRAPH_SOURCE_TOP_K"
  --external-page-graph-target-top-k "$EXTERNAL_PAGE_GRAPH_TARGET_TOP_K"
  --doc-doc-edge-mode "$DOC_DOC_EDGE_MODE"
  --doc-doc-edge-weight "$DOC_DOC_EDGE_WEIGHT"
  --doc-doc-top-docs "$DOC_DOC_TOP_DOCS"
  --doc-doc-max-edges-per-doc "$DOC_DOC_MAX_EDGES_PER_DOC"
  --doc-doc-min-shared-signals "$DOC_DOC_MIN_SHARED_SIGNALS"
  --doc-doc-max-signal-doc-matches "$DOC_DOC_MAX_SIGNAL_DOC_MATCHES"
  --doc-doc-min-semantic-similarity "$DOC_DOC_MIN_SEMANTIC_SIMILARITY"
  --doc-doc-semantic-top-terms "$DOC_DOC_SEMANTIC_TOP_TERMS"
  --doc-doc-page-embedding-dir "$DOC_DOC_PAGE_EMBEDDING_DIR"
  --doc-doc-embedding-pooling "$DOC_DOC_EMBEDDING_POOLING"
  --doc-doc-embedding-page-top-k "$DOC_DOC_EMBEDDING_PAGE_TOP_K"
  --doc-doc-embedding-min-similarity "$DOC_DOC_EMBEDDING_MIN_SIMILARITY"
  --doc-doc-embedding-cache-docs "$DOC_DOC_EMBEDDING_CACHE_DOCS"
  --doc-doc-hyperlink-weight-mode "$DOC_DOC_HYPERLINK_WEIGHT_MODE"
  --doc-doc-hyperlink-init-mode "$DOC_DOC_HYPERLINK_INIT_MODE"
  --doc-doc-hyperlink-source-seed-floor "$DOC_DOC_HYPERLINK_SOURCE_SEED_FLOOR"
  --doc-doc-hyperlink-source-seed-scale "$DOC_DOC_HYPERLINK_SOURCE_SEED_SCALE"
  --doc-doc-hyperlink-target-support-floor "$DOC_DOC_HYPERLINK_TARGET_SUPPORT_FLOOR"
  --doc-doc-hyperlink-target-support-scale "$DOC_DOC_HYPERLINK_TARGET_SUPPORT_SCALE"
  --restart-prob "$RESTART_PROB"
  --ppr-iters "$PPR_ITERS"
  --ppr-iteration-mode "$PPR_ITERATION_MODE"
  --ppr-graph-size-metric "$PPR_GRAPH_SIZE_METRIC"
  --ppr-graph-size-reference "$PPR_GRAPH_SIZE_REFERENCE"
  --ppr-graph-size-min-iters "$PPR_GRAPH_SIZE_MIN_ITERS"
  --ppr-graph-size-max-iters "$PPR_GRAPH_SIZE_MAX_ITERS"
  --ppr-convergence-tol "$PPR_CONVERGENCE_TOL"
  --ppr-convergence-min-iters "$PPR_CONVERGENCE_MIN_ITERS"
  --page-doc-edge-weight "$PAGE_DOC_EDGE_WEIGHT"
  --page-to-doc-edge-weight "$PAGE_TO_DOC_EDGE_WEIGHT"
  --doc-to-page-edge-weight "$DOC_TO_PAGE_EDGE_WEIGHT"
  --same-doc-window "$SAME_DOC_WINDOW"
  --adjacent-page-edge-weight "$ADJACENT_PAGE_EDGE_WEIGHT"
  --final-page-seed-weight "$FINAL_PAGE_SEED_WEIGHT"
  --final-ppr-page-weight "$FINAL_PPR_PAGE_WEIGHT"
  --final-ppr-doc-weight "$FINAL_PPR_DOC_WEIGHT"
  --final-selection-mode "$FINAL_SELECTION_MODE"
  --final-selection-top-k "$FINAL_SELECTION_TOP_K"
  --final-selection-candidate-pool "$FINAL_SELECTION_CANDIDATE_POOL"
  --final-selection-new-doc-bonus "$FINAL_SELECTION_NEW_DOC_BONUS"
  --final-selection-same-doc-penalty "$FINAL_SELECTION_SAME_DOC_PENALTY"
  --output-prediction-json "$PRED_OUT"
  --output-summary-json "$SUMMARY_OUT"
)
if [[ "$ADAPTIVE_SOURCE_RECIPROCAL_PRESERVE_TOTAL" == "0" ]]; then
  GRAPH_ARGS+=(--no-adaptive-source-reciprocal-preserve-total)
else
  GRAPH_ARGS+=(--adaptive-source-reciprocal-preserve-total)
fi
if [[ "$ADAPTIVE_RESTART_PRESERVE_SOURCE_TOTAL" == "0" ]]; then
  GRAPH_ARGS+=(--no-adaptive-restart-preserve-source-total)
else
  GRAPH_ARGS+=(--adaptive-restart-preserve-source-total)
fi
if [[ "$ADAPTIVE_TRANSITION_GATE_ADJACENT" == "0" ]]; then
  GRAPH_ARGS+=(--no-adaptive-transition-gate-adjacent)
else
  GRAPH_ARGS+=(--adaptive-transition-gate-adjacent)
fi
if [[ "$ADAPTIVE_TRANSITION_GATE_PAGE_DOC" == "0" ]]; then
  GRAPH_ARGS+=(--no-adaptive-transition-gate-page-doc)
else
  GRAPH_ARGS+=(--adaptive-transition-gate-page-doc)
fi
if [[ "$ADAPTIVE_TRANSITION_GATE_PAGE_TO_DOC" == "0" ]]; then
  GRAPH_ARGS+=(--no-adaptive-transition-gate-page-to-doc)
else
  GRAPH_ARGS+=(--adaptive-transition-gate-page-to-doc)
fi
if [[ "$ADAPTIVE_TRANSITION_GATE_DOC_TO_PAGE" == "0" ]]; then
  GRAPH_ARGS+=(--no-adaptive-transition-gate-doc-to-page)
else
  GRAPH_ARGS+=(--adaptive-transition-gate-doc-to-page)
fi
if [[ -n "$DOC_PAGES_JSONL" && -f "$DOC_PAGES_JSONL" ]]; then
  GRAPH_ARGS+=(--doc-pages-jsonl "$DOC_PAGES_JSONL")
fi
if [[ "$LEARNED_PAGE_PRIOR_NORMALIZE" == "0" ]]; then
  GRAPH_ARGS+=(--no-learned-page-prior-normalize)
else
  GRAPH_ARGS+=(--learned-page-prior-normalize)
fi
if [[ -n "$LEARNED_PAGE_PRIOR_JSONL" && -f "$LEARNED_PAGE_PRIOR_JSONL" ]]; then
  GRAPH_ARGS+=(--learned-page-prior-jsonl "$LEARNED_PAGE_PRIOR_JSONL")
fi
if [[ -n "$PDF_HYPERLINK_EDGES_JSONL" && -f "$PDF_HYPERLINK_EDGES_JSONL" ]]; then
  GRAPH_ARGS+=(--pdf-hyperlink-edges-jsonl "$PDF_HYPERLINK_EDGES_JSONL")
fi
if [[ -n "$EXTERNAL_PAGE_GRAPH_JSONL" && -f "$EXTERNAL_PAGE_GRAPH_JSONL" ]]; then
  GRAPH_ARGS+=(--external-page-graph-jsonl "$EXTERNAL_PAGE_GRAPH_JSONL")
fi
if [[ -n "$SPLADE_INDEX_PT" && -f "$SPLADE_INDEX_PT" ]]; then
  GRAPH_ARGS+=(--splade-index-pt "$SPLADE_INDEX_PT")
fi
if [[ "$NEIGHBOR_EXPANSION_WINDOW" -gt 0 ]]; then
  GRAPH_ARGS+=(
    --neighbor-expansion-window "$NEIGHBOR_EXPANSION_WINDOW"
    --neighbor-seed-weight "$NEIGHBOR_SEED_WEIGHT"
    --expand-neighbors-from-top-dense-pages "$EXPAND_NEIGHBORS_FROM_TOP_DENSE_PAGES"
    --expand-neighbors-from-top-sparse-pages "$EXPAND_NEIGHBORS_FROM_TOP_SPARSE_PAGES"
  )
fi
if [[ -n "$QUESTION_TYPE_FILTER" ]]; then
  GRAPH_ARGS+=(--question-type "$QUESTION_TYPE_FILTER")
fi
"$PYTHON_BIN" "$REPO_ROOT/scripts/graph_rerank_page_retrieval_predictions.py" "${GRAPH_ARGS[@]}"

"$PYTHON_BIN" "$REPO_ROOT/scripts/analyze_m3docvqa_retrieval.py" \
  --pred "$PRED_OUT" \
  --gold "$GOLD" \
  --summary-only \
  --recall-k $RECALL_K_VALUES \
  --json > "$ANALYSIS_OUT"

"$PYTHON_BIN" "$REPO_ROOT/scripts/compare_m3docvqa_retrieval_runs.py" \
  --baseline "$DENSE_PRED" \
  --candidate "$PRED_OUT" \
  --gold "$GOLD" \
  --recall-k $RECALL_K_VALUES \
  --json > "$VS_DENSE_OUT"

"$PYTHON_BIN" "$REPO_ROOT/scripts/compare_m3docvqa_retrieval_runs.py" \
  --baseline "$SPARSE_PRED" \
  --candidate "$PRED_OUT" \
  --gold "$GOLD" \
  --recall-k $RECALL_K_VALUES \
  --json > "$VS_SPLADE_OUT"

echo "saved_prediction=$PRED_OUT"
echo "saved_summary=$SUMMARY_OUT"
echo "saved_retrieval_analysis=$ANALYSIS_OUT"
echo "saved_vs_dense=$VS_DENSE_OUT"
echo "saved_vs_splade=$VS_SPLADE_OUT"
