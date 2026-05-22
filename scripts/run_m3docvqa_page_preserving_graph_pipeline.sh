#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/m3docvqa_internal_env.sh"

DEFAULT_GRAPH_OUT_DIR="$LOCAL_OUTPUT_DIR/m3docvqa_graph_pagepreserve_mmqa_${SPLIT}"
GRAPH_OUT_DIR="${GRAPH_OUT_DIR:-$DEFAULT_GRAPH_OUT_DIR}"

DENSE_PRED="${DENSE_PRED:-$LOCAL_OUTPUT_DIR/m3docvqa_plain_top224_mmqa_${SPLIT}/mmqa_${SPLIT}_plain_top224_nprobe${FAISS_NPROBE}_effdiag_all.prediction.json}"
SPARSE_PRED="${SPARSE_PRED:-$LOCAL_OUTPUT_DIR/m3docvqa_splade_mmqa_${SPLIT}/mmqa_${SPLIT}_splade.prediction.json}"

GRAPH_PROFILE="${GRAPH_PROFILE:-denseheavy125_medium_both}"
GRAPH_LABEL="${GRAPH_LABEL:-mmqa_${SPLIT}_plain_top224_splade_graph_pagepreserve_${GRAPH_PROFILE}}"
PRED_OUT="${PRED_OUT:-$GRAPH_OUT_DIR/${GRAPH_LABEL}.prediction.json}"
SUMMARY_OUT="${SUMMARY_OUT:-$GRAPH_OUT_DIR/${GRAPH_LABEL}.summary.json}"
ANALYSIS_OUT="${ANALYSIS_OUT:-$GRAPH_OUT_DIR/${GRAPH_LABEL}.retrieval_analysis.json}"
VS_DENSE_OUT="${VS_DENSE_OUT:-$GRAPH_OUT_DIR/${GRAPH_LABEL}.vs_dense.json}"
VS_SPLADE_OUT="${VS_SPLADE_OUT:-$GRAPH_OUT_DIR/${GRAPH_LABEL}.vs_splade.json}"

RECALL_K_VALUES="${RECALL_K_VALUES:-1 2 4 5 10 20 50 100 500 1000}"

mkdir -p "$GRAPH_OUT_DIR"

echo "using_dense_pred=$DENSE_PRED"
echo "using_sparse_pred=$SPARSE_PRED"
echo "using_gold=$GOLD"
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
    ;;
  denseheavy_lightboth)
    PROFILE_FINAL_TOP_PAGES=1000
    PROFILE_PER_DOC_PAGE_LIMIT=0
    PROFILE_DENSE_WEIGHT=1.25
    PROFILE_SPARSE_WEIGHT=0.75
    PROFILE_FINAL_PPR_PAGE_WEIGHT=0.25
    PROFILE_FINAL_PPR_DOC_WEIGHT=0.25
    ;;
  denseheavy150_m3best_pagepreserve)
    PROFILE_FINAL_TOP_PAGES=1000
    PROFILE_PER_DOC_PAGE_LIMIT=0
    PROFILE_DENSE_WEIGHT=1.5
    PROFILE_SPARSE_WEIGHT=0.5
    PROFILE_FINAL_PPR_PAGE_WEIGHT=1.5
    PROFILE_FINAL_PPR_DOC_WEIGHT=0.75
    ;;
  doc_shortlist_best|graph1000)
    PROFILE_FINAL_TOP_PAGES=20
    PROFILE_PER_DOC_PAGE_LIMIT=1
    PROFILE_DENSE_WEIGHT=1.0
    PROFILE_SPARSE_WEIGHT=1.0
    PROFILE_FINAL_PPR_PAGE_WEIGHT=1.5
    PROFILE_FINAL_PPR_DOC_WEIGHT=0.75
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
DOC_SEED_WEIGHT="${DOC_SEED_WEIGHT:-0.0}"
RESTART_PROB="${RESTART_PROB:-0.15}"
PPR_ITERS="${PPR_ITERS:-30}"
PAGE_DOC_EDGE_WEIGHT="${PAGE_DOC_EDGE_WEIGHT:-1.0}"
SAME_DOC_WINDOW="${SAME_DOC_WINDOW:-1}"
ADJACENT_PAGE_EDGE_WEIGHT="${ADJACENT_PAGE_EDGE_WEIGHT:-0.25}"
FINAL_PAGE_SEED_WEIGHT="${FINAL_PAGE_SEED_WEIGHT:-1.0}"
FINAL_PPR_PAGE_WEIGHT="${FINAL_PPR_PAGE_WEIGHT:-$PROFILE_FINAL_PPR_PAGE_WEIGHT}"
FINAL_PPR_DOC_WEIGHT="${FINAL_PPR_DOC_WEIGHT:-$PROFILE_FINAL_PPR_DOC_WEIGHT}"

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
  --doc-seed-weight "$DOC_SEED_WEIGHT"
  --restart-prob "$RESTART_PROB"
  --ppr-iters "$PPR_ITERS"
  --page-doc-edge-weight "$PAGE_DOC_EDGE_WEIGHT"
  --same-doc-window "$SAME_DOC_WINDOW"
  --adjacent-page-edge-weight "$ADJACENT_PAGE_EDGE_WEIGHT"
  --final-page-seed-weight "$FINAL_PAGE_SEED_WEIGHT"
  --final-ppr-page-weight "$FINAL_PPR_PAGE_WEIGHT"
  --final-ppr-doc-weight "$FINAL_PPR_DOC_WEIGHT"
  --output-prediction-json "$PRED_OUT"
  --output-summary-json "$SUMMARY_OUT"
)
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
