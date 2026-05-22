#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/env/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

OUT_DIR="${OUT_DIR:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_graph_pagepreserve_mmqa_dev}"
DATA_NAME="${DATA_NAME:-m3docvqa-mmqa}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data/m3-docvqa/multimodalqa}"
SPLIT="${SPLIT:-dev}"
GOLD="${GOLD:-$REPO_ROOT/data/m3-docvqa/multimodalqa/MMQA_dev.jsonl}"

DENSE_PRED="${DENSE_PRED:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/mmqa_dev_plain_top224_nprobe4_effdiag_all.prediction.json}"
SPARSE_PRED="${SPARSE_PRED:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_splade_mmqa_dev/mmqa_dev_splade.prediction.json}"
QUESTION_TYPE_FILTER="${QUESTION_TYPE_FILTER:-}"

GRAPH_LABEL="${GRAPH_LABEL:-mmqa_dev_plain_top224_splade_graph_pagepreserve_denseheavy_lightboth}"
PRED_OUT="${PRED_OUT:-$OUT_DIR/${GRAPH_LABEL}.prediction.json}"
SUMMARY_OUT="${SUMMARY_OUT:-$OUT_DIR/${GRAPH_LABEL}.summary.json}"
ANALYSIS_OUT="${ANALYSIS_OUT:-$OUT_DIR/${GRAPH_LABEL}.retrieval_analysis.json}"
VS_DENSE_OUT="${VS_DENSE_OUT:-$OUT_DIR/${GRAPH_LABEL}.vs_dense.json}"
VS_SPLADE_OUT="${VS_SPLADE_OUT:-$OUT_DIR/${GRAPH_LABEL}.vs_splade.json}"

RECALL_K_VALUES="${RECALL_K_VALUES:-1 2 4 5 10 20 50 100 500 1000}"

mkdir -p "$OUT_DIR"

echo "using_dense_pred=$DENSE_PRED"
echo "using_sparse_pred=$SPARSE_PRED"
echo "using_gold=$GOLD"
echo "using_out_dir=$OUT_DIR"
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

# Best current page-preserving transfer config from the handoff:
# denseheavy_lightboth
GRAPH_ARGS=(
  --dense-prediction-json "$DENSE_PRED"
  --sparse-prediction-json "$SPARSE_PRED"
  --gold "$GOLD"
  --dense-top-pages 1000
  --sparse-top-pages 1000
  --final-top-pages 1000
  --per-doc-page-limit 0
  --rrf-k 10
  --dense-weight 1.25
  --sparse-weight 0.75
  --doc-seed-weight 0.0
  --restart-prob 0.15
  --ppr-iters 30
  --page-doc-edge-weight 1.0
  --same-doc-window 1
  --adjacent-page-edge-weight 0.25
  --final-page-seed-weight 1.0
  --final-ppr-page-weight 0.25
  --final-ppr-doc-weight 0.25
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
