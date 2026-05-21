#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse dense and sparse page-retrieval predictions into a doc-ordered shortlist "
            "suitable for cue-verifier experiments."
        )
    )
    parser.add_argument("--dense-prediction-json", required=True)
    parser.add_argument("--sparse-prediction-json", required=True)
    parser.add_argument("--gold", help="Optional MMQA_<split>.jsonl for summary metrics.")
    parser.add_argument("--dense-top-docs", type=int, default=20)
    parser.add_argument("--sparse-top-docs", type=int, default=20)
    parser.add_argument("--dense-keep-docs", type=int, default=8)
    parser.add_argument("--sparse-add-docs", type=int, default=4)
    parser.add_argument("--output-prediction-json", required=True)
    parser.add_argument("--output-summary-json", required=True)
    return parser.parse_args()


def load_prediction(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Prediction JSON must be an object: {path}")
    return payload


def dedupe_doc_best_rows(rows: list[list[object]], top_docs: int) -> list[list[object]]:
    result: list[list[object]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, list) or len(row) < 3:
            continue
        doc_id = str(row[0]).strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        result.append([doc_id, int(row[1]), float(row[2])])
        if len(result) >= top_docs:
            break
    return result


def load_gold_rows(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("qid", "")).strip()
            if qid:
                rows[qid] = row
    return rows


def first_gold_doc_rank(rows: list[list[object]], gold_doc_ids: set[str]) -> int | None:
    seen: set[str] = set()
    rank = 0
    for row in rows:
        doc_id = str(row[0])
        if doc_id in seen:
            continue
        seen.add(doc_id)
        rank += 1
        if doc_id in gold_doc_ids:
            return rank
    return None


def main() -> None:
    args = parse_args()

    dense_pred = load_prediction(Path(args.dense_prediction_json))
    sparse_pred = load_prediction(Path(args.sparse_prediction_json))
    if set(dense_pred) != set(sparse_pred):
        missing_in_sparse = sorted(set(dense_pred) - set(sparse_pred))
        missing_in_dense = sorted(set(sparse_pred) - set(dense_pred))
        raise ValueError(
            "Dense/sparse qid sets differ: "
            f"missing_in_sparse={missing_in_sparse[:10]} "
            f"missing_in_dense={missing_in_dense[:10]}"
        )

    fused_payload: dict[str, dict] = {}
    summary_rows: list[dict] = []
    gold_rows = load_gold_rows(Path(args.gold)) if args.gold else {}

    for qid in sorted(dense_pred):
        dense_rows = dense_pred[qid].get("page_retrieval_results", [])
        sparse_rows = sparse_pred[qid].get("page_retrieval_results", [])
        dense_docs = dedupe_doc_best_rows(dense_rows, int(args.dense_top_docs))
        sparse_docs = dedupe_doc_best_rows(sparse_rows, int(args.sparse_top_docs))

        selected: list[list[object]] = []
        seen_docs: set[str] = set()
        for row in dense_docs[: int(args.dense_keep_docs)]:
            doc_id = str(row[0])
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            selected.append(row)
        for row in sparse_docs:
            if len(selected) >= int(args.dense_keep_docs) + int(args.sparse_add_docs):
                break
            doc_id = str(row[0])
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            selected.append(row)

        fused_payload[qid] = {
            "pred_answer": dense_pred[qid].get("pred_answer", ""),
            "page_retrieval_results": selected,
            "qid": qid,
            "question": dense_pred[qid].get("question", ""),
            "top_retrieved_docs": [row[0] for row in selected[:10]],
            "reranker_metadata": {
                "fusion_method": "dense_sparse_doc_union",
                "dense_prediction_json": args.dense_prediction_json,
                "sparse_prediction_json": args.sparse_prediction_json,
                "dense_keep_docs": int(args.dense_keep_docs),
                "sparse_add_docs": int(args.sparse_add_docs),
                "dense_top_docs": int(args.dense_top_docs),
                "sparse_top_docs": int(args.sparse_top_docs),
            },
        }

        row_summary = {
            "qid": qid,
            "fused_doc_count": len(selected),
            "fused_top_doc_ids": [row[0] for row in selected],
        }
        if gold_rows:
            gold_doc_ids = {
                str(item["doc_id"]).strip() for item in gold_rows[qid].get("supporting_context", [])
            }
            row_summary["reranked_first_gold_doc_rank"] = first_gold_doc_rank(selected, gold_doc_ids)
        summary_rows.append(row_summary)

    output_prediction_json = Path(args.output_prediction_json)
    output_prediction_json.parent.mkdir(parents=True, exist_ok=True)
    output_prediction_json.write_text(json.dumps(fused_payload, indent=2) + "\n", encoding="utf-8")

    summary = {
        "fusion_method": "dense_sparse_doc_union",
        "qid_count": len(summary_rows),
        "dense_keep_docs": int(args.dense_keep_docs),
        "sparse_add_docs": int(args.sparse_add_docs),
        "dense_top_docs": int(args.dense_top_docs),
        "sparse_top_docs": int(args.sparse_top_docs),
        "mean_fused_doc_count": (
            sum(int(row["fused_doc_count"]) for row in summary_rows) / len(summary_rows)
            if summary_rows
            else None
        ),
        "per_qid": summary_rows,
    }
    if gold_rows:
        summary["reranked_top4_doc_count"] = sum(
            1
            for row in summary_rows
            if row.get("reranked_first_gold_doc_rank") is not None
            and int(row["reranked_first_gold_doc_rank"]) <= 4
        )
        summary["reranked_top20_doc_count"] = sum(
            1
            for row in summary_rows
            if row.get("reranked_first_gold_doc_rank") is not None
            and int(row["reranked_first_gold_doc_rank"]) <= 20
        )

    output_summary_json = Path(args.output_summary_json)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"saved_prediction: {output_prediction_json}")
    print(f"saved_summary: {output_summary_json}")
    print(f"qid_count: {len(summary_rows)}")
    if gold_rows:
        print(f"reranked_top4_doc_count: {summary['reranked_top4_doc_count']}")
        print(f"reranked_top20_doc_count: {summary['reranked_top20_doc_count']}")


if __name__ == "__main__":
    main()
