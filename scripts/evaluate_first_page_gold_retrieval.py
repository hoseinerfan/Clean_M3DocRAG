#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate retrieval with synthetic page gold labels constructed as "
            "page N of each gold supporting-context document. Useful for "
            "M3DocVQA/ImageListQ diagnostics when page-level gold is unavailable."
        )
    )
    parser.add_argument("--pred", required=True, help="Prediction JSON keyed by qid.")
    parser.add_argument("--gold", required=True, help="MMQA-style JSONL with supporting_context.")
    parser.add_argument(
        "--question-type",
        default="",
        help="Optional metadata.type filter, e.g. ImageListQ.",
    )
    parser.add_argument(
        "--first-page-idx",
        type=int,
        default=0,
        help="Synthetic gold page index for every gold doc. Default: 0.",
    )
    parser.add_argument(
        "--recall-k",
        dest="recall_ks",
        type=int,
        nargs="+",
        default=[1, 2, 4, 5, 10, 20, 50, 100, 500, 1000],
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON payload.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def gold_doc_ids(row: dict[str, Any]) -> list[str]:
    return sorted(
        {
            str(ctx.get("doc_id", "")).strip()
            for ctx in row.get("supporting_context", [])
            if str(ctx.get("doc_id", "")).strip()
        }
    )


def synthetic_first_page_uids(row: dict[str, Any], first_page_idx: int) -> list[str]:
    return [f"{doc_id}_page{int(first_page_idx)}" for doc_id in gold_doc_ids(row)]


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / len(gold)


def first_rank(ranked: list[str], gold: set[str]) -> int | None:
    for idx, item in enumerate(ranked, start=1):
        if item in gold:
            return idx
    return None


def metadata_type(row: dict[str, Any]) -> str:
    return str(row.get("metadata", {}).get("type", "")).strip()


def main() -> None:
    args = parse_args()
    pred = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    if not isinstance(pred, dict):
        raise TypeError(f"Prediction JSON must be an object keyed by qid: {args.pred}")
    gold_rows = read_jsonl(Path(args.gold))
    wanted_type = str(args.question_type).strip()

    per_qid: list[dict[str, Any]] = []
    skipped_no_prediction = 0
    skipped_no_gold_doc = 0
    for gold_row in gold_rows:
        if wanted_type and metadata_type(gold_row) != wanted_type:
            continue
        qid = str(gold_row.get("qid", "")).strip()
        if not qid:
            continue
        if qid not in pred:
            skipped_no_prediction += 1
            continue
        doc_gold = set(gold_doc_ids(gold_row))
        if not doc_gold:
            skipped_no_gold_doc += 1
            continue
        page_gold = set(synthetic_first_page_uids(gold_row, int(args.first_page_idx)))
        retrieval_rows = pred[qid].get("page_retrieval_results", [])
        ranked_pages = [f"{row[0]}_page{int(row[1])}" for row in retrieval_rows]
        ranked_docs: list[str] = []
        seen_docs: set[str] = set()
        for row in retrieval_rows:
            doc_id = str(row[0])
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                ranked_docs.append(doc_id)

        item = {
            "qid": qid,
            "question": gold_row.get("question", ""),
            "metadata_type": metadata_type(gold_row),
            "gold_doc_ids": sorted(doc_gold),
            "synthetic_gold_page_uids": sorted(page_gold),
            "first_gold_doc_rank": first_rank(ranked_docs, doc_gold),
            "first_synthetic_gold_page_rank": first_rank(ranked_pages, page_gold),
            "doc_recall_at_k": {str(k): recall_at_k(ranked_docs, doc_gold, k) for k in args.recall_ks},
            "synthetic_page_recall_at_k": {
                str(k): recall_at_k(ranked_pages, page_gold, k) for k in args.recall_ks
            },
        }
        per_qid.append(item)

    summary: dict[str, Any] = {
        "n_qids": len(per_qid),
        "question_type": wanted_type,
        "first_page_idx": int(args.first_page_idx),
        "skipped_no_prediction": skipped_no_prediction,
        "skipped_no_gold_doc": skipped_no_gold_doc,
        "synthetic_page_recall_at_k": {},
        "doc_recall_at_k": {},
    }
    for k in args.recall_ks:
        key = str(k)
        summary["synthetic_page_recall_at_k"][key] = (
            sum(item["synthetic_page_recall_at_k"][key] for item in per_qid) / len(per_qid)
            if per_qid
            else 0.0
        )
        summary["doc_recall_at_k"][key] = (
            sum(item["doc_recall_at_k"][key] for item in per_qid) / len(per_qid)
            if per_qid
            else 0.0
        )
    summary["synthetic_page_hit_at_4_count"] = sum(
        1
        for item in per_qid
        if item["first_synthetic_gold_page_rank"] is not None
        and int(item["first_synthetic_gold_page_rank"]) <= 4
    )
    summary["doc_hit_at_4_count"] = sum(
        1
        for item in per_qid
        if item["first_gold_doc_rank"] is not None and int(item["first_gold_doc_rank"]) <= 4
    )

    payload = {"summary": summary, "per_qid": per_qid}
    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(f"n_qids {summary['n_qids']}")
    print(f"question_type {summary['question_type']}")
    print(f"first_page_idx {summary['first_page_idx']}")
    print(f"skipped_no_prediction {summary['skipped_no_prediction']}")
    print(f"skipped_no_gold_doc {summary['skipped_no_gold_doc']}")
    print(f"synthetic_page_recall_at_k {summary['synthetic_page_recall_at_k']}")
    print(f"doc_recall_at_k {summary['doc_recall_at_k']}")
    print(f"synthetic_page_hit_at_4_count {summary['synthetic_page_hit_at_4_count']}")
    print(f"doc_hit_at_4_count {summary['doc_hit_at_4_count']}")


if __name__ == "__main__":
    main()
