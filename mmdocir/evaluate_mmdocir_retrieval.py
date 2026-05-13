#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MMDocIR retrieval JSON against exact gold pages.")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gold", required=True, help="Converted MMQA_dev.jsonl")
    parser.add_argument("--recall-k", dest="recall_ks", type=int, nargs="+", default=[1, 2, 4, 5, 10, 20, 50, 100, 500, 1000])
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def gold_page_uids(row: dict) -> list[str]:
    metadata = row.get("metadata", {})
    uids = {
        str(value).strip()
        for value in metadata.get("gold_page_uids", [])
        if str(value).strip()
    }
    for ctx in row.get("supporting_context", []):
        doc_id = str(ctx.get("doc_id", "")).strip()
        page_idx = ctx.get("page_idx", ctx.get("page_id"))
        if doc_id and page_idx is not None:
            uids.add(f"{doc_id}_page{int(page_idx)}")
    return sorted(uids)


def gold_doc_ids(row: dict) -> list[str]:
    return sorted({str(ctx["doc_id"]).strip() for ctx in row.get("supporting_context", []) if str(ctx.get("doc_id", "")).strip()})


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / len(gold)


def first_rank(ranked: list[str], gold: set[str]) -> int | None:
    for idx, item in enumerate(ranked, start=1):
        if item in gold:
            return idx
    return None


def main() -> None:
    args = parse_args()
    pred = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    gold_rows = read_jsonl(Path(args.gold))

    per_qid = []
    for gold_row in gold_rows:
        qid = str(gold_row["qid"])
        if qid not in pred:
            continue
        retrieval_rows = pred[qid].get("page_retrieval_results", [])
        ranked_pages = [f"{row[0]}_page{int(row[1])}" for row in retrieval_rows]
        ranked_docs = []
        seen_docs = set()
        for row in retrieval_rows:
            doc_id = str(row[0])
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                ranked_docs.append(doc_id)

        page_gold = set(gold_page_uids(gold_row))
        doc_gold = set(gold_doc_ids(gold_row))
        item = {
            "qid": qid,
            "question": gold_row.get("question", ""),
            "gold_page_uids": sorted(page_gold),
            "gold_doc_ids": sorted(doc_gold),
            "first_gold_page_rank": first_rank(ranked_pages, page_gold),
            "first_gold_doc_rank": first_rank(ranked_docs, doc_gold),
            "page_recall_at_k": {str(k): recall_at_k(ranked_pages, page_gold, k) for k in args.recall_ks},
            "doc_recall_at_k": {str(k): recall_at_k(ranked_docs, doc_gold, k) for k in args.recall_ks},
        }
        per_qid.append(item)

    summary = {"n_qids": len(per_qid), "page_recall_at_k": {}, "doc_recall_at_k": {}}
    for k in args.recall_ks:
        key = str(k)
        summary["page_recall_at_k"][key] = (
            sum(item["page_recall_at_k"][key] for item in per_qid) / len(per_qid)
            if per_qid
            else 0.0
        )
        summary["doc_recall_at_k"][key] = (
            sum(item["doc_recall_at_k"][key] for item in per_qid) / len(per_qid)
            if per_qid
            else 0.0
        )
    summary["page_hit_at_4_count"] = sum(
        1 for item in per_qid if item["first_gold_page_rank"] is not None and item["first_gold_page_rank"] <= 4
    )
    summary["doc_hit_at_4_count"] = sum(
        1 for item in per_qid if item["first_gold_doc_rank"] is not None and item["first_gold_doc_rank"] <= 4
    )

    payload = {"summary": summary, "per_qid": per_qid}
    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(f"n_qids {summary['n_qids']}")
    print(f"page_recall_at_k {summary['page_recall_at_k']}")
    print(f"doc_recall_at_k {summary['doc_recall_at_k']}")
    print(f"page_hit_at_4_count {summary['page_hit_at_4_count']}")
    print(f"doc_hit_at_4_count {summary['doc_hit_at_4_count']}")


if __name__ == "__main__":
    main()

