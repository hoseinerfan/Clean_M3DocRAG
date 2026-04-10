#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze M3DocVQA retrieval outputs against gold supporting docs. "
            "Reports gold doc rank and a page-row proxy rank."
        )
    )
    parser.add_argument("--pred", required=True, help="Prediction JSON from run_rag_m3docvqa.py")
    parser.add_argument("--gold", required=True, help="Gold MMQA_<split>.jsonl")
    parser.add_argument(
        "--qid",
        dest="qids",
        action="append",
        default=[],
        help="Restrict to one or more qids; pass multiple times",
    )
    parser.add_argument(
        "--topn",
        type=int,
        default=10,
        help="How many retrieved rows to print per qid",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of text",
    )
    return parser.parse_args()


def first_unique_doc_ranks(retrieval_rows: list[list]) -> dict[str, int]:
    doc2rank: dict[str, int] = {}
    seen: set[str] = set()
    rank = 0
    for row in retrieval_rows:
        doc_id = row[0]
        if doc_id in seen:
            continue
        seen.add(doc_id)
        rank += 1
        doc2rank[doc_id] = rank
    return doc2rank


def analyze_one(qid: str, pred_item: dict, gold_item: dict, topn: int) -> dict:
    retrieval_rows = pred_item.get("page_retrieval_results", [])
    gold_doc_ids = sorted({ctx["doc_id"] for ctx in gold_item["supporting_context"]})
    doc_rank_map = first_unique_doc_ranks(retrieval_rows)

    gold_doc_unique_ranks = {
        doc_id: doc_rank_map.get(doc_id) for doc_id in gold_doc_ids
    }

    gold_page_row_ranks = []
    for idx, row in enumerate(retrieval_rows, start=1):
        doc_id = row[0]
        if doc_id in gold_doc_ids:
            gold_page_row_ranks.append(
                {
                    "row_rank": idx,
                    "doc_id": doc_id,
                    "page_idx": row[1],
                    "score": row[2],
                }
            )

    deduped_rows = []
    seen_docs: set[str] = set()
    for idx, row in enumerate(retrieval_rows, start=1):
        doc_id = row[0]
        if doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)
        deduped_rows.append(
            {
                "deduped_rank": len(deduped_rows) + 1,
                "original_row_rank": idx,
                "doc_id": row[0],
                "page_idx": row[1],
                "score": row[2],
            }
        )

    gold_page_row_ranks_deduped = [
        row for row in deduped_rows if row["doc_id"] in gold_doc_ids
    ]

    first_gold_doc_rank = min(
        (rank for rank in gold_doc_unique_ranks.values() if rank is not None),
        default=None,
    )
    first_gold_page_row_rank = (
        gold_page_row_ranks[0]["row_rank"] if gold_page_row_ranks else None
    )
    first_gold_page_row_rank_deduped = (
        gold_page_row_ranks_deduped[0]["deduped_rank"]
        if gold_page_row_ranks_deduped
        else None
    )

    return {
        "qid": qid,
        "question": gold_item["question"],
        "gold_answers": [a["answer"] for a in gold_item["answers"]],
        "gold_doc_ids": gold_doc_ids,
        "first_gold_doc_rank": first_gold_doc_rank,
        "gold_doc_unique_ranks": gold_doc_unique_ranks,
        "first_gold_page_row_rank": first_gold_page_row_rank,
        "first_gold_page_row_rank_deduped": first_gold_page_row_rank_deduped,
        "gold_page_row_hits": gold_page_row_ranks,
        "gold_page_ranks_without_deduping": [row["row_rank"] for row in gold_page_row_ranks],
        "gold_page_ranks_with_deduping": [row["deduped_rank"] for row in gold_page_row_ranks_deduped],
        "gold_page_hits_with_deduping": gold_page_row_ranks_deduped,
        "pred_answer": pred_item.get("pred_answer", ""),
        "time_retrieval": pred_item.get("time_retrieval"),
        "time_qa": pred_item.get("time_qa"),
        "top_retrieval_rows": retrieval_rows[:topn],
    }


def main() -> None:
    args = parse_args()

    pred_path = Path(args.pred)
    gold_path = Path(args.gold)

    pred = json.loads(pred_path.read_text())
    gold_rows = load_jsonl(gold_path)
    gold_by_qid = {row["qid"]: row for row in gold_rows}

    if args.qids:
        qids = args.qids
    else:
        qids = list(pred.keys())

    analyses = []
    for qid in qids:
        if qid not in pred:
            raise KeyError(f"QID missing in prediction file: {qid}")
        if qid not in gold_by_qid:
            raise KeyError(f"QID missing in gold file: {qid}")
        analyses.append(analyze_one(qid, pred[qid], gold_by_qid[qid], args.topn))

    if args.json:
        print(json.dumps(analyses, indent=2))
        return

    print(
        "Note: M3DocVQA does not provide true gold page_idx labels. "
        "The page-rank fields below mean retrieved page rows whose doc_id matches a gold supporting doc."
    )
    for item in analyses:
        print("-" * 80)
        print(f"qid {item['qid']}")
        print(f"question {item['question']}")
        print(f"gold_answers {item['gold_answers']}")
        print(f"gold_doc_ids {item['gold_doc_ids']}")
        print(f"first_gold_doc_rank {item['first_gold_doc_rank']}")
        print(f"gold_doc_unique_ranks {item['gold_doc_unique_ranks']}")
        print(f"first_gold_page_row_rank {item['first_gold_page_row_rank']}")
        print(f"first_gold_page_row_rank_deduped {item['first_gold_page_row_rank_deduped']}")
        print(f"gold_page_ranks_without_deduping {item['gold_page_ranks_without_deduping']}")
        print(f"gold_page_ranks_with_deduping {item['gold_page_ranks_with_deduping']}")
        print(f"gold_page_row_hits {item['gold_page_row_hits'][:5]}")
        print(f"gold_page_hits_with_deduping {item['gold_page_hits_with_deduping'][:5]}")
        print("top_retrieval_rows")
        for row in item["top_retrieval_rows"]:
            print(row)
        print(f"pred_answer {item['pred_answer']}")


if __name__ == "__main__":
    main()
