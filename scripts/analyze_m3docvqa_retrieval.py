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
    parser.add_argument(
        "--recall-k",
        dest="recall_ks",
        type=int,
        nargs="+",
        default=[20, 50, 100, 500, 1000],
        help="Recall@k levels to compute",
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


def dedupe_rows_by_doc(retrieval_rows: list[list]) -> list[list]:
    deduped = []
    seen_docs: set[str] = set()
    for row in retrieval_rows:
        doc_id = row[0]
        if doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)
        deduped.append(row)
    return deduped


def compute_recall_dict(
    retrieval_rows: list[list], gold_doc_ids: set[str], recall_ks: list[int], dedupe_first: bool
) -> dict[int, float]:
    n_relevant = len(gold_doc_ids)
    if dedupe_first:
        ranked_rows = dedupe_rows_by_doc(retrieval_rows)
    else:
        ranked_rows = retrieval_rows

    recalls = {}
    for k in recall_ks:
        top_k_rows = ranked_rows[:k]
        top_k_doc_ids = {row[0] for row in top_k_rows}
        recalls[k] = len(top_k_doc_ids & gold_doc_ids) / n_relevant if n_relevant > 0 else 0.0
    return recalls


def analyze_one(qid: str, pred_item: dict, gold_item: dict, topn: int, recall_ks: list[int]) -> dict:
    retrieval_rows = pred_item.get("page_retrieval_results", [])
    gold_doc_ids = sorted({ctx["doc_id"] for ctx in gold_item["supporting_context"]})
    gold_doc_id_set = set(gold_doc_ids)
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
        row for row in deduped_rows if row["doc_id"] in gold_doc_id_set
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
        "recall_at_k_without_deduping": compute_recall_dict(
            retrieval_rows, gold_doc_id_set, recall_ks, dedupe_first=False
        ),
        "recall_at_k_with_deduping": compute_recall_dict(
            retrieval_rows, gold_doc_id_set, recall_ks, dedupe_first=True
        ),
        "pred_answer": pred_item.get("pred_answer", ""),
        "time_retrieval": pred_item.get("time_retrieval"),
        "time_qa": pred_item.get("time_qa"),
        "top_retrieval_rows": retrieval_rows[:topn],
    }


def average_recall(analyses: list[dict], key: str, recall_ks: list[int]) -> dict[int, float]:
    return {
        k: (
            sum(item[key][k] for item in analyses) / len(analyses)
            if analyses
            else 0.0
        )
        for k in recall_ks
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
        analyses.append(analyze_one(qid, pred[qid], gold_by_qid[qid], args.topn, args.recall_ks))

    summary = {
        "n_qids": len(analyses),
        "average_recall_at_k_without_deduping": average_recall(
            analyses, "recall_at_k_without_deduping", args.recall_ks
        ),
        "average_recall_at_k_with_deduping": average_recall(
            analyses, "recall_at_k_with_deduping", args.recall_ks
        ),
    }

    if args.json:
        print(json.dumps({"summary": summary, "per_qid": analyses}, indent=2))
        return

    print(
        "Note: M3DocVQA does not provide true gold page_idx labels. "
        "The page-rank fields below mean retrieved page rows whose doc_id matches a gold supporting doc."
    )
    print("summary")
    print(f"n_qids {summary['n_qids']}")
    print(
        "average_recall_at_k_without_deduping "
        f"{summary['average_recall_at_k_without_deduping']}"
    )
    print(
        "average_recall_at_k_with_deduping "
        f"{summary['average_recall_at_k_with_deduping']}"
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
        print(f"recall_at_k_without_deduping {item['recall_at_k_without_deduping']}")
        print(f"recall_at_k_with_deduping {item['recall_at_k_with_deduping']}")
        print(f"gold_page_row_hits {item['gold_page_row_hits'][:5]}")
        print(f"gold_page_hits_with_deduping {item['gold_page_hits_with_deduping'][:5]}")
        print("top_retrieval_rows")
        for row in item["top_retrieval_rows"]:
            print(row)
        print(f"pred_answer {item['pred_answer']}")


if __name__ == "__main__":
    main()
