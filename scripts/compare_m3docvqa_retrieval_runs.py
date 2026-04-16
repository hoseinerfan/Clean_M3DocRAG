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
            "Compare two M3DocVQA retrieval prediction files against the same gold file. "
            "Reports recall deltas and per-qid gold-doc rank movement."
        )
    )
    parser.add_argument("--baseline", required=True, help="Baseline prediction JSON")
    parser.add_argument("--candidate", required=True, help="Candidate prediction JSON")
    parser.add_argument("--gold", required=True, help="Gold MMQA_<split>.jsonl")
    parser.add_argument(
        "--qid",
        dest="qids",
        action="append",
        default=[],
        help="Restrict to one or more qids; pass multiple times",
    )
    parser.add_argument(
        "--recall-k",
        dest="recall_ks",
        type=int,
        nargs="+",
        default=[1, 2, 4, 5, 10, 20, 50, 100, 500, 1000],
        help="Recall@k levels to compute",
    )
    parser.add_argument(
        "--topn",
        type=int,
        default=20,
        help="How many improved / worsened qids to print",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of plain text",
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
    ranked_rows = dedupe_rows_by_doc(retrieval_rows) if dedupe_first else retrieval_rows

    recalls = {}
    for k in recall_ks:
        top_k_rows = ranked_rows[:k]
        top_k_doc_ids = {row[0] for row in top_k_rows}
        recalls[k] = len(top_k_doc_ids & gold_doc_ids) / n_relevant if n_relevant > 0 else 0.0
    return recalls


def analyze_one(pred_item: dict, gold_item: dict, recall_ks: list[int]) -> dict:
    retrieval_rows = pred_item.get("page_retrieval_results", [])
    gold_doc_ids = sorted({ctx["doc_id"] for ctx in gold_item["supporting_context"]})
    gold_doc_id_set = set(gold_doc_ids)
    doc_rank_map = first_unique_doc_ranks(retrieval_rows)
    gold_doc_unique_ranks = {doc_id: doc_rank_map.get(doc_id) for doc_id in gold_doc_ids}

    first_gold_doc_rank = min(
        (rank for rank in gold_doc_unique_ranks.values() if rank is not None),
        default=None,
    )

    return {
        "question": gold_item["question"],
        "gold_answers": [a["answer"] for a in gold_item["answers"]],
        "gold_doc_ids": gold_doc_ids,
        "first_gold_doc_rank": first_gold_doc_rank,
        "recall_at_k_without_deduping": compute_recall_dict(
            retrieval_rows, gold_doc_id_set, recall_ks, dedupe_first=False
        ),
        "recall_at_k_with_deduping": compute_recall_dict(
            retrieval_rows, gold_doc_id_set, recall_ks, dedupe_first=True
        ),
    }


def average_recall(analyses: list[dict], key: str, recall_ks: list[int]) -> dict[int, float]:
    return {
        k: (sum(item[key][k] for item in analyses) / len(analyses) if analyses else 0.0)
        for k in recall_ks
    }


def delta_recall(candidate: dict[int, float], baseline: dict[int, float]) -> dict[int, float]:
    return {k: candidate[k] - baseline[k] for k in baseline}


def compare_ranks(qid: str, baseline_item: dict, candidate_item: dict) -> dict:
    b = baseline_item["first_gold_doc_rank"]
    c = candidate_item["first_gold_doc_rank"]

    if b is None and c is None:
        movement = "missing_in_both"
        delta = None
    elif b is None and c is not None:
        movement = "newly_found"
        delta = None
    elif b is not None and c is None:
        movement = "newly_lost"
        delta = None
    else:
        delta = b - c
        if c < b:
            movement = "improved"
        elif c > b:
            movement = "worsened"
        else:
            movement = "unchanged"

    return {
        "qid": qid,
        "question": baseline_item["question"],
        "gold_answers": baseline_item["gold_answers"],
        "gold_doc_ids": baseline_item["gold_doc_ids"],
        "baseline_first_gold_doc_rank": b,
        "candidate_first_gold_doc_rank": c,
        "rank_improvement": delta,
        "movement": movement,
    }


def main() -> None:
    args = parse_args()

    baseline = json.loads(Path(args.baseline).read_text())
    candidate = json.loads(Path(args.candidate).read_text())
    gold_rows = load_jsonl(Path(args.gold))
    gold_by_qid = {row["qid"]: row for row in gold_rows}

    qids = args.qids if args.qids else sorted(set(baseline.keys()) & set(candidate.keys()))

    baseline_analyses = []
    candidate_analyses = []
    comparisons = []
    for qid in qids:
        if qid not in gold_by_qid:
            raise KeyError(f"QID missing in gold file: {qid}")
        if qid not in baseline:
            raise KeyError(f"QID missing in baseline file: {qid}")
        if qid not in candidate:
            raise KeyError(f"QID missing in candidate file: {qid}")

        baseline_item = analyze_one(baseline[qid], gold_by_qid[qid], args.recall_ks)
        candidate_item = analyze_one(candidate[qid], gold_by_qid[qid], args.recall_ks)
        baseline_analyses.append(baseline_item)
        candidate_analyses.append(candidate_item)
        comparisons.append(compare_ranks(qid, baseline_item, candidate_item))

    baseline_recall_wo = average_recall(baseline_analyses, "recall_at_k_without_deduping", args.recall_ks)
    candidate_recall_wo = average_recall(candidate_analyses, "recall_at_k_without_deduping", args.recall_ks)
    baseline_recall_w = average_recall(baseline_analyses, "recall_at_k_with_deduping", args.recall_ks)
    candidate_recall_w = average_recall(candidate_analyses, "recall_at_k_with_deduping", args.recall_ks)

    counts = {
        "improved": sum(item["movement"] == "improved" for item in comparisons),
        "worsened": sum(item["movement"] == "worsened" for item in comparisons),
        "unchanged": sum(item["movement"] == "unchanged" for item in comparisons),
        "newly_found": sum(item["movement"] == "newly_found" for item in comparisons),
        "newly_lost": sum(item["movement"] == "newly_lost" for item in comparisons),
        "missing_in_both": sum(item["movement"] == "missing_in_both" for item in comparisons),
    }

    improved = sorted(
        [item for item in comparisons if item["movement"] == "improved"],
        key=lambda x: (-x["rank_improvement"], x["qid"]),
    )
    worsened = sorted(
        [item for item in comparisons if item["movement"] == "worsened"],
        key=lambda x: (x["rank_improvement"], x["qid"]),
    )

    payload = {
        "n_qids": len(qids),
        "baseline": {
            "average_recall_at_k_without_deduping": baseline_recall_wo,
            "average_recall_at_k_with_deduping": baseline_recall_w,
        },
        "candidate": {
            "average_recall_at_k_without_deduping": candidate_recall_wo,
            "average_recall_at_k_with_deduping": candidate_recall_w,
        },
        "delta": {
            "average_recall_at_k_without_deduping": delta_recall(candidate_recall_wo, baseline_recall_wo),
            "average_recall_at_k_with_deduping": delta_recall(candidate_recall_w, baseline_recall_w),
        },
        "counts": counts,
        "top_improved": improved[: args.topn],
        "top_worsened": worsened[: args.topn],
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print("summary")
    print(f"n_qids {payload['n_qids']}")
    print(f"counts {counts}")
    print(f"baseline_average_recall_at_k_without_deduping {baseline_recall_wo}")
    print(f"candidate_average_recall_at_k_without_deduping {candidate_recall_wo}")
    print(f"delta_average_recall_at_k_without_deduping {payload['delta']['average_recall_at_k_without_deduping']}")
    print(f"baseline_average_recall_at_k_with_deduping {baseline_recall_w}")
    print(f"candidate_average_recall_at_k_with_deduping {candidate_recall_w}")
    print(f"delta_average_recall_at_k_with_deduping {payload['delta']['average_recall_at_k_with_deduping']}")

    print("-" * 80)
    print(f"top_improved_first_gold_doc_rank {len(payload['top_improved'])}")
    for item in payload["top_improved"]:
        print(
            f"{item['qid']} | {item['baseline_first_gold_doc_rank']} -> "
            f"{item['candidate_first_gold_doc_rank']} | +{item['rank_improvement']} | {item['question']}"
        )

    print("-" * 80)
    print(f"top_worsened_first_gold_doc_rank {len(payload['top_worsened'])}")
    for item in payload["top_worsened"]:
        print(
            f"{item['qid']} | {item['baseline_first_gold_doc_rank']} -> "
            f"{item['candidate_first_gold_doc_rank']} | {item['rank_improvement']} | {item['question']}"
        )


if __name__ == "__main__":
    main()
