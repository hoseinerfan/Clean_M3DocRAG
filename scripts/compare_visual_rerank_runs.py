#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl_by_qid(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row["qid"]
            rows[qid] = row
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two run_visual_rerank_batch JSONL outputs. "
            "Reports per-qid reranked gold-doc-rank movement and top-4 set changes."
        )
    )
    parser.add_argument("--baseline-run", required=True, help="Reference rerank-batch JSONL")
    parser.add_argument("--candidate-run", required=True, help="Candidate rerank-batch JSONL")
    parser.add_argument("--name-a", default="baseline_run", help="Display name for reference run")
    parser.add_argument("--name-b", default="candidate_run", help="Display name for candidate run")
    parser.add_argument(
        "--qid",
        dest="qids",
        action="append",
        default=[],
        help="Restrict to one or more qids; pass multiple times",
    )
    parser.add_argument("--topn", type=int, default=20, help="How many qids to print per bucket")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument(
        "--output-json",
        help="Optional path to save the JSON payload",
    )
    return parser.parse_args()


def is_top4(rank: int | None) -> bool:
    return rank is not None and rank <= 4


def compare_one(qid: str, row_a: dict[str, Any], row_b: dict[str, Any]) -> dict[str, Any]:
    baseline_rank_a = row_a.get("baseline_first_gold_doc_rank")
    baseline_rank_b = row_b.get("baseline_first_gold_doc_rank")
    reranked_rank_a = row_a.get("reranked_first_gold_doc_rank")
    reranked_rank_b = row_b.get("reranked_first_gold_doc_rank")

    if reranked_rank_a is None and reranked_rank_b is None:
        movement = "missing_in_both"
        rank_delta = None
    elif reranked_rank_a is None and reranked_rank_b is not None:
        movement = "newly_found"
        rank_delta = None
    elif reranked_rank_a is not None and reranked_rank_b is None:
        movement = "newly_lost"
        rank_delta = None
    else:
        rank_delta = reranked_rank_a - reranked_rank_b
        if reranked_rank_b < reranked_rank_a:
            movement = "improved"
        elif reranked_rank_b > reranked_rank_a:
            movement = "worsened"
        else:
            movement = "unchanged"

    return {
        "qid": qid,
        "question": row_a.get("question"),
        "baseline_first_gold_doc_rank_run_a": baseline_rank_a,
        "baseline_first_gold_doc_rank_run_b": baseline_rank_b,
        "baseline_rank_mismatch": baseline_rank_a != baseline_rank_b,
        "reranked_first_gold_doc_rank_run_a": reranked_rank_a,
        "reranked_first_gold_doc_rank_run_b": reranked_rank_b,
        "reranked_rank_improvement": rank_delta,
        "movement": movement,
        "top4_run_a": is_top4(reranked_rank_a),
        "top4_run_b": is_top4(reranked_rank_b),
    }


def build_payload(
    rows_a: dict[str, dict[str, Any]],
    rows_b: dict[str, dict[str, Any]],
    qids: list[str],
    name_a: str,
    name_b: str,
    topn: int,
) -> dict[str, Any]:
    comparisons = [compare_one(qid, rows_a[qid], rows_b[qid]) for qid in qids]

    counts = {
        "improved": sum(item["movement"] == "improved" for item in comparisons),
        "worsened": sum(item["movement"] == "worsened" for item in comparisons),
        "unchanged": sum(item["movement"] == "unchanged" for item in comparisons),
        "newly_found": sum(item["movement"] == "newly_found" for item in comparisons),
        "newly_lost": sum(item["movement"] == "newly_lost" for item in comparisons),
        "missing_in_both": sum(item["movement"] == "missing_in_both" for item in comparisons),
        "baseline_rank_mismatch": sum(item["baseline_rank_mismatch"] for item in comparisons),
        "top4_only_in_run_a": sum(item["top4_run_a"] and not item["top4_run_b"] for item in comparisons),
        "top4_only_in_run_b": sum(item["top4_run_b"] and not item["top4_run_a"] for item in comparisons),
        "top4_in_both": sum(item["top4_run_a"] and item["top4_run_b"] for item in comparisons),
        "top4_in_neither": sum((not item["top4_run_a"]) and (not item["top4_run_b"]) for item in comparisons),
    }

    improved = sorted(
        [item for item in comparisons if item["movement"] == "improved"],
        key=lambda item: (
            -(item["reranked_rank_improvement"] if item["reranked_rank_improvement"] is not None else -10**9),
            item["qid"],
        ),
    )
    worsened = sorted(
        [item for item in comparisons if item["movement"] == "worsened"],
        key=lambda item: (
            item["reranked_rank_improvement"] if item["reranked_rank_improvement"] is not None else 10**9,
            item["qid"],
        ),
    )
    top4_only_in_run_a = [
        item for item in comparisons if item["top4_run_a"] and not item["top4_run_b"]
    ]
    top4_only_in_run_b = [
        item for item in comparisons if item["top4_run_b"] and not item["top4_run_a"]
    ]
    top4_only_in_run_a.sort(
        key=lambda item: (
            item["reranked_first_gold_doc_rank_run_a"] if item["reranked_first_gold_doc_rank_run_a"] is not None else 10**9,
            item["qid"],
        )
    )
    top4_only_in_run_b.sort(
        key=lambda item: (
            item["reranked_first_gold_doc_rank_run_b"] if item["reranked_first_gold_doc_rank_run_b"] is not None else 10**9,
            item["qid"],
        )
    )

    return {
        "n_qids": len(qids),
        "name_a": name_a,
        "name_b": name_b,
        "counts": counts,
        "top_improved": improved[:topn],
        "top_worsened": worsened[:topn],
        "top4_only_in_run_a": top4_only_in_run_a[:topn],
        "top4_only_in_run_b": top4_only_in_run_b[:topn],
    }


def main() -> None:
    args = parse_args()

    rows_a = load_jsonl_by_qid(Path(args.baseline_run))
    rows_b = load_jsonl_by_qid(Path(args.candidate_run))

    qids = args.qids if args.qids else sorted(set(rows_a) & set(rows_b))
    missing_in_a = sorted(set(qids) - set(rows_a))
    missing_in_b = sorted(set(qids) - set(rows_b))
    if missing_in_a:
        raise KeyError(f"QIDs missing in {args.name_a}: {missing_in_a[:10]}")
    if missing_in_b:
        raise KeyError(f"QIDs missing in {args.name_b}: {missing_in_b[:10]}")

    payload = build_payload(rows_a, rows_b, qids, args.name_a, args.name_b, args.topn)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print("summary")
    print(f"n_qids {payload['n_qids']}")
    print(f"name_a {payload['name_a']}")
    print(f"name_b {payload['name_b']}")
    print(f"counts {payload['counts']}")

    print("-" * 80)
    print(f"top_improved {len(payload['top_improved'])}")
    for item in payload["top_improved"]:
        print(
            f"{item['qid']} | "
            f"{item['reranked_first_gold_doc_rank_run_a']} -> {item['reranked_first_gold_doc_rank_run_b']} | "
            f"+{item['reranked_rank_improvement']} | {item['question']}"
        )

    print("-" * 80)
    print(f"top_worsened {len(payload['top_worsened'])}")
    for item in payload["top_worsened"]:
        print(
            f"{item['qid']} | "
            f"{item['reranked_first_gold_doc_rank_run_a']} -> {item['reranked_first_gold_doc_rank_run_b']} | "
            f"{item['reranked_rank_improvement']} | {item['question']}"
        )

    print("-" * 80)
    print(f"top4_only_in_{payload['name_a']} {len(payload['top4_only_in_run_a'])}")
    for item in payload["top4_only_in_run_a"]:
        print(
            f"{item['qid']} | "
            f"{item['reranked_first_gold_doc_rank_run_a']} -> {item['reranked_first_gold_doc_rank_run_b']} | "
            f"{item['question']}"
        )

    print("-" * 80)
    print(f"top4_only_in_{payload['name_b']} {len(payload['top4_only_in_run_b'])}")
    for item in payload["top4_only_in_run_b"]:
        print(
            f"{item['qid']} | "
            f"{item['reranked_first_gold_doc_rank_run_a']} -> {item['reranked_first_gold_doc_rank_run_b']} | "
            f"{item['question']}"
        )


if __name__ == "__main__":
    main()
