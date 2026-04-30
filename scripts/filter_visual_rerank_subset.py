#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a run_visual_rerank_batch JSONL into a smaller qid JSONL subset. "
            "Useful for extracting near-miss, regression, or missing-gold groups."
        )
    )
    parser.add_argument("--input-jsonl", required=True, help="run_visual_rerank_batch JSONL")
    parser.add_argument("--output-jsonl", required=True, help="Subset qid JSONL to write")
    parser.add_argument(
        "--rank-source",
        choices=["baseline", "reranked"],
        default="reranked",
        help="Which gold-doc rank field to filter on",
    )
    parser.add_argument("--min-rank", type=int, help="Keep qids with rank >= this value")
    parser.add_argument("--max-rank", type=int, help="Keep qids with rank <= this value")
    parser.add_argument(
        "--require-not-top4",
        action="store_true",
        help="Keep only qids whose selected rank is missing or > 4",
    )
    parser.add_argument(
        "--require-improved-vs-baseline",
        action="store_true",
        help="Keep only qids where reranked_first_gold_doc_rank < baseline_first_gold_doc_rank",
    )
    parser.add_argument(
        "--require-worsened-vs-baseline",
        action="store_true",
        help="Keep only qids where reranked_first_gold_doc_rank > baseline_first_gold_doc_rank",
    )
    parser.add_argument(
        "--include-none-ranks",
        action="store_true",
        help="Allow qids with selected rank=None to pass rank filters",
    )
    parser.add_argument("--max-qids", type=int, help="Optional cap after filtering")
    return parser.parse_args()


def selected_rank(row: dict[str, Any], rank_source: str) -> int | None:
    if rank_source == "baseline":
        return row.get("baseline_first_gold_doc_rank")
    return row.get("reranked_first_gold_doc_rank")


def should_keep(row: dict[str, Any], args: argparse.Namespace) -> bool:
    rank = selected_rank(row, args.rank_source)
    baseline_rank = row.get("baseline_first_gold_doc_rank")
    reranked_rank = row.get("reranked_first_gold_doc_rank")

    if args.require_not_top4 and rank is not None and rank <= 4:
        return False
    if args.require_not_top4 and rank is None and not args.include_none_ranks:
        return False

    if args.require_improved_vs_baseline:
        if baseline_rank is None or reranked_rank is None or reranked_rank >= baseline_rank:
            return False

    if args.require_worsened_vs_baseline:
        if baseline_rank is None or reranked_rank is None or reranked_rank <= baseline_rank:
            return False

    if rank is None:
        return args.include_none_ranks

    if args.min_rank is not None and rank < args.min_rank:
        return False
    if args.max_rank is not None and rank > args.max_rank:
        return False

    return True


def main() -> None:
    args = parse_args()
    if args.require_improved_vs_baseline and args.require_worsened_vs_baseline:
        raise ValueError("Cannot require both improved-vs-baseline and worsened-vs-baseline.")

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)

    kept: list[dict[str, Any]] = []
    total = 0
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)
            if not should_keep(row, args):
                continue
            kept.append(
                {
                    "qid": row["qid"],
                    "question": row.get("question"),
                    "baseline_first_gold_doc_rank": row.get("baseline_first_gold_doc_rank"),
                    "reranked_first_gold_doc_rank": row.get("reranked_first_gold_doc_rank"),
                }
            )
            if args.max_qids is not None and len(kept) >= args.max_qids:
                break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"input_jsonl: {input_path}")
    print(f"output_jsonl: {output_path}")
    print(f"total_qids_seen: {total}")
    print(f"kept_qids: {len(kept)}")


if __name__ == "__main__":
    main()
