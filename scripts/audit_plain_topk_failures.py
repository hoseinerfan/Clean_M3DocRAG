#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a top-k pruning failure subset by comparing the approximate top-k run "
            "against an exact-page control on the same saved retrieval pool."
        )
    )
    parser.add_argument(
        "--topk-run",
        required=True,
        help="run_visual_rerank_batch JSONL for the approximate top-k method on a failure subset.",
    )
    parser.add_argument(
        "--exact-run",
        required=True,
        help="run_visual_rerank_batch JSONL for the exact-page control on the same qids/pool.",
    )
    parser.add_argument(
        "--failure-topk",
        type=int,
        default=4,
        help="Failure cutoff used to define unsuccessful retrieval. Default: 4.",
    )
    parser.add_argument(
        "--near-miss-max-rank",
        type=int,
        default=10,
        help="Exact-page ranks up to this value are counted as near-boundary misses. Default: 10.",
    )
    parser.add_argument(
        "--pruning-damage-min-rank-gain",
        type=int,
        default=5,
        help=(
            "Minimum exact-rank improvement over the top-k run needed to call a miss "
            "a pruning-damage case. Default: 5."
        ),
    )
    parser.add_argument(
        "--output-summary-json",
        help="Optional path to save the full audit summary JSON.",
    )
    parser.add_argument(
        "--output-bucket-dir",
        help="Optional directory to write one JSONL file per bucket.",
    )
    return parser.parse_args()


def load_run(path: Path) -> dict[str, dict]:
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
    if not rows:
        raise ValueError(f"No qids found in {path}")
    return rows


def rank_gain(old_rank: int | None, new_rank: int | None) -> int | None:
    if old_rank is None or new_rank is None:
        return None
    return int(old_rank) - int(new_rank)


def classify_record(
    *,
    topk_row: dict,
    exact_row: dict,
    failure_topk: int,
    near_miss_max_rank: int,
    pruning_damage_min_rank_gain: int,
) -> dict:
    baseline_rank = topk_row.get("baseline_first_gold_doc_rank")
    topk_rank = topk_row.get("reranked_first_gold_doc_rank")
    exact_rank = exact_row.get("reranked_first_gold_doc_rank")

    gain_exact_vs_topk = rank_gain(topk_rank, exact_rank)
    gain_topk_vs_baseline = rank_gain(baseline_rank, topk_rank)
    gain_exact_vs_baseline = rank_gain(baseline_rank, exact_rank)

    diagnostics = topk_row.get("token_pruning_diagnostic_summary", {})

    if exact_rank is None:
        bucket = "retrieval_missing"
    elif int(exact_rank) <= failure_topk:
        bucket = "pruning_damage_exact_recovers_topk"
    elif gain_exact_vs_topk is not None and int(gain_exact_vs_topk) >= pruning_damage_min_rank_gain:
        bucket = "pruning_damage_exact_improves_but_still_fails"
    elif failure_topk < int(exact_rank) <= near_miss_max_rank:
        bucket = "boundary_exact_still_fails"
    else:
        bucket = "hard_fail_even_exact"

    return {
        "qid": topk_row["qid"],
        "question": topk_row.get("question"),
        "bucket": bucket,
        "baseline_first_gold_doc_rank": baseline_rank,
        "topk_first_gold_doc_rank": topk_rank,
        "exact_first_gold_doc_rank": exact_rank,
        "exact_rank_gain_vs_topk": gain_exact_vs_topk,
        "topk_rank_gain_vs_baseline": gain_topk_vs_baseline,
        "exact_rank_gain_vs_baseline": gain_exact_vs_baseline,
        "gold_doc_ids": topk_row.get("gold_doc_ids", []),
        "topk_mean_exact_score_loss": diagnostics.get("mean_exact_score_loss"),
        "topk_mean_shifted_score_preservation_ratio": diagnostics.get(
            "mean_shifted_score_preservation_ratio"
        ),
        "topk_mean_argmax_retention_ratio": diagnostics.get("mean_argmax_retention_ratio"),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def bucket_sort_key(item: dict) -> tuple:
    exact_rank = item.get("exact_first_gold_doc_rank")
    topk_rank = item.get("topk_first_gold_doc_rank")
    gain = item.get("exact_rank_gain_vs_topk")
    exact_rank_value = 10**9 if exact_rank is None else int(exact_rank)
    topk_rank_value = 10**9 if topk_rank is None else int(topk_rank)
    gain_value = -10**9 if gain is None else int(gain)
    return (-gain_value, exact_rank_value, topk_rank_value, item["qid"])


def main() -> None:
    args = parse_args()
    topk_rows = load_run(Path(args.topk_run))
    exact_rows = load_run(Path(args.exact_run))

    shared_qids = sorted(set(topk_rows) & set(exact_rows))
    missing_from_exact = sorted(set(topk_rows) - set(exact_rows))
    missing_from_topk = sorted(set(exact_rows) - set(topk_rows))
    if missing_from_exact:
        raise KeyError(f"QIDs missing in exact run: {missing_from_exact[:10]}")
    if missing_from_topk:
        raise KeyError(f"QIDs missing in top-k run: {missing_from_topk[:10]}")

    records = [
        classify_record(
            topk_row=topk_rows[qid],
            exact_row=exact_rows[qid],
            failure_topk=int(args.failure_topk),
            near_miss_max_rank=int(args.near_miss_max_rank),
            pruning_damage_min_rank_gain=int(args.pruning_damage_min_rank_gain),
        )
        for qid in shared_qids
    ]

    buckets: dict[str, list[dict]] = {
        "retrieval_missing": [],
        "pruning_damage_exact_recovers_topk": [],
        "pruning_damage_exact_improves_but_still_fails": [],
        "boundary_exact_still_fails": [],
        "hard_fail_even_exact": [],
    }
    for item in records:
        buckets[item["bucket"]].append(item)
    for rows in buckets.values():
        rows.sort(key=bucket_sort_key)

    summary = {
        "topk_run": args.topk_run,
        "exact_run": args.exact_run,
        "failure_topk": int(args.failure_topk),
        "near_miss_max_rank": int(args.near_miss_max_rank),
        "pruning_damage_min_rank_gain": int(args.pruning_damage_min_rank_gain),
        "qid_count": len(shared_qids),
        "bucket_counts": {label: len(rows) for label, rows in buckets.items()},
        "bucket_examples": {
            label: rows[:10]
            for label, rows in buckets.items()
        },
    }

    if args.output_summary_json:
        out_path = Path(args.output_summary_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.output_bucket_dir:
        bucket_dir = Path(args.output_bucket_dir)
        bucket_dir.mkdir(parents=True, exist_ok=True)
        for label, rows in buckets.items():
            write_jsonl(bucket_dir / f"{label}.jsonl", rows)

    print(f"qid_count: {summary['qid_count']}")
    for label, count in summary["bucket_counts"].items():
        print(f"{label}: {count}")


if __name__ == "__main__":
    main()
