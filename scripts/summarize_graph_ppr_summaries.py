#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from pathlib import Path
from typing import Any


TABLE_COLUMNS = [
    "label",
    "question_type",
    "qid_count",
    "dense_top_pages",
    "sparse_top_pages",
    "reranked_top4_doc_count",
    "reranked_top20_doc_count",
    "dense_top4_doc_count",
    "sparse_top4_doc_count",
    "candidate_gold_doc_count",
    "candidate_gold_doc_miss_count",
    "mean_candidate_page_count",
    "mean_candidate_doc_count",
    "graph_recovers_top4_doc_vs_dense_count",
    "graph_loses_top4_doc_vs_dense_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize graph/PPR retrieval summary JSONs into compact tables, "
            "and optionally compare two summaries with paired doc@k tests."
        )
    )
    parser.add_argument(
        "summary_jsons",
        nargs="*",
        help="Summary JSON files. Shell globs are accepted when quoted.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "tsv", "csv"],
        default="markdown",
        help="Output table format for summary rows.",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE_SUMMARY", "CANDIDATE_SUMMARY"),
        help="Print paired doc@4/doc@20 comparison for two summary JSON files.",
    )
    return parser.parse_args()


def expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return paths


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if not isinstance(summary, dict):
        raise TypeError(f"Expected object summary JSON: {path}")
    return summary


def compact_label(path: Path) -> str:
    name = path.name
    if name.endswith(".summary.json"):
        return name[: -len(".summary.json")]
    return path.stem


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def print_markdown(rows: list[dict[str, Any]]) -> None:
    print("| " + " | ".join(TABLE_COLUMNS) + " |")
    print("| " + " | ".join("---" for _ in TABLE_COLUMNS) + " |")
    for row in rows:
        print("| " + " | ".join(format_value(row.get(column)) for column in TABLE_COLUMNS) + " |")


def print_delimited(rows: list[dict[str, Any]], *, delimiter: str) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=TABLE_COLUMNS, delimiter=delimiter)
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column) for column in TABLE_COLUMNS})


def row_from_summary(path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    row = {column: summary.get(column) for column in TABLE_COLUMNS}
    row["label"] = compact_label(path)
    return row


def first_doc_rank_by_qid(summary: dict[str, Any]) -> dict[str, int | None]:
    ranks: dict[str, int | None] = {}
    for row in summary.get("per_qid", []):
        if not isinstance(row, dict):
            continue
        qid = str(row.get("qid", "")).strip()
        if not qid:
            continue
        rank = row.get("reranked_first_gold_doc_rank")
        ranks[qid] = int(rank) if rank is not None else None
    return ranks


def logsumexp(values: list[float]) -> float:
    if not values:
        return float("-inf")
    offset = max(values)
    return offset + math.log(sum(math.exp(value - offset) for value in values))


def binomial_two_sided_pvalue(successes: int, failures: int) -> float:
    n = int(successes) + int(failures)
    if n <= 0:
        return 1.0
    tail = min(int(successes), int(failures))
    log_terms = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2.0)
        for i in range(tail + 1)
    ]
    return min(1.0, 2.0 * math.exp(logsumexp(log_terms)))


def compare_summaries(base_path: Path, candidate_path: Path) -> None:
    base = load_summary(base_path)
    candidate = load_summary(candidate_path)
    base_ranks = first_doc_rank_by_qid(base)
    candidate_ranks = first_doc_rank_by_qid(candidate)
    qids = sorted(set(base_ranks) & set(candidate_ranks))
    print(f"baseline: {base_path}")
    print(f"candidate: {candidate_path}")
    print(f"paired_qids: {len(qids)}")
    for k in [4, 20]:
        candidate_only = 0
        baseline_only = 0
        both = 0
        neither = 0
        for qid in qids:
            base_hit = base_ranks[qid] is not None and int(base_ranks[qid]) <= k
            candidate_hit = candidate_ranks[qid] is not None and int(candidate_ranks[qid]) <= k
            if base_hit and candidate_hit:
                both += 1
            elif candidate_hit:
                candidate_only += 1
            elif base_hit:
                baseline_only += 1
            else:
                neither += 1
        p_value = binomial_two_sided_pvalue(candidate_only, baseline_only)
        print(
            f"doc@{k}: baseline={both + baseline_only} "
            f"candidate={both + candidate_only} "
            f"net={candidate_only - baseline_only} "
            f"candidate_only={candidate_only} baseline_only={baseline_only} "
            f"both={both} neither={neither} sign_test_p={p_value:.6g}"
        )


def main() -> None:
    args = parse_args()
    if args.compare:
        compare_summaries(Path(args.compare[0]), Path(args.compare[1]))
        if not args.summary_jsons:
            return

    paths = expand_paths(args.summary_jsons)
    if not paths:
        return
    rows = [row_from_summary(path, load_summary(path)) for path in paths]
    if args.format == "markdown":
        print_markdown(rows)
    elif args.format == "tsv":
        print_delimited(rows, delimiter="\t")
    else:
        print_delimited(rows, delimiter=",")


if __name__ == "__main__":
    main()
