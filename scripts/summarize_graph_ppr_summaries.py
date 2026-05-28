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

RECALL_K_VALUES = [1, 2, 4, 5, 10, 20, 50, 100]
RECALL_TABLE_COLUMNS = [
    "label",
    "qid_count",
    "dense_weight",
    "sparse_weight",
    "final_top_pages",
    "per_doc_page_limit",
    "final_selection_mode",
    "final_selection_top_k",
    "final_selection_candidate_pool",
    "final_selection_new_doc_bonus",
    "final_selection_same_doc_penalty",
    "final_selection_reordered_count",
    "mean_final_selection_selected_doc_count",
    "ppr_iters",
    "final_page_seed_weight",
    "final_ppr_page_weight",
    "final_ppr_doc_weight",
    "page_doc_edge_weight",
    "same_doc_window",
    "adjacent_page_edge_weight",
    *[f"page@{k}" for k in RECALL_K_VALUES],
    *[f"doc@{k}" for k in RECALL_K_VALUES],
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
    parser.add_argument(
        "--recall-table",
        action="store_true",
        help="Print derived page/doc recall@k columns from per-qid gold ranks.",
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
    columns = RECALL_TABLE_COLUMNS if rows and "page@1" in rows[0] else TABLE_COLUMNS
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        print("| " + " | ".join(format_value(row.get(column)) for column in columns) + " |")


def print_delimited(rows: list[dict[str, Any]], *, delimiter: str, recall_table: bool = False) -> None:
    columns = RECALL_TABLE_COLUMNS if recall_table else TABLE_COLUMNS
    writer = csv.DictWriter(sys.stdout, fieldnames=columns, delimiter=delimiter)
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column) for column in columns})


def row_from_summary(path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    row = {column: summary.get(column) for column in TABLE_COLUMNS}
    row["label"] = compact_label(path)
    return row


def recall_at_k(ranks: list[int | None], k: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return sum(1 for rank in ranks if rank is not None and int(rank) <= k) / denominator


def recall_row_from_summary(path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    row = {column: summary.get(column) for column in RECALL_TABLE_COLUMNS}
    row["label"] = compact_label(path)
    denominator = int(summary.get("qid_count") or 0)
    per_qid = summary.get("per_qid", [])
    if not isinstance(per_qid, list):
        per_qid = []
    page_ranks = [
        item.get("reranked_first_gold_page_rank")
        for item in per_qid
        if isinstance(item, dict)
    ]
    doc_ranks = [
        item.get("reranked_first_gold_doc_rank")
        for item in per_qid
        if isinstance(item, dict)
    ]
    for k in RECALL_K_VALUES:
        row[f"page@{k}"] = recall_at_k(page_ranks, k, denominator)
        row[f"doc@{k}"] = recall_at_k(doc_ranks, k, denominator)
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
    rows = [
        recall_row_from_summary(path, load_summary(path))
        if args.recall_table
        else row_from_summary(path, load_summary(path))
        for path in paths
    ]
    if args.format == "markdown":
        print_markdown(rows)
    elif args.format == "tsv":
        print_delimited(rows, delimiter="\t", recall_table=bool(args.recall_table))
    else:
        print_delimited(rows, delimiter=",", recall_table=bool(args.recall_table))


if __name__ == "__main__":
    main()
