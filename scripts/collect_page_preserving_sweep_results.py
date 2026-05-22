#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import csv
import re
import sys
from pathlib import Path
from typing import Any


DEFAULT_COLUMNS = [
    "dataset",
    "role",
    "label",
    "dense_weight",
    "sparse_weight",
    "ppr_iters",
    "final_ppr_page_weight",
    "final_ppr_doc_weight",
    "page@1",
    "page@4",
    "page@10",
    "page@20",
    "doc@4",
    "doc@20",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect page-preserving Graph-PPR sweep recall CSVs into a compact "
            "cross-dataset table."
        )
    )
    parser.add_argument(
        "--dataset",
        action="append",
        nargs=2,
        metavar=("NAME", "RECALL_TABLE_CSV"),
        required=True,
        help="Dataset name and sweep recall_table.csv path. Repeat for each dataset.",
    )
    parser.add_argument(
        "--plain-eval",
        action="append",
        nargs=2,
        metavar=("NAME", "PLAIN_EVAL_TXT"),
        default=[],
        help="Optional dataset name and plain_top224 eval txt path.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top page@4 configs to include per dataset.",
    )
    parser.add_argument(
        "--sort-metric",
        default="page@4",
        help="Metric used to select top configs from each CSV.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "csv"],
        default="markdown",
    )
    return parser.parse_args()


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def read_recall_csv(dataset: str, path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    parsed: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {"dataset": dataset, "role": "sweep", **row}
        for key, value in list(item.items()):
            if key.startswith("page@") or key.startswith("doc@") or key.endswith("_weight"):
                item[key] = as_float(value)
        item["ppr_iters"] = int(float(item["ppr_iters"])) if str(item.get("ppr_iters", "")).strip() else ""
        parsed.append(item)
    return parsed


def parse_metric_dict(text: str, metric_name: str) -> dict[str, float]:
    match = re.search(rf"{re.escape(metric_name)}\s+(\{{[^\n]+\}})", text)
    if not match:
        return {}
    raw = ast.literal_eval(match.group(1))
    if not isinstance(raw, dict):
        return {}
    return {str(key): float(value) for key, value in raw.items()}


def read_plain_eval(dataset: str, path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    page = parse_metric_dict(text, "page_recall_at_k")
    doc = parse_metric_dict(text, "doc_recall_at_k")
    return {
        "dataset": dataset,
        "role": "plain_top224",
        "label": "plain_top224",
        "dense_weight": "",
        "sparse_weight": "",
        "ppr_iters": "",
        "final_ppr_page_weight": "",
        "final_ppr_doc_weight": "",
        "page@1": page.get("1"),
        "page@4": page.get("4"),
        "page@10": page.get("10"),
        "page@20": page.get("20"),
        "doc@4": doc.get("4"),
        "doc@20": doc.get("20"),
    }


def print_markdown(rows: list[dict[str, Any]]) -> None:
    print("| " + " | ".join(DEFAULT_COLUMNS) + " |")
    print("| " + " | ".join("---" for _ in DEFAULT_COLUMNS) + " |")
    for row in rows:
        print("| " + " | ".join(format_value(row.get(column)) for column in DEFAULT_COLUMNS) + " |")


def print_csv(rows: list[dict[str, Any]]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=DEFAULT_COLUMNS)
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column) for column in DEFAULT_COLUMNS})


def main() -> None:
    args = parse_args()
    plain_by_dataset = {name: Path(path) for name, path in args.plain_eval}
    output_rows: list[dict[str, Any]] = []

    for dataset, csv_path in args.dataset:
        rows = read_recall_csv(dataset, Path(csv_path))
        if dataset in plain_by_dataset:
            output_rows.append(read_plain_eval(dataset, plain_by_dataset[dataset]))
        best_rows = sorted(
            rows,
            key=lambda row: as_float(row.get(args.sort_metric)) if as_float(row.get(args.sort_metric)) is not None else -1.0,
            reverse=True,
        )[: int(args.top_n)]
        for row in best_rows:
            row["role"] = f"top{args.top_n}_by_{args.sort_metric}"
            output_rows.append(row)

    if args.format == "markdown":
        print_markdown(output_rows)
    else:
        print_csv(output_rows)


if __name__ == "__main__":
    main()
