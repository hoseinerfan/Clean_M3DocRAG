#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any


VARIANT_ORDER = [
    "seed_only",
    "adjacent_only",
    "doc_edges_page_score_only",
    "doc_prior_only",
    "full_no_explicit_doc_score",
    "current_full_graph",
]
OUTPUT_COLUMNS = [
    "dataset",
    "variant",
    "page@1",
    "page@4",
    "delta_page@4_vs_seed",
    "page@20",
    "delta_page@20_vs_seed",
    "doc@4",
    "delta_doc@4_vs_seed",
    "doc@20",
    "delta_doc@20_vs_seed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect structural Graph-PPR ablation recall tables across datasets."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        nargs=2,
        metavar=("NAME", "RECALL_TABLE_CSV"),
        required=True,
        help="Dataset display name and structural-ablation recall CSV. Repeat per dataset.",
    )
    parser.add_argument("--format", choices=["markdown", "csv"], default="markdown")
    return parser.parse_args()


def as_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    return float(value)


def find_variant(label: str) -> str | None:
    for variant in VARIANT_ORDER:
        if label.endswith(f"_{variant}"):
            return variant
    return None


def read_rows(dataset: str, path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    rows: dict[str, dict[str, Any]] = {}
    for source in source_rows:
        variant = find_variant(source.get("label", ""))
        if variant is None:
            continue
        rows[variant] = {
            "dataset": dataset,
            "variant": variant,
            **{
                key: as_float(source.get(key))
                for key in ("page@1", "page@4", "page@20", "doc@4", "doc@20")
            },
        }
    missing = [variant for variant in VARIANT_ORDER if variant not in rows]
    if missing:
        raise ValueError(f"{path} is missing variants: {', '.join(missing)}")
    seed = rows["seed_only"]
    output: list[dict[str, Any]] = []
    for variant in VARIANT_ORDER:
        row = rows[variant]
        for metric in ("page@4", "page@20", "doc@4", "doc@20"):
            value = row[metric]
            base = seed[metric]
            row[f"delta_{metric}_vs_seed"] = (
                value - base if value is not None and base is not None else None
            )
        output.append(row)
    return output


def formatted(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_markdown(rows: list[dict[str, Any]]) -> None:
    print("| " + " | ".join(OUTPUT_COLUMNS) + " |")
    print("| " + " | ".join("---" for _ in OUTPUT_COLUMNS) + " |")
    for row in rows:
        print("| " + " | ".join(formatted(row.get(column)) for column in OUTPUT_COLUMNS) + " |")


def print_csv(rows: list[dict[str, Any]]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=OUTPUT_COLUMNS)
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column) for column in OUTPUT_COLUMNS})


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for dataset, path in args.dataset:
        rows.extend(read_rows(dataset, Path(path)))
    if args.format == "markdown":
        print_markdown(rows)
    else:
        print_csv(rows)


if __name__ == "__main__":
    main()
