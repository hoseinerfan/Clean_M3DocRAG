#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path
from typing import Any


RECALL_K_VALUES = [1, 2, 4, 5, 10, 20, 50, 100]
COLUMNS = [
    "label",
    "n_qids",
    *[f"doc@{k}" for k in RECALL_K_VALUES],
    *[f"row@{k}" for k in RECALL_K_VALUES],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize M3DocVQA retrieval analysis JSON files produced by "
            "scripts/analyze_m3docvqa_retrieval.py."
        )
    )
    parser.add_argument("analysis_jsons", nargs="+", help="Analysis JSON paths or quoted globs.")
    parser.add_argument("--format", choices=["markdown", "csv"], default="markdown")
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


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected object JSON: {path}")
    return payload


def compact_label(path: Path) -> str:
    name = path.name
    suffix = ".retrieval_analysis.json"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return path.stem


def recall_value(summary: dict[str, Any], key: str, k: int) -> float | None:
    values = summary.get(key, {})
    if not isinstance(values, dict):
        return None
    value = values.get(str(k), values.get(k))
    return float(value) if value is not None else None


def row_from_analysis(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    row: dict[str, Any] = {
        "label": compact_label(path),
        "n_qids": int(summary.get("n_qids") or 0),
    }
    for k in RECALL_K_VALUES:
        row[f"doc@{k}"] = recall_value(summary, "average_recall_at_k_with_deduping", k)
        row[f"row@{k}"] = recall_value(summary, "average_recall_at_k_without_deduping", k)
    return row


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def print_markdown(rows: list[dict[str, Any]]) -> None:
    print("| " + " | ".join(COLUMNS) + " |")
    print("| " + " | ".join("---" for _ in COLUMNS) + " |")
    for row in rows:
        print("| " + " | ".join(format_value(row.get(column)) for column in COLUMNS) + " |")


def print_csv(rows: list[dict[str, Any]]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=COLUMNS)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    rows = [row_from_analysis(path, load_json(path)) for path in expand_paths(args.analysis_jsons)]
    if args.format == "markdown":
        print_markdown(rows)
    else:
        print_csv(rows)


if __name__ == "__main__":
    main()
