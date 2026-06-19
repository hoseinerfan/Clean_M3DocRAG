#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter an M3DocVQA/MMQA gold JSONL to QIDs present in every supplied "
            "prediction JSON."
        )
    )
    parser.add_argument("--gold", required=True, help="Input MMQA gold JSONL.")
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="Prediction JSON. Repeat for every run in the paired comparison.",
    )
    parser.add_argument("--output-gold", required=True)
    parser.add_argument("--output-summary", default="")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prediction_qids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Prediction must be a JSON object: {path}")
    return {str(qid) for qid, row in payload.items() if isinstance(row, dict)}


def main() -> None:
    args = parse_args()
    gold_path = Path(args.gold)
    prediction_paths = [Path(value) for value in args.prediction]
    gold_rows = load_jsonl(gold_path)
    gold_qids = {str(row.get("qid", "")).strip() for row in gold_rows}

    per_prediction_qids = {str(path): prediction_qids(path) for path in prediction_paths}
    common_qids = set(gold_qids)
    for qids in per_prediction_qids.values():
        common_qids.intersection_update(qids)

    filtered_rows = [row for row in gold_rows if str(row.get("qid", "")).strip() in common_qids]
    output_path = Path(args.output_gold)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in filtered_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_gold": str(gold_path),
        "input_gold_qids": len(gold_qids),
        "prediction_qid_counts": {path: len(qids) for path, qids in per_prediction_qids.items()},
        "intersection_qids": len(common_qids),
        "excluded_gold_qids": len(gold_qids - common_qids),
        "output_gold": str(output_path),
    }
    if args.output_summary:
        summary_path = Path(args.output_summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"saved_summary={summary_path}")

    print(f"saved_filtered_gold={output_path}")
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
