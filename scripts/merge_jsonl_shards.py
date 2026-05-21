#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge JSONL shard files.")
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument(
        "--dedupe-key",
        default="",
        help="Optional JSON key to deduplicate on, keeping the first row.",
    )
    return parser.parse_args()


def natural_key(path: str) -> list[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path)]


def main() -> None:
    args = parse_args()
    input_paths = [Path(path) for path in sorted(glob.glob(args.input_glob), key=natural_key)]
    if not input_paths:
        raise FileNotFoundError(f"No files matched --input-glob: {args.input_glob}")

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json = Path(args.output_summary_json)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    row_count = 0
    skipped_duplicate_count = 0
    input_counts: dict[str, int] = {}
    with output_jsonl.open("w", encoding="utf-8") as out:
        for path in input_paths:
            count = 0
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    if args.dedupe_key:
                        row = json.loads(stripped)
                        key = str(row.get(args.dedupe_key, "") or "")
                        if key in seen:
                            skipped_duplicate_count += 1
                            continue
                        seen.add(key)
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    else:
                        out.write(stripped + "\n")
                    count += 1
                    row_count += 1
            input_counts[str(path)] = count

    summary = {
        "input_glob": args.input_glob,
        "input_files": [str(path) for path in input_paths],
        "input_file_count": len(input_paths),
        "input_counts": input_counts,
        "output_jsonl": str(output_jsonl),
        "row_count": row_count,
        "dedupe_key": args.dedupe_key,
        "skipped_duplicate_count": skipped_duplicate_count,
    }
    output_summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"input_files={len(input_paths)}")
    print(f"row_count={row_count}")
    print(f"saved_jsonl={output_jsonl}")
    print(f"saved_summary={output_summary_json}")


if __name__ == "__main__":
    main()
