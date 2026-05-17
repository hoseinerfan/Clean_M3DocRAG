#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded retrieval prediction JSON files.")
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--gold", default="", help="Optional MMQA jsonl file used to verify expected qids.")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--overwrite-duplicates", action="store_true")
    return parser.parse_args()


def read_expected_qids(path: Path) -> list[str]:
    qids = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            qids.append(str(json.loads(line)["qid"]))
    return qids


def main() -> None:
    args = parse_args()
    input_paths = sorted(Path(path) for path in glob.glob(args.input_glob))
    if not input_paths:
        raise FileNotFoundError(f"No files matched: {args.input_glob}")

    merged: dict[str, dict] = {}
    duplicate_qids = []
    for path in input_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"Prediction shard is not a JSON object: {path}")
        for qid, row in payload.items():
            if qid in merged and not args.overwrite_duplicates:
                duplicate_qids.append(qid)
                continue
            merged[qid] = row

    if duplicate_qids:
        sample = duplicate_qids[:10]
        raise ValueError(f"Found duplicate qids across shards: count={len(duplicate_qids)} sample={sample}")

    missing_qids = []
    if args.gold:
        expected_qids = read_expected_qids(Path(args.gold))
        expected_set = set(expected_qids)
        missing_qids = [qid for qid in expected_qids if qid not in merged]
        extra_qids = sorted(set(merged) - expected_set)
        if missing_qids and not args.allow_missing:
            raise ValueError(f"Missing qids: count={len(missing_qids)} sample={missing_qids[:10]}")
        if extra_qids:
            raise ValueError(f"Unexpected qids: count={len(extra_qids)} sample={extra_qids[:10]}")

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")

    print(f"input_files={len(input_paths)}")
    print(f"merged_qids={len(merged)}")
    if args.gold:
        print(f"missing_qids={len(missing_qids)}")
    print(f"saved_prediction={output_path}")


if __name__ == "__main__":
    main()
