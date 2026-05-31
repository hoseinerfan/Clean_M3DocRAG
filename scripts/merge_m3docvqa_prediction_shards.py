#!/usr/bin/env python3
import argparse
import glob
import json
from pathlib import Path

from m3docrag.datasets.m3_docvqa import evaluate_prediction_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge M3DocVQA prediction JSON shards and optionally evaluate the merged file."
    )
    parser.add_argument("--input-glob", required=True, help="Glob matching shard prediction JSON files.")
    parser.add_argument("--output-pred", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--output-eval", default="")
    parser.add_argument("--expected-count", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = sorted(Path(path) for path in glob.glob(args.input_glob))
    if not paths:
        raise SystemExit(f"No shard files matched: {args.input_glob}")

    merged = {}
    duplicate_qids = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            shard = json.load(handle)
        if not isinstance(shard, dict):
            raise ValueError(f"Expected JSON object in {path}")
        for qid, row in shard.items():
            if qid in merged:
                duplicate_qids.append(qid)
            merged[qid] = row

    if duplicate_qids:
        sample = ", ".join(duplicate_qids[:5])
        raise ValueError(f"Found {len(duplicate_qids)} duplicate qids while merging, sample: {sample}")
    if args.expected_count and len(merged) != args.expected_count:
        raise ValueError(
            f"Merged qid count mismatch: got {len(merged)}, expected {args.expected_count}"
        )

    output_pred = Path(args.output_pred)
    output_pred.parent.mkdir(parents=True, exist_ok=True)
    with output_pred.open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2)
    print(f"saved_prediction={output_pred}")
    print(f"merged_qids={len(merged)}")
    print(f"merged_shards={len(paths)}")

    if args.output_eval:
        scores = evaluate_prediction_file(merged, args.gold)
        output_eval = Path(args.output_eval)
        output_eval.parent.mkdir(parents=True, exist_ok=True)
        with output_eval.open("w", encoding="utf-8") as handle:
            json.dump(scores, handle, indent=2)
        print(f"saved_eval={output_eval}")


if __name__ == "__main__":
    main()
