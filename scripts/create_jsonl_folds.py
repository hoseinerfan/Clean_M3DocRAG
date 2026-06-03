#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic train/held-out folds from a JSONL file with qid fields."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--label", default="folds")
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--summary-json", default="")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("qid", "")).strip()
            if not qid:
                raise ValueError(f"Missing qid at {path}:{line_no}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    args = parse_args()
    if int(args.fold_count) < 2:
        raise ValueError("--fold-count must be at least 2.")

    rows = read_jsonl(Path(args.input_jsonl))
    qids = [str(row["qid"]) for row in rows]
    if len(set(qids)) != len(qids):
        raise ValueError("Input JSONL contains duplicate qids.")

    shuffled = list(qids)
    rng = random.Random(int(args.seed))
    rng.shuffle(shuffled)
    fold_by_qid = {
        qid: idx % int(args.fold_count)
        for idx, qid in enumerate(shuffled)
    }

    out_dir = Path(args.out_dir)
    summary_rows: list[dict[str, Any]] = []
    for fold_idx in range(int(args.fold_count)):
        heldout_rows = [row for row in rows if fold_by_qid[str(row["qid"])] == fold_idx]
        train_rows = [row for row in rows if fold_by_qid[str(row["qid"])] != fold_idx]
        train_path = out_dir / f"{args.label}_fold{fold_idx}_train.jsonl"
        heldout_path = out_dir / f"{args.label}_fold{fold_idx}_heldout.jsonl"
        write_jsonl(train_path, train_rows)
        write_jsonl(heldout_path, heldout_rows)
        summary_rows.append(
            {
                "fold": int(fold_idx),
                "train_qid_count": int(len(train_rows)),
                "heldout_qid_count": int(len(heldout_rows)),
                "train_jsonl": str(train_path),
                "heldout_jsonl": str(heldout_path),
            }
        )

    summary = {
        "input_jsonl": args.input_jsonl,
        "label": args.label,
        "fold_count": int(args.fold_count),
        "seed": int(args.seed),
        "qid_count": int(len(rows)),
        "folds": summary_rows,
    }
    summary_path = Path(args.summary_json) if args.summary_json else out_dir / f"{args.label}_folds.summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"saved_summary={summary_path}")
    for row in summary_rows:
        print(
            f"fold={row['fold']} train_qids={row['train_qid_count']} "
            f"heldout_qids={row['heldout_qid_count']}"
        )


if __name__ == "__main__":
    main()
