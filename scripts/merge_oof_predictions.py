#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge out-of-fold prediction JSON files by taking only qids listed in each "
            "fold's held-out JSONL file."
        )
    )
    parser.add_argument(
        "--fold",
        action="append",
        required=True,
        help="HELDOUT_JSONL=PREDICTION_JSON. May be repeated.",
    )
    parser.add_argument("--expected-jsonl", default="", help="Optional JSONL whose qids must all be covered.")
    parser.add_argument("--output-prediction-json", required=True)
    parser.add_argument("--output-summary-json", required=True)
    return parser.parse_args()


def read_qids(path: Path) -> list[str]:
    qids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("qid", "")).strip()
            if not qid:
                raise ValueError(f"Missing qid at {path}:{line_no}")
            qids.append(qid)
    return qids


def load_prediction(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("predictions"), (dict, list)):
        payload = payload["predictions"]
    if not isinstance(payload, dict):
        raise TypeError(f"Prediction JSON must be an object keyed by qid: {path}")
    out: dict[str, dict[str, Any]] = {}
    for key, row in payload.items():
        if not isinstance(row, dict):
            continue
        qid = str(row.get("qid") or key).strip()
        if qid:
            out[qid] = row
    return out


def parse_fold_spec(spec: str) -> tuple[Path, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected HELDOUT_JSONL=PREDICTION_JSON, got {spec!r}")
    heldout, pred = spec.split("=", 1)
    return Path(heldout), Path(pred)


def main() -> None:
    args = parse_args()
    merged: dict[str, dict[str, Any]] = {}
    duplicate_qids: list[str] = []
    missing_in_fold_predictions: list[str] = []
    fold_summaries: list[dict[str, Any]] = []

    for fold_idx, spec in enumerate(args.fold):
        heldout_path, pred_path = parse_fold_spec(spec)
        heldout_qids = read_qids(heldout_path)
        pred = load_prediction(pred_path)
        fold_missing: list[str] = []
        for qid in heldout_qids:
            if qid not in pred:
                fold_missing.append(qid)
                missing_in_fold_predictions.append(qid)
                continue
            if qid in merged:
                duplicate_qids.append(qid)
                continue
            merged[qid] = pred[qid]
        fold_summaries.append(
            {
                "fold": int(fold_idx),
                "heldout_jsonl": str(heldout_path),
                "prediction_json": str(pred_path),
                "heldout_qid_count": int(len(heldout_qids)),
                "prediction_qid_count": int(len(pred)),
                "merged_qid_count": int(sum(1 for qid in heldout_qids if qid in merged)),
                "missing_in_prediction_count": int(len(fold_missing)),
                "missing_sample": fold_missing[:10],
            }
        )

    expected_qids: list[str] = read_qids(Path(args.expected_jsonl)) if args.expected_jsonl else []
    missing_expected: list[str] = []
    duplicate_expected_count = 0
    if expected_qids:
        counts = Counter(expected_qids)
        duplicate_expected_count = sum(1 for _, count in counts.items() if count > 1)
        expected_set = set(expected_qids)
        missing_expected = sorted(qid for qid in expected_set if qid not in merged)

    if duplicate_qids:
        raise ValueError(f"Duplicate held-out qids across folds: {duplicate_qids[:10]}")
    if missing_in_fold_predictions:
        raise ValueError(f"Fold predictions missing held-out qids: {missing_in_fold_predictions[:10]}")
    if missing_expected:
        raise ValueError(f"Merged OOF prediction missing expected qids: {missing_expected[:10]}")
    if duplicate_expected_count:
        raise ValueError(f"Expected JSONL contains duplicate qids: {duplicate_expected_count}")

    output_pred = Path(args.output_prediction_json)
    output_pred.parent.mkdir(parents=True, exist_ok=True)
    output_pred.write_text(json.dumps(merged) + "\n", encoding="utf-8")

    summary = {
        "fold_count": int(len(args.fold)),
        "merged_qid_count": int(len(merged)),
        "expected_qid_count": int(len(expected_qids)) if expected_qids else None,
        "output_prediction_json": str(output_pred),
        "folds": fold_summaries,
    }
    output_summary = Path(args.output_summary_json)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"saved_prediction={output_pred}")
    print(f"saved_summary={output_summary}")
    print(f"merged_qid_count={len(merged)}")


if __name__ == "__main__":
    main()
