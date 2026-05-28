#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_RECALL_KS = [1, 2, 4, 5, 10, 20]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize M3DocVQA prediction runs by qid group. Each group table includes "
            "the baseline row, candidate rows, and candidate deltas/movement counts vs baseline."
        )
    )
    parser.add_argument("--gold", required=True, help="Gold MMQA_<split>.jsonl")
    parser.add_argument("--baseline", required=True, help="Baseline prediction JSON")
    parser.add_argument("--baseline-label", default="base", help="Label for the baseline row")
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        help="Group as NAME=path/to/qids.txt. Can be passed multiple times. Defaults to all qids.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate as LABEL=prediction.json, prediction.json, or a quoted glob.",
    )
    parser.add_argument(
        "--candidate-glob",
        action="append",
        default=[],
        help="Candidate prediction JSON glob. Labels are inferred from filenames.",
    )
    parser.add_argument(
        "--recall-k",
        type=int,
        nargs="+",
        default=DEFAULT_RECALL_KS,
        help="Recall@k levels to report.",
    )
    parser.add_argument("--format", choices=["markdown", "csv", "json"], default="markdown")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_prediction(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "predictions" in payload and isinstance(
        payload["predictions"], (dict, list)
    ):
        payload = payload["predictions"]

    if isinstance(payload, list):
        iterable = enumerate(payload)
    elif isinstance(payload, dict):
        iterable = payload.items()
    else:
        raise TypeError(f"Prediction JSON must be a list or object: {path}")

    rows_by_qid: dict[str, dict[str, Any]] = {}
    for raw_key, row in iterable:
        if not isinstance(row, dict):
            raise TypeError(f"Prediction row must be object: {path} key={raw_key!r}")
        qid = str(row.get("qid", "")).strip() or str(raw_key).strip()
        if not qid:
            raise ValueError(f"Prediction row missing qid: {path} key={raw_key!r}")
        if qid in rows_by_qid:
            raise ValueError(f"Duplicate qid after normalization: {qid} ({path})")
        rows_by_qid[qid] = row
    return rows_by_qid


def load_qids(path: Path) -> list[str]:
    qids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value and not value.startswith("#"):
                qids.append(value)
    return ordered_unique(qids)


def ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def compact_label(path: Path) -> str:
    suffix = ".prediction.json"
    name = path.name
    return name[: -len(suffix)] if name.endswith(suffix) else path.stem


def parse_labeled_path(spec: str) -> tuple[str | None, str]:
    if "=" not in spec:
        return None, spec
    label, path = spec.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"Invalid labeled path: {spec!r}")
    return label, path


def expand_candidate_specs(specs: list[str], glob_specs: list[str]) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []
    for spec in [*specs, *glob_specs]:
        label, pattern = parse_labeled_path(spec)
        matches = sorted(glob.glob(pattern))
        paths = [Path(match) for match in matches] if matches else [Path(pattern)]
        if label and len(paths) != 1:
            raise ValueError(f"Labeled candidate matched {len(paths)} paths, expected 1: {spec!r}")
        for path in paths:
            candidates.append((label or compact_label(path), path))
    return candidates


def parse_groups(group_specs: list[str], default_qids: list[str]) -> list[tuple[str, list[str]]]:
    if not group_specs:
        return [("all", default_qids)]
    groups: list[tuple[str, list[str]]] = []
    for spec in group_specs:
        label, path = parse_labeled_path(spec)
        if label is None:
            raise ValueError(f"Group must be NAME=path/to/qids.txt: {spec!r}")
        groups.append((label, load_qids(Path(path))))
    return groups


def retrieval_rows(pred_item: dict[str, Any]) -> list[list[Any]]:
    rows = pred_item.get("page_retrieval_results", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, list) and row]


def dedupe_rows_by_doc(rows: list[list[Any]]) -> list[list[Any]]:
    deduped: list[list[Any]] = []
    seen_docs: set[str] = set()
    for row in rows:
        doc_id = str(row[0])
        if doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)
        deduped.append(row)
    return deduped


def first_gold_doc_rank(rows: list[list[Any]], gold_doc_ids: set[str]) -> int | None:
    seen_docs: set[str] = set()
    rank = 0
    for row in rows:
        doc_id = str(row[0])
        if doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)
        rank += 1
        if doc_id in gold_doc_ids:
            return rank
    return None


def recall_at_k(
    rows: list[list[Any]],
    gold_doc_ids: set[str],
    recall_ks: list[int],
    *,
    dedupe_first: bool,
) -> dict[int, float]:
    ranked_rows = dedupe_rows_by_doc(rows) if dedupe_first else rows
    denom = len(gold_doc_ids)
    values: dict[int, float] = {}
    for k in recall_ks:
        top_docs = {str(row[0]) for row in ranked_rows[:k]}
        values[k] = len(top_docs & gold_doc_ids) / denom if denom else 0.0
    return values


def gold_doc_ids(row: dict[str, Any]) -> set[str]:
    return {
        str(ctx.get("doc_id", "")).strip()
        for ctx in row.get("supporting_context", [])
        if isinstance(ctx, dict) and str(ctx.get("doc_id", "")).strip()
    }


def analyze_run(
    *,
    pred: dict[str, dict[str, Any]],
    gold_by_qid: dict[str, dict[str, Any]],
    qids: list[str],
    recall_ks: list[int],
) -> dict[str, Any]:
    doc_recalls = {k: [] for k in recall_ks}
    row_recalls = {k: [] for k in recall_ks}
    first_ranks: dict[str, int | None] = {}
    for qid in qids:
        if qid not in gold_by_qid:
            raise KeyError(f"QID missing from gold: {qid}")
        if qid not in pred:
            raise KeyError(f"QID missing from prediction: {qid}")
        docs = gold_doc_ids(gold_by_qid[qid])
        rows = retrieval_rows(pred[qid])
        first_ranks[qid] = first_gold_doc_rank(rows, docs)
        for k, value in recall_at_k(rows, docs, recall_ks, dedupe_first=True).items():
            doc_recalls[k].append(value)
        for k, value in recall_at_k(rows, docs, recall_ks, dedupe_first=False).items():
            row_recalls[k].append(value)
    return {
        "doc_recall": {
            k: (sum(values) / len(values) if values else 0.0)
            for k, values in doc_recalls.items()
        },
        "row_recall": {
            k: (sum(values) / len(values) if values else 0.0)
            for k, values in row_recalls.items()
        },
        "first_ranks": first_ranks,
    }


def movement_counts(
    baseline_ranks: dict[str, int | None],
    candidate_ranks: dict[str, int | None],
) -> dict[str, int]:
    counts = {
        "improved": 0,
        "worsened": 0,
        "unchanged": 0,
        "newly_found": 0,
        "newly_lost": 0,
        "missing_in_both": 0,
    }
    for qid, base_rank in baseline_ranks.items():
        cand_rank = candidate_ranks[qid]
        if base_rank is None and cand_rank is None:
            counts["missing_in_both"] += 1
        elif base_rank is None:
            counts["newly_found"] += 1
        elif cand_rank is None:
            counts["newly_lost"] += 1
        elif cand_rank < base_rank:
            counts["improved"] += 1
        elif cand_rank > base_rank:
            counts["worsened"] += 1
        else:
            counts["unchanged"] += 1
    return counts


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    gold_rows = load_jsonl(Path(args.gold))
    gold_by_qid = {str(row["qid"]): row for row in gold_rows}
    default_qids = sorted(gold_by_qid)
    groups = parse_groups(args.group, default_qids)

    baseline_path = Path(args.baseline)
    baseline_pred = load_prediction(baseline_path)
    candidates = expand_candidate_specs(args.candidate, args.candidate_glob)

    rows: list[dict[str, Any]] = []
    for group_name, group_qids in groups:
        baseline = analyze_run(
            pred=baseline_pred,
            gold_by_qid=gold_by_qid,
            qids=group_qids,
            recall_ks=args.recall_k,
        )
        base_row = {
            "group": group_name,
            "label": args.baseline_label,
            "n_qids": len(group_qids),
            "improved": "",
            "worsened": "",
            "net": "",
            "missing": "",
            "delta_doc@4": "",
            "delta_row@4": "",
        }
        for k in args.recall_k:
            base_row[f"doc@{k}"] = baseline["doc_recall"][k]
            base_row[f"row@{k}"] = baseline["row_recall"][k]
        rows.append(base_row)

        for label, path in candidates:
            candidate = analyze_run(
                pred=load_prediction(path),
                gold_by_qid=gold_by_qid,
                qids=group_qids,
                recall_ks=args.recall_k,
            )
            counts = movement_counts(baseline["first_ranks"], candidate["first_ranks"])
            row: dict[str, Any] = {
                "group": group_name,
                "label": label,
                "n_qids": len(group_qids),
                "improved": counts["improved"] + counts["newly_found"],
                "worsened": counts["worsened"] + counts["newly_lost"],
                "net": (
                    counts["improved"]
                    + counts["newly_found"]
                    - counts["worsened"]
                    - counts["newly_lost"]
                ),
                "missing": counts["missing_in_both"],
                "delta_doc@4": candidate["doc_recall"].get(4, 0.0) - baseline["doc_recall"].get(4, 0.0),
                "delta_row@4": candidate["row_recall"].get(4, 0.0) - baseline["row_recall"].get(4, 0.0),
            }
            for k in args.recall_k:
                row[f"doc@{k}"] = candidate["doc_recall"][k]
                row[f"row@{k}"] = candidate["row_recall"][k]
            rows.append(row)
    return rows


def columns(recall_ks: list[int]) -> list[str]:
    return [
        "group",
        "label",
        "n_qids",
        "improved",
        "worsened",
        "net",
        "missing",
        *[f"doc@{k}" for k in recall_ks],
        *[f"row@{k}" for k in recall_ks],
        "delta_doc@4",
        "delta_row@4",
    ]


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, dict):
        return ""
    return str(value)


def print_markdown(rows: list[dict[str, Any]], cols: list[str]) -> None:
    print("| " + " | ".join(cols) + " |")
    print("| " + " | ".join("---" for _ in cols) + " |")
    for row in rows:
        print("| " + " | ".join(format_value(row.get(col, "")) for col in cols) + " |")


def print_csv(rows: list[dict[str, Any]], cols: list[str]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=cols)
    writer.writeheader()
    for row in rows:
        writer.writerow({col: row.get(col, "") for col in cols})


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    cols = columns(args.recall_k)
    if args.format == "json":
        print(json.dumps(rows, indent=2))
    elif args.format == "csv":
        print_csv(rows, cols)
    else:
        print_markdown(rows, cols)


if __name__ == "__main__":
    main()
