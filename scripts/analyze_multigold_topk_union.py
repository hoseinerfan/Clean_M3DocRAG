#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze multi-gold qids in rerank batch JSONL outputs. "
            "Supports both single-run partial top-K coverage extraction and "
            "multi-run union-of-top-K gold-doc coverage analysis."
        )
    )
    parser.add_argument(
        "--run-jsonl",
        action="append",
        required=True,
        help="Labeled run JSONL in the form label=/abs/path/run.jsonl .",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=4,
        help="Gold-doc coverage cutoff inside each run. Default: 4.",
    )
    parser.add_argument(
        "--output-summary-json",
        help="Optional path to write the full summary JSON.",
    )
    parser.add_argument(
        "--output-partial-qid-jsonl",
        help=(
            "Optional JSONL export of qids with >=2 gold docs and partial coverage "
            "in the first/reference run."
        ),
    )
    parser.add_argument(
        "--output-union-recovered-qid-jsonl",
        help=(
            "Optional JSONL export of multi-gold qids not fully covered by the first/reference run "
            "but fully covered by the union across all runs."
        ),
    )
    return parser.parse_args()


def parse_labeled_paths(items: list[str]) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for item in items:
        if "=" not in item:
            raise ValueError(f"--run-jsonl must be label=/abs/path, got {item!r}")
        label, path_str = item.split("=", 1)
        label = label.strip()
        path = Path(path_str.strip())
        if not label:
            raise ValueError(f"Missing label in {item!r}")
        if label in seen:
            raise ValueError(f"Duplicate label: {label!r}")
        if not path.exists():
            raise FileNotFoundError(path)
        seen.add(label)
        runs.append((label, path))
    return runs


def load_run(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("qid", "")).strip()
            if qid:
                rows[qid] = row
    if not rows:
        raise ValueError(f"No qids found in {path}")
    return rows


def gold_docs_hit_within_topk(row: dict, topk: int) -> list[str]:
    hits: list[str] = []
    for item in row.get("reranked_gold_doc_ranks", []):
        if int(item["rank"]) <= topk:
            hits.append(str(item["doc_id"]))
    return hits


def qid_record_for_run(row: dict, topk: int) -> dict:
    gold_doc_ids = [str(x) for x in row.get("gold_doc_ids", [])]
    hit_doc_ids = gold_docs_hit_within_topk(row, topk)
    missing_doc_ids = [doc_id for doc_id in gold_doc_ids if doc_id not in set(hit_doc_ids)]
    return {
        "qid": row["qid"],
        "question": row.get("question"),
        "gold_doc_ids": gold_doc_ids,
        "gold_doc_count": len(gold_doc_ids),
        "hit_doc_ids_at_k": hit_doc_ids,
        "hit_doc_count_at_k": len(hit_doc_ids),
        "missing_doc_ids_at_k": missing_doc_ids,
        "reranked_first_gold_doc_rank": row.get("reranked_first_gold_doc_rank"),
        "reranked_gold_doc_ranks": row.get("reranked_gold_doc_ranks", []),
    }


def summarize_single_run(rows_by_qid: dict[str, dict], topk: int) -> tuple[dict, list[dict]]:
    multi_gold_rows: list[dict] = []
    partial_rows: list[dict] = []
    full_rows: list[dict] = []
    nohit_rows: list[dict] = []

    for row in rows_by_qid.values():
        record = qid_record_for_run(row, topk)
        if record["gold_doc_count"] < 2:
            continue
        multi_gold_rows.append(record)
        if record["hit_doc_count_at_k"] == 0:
            nohit_rows.append(record)
        elif record["hit_doc_count_at_k"] < record["gold_doc_count"]:
            partial_rows.append(record)
        else:
            full_rows.append(record)

    summary = {
        "qid_count": len(rows_by_qid),
        "multi_gold_qid_count": len(multi_gold_rows),
        "multi_gold_no_hit_qid_count": len(nohit_rows),
        "multi_gold_partial_hit_qid_count": len(partial_rows),
        "multi_gold_full_hit_qid_count": len(full_rows),
    }
    partial_rows.sort(
        key=lambda item: (
            item["hit_doc_count_at_k"],
            item["gold_doc_count"],
            item["qid"],
        )
    )
    return summary, partial_rows


def summarize_union(runs: list[tuple[str, dict[str, dict]]], topk: int) -> tuple[dict, list[dict]]:
    reference_label, reference_rows = runs[0]
    shared_qids = sorted(set.intersection(*(set(rows) for _label, rows in runs)))

    union_rows: list[dict] = []
    union_recovered: list[dict] = []
    union_full = 0
    union_partial = 0
    union_none = 0

    for qid in shared_qids:
        ref_row = reference_rows[qid]
        gold_doc_ids = [str(x) for x in ref_row.get("gold_doc_ids", [])]
        if len(gold_doc_ids) < 2:
            continue

        per_run = {}
        union_hit_set: set[str] = set()
        for label, rows in runs:
            record = qid_record_for_run(rows[qid], topk)
            per_run[label] = {
                "hit_doc_ids_at_k": record["hit_doc_ids_at_k"],
                "hit_doc_count_at_k": record["hit_doc_count_at_k"],
                "reranked_first_gold_doc_rank": record["reranked_first_gold_doc_rank"],
            }
            union_hit_set.update(record["hit_doc_ids_at_k"])

        union_hit_doc_ids = [doc_id for doc_id in gold_doc_ids if doc_id in union_hit_set]
        missing_union_doc_ids = [doc_id for doc_id in gold_doc_ids if doc_id not in union_hit_set]
        reference_hit_count = per_run[reference_label]["hit_doc_count_at_k"]

        record = {
            "qid": qid,
            "question": ref_row.get("question"),
            "gold_doc_ids": gold_doc_ids,
            "gold_doc_count": len(gold_doc_ids),
            "reference_label": reference_label,
            "reference_hit_doc_count_at_k": reference_hit_count,
            "union_hit_doc_ids_at_k": union_hit_doc_ids,
            "union_hit_doc_count_at_k": len(union_hit_doc_ids),
            "union_missing_doc_ids_at_k": missing_union_doc_ids,
            "per_run": per_run,
        }
        union_rows.append(record)

        if len(union_hit_doc_ids) == 0:
            union_none += 1
        elif len(union_hit_doc_ids) < len(gold_doc_ids):
            union_partial += 1
        else:
            union_full += 1
            if reference_hit_count < len(gold_doc_ids):
                union_recovered.append(record)

    summary = {
        "shared_qid_count": len(shared_qids),
        "shared_multi_gold_qid_count": len(union_rows),
        "union_full_hit_multi_gold_qid_count": union_full,
        "union_partial_hit_multi_gold_qid_count": union_partial,
        "union_no_hit_multi_gold_qid_count": union_none,
        "reference_not_full_but_union_full_qid_count": len(union_recovered),
    }
    union_recovered.sort(
        key=lambda item: (
            item["reference_hit_doc_count_at_k"],
            -item["union_hit_doc_count_at_k"],
            item["qid"],
        )
    )
    return summary, union_recovered


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    labeled_paths = parse_labeled_paths(args.run_jsonl)
    runs = [(label, load_run(path)) for label, path in labeled_paths]

    summary = {
        "topk": args.topk,
        "reference_label": runs[0][0],
        "single_run": {},
    }

    partial_rows_by_run: dict[str, list[dict]] = {}
    for label, rows in runs:
        run_summary, partial_rows = summarize_single_run(rows, args.topk)
        summary["single_run"][label] = run_summary
        partial_rows_by_run[label] = partial_rows

    union_summary = None
    union_recovered_rows: list[dict] = []
    if len(runs) >= 2:
        union_summary, union_recovered_rows = summarize_union(runs, args.topk)
        summary["union"] = union_summary

    if args.output_summary_json:
        output_path = Path(args.output_summary_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.output_partial_qid_jsonl:
        write_jsonl(
            Path(args.output_partial_qid_jsonl),
            partial_rows_by_run[runs[0][0]],
        )

    if args.output_union_recovered_qid_jsonl and len(runs) >= 2:
        write_jsonl(Path(args.output_union_recovered_qid_jsonl), union_recovered_rows)

    print(f"reference_label: {runs[0][0]}")
    for label, run_summary in summary["single_run"].items():
        print(label)
        print(f"  multi_gold_qid_count: {run_summary['multi_gold_qid_count']}")
        print(f"  multi_gold_partial_hit_qid_count: {run_summary['multi_gold_partial_hit_qid_count']}")
        print(f"  multi_gold_full_hit_qid_count: {run_summary['multi_gold_full_hit_qid_count']}")
        print(f"  multi_gold_no_hit_qid_count: {run_summary['multi_gold_no_hit_qid_count']}")

    if union_summary is not None:
        print("union")
        print(f"  shared_multi_gold_qid_count: {union_summary['shared_multi_gold_qid_count']}")
        print(
            "  reference_not_full_but_union_full_qid_count: "
            f"{union_summary['reference_not_full_but_union_full_qid_count']}"
        )


if __name__ == "__main__":
    main()
