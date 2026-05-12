#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare budget-matched multi-gold shortlist strategies. "
            "Typical use: reference top-8/top-12 from one run versus union of 2 or 3 "
            "top-4 lists from different fixed global variants."
        )
    )
    parser.add_argument(
        "--run-jsonl",
        action="append",
        required=True,
        help="Labeled run JSONL in the form label=/abs/path/run.jsonl .",
    )
    parser.add_argument(
        "--reference-label",
        help="Optional reference run label. Defaults to the first --run-jsonl label.",
    )
    parser.add_argument(
        "--base-topk",
        type=int,
        default=4,
        help="Per-run cutoff used before unioning docs across runs. Default: 4.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        action="append",
        default=None,
        help="Total shortlist budgets to compare. Default: 8 and 12.",
    )
    parser.add_argument(
        "--restrict-to-reference-partial-at",
        type=int,
        default=4,
        help=(
            "If > 0, restrict analysis to qids where the reference run has >=1 but not all "
            "gold docs within this top-K. Default: 4."
        ),
    )
    parser.add_argument(
        "--output-summary-json",
        help="Optional path to save the full summary JSON.",
    )
    return parser.parse_args()


def parse_labeled_paths(items: list[str]) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    seen_labels: set[str] = set()
    for item in items:
        if "=" not in item:
            raise ValueError(f"--run-jsonl must be label=/abs/path, got {item!r}")
        label, path_str = item.split("=", 1)
        label = label.strip()
        path = Path(path_str.strip())
        if not label:
            raise ValueError(f"Missing label in {item!r}")
        if label in seen_labels:
            raise ValueError(f"Duplicate label: {label!r}")
        if not path.exists():
            raise FileNotFoundError(path)
        seen_labels.add(label)
        runs.append((label, path))
    if len(runs) < 2:
        raise ValueError("Provide at least two --run-jsonl inputs.")
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


def topk_hit_doc_ids(row: dict, topk: int) -> list[str]:
    hits: list[str] = []
    for item in row.get("reranked_gold_doc_ranks", []):
        if int(item["rank"]) <= topk:
            hits.append(str(item["doc_id"]))
    return hits


def full_hit(hit_doc_ids: list[str], gold_doc_ids: list[str]) -> bool:
    return len(hit_doc_ids) == len(gold_doc_ids) and len(gold_doc_ids) > 0


def reference_filter_qids(
    *,
    reference_rows: dict[str, dict],
    shared_qids: list[str],
    restrict_to_reference_partial_at: int,
) -> list[str]:
    kept: list[str] = []
    for qid in shared_qids:
        row = reference_rows[qid]
        gold_doc_ids = [str(x) for x in row.get("gold_doc_ids", [])]
        if len(gold_doc_ids) < 2:
            continue
        if restrict_to_reference_partial_at > 0:
            hits = topk_hit_doc_ids(row, restrict_to_reference_partial_at)
            if not (0 < len(hits) < len(gold_doc_ids)):
                continue
        kept.append(qid)
    return kept


def evaluate_single_run(
    *,
    rows_by_qid: dict[str, dict],
    qids: list[str],
    topk: int,
) -> dict:
    full_qids: list[str] = []
    for qid in qids:
        row = rows_by_qid[qid]
        gold_doc_ids = [str(x) for x in row.get("gold_doc_ids", [])]
        hits = topk_hit_doc_ids(row, topk)
        if full_hit(hits, gold_doc_ids):
            full_qids.append(qid)
    return {
        "topk": topk,
        "full_hit_qid_count": len(full_qids),
        "full_hit_qids": full_qids,
    }


def evaluate_union_combo(
    *,
    combo: tuple[str, ...],
    run_rows: dict[str, dict[str, dict]],
    qids: list[str],
    base_topk: int,
) -> dict:
    full_qids: list[str] = []
    for qid in qids:
        gold_doc_ids = [str(x) for x in run_rows[combo[0]][qid].get("gold_doc_ids", [])]
        union_hit_set: set[str] = set()
        for label in combo:
            union_hit_set.update(topk_hit_doc_ids(run_rows[label][qid], base_topk))
        union_hit_doc_ids = [doc_id for doc_id in gold_doc_ids if doc_id in union_hit_set]
        if full_hit(union_hit_doc_ids, gold_doc_ids):
            full_qids.append(qid)
    return {
        "labels": list(combo),
        "full_hit_qid_count": len(full_qids),
        "full_hit_qids": full_qids,
    }


def best_single_run(
    *,
    run_rows: dict[str, dict[str, dict]],
    qids: list[str],
    topk: int,
) -> dict:
    candidates: list[dict] = []
    for label, rows in run_rows.items():
        result = evaluate_single_run(rows_by_qid=rows, qids=qids, topk=topk)
        result["label"] = label
        candidates.append(result)
    candidates.sort(key=lambda item: (-int(item["full_hit_qid_count"]), item["label"]))
    return candidates[0]


def best_union_of_size(
    *,
    run_rows: dict[str, dict[str, dict]],
    qids: list[str],
    base_topk: int,
    union_size: int,
) -> dict:
    labels = sorted(run_rows)
    best_result: dict | None = None
    for combo in itertools.combinations(labels, union_size):
        result = evaluate_union_combo(
            combo=combo,
            run_rows=run_rows,
            qids=qids,
            base_topk=base_topk,
        )
        if best_result is None:
            best_result = result
            continue
        if int(result["full_hit_qid_count"]) > int(best_result["full_hit_qid_count"]):
            best_result = result
            continue
        if (
            int(result["full_hit_qid_count"]) == int(best_result["full_hit_qid_count"])
            and tuple(result["labels"]) < tuple(best_result["labels"])
        ):
            best_result = result
    if best_result is None:
        raise ValueError(f"No union combinations of size {union_size}")
    return best_result


def main() -> None:
    args = parse_args()
    labeled_paths = parse_labeled_paths(args.run_jsonl)
    run_rows = {label: load_run(path) for label, path in labeled_paths}

    reference_label = args.reference_label or labeled_paths[0][0]
    if reference_label not in run_rows:
        raise KeyError(f"Unknown --reference-label: {reference_label!r}")

    shared_qids = sorted(set.intersection(*(set(rows) for rows in run_rows.values())))
    analysis_qids = reference_filter_qids(
        reference_rows=run_rows[reference_label],
        shared_qids=shared_qids,
        restrict_to_reference_partial_at=args.restrict_to_reference_partial_at,
    )

    raw_budgets = args.budget if args.budget is not None else [8, 12]
    budgets = sorted({int(value) for value in raw_budgets if int(value) > 0})
    comparisons: list[dict] = []
    for budget in budgets:
        if budget % int(args.base_topk) != 0:
            raise ValueError(
                f"Budget {budget} is not divisible by base_topk={args.base_topk}; "
                "union size would be ambiguous."
            )
        union_size = budget // int(args.base_topk)
        if union_size < 2:
            raise ValueError(
                f"Budget {budget} with base_topk={args.base_topk} gives union_size={union_size}; "
                "use a larger budget."
            )

        reference_single = evaluate_single_run(
            rows_by_qid=run_rows[reference_label],
            qids=analysis_qids,
            topk=budget,
        )
        best_single = best_single_run(
            run_rows=run_rows,
            qids=analysis_qids,
            topk=budget,
        )
        best_union = best_union_of_size(
            run_rows=run_rows,
            qids=analysis_qids,
            base_topk=args.base_topk,
            union_size=union_size,
        )

        best_union_only_qids = sorted(
            set(best_union["full_hit_qids"]) - set(reference_single["full_hit_qids"])
        )
        comparisons.append(
            {
                "budget": budget,
                "base_topk": args.base_topk,
                "union_size": union_size,
                "reference_single": {
                    "label": reference_label,
                    **reference_single,
                },
                "best_single": best_single,
                "best_union": best_union,
                "best_union_only_vs_reference_single_qids": best_union_only_qids,
            }
        )

    summary = {
        "reference_label": reference_label,
        "base_topk": args.base_topk,
        "restrict_to_reference_partial_at": args.restrict_to_reference_partial_at,
        "shared_qid_count": len(shared_qids),
        "analysis_qid_count": len(analysis_qids),
        "analysis_qids": analysis_qids,
        "comparisons": comparisons,
    }

    if args.output_summary_json:
        output_path = Path(args.output_summary_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"reference_label: {reference_label}")
    print(f"analysis_qid_count: {len(analysis_qids)}")
    for item in comparisons:
        print(f"budget={item['budget']}")
        print(
            "  reference_single_full_hit_qid_count: "
            f"{item['reference_single']['full_hit_qid_count']}"
        )
        print(
            "  best_single: "
            f"{item['best_single']['label']} -> {item['best_single']['full_hit_qid_count']}"
        )
        print(
            "  best_union: "
            f"{item['best_union']['labels']} -> {item['best_union']['full_hit_qid_count']}"
        )
        print(
            "  best_union_only_vs_reference_single_qid_count: "
            f"{len(item['best_union_only_vs_reference_single_qids'])}"
        )


if __name__ == "__main__":
    main()
