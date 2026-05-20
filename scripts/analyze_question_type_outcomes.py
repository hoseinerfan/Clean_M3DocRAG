#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize successful and unsuccessful qids by question type from "
            "run_visual_rerank_batch.py JSONL outputs, with optional coarse-vs-final "
            "diagnostics when those fields are present."
        )
    )
    parser.add_argument(
        "--run-jsonl",
        action="append",
        required=True,
        help="Run JSONL in the form label=/abs/path/run.jsonl . The first run is the reference when comparing runs.",
    )
    parser.add_argument(
        "--success-rank-threshold",
        type=int,
        default=4,
        help="Treat reranked/coarse first-gold-doc rank <= this value as success. Defaults to 4.",
    )
    parser.add_argument(
        "--output-summary-json",
        help="Optional path to write the full summary JSON.",
    )
    return parser.parse_args()


def parse_labeled_paths(items: list[str]) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    seen_labels: set[str] = set()
    for item in items:
        if "=" not in item:
            raise ValueError(f"--run-jsonl must be label=/abs/path, got: {item!r}")
        label, raw_path = item.split("=", 1)
        label = label.strip()
        path = Path(raw_path.strip())
        if not label:
            raise ValueError(f"Missing label in --run-jsonl={item!r}")
        if label in seen_labels:
            raise ValueError(f"Duplicate run label: {label!r}")
        if not path.exists():
            raise FileNotFoundError(path)
        seen_labels.add(label)
        runs.append((label, path))
    return runs


def load_run_rows(path: Path) -> dict[str, dict]:
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


def median_or_none(values: list[int]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.fmean(values))


def question_type_of(row: dict) -> str:
    return str(row.get("question_type", "UNKNOWN")).strip() or "UNKNOWN"


def summarize_single_run(rows_by_qid: dict[str, dict], success_rank_threshold: int) -> dict:
    per_qtype: dict[str, dict] = {}
    has_coarse = False

    for qid, row in rows_by_qid.items():
        qtype = question_type_of(row)
        bucket = per_qtype.setdefault(
            qtype,
            {
                "question_type": qtype,
                "total_qids": 0,
                "success_count": 0,
                "failure_count": 0,
                "success_qids": [],
                "failure_qids": [],
                "reranked_doc_ranks": [],
                "baseline_doc_ranks": [],
                "coarse_success_count": 0,
                "coarse_failure_count": 0,
                "coarse_success_qids": [],
                "coarse_failure_qids": [],
                "coarse_doc_ranks": [],
                "final_better_than_coarse_count": 0,
                "final_worse_than_coarse_count": 0,
                "final_same_as_coarse_count": 0,
                "coarse_minus_final_rank_deltas": [],
            },
        )

        bucket["total_qids"] += 1
        reranked_rank = row.get("reranked_first_gold_doc_rank")
        if reranked_rank is not None:
            reranked_rank = int(reranked_rank)
            bucket["reranked_doc_ranks"].append(reranked_rank)
            if reranked_rank <= success_rank_threshold:
                bucket["success_count"] += 1
                bucket["success_qids"].append(qid)
            else:
                bucket["failure_count"] += 1
                bucket["failure_qids"].append(qid)

        baseline_rank = row.get("baseline_first_gold_doc_rank")
        if baseline_rank is not None:
            bucket["baseline_doc_ranks"].append(int(baseline_rank))

        coarse_rank = row.get("coarse_pre_exact_first_gold_doc_rank")
        if coarse_rank is not None:
            has_coarse = True
            coarse_rank = int(coarse_rank)
            bucket["coarse_doc_ranks"].append(coarse_rank)
            if coarse_rank <= success_rank_threshold:
                bucket["coarse_success_count"] += 1
                bucket["coarse_success_qids"].append(qid)
            else:
                bucket["coarse_failure_count"] += 1
                bucket["coarse_failure_qids"].append(qid)
            if reranked_rank is not None:
                bucket["coarse_minus_final_rank_deltas"].append(float(coarse_rank - reranked_rank))
                if reranked_rank < coarse_rank:
                    bucket["final_better_than_coarse_count"] += 1
                elif reranked_rank > coarse_rank:
                    bucket["final_worse_than_coarse_count"] += 1
                else:
                    bucket["final_same_as_coarse_count"] += 1

    qtype_rows: list[dict] = []
    for qtype in sorted(per_qtype):
        bucket = per_qtype[qtype]
        total_qids = int(bucket["total_qids"])
        success_count = int(bucket["success_count"])
        failure_count = int(bucket["failure_count"])
        row = {
            "question_type": qtype,
            "total_qids": total_qids,
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": (float(success_count) / total_qids) if total_qids > 0 else None,
            "median_reranked_doc_rank": median_or_none(bucket["reranked_doc_ranks"]),
            "median_baseline_doc_rank": median_or_none(bucket["baseline_doc_ranks"]),
            "success_qids": bucket["success_qids"],
            "failure_qids": bucket["failure_qids"],
        }
        if has_coarse:
            row.update(
                {
                    "coarse_success_count": int(bucket["coarse_success_count"]),
                    "coarse_failure_count": int(bucket["coarse_failure_count"]),
                    "coarse_success_rate": (
                        float(bucket["coarse_success_count"]) / total_qids
                        if total_qids > 0
                        else None
                    ),
                    "median_coarse_doc_rank": median_or_none(bucket["coarse_doc_ranks"]),
                    "final_better_than_coarse_count": int(bucket["final_better_than_coarse_count"]),
                    "final_worse_than_coarse_count": int(bucket["final_worse_than_coarse_count"]),
                    "final_same_as_coarse_count": int(bucket["final_same_as_coarse_count"]),
                    "mean_coarse_minus_final_rank_delta": mean_or_none(
                        bucket["coarse_minus_final_rank_deltas"]
                    ),
                    "coarse_success_qids": bucket["coarse_success_qids"],
                    "coarse_failure_qids": bucket["coarse_failure_qids"],
                }
            )
        qtype_rows.append(row)

    qtype_rows.sort(
        key=lambda item: (
            item["success_rate"] if item["success_rate"] is not None else -1.0,
            -item["total_qids"],
            item["question_type"],
        )
    )

    overall_success = sum(int(item["success_count"]) for item in qtype_rows)
    overall_total = sum(int(item["total_qids"]) for item in qtype_rows)
    summary = {
        "qid_count": len(rows_by_qid),
        "success_rank_threshold": int(success_rank_threshold),
        "success_count": overall_success,
        "failure_count": overall_total - overall_success,
        "success_rate": (float(overall_success) / overall_total) if overall_total > 0 else None,
        "has_coarse_pre_exact": has_coarse,
        "question_type_summary": qtype_rows,
    }
    if has_coarse:
        overall_coarse_success = sum(int(item["coarse_success_count"]) for item in qtype_rows)
        summary.update(
            {
                "coarse_success_count": overall_coarse_success,
                "coarse_failure_count": overall_total - overall_coarse_success,
                "coarse_success_rate": (
                    float(overall_coarse_success) / overall_total if overall_total > 0 else None
                ),
                "final_better_than_coarse_count": sum(
                    int(item["final_better_than_coarse_count"]) for item in qtype_rows
                ),
                "final_worse_than_coarse_count": sum(
                    int(item["final_worse_than_coarse_count"]) for item in qtype_rows
                ),
                "final_same_as_coarse_count": sum(
                    int(item["final_same_as_coarse_count"]) for item in qtype_rows
                ),
            }
        )
    return summary


def compare_against_reference(
    reference_rows: dict[str, dict],
    candidate_rows: dict[str, dict],
    success_rank_threshold: int,
) -> list[dict]:
    qtypes = sorted(
        {
            question_type_of(row)
            for row in reference_rows.values()
        }
        | {
            question_type_of(row)
            for row in candidate_rows.values()
        }
    )
    result: list[dict] = []
    for qtype in qtypes:
        shared_qids = sorted(
            qid
            for qid in set(reference_rows) & set(candidate_rows)
            if question_type_of(reference_rows[qid]) == qtype
            and question_type_of(candidate_rows[qid]) == qtype
        )
        if not shared_qids:
            continue
        ref_success = 0
        cand_success = 0
        improved = 0
        worsened = 0
        same = 0
        for qid in shared_qids:
            ref_rank = reference_rows[qid].get("reranked_first_gold_doc_rank")
            cand_rank = candidate_rows[qid].get("reranked_first_gold_doc_rank")
            if ref_rank is None or cand_rank is None:
                continue
            ref_rank = int(ref_rank)
            cand_rank = int(cand_rank)
            if ref_rank <= success_rank_threshold:
                ref_success += 1
            if cand_rank <= success_rank_threshold:
                cand_success += 1
            if cand_rank < ref_rank:
                improved += 1
            elif cand_rank > ref_rank:
                worsened += 1
            else:
                same += 1
        result.append(
            {
                "question_type": qtype,
                "shared_qid_count": len(shared_qids),
                "reference_success_count": ref_success,
                "candidate_success_count": cand_success,
                "candidate_minus_reference_success_count": cand_success - ref_success,
                "candidate_better_rank_count": improved,
                "candidate_worse_rank_count": worsened,
                "candidate_same_rank_count": same,
            }
        )
    result.sort(
        key=lambda item: (
            item["candidate_minus_reference_success_count"],
            item["candidate_better_rank_count"] - item["candidate_worse_rank_count"],
            -item["shared_qid_count"],
            item["question_type"],
        )
    )
    return result


def print_run_summary(label: str, summary: dict) -> None:
    print()
    print(label)
    print("  qid_count:", summary["qid_count"])
    print("  success_count:", summary["success_count"])
    print("  failure_count:", summary["failure_count"])
    print("  success_rate:", summary["success_rate"])
    if summary.get("has_coarse_pre_exact"):
        print("  coarse_success_count:", summary.get("coarse_success_count"))
        print("  coarse_failure_count:", summary.get("coarse_failure_count"))
        print("  coarse_success_rate:", summary.get("coarse_success_rate"))
        print("  final_better_than_coarse_count:", summary.get("final_better_than_coarse_count"))
        print("  final_worse_than_coarse_count:", summary.get("final_worse_than_coarse_count"))
        print("  final_same_as_coarse_count:", summary.get("final_same_as_coarse_count"))
    print("  by_question_type:")
    for item in summary["question_type_summary"]:
        base = (
            f"    {item['question_type']}: total={item['total_qids']} "
            f"success={item['success_count']} fail={item['failure_count']} "
            f"rate={item['success_rate']:.4f}"
        )
        if summary.get("has_coarse_pre_exact"):
            coarse_part = (
                f" coarse_success={item['coarse_success_count']} "
                f"coarse_rate={item['coarse_success_rate']:.4f} "
                f"final_better={item['final_better_than_coarse_count']} "
                f"final_worse={item['final_worse_than_coarse_count']} "
                f"same={item['final_same_as_coarse_count']}"
            )
            print(base + coarse_part)
        else:
            print(base)


def main() -> None:
    args = parse_args()
    runs = parse_labeled_paths(args.run_jsonl)

    loaded_runs = [(label, load_run_rows(path)) for label, path in runs]
    summaries = []
    for label, rows_by_qid in loaded_runs:
        summary = summarize_single_run(
            rows_by_qid=rows_by_qid,
            success_rank_threshold=int(args.success_rank_threshold),
        )
        summaries.append({"label": label, **summary})
        print_run_summary(label=label, summary=summary)

    comparisons: list[dict] = []
    if len(loaded_runs) >= 2:
        reference_label, reference_rows = loaded_runs[0]
        for candidate_label, candidate_rows in loaded_runs[1:]:
            comparison_rows = compare_against_reference(
                reference_rows=reference_rows,
                candidate_rows=candidate_rows,
                success_rank_threshold=int(args.success_rank_threshold),
            )
            comparison_payload = {
                "reference_label": reference_label,
                "candidate_label": candidate_label,
                "question_type_comparison": comparison_rows,
            }
            comparisons.append(comparison_payload)
            print()
            print(f"{candidate_label} vs {reference_label}")
            for item in comparison_rows:
                print(
                    "    "
                    f"{item['question_type']}: shared={item['shared_qid_count']} "
                    f"delta_success={item['candidate_minus_reference_success_count']} "
                    f"better={item['candidate_better_rank_count']} "
                    f"worse={item['candidate_worse_rank_count']} "
                    f"same={item['candidate_same_rank_count']}"
                )

    if args.output_summary_json:
        payload = {
            "success_rank_threshold": int(args.success_rank_threshold),
            "runs": summaries,
            "comparisons": comparisons,
        }
        out_path = Path(args.output_summary_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
