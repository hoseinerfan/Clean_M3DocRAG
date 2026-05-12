#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare top-256 page-token selector runs from run_visual_rerank_batch.py JSONL outputs. "
            "This reports both final rerank metrics and objective-aligned pruning diagnostics."
        )
    )
    parser.add_argument(
        "--run-jsonl",
        action="append",
        required=True,
        help="Labeled batch JSONL in the form label=/abs/path/run.jsonl . The first run is the reference.",
    )
    parser.add_argument(
        "--top-example-count",
        type=int,
        default=10,
        help="How many strongest improvement/regression examples to keep per comparison.",
    )
    parser.add_argument(
        "--output-summary-json",
        help="Optional path to write the full comparison summary as JSON.",
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
    if len(runs) < 2:
        raise ValueError("Provide at least two --run-jsonl inputs.")
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
            if not qid:
                continue
            rows[qid] = row
    if not rows:
        raise ValueError(f"No qids found in {path}")
    return rows


def mean_or_none(values: list[float]) -> float | None:
    return None if not values else float(statistics.fmean(values))


def median_or_none(values: list[float]) -> float | None:
    return None if not values else float(statistics.median(values))


def summarize_run(rows_by_qid: dict[str, dict]) -> dict:
    rows = list(rows_by_qid.values())
    doc_ranks = [
        int(row["reranked_first_gold_doc_rank"])
        for row in rows
        if row.get("reranked_first_gold_doc_rank") is not None
    ]
    page_ranks = [
        int(row["reranked_first_gold_page_rank_any_gold_doc_page"])
        for row in rows
        if row.get("reranked_first_gold_page_rank_any_gold_doc_page") is not None
    ]
    top4_doc_count = sum(
        1
        for row in rows
        if row.get("reranked_first_gold_doc_rank") is not None
        and int(row["reranked_first_gold_doc_rank"]) <= 4
    )

    diag_rows = [
        row["token_pruning_diagnostic_summary"]
        for row in rows
        if row.get("token_pruning_diagnostic_summary", {}).get("enabled")
    ]
    mean_exact_score_loss = mean_or_none(
        [
            float(item["mean_exact_score_loss"])
            for item in diag_rows
            if item.get("mean_exact_score_loss") is not None
        ]
    )
    mean_shifted_preservation = mean_or_none(
        [
            float(item["mean_shifted_score_preservation_ratio"])
            for item in diag_rows
            if item.get("mean_shifted_score_preservation_ratio") is not None
        ]
    )
    mean_argmax_retention = mean_or_none(
        [
            float(item["mean_argmax_retention_ratio"])
            for item in diag_rows
            if item.get("mean_argmax_retention_ratio") is not None
        ]
    )
    median_argmax_retention = median_or_none(
        [
            float(item["mean_argmax_retention_ratio"])
            for item in diag_rows
            if item.get("mean_argmax_retention_ratio") is not None
        ]
    )

    return {
        "qid_count": len(rows),
        "reranked_top4_doc_count": top4_doc_count,
        "reranked_doc_rank_mean": mean_or_none([float(x) for x in doc_ranks]),
        "reranked_doc_rank_median": median_or_none([float(x) for x in doc_ranks]),
        "reranked_page_rank_mean": mean_or_none([float(x) for x in page_ranks]),
        "reranked_page_rank_median": median_or_none([float(x) for x in page_ranks]),
        "qid_with_pruning_diagnostics_count": len(diag_rows),
        "qid_mean_exact_score_loss": mean_exact_score_loss,
        "qid_mean_shifted_score_preservation_ratio": mean_shifted_preservation,
        "qid_mean_argmax_retention_ratio": mean_argmax_retention,
        "qid_median_argmax_retention_ratio": median_argmax_retention,
    }


def compare_runs(
    reference_rows: dict[str, dict],
    candidate_rows: dict[str, dict],
    *,
    top_example_count: int,
) -> dict:
    shared_qids = sorted(set(reference_rows) & set(candidate_rows))
    better_doc = 0
    worse_doc = 0
    tie_doc = 0
    better_page = 0
    worse_page = 0
    tie_page = 0
    improved_examples: list[dict] = []
    regressed_examples: list[dict] = []

    for qid in shared_qids:
        ref_row = reference_rows[qid]
        cand_row = candidate_rows[qid]
        ref_doc_rank = ref_row.get("reranked_first_gold_doc_rank")
        cand_doc_rank = cand_row.get("reranked_first_gold_doc_rank")
        ref_page_rank = ref_row.get("reranked_first_gold_page_rank_any_gold_doc_page")
        cand_page_rank = cand_row.get("reranked_first_gold_page_rank_any_gold_doc_page")

        if ref_doc_rank is not None and cand_doc_rank is not None:
            ref_doc_rank = int(ref_doc_rank)
            cand_doc_rank = int(cand_doc_rank)
            if cand_doc_rank < ref_doc_rank:
                better_doc += 1
                improved_examples.append(
                    {
                        "qid": qid,
                        "question": cand_row.get("question"),
                        "reference_doc_rank": ref_doc_rank,
                        "candidate_doc_rank": cand_doc_rank,
                        "doc_rank_delta": ref_doc_rank - cand_doc_rank,
                        "reference_page_rank": ref_page_rank,
                        "candidate_page_rank": cand_page_rank,
                        "reference_argmax_retention": ref_row.get("token_pruning_diagnostic_summary", {}).get(
                            "mean_argmax_retention_ratio"
                        ),
                        "candidate_argmax_retention": cand_row.get("token_pruning_diagnostic_summary", {}).get(
                            "mean_argmax_retention_ratio"
                        ),
                    }
                )
            elif cand_doc_rank > ref_doc_rank:
                worse_doc += 1
                regressed_examples.append(
                    {
                        "qid": qid,
                        "question": cand_row.get("question"),
                        "reference_doc_rank": ref_doc_rank,
                        "candidate_doc_rank": cand_doc_rank,
                        "doc_rank_delta": cand_doc_rank - ref_doc_rank,
                        "reference_page_rank": ref_page_rank,
                        "candidate_page_rank": cand_page_rank,
                        "reference_argmax_retention": ref_row.get("token_pruning_diagnostic_summary", {}).get(
                            "mean_argmax_retention_ratio"
                        ),
                        "candidate_argmax_retention": cand_row.get("token_pruning_diagnostic_summary", {}).get(
                            "mean_argmax_retention_ratio"
                        ),
                    }
                )
            else:
                tie_doc += 1

        if ref_page_rank is not None and cand_page_rank is not None:
            ref_page_rank = int(ref_page_rank)
            cand_page_rank = int(cand_page_rank)
            if cand_page_rank < ref_page_rank:
                better_page += 1
            elif cand_page_rank > ref_page_rank:
                worse_page += 1
            else:
                tie_page += 1

    improved_examples.sort(key=lambda item: (-int(item["doc_rank_delta"]), int(item["candidate_doc_rank"])))
    regressed_examples.sort(key=lambda item: (-int(item["doc_rank_delta"]), int(item["reference_doc_rank"])))

    return {
        "shared_qid_count": len(shared_qids),
        "candidate_better_doc_rank_count": better_doc,
        "candidate_worse_doc_rank_count": worse_doc,
        "candidate_tied_doc_rank_count": tie_doc,
        "candidate_better_page_rank_count": better_page,
        "candidate_worse_page_rank_count": worse_page,
        "candidate_tied_page_rank_count": tie_page,
        "top_improvements": improved_examples[:top_example_count],
        "top_regressions": regressed_examples[:top_example_count],
    }


def main() -> None:
    args = parse_args()
    labeled_paths = parse_labeled_paths(args.run_jsonl)
    runs = [(label, path, load_run_rows(path)) for label, path in labeled_paths]

    summary = {
        "reference_label": runs[0][0],
        "run_summaries": {},
        "pairwise_vs_reference": {},
    }

    for label, path, rows in runs:
        summary["run_summaries"][label] = {
            "input_jsonl": str(path),
            **summarize_run(rows),
        }

    reference_label, _reference_path, reference_rows = runs[0]
    for label, _path, rows in runs[1:]:
        summary["pairwise_vs_reference"][label] = compare_runs(
            reference_rows,
            rows,
            top_example_count=args.top_example_count,
        )

    if args.output_summary_json:
        output_path = Path(args.output_summary_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"reference_label: {reference_label}")
    for label in summary["run_summaries"]:
        run_summary = summary["run_summaries"][label]
        print(label)
        print(f"  reranked_top4_doc_count: {run_summary['reranked_top4_doc_count']}")
        print(f"  reranked_doc_rank_median: {run_summary['reranked_doc_rank_median']}")
        print(f"  reranked_page_rank_median: {run_summary['reranked_page_rank_median']}")
        print(f"  qid_mean_exact_score_loss: {run_summary['qid_mean_exact_score_loss']}")
        print(
            "  qid_mean_shifted_score_preservation_ratio: "
            f"{run_summary['qid_mean_shifted_score_preservation_ratio']}"
        )
        print(
            "  qid_mean_argmax_retention_ratio: "
            f"{run_summary['qid_mean_argmax_retention_ratio']}"
        )

    for label, pairwise in summary["pairwise_vs_reference"].items():
        print(label)
        print(f"  shared_qid_count: {pairwise['shared_qid_count']}")
        print(
            "  candidate_better_doc_rank_count: "
            f"{pairwise['candidate_better_doc_rank_count']}"
        )
        print(
            "  candidate_worse_doc_rank_count: "
            f"{pairwise['candidate_worse_doc_rank_count']}"
        )
        print(
            "  candidate_better_page_rank_count: "
            f"{pairwise['candidate_better_page_rank_count']}"
        )
        print(
            "  candidate_worse_page_rank_count: "
            f"{pairwise['candidate_worse_page_rank_count']}"
        )


if __name__ == "__main__":
    main()
