#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare run_visual_rerank_batch.py JSONL outputs on quality, runtime, "
            "retained-token ratios, and best-effort memory diagnostics."
        )
    )
    parser.add_argument(
        "--run-jsonl",
        action="append",
        required=True,
        help="Labeled batch JSONL in the form label=/abs/path/run.jsonl . The first run is the reference.",
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


def weighted_mean(values: list[tuple[float, int]]) -> float | None:
    numerator = 0.0
    denominator = 0
    for value, weight in values:
        if weight <= 0:
            continue
        numerator += float(value) * int(weight)
        denominator += int(weight)
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def summarize_run(rows_by_qid: dict[str, dict]) -> dict:
    rows = list(rows_by_qid.values())
    reranked_top4_doc_count = sum(
        1
        for row in rows
        if row.get("reranked_first_gold_doc_rank") is not None
        and int(row["reranked_first_gold_doc_rank"]) <= 4
    )

    efficiency_rows = [
        row.get("efficiency_diagnostic_summary", {})
        for row in rows
        if row.get("efficiency_diagnostic_summary", {}).get("enabled")
    ]
    pruning_rows = [
        row.get("token_pruning_diagnostic_summary", {})
        for row in rows
        if row.get("token_pruning_diagnostic_summary", {}).get("enabled")
    ]

    total_candidate_pages = sum(int(item.get("candidate_page_count", 0)) for item in efficiency_rows)
    total_page_feature_wall_time_sec = sum(
        float(item.get("page_feature_wall_time_sec", 0.0))
        for item in efficiency_rows
        if item.get("page_feature_wall_time_sec") is not None
    )
    total_qid_wall_time_sec = sum(
        float(item.get("qid_wall_time_sec", 0.0))
        for item in efficiency_rows
        if item.get("qid_wall_time_sec") is not None
    )

    weighted_selected_token_fraction = weighted_mean(
        [
            (float(item["mean_selected_token_fraction"]), int(item.get("page_count", 0)))
            for item in pruning_rows
            if item.get("mean_selected_token_fraction") is not None
        ]
    )
    weighted_candidate_token_fraction = weighted_mean(
        [
            (float(item["mean_candidate_token_fraction"]), int(item.get("page_count", 0)))
            for item in pruning_rows
            if item.get("mean_candidate_token_fraction") is not None
        ]
    )
    weighted_selected_token_count = weighted_mean(
        [
            (float(item["mean_selected_token_count"]), int(item.get("page_count", 0)))
            for item in pruning_rows
            if item.get("mean_selected_token_count") is not None
        ]
    )
    weighted_full_token_count = weighted_mean(
        [
            (float(item["mean_full_token_count"]), int(item.get("page_count", 0)))
            for item in pruning_rows
            if item.get("mean_full_token_count") is not None
        ]
    )
    weighted_exact_score_loss = weighted_mean(
        [
            (float(item["mean_exact_score_loss"]), int(item.get("page_count", 0)))
            for item in pruning_rows
            if item.get("mean_exact_score_loss") is not None
        ]
    )
    weighted_shifted_preservation = weighted_mean(
        [
            (float(item["mean_shifted_score_preservation_ratio"]), int(item.get("page_count", 0)))
            for item in pruning_rows
            if item.get("mean_shifted_score_preservation_ratio") is not None
        ]
    )

    return {
        "qid_count": len(rows),
        "reranked_top4_doc_count": reranked_top4_doc_count,
        "qid_with_efficiency_diagnostics_count": len(efficiency_rows),
        "qid_with_pruning_diagnostics_count": len(pruning_rows),
        "total_qid_wall_time_sec": total_qid_wall_time_sec,
        "mean_qid_wall_time_sec": mean_or_none(
            [
                float(item["qid_wall_time_sec"])
                for item in efficiency_rows
                if item.get("qid_wall_time_sec") is not None
            ]
        ),
        "median_qid_wall_time_sec": median_or_none(
            [
                float(item["qid_wall_time_sec"])
                for item in efficiency_rows
                if item.get("qid_wall_time_sec") is not None
            ]
        ),
        "mean_page_feature_wall_time_sec": mean_or_none(
            [
                float(item["page_feature_wall_time_sec"])
                for item in efficiency_rows
                if item.get("page_feature_wall_time_sec") is not None
            ]
        ),
        "mean_final_ranking_wall_time_sec": mean_or_none(
            [
                float(item["final_ranking_wall_time_sec"])
                for item in efficiency_rows
                if item.get("final_ranking_wall_time_sec") is not None
            ]
        ),
        "mean_candidate_page_count": mean_or_none(
            [
                float(item["candidate_page_count"])
                for item in efficiency_rows
                if item.get("candidate_page_count") is not None
            ]
        ),
        "total_candidate_page_count": total_candidate_pages,
        "page_feature_pages_per_second": (
            None
            if total_page_feature_wall_time_sec <= 0.0
            else float(total_candidate_pages / total_page_feature_wall_time_sec)
        ),
        "mean_cuda_peak_memory_allocated_mb": mean_or_none(
            [
                float(item["cuda_peak_memory_allocated_mb"])
                for item in efficiency_rows
                if item.get("cuda_peak_memory_allocated_mb") is not None
            ]
        ),
        "max_cuda_peak_memory_allocated_mb": (
            None
            if not [
                float(item["cuda_peak_memory_allocated_mb"])
                for item in efficiency_rows
                if item.get("cuda_peak_memory_allocated_mb") is not None
            ]
            else float(
                max(
                    float(item["cuda_peak_memory_allocated_mb"])
                    for item in efficiency_rows
                    if item.get("cuda_peak_memory_allocated_mb") is not None
                )
            )
        ),
        "mean_selected_token_count": weighted_selected_token_count,
        "mean_full_token_count": weighted_full_token_count,
        "mean_selected_token_fraction": weighted_selected_token_fraction,
        "mean_candidate_token_fraction": weighted_candidate_token_fraction,
        "mean_estimated_selected_compute_reduction_ratio": (
            None
            if weighted_selected_token_fraction is None
            else float(max(0.0, 1.0 - weighted_selected_token_fraction))
        ),
        "mean_exact_score_loss": weighted_exact_score_loss,
        "mean_shifted_score_preservation_ratio": weighted_shifted_preservation,
    }


def compare_runs(reference_rows: dict[str, dict], candidate_rows: dict[str, dict]) -> dict:
    shared_qids = sorted(set(reference_rows) & set(candidate_rows))
    reference_top4 = 0
    candidate_top4 = 0
    qid_wall_time_ratios: list[float] = []
    page_feature_wall_time_ratios: list[float] = []
    selected_token_fraction_deltas: list[float] = []
    exact_score_loss_deltas: list[float] = []

    for qid in shared_qids:
        ref_row = reference_rows[qid]
        cand_row = candidate_rows[qid]

        ref_doc_rank = ref_row.get("reranked_first_gold_doc_rank")
        cand_doc_rank = cand_row.get("reranked_first_gold_doc_rank")
        if ref_doc_rank is not None and int(ref_doc_rank) <= 4:
            reference_top4 += 1
        if cand_doc_rank is not None and int(cand_doc_rank) <= 4:
            candidate_top4 += 1

        ref_eff = ref_row.get("efficiency_diagnostic_summary", {})
        cand_eff = cand_row.get("efficiency_diagnostic_summary", {})
        ref_qid_time = ref_eff.get("qid_wall_time_sec")
        cand_qid_time = cand_eff.get("qid_wall_time_sec")
        if ref_qid_time is not None and cand_qid_time is not None and float(ref_qid_time) > 0.0:
            qid_wall_time_ratios.append(float(cand_qid_time) / float(ref_qid_time))

        ref_page_time = ref_eff.get("page_feature_wall_time_sec")
        cand_page_time = cand_eff.get("page_feature_wall_time_sec")
        if ref_page_time is not None and cand_page_time is not None and float(ref_page_time) > 0.0:
            page_feature_wall_time_ratios.append(float(cand_page_time) / float(ref_page_time))

        ref_prune = ref_row.get("token_pruning_diagnostic_summary", {})
        cand_prune = cand_row.get("token_pruning_diagnostic_summary", {})
        ref_selected_fraction = ref_prune.get("mean_selected_token_fraction")
        cand_selected_fraction = cand_prune.get("mean_selected_token_fraction")
        if ref_selected_fraction is not None and cand_selected_fraction is not None:
            selected_token_fraction_deltas.append(
                float(cand_selected_fraction) - float(ref_selected_fraction)
            )

        ref_exact_loss = ref_prune.get("mean_exact_score_loss")
        cand_exact_loss = cand_prune.get("mean_exact_score_loss")
        if ref_exact_loss is not None and cand_exact_loss is not None:
            exact_score_loss_deltas.append(float(cand_exact_loss) - float(ref_exact_loss))

    return {
        "shared_qid_count": len(shared_qids),
        "reference_top4_doc_count_on_shared_qids": reference_top4,
        "candidate_top4_doc_count_on_shared_qids": candidate_top4,
        "candidate_minus_reference_top4_doc_count": candidate_top4 - reference_top4,
        "mean_qid_wall_time_ratio_vs_reference": mean_or_none(qid_wall_time_ratios),
        "median_qid_wall_time_ratio_vs_reference": median_or_none(qid_wall_time_ratios),
        "mean_page_feature_wall_time_ratio_vs_reference": mean_or_none(page_feature_wall_time_ratios),
        "median_page_feature_wall_time_ratio_vs_reference": median_or_none(page_feature_wall_time_ratios),
        "mean_selected_token_fraction_delta_vs_reference": mean_or_none(selected_token_fraction_deltas),
        "mean_exact_score_loss_delta_vs_reference": mean_or_none(exact_score_loss_deltas),
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
        summary["pairwise_vs_reference"][label] = compare_runs(reference_rows, rows)

    if args.output_summary_json:
        output_path = Path(args.output_summary_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"reference_label: {reference_label}")
    for label, run_summary in summary["run_summaries"].items():
        print(label)
        print(f"  reranked_top4_doc_count: {run_summary['reranked_top4_doc_count']}")
        print(f"  mean_qid_wall_time_sec: {run_summary['mean_qid_wall_time_sec']}")
        print(f"  mean_page_feature_wall_time_sec: {run_summary['mean_page_feature_wall_time_sec']}")
        print(f"  page_feature_pages_per_second: {run_summary['page_feature_pages_per_second']}")
        print(f"  mean_selected_token_count: {run_summary['mean_selected_token_count']}")
        print(f"  mean_selected_token_fraction: {run_summary['mean_selected_token_fraction']}")
        print(
            "  mean_estimated_selected_compute_reduction_ratio: "
            f"{run_summary['mean_estimated_selected_compute_reduction_ratio']}"
        )
        print(f"  mean_exact_score_loss: {run_summary['mean_exact_score_loss']}")
        print(
            "  mean_shifted_score_preservation_ratio: "
            f"{run_summary['mean_shifted_score_preservation_ratio']}"
        )
        print(f"  mean_cuda_peak_memory_allocated_mb: {run_summary['mean_cuda_peak_memory_allocated_mb']}")

    for label, pairwise in summary["pairwise_vs_reference"].items():
        print(label)
        print(f"  shared_qid_count: {pairwise['shared_qid_count']}")
        print(
            "  candidate_minus_reference_top4_doc_count: "
            f"{pairwise['candidate_minus_reference_top4_doc_count']}"
        )
        print(
            "  mean_qid_wall_time_ratio_vs_reference: "
            f"{pairwise['mean_qid_wall_time_ratio_vs_reference']}"
        )
        print(
            "  mean_page_feature_wall_time_ratio_vs_reference: "
            f"{pairwise['mean_page_feature_wall_time_ratio_vs_reference']}"
        )
        print(
            "  mean_selected_token_fraction_delta_vs_reference: "
            f"{pairwise['mean_selected_token_fraction_delta_vs_reference']}"
        )
        print(
            "  mean_exact_score_loss_delta_vs_reference: "
            f"{pairwise['mean_exact_score_loss_delta_vs_reference']}"
        )


if __name__ == "__main__":
    main()
