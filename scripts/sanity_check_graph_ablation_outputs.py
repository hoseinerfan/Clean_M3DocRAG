#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RECALL_K_VALUES = (1, 2, 4, 5, 10, 20, 50, 100)
METRIC_KEYS = (
    "reranked_top4_page_count",
    "reranked_top20_page_count",
    "reranked_top4_doc_count",
    "reranked_top20_doc_count",
)
BASELINE_CONFIG_KEYS = (
    "qid_count",
    "dense_prediction_json",
    "sparse_prediction_json",
    "gold",
    "final_top_pages",
    "per_doc_page_limit",
    "dense_weight",
    "sparse_weight",
    "restart_prob",
    "ppr_iters",
    "page_doc_edge_weight",
    "same_doc_window",
    "adjacent_page_edge_weight",
    "final_page_seed_weight",
    "final_ppr_page_weight",
    "final_ppr_doc_weight",
)
COMMON_CONFIG_KEYS = (
    "final_top_pages",
    "per_doc_page_limit",
    "dense_weight",
    "sparse_weight",
    "restart_prob",
    "ppr_iters",
    "page_doc_edge_weight",
    "same_doc_window",
    "adjacent_page_edge_weight",
    "final_page_seed_weight",
    "final_ppr_page_weight",
    "final_ppr_doc_weight",
)


@dataclass(frozen=True)
class SummaryRecord:
    path: Path
    label: str
    dataset: str
    ablation: str
    variant: str
    summary: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sanity-check Graph Page Preserve doc-doc edge and doc-seed ablation summaries."
        )
    )
    parser.add_argument(
        "--doc-doc-summary-glob",
        action="append",
        default=[],
        help="Glob for doc-doc edge ablation summary JSONs. Repeat per dataset.",
    )
    parser.add_argument(
        "--doc-seed-summary-glob",
        action="append",
        default=[],
        help="Glob for doc-seed ablation summary JSONs. Repeat per dataset.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any warning or failure is found.",
    )
    return parser.parse_args()


def compact_label(path: Path) -> str:
    name = path.name
    if name.endswith(".summary.json"):
        return name[: -len(".summary.json")]
    return path.stem


def expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(f"WARN no_matches_for_glob: {pattern}", file=sys.stderr)
            continue
        paths.extend(Path(match) for match in matches)
    return sorted(dict.fromkeys(paths))


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def parse_label(path: Path, ablation: str) -> tuple[str, str]:
    label = compact_label(path)
    marker = "_docdoc_ablation_" if ablation == "docdoc" else "_docseed_ablation_"
    if marker not in label:
        raise ValueError(f"Cannot parse {ablation} label: {label}")
    dataset, variant = label.split(marker, 1)
    return dataset, variant


def load_records(paths: list[Path], ablation: str) -> list[SummaryRecord]:
    records: list[SummaryRecord] = []
    for path in paths:
        dataset, variant = parse_label(path, ablation)
        records.append(
            SummaryRecord(
                path=path,
                label=compact_label(path),
                dataset=dataset,
                ablation=ablation,
                variant=variant,
                summary=load_summary(path),
            )
        )
    return records


def per_qid(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = summary.get("per_qid", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def rank_count(summary: dict[str, Any], rank_key: str, k: int) -> int:
    return sum(
        1
        for row in per_qid(summary)
        if row.get(rank_key) is not None and int(row[rank_key]) <= k
    )


def rate(summary: dict[str, Any], rank_key: str, k: int) -> float:
    qid_count = int(summary.get("qid_count") or 0)
    if qid_count <= 0:
        return float("nan")
    return rank_count(summary, rank_key, k) / qid_count


def fmt_rate(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.3f}"


def almost_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) or isinstance(right, (int, float)):
        try:
            return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return left == right


def qid_rank_map(summary: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    ranks: dict[str, tuple[Any, Any]] = {}
    for row in per_qid(summary):
        qid = str(row.get("qid", "")).strip()
        if not qid:
            continue
        ranks[qid] = (
            row.get("reranked_first_gold_page_rank"),
            row.get("reranked_first_gold_doc_rank"),
        )
    return ranks


def print_delta_table(title: str, records: list[SummaryRecord], baseline_variant: str) -> None:
    grouped: dict[str, dict[str, SummaryRecord]] = {}
    for record in records:
        grouped.setdefault(record.dataset, {})[record.variant] = record

    print()
    print(title)
    print(
        "| dataset | variant | page@4 count | delta page@4 | page@4 | "
        "doc@4 count | delta doc@4 | doc@4 | edge qids | mean edges |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for dataset in sorted(grouped):
        variants = grouped[dataset]
        baseline = variants.get(baseline_variant)
        if baseline is None:
            continue
        base_page4 = int(baseline.summary.get("reranked_top4_page_count") or 0)
        base_doc4 = int(baseline.summary.get("reranked_top4_doc_count") or 0)
        for variant in sorted(variants):
            record = variants[variant]
            summary = record.summary
            page4 = int(summary.get("reranked_top4_page_count") or 0)
            doc4 = int(summary.get("reranked_top4_doc_count") or 0)
            edge_qids = summary.get("doc_doc_edge_qid_count", "")
            mean_edges = summary.get("mean_doc_doc_edge_pair_count", "")
            mean_edges_text = "" if mean_edges in (None, "") else f"{float(mean_edges):.2f}"
            print(
                f"| {dataset} | {variant} | {page4} | {page4 - base_page4:+d} | "
                f"{fmt_rate(rate(summary, 'reranked_first_gold_page_rank', 4))} | "
                f"{doc4} | {doc4 - base_doc4:+d} | "
                f"{fmt_rate(rate(summary, 'reranked_first_gold_doc_rank', 4))} | "
                f"{edge_qids} | {mean_edges_text} |"
            )


def check_summary_counts(record: SummaryRecord, problems: list[str]) -> None:
    summary = record.summary
    expected = {
        "reranked_top4_page_count": rank_count(summary, "reranked_first_gold_page_rank", 4),
        "reranked_top20_page_count": rank_count(summary, "reranked_first_gold_page_rank", 20),
        "reranked_top4_doc_count": rank_count(summary, "reranked_first_gold_doc_rank", 4),
        "reranked_top20_doc_count": rank_count(summary, "reranked_first_gold_doc_rank", 20),
    }
    for key, value in expected.items():
        observed = int(summary.get(key) or 0)
        if observed != value:
            problems.append(f"{record.label}: {key}={observed}, recomputed={value}")


def check_variant_configs(records: list[SummaryRecord], problems: list[str]) -> None:
    expected_docdoc_modes = {
        "no_doc_doc": ("none", 0.0),
        "dense_sparse_agreement": ("dense_sparse_agreement", None),
        "shared_entity_title_topic": ("shared_entity_title_topic", None),
        "semantic_similarity": ("semantic_similarity", None),
        "all_doc_doc_features": ("all", None),
    }
    expected_docseed = {
        "docseed_none": ("rrf", 0.0),
        "docseed_rrf_0p25": ("rrf", 0.25),
        "docseed_rrf_0p50": ("rrf", 0.50),
        "docseed_rrf_1p00": ("rrf", 1.00),
        "docseed_graphsize_0p50": ("graph_size_adaptive", 0.50),
        "docseed_avgpage_0p50": ("avg_page_seed", 0.50),
        "docseed_avgpage_graphsize_0p50": ("avg_page_seed_graph_size", 0.50),
    }
    for record in records:
        summary = record.summary
        if record.ablation == "docdoc":
            expected = expected_docdoc_modes.get(record.variant)
            if expected is None:
                continue
            mode, weight = expected
            if summary.get("doc_doc_edge_mode") != mode:
                problems.append(
                    f"{record.label}: doc_doc_edge_mode={summary.get('doc_doc_edge_mode')} "
                    f"expected={mode}"
                )
            if weight is not None and not almost_equal(summary.get("doc_doc_edge_weight"), weight):
                problems.append(
                    f"{record.label}: doc_doc_edge_weight={summary.get('doc_doc_edge_weight')} "
                    f"expected={weight}"
                )
            if not almost_equal(summary.get("doc_seed_weight"), 0.0):
                problems.append(f"{record.label}: doc_seed_weight should be 0.0")
        else:
            expected = expected_docseed.get(record.variant)
            if expected is None:
                continue
            mode, weight = expected
            if summary.get("doc_seed_mode") != mode:
                problems.append(
                    f"{record.label}: doc_seed_mode={summary.get('doc_seed_mode')} expected={mode}"
                )
            if not almost_equal(summary.get("doc_seed_weight"), weight):
                problems.append(
                    f"{record.label}: doc_seed_weight={summary.get('doc_seed_weight')} "
                    f"expected={weight}"
                )
            if summary.get("doc_doc_edge_mode") != "none":
                problems.append(f"{record.label}: doc_doc_edge_mode should be none")


def check_baselines(
    docdoc_records: list[SummaryRecord],
    docseed_records: list[SummaryRecord],
    problems: list[str],
) -> None:
    docdoc = {
        record.dataset: record
        for record in docdoc_records
        if record.variant == "no_doc_doc"
    }
    docseed = {
        record.dataset: record
        for record in docseed_records
        if record.variant == "docseed_none"
    }
    print()
    print("Baseline Cross-Check")
    print("| dataset | config match | metrics match | per-qid ranks match |")
    print("|---|---:|---:|---:|")
    for dataset in sorted(set(docdoc) | set(docseed)):
        left = docdoc.get(dataset)
        right = docseed.get(dataset)
        if left is None or right is None:
            problems.append(f"{dataset}: missing baseline in one ablation family")
            print(f"| {dataset} | no | no | no |")
            continue
        config_match = all(
            almost_equal(left.summary.get(key), right.summary.get(key))
            for key in BASELINE_CONFIG_KEYS
        )
        metrics_match = all(
            almost_equal(left.summary.get(key), right.summary.get(key))
            for key in METRIC_KEYS
        )
        ranks_match = qid_rank_map(left.summary) == qid_rank_map(right.summary)
        if not config_match:
            problems.append(f"{dataset}: baseline config mismatch between docdoc/docseed")
        if not metrics_match:
            problems.append(f"{dataset}: baseline metric mismatch between docdoc/docseed")
        if not ranks_match:
            problems.append(f"{dataset}: baseline per-qid rank mismatch between docdoc/docseed")
        print(
            f"| {dataset} | {'yes' if config_match else 'no'} | "
            f"{'yes' if metrics_match else 'no'} | {'yes' if ranks_match else 'no'} |"
        )


def check_common_config(records: list[SummaryRecord], problems: list[str]) -> None:
    grouped: dict[tuple[str, str], list[SummaryRecord]] = {}
    for record in records:
        grouped.setdefault((record.ablation, record.dataset), []).append(record)
    for (ablation, dataset), items in grouped.items():
        if not items:
            continue
        baseline = sorted(items, key=lambda item: item.variant)[0]
        for item in items[1:]:
            for key in COMMON_CONFIG_KEYS:
                if not almost_equal(baseline.summary.get(key), item.summary.get(key)):
                    problems.append(
                        f"{ablation}/{dataset}: {key} differs between "
                        f"{baseline.label} and {item.label}"
                    )


def check_required_variants(records: list[SummaryRecord], problems: list[str]) -> None:
    required = {
        "docdoc": {
            "no_doc_doc",
            "dense_sparse_agreement",
            "shared_entity_title_topic",
            "semantic_similarity",
            "all_doc_doc_features",
        },
        "docseed": {
            "docseed_none",
            "docseed_rrf_0p25",
            "docseed_rrf_0p50",
            "docseed_rrf_1p00",
            "docseed_graphsize_0p50",
            "docseed_avgpage_0p50",
            "docseed_avgpage_graphsize_0p50",
        },
    }
    grouped: dict[tuple[str, str], set[str]] = {}
    for record in records:
        grouped.setdefault((record.ablation, record.dataset), set()).add(record.variant)
    for (ablation, dataset), variants in sorted(grouped.items()):
        missing = sorted(required[ablation] - variants)
        if missing:
            problems.append(f"{ablation}/{dataset}: missing variants {', '.join(missing)}")


def main() -> None:
    args = parse_args()
    docdoc_paths = expand_paths(args.doc_doc_summary_glob)
    docseed_paths = expand_paths(args.doc_seed_summary_glob)
    docdoc_records = load_records(docdoc_paths, "docdoc")
    docseed_records = load_records(docseed_paths, "docseed")
    records = docdoc_records + docseed_records

    problems: list[str] = []
    if args.doc_doc_summary_glob and not docdoc_records:
        problems.append("no doc-doc summary JSONs loaded")
    if args.doc_seed_summary_glob and not docseed_records:
        problems.append("no doc-seed summary JSONs loaded")
    if not records:
        problems.append("no summary JSONs loaded")
    labels = [record.label for record in records]
    if len(labels) != len(set(labels)):
        problems.append("duplicate summary labels found")
    for record in records:
        check_summary_counts(record, problems)
    check_required_variants(records, problems)
    check_variant_configs(records, problems)
    check_common_config(records, problems)
    if docdoc_records and docseed_records:
        check_baselines(docdoc_records, docseed_records, problems)

    if docdoc_records:
        print_delta_table("Doc-Doc Edge Deltas", docdoc_records, "no_doc_doc")
    if docseed_records:
        print_delta_table("Doc-Seed Deltas", docseed_records, "docseed_none")

    print()
    if problems:
        print("Sanity Check Problems")
        for problem in problems:
            print(f"- {problem}")
        if args.strict:
            raise SystemExit(1)
    else:
        print("Sanity checks passed.")


if __name__ == "__main__":
    main()
