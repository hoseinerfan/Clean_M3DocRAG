#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


VARIANT_ORDER = [
    "control",
    "no_doc_rank_cap",
    "require_topk_doc",
    "relax_overlap",
    "relax_support",
    "combined_relaxed",
]
VARIANT_LABELS = {
    "control": "control",
    "no_doc_rank_cap": "no doc-rank cap",
    "require_topk_doc": "true top-k doc",
    "relax_overlap": "overlap -1",
    "relax_support": "one support vote",
    "combined_relaxed": "combined relaxed",
}
CSV_COLUMNS = [
    "dataset",
    "variant",
    "hit_k",
    "min_page_overlap",
    "min_support_page_votes",
    "promoted_doc_max_base_rank",
    "require_promoted_doc_in_base_topk",
    "accepted",
    "base_page_hit_at_k",
    "candidate_page_hit_at_k",
    "page_hit_at_k",
    "delta_page_vs_control",
    "recovered",
    "lost",
    "page_net",
    "base_doc_hit_at_k",
    "doc_hit_at_k",
    "delta_doc_vs_control",
    "doc_net",
    "overlap_rejects",
    "doc_rejects",
    "topk_doc_rejects",
    "support_rejects",
    "body_rejects",
    "summary_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect gate-only agreement/document-constraint ablation summaries."
    )
    parser.add_argument(
        "--summary",
        action="append",
        nargs=3,
        metavar=("DATASET", "VARIANT", "SUMMARY_JSON"),
        required=True,
        help="Dataset label, policy variant, and safe-gate summary JSON. Repeat per output.",
    )
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def count(summary: dict[str, Any], bucket: str, key: str) -> int:
    values = summary.get(bucket) or {}
    return int(values.get(key, 0))


def read_row(dataset: str, variant: str, path: Path) -> dict[str, Any]:
    if variant not in VARIANT_LABELS:
        raise ValueError(f"Unknown policy variant: {variant}")
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    config = summary.get("config") or {}
    return {
        "dataset": dataset,
        "variant": variant,
        "hit_k": int(config["hit_k"]),
        "min_page_overlap": int(config["min_page_overlap"]),
        "min_support_page_votes": int(config["min_support_page_votes"]),
        "promoted_doc_max_base_rank": int(config["promoted_doc_max_base_rank"]),
        "require_promoted_doc_in_base_topk": bool(
            config.get("require_promoted_doc_in_base_topk", False)
        ),
        "accepted": int(summary["accepted_count"]),
        "base_page_hit_at_k": summary.get("base_page_hit_at_k_count"),
        "candidate_page_hit_at_k": summary.get("candidate_page_hit_at_k_count"),
        "page_hit_at_k": summary.get("page_hit_at_k_count"),
        "recovered": summary.get("recovered"),
        "lost": summary.get("lost"),
        "page_net": summary.get("net_recovered"),
        "base_doc_hit_at_k": summary.get("base_doc_hit_at_k_count"),
        "doc_hit_at_k": summary.get("doc_hit_at_k_count"),
        "doc_net": summary.get("doc_net_recovered"),
        "overlap_rejects": count(
            summary, "selection_reason_counts", "page_overlap_below_min"
        ),
        "doc_rejects": count(
            summary, "rejected_promotion_reason_counts", "promoted_doc_after_allowed_rank"
        ),
        "topk_doc_rejects": count(
            summary, "rejected_promotion_reason_counts", "promoted_doc_not_in_base_topk"
        ),
        "support_rejects": count(
            summary, "rejected_promotion_reason_counts", "support_page_votes_below_min"
        ),
        "body_rejects": count(
            summary, "rejected_promotion_reason_counts", "promoted_body_score_not_above_base"
        ),
        "summary_path": str(path),
    }


def delta(value: Any, control_value: Any) -> int | None:
    if value is None or control_value is None:
        return None
    return int(value) - int(control_value)


def collect_rows(specs: Iterable[tuple[str, str, Path]]) -> tuple[list[dict[str, Any]], int]:
    rows = [read_row(dataset, variant, path) for dataset, variant, path in specs]
    hit_ks = {int(row["hit_k"]) for row in rows}
    if len(hit_ks) != 1:
        raise ValueError(f"Expected a single hit_k across summaries, found: {sorted(hit_ks)}")
    output: list[dict[str, Any]] = []
    datasets = list(dict.fromkeys(str(row["dataset"]) for row in rows))
    for dataset in datasets:
        dataset_rows = {
            str(row["variant"]): row for row in rows if row["dataset"] == dataset
        }
        if "control" not in dataset_rows:
            raise ValueError(f"{dataset} is missing the control policy summary")
        control = dataset_rows["control"]
        for variant in VARIANT_ORDER:
            if variant not in dataset_rows:
                continue
            row = dataset_rows[variant]
            row["delta_page_vs_control"] = delta(
                row["page_hit_at_k"], control["page_hit_at_k"]
            )
            row["delta_doc_vs_control"] = delta(
                row["doc_hit_at_k"], control["doc_hit_at_k"]
            )
            output.append(row)
    return output, hit_ks.pop()


def display(value: Any) -> str:
    if value is None:
        return "NA"
    return str(value)


def signed(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{int(value):+d}"


def doc_guard(row: dict[str, Any]) -> str:
    if bool(row["require_promoted_doc_in_base_topk"]):
        return "true top-k membership"
    value = int(row["promoted_doc_max_base_rank"])
    return "off" if value == 0 else f"first {value} distinct docs"


def render_markdown(rows: list[dict[str, Any]], hit_k: int) -> str:
    lines = [
        f"# Safe Gate Policy Ablation: hit@{hit_k}",
        "",
        "All rows reuse completed graph-view predictions; only the conservative rescue gate policy changes.",
        "The control row is the EvidenceGuard-PPR boundary policy for this cutoff.",
        "",
        "## Policy Definitions",
        "",
        "| policy | top-k page overlap minimum | support-page votes minimum | promoted-document guard |",
        "|---|---:|---:|---|",
    ]
    seen: set[str] = set()
    for row in rows:
        variant = str(row["variant"])
        if variant in seen:
            continue
        seen.add(variant)
        lines.append(
            "| {label} | {overlap} | {votes} | {doc_max} |".format(
                label=VARIANT_LABELS[variant],
                overlap=row["min_page_overlap"],
                votes=row["min_support_page_votes"],
                doc_max=doc_guard(row),
            )
        )
    lines.extend(
        [
            "",
            "## Outcomes",
            "",
            (
                f"| dataset | policy | accepted | base page@{hit_k} | candidate page@{hit_k} | "
                f"gated page@{hit_k} | delta page vs control | recovered | lost | page net | "
                f"base doc@{hit_k} | gated doc@{hit_k} | delta doc vs control | doc net | "
                "overlap rejects | doc-rank rejects | top-k-doc rejects | support rejects | body rejects |"
            ),
            (
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
                "---:|---:|---:|---:|---:|---:|---:|---:|---:|"
            ),
        ]
    )
    for row in rows:
        lines.append(
            "| {dataset} | {label} | {accepted} | {base_page} | {candidate_page} | {page} | "
            "{delta_page} | {recovered} | {lost} | {page_net} | {base_doc} | {doc} | "
            "{delta_doc} | {doc_net} | {overlap_rejects} | {doc_rejects} | "
            "{topk_doc_rejects} | {support_rejects} | {body_rejects} |".format(
                dataset=row["dataset"],
                label=VARIANT_LABELS[str(row["variant"])],
                accepted=row["accepted"],
                base_page=display(row["base_page_hit_at_k"]),
                candidate_page=display(row["candidate_page_hit_at_k"]),
                page=display(row["page_hit_at_k"]),
                delta_page=signed(row["delta_page_vs_control"]),
                recovered=display(row["recovered"]),
                lost=display(row["lost"]),
                page_net=signed(row["page_net"]),
                base_doc=display(row["base_doc_hit_at_k"]),
                doc=display(row["doc_hit_at_k"]),
                delta_doc=signed(row["delta_doc_vs_control"]),
                doc_net=signed(row["doc_net"]),
                overlap_rejects=row["overlap_rejects"],
                doc_rejects=row["doc_rejects"],
                topk_doc_rejects=row["topk_doc_rejects"],
                support_rejects=row["support_rejects"],
                body_rejects=row["body_rejects"],
            )
        )
    return "\n".join(lines) + "\n"


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in CSV_COLUMNS})


def main() -> None:
    args = parse_args()
    rows, hit_k = collect_rows(
        (dataset, variant, Path(path)) for dataset, variant, path in args.summary
    )
    output_md = Path(args.output_md)
    output_csv = Path(args.output_csv)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(rows, hit_k)
    output_md.write_text(markdown, encoding="utf-8")
    write_csv(rows, output_csv)
    print(markdown, end="")
    print(f"saved_report_md={output_md}")
    print(f"saved_report_csv={output_csv}")


if __name__ == "__main__":
    main()
