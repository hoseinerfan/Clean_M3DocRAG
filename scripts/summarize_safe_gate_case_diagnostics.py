#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize safe-gate case diagnostics across datasets and policy variants."
    )
    parser.add_argument(
        "--entry",
        action="append",
        nargs=4,
        metavar=("DATASET", "VARIANT", "SUMMARY_JSON", "CASES_JSON"),
        required=True,
        help="Dataset, variant label, summary JSON, and cases JSON. Repeat per output.",
    )
    parser.add_argument("--topn", type=int, default=8)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_cases(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
        payload = payload["cases"]
    if not isinstance(payload, list):
        raise TypeError(f"Expected case list or object with cases list: {path}")
    return [case for case in payload if isinstance(case, dict)]


def display(value: Any) -> str:
    if value is None:
        return "NA"
    return str(value)


def signed(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{int(value):+d}"


def count_map(values: Iterable[Any]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for value in values:
        if value is None:
            continue
        counter[str(value)] += 1
    return dict(sorted(counter.items(), key=lambda item: (item[0])))


def top_counts(counter: dict[str, int], limit: int = 4) -> str:
    if not counter:
        return "none"
    items = sorted(counter.items(), key=lambda item: (-int(item[1]), str(item[0])))[:limit]
    return ", ".join(f"{key}:{value}" for key, value in items)


def rank_counts(values: Iterable[Any]) -> str:
    counts: Counter[int] = Counter()
    for value in values:
        if value is None:
            continue
        try:
            counts[int(value)] += 1
        except (TypeError, ValueError):
            continue
    if not counts:
        return "none"
    return ", ".join(f"{rank}:{counts[rank]}" for rank in sorted(counts))


def accepted_promotions(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item
        for case in cases
        if bool(case.get("accepted"))
        for item in case.get("accepted_promoted_pages", [])
        if isinstance(item, dict)
    ]


def rejected_promotions(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item
        for case in cases
        for item in case.get("rejected_promoted_pages", [])
        if isinstance(item, dict)
    ]


def case_sort_key(case: dict[str, Any]) -> tuple[int, int, str]:
    output_rank = case.get("output_first_gold_page_rank")
    base_rank = case.get("base_first_gold_page_rank")
    return (
        int(output_rank) if output_rank is not None else 10**9,
        int(base_rank) if base_rank is not None else 10**9,
        str(case.get("qid", "")),
    )


def promotion_brief(items: list[dict[str, Any]], limit: int = 2) -> str:
    parts: list[str] = []
    for item in items[:limit]:
        parts.append(
            "{uid} c{cand}/b{base}/d{doc}/v{votes}".format(
                uid=item.get("page_uid", ""),
                cand=display(item.get("candidate_rank")),
                base=display(item.get("base_rank")),
                doc=display(item.get("base_doc_rank")),
                votes=display(item.get("support_page_vote_count")),
            )
        )
    return "; ".join(parts) if parts else "none"


def case_brief(case: dict[str, Any]) -> str:
    return (
        f"`{case.get('qid', '')}` "
        f"base={display(case.get('base_first_gold_page_rank'))} "
        f"cand={display(case.get('candidate_first_gold_page_rank'))} "
        f"out={display(case.get('output_first_gold_page_rank'))} "
        f"promoted={promotion_brief(case.get('accepted_promoted_pages', []))} "
        f"question={str(case.get('question', '')).strip()[:220]}"
    )


def build_entry(dataset: str, variant: str, summary_path: Path, cases_path: Path) -> dict[str, Any]:
    summary = read_json(summary_path)
    cases = read_cases(cases_path)
    accepted_cases = [case for case in cases if bool(case.get("accepted"))]
    accepted = accepted_promotions(cases)
    rejected = rejected_promotions(cases)
    accepted_movements = count_map(case.get("movement_vs_base") for case in accepted_cases)
    accepted_candidate_movements = count_map(
        case.get("candidate_movement_vs_base") for case in accepted_cases
    )
    rejected_reasons = count_map(item.get("reject_reason") for item in rejected)
    config = summary.get("config") or {}
    return {
        "dataset": dataset,
        "variant": variant,
        "hit_k": int(config.get("hit_k", 0) or 0),
        "accepted": int(summary.get("accepted_count", 0) or 0),
        "base_page": summary.get("base_page_hit_at_k_count"),
        "candidate_page": summary.get("candidate_page_hit_at_k_count"),
        "gated_page": summary.get("page_hit_at_k_count"),
        "recovered": summary.get("recovered"),
        "lost": summary.get("lost"),
        "net": summary.get("net_recovered"),
        "base_doc": summary.get("base_doc_hit_at_k_count"),
        "candidate_doc": summary.get("candidate_doc_hit_at_k_count"),
        "gated_doc": summary.get("doc_hit_at_k_count"),
        "doc_net": summary.get("doc_net_recovered"),
        "candidate_recovered": summary.get("candidate_recovered"),
        "candidate_lost": summary.get("candidate_lost"),
        "candidate_net": summary.get("candidate_net_recovered"),
        "accepted_movements": accepted_movements,
        "accepted_candidate_movements": accepted_candidate_movements,
        "rejected_reasons": rejected_reasons,
        "accepted_base_rank_counts": rank_counts(item.get("base_rank") for item in accepted),
        "accepted_doc_rank_counts": rank_counts(item.get("base_doc_rank") for item in accepted),
        "accepted_support_vote_counts": rank_counts(
            item.get("support_page_vote_count") for item in accepted
        ),
        "lost_cases": sorted(
            [case for case in cases if case.get("movement_vs_base") == "lost"],
            key=case_sort_key,
        ),
        "recovered_cases": sorted(
            [case for case in cases if case.get("movement_vs_base") == "recovered"],
            key=case_sort_key,
        ),
        "summary_path": str(summary_path),
        "cases_path": str(cases_path),
    }


def aggregate_by_variant(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[str(entry["variant"])].append(entry)
    output: list[dict[str, Any]] = []
    for variant, rows in grouped.items():
        output.append(
            {
                "variant": variant,
                "datasets": len(rows),
                "accepted": sum(int(row["accepted"]) for row in rows),
                "base_page": sum(int(row["base_page"] or 0) for row in rows),
                "candidate_page": sum(int(row["candidate_page"] or 0) for row in rows),
                "gated_page": sum(int(row["gated_page"] or 0) for row in rows),
                "recovered": sum(int(row["recovered"] or 0) for row in rows),
                "lost": sum(int(row["lost"] or 0) for row in rows),
                "net": sum(int(row["net"] or 0) for row in rows),
                "doc_net": sum(int(row["doc_net"] or 0) for row in rows),
            }
        )
    return sorted(output, key=lambda row: str(row["variant"]))


def render_markdown(entries: list[dict[str, Any]], topn: int) -> str:
    hit_ks = sorted({entry["hit_k"] for entry in entries if entry["hit_k"]})
    hit_label = hit_ks[0] if len(hit_ks) == 1 else "mixed"
    lines: list[str] = [
        f"# Safe Gate Case Diagnostics: top-{hit_label}",
        "",
        "This report reads completed safe-gate case JSON files. It does not rerun retrieval.",
        "",
        "## Aggregate By Variant",
        "",
        "| variant | datasets | accepted | base page | candidate page | gated page | recovered | lost | net | doc net |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_by_variant(entries):
        lines.append(
            "| {variant} | {datasets} | {accepted} | {base_page} | {candidate_page} | "
            "{gated_page} | {recovered} | {lost} | {net} | {doc_net} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Dataset/Variant Diagnostics",
            "",
            "| dataset | variant | accepted | base page | candidate page | gated page | recovered | lost | net | candidate net | accepted movements | accepted base ranks | accepted doc ranks | support votes | top rejected reasons |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|",
        ]
    )
    for entry in entries:
        lines.append(
            "| {dataset} | {variant} | {accepted} | {base_page} | {candidate_page} | {gated_page} | "
            "{recovered} | {lost} | {net} | {candidate_net} | {accepted_movements} | "
            "{base_ranks} | {doc_ranks} | {support_votes} | {rejected} |".format(
                dataset=entry["dataset"],
                variant=entry["variant"],
                accepted=entry["accepted"],
                base_page=display(entry["base_page"]),
                candidate_page=display(entry["candidate_page"]),
                gated_page=display(entry["gated_page"]),
                recovered=display(entry["recovered"]),
                lost=display(entry["lost"]),
                net=signed(entry["net"]),
                candidate_net=signed(entry["candidate_net"]),
                accepted_movements=top_counts(entry["accepted_movements"]),
                base_ranks=entry["accepted_base_rank_counts"],
                doc_ranks=entry["accepted_doc_rank_counts"],
                support_votes=entry["accepted_support_vote_counts"],
                rejected=top_counts(entry["rejected_reasons"]),
            )
        )

    lines.extend(["", "## Lost Cases", ""])
    any_lost = False
    for entry in entries:
        if not entry["lost_cases"]:
            continue
        any_lost = True
        lines.append(f"### {entry['dataset']} / {entry['variant']}")
        for case in entry["lost_cases"][:topn]:
            lines.append(f"- {case_brief(case)}")
        lines.append("")
    if not any_lost:
        lines.append("No page-level lost cases in the selected entries.")
        lines.append("")

    lines.extend(["## Recovered Cases", ""])
    any_recovered = False
    for entry in entries:
        if not entry["recovered_cases"]:
            continue
        any_recovered = True
        lines.append(f"### {entry['dataset']} / {entry['variant']}")
        for case in entry["recovered_cases"][:topn]:
            lines.append(f"- {case_brief(case)}")
        lines.append("")
    if not any_recovered:
        lines.append("No page-level recovered cases in the selected entries.")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    entries = [
        build_entry(dataset, variant, Path(summary), Path(cases))
        for dataset, variant, summary, cases in args.entry
    ]
    output_md = Path(args.output_md)
    output_json = Path(args.output_json)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output = {"entries": entries, "aggregate_by_variant": aggregate_by_variant(entries)}
    output_md.write_text(render_markdown(entries, int(args.topn)), encoding="utf-8")
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"saved_md: {output_md}")
    print(f"saved_json: {output_json}")


if __name__ == "__main__":
    main()
