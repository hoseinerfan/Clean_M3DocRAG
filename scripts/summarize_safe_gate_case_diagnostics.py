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
    parser.add_argument(
        "--baseline-prediction",
        action="append",
        nargs=2,
        metavar=("DATASET", "PREDICTION_JSON"),
        default=[],
        help=(
            "Optional original M3DocRAG/plain-top224 prediction JSON for a dataset. "
            "When supplied, the report adds baseline page@k before Graph-PPR/gating."
        ),
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


def load_prediction(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    if isinstance(payload, dict) and "predictions" in payload and isinstance(
        payload["predictions"], (dict, list)
    ):
        payload = payload["predictions"]

    rows_by_qid: dict[str, dict[str, Any]] = {}
    if isinstance(payload, list):
        iterable = enumerate(payload)
    elif isinstance(payload, dict):
        iterable = payload.items()
    else:
        raise TypeError(f"Prediction JSON must be a list or object: {path}")

    for raw_key, row in iterable:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("qid", "")).strip() or str(raw_key).strip()
        if qid:
            rows_by_qid[qid] = row
    return rows_by_qid


def page_uid(doc_id: object, page_idx: object) -> str:
    return f"{doc_id}_page{int(page_idx)}"


def row_page_uid(row: Any) -> str | None:
    if not isinstance(row, list) or len(row) < 2:
        return None
    try:
        return page_uid(row[0], row[1])
    except (TypeError, ValueError):
        return None


def ranked_page_uids(row: dict[str, Any]) -> list[str]:
    pages: list[str] = []
    seen: set[str] = set()
    raw_rows = row.get("page_retrieval_results", [])
    if not isinstance(raw_rows, list):
        return pages
    for raw_row in raw_rows:
        uid = row_page_uid(raw_row)
        if uid is None or uid in seen:
            continue
        seen.add(uid)
        pages.append(uid)
    return pages


def first_rank(ranked: list[str], gold: set[str]) -> int | None:
    for index, value in enumerate(ranked, start=1):
        if value in gold:
            return index
    return None


def hit_at(rank: int | None, topk: int) -> bool:
    return rank is not None and int(rank) <= int(topk)


def baseline_page_hit_count(
    cases: list[dict[str, Any]],
    prediction: dict[str, dict[str, Any]] | None,
    hit_k: int,
) -> int | None:
    if prediction is None:
        return None
    count = 0
    for case in cases:
        gold = {str(uid).strip() for uid in case.get("gold_page_uids", []) if str(uid).strip()}
        if not gold:
            continue
        pred_row = prediction.get(str(case.get("qid", "")).strip())
        if not pred_row:
            continue
        if hit_at(first_rank(ranked_page_uids(pred_row), gold), hit_k):
            count += 1
    return count


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


def boundary_opportunity_count(cases: list[dict[str, Any]], hit_k: int) -> int:
    boundary_rank = int(hit_k) + 1
    count = 0
    for case in cases:
        if not case.get("gold_page_uids"):
            continue
        try:
            rank = int(case.get("base_first_gold_page_rank"))
        except (TypeError, ValueError):
            continue
        if rank == boundary_rank:
            count += 1
    return count


def build_entry(
    dataset: str,
    variant: str,
    summary_path: Path,
    cases_path: Path,
    baseline_prediction: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
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
    hit_k = int(config.get("hit_k", 0) or 0)
    return {
        "dataset": dataset,
        "variant": variant,
        "hit_k": hit_k,
        "accepted": int(summary.get("accepted_count", 0) or 0),
        "baseline_page": baseline_page_hit_count(cases, baseline_prediction, hit_k)
        if hit_k
        else None,
        "base_page": summary.get("base_page_hit_at_k_count"),
        "candidate_page": summary.get("candidate_page_hit_at_k_count"),
        "gated_page": summary.get("page_hit_at_k_count"),
        "boundary_opportunities": boundary_opportunity_count(cases, hit_k) if hit_k else None,
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
                "baseline_page": (
                    sum(int(row["baseline_page"] or 0) for row in rows)
                    if all(row.get("baseline_page") is not None for row in rows)
                    else None
                ),
                "base_page": sum(int(row["base_page"] or 0) for row in rows),
                "candidate_page": sum(int(row["candidate_page"] or 0) for row in rows),
                "gated_page": sum(int(row["gated_page"] or 0) for row in rows),
                "boundary_opportunities": sum(
                    int(row["boundary_opportunities"] or 0) for row in rows
                ),
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
        "| variant | datasets | accepted | M3DocRAG page | base page | candidate page | gated page | boundary opportunities | recovered | lost | net | doc net |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_by_variant(entries):
        lines.append(
            "| {variant} | {datasets} | {accepted} | {baseline_page} | {base_page} | "
            "{candidate_page} | {gated_page} | {boundary_opportunities} | {recovered} | "
            "{lost} | {net} | {doc_net} |".format(
                **{**row, "baseline_page": display(row.get("baseline_page"))}
            )
        )

    lines.extend(
        [
            "",
            "## Dataset/Variant Diagnostics",
            "",
            "| dataset | variant | accepted | M3DocRAG page | base page | candidate page | gated page | boundary opportunities | recovered | lost | net | candidate net | accepted movements | accepted base ranks | accepted doc ranks | support votes | top rejected reasons |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|",
        ]
    )
    for entry in entries:
        lines.append(
            "| {dataset} | {variant} | {accepted} | {baseline_page} | {base_page} | "
            "{candidate_page} | {gated_page} | {boundary_opportunities} | {recovered} | "
            "{lost} | {net} | {candidate_net} | {accepted_movements} | {base_ranks} | "
            "{doc_ranks} | {support_votes} | {rejected} |".format(
                dataset=entry["dataset"],
                variant=entry["variant"],
                accepted=entry["accepted"],
                baseline_page=display(entry.get("baseline_page")),
                base_page=display(entry["base_page"]),
                candidate_page=display(entry["candidate_page"]),
                gated_page=display(entry["gated_page"]),
                boundary_opportunities=display(entry.get("boundary_opportunities")),
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
    baseline_predictions = {
        str(dataset): load_prediction(Path(path))
        for dataset, path in args.baseline_prediction
        if Path(path).is_file()
    }
    entries = [
        build_entry(
            dataset,
            variant,
            Path(summary),
            Path(cases),
            baseline_prediction=baseline_predictions.get(dataset),
        )
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
