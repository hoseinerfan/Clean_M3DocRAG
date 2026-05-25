#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from analyze_layout_evidence_gate import (
    first_rank,
    gold_doc_ids,
    gold_page_uids,
    load_case_json,
    load_prediction,
    page_doc,
    ranked_pages,
    read_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose Gaussian boundary graph reranker cases. The report explains recovered, "
            "lost, and unchanged swaps using boundary/gold overlap, support ranks, and z margins."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=6,
        metavar=("LABEL", "GOLD", "BASE", "CANDIDATE", "CASES", "SUPPORT"),
        default=[],
        help="Dataset tuple. Repeat for side-by-side reports.",
    )
    parser.add_argument("--hit-k", type=int, default=4)
    parser.add_argument("--boundary-top-pages", type=int, default=10)
    parser.add_argument("--top-examples", type=int, default=12)
    parser.add_argument("--output-md", default="")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def question_text(row: dict[str, Any]) -> str:
    for key in ("question", "query", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"].strip()
    return ""


def support_rank_map(prediction: dict[str, Any] | None) -> dict[str, int]:
    return {uid: rank for rank, uid in enumerate(ranked_pages(prediction), start=1)}


def safe_median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(median(values))


def movement(base_rank: int | None, candidate_rank: int | None, hit_k: int) -> str:
    base_hit = base_rank is not None and base_rank <= hit_k
    candidate_hit = candidate_rank is not None and candidate_rank <= hit_k
    if not base_hit and candidate_hit:
        return "recovered"
    if base_hit and not candidate_hit:
        return "lost"
    if base_hit and candidate_hit:
        if candidate_rank is not None and base_rank is not None and candidate_rank < base_rank:
            return "improved_rank"
        if candidate_rank is not None and base_rank is not None and candidate_rank > base_rank:
            return "worsened_rank"
    return "unchanged"


def analyze_run(
    *,
    label: str,
    gold_path: Path,
    base_path: Path,
    candidate_path: Path,
    case_path: Path,
    support_path: Path,
    hit_k: int,
    boundary_top_pages: int,
    top_examples: int,
) -> dict[str, Any]:
    gold = {str(row["qid"]): row for row in read_jsonl(gold_path)}
    base = load_prediction(base_path)
    candidate = load_prediction(candidate_path)
    cases = load_case_json(case_path)
    support = load_prediction(support_path)

    rows: list[dict[str, Any]] = []
    movement_counts: Counter[str] = Counter()
    feature_by_movement: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    bool_by_movement: dict[str, Counter[str]] = defaultdict(Counter)
    base_gold_rank_counts: Counter[str] = Counter()
    accepted_counts: Counter[str] = Counter()

    qids = sorted(set(gold) & set(base) & set(candidate))
    for qid in qids:
        gold_pages = gold_page_uids(gold[qid])
        gold_docs = gold_doc_ids(gold[qid])
        base_pages = ranked_pages(base[qid])
        candidate_pages = ranked_pages(candidate[qid])
        support_ranks = support_rank_map(support.get(qid))
        base_rank = first_rank(base_pages, gold_pages)
        candidate_rank = first_rank(candidate_pages, gold_pages)
        move = movement(base_rank, candidate_rank, hit_k)
        movement_counts[move] += 1

        base_top = base_pages[:hit_k]
        boundary = base_pages[hit_k:boundary_top_pages]
        case = cases.get(qid, {})
        accepted = bool(case.get("accepted"))
        accepted_counts["accepted" if accepted else "kept_base"] += 1
        best_boundary = str(case.get("best_boundary_page", ""))
        weakest_top = str(case.get("weakest_top_page", ""))
        best_boundary_doc = page_doc(best_boundary) if best_boundary else ""
        weakest_top_doc = page_doc(weakest_top) if weakest_top else ""
        boundary_gold = sorted(set(boundary) & gold_pages)
        top_gold = sorted(set(base_top) & gold_pages)

        if base_rank is None:
            base_gold_rank_counts["missing"] += 1
        elif base_rank <= hit_k:
            base_gold_rank_counts["top_hit"] += 1
        elif base_rank <= boundary_top_pages:
            base_gold_rank_counts["boundary"] += 1
        else:
            base_gold_rank_counts["below_boundary"] += 1

        support_gold_ranks = [
            float(rank)
            for uid, rank in support_ranks.items()
            if uid in gold_pages
        ]
        support_best_rank = support_ranks.get(best_boundary)
        z_margin = case.get("z_margin")
        best_boundary_z = case.get("best_boundary_z")
        weakest_top_z = case.get("weakest_top_z")

        row = {
            "qid": qid,
            "movement": move,
            "accepted": accepted,
            "base_first_gold_page_rank": base_rank,
            "candidate_first_gold_page_rank": candidate_rank,
            "best_boundary_page": best_boundary,
            "best_boundary_base_rank": case.get("best_boundary_base_rank"),
            "weakest_top_page": weakest_top,
            "weakest_top_base_rank": case.get("weakest_top_base_rank"),
            "best_boundary_is_gold": best_boundary in gold_pages,
            "best_boundary_doc_is_gold": best_boundary_doc in gold_docs,
            "weakest_top_is_gold": weakest_top in gold_pages,
            "weakest_top_doc_is_gold": weakest_top_doc in gold_docs,
            "boundary_contains_gold": bool(boundary_gold),
            "base_top_contains_gold": bool(top_gold),
            "support_rank_best_boundary": support_best_rank,
            "support_best_gold_rank": min(support_gold_ranks) if support_gold_ranks else None,
            "z_margin": z_margin,
            "best_boundary_z": best_boundary_z,
            "weakest_top_z": weakest_top_z,
            "gold_pages": sorted(gold_pages),
            "question": question_text(gold[qid]),
        }
        rows.append(row)

        for key in ("z_margin", "best_boundary_z", "weakest_top_z"):
            if isinstance(row[key], (int, float)):
                feature_by_movement[move][key].append(float(row[key]))
        for key in ("support_rank_best_boundary", "support_best_gold_rank"):
            if isinstance(row[key], (int, float)):
                feature_by_movement[move][key].append(float(row[key]))
        for key in (
            "accepted",
            "best_boundary_is_gold",
            "best_boundary_doc_is_gold",
            "weakest_top_is_gold",
            "weakest_top_doc_is_gold",
            "boundary_contains_gold",
            "base_top_contains_gold",
        ):
            if row[key]:
                bool_by_movement[move][key] += 1

    movement_summary: dict[str, Any] = {}
    for move, count in sorted(movement_counts.items()):
        movement_summary[move] = {
            "n": count,
            **{
                f"{key}_count": bool_by_movement[move].get(key, 0)
                for key in (
                    "accepted",
                    "best_boundary_is_gold",
                    "best_boundary_doc_is_gold",
                    "weakest_top_is_gold",
                    "weakest_top_doc_is_gold",
                    "boundary_contains_gold",
                    "base_top_contains_gold",
                )
            },
            **{
                f"{key}_median": safe_median(values)
                for key, values in sorted(feature_by_movement[move].items())
            },
        }

    examples: dict[str, list[dict[str, Any]]] = {}
    for move in ("lost", "recovered", "unchanged", "worsened_rank", "improved_rank"):
        selected = [row for row in rows if row["movement"] == move]
        selected.sort(
            key=lambda row: (
                0 if row.get("accepted") else 1,
                -(float(row["z_margin"]) if isinstance(row.get("z_margin"), (int, float)) else -1e9),
            )
        )
        examples[move] = selected[:top_examples]

    return {
        "label": label,
        "n": len(qids),
        "accepted_counts": dict(accepted_counts),
        "movement_counts": dict(sorted(movement_counts.items())),
        "base_gold_rank_counts": dict(sorted(base_gold_rank_counts.items())),
        "movement_summary": movement_summary,
        "examples": examples,
    }


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def render_md(report: dict[str, Any]) -> str:
    lines: list[str] = ["# Boundary Gaussian Diagnostics", ""]
    for run in report["runs"]:
        lines.append(f"## {run['label']}")
        lines.append("")
        lines.append(f"n: {run['n']}")
        lines.append(f"accepted: {run['accepted_counts']}")
        lines.append(f"movement: {run['movement_counts']}")
        lines.append(f"base_gold_rank: {run['base_gold_rank_counts']}")
        lines.append("")
        lines.append("### Movement Summary")
        lines.append("")
        headers = [
            "movement",
            "n",
            "accepted",
            "best_boundary_gold",
            "best_boundary_gold_doc",
            "weakest_top_gold",
            "boundary_has_gold",
            "base_top_has_gold",
            "z_margin_med",
            "support_best_boundary_med",
            "support_best_gold_med",
        ]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for move, row in run["movement_summary"].items():
            values = [
                move,
                row.get("n"),
                row.get("accepted_count"),
                row.get("best_boundary_is_gold_count"),
                row.get("best_boundary_doc_is_gold_count"),
                row.get("weakest_top_is_gold_count"),
                row.get("boundary_contains_gold_count"),
                row.get("base_top_contains_gold_count"),
                row.get("z_margin_median"),
                row.get("support_rank_best_boundary_median"),
                row.get("support_best_gold_rank_median"),
            ]
            lines.append("| " + " | ".join(fmt(value) for value in values) + " |")
        lines.append("")

        for move in ("lost", "recovered"):
            examples = run["examples"].get(move, [])
            if not examples:
                continue
            lines.append(f"### {move.title()} Examples")
            lines.append("")
            lines.append(
                "| qid | base_rank | cand_rank | weakest_top | best_boundary | boundary_gold | "
                "boundary_gold_doc | z_margin | support_best_boundary | support_best_gold | question |"
            )
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for row in examples:
                question = str(row.get("question", "")).replace("|", "\\|")[:180]
                values = [
                    row.get("qid"),
                    row.get("base_first_gold_page_rank"),
                    row.get("candidate_first_gold_page_rank"),
                    row.get("weakest_top_page"),
                    row.get("best_boundary_page"),
                    row.get("best_boundary_is_gold"),
                    row.get("best_boundary_doc_is_gold"),
                    row.get("z_margin"),
                    row.get("support_rank_best_boundary"),
                    row.get("support_best_gold_rank"),
                    question,
                ]
                lines.append("| " + " | ".join(fmt(value) for value in values) + " |")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    runs = [
        analyze_run(
            label=label,
            gold_path=Path(gold),
            base_path=Path(base),
            candidate_path=Path(candidate),
            case_path=Path(cases),
            support_path=Path(support),
            hit_k=int(args.hit_k),
            boundary_top_pages=int(args.boundary_top_pages),
            top_examples=int(args.top_examples),
        )
        for label, gold, base, candidate, cases, support in args.run
    ]
    report = {"runs": runs}
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved_json: {path}")
    md = render_md(report)
    if args.output_md:
        path = Path(args.output_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(md, encoding="utf-8")
        print(f"saved_md: {path}")
    else:
        print(md)


if __name__ == "__main__":
    main()
