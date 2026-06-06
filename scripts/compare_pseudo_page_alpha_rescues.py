#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_pseudo_page_retrieval import (
    first_rank,
    gold_pages,
    load_gold,
    load_prediction,
    parse_labeled_path,
    ranked_pages,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare pseudo-page rescues from two or more fixed-alpha retrieval runs. "
            "The report checks whether different alpha values rescue different qids "
            "and cover different gold page uids at the chosen top-k boundary."
        )
    )
    parser.add_argument("--gold", required=True, help="Augmented gold JSONL with pseudo page labels.")
    parser.add_argument("--baseline", required=True, help="Baseline prediction JSON.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Candidate run as LABEL=path/to/prediction.json. Repeat for each alpha.",
    )
    parser.add_argument("--hit-k", type=int, default=5, help="Top-k cutoff for rescue and coverage.")
    parser.add_argument("--examples-per-section", type=int, default=12)
    parser.add_argument("--output-md", default="")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def topk_covered_gold_pages(pred_row: dict[str, Any] | None, gold: set[str], k: int) -> set[str]:
    return set(ranked_pages(pred_row)[: int(k)]) & gold


def movement(base_hit: bool, cand_hit: bool, base_rank: int | None, cand_rank: int | None) -> str:
    if not base_hit and cand_hit:
        return "recovered"
    if base_hit and not cand_hit:
        return "lost"
    if base_rank == cand_rank:
        return "unchanged"
    if cand_rank is None:
        return "worsened_rank"
    if base_rank is None or cand_rank < base_rank:
        return "improved_rank"
    return "worsened_rank"


def truncate(text: str, limit: int = 130) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def build_run_cases(
    *,
    label: str,
    gold: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    hit_k: int,
) -> dict[str, Any]:
    cases: dict[str, dict[str, Any]] = {}
    skipped_no_gold = 0
    missing_baseline = 0
    missing_candidate = 0

    for qid in sorted(gold):
        gold_row = gold[qid]
        gp = gold_pages(gold_row)
        if not gp:
            skipped_no_gold += 1
            continue
        base_row = baseline.get(qid)
        cand_row = candidate.get(qid)
        if base_row is None:
            missing_baseline += 1
            continue
        if cand_row is None:
            missing_candidate += 1
            continue

        base_pages = ranked_pages(base_row)
        cand_pages = ranked_pages(cand_row)
        base_covered = topk_covered_gold_pages(base_row, gp, hit_k)
        cand_covered = topk_covered_gold_pages(cand_row, gp, hit_k)
        base_rank = first_rank(base_pages, gp)
        cand_rank = first_rank(cand_pages, gp)
        move = movement(
            bool(base_covered),
            bool(cand_covered),
            base_rank,
            cand_rank,
        )
        cases[qid] = {
            "qid": qid,
            "question": gold_row.get("question", ""),
            "gold_page_uids": sorted(gp),
            "base_first_gold_rank": base_rank,
            "candidate_first_gold_rank": cand_rank,
            "base_covered_gold_pages": sorted(base_covered),
            "candidate_covered_gold_pages": sorted(cand_covered),
            "gained_gold_pages": sorted(cand_covered - base_covered),
            "dropped_gold_pages": sorted(base_covered - cand_covered),
            "movement": move,
        }

    n_eval = len(cases)
    recovered = {qid for qid, row in cases.items() if row["movement"] == "recovered"}
    lost = {qid for qid, row in cases.items() if row["movement"] == "lost"}
    improved_rank = {qid for qid, row in cases.items() if row["movement"] == "improved_rank"}
    worsened_rank = {qid for qid, row in cases.items() if row["movement"] == "worsened_rank"}
    hit_qids = {qid for qid, row in cases.items() if row["candidate_covered_gold_pages"]}
    covered_pairs = {
        (qid, page_uid)
        for qid, row in cases.items()
        for page_uid in row["candidate_covered_gold_pages"]
    }
    gold_pairs = {
        (qid, page_uid)
        for qid, row in cases.items()
        for page_uid in row["gold_page_uids"]
    }
    full_coverage_qids = {
        qid
        for qid, row in cases.items()
        if set(row["candidate_covered_gold_pages"]) == set(row["gold_page_uids"])
    }

    summary = {
        "label": label,
        "hit_k": int(hit_k),
        "n_eval": n_eval,
        "skipped_no_page_gold": skipped_no_gold,
        "missing_baseline": missing_baseline,
        "missing_candidate": missing_candidate,
        f"page@{hit_k}": len(hit_qids) / float(n_eval) if n_eval else 0.0,
        f"full_gold_page_coverage@{hit_k}": len(full_coverage_qids) / float(n_eval) if n_eval else 0.0,
        f"covered_gold_page_pairs@{hit_k}": len(covered_pairs),
        "gold_page_pairs": len(gold_pairs),
        f"gold_page_pair_coverage@{hit_k}": len(covered_pairs) / float(len(gold_pairs)) if gold_pairs else 0.0,
        "recovered": len(recovered),
        "lost": len(lost),
        "net": len(recovered) - len(lost),
        "improved_rank": len(improved_rank),
        "worsened_rank": len(worsened_rank),
    }
    return {
        "label": label,
        "summary": summary,
        "cases": cases,
        "sets": {
            "recovered": recovered,
            "lost": lost,
            "hit_qids": hit_qids,
            "covered_pairs": covered_pairs,
            "full_coverage_qids": full_coverage_qids,
        },
    }


def jaccard(left: set[Any], right: set[Any]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / float(len(union))


def pair_summary(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_sets = left["sets"]
    right_sets = right["sets"]
    recovered_left = left_sets["recovered"]
    recovered_right = right_sets["recovered"]
    covered_left = left_sets["covered_pairs"]
    covered_right = right_sets["covered_pairs"]
    hit_left = left_sets["hit_qids"]
    hit_right = right_sets["hit_qids"]
    return {
        "left": left["label"],
        "right": right["label"],
        "recovered_left": len(recovered_left),
        "recovered_right": len(recovered_right),
        "recovered_overlap": len(recovered_left & recovered_right),
        "recovered_left_only": len(recovered_left - recovered_right),
        "recovered_right_only": len(recovered_right - recovered_left),
        "recovered_jaccard": jaccard(recovered_left, recovered_right),
        "hit_qid_left_only": len(hit_left - hit_right),
        "hit_qid_right_only": len(hit_right - hit_left),
        "covered_pairs_left": len(covered_left),
        "covered_pairs_right": len(covered_right),
        "covered_pairs_overlap": len(covered_left & covered_right),
        "covered_pairs_left_only": len(covered_left - covered_right),
        "covered_pairs_right_only": len(covered_right - covered_left),
        "covered_pairs_jaccard": jaccard(covered_left, covered_right),
    }


def serializable_pair(pair: tuple[str, str]) -> str:
    return f"{pair[0]}::{pair[1]}"


def render_table(headers: list[str], rows: list[dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values: list[str] = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def example_rows(
    *,
    run: dict[str, Any],
    qids: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    rows = []
    for qid in sorted(
        qids,
        key=lambda q: (
            run["cases"][q]["candidate_first_gold_rank"]
            if run["cases"][q]["candidate_first_gold_rank"] is not None
            else 10**9,
            run["cases"][q]["base_first_gold_rank"]
            if run["cases"][q]["base_first_gold_rank"] is not None
            else 10**9,
            q,
        ),
    )[: int(limit)]:
        case = run["cases"][qid]
        rows.append(
            {
                "qid": qid,
                "base_rank": case["base_first_gold_rank"],
                "run_rank": case["candidate_first_gold_rank"],
                "gained_gold_pages": ", ".join(case["gained_gold_pages"]),
                "question": truncate(case["question"]),
            }
        )
    return rows


def render_markdown(payload: dict[str, Any], examples_per_section: int) -> str:
    hit_k = payload["hit_k"]
    lines = [
        f"# Fixed-Alpha Rescue Overlap Audit",
        "",
        f"Hit cutoff: top-{hit_k}",
        "",
        "## Run Summary",
        "",
        render_table(
            [
                "label",
                "n_eval",
                f"page@{hit_k}",
                f"full_gold_page_coverage@{hit_k}",
                f"gold_page_pair_coverage@{hit_k}",
                "recovered",
                "lost",
                "net",
                "improved_rank",
                "worsened_rank",
            ],
            [run["summary"] for run in payload["runs"]],
        ),
        "",
        "## Pairwise Overlap",
        "",
        render_table(
            [
                "left",
                "right",
                "recovered_overlap",
                "recovered_left_only",
                "recovered_right_only",
                "recovered_jaccard",
                "covered_pairs_overlap",
                "covered_pairs_left_only",
                "covered_pairs_right_only",
                "covered_pairs_jaccard",
            ],
            payload["pairwise"],
        ),
    ]
    for pair in payload["pairwise"]:
        left_run = next(run for run in payload["runs"] if run["label"] == pair["left"])
        right_run = next(run for run in payload["runs"] if run["label"] == pair["right"])
        left_only = left_run["sets"]["recovered"] - right_run["sets"]["recovered"]
        right_only = right_run["sets"]["recovered"] - left_run["sets"]["recovered"]
        lines.extend(
            [
                "",
                f"## Unique Rescues: {pair['left']} vs {pair['right']}",
                "",
                f"### {pair['left']} only",
                "",
                render_table(
                    ["qid", "base_rank", "run_rank", "gained_gold_pages", "question"],
                    example_rows(run=left_run, qids=left_only, limit=examples_per_section),
                )
                if left_only
                else "_None._",
                "",
                f"### {pair['right']} only",
                "",
                render_table(
                    ["qid", "base_rank", "run_rank", "gained_gold_pages", "question"],
                    example_rows(run=right_run, qids=right_only, limit=examples_per_section),
                )
                if right_only
                else "_None._",
            ]
        )
    return "\n".join(lines) + "\n"


def make_json_safe(payload: dict[str, Any]) -> dict[str, Any]:
    safe_runs = []
    for run in payload["runs"]:
        safe_sets = {
            key: sorted(serializable_pair(value) if isinstance(value, tuple) else value for value in values)
            for key, values in run["sets"].items()
        }
        safe_runs.append(
            {
                "label": run["label"],
                "summary": run["summary"],
                "sets": safe_sets,
                "cases": run["cases"],
            }
        )
    return {
        "hit_k": payload["hit_k"],
        "runs": safe_runs,
        "pairwise": payload["pairwise"],
    }


def main() -> None:
    args = parse_args()
    if len(args.run) < 2:
        raise ValueError("Pass at least two --run entries to compare fixed-alpha outputs.")

    gold = load_gold(Path(args.gold))
    baseline = load_prediction(Path(args.baseline))
    runs = [
        build_run_cases(
            label=label,
            gold=gold,
            baseline=baseline,
            candidate=load_prediction(path),
            hit_k=int(args.hit_k),
        )
        for label, path in map(parse_labeled_path, args.run)
    ]
    pairs = []
    for left_idx in range(len(runs)):
        for right_idx in range(left_idx + 1, len(runs)):
            pairs.append(pair_summary(runs[left_idx], runs[right_idx]))

    payload = {
        "hit_k": int(args.hit_k),
        "runs": runs,
        "pairwise": pairs,
    }
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(make_json_safe(payload), indent=2), encoding="utf-8")
        print(f"saved_output_json={path}")

    text = render_markdown(payload, int(args.examples_per_section))
    if args.output_md:
        path = Path(args.output_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"saved_output_md={path}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
