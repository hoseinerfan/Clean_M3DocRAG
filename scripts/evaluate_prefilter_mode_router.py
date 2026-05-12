#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

from scripts.rerank_target_docs_visual_aware import route_rule_matches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a simple rule-based router that chooses between two merged prefilter audit runs "
            "using query route features."
        )
    )
    parser.add_argument("--run-default", required=True, help="Merged audit JSON for the default mode.")
    parser.add_argument("--run-specialist", required=True, help="Merged audit JSON for the specialist mode.")
    parser.add_argument("--default-label", default="default")
    parser.add_argument("--specialist-label", default="specialist")
    parser.add_argument(
        "--specialist-token",
        action="append",
        default=[],
        help="Add an OR rule that routes to the specialist if this informative visual token is present.",
    )
    parser.add_argument(
        "--specialist-rule-json",
        action="append",
        default=[],
        help="Explicit specialist rule JSON. Can be passed multiple times.",
    )
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument("--output-routed-qid-jsonl", help="Optional qid JSONL for qids routed to the specialist.")
    return parser.parse_args()


def load_audits(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    audits = payload.get("audits", [])
    return {str(row["qid"]): row for row in audits}


def rank_of(row: dict) -> int | None:
    checklist = row.get("reliability_checklist", {})
    value = checklist.get("gold_visual_rank")
    return None if value is None else int(value)


def median(values: list[int]) -> float | None:
    if not values:
        return None
    values_sorted = sorted(values)
    n = len(values_sorted)
    mid = n // 2
    if n % 2 == 1:
        return float(values_sorted[mid])
    return float(values_sorted[mid - 1] + values_sorted[mid]) / 2.0


def build_specialist_rules(args: argparse.Namespace) -> list[dict]:
    rules: list[dict] = []
    for token in args.specialist_token:
        token_text = str(token).strip()
        if token_text:
            rules.append({"any_informative_visual_tokens": [token_text]})
    for payload in args.specialist_rule_json:
        rule = json.loads(payload)
        if not isinstance(rule, dict):
            raise ValueError(f"Rule must decode to a JSON object: {payload!r}")
        rules.append(rule)
    if not rules:
        raise ValueError("Provide at least one --specialist-token or --specialist-rule-json.")
    return rules


def should_route_to_specialist(route_features: dict, rules: list[dict]) -> bool:
    return any(route_rule_matches(route_features=route_features, rule=rule) for rule in rules)


def summarize_ranks(rows: list[dict]) -> dict[str, object]:
    ranks = [int(row["chosen_rank"]) for row in rows if row.get("chosen_rank") is not None]
    return {
        "count": len(rows),
        "top50_gold_pages": sum(int(row["chosen_rank"]) <= 50 for row in rows if row.get("chosen_rank") is not None),
        "mean_gold_rank": (sum(ranks) / len(ranks)) if ranks else None,
        "median_gold_rank": median(ranks),
    }


def main() -> None:
    args = parse_args()
    rules = build_specialist_rules(args)

    default_run = load_audits(Path(args.run_default))
    specialist_run = load_audits(Path(args.run_specialist))
    shared_qids = sorted(set(default_run) & set(specialist_run))
    if not shared_qids:
        raise ValueError("No shared qids found between the two runs.")

    routed_rows: list[dict] = []
    specialist_qids: list[str] = []
    specialist_better = 0
    default_better = 0
    ties = 0
    for qid in shared_qids:
        default_row = default_run[qid]
        specialist_row = specialist_run[qid]
        route_features = dict(default_row.get("route_features", {}))
        route_to_specialist = should_route_to_specialist(route_features, rules)
        chosen_row = specialist_row if route_to_specialist else default_row
        chosen_rank = rank_of(chosen_row)
        default_rank = rank_of(default_row)
        specialist_rank = rank_of(specialist_row)
        if route_to_specialist:
            specialist_qids.append(qid)
        if default_rank is not None and specialist_rank is not None:
            if specialist_rank < default_rank:
                specialist_better += 1
            elif default_rank < specialist_rank:
                default_better += 1
            else:
                ties += 1
        routed_rows.append(
            {
                "qid": qid,
                "question": default_row["question"],
                "question_type": route_features.get("question_type", "UNKNOWN"),
                "informative_visual_query_tokens": route_features.get("informative_visual_query_tokens", []),
                "route_to_specialist": route_to_specialist,
                "chosen_mode": args.specialist_label if route_to_specialist else args.default_label,
                "chosen_rank": chosen_rank,
                "default_rank": default_rank,
                "specialist_rank": specialist_rank,
            }
        )

    summary = {
        "labels": {
            "default": args.default_label,
            "specialist": args.specialist_label,
        },
        "n_shared_qids": len(shared_qids),
        "specialist_rules": rules,
        "routed_to_specialist_qids": len(specialist_qids),
        "route_summary": summarize_ranks(routed_rows),
        "default_summary": summarize_ranks(
            [{"chosen_rank": rank_of(default_run[qid])} for qid in shared_qids]
        ),
        "specialist_summary": summarize_ranks(
            [{"chosen_rank": rank_of(specialist_run[qid])} for qid in shared_qids]
        ),
        "pairwise_mode_wins": {
            args.default_label: default_better,
            args.specialist_label: specialist_better,
            "tie": ties,
        },
        "routed_examples": [row for row in routed_rows if row["route_to_specialist"]][:50],
    }

    output_path = Path(args.output_summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.output_routed_qid_jsonl:
        qid_path = Path(args.output_routed_qid_jsonl)
        qid_path.parent.mkdir(parents=True, exist_ok=True)
        with qid_path.open("w", encoding="utf-8") as f:
            for qid in specialist_qids:
                f.write(json.dumps({"qid": qid}) + "\n")

    print("n_shared_qids:", len(shared_qids))
    print("routed_to_specialist_qids:", len(specialist_qids))
    print("route_summary:", summary["route_summary"])
    print("default_summary:", summary["default_summary"])
    print("specialist_summary:", summary["specialist_summary"])
    print("pairwise_mode_wins:", summary["pairwise_mode_wins"])
    print("saved_summary:", output_path)
    if args.output_routed_qid_jsonl:
        print("saved_routed_qids:", args.output_routed_qid_jsonl)


if __name__ == "__main__":
    main()
