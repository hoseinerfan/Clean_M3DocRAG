#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
            "Compare two merged qid-audit JSON files, extract winner subsets, and summarize "
            "simple route-feature rules that separate the winner families."
        )
    )
    parser.add_argument("--run-a", required=True, help="Merged audit JSON for mode A.")
    parser.add_argument("--run-b", required=True, help="Merged audit JSON for mode B.")
    parser.add_argument("--label-a", default="mode_a")
    parser.add_argument("--label-b", default="mode_b")
    parser.add_argument("--qid-field", default="qid")
    parser.add_argument("--top-k", type=int, default=20, help="How many top question types / tokens / rules to keep.")
    parser.add_argument("--min-rule-support", type=int, default=3)
    parser.add_argument("--min-rule-precision", type=float, default=0.70)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument("--output-a-qid-jsonl", help="Optional qid JSONL for qids where mode A wins.")
    parser.add_argument("--output-b-qid-jsonl", help="Optional qid JSONL for qids where mode B wins.")
    parser.add_argument("--output-tie-qid-jsonl", help="Optional qid JSONL for tied qids.")
    return parser.parse_args()


def load_audits(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    audits = payload.get("audits", [])
    return {str(row["qid"]): row for row in audits}


def rank_of(row: dict) -> int | None:
    checklist = row.get("reliability_checklist", {})
    value = checklist.get("gold_visual_rank")
    return None if value is None else int(value)


def token_list(row: dict) -> list[str]:
    route_features = row.get("route_features", {})
    return [str(token).strip() for token in route_features.get("informative_visual_query_tokens", []) if str(token).strip()]


def question_type(row: dict) -> str:
    route_features = row.get("route_features", {})
    return str(route_features.get("question_type", "UNKNOWN")).strip() or "UNKNOWN"


def route_features(row: dict) -> dict:
    return dict(row.get("route_features", {}))


def write_qid_jsonl(path: Path, rows: list[dict], qid_field: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({qid_field: row["qid"]}) + "\n")


def summarize_bucket(rows: list[dict]) -> dict[str, object]:
    ranks = [int(row["winner_rank"]) for row in rows if row.get("winner_rank") is not None]
    qtype_counts = Counter(question_type(row) for row in rows)
    token_counts = Counter(token for row in rows for token in token_list(row))
    return {
        "count": len(rows),
        "mean_rank": (sum(ranks) / len(ranks)) if ranks else None,
        "median_rank": median(ranks) if ranks else None,
        "question_type_counts": [
            {"question_type": key, "count": value}
            for key, value in qtype_counts.most_common()
        ],
        "informative_visual_token_counts": [
            {"token": key, "count": value}
            for key, value in token_counts.most_common()
        ],
    }


def median(values: list[int]) -> float | None:
    if not values:
        return None
    values_sorted = sorted(values)
    n = len(values_sorted)
    mid = n // 2
    if n % 2 == 1:
        return float(values_sorted[mid])
    return float(values_sorted[mid - 1] + values_sorted[mid]) / 2.0


def candidate_rules(features_by_qid: dict[str, dict]) -> list[dict]:
    question_types = sorted(
        {
            str(features.get("question_type", "")).strip()
            for features in features_by_qid.values()
            if str(features.get("question_type", "")).strip()
            and str(features.get("question_type", "")).strip() != "UNKNOWN"
        }
    )
    informative_counts = sorted(
        {
            int(features.get("informative_visual_query_count", 0))
            for features in features_by_qid.values()
            if int(features.get("informative_visual_query_count", 0)) > 0
        }
    )
    visual_counts = sorted(
        {
            int(features.get("visual_query_token_count", 0))
            for features in features_by_qid.values()
            if int(features.get("visual_query_token_count", 0)) > 0
        }
    )
    informative_tokens = sorted(
        {
            str(token).strip()
            for features in features_by_qid.values()
            for token in features.get("informative_visual_query_tokens", []) or []
            if str(token).strip()
        }
    )

    rules: list[dict] = []
    for question_type_value in question_types:
        rules.append({"question_types": [question_type_value]})
    for count in informative_counts:
        rules.append({"min_informative_visual_count": count})
    for count in visual_counts:
        rules.append({"min_visual_query_token_count": count})
    for token in informative_tokens:
        rules.append({"any_informative_visual_tokens": [token]})
    for question_type_value in question_types:
        for token in informative_tokens:
            rules.append(
                {
                    "question_types": [question_type_value],
                    "any_informative_visual_tokens": [token],
                }
            )
        for count in informative_counts:
            rules.append(
                {
                    "question_types": [question_type_value],
                    "min_informative_visual_count": count,
                }
            )

    deduped: list[dict] = []
    seen: set[str] = set()
    for rule in rules:
        key = json.dumps(rule, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rule)
    return deduped


def evaluate_rules(
    *,
    all_rows: list[dict],
    target_winner_label: str,
    min_support: int,
    min_precision: float,
    top_k: int,
) -> list[dict]:
    features_by_qid = {row["qid"]: route_features(row) for row in all_rows}
    target_total = sum(1 for row in all_rows if row["winner"] == target_winner_label)
    scored_rules: list[dict] = []
    for rule in candidate_rules(features_by_qid):
        matched = [row for row in all_rows if route_rule_matches(route_features=features_by_qid[row["qid"]], rule=rule)]
        support = len(matched)
        if support < min_support:
            continue
        target_hits = sum(1 for row in matched if row["winner"] == target_winner_label)
        other_hits = sum(1 for row in matched if row["winner"] not in {target_winner_label, "tie"})
        tie_hits = sum(1 for row in matched if row["winner"] == "tie")
        precision = target_hits / support
        if precision < min_precision:
            continue
        recall = (target_hits / target_total) if target_total > 0 else 0.0
        scored_rules.append(
            {
                "rule": rule,
                "support": support,
                "target_hits": target_hits,
                "other_hits": other_hits,
                "tie_hits": tie_hits,
                "precision": precision,
                "recall": recall,
            }
        )
    scored_rules.sort(
        key=lambda item: (
            -float(item["precision"]),
            -int(item["support"]),
            -float(item["recall"]),
            json.dumps(item["rule"], sort_keys=True),
        )
    )
    return scored_rules[:top_k]


def main() -> None:
    args = parse_args()

    run_a = load_audits(Path(args.run_a))
    run_b = load_audits(Path(args.run_b))
    shared_qids = sorted(set(run_a) & set(run_b))
    if not shared_qids:
        raise ValueError("No shared qids found between the two runs.")

    rows: list[dict] = []
    for qid in shared_qids:
        row_a = run_a[qid]
        row_b = run_b[qid]
        rank_a = rank_of(row_a)
        rank_b = rank_of(row_b)
        if rank_a is None or rank_b is None:
            winner = "tie"
            winner_rank = None
        elif rank_a < rank_b:
            winner = args.label_a
            winner_rank = rank_a
        elif rank_b < rank_a:
            winner = args.label_b
            winner_rank = rank_b
        else:
            winner = "tie"
            winner_rank = rank_a
        rows.append(
            {
                "qid": qid,
                "question": row_a["question"],
                "question_type": question_type(row_a),
                "route_features": route_features(row_a),
                "rank_a": rank_a,
                "rank_b": rank_b,
                "verdict_a": row_a.get("reliability_checklist", {}).get("verdict"),
                "verdict_b": row_b.get("reliability_checklist", {}).get("verdict"),
                "winner": winner,
                "winner_rank": winner_rank,
            }
        )

    a_wins = [row for row in rows if row["winner"] == args.label_a]
    b_wins = [row for row in rows if row["winner"] == args.label_b]
    ties = [row for row in rows if row["winner"] == "tie"]

    if args.output_a_qid_jsonl:
        write_qid_jsonl(Path(args.output_a_qid_jsonl), a_wins, args.qid_field)
    if args.output_b_qid_jsonl:
        write_qid_jsonl(Path(args.output_b_qid_jsonl), b_wins, args.qid_field)
    if args.output_tie_qid_jsonl:
        write_qid_jsonl(Path(args.output_tie_qid_jsonl), ties, args.qid_field)

    summary = {
        "labels": {
            "run_a": args.label_a,
            "run_b": args.label_b,
        },
        "n_shared_qids": len(shared_qids),
        "winner_counts": {
            args.label_a: len(a_wins),
            args.label_b: len(b_wins),
            "tie": len(ties),
        },
        "run_a_wins_summary": summarize_bucket(a_wins),
        "run_b_wins_summary": summarize_bucket(b_wins),
        "ties_summary": summarize_bucket(ties),
        "top_rules_for_run_a": evaluate_rules(
            all_rows=rows,
            target_winner_label=args.label_a,
            min_support=args.min_rule_support,
            min_precision=args.min_rule_precision,
            top_k=args.top_k,
        ),
        "top_rules_for_run_b": evaluate_rules(
            all_rows=rows,
            target_winner_label=args.label_b,
            min_support=args.min_rule_support,
            min_precision=args.min_rule_precision,
            top_k=args.top_k,
        ),
        "run_a_win_examples": a_wins[: args.top_k],
        "run_b_win_examples": b_wins[: args.top_k],
    }

    output_path = Path(args.output_summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("n_shared_qids:", len(shared_qids))
    print(f"{args.label_a}_wins:", len(a_wins))
    print(f"{args.label_b}_wins:", len(b_wins))
    print("ties:", len(ties))
    print("saved_summary:", output_path)
    if args.output_a_qid_jsonl:
        print("saved_a_qids:", args.output_a_qid_jsonl)
    if args.output_b_qid_jsonl:
        print("saved_b_qids:", args.output_b_qid_jsonl)
    if args.output_tie_qid_jsonl:
        print("saved_tie_qids:", args.output_tie_qid_jsonl)


if __name__ == "__main__":
    main()
