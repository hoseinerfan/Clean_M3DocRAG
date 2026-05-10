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

from scripts.rerank_target_docs_visual_aware import (
    BIG_RANK,
    clean_token_label,
    decide_query_route,
    extract_query_route_features,
    load_splice_query_axis_classes,
    resolve_model_path,
    route_rule_matches,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a simple binary query-routing config from two existing run JSONLs "
            "and optionally materialize the routed mixture offline."
        )
    )
    parser.add_argument("--baseline-run", required=True, help="Reference run_visual_rerank_batch JSONL")
    parser.add_argument("--candidate-run", required=True, help="Candidate-arm run_visual_rerank_batch JSONL")
    parser.add_argument("--baseline-label", default="baseline", help="Human-readable label for --baseline-run.")
    parser.add_argument("--candidate-label", default="candidate", help="Human-readable label for --candidate-run.")
    parser.add_argument("--gold", required=True, help="Path to MMQA_<split>.jsonl")
    parser.add_argument("--splice-query-token-labels", required=True)
    parser.add_argument("--retrieval_model_name_or_path", default="colpaligemma-3b-pt-448-base")
    parser.add_argument("--retrieval_adapter_model_name_or_path", default="colpali-v1.2")
    parser.add_argument("--query-token-filter", default="full", choices=["full", "drop_pad_like", "semantic_only"])
    parser.add_argument("--qid-jsonl", help="Optional JSONL subset of qids to fit on.")
    parser.add_argument("--qid-field", default="qid")
    parser.add_argument("--max-rules", type=int, default=1, help="Greedy OR-of-rules search depth.")
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to save the learned route config JSON.",
    )
    parser.add_argument(
        "--output-routed-jsonl",
        default="",
        help="Optional JSONL path to save the offline routed mixture of --baseline-run and --candidate-run.",
    )
    parser.add_argument(
        "--output-summary-json",
        default="",
        help="Optional JSON path to save an offline routed summary.",
    )
    return parser.parse_args()


def load_run(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("qid", "")).strip()
            if qid:
                rows[qid] = row
    if not rows:
        raise ValueError(f"No qids found in {path}")
    return rows


def load_subset_qids(path: Path, qid_field: str) -> list[str]:
    qids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get(qid_field, "")).strip()
            if qid:
                qids.append(qid)
    if not qids:
        raise ValueError(f"No qids found in {path} using field {qid_field!r}")
    return qids


def load_gold_rows(path: Path, qids: set[str]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            qid = str(row.get("qid", "")).strip()
            if qid in qids:
                rows[qid] = row
    missing = sorted(qids - set(rows))
    if missing:
        raise KeyError(f"Missing {len(missing)} qids in gold file: {missing[:10]}")
    return rows


def rank_value(value: int | None) -> int:
    return BIG_RANK if value is None else int(value)


def top4_count(rows: dict[str, dict], qids: list[str]) -> int:
    return sum(rank_value(rows[qid].get("reranked_first_gold_doc_rank")) <= 4 for qid in qids)


def evaluate_selected_rules(
    *,
    qids: list[str],
    baseline_rows: dict[str, dict],
    candidate_rows: dict[str, dict],
    route_features_by_qid: dict[str, dict],
    selected_rules: list[dict],
) -> dict:
    config = {
        "default_route": "base",
        "visual_rules": selected_rules,
    }
    routed_top4_doc_count = 0
    improved_doc_rank_count = 0
    worsened_doc_rank_count = 0
    routed_visual_qids: list[str] = []

    for qid in qids:
        route_info = decide_query_route(
            route_config=config,
            route_features=route_features_by_qid[qid],
        )
        use_candidate = route_info["route_decision"] == "visual"
        baseline_rank = rank_value(baseline_rows[qid].get("reranked_first_gold_doc_rank"))
        candidate_rank = rank_value(candidate_rows[qid].get("reranked_first_gold_doc_rank"))
        effective_rank = candidate_rank if use_candidate else baseline_rank
        if effective_rank <= 4:
            routed_top4_doc_count += 1
        if effective_rank < baseline_rank:
            improved_doc_rank_count += 1
        elif effective_rank > baseline_rank:
            worsened_doc_rank_count += 1
        if use_candidate:
            routed_visual_qids.append(qid)

    return {
        "routed_top4_doc_count": routed_top4_doc_count,
        "improved_doc_rank_count": improved_doc_rank_count,
        "worsened_doc_rank_count": worsened_doc_rank_count,
        "routed_visual_qids": routed_visual_qids,
        "score_tuple": (
            routed_top4_doc_count,
            improved_doc_rank_count,
            -worsened_doc_rank_count,
            -len(routed_visual_qids),
        ),
    }


def build_routed_rows(
    *,
    qids: list[str],
    baseline_rows: dict[str, dict],
    candidate_rows: dict[str, dict],
    route_features_by_qid: dict[str, dict],
    route_config: dict,
    baseline_label: str,
    candidate_label: str,
) -> list[dict]:
    routed_rows: list[dict] = []
    for qid in qids:
        route_info = decide_query_route(
            route_config=route_config,
            route_features=route_features_by_qid[qid],
        )
        use_candidate = route_info["route_decision"] == "visual"
        source_row = candidate_rows[qid] if use_candidate else baseline_rows[qid]
        routed_row = dict(source_row)
        routed_row["route_decision"] = route_info["route_decision"]
        routed_row["route_matched_rule_index"] = route_info["matched_rule_index"]
        routed_row["route_matched_rule"] = route_info["matched_rule"]
        routed_row["route_arm_label"] = candidate_label if use_candidate else baseline_label
        routed_row["route_baseline_label"] = baseline_label
        routed_row["route_candidate_label"] = candidate_label
        routed_rows.append(routed_row)
    return routed_rows


def candidate_rules(route_features_by_qid: dict[str, dict]) -> list[dict]:
    question_types = sorted(
        {
            features["question_type"]
            for features in route_features_by_qid.values()
            if str(features.get("question_type", "")).strip()
            and str(features.get("question_type", "")).strip() != "UNKNOWN"
        }
    )
    informative_counts = sorted(
        {
            int(features.get("informative_visual_query_count", 0))
            for features in route_features_by_qid.values()
            if int(features.get("informative_visual_query_count", 0)) > 0
        }
    )
    visual_counts = sorted(
        {
            int(features.get("visual_query_token_count", 0))
            for features in route_features_by_qid.values()
            if int(features.get("visual_query_token_count", 0)) > 0
        }
    )
    informative_tokens = sorted(
        {
            token
            for features in route_features_by_qid.values()
            for token in features.get("informative_visual_query_tokens", [])
        }
    )

    rules: list[dict] = []
    for question_type in question_types:
        rules.append({"question_types": [question_type]})
    for count in informative_counts:
        rules.append({"min_informative_visual_count": count})
    for count in visual_counts:
        rules.append({"min_visual_query_token_count": count})
    for token in informative_tokens:
        rules.append({"any_informative_visual_tokens": [token]})
    for question_type in question_types:
        for count in informative_counts:
            rules.append(
                {
                    "question_types": [question_type],
                    "min_informative_visual_count": count,
                }
            )
        for token in informative_tokens:
            rules.append(
                {
                    "question_types": [question_type],
                    "any_informative_visual_tokens": [token],
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


def main() -> None:
    args = parse_args()

    baseline_rows = load_run(Path(args.baseline_run))
    candidate_rows = load_run(Path(args.candidate_run))
    shared_qids = sorted(set(baseline_rows) & set(candidate_rows))
    if args.qid_jsonl:
        subset_qids = set(load_subset_qids(Path(args.qid_jsonl), args.qid_field))
        shared_qids = [qid for qid in shared_qids if qid in subset_qids]
    if not shared_qids:
        raise ValueError("No shared qids remain after intersecting runs and optional subset.")

    gold_rows = load_gold_rows(Path(args.gold), set(shared_qids))

    import torch
    from m3docrag.retrieval import ColPaliRetrievalModel

    retrieval_model = ColPaliRetrievalModel(
        backbone_name_or_path=resolve_model_path(args.retrieval_model_name_or_path),
        adapter_name_or_path=resolve_model_path(args.retrieval_adapter_model_name_or_path),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    route_features_by_qid: dict[str, dict] = {}
    with torch.no_grad():
        for qid in shared_qids:
            gold_row = gold_rows[qid]
            query_text = gold_row["question"]
            query_meta = retrieval_model.encode_query_with_metadata(
                query=query_text,
                to_cpu=True,
                query_token_filter=args.query_token_filter,
            )
            query_raw_tokens = query_meta.get("raw_tokens", [])
            query_token_labels = [clean_token_label(token) for token in query_raw_tokens]
            query_axis_classes = load_splice_query_axis_classes(
                query_labels_path=args.splice_query_token_labels,
                qid=qid,
                query_token_labels=query_token_labels,
                query_raw_tokens=query_raw_tokens,
            )
            route_features_by_qid[qid] = extract_query_route_features(
                question_type=str(gold_row.get("metadata", {}).get("type", "UNKNOWN")).strip() or "UNKNOWN",
                query_axis_classes=query_axis_classes,
                query_token_labels=query_token_labels,
            )

    baseline_top4_doc_count = top4_count(baseline_rows, shared_qids)
    candidate_top4_doc_count = top4_count(candidate_rows, shared_qids)
    oracle_top4_doc_count = sum(
        min(
            rank_value(baseline_rows[qid].get("reranked_first_gold_doc_rank")),
            rank_value(candidate_rows[qid].get("reranked_first_gold_doc_rank")),
        ) <= 4
        for qid in shared_qids
    )

    all_rules = candidate_rules(route_features_by_qid)
    selected_rules: list[dict] = []
    used_rule_keys: set[str] = set()
    best_eval = evaluate_selected_rules(
        qids=shared_qids,
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
        route_features_by_qid=route_features_by_qid,
        selected_rules=selected_rules,
    )
    search_trace: list[dict] = []

    for _ in range(max(0, int(args.max_rules))):
        iteration_best_rule = None
        iteration_best_eval = None
        for rule in all_rules:
            key = json.dumps(rule, sort_keys=True)
            if key in used_rule_keys:
                continue
            eval_result = evaluate_selected_rules(
                qids=shared_qids,
                baseline_rows=baseline_rows,
                candidate_rows=candidate_rows,
                route_features_by_qid=route_features_by_qid,
                selected_rules=selected_rules + [rule],
            )
            if iteration_best_eval is None or eval_result["score_tuple"] > iteration_best_eval["score_tuple"]:
                iteration_best_eval = eval_result
                iteration_best_rule = rule
        if iteration_best_rule is None or iteration_best_eval is None:
            break
        if iteration_best_eval["score_tuple"] <= best_eval["score_tuple"]:
            break
        selected_rules.append(iteration_best_rule)
        used_rule_keys.add(json.dumps(iteration_best_rule, sort_keys=True))
        best_eval = iteration_best_eval
        search_trace.append(
            {
                "rule": iteration_best_rule,
                "score_tuple": list(iteration_best_eval["score_tuple"]),
                "routed_top4_doc_count": iteration_best_eval["routed_top4_doc_count"],
                "improved_doc_rank_count": iteration_best_eval["improved_doc_rank_count"],
                "worsened_doc_rank_count": iteration_best_eval["worsened_doc_rank_count"],
                "routed_visual_qid_count": len(iteration_best_eval["routed_visual_qids"]),
            }
        )

    output = {
        "router_type": "rule_or",
        "default_route": "base",
        "visual_rules": selected_rules,
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "search_summary": {
            "baseline_run": args.baseline_run,
            "candidate_run": args.candidate_run,
            "baseline_label": args.baseline_label,
            "candidate_label": args.candidate_label,
            "gold": args.gold,
            "splice_query_token_labels": args.splice_query_token_labels,
            "query_token_filter": args.query_token_filter,
            "num_qids": len(shared_qids),
            "baseline_top4_doc_count": baseline_top4_doc_count,
            "candidate_top4_doc_count": candidate_top4_doc_count,
            "oracle_top4_doc_count": oracle_top4_doc_count,
            "routed_top4_doc_count": best_eval["routed_top4_doc_count"],
            "improved_doc_rank_count": best_eval["improved_doc_rank_count"],
            "worsened_doc_rank_count": best_eval["worsened_doc_rank_count"],
            "routed_visual_qid_count": len(best_eval["routed_visual_qids"]),
            "routed_visual_qids": best_eval["routed_visual_qids"],
            "search_trace": search_trace,
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    routed_summary = {
        "baseline_run": args.baseline_run,
        "candidate_run": args.candidate_run,
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "route_config_json": str(output_path),
        "num_qids": len(shared_qids),
        "baseline_top4_doc_count": baseline_top4_doc_count,
        "candidate_top4_doc_count": candidate_top4_doc_count,
        "oracle_top4_doc_count": oracle_top4_doc_count,
        "routed_top4_doc_count": best_eval["routed_top4_doc_count"],
        "improved_doc_rank_count": best_eval["improved_doc_rank_count"],
        "worsened_doc_rank_count": best_eval["worsened_doc_rank_count"],
        "routed_candidate_qid_count": len(best_eval["routed_visual_qids"]),
        "routed_candidate_qids": best_eval["routed_visual_qids"],
        "selected_rules": selected_rules,
    }

    if args.output_routed_jsonl or args.output_summary_json:
        route_config = {
            "default_route": "base",
            "visual_rules": selected_rules,
        }
        routed_rows = build_routed_rows(
            qids=shared_qids,
            baseline_rows=baseline_rows,
            candidate_rows=candidate_rows,
            route_features_by_qid=route_features_by_qid,
            route_config=route_config,
            baseline_label=args.baseline_label,
            candidate_label=args.candidate_label,
        )
        if args.output_routed_jsonl:
            routed_path = Path(args.output_routed_jsonl)
            routed_path.parent.mkdir(parents=True, exist_ok=True)
            with routed_path.open("w", encoding="utf-8") as handle:
                for row in routed_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"saved_jsonl: {routed_path}")
        if args.output_summary_json:
            summary_path = Path(args.output_summary_json)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(routed_summary, indent=2) + "\n", encoding="utf-8")
            print(f"saved_summary: {summary_path}")

    print(f"saved_route_config: {output_path}")
    print(f"num_qids: {len(shared_qids)}")
    print(f"baseline_top4_doc_count: {baseline_top4_doc_count}")
    print(f"candidate_top4_doc_count: {candidate_top4_doc_count}")
    print(f"oracle_top4_doc_count: {oracle_top4_doc_count}")
    print(f"routed_top4_doc_count: {best_eval['routed_top4_doc_count']}")
    print(f"routed_visual_qid_count: {len(best_eval['routed_visual_qids'])}")
    print("selected_rules:")
    for rule in selected_rules:
        print(" ", json.dumps(rule, sort_keys=True))


if __name__ == "__main__":
    main()
