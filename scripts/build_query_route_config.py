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

from scripts.rerank_target_docs_visual_aware import (
    BIG_RANK,
    LEARNED_QUERY_ROUTE_NUMERIC_FEATURE_NAMES,
    build_learned_query_route_feature_vector,
    clean_token_label,
    decide_query_route,
    extract_query_route_features,
    load_splice_query_axis_classes,
    resolve_model_path,
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
    parser.add_argument(
        "--router-type",
        default="learned_linear",
        choices=["learned_linear", "rule_or"],
        help="Routing model family. 'learned_linear' is the new default; 'rule_or' keeps the old greedy OR-of-rules search.",
    )
    parser.add_argument("--max-rules", type=int, default=1, help="Greedy OR-of-rules search depth when --router-type=rule_or.")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs for --router-type=learned_linear.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate for the learned linear router.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for the learned linear router.")
    parser.add_argument("--min-token-df", type=int, default=2, help="Minimum qid frequency for informative visual tokens to enter the learned router vocabulary.")
    parser.add_argument("--route-positive-weight", type=float, default=1.0, help="Extra loss weight for qids where the candidate arm beats the baseline.")
    parser.add_argument("--top4-transition-weight", type=float, default=3.0, help="Extra loss weight for qids where baseline and candidate differ on top-4 success.")
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


def evaluate_route_config(
    *,
    qids: list[str],
    baseline_rows: dict[str, dict],
    candidate_rows: dict[str, dict],
    route_features_by_qid: dict[str, dict],
    route_config: dict,
) -> dict:
    routed_top4_doc_count = 0
    improved_doc_rank_count = 0
    worsened_doc_rank_count = 0
    routed_visual_qids: list[str] = []

    for qid in qids:
        route_info = decide_query_route(
            route_config=route_config,
            route_features=route_features_by_qid[qid],
        )
        use_candidate = route_info["route_decision"] == str(route_config.get("candidate_route", "visual")).strip().lower()
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


def evaluate_selected_rules(
    *,
    qids: list[str],
    baseline_rows: dict[str, dict],
    candidate_rows: dict[str, dict],
    route_features_by_qid: dict[str, dict],
    selected_rules: list[dict],
) -> dict:
    return evaluate_route_config(
        qids=qids,
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
        route_features_by_qid=route_features_by_qid,
        route_config={
        "default_route": "base",
            "candidate_route": "visual",
        "visual_rules": selected_rules,
            "router_type": "rule_or",
        },
    )


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
    candidate_route = str(route_config.get("candidate_route", "visual")).strip().lower() or "visual"
    for qid in qids:
        route_info = decide_query_route(
            route_config=route_config,
            route_features=route_features_by_qid[qid],
        )
        use_candidate = route_info["route_decision"] == candidate_route
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


def build_learned_router_payload(
    *,
    route_features_by_qid: dict[str, dict],
    qids: list[str],
    min_token_df: int,
) -> dict:
    question_type_vocabulary = sorted(
        {
            str(route_features_by_qid[qid].get("question_type", "UNKNOWN")).strip() or "UNKNOWN"
            for qid in qids
        }
    )
    token_counter: Counter[str] = Counter()
    for qid in qids:
        for token in route_features_by_qid[qid].get("informative_visual_query_tokens", []) or []:
            normalized = str(token).strip()
            if normalized:
                token_counter[normalized] += 1
    token_vocabulary = sorted(
        token for token, count in token_counter.items() if int(count) >= max(1, int(min_token_df))
    )

    numeric_feature_names = list(LEARNED_QUERY_ROUTE_NUMERIC_FEATURE_NAMES)
    raw_numeric_rows: list[list[float]] = []
    for qid in qids:
        features = route_features_by_qid[qid]
        visual_count = float(int(features.get("visual_query_token_count", 0)))
        non_visual_count = float(int(features.get("non_visual_query_token_count", 0)))
        unknown_count = float(int(features.get("unknown_query_token_count", 0)))
        total_count = float(int(features.get("total_query_token_count", 0)))
        informative_count = float(int(features.get("informative_visual_query_count", 0)))
        raw_numeric_rows.append(
            [
                visual_count,
                non_visual_count,
                unknown_count,
                total_count,
                informative_count,
                1.0 if informative_count > 0.0 else 0.0,
                informative_count / max(1.0, visual_count),
                informative_count / max(1.0, total_count),
            ]
        )

    numeric_feature_means: list[float] = []
    numeric_feature_stds: list[float] = []
    for idx in range(len(numeric_feature_names)):
        column = [row[idx] for row in raw_numeric_rows]
        mean = sum(column) / max(1, len(column))
        variance = sum((value - mean) ** 2 for value in column) / max(1, len(column))
        std = variance ** 0.5
        numeric_feature_means.append(float(mean))
        numeric_feature_stds.append(float(std if std > 1e-8 else 1.0))

    return {
        "router_type": "learned_linear",
        "default_route": "base",
        "candidate_route": "visual",
        "numeric_feature_names": numeric_feature_names,
        "numeric_feature_means": numeric_feature_means,
        "numeric_feature_stds": numeric_feature_stds,
        "question_type_vocabulary": question_type_vocabulary,
        "token_vocabulary": token_vocabulary,
    }


def build_learned_router_dataset(
    *,
    qids: list[str],
    route_features_by_qid: dict[str, dict],
    baseline_rows: dict[str, dict],
    candidate_rows: dict[str, dict],
    model_payload: dict,
    route_positive_weight: float,
    top4_transition_weight: float,
):
    import torch

    features: list[list[float]] = []
    labels: list[float] = []
    weights: list[float] = []
    qid_order: list[str] = []

    for qid in qids:
        baseline_rank = rank_value(baseline_rows[qid].get("reranked_first_gold_doc_rank"))
        candidate_rank = rank_value(candidate_rows[qid].get("reranked_first_gold_doc_rank"))
        label = 1.0 if candidate_rank < baseline_rank else 0.0
        sample_weight = 1.0
        if label > 0.0:
            sample_weight *= max(1.0, float(route_positive_weight))
        baseline_top4 = baseline_rank <= 4
        candidate_top4 = candidate_rank <= 4
        if baseline_top4 != candidate_top4:
            sample_weight *= max(1.0, float(top4_transition_weight))
        features.append(
            build_learned_query_route_feature_vector(
                route_features=route_features_by_qid[qid],
                route_config=model_payload,
            )
        )
        labels.append(label)
        weights.append(sample_weight)
        qid_order.append(qid)

    x_tensor = torch.tensor(features, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.float32)
    w_tensor = torch.tensor(weights, dtype=torch.float32)
    return x_tensor, y_tensor, w_tensor, qid_order


def fit_learned_router(
    *,
    qids: list[str],
    baseline_rows: dict[str, dict],
    candidate_rows: dict[str, dict],
    route_features_by_qid: dict[str, dict],
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    min_token_df: int,
    route_positive_weight: float,
    top4_transition_weight: float,
) -> tuple[dict, dict]:
    import torch

    model_payload = build_learned_router_payload(
        route_features_by_qid=route_features_by_qid,
        qids=qids,
        min_token_df=min_token_df,
    )
    feature_names = (
        list(model_payload["numeric_feature_names"])
        + [f"question_type::{value}" for value in model_payload["question_type_vocabulary"]]
        + [f"informative_token::{value}" for value in model_payload["token_vocabulary"]]
    )
    x_tensor, y_tensor, sample_weights, _qid_order = build_learned_router_dataset(
        qids=qids,
        route_features_by_qid=route_features_by_qid,
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
        model_payload=model_payload,
        route_positive_weight=route_positive_weight,
        top4_transition_weight=top4_transition_weight,
    )

    if x_tensor.numel() == 0:
        raise ValueError("No route-training examples available.")

    weight_vector = torch.zeros(x_tensor.shape[1], dtype=torch.float32, requires_grad=True)
    bias = torch.zeros(1, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([weight_vector, bias], lr=float(learning_rate), weight_decay=float(weight_decay))

    best_record = None
    training_trace: list[dict] = []
    for epoch in range(max(1, int(epochs))):
        logits = x_tensor.matmul(weight_vector) + bias
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            y_tensor,
            reduction="none",
        )
        loss = (losses * sample_weights).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            probabilities = torch.sigmoid(x_tensor.matmul(weight_vector) + bias).cpu().tolist()

        unique_thresholds = sorted({0.0, 1.000001, *[float(value) for value in probabilities]})
        epoch_best_threshold = 0.5
        epoch_best_eval = None
        for threshold in unique_thresholds:
            trial_payload = dict(model_payload)
            trial_payload.update(
                {
                    "weights": [float(value) for value in weight_vector.detach().cpu().tolist()],
                    "bias": float(bias.detach().cpu().item()),
                    "probability_threshold": float(threshold),
                }
            )
            eval_result = evaluate_route_config(
                qids=qids,
                baseline_rows=baseline_rows,
                candidate_rows=candidate_rows,
                route_features_by_qid=route_features_by_qid,
                route_config=trial_payload,
            )
            if epoch_best_eval is None or eval_result["score_tuple"] > epoch_best_eval["score_tuple"]:
                epoch_best_eval = eval_result
                epoch_best_threshold = float(threshold)

        assert epoch_best_eval is not None
        epoch_record = {
            "epoch": epoch,
            "loss": float(loss.detach().cpu().item()),
            "threshold": epoch_best_threshold,
            "score_tuple": list(epoch_best_eval["score_tuple"]),
            "routed_top4_doc_count": epoch_best_eval["routed_top4_doc_count"],
            "improved_doc_rank_count": epoch_best_eval["improved_doc_rank_count"],
            "worsened_doc_rank_count": epoch_best_eval["worsened_doc_rank_count"],
            "routed_visual_qid_count": len(epoch_best_eval["routed_visual_qids"]),
            "weights": [float(value) for value in weight_vector.detach().cpu().tolist()],
            "bias": float(bias.detach().cpu().item()),
        }
        training_trace.append(epoch_record)
        if best_record is None or tuple(epoch_record["score_tuple"]) > tuple(best_record["score_tuple"]):
            best_record = epoch_record

    assert best_record is not None
    final_payload = dict(model_payload)
    top_weight_features = sorted(
        zip(feature_names, best_record["weights"]),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )[:20]
    final_payload.update(
        {
            "feature_names": feature_names,
            "weights": best_record["weights"],
            "bias": best_record["bias"],
            "probability_threshold": best_record["threshold"],
            "training_trace": training_trace,
            "best_epoch": best_record["epoch"],
            "top_weight_features": [
                {"feature_name": str(name), "weight": float(weight)}
                for name, weight in top_weight_features
            ],
        }
    )
    final_eval = evaluate_route_config(
        qids=qids,
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
        route_features_by_qid=route_features_by_qid,
        route_config=final_payload,
    )
    return final_payload, final_eval


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

    if args.router_type == "rule_or":
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
            "candidate_route": "visual",
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
    else:
        output, best_eval = fit_learned_router(
            qids=shared_qids,
            baseline_rows=baseline_rows,
            candidate_rows=candidate_rows,
            route_features_by_qid=route_features_by_qid,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            min_token_df=args.min_token_df,
            route_positive_weight=args.route_positive_weight,
            top4_transition_weight=args.top4_transition_weight,
        )
        output["baseline_label"] = args.baseline_label
        output["candidate_label"] = args.candidate_label
        output["search_summary"] = {
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
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "min_token_df": int(args.min_token_df),
            "route_positive_weight": float(args.route_positive_weight),
            "top4_transition_weight": float(args.top4_transition_weight),
            "best_epoch": output.get("best_epoch"),
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
        "router_type": output["router_type"],
        "selected_rules": output.get("visual_rules", []),
        "probability_threshold": output.get("probability_threshold"),
        "best_epoch": output.get("best_epoch"),
    }

    if args.output_routed_jsonl or args.output_summary_json:
        routed_rows = build_routed_rows(
            qids=shared_qids,
            baseline_rows=baseline_rows,
            candidate_rows=candidate_rows,
            route_features_by_qid=route_features_by_qid,
            route_config=output,
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
    if output["router_type"] == "rule_or":
        print("selected_rules:")
        for rule in output.get("visual_rules", []):
            print(" ", json.dumps(rule, sort_keys=True))
    else:
        print(f"router_type: {output['router_type']}")
        print(f"probability_threshold: {output.get('probability_threshold')}")
        print(f"best_epoch: {output.get('best_epoch')}")


if __name__ == "__main__":
    main()
