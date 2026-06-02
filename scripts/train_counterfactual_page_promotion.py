#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import train_content_aware_pseudo_page_reranker as ca


DEFAULT_THRESHOLD_GRID = "0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70,0.80"

ACTION_FEATURE_NAMES = [
    "candidate_rank_minus_repair_k_log",
    "candidate_rank_over_pool",
    "candidate_rank_over_max_rank",
    "candidate_doc_in_topk",
    "candidate_doc_count_in_topk",
    "candidate_doc_best_topk_rank_recip",
    "candidate_adds_new_doc",
    "topk_unique_doc_count_log",
    "topk_max_base_norm_score",
    "topk_mean_base_norm_score",
    "candidate_minus_topk_max_base_norm_score",
    "candidate_minus_topk_mean_base_norm_score",
    "topk_max_question_token_recall",
    "topk_mean_question_token_recall",
    "candidate_minus_topk_max_question_token_recall",
    "candidate_minus_topk_mean_question_token_recall",
    "topk_max_anchor_token_recall",
    "topk_mean_anchor_token_recall",
    "candidate_minus_topk_max_anchor_token_recall",
    "candidate_minus_topk_mean_anchor_token_recall",
    "topk_max_phrase_match_fraction",
    "topk_mean_phrase_match_fraction",
    "candidate_minus_topk_max_phrase_match_fraction",
    "candidate_minus_topk_mean_phrase_match_fraction",
    "topk_max_source_present_count",
    "topk_mean_source_present_count",
    "candidate_minus_topk_max_source_present_count",
    "candidate_minus_topk_mean_source_present_count",
]

FEATURE_NAMES = ca.FEATURE_NAMES + ACTION_FEATURE_NAMES
FEATURE_INDEX = {name: idx for idx, name in enumerate(ca.FEATURE_NAMES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a counterfactual top-k page-promotion model. The model learns whether "
            "promoting a candidate page would repair the current top-k evidence set, rather "
            "than scoring every page as independently relevant."
        )
    )
    parser.add_argument("--train-gold", required=True)
    parser.add_argument("--eval-gold", required=True)
    parser.add_argument("--train-base-pred", required=True)
    parser.add_argument("--eval-base-pred", required=True)
    parser.add_argument("--train-page-text-jsonl", required=True)
    parser.add_argument("--eval-page-text-jsonl", required=True)
    parser.add_argument("--train-source", action="append", default=[], help="Optional LABEL=prediction.json")
    parser.add_argument("--eval-source", action="append", default=[], help="Optional LABEL=prediction.json")
    parser.add_argument("--candidate-top-k", type=int, default=1000)
    parser.add_argument("--repair-hit-k", type=int, default=5)
    parser.add_argument("--insert-rank", type=int, default=5)
    parser.add_argument("--promotion-rank-min", type=int, default=6)
    parser.add_argument("--promotion-rank-max", type=int, default=200)
    parser.add_argument("--max-promotions-per-qid", type=int, default=1)
    parser.add_argument("--negatives-per-band", type=int, default=24)
    parser.add_argument("--max-negatives-per-qid", type=int, default=160)
    parser.add_argument("--already-hit-negatives-per-qid", type=int, default=48)
    parser.add_argument("--positive-weight-cap", type=float, default=50.0)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-samples-leaf", type=int, default=30)
    parser.add_argument("--l2-regularization", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--auto-tune-threshold", action="store_true")
    parser.add_argument("--tune-fraction", type=float, default=0.20)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--threshold-grid", default=DEFAULT_THRESHOLD_GRID)
    parser.add_argument("--lost-penalty", type=float, default=1.0)
    parser.add_argument("--promotion-penalty", type=float, default=0.002)
    parser.add_argument("--skip-retrain-after-tuning", action="store_true")
    parser.add_argument("--recall-k", type=int, nargs="+", default=ca.DEFAULT_RECALL_KS)
    parser.add_argument("--output-model-json", required=True)
    parser.add_argument("--output-prediction-json", required=True)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument("--output-table-md", default="")
    parser.add_argument("--output-actions-jsonl", default="")
    return parser.parse_args()


def parse_threshold_grid(raw: str) -> list[float]:
    values: list[float] = []
    seen: set[float] = set()
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {value}")
        key = round(value, 8)
        if key in seen:
            continue
        seen.add(key)
        values.append(value)
    if not values:
        raise ValueError("Empty threshold grid.")
    return values


def feature_rows_for_qid(
    *,
    qid: str,
    gold_row: dict[str, Any],
    records: list[dict[str, Any]],
    page_features: dict[str, dict[str, Any]],
    source_maps_by_label: dict[str, dict[str, dict[str, float]]],
) -> np.ndarray:
    if not records:
        return np.zeros((0, len(ca.FEATURE_NAMES)), dtype=np.float32)
    base_norm_scores = ca.normalize_scores(records)
    doc_rank, doc_page_rank, doc_page_count = ca.doc_rank_maps(records)
    q_profile = ca.question_profile(gold_row)
    vectors = [
        ca.feature_vector(
            record,
            records=records,
            base_norm_scores=base_norm_scores,
            doc_rank=doc_rank,
            doc_page_rank=doc_page_rank,
            doc_page_count=doc_page_count,
            question=q_profile,
            page_features=page_features,
            source_maps_by_label=source_maps_by_label,
            qid=qid,
        )
        for record in records
    ]
    return np.asarray(vectors, dtype=np.float32)


def feature_value(vector: np.ndarray, name: str) -> float:
    return float(vector[FEATURE_INDEX[name]])


def aggregate(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return float(max(values)), float(sum(values) / len(values))


def action_features(
    *,
    record: dict[str, Any],
    record_vector: np.ndarray,
    records: list[dict[str, Any]],
    vectors: np.ndarray,
    args: argparse.Namespace,
) -> list[float]:
    repair_k = int(args.repair_hit_k)
    rank = int(record["base_rank"])
    pool_size = max(len(records), 1)
    max_rank = max(int(args.promotion_rank_max), 1)
    top_indices = list(range(min(repair_k, len(records))))
    top_records = [records[idx] for idx in top_indices]
    top_vectors = [vectors[idx] for idx in top_indices]
    candidate_doc = str(record["doc_id"])
    top_doc_ranks = [
        int(row["base_rank"])
        for row in top_records
        if str(row["doc_id"]) == candidate_doc
    ]
    top_docs = {str(row["doc_id"]) for row in top_records}

    def top_stats(name: str) -> tuple[float, float, float, float]:
        cand = feature_value(record_vector, name)
        top_max, top_mean = aggregate([feature_value(vector, name) for vector in top_vectors])
        return top_max, top_mean, cand - top_max, cand - top_mean

    base_max, base_mean, base_delta_max, base_delta_mean = top_stats("base_norm_score")
    q_max, q_mean, q_delta_max, q_delta_mean = top_stats("question_token_recall")
    anchor_max, anchor_mean, anchor_delta_max, anchor_delta_mean = top_stats("anchor_token_recall")
    phrase_max, phrase_mean, phrase_delta_max, phrase_delta_mean = top_stats("phrase_match_fraction")
    source_max, source_mean, source_delta_max, source_delta_mean = top_stats("source_present_count")

    return [
        math.log1p(float(max(rank - repair_k, 0))),
        float(rank) / float(pool_size),
        float(rank) / float(max_rank),
        1.0 if top_doc_ranks else 0.0,
        float(len(top_doc_ranks)),
        0.0 if not top_doc_ranks else 1.0 / float(min(top_doc_ranks)),
        0.0 if candidate_doc in top_docs else 1.0,
        math.log1p(float(len(top_docs))),
        base_max,
        base_mean,
        base_delta_max,
        base_delta_mean,
        q_max,
        q_mean,
        q_delta_max,
        q_delta_mean,
        anchor_max,
        anchor_mean,
        anchor_delta_max,
        anchor_delta_mean,
        phrase_max,
        phrase_mean,
        phrase_delta_max,
        phrase_delta_mean,
        source_max,
        source_mean,
        source_delta_max,
        source_delta_mean,
    ]


def counterfactual_vector(
    *,
    record_idx: int,
    records: list[dict[str, Any]],
    vectors: np.ndarray,
    args: argparse.Namespace,
) -> list[float]:
    base_vector = vectors[record_idx]
    action_vector = action_features(
        record=records[record_idx],
        record_vector=base_vector,
        records=records,
        vectors=vectors,
        args=args,
    )
    return [float(value) for value in base_vector.tolist()] + action_vector


def base_page_hit(records: list[dict[str, Any]], positive_uids: set[str], hit_k: int) -> bool:
    return any(str(row["uid"]) in positive_uids for row in records[: int(hit_k)])


def candidate_indices(records: list[dict[str, Any]], args: argparse.Namespace) -> list[int]:
    lo = max(int(args.promotion_rank_min), int(args.repair_hit_k) + 1)
    hi = int(args.promotion_rank_max)
    return [
        idx
        for idx, record in enumerate(records)
        if lo <= int(record["base_rank"]) <= hi
    ]


def sample_negative_indices(
    *,
    records: list[dict[str, Any]],
    candidate_idxs: list[int],
    positive_idxs: set[int],
    base_hit: bool,
    args: argparse.Namespace,
    rng: random.Random,
) -> list[int]:
    if base_hit:
        pool = [idx for idx in candidate_idxs if idx not in positive_idxs]
        rng.shuffle(pool)
        return pool[: int(args.already_hit_negatives_per_qid)]

    selected: list[int] = []
    selected_set: set[int] = set()
    bands = [
        (int(args.promotion_rank_min), 20),
        (21, 50),
        (51, 100),
        (101, int(args.promotion_rank_max)),
    ]
    for lo, hi in bands:
        band = [
            idx
            for idx in candidate_idxs
            if idx not in positive_idxs
            and idx not in selected_set
            and lo <= int(records[idx]["base_rank"]) <= hi
        ]
        rng.shuffle(band)
        for idx in band[: int(args.negatives_per_band)]:
            selected.append(idx)
            selected_set.add(idx)
            if len(selected) >= int(args.max_negatives_per_qid):
                return selected
    return selected[: int(args.max_negatives_per_qid)]


def build_counterfactual_matrix(
    *,
    gold: dict[str, dict[str, Any]],
    base_pred: dict[str, dict[str, Any]],
    page_features: dict[str, dict[str, Any]],
    source_maps_by_label: dict[str, dict[str, dict[str, float]]],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rows: list[list[float]] = []
    labels: list[int] = []
    stats = Counter()
    rng = random.Random(int(args.seed))

    for qid, gold_row in gold.items():
        positive_uids = ca.gold_page_uids(gold_row)
        if not positive_uids:
            stats["skipped_no_page_gold"] += 1
            continue
        records = ca.ranked_page_records(base_pred.get(qid), int(args.candidate_top_k))
        if not records:
            stats["skipped_missing_prediction"] += 1
            continue
        stats["qid_with_page_gold_and_base"] += 1
        candidate_idxs = candidate_indices(records, args)
        if not candidate_idxs:
            stats["skipped_no_candidate_range"] += 1
            continue

        hit = base_page_hit(records, positive_uids, int(args.repair_hit_k))
        positive_idxs = {
            idx
            for idx in candidate_idxs
            if (not hit) and str(records[idx]["uid"]) in positive_uids
        }
        if not hit and not positive_idxs:
            stats["skipped_miss_without_repair_candidate"] += 1
            continue

        negative_idxs = sample_negative_indices(
            records=records,
            candidate_idxs=candidate_idxs,
            positive_idxs=positive_idxs,
            base_hit=hit,
            args=args,
            rng=rng,
        )
        selected = sorted(positive_idxs | set(negative_idxs), key=lambda idx: int(records[idx]["base_rank"]))
        if len(selected) < 2:
            stats["skipped_too_few_training_rows"] += 1
            continue

        vectors = feature_rows_for_qid(
            qid=qid,
            gold_row=gold_row,
            records=records,
            page_features=page_features,
            source_maps_by_label=source_maps_by_label,
        )
        for idx in selected:
            label = 1 if idx in positive_idxs else 0
            rows.append(counterfactual_vector(record_idx=idx, records=records, vectors=vectors, args=args))
            labels.append(label)
            stats["positive_count" if label else "negative_count"] += 1
        stats["train_qid_count"] += 1
        stats["base_hit_qid_count" if hit else "base_miss_qid_count"] += 1
        if positive_idxs:
            stats["repairable_miss_qid_count"] += 1

    if not rows:
        raise ValueError("No counterfactual training rows were produced.")
    if not any(labels):
        raise ValueError("No positive counterfactual repair examples were produced.")

    metadata = {key: int(value) for key, value in stats.items()}
    metadata["row_count"] = int(len(rows))
    metadata["positive_fraction"] = float(sum(labels) / max(len(labels), 1))
    metadata["feature_count"] = int(len(FEATURE_NAMES))
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int32), metadata


def train_classifier(X: np.ndarray, y: np.ndarray, args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("scikit-learn is required for counterfactual page promotion.") from exc

    positives = float(y.sum())
    negatives = float(len(y) - y.sum())
    pos_weight = min(float(args.positive_weight_cap), negatives / max(positives, 1.0))
    sample_weight = np.where(y > 0, pos_weight, 1.0).astype(np.float32)
    model = HistGradientBoostingClassifier(
        max_iter=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        max_leaf_nodes=int(args.num_leaves),
        min_samples_leaf=int(args.min_samples_leaf),
        l2_regularization=float(args.l2_regularization),
        random_state=int(args.seed),
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model, {
        "backend": "sklearn",
        "objective": "counterfactual_binary_repair",
        "params": {
            "n_estimators": int(args.n_estimators),
            "learning_rate": float(args.learning_rate),
            "num_leaves": int(args.num_leaves),
            "min_samples_leaf": int(args.min_samples_leaf),
            "l2_regularization": float(args.l2_regularization),
            "positive_weight": float(pos_weight),
            "seed": int(args.seed),
        },
    }


def predict_repair_scores(model: Any, X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return np.asarray([], dtype=np.float32)
    if hasattr(model, "predict_proba"):
        values = model.predict_proba(X)
        if values.ndim == 2 and values.shape[1] >= 2:
            return np.asarray(values[:, 1], dtype=np.float32)
    return np.asarray(model.predict(X), dtype=np.float32)


def score_candidate_records(
    *,
    qid: str,
    gold_row: dict[str, Any],
    records: list[dict[str, Any]],
    page_features: dict[str, dict[str, Any]],
    source_maps_by_label: dict[str, dict[str, dict[str, float]]],
    model: Any,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    candidate_idxs = candidate_indices(records, args)
    if not records or not candidate_idxs:
        return []
    vectors = feature_rows_for_qid(
        qid=qid,
        gold_row=gold_row,
        records=records,
        page_features=page_features,
        source_maps_by_label=source_maps_by_label,
    )
    X = np.asarray(
        [counterfactual_vector(record_idx=idx, records=records, vectors=vectors, args=args) for idx in candidate_idxs],
        dtype=np.float32,
    )
    scores = predict_repair_scores(model, X)
    scored: list[dict[str, Any]] = []
    for idx, score in zip(candidate_idxs, scores):
        row = dict(records[idx])
        row["counterfactual_score"] = float(score)
        scored.append(row)
    return scored


def insert_promotions(
    records: list[dict[str, Any]],
    promotions: list[dict[str, Any]],
    *,
    insert_rank: int,
) -> list[dict[str, Any]]:
    if not promotions:
        return list(records)
    promoted_uids = {str(row["uid"]) for row in promotions}
    remaining = [row for row in records if str(row["uid"]) not in promoted_uids]
    insert_idx = max(0, min(int(insert_rank) - 1, len(remaining)))
    return remaining[:insert_idx] + promotions + remaining[insert_idx:]


def choose_promotions(scored_candidates: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in scored_candidates
        if float(row.get("counterfactual_score", 0.0)) >= float(args.threshold)
    ]
    eligible.sort(
        key=lambda row: (
            -float(row.get("counterfactual_score", 0.0)),
            int(row["base_rank"]),
            str(row["uid"]),
        )
    )
    return eligible[: int(args.max_promotions_per_qid)]


def apply_counterfactual_promoter(
    *,
    gold: dict[str, dict[str, Any]],
    base_pred: dict[str, dict[str, Any]],
    page_features: dict[str, dict[str, Any]],
    source_maps_by_label: dict[str, dict[str, dict[str, float]]],
    model: Any,
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    output: dict[str, dict[str, Any]] = {}
    action_rows: list[dict[str, Any]] = []

    for qid, base_row in base_pred.items():
        records = ca.ranked_page_records(base_row, int(args.candidate_top_k))
        gold_row = gold.get(qid, {"qid": qid, "question": base_row.get("question", "")})
        scored_candidates = score_candidate_records(
            qid=qid,
            gold_row=gold_row,
            records=records,
            page_features=page_features,
            source_maps_by_label=source_maps_by_label,
            model=model,
            args=args,
        )
        promotions = choose_promotions(scored_candidates, args)
        promotion_uids = {str(row["uid"]) for row in promotions}
        reranked = insert_promotions(records, promotions, insert_rank=int(args.insert_rank))
        reranked_uids = {str(row["uid"]) for row in reranked}
        raw_by_uid = {str(row["uid"]): row["raw"] for row in records}
        output_rows = [raw_by_uid[str(row["uid"])] for row in reranked if str(row["uid"]) in raw_by_uid]
        output_rows.extend(
            raw_by_uid[str(row["uid"])]
            for row in records
            if str(row["uid"]) not in reranked_uids and str(row["uid"]) in raw_by_uid
        )

        out_row = dict(base_row)
        if "page_retrieval_results" in out_row:
            out_row["page_retrieval_results"] = output_rows
        elif "retrieval_results" in out_row:
            out_row["retrieval_results"] = output_rows
        else:
            out_row["page_retrieval_results"] = output_rows
        out_row["reranker_metadata"] = {
            **(out_row.get("reranker_metadata", {}) if isinstance(out_row.get("reranker_metadata"), dict) else {}),
            "counterfactual_page_promotion": {
                "repair_hit_k": int(args.repair_hit_k),
                "insert_rank": int(args.insert_rank),
                "promotion_rank_min": int(args.promotion_rank_min),
                "promotion_rank_max": int(args.promotion_rank_max),
                "max_promotions_per_qid": int(args.max_promotions_per_qid),
                "threshold": float(args.threshold),
            },
        }
        output[qid] = out_row

        for row in scored_candidates:
            action_rows.append(
                {
                    "qid": qid,
                    "page_uid": row["uid"],
                    "doc_id": row["doc_id"],
                    "page_idx": int(row["page_idx"]),
                    "base_rank": int(row["base_rank"]),
                    "counterfactual_score": float(row.get("counterfactual_score", 0.0)),
                    "promoted": str(row["uid"]) in promotion_uids,
                    "threshold": float(args.threshold),
                }
            )
    return output, action_rows


def tune_threshold(
    *,
    tune_gold: dict[str, dict[str, Any]],
    base_pred: dict[str, dict[str, Any]],
    page_features: dict[str, dict[str, Any]],
    source_maps_by_label: dict[str, dict[str, dict[str, float]]],
    model: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    thresholds = parse_threshold_grid(str(args.threshold_grid))
    rows_by_threshold: dict[float, Counter[str]] = {value: Counter() for value in thresholds}
    evaluated = 0
    skipped_no_page_gold = 0
    skipped_missing_prediction = 0

    for qid, gold_row in tune_gold.items():
        pages_gold = ca.gold_page_uids(gold_row)
        if not pages_gold:
            skipped_no_page_gold += 1
            continue
        records = ca.ranked_page_records(base_pred.get(qid), int(args.candidate_top_k))
        if not records:
            skipped_missing_prediction += 1
            continue
        base_hit = base_page_hit(records, pages_gold, int(args.repair_hit_k))
        scored_candidates = score_candidate_records(
            qid=qid,
            gold_row=gold_row,
            records=records,
            page_features=page_features,
            source_maps_by_label=source_maps_by_label,
            model=model,
            args=args,
        )
        evaluated += 1
        for threshold in thresholds:
            candidate_args = copy.copy(args)
            candidate_args.threshold = float(threshold)
            promotions = choose_promotions(scored_candidates, candidate_args)
            reranked = insert_promotions(records, promotions, insert_rank=int(args.insert_rank))
            cand_hit = base_page_hit(reranked, pages_gold, int(args.repair_hit_k))
            counter = rows_by_threshold[threshold]
            counter["hit"] += int(cand_hit)
            counter["base_hit"] += int(base_hit)
            counter["recovered"] += int((not base_hit) and cand_hit)
            counter["lost"] += int(base_hit and not cand_hit)
            counter["promotions"] += len(promotions)

    if evaluated <= 0:
        raise ValueError("No held-out train qids with pseudo-page labels were available for threshold tuning.")

    scores: list[dict[str, Any]] = []
    for threshold in thresholds:
        counter = rows_by_threshold[threshold]
        hit_rate = float(counter["hit"]) / float(evaluated)
        lost_rate = float(counter["lost"]) / float(evaluated)
        promotion_rate = float(counter["promotions"]) / float(evaluated)
        objective = hit_rate - float(args.lost_penalty) * lost_rate - float(args.promotion_penalty) * promotion_rate
        scores.append(
            {
                "threshold": float(threshold),
                f"page@{int(args.repair_hit_k)}": hit_rate,
                "objective": float(objective),
                "hit_count": int(counter["hit"]),
                "base_hit_count": int(counter["base_hit"]),
                "recovered": int(counter["recovered"]),
                "lost": int(counter["lost"]),
                "promotions": int(counter["promotions"]),
            }
        )

    best = max(
        scores,
        key=lambda row: (
            float(row["objective"]),
            float(row[f"page@{int(args.repair_hit_k)}"]),
            -float(row["lost"]),
            -float(row["promotions"]),
            float(row["threshold"]),
        ),
    )
    return {
        "selected_threshold": float(best["threshold"]),
        "optimized_metric": f"page@{int(args.repair_hit_k)}_minus_lost_penalty",
        "optimized_metric_value": float(best["objective"]),
        "tune_eval_qid_count": int(evaluated),
        "skipped_no_page_gold": int(skipped_no_page_gold),
        "skipped_missing_prediction": int(skipped_missing_prediction),
        "threshold_scores": scores,
    }


def write_outputs(
    *,
    args: argparse.Namespace,
    model_info: dict[str, Any],
    train_meta: dict[str, Any],
    tuning_summary: dict[str, Any] | None,
    eval_base: dict[str, dict[str, Any]],
    eval_gold: dict[str, dict[str, Any]],
    output_pred: dict[str, dict[str, Any]],
    action_rows: list[dict[str, Any]],
) -> None:
    prediction_path = Path(args.output_prediction_json)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_text(json.dumps(output_pred) + "\n", encoding="utf-8")

    model_payload = {
        "feature_names": FEATURE_NAMES,
        "base_feature_names": ca.FEATURE_NAMES,
        "action_feature_names": ACTION_FEATURE_NAMES,
        "model_info": model_info,
        "train_metadata": train_meta,
        "tuning_summary": tuning_summary,
        "args": {
            "candidate_top_k": int(args.candidate_top_k),
            "repair_hit_k": int(args.repair_hit_k),
            "insert_rank": int(args.insert_rank),
            "promotion_rank_min": int(args.promotion_rank_min),
            "promotion_rank_max": int(args.promotion_rank_max),
            "max_promotions_per_qid": int(args.max_promotions_per_qid),
            "threshold": float(args.threshold),
            "auto_tune_threshold": bool(args.auto_tune_threshold),
        },
    }
    model_path = Path(args.output_model_json)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(model_payload, indent=2) + "\n", encoding="utf-8")

    metrics = [
        ca.evaluate_run(label="base", pred=eval_base, gold=eval_gold, recall_ks=list(args.recall_k)),
        ca.evaluate_run(label="counterfactual_page_promotion", pred=output_pred, gold=eval_gold, recall_ks=list(args.recall_k)),
    ]
    summary = {
        "train_gold": args.train_gold,
        "eval_gold": args.eval_gold,
        "train_base_pred": args.train_base_pred,
        "eval_base_pred": args.eval_base_pred,
        "feature_names": FEATURE_NAMES,
        "model_info": model_info,
        "train_metadata": train_meta,
        "tuning_summary": tuning_summary,
        "metrics": metrics,
        "movement_vs_base": ca.movement_vs_base(
            base_pred=eval_base,
            candidate_pred=output_pred,
            gold=eval_gold,
            hit_k=int(args.repair_hit_k),
        ),
        "action_counts": {
            "candidate_action_count": int(len(action_rows)),
            "promoted_action_count": int(sum(1 for row in action_rows if row.get("promoted"))),
        },
    }
    summary_path = Path(args.output_summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.output_table_md:
        ca.write_table(Path(args.output_table_md), metrics, list(args.recall_k))
    if args.output_actions_jsonl:
        action_path = Path(args.output_actions_jsonl)
        action_path.parent.mkdir(parents=True, exist_ok=True)
        with action_path.open("w", encoding="utf-8") as handle:
            for row in action_rows:
                handle.write(json.dumps(row) + "\n")

    print(f"saved_model={model_path}")
    print(f"saved_prediction={prediction_path}")
    print(f"saved_summary={summary_path}")
    if args.output_table_md:
        print(f"saved_table={args.output_table_md}")
    if args.output_actions_jsonl:
        print(f"saved_actions={args.output_actions_jsonl}")
    print(f"objective={model_info.get('objective')}")
    print(f"train_metadata={train_meta}")
    print(f"movement_vs_base={summary['movement_vs_base']['counts']}")
    print(f"action_counts={summary['action_counts']}")
    for row in metrics:
        print(row)


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    train_gold = ca.load_gold(Path(args.train_gold))
    eval_gold = ca.load_gold(Path(args.eval_gold))
    train_base = ca.load_prediction(Path(args.train_base_pred))
    eval_base = ca.load_prediction(Path(args.eval_base_pred))
    train_page_features = ca.load_page_features(Path(args.train_page_text_jsonl))
    eval_page_features = ca.load_page_features(Path(args.eval_page_text_jsonl))

    train_source_maps: dict[str, dict[str, dict[str, float]]] = {}
    eval_source_maps: dict[str, dict[str, dict[str, float]]] = {}
    for spec in args.train_source:
        label, path = ca.parse_labeled_path(spec)
        train_source_maps[label] = ca.source_maps(ca.load_prediction(path), int(args.candidate_top_k))
    for spec in args.eval_source:
        label, path = ca.parse_labeled_path(spec)
        eval_source_maps[label] = ca.source_maps(ca.load_prediction(path), int(args.candidate_top_k))

    fit_gold = train_gold
    tune_gold: dict[str, dict[str, Any]] = {}
    if bool(args.auto_tune_threshold):
        fit_gold, tune_gold = ca.split_gold_for_tuning(
            train_gold,
            tune_fraction=float(args.tune_fraction),
            seed=int(args.seed),
        )

    X, y, train_meta = build_counterfactual_matrix(
        gold=fit_gold,
        base_pred=train_base,
        page_features=train_page_features,
        source_maps_by_label=train_source_maps,
        args=args,
    )
    model, model_info = train_classifier(X, y, args)

    tuning_summary: dict[str, Any] | None = None
    if bool(args.auto_tune_threshold):
        fit_train_meta = dict(train_meta)
        tuning_summary = tune_threshold(
            tune_gold=tune_gold,
            base_pred=train_base,
            page_features=train_page_features,
            source_maps_by_label=train_source_maps,
            model=model,
            args=args,
        )
        args.threshold = float(tuning_summary["selected_threshold"])
        if not bool(args.skip_retrain_after_tuning):
            X, y, train_meta = build_counterfactual_matrix(
                gold=train_gold,
                base_pred=train_base,
                page_features=train_page_features,
                source_maps_by_label=train_source_maps,
                args=args,
            )
            model, model_info = train_classifier(X, y, args)
        train_meta = {
            **train_meta,
            "auto_tune_threshold": True,
            "retrained_after_tuning": not bool(args.skip_retrain_after_tuning),
            "fit_qid_count": int(len(fit_gold)),
            "tune_qid_count": int(len(tune_gold)),
            "selected_threshold": float(args.threshold),
            "fit_train_metadata": fit_train_meta,
            "tuning_summary": tuning_summary,
        }

    output_pred, action_rows = apply_counterfactual_promoter(
        gold=eval_gold,
        base_pred=eval_base,
        page_features=eval_page_features,
        source_maps_by_label=eval_source_maps,
        model=model,
        args=args,
    )
    write_outputs(
        args=args,
        model_info=model_info,
        train_meta=train_meta,
        tuning_summary=tuning_summary,
        eval_base=eval_base,
        eval_gold=eval_gold,
        output_pred=output_pred,
        action_rows=action_rows,
    )


if __name__ == "__main__":
    main()
