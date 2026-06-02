#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import train_content_aware_pseudo_page_reranker as ca


DEFAULT_ALPHA_GRID = "0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a graph/content-aware boosted-tree page LTR reranker for M3DocVQA. "
            "LightGBM LambdaMART is used when available; a sklearn pointwise GBDT fallback "
            "keeps the experiment runnable in environments without LightGBM."
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
    parser.add_argument("--backend", choices=["auto", "lightgbm", "sklearn"], default="auto")
    parser.add_argument("--candidate-top-k", type=int, default=1000)
    parser.add_argument("--negatives-per-band", type=int, default=24)
    parser.add_argument("--max-negatives-per-qid", type=int, default=160)
    parser.add_argument("--max-same-doc-pages-per-qid", type=int, default=24)
    parser.add_argument("--same-doc-label", type=int, default=1)
    parser.add_argument("--pseudo-page-label", type=int, default=2)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-data-in-leaf", type=int, default=30)
    parser.add_argument("--subsample", type=float, default=0.90)
    parser.add_argument("--colsample-bytree", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument(
        "--inference-mode",
        choices=["blend_rerank", "full_rerank", "safe_promote", "doc_head_blend", "doc_slot_blend"],
        default="blend_rerank",
    )
    parser.add_argument("--blend-alpha", type=float, default=0.30)
    parser.add_argument("--auto-tune-blend-alpha", action="store_true")
    parser.add_argument("--tune-fraction", type=float, default=0.20)
    parser.add_argument("--tune-blend-alpha-grid", default=DEFAULT_ALPHA_GRID)
    parser.add_argument("--tune-hit-k", type=int, default=5)
    parser.add_argument("--skip-retrain-after-tuning", action="store_true")
    parser.add_argument("--anchor-top-k", type=int, default=4)
    parser.add_argument("--promotion-rank-min", type=int, default=5)
    parser.add_argument("--promotion-rank-max", type=int, default=200)
    parser.add_argument("--max-promotions-per-qid", type=int, default=2)
    parser.add_argument("--promotion-margin", type=float, default=0.05)
    parser.add_argument("--recall-k", type=int, nargs="+", default=ca.DEFAULT_RECALL_KS)
    parser.add_argument("--output-model-json", required=True)
    parser.add_argument("--output-prediction-json", required=True)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument("--output-table-md", default="")
    parser.add_argument("--output-eval-prior-jsonl", default="")
    return parser.parse_args()


def label_for_record(record: dict[str, Any], pseudo_pages: set[str], gold_docs: set[str], args: argparse.Namespace) -> int:
    if str(record["uid"]) in pseudo_pages:
        return int(args.pseudo_page_label)
    if str(record["doc_id"]) in gold_docs:
        return int(args.same_doc_label)
    return 0


def pick_ltr_indices(records: list[dict[str, Any]], labels: list[int], args: argparse.Namespace) -> list[int]:
    selected: list[int] = []
    selected_set: set[int] = set()

    for idx, label in enumerate(labels):
        if label >= int(args.pseudo_page_label):
            selected.append(idx)
            selected_set.add(idx)

    same_doc_added = 0
    for idx, label in enumerate(labels):
        if idx in selected_set or label <= 0:
            continue
        selected.append(idx)
        selected_set.add(idx)
        same_doc_added += 1
        if same_doc_added >= int(args.max_same_doc_pages_per_qid):
            break

    rng = random.Random(int(args.seed) + len(records) + sum(labels))
    bands = [(1, 4), (5, 20), (21, 100), (101, 500), (501, int(args.candidate_top_k))]
    negatives: list[int] = []
    for lo, hi in bands:
        band = [
            idx
            for idx, record in enumerate(records)
            if idx not in selected_set
            and labels[idx] <= 0
            and lo <= int(record["base_rank"]) <= hi
        ]
        rng.shuffle(band)
        for idx in band[: int(args.negatives_per_band)]:
            negatives.append(idx)
            selected_set.add(idx)
            if len(negatives) >= int(args.max_negatives_per_qid):
                break
        if len(negatives) >= int(args.max_negatives_per_qid):
            break

    selected.extend(negatives[: int(args.max_negatives_per_qid)])
    selected = sorted(set(selected), key=lambda idx: int(records[idx]["base_rank"]))
    return selected


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


def build_ltr_dataset(
    *,
    gold: dict[str, dict[str, Any]],
    base_pred: dict[str, dict[str, Any]],
    page_features: dict[str, dict[str, Any]],
    source_maps_by_label: dict[str, dict[str, dict[str, float]]],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[int], dict[str, Any]]:
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    groups: list[int] = []
    stats = Counter()

    for qid, gold_row in gold.items():
        pseudo_pages = ca.gold_page_uids(gold_row)
        if not pseudo_pages:
            stats["skipped_no_page_gold"] += 1
            continue
        records = ca.ranked_page_records(base_pred.get(qid), int(args.candidate_top_k))
        if not records:
            stats["skipped_missing_prediction"] += 1
            continue
        labels_all = [label_for_record(record, pseudo_pages, ca.gold_doc_ids(gold_row), args) for record in records]
        if max(labels_all, default=0) < int(args.pseudo_page_label):
            stats["skipped_no_pseudo_page_in_pool"] += 1
            continue
        selected = pick_ltr_indices(records, labels_all, args)
        if len(selected) < 2:
            stats["skipped_too_small_group"] += 1
            continue
        selected_records = [records[idx] for idx in selected]
        selected_labels = [labels_all[idx] for idx in selected]
        if len(set(selected_labels)) < 2:
            stats["skipped_no_label_variation"] += 1
            continue
        X_qid = feature_rows_for_qid(
            qid=qid,
            gold_row=gold_row,
            records=selected_records,
            page_features=page_features,
            source_maps_by_label=source_maps_by_label,
        )
        X_parts.append(X_qid)
        y_parts.append(np.asarray(selected_labels, dtype=np.float32))
        groups.append(len(selected_labels))
        stats["train_qid_count"] += 1
        stats["train_row_count"] += len(selected_labels)
        stats["pseudo_page_label_count"] += sum(1 for value in selected_labels if value >= int(args.pseudo_page_label))
        stats["same_doc_label_count"] += sum(1 for value in selected_labels if 0 < value < int(args.pseudo_page_label))
        stats["negative_label_count"] += sum(1 for value in selected_labels if value <= 0)

    if not X_parts:
        raise ValueError("No LTR training groups were produced.")
    X = np.vstack(X_parts).astype(np.float32)
    y = np.concatenate(y_parts).astype(np.float32)
    metadata = {key: int(value) for key, value in stats.items()}
    metadata["group_count"] = int(len(groups))
    metadata["mean_group_size"] = float(sum(groups) / max(len(groups), 1))
    return X, y, groups, metadata


def train_lightgbm(X: np.ndarray, y: np.ndarray, groups: list[int], args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    try:
        import lightgbm as lgb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("LightGBM is not installed. Use --backend sklearn or install lightgbm.") from exc

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        num_leaves=int(args.num_leaves),
        min_data_in_leaf=int(args.min_data_in_leaf),
        subsample=float(args.subsample),
        colsample_bytree=float(args.colsample_bytree),
        random_state=int(args.seed),
        n_jobs=int(args.n_jobs),
        verbosity=-1,
    )
    model.fit(X, y.astype(int), group=groups, eval_at=[4, 5, 10])
    info = {
        "backend": "lightgbm",
        "objective": "lambdarank",
        "params": model.get_params(),
        "feature_importance": {
            name: float(value)
            for name, value in zip(ca.FEATURE_NAMES, model.feature_importances_.tolist())
        },
    }
    return model, info


def train_sklearn(X: np.ndarray, y: np.ndarray, args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("scikit-learn is not installed; cannot use sklearn fallback.") from exc

    model = HistGradientBoostingRegressor(
        max_iter=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        max_leaf_nodes=int(args.num_leaves),
        min_samples_leaf=int(args.min_data_in_leaf),
        l2_regularization=0.0,
        random_state=int(args.seed),
    )
    model.fit(X, y)
    info = {
        "backend": "sklearn",
        "objective": "pointwise_gbdt_regression_fallback",
        "params": {
            "n_estimators": int(args.n_estimators),
            "learning_rate": float(args.learning_rate),
            "num_leaves": int(args.num_leaves),
            "min_data_in_leaf": int(args.min_data_in_leaf),
            "seed": int(args.seed),
        },
        "note": "This fallback is boosted-tree LTR-style, not LambdaMART. Use --backend lightgbm for grouped LambdaRank.",
    }
    return model, info


def train_model(X: np.ndarray, y: np.ndarray, groups: list[int], args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    if args.backend == "lightgbm":
        return train_lightgbm(X, y, groups, args)
    if args.backend == "sklearn":
        return train_sklearn(X, y, args)
    try:
        return train_lightgbm(X, y, groups, args)
    except ModuleNotFoundError:
        return train_sklearn(X, y, args)


def predict_scores(model: Any, X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return np.asarray([], dtype=np.float32)
    values = model.predict(X)
    return np.asarray(values, dtype=np.float32)


def score_records(
    *,
    qid: str,
    gold_row: dict[str, Any],
    records: list[dict[str, Any]],
    page_features: dict[str, dict[str, Any]],
    source_maps_by_label: dict[str, dict[str, dict[str, float]]],
    model: Any,
) -> list[dict[str, Any]]:
    X = feature_rows_for_qid(
        qid=qid,
        gold_row=gold_row,
        records=records,
        page_features=page_features,
        source_maps_by_label=source_maps_by_label,
    )
    scores = predict_scores(model, X)
    for record, score in zip(records, scores):
        record["learned_score"] = float(score)
    return records


def apply_ltr(
    *,
    gold: dict[str, dict[str, Any]],
    base_pred: dict[str, dict[str, Any]],
    page_features: dict[str, dict[str, Any]],
    source_maps_by_label: dict[str, dict[str, dict[str, float]]],
    model: Any,
    model_info: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    output: dict[str, dict[str, Any]] = {}
    prior_rows: list[dict[str, Any]] = []
    for qid, base_row in base_pred.items():
        records = ca.ranked_page_records(base_row, int(args.candidate_top_k))
        gold_row = gold.get(qid, {"qid": qid, "question": base_row.get("question", "")})
        scored_records = score_records(
            qid=qid,
            gold_row=gold_row,
            records=records,
            page_features=page_features,
            source_maps_by_label=source_maps_by_label,
            model=model,
        )
        reranked = ca.rerank_records(scored_records, args)
        reranked_uids = {row["uid"] for row in reranked}
        raw_by_uid = {row["uid"]: row["raw"] for row in records}
        output_rows = [raw_by_uid[row["uid"]] for row in reranked if row["uid"] in raw_by_uid]
        output_rows.extend(
            raw_by_uid[row["uid"]]
            for row in records
            if row["uid"] not in reranked_uids and row["uid"] in raw_by_uid
        )
        out_row = dict(base_row)
        out_row["page_retrieval_results"] = output_rows
        out_row["reranker_metadata"] = {
            **(out_row.get("reranker_metadata", {}) if isinstance(out_row.get("reranker_metadata"), dict) else {}),
            "graph_aware_ltr_page_reranker": {
                "backend": model_info.get("backend"),
                "objective": model_info.get("objective"),
                "inference_mode": args.inference_mode,
                "blend_alpha": float(args.blend_alpha),
            },
        }
        output[qid] = out_row
        for row in scored_records:
            prior_rows.append(
                {
                    "qid": qid,
                    "page_uid": row["uid"],
                    "doc_id": row["doc_id"],
                    "page_idx": int(row["page_idx"]),
                    "base_rank": int(row["base_rank"]),
                    "learned_score": float(row.get("learned_score", 0.0)),
                }
            )
    return output, prior_rows


def tune_blend_alpha(
    *,
    tune_gold: dict[str, dict[str, Any]],
    base_pred: dict[str, dict[str, Any]],
    page_features: dict[str, dict[str, Any]],
    source_maps_by_label: dict[str, dict[str, dict[str, float]]],
    model: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    alpha_grid = ca.parse_alpha_grid(str(args.tune_blend_alpha_grid))
    hit_k = int(args.tune_hit_k)
    if hit_k <= 0:
        raise ValueError(f"tune_hit_k must be positive, got {hit_k}")
    candidate_args = copy.copy(args)
    candidate_args.inference_mode = "blend_rerank"
    hits = {alpha: 0 for alpha in alpha_grid}
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
        scored = score_records(
            qid=qid,
            gold_row=gold_row,
            records=records,
            page_features=page_features,
            source_maps_by_label=source_maps_by_label,
            model=model,
        )
        evaluated += 1
        for alpha in alpha_grid:
            candidate_args.blend_alpha = float(alpha)
            ranked = ca.rerank_records([dict(row) for row in scored], candidate_args)
            ranked_uids = [str(row["uid"]) for row in ranked]
            if any(uid in pages_gold for uid in ranked_uids[:hit_k]):
                hits[alpha] += 1
    if evaluated <= 0:
        raise ValueError("No held-out train qids with pseudo-page labels were available for blend-alpha tuning.")
    scores = [
        {
            "blend_alpha": float(alpha),
            f"page@{hit_k}": float(hits[alpha]) / float(evaluated),
            "hit_count": int(hits[alpha]),
        }
        for alpha in alpha_grid
    ]
    best = max(scores, key=lambda row: (float(row[f"page@{hit_k}"]), -float(row["blend_alpha"])))
    return {
        "selected_blend_alpha": float(best["blend_alpha"]),
        "optimized_metric": f"page@{hit_k}",
        "optimized_metric_value": float(best[f"page@{hit_k}"]),
        "tune_eval_qid_count": int(evaluated),
        "skipped_no_page_gold": int(skipped_no_page_gold),
        "skipped_missing_prediction": int(skipped_missing_prediction),
        "alpha_scores": scores,
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
    prior_rows: list[dict[str, Any]],
) -> None:
    prediction_path = Path(args.output_prediction_json)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_text(json.dumps(output_pred) + "\n", encoding="utf-8")

    model_payload = {
        "feature_names": ca.FEATURE_NAMES,
        "model_info": model_info,
        "train_metadata": train_meta,
        "tuning_summary": tuning_summary,
        "args": {
            "candidate_top_k": int(args.candidate_top_k),
            "backend": args.backend,
            "inference_mode": args.inference_mode,
            "blend_alpha": float(args.blend_alpha),
            "auto_tune_blend_alpha": bool(args.auto_tune_blend_alpha),
        },
    }
    model_path = Path(args.output_model_json)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(model_payload, indent=2) + "\n", encoding="utf-8")

    metrics = [
        ca.evaluate_run(label="base", pred=eval_base, gold=eval_gold, recall_ks=list(args.recall_k)),
        ca.evaluate_run(label="graph_aware_ltr_page_reranker", pred=output_pred, gold=eval_gold, recall_ks=list(args.recall_k)),
    ]
    summary = {
        "train_gold": args.train_gold,
        "eval_gold": args.eval_gold,
        "train_base_pred": args.train_base_pred,
        "eval_base_pred": args.eval_base_pred,
        "feature_names": ca.FEATURE_NAMES,
        "model_info": model_info,
        "train_metadata": train_meta,
        "tuning_summary": tuning_summary,
        "metrics": metrics,
        "movement_vs_base": ca.movement_vs_base(base_pred=eval_base, candidate_pred=output_pred, gold=eval_gold),
    }
    summary_path = Path(args.output_summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.output_table_md:
        ca.write_table(Path(args.output_table_md), metrics, list(args.recall_k))
    if args.output_eval_prior_jsonl:
        prior_path = Path(args.output_eval_prior_jsonl)
        prior_path.parent.mkdir(parents=True, exist_ok=True)
        with prior_path.open("w", encoding="utf-8") as handle:
            for row in prior_rows:
                handle.write(json.dumps(row) + "\n")

    print(f"saved_model={model_path}")
    print(f"saved_prediction={prediction_path}")
    print(f"saved_summary={summary_path}")
    if args.output_table_md:
        print(f"saved_table={args.output_table_md}")
    if args.output_eval_prior_jsonl:
        print(f"saved_eval_prior={args.output_eval_prior_jsonl}")
    print(f"backend={model_info.get('backend')}")
    print(f"objective={model_info.get('objective')}")
    print(f"train_metadata={train_meta}")
    print(f"movement_vs_base={summary['movement_vs_base']['counts']}")
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
    if bool(args.auto_tune_blend_alpha):
        fit_gold, tune_gold = ca.split_gold_for_tuning(
            train_gold,
            tune_fraction=float(args.tune_fraction),
            seed=int(args.seed),
        )

    X, y, groups, train_meta = build_ltr_dataset(
        gold=fit_gold,
        base_pred=train_base,
        page_features=train_page_features,
        source_maps_by_label=train_source_maps,
        args=args,
    )
    model, model_info = train_model(X, y, groups, args)

    tuning_summary: dict[str, Any] | None = None
    if bool(args.auto_tune_blend_alpha):
        fit_train_meta = dict(train_meta)
        tuning_summary = tune_blend_alpha(
            tune_gold=tune_gold,
            base_pred=train_base,
            page_features=train_page_features,
            source_maps_by_label=train_source_maps,
            model=model,
            args=args,
        )
        args.blend_alpha = float(tuning_summary["selected_blend_alpha"])
        if not bool(args.skip_retrain_after_tuning):
            X, y, groups, train_meta = build_ltr_dataset(
                gold=train_gold,
                base_pred=train_base,
                page_features=train_page_features,
                source_maps_by_label=train_source_maps,
                args=args,
            )
            model, model_info = train_model(X, y, groups, args)
        train_meta = {
            **train_meta,
            "auto_tune_blend_alpha": True,
            "retrained_after_tuning": not bool(args.skip_retrain_after_tuning),
            "fit_qid_count": int(len(fit_gold)),
            "tune_qid_count": int(len(tune_gold)),
            "selected_blend_alpha": float(args.blend_alpha),
            "fit_train_metadata": fit_train_meta,
            "tuning_summary": tuning_summary,
        }

    output_pred, prior_rows = apply_ltr(
        gold=eval_gold,
        base_pred=eval_base,
        page_features=eval_page_features,
        source_maps_by_label=eval_source_maps,
        model=model,
        model_info=model_info,
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
        prior_rows=prior_rows,
    )


if __name__ == "__main__":
    main()
