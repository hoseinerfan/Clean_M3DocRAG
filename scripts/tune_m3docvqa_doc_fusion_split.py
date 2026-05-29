#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
from pathlib import Path
from typing import Any


DEFAULT_RECALL_KS = [1, 2, 4, 5, 10, 20, 50, 100]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune a simple document-level fusion over existing M3DocVQA prediction runs "
            "using a train/eval qid split. This uses only gold doc_id labels, not page labels."
        )
    )
    parser.add_argument("--gold", required=True, help="Gold MMQA_<split>.jsonl")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Source prediction as LABEL=path/to/prediction.json. Repeat for dense/SPLADE/GPP runs.",
    )
    parser.add_argument("--train-qids", help="Optional train qid file. If omitted, split --gold qids.")
    parser.add_argument("--eval-qids", help="Optional eval qid file. If omitted, split --gold qids.")
    parser.add_argument("--holdout-frac", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--top-rows",
        type=int,
        default=1000,
        help="Rows to read per source/qid before aggregating by doc. Use 0 for all rows.",
    )
    parser.add_argument(
        "--source-score-mode",
        choices=["rank", "score", "rank_score"],
        default="rank_score",
        help=(
            "Per-source doc contribution before weight tuning: first-doc rank, normalized "
            "max page score, or their sum."
        ),
    )
    parser.add_argument(
        "--weight-grid",
        default="0,0.25,0.5,1,2,4",
        help="Comma-separated non-negative source weights searched by grid.",
    )
    parser.add_argument(
        "--objective-k",
        type=int,
        default=4,
        help="Primary doc recall@k optimized on the train fold.",
    )
    parser.add_argument("--recall-k", type=int, nargs="+", default=DEFAULT_RECALL_KS)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument("--output-table-md")
    parser.add_argument("--output-prediction-json")
    parser.add_argument(
        "--prediction-scope",
        choices=["eval", "train", "all"],
        default="eval",
        help="Qids written to --output-prediction-json.",
    )
    parser.add_argument("--output-train-qids")
    parser.add_argument("--output-eval-qids")
    return parser.parse_args()


def parse_labeled_path(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected LABEL=path for --source, got: {spec!r}")
    label, path = spec.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"Invalid --source value: {spec!r}")
    return label, Path(path)


def ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_prediction(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "predictions" in payload and isinstance(
        payload["predictions"], (dict, list)
    ):
        payload = payload["predictions"]

    if isinstance(payload, list):
        iterable = enumerate(payload)
    elif isinstance(payload, dict):
        iterable = payload.items()
    else:
        raise TypeError(f"Prediction JSON must be a list or object: {path}")

    rows_by_qid: dict[str, dict[str, Any]] = {}
    for raw_key, row in iterable:
        if not isinstance(row, dict):
            raise TypeError(f"Prediction row must be object: {path} key={raw_key!r}")
        qid = str(row.get("qid", "")).strip() or str(raw_key).strip()
        if not qid:
            raise ValueError(f"Prediction row missing qid: {path} key={raw_key!r}")
        if qid in rows_by_qid:
            raise ValueError(f"Duplicate qid after normalization: {qid} ({path})")
        rows_by_qid[qid] = row
    return rows_by_qid


def load_qids(path: Path) -> list[str]:
    qids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value and not value.startswith("#"):
                qids.append(value)
    return ordered_unique(qids)


def write_qids(path: Path, qids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{qid}\n" for qid in qids), encoding="utf-8")


def gold_doc_ids(row: dict[str, Any]) -> set[str]:
    return {
        str(ctx.get("doc_id", "")).strip()
        for ctx in row.get("supporting_context", [])
        if isinstance(ctx, dict) and str(ctx.get("doc_id", "")).strip()
    }


def prediction_rows(pred_item: dict[str, Any]) -> list[Any]:
    rows = pred_item.get("page_retrieval_results", [])
    return rows if isinstance(rows, list) else []


def parse_retrieval_row(row: Any) -> tuple[str, int, float] | None:
    if isinstance(row, list) and row:
        doc_id = str(row[0]).strip()
        if not doc_id:
            return None
        page_idx = 0
        if len(row) > 1:
            try:
                page_idx = int(row[1])
            except (TypeError, ValueError):
                page_idx = 0
        score = 0.0
        if len(row) > 2:
            try:
                score = float(row[2])
            except (TypeError, ValueError):
                score = 0.0
        return doc_id, page_idx, score

    if isinstance(row, dict):
        doc_id = str(row.get("doc_id", "")).strip()
        if not doc_id:
            return None
        try:
            page_idx = int(row.get("page_idx", row.get("page_id", 0)) or 0)
        except (TypeError, ValueError):
            page_idx = 0
        score_value = row.get("fused_page_score", row.get("score", row.get("base_page_score", 0.0)))
        try:
            score = float(score_value)
        except (TypeError, ValueError):
            score = 0.0
        return doc_id, page_idx, score

    return None


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return [1.0 for _value in values]
    return [(value - lo) / (hi - lo) for value in values]


def doc_scores_for_qid(
    pred_item: dict[str, Any],
    *,
    top_rows: int,
    score_mode: str,
) -> dict[str, dict[str, Any]]:
    parsed_rows: list[tuple[str, int, float]] = []
    for row in prediction_rows(pred_item):
        parsed = parse_retrieval_row(row)
        if parsed is None:
            continue
        parsed_rows.append(parsed)
        if top_rows > 0 and len(parsed_rows) >= top_rows:
            break

    normalized_scores = minmax([score for _doc_id, _page_idx, score in parsed_rows])
    by_doc: dict[str, dict[str, Any]] = {}
    unique_doc_rank = 0
    seen_docs: set[str] = set()
    for row_idx, (doc_id, page_idx, raw_score) in enumerate(parsed_rows):
        norm_score = normalized_scores[row_idx] if row_idx < len(normalized_scores) else 0.0
        if doc_id not in seen_docs:
            seen_docs.add(doc_id)
            unique_doc_rank += 1
        record = by_doc.get(doc_id)
        if record is None:
            record = {
                "doc_id": doc_id,
                "rank": unique_doc_rank,
                "best_page_idx": page_idx,
                "raw_max_score": raw_score,
                "norm_max_score": norm_score,
                "row_count": 0,
            }
            by_doc[doc_id] = record
        record["row_count"] = int(record["row_count"]) + 1
        if norm_score > float(record["norm_max_score"]):
            record["norm_max_score"] = norm_score
            record["raw_max_score"] = raw_score
            record["best_page_idx"] = page_idx

    for record in by_doc.values():
        rank_score = 1.0 / max(1.0, float(record["rank"]))
        norm_score = float(record["norm_max_score"])
        if score_mode == "rank":
            contribution = rank_score
        elif score_mode == "score":
            contribution = norm_score
        else:
            contribution = rank_score + norm_score
        record["source_doc_score"] = float(contribution)
    return by_doc


def build_source_doc_scores(
    sources: list[tuple[str, dict[str, dict[str, Any]]]],
    *,
    qids: list[str],
    top_rows: int,
    score_mode: str,
) -> dict[str, list[dict[str, dict[str, Any]]]]:
    by_qid: dict[str, list[dict[str, dict[str, Any]]]] = {}
    for qid in qids:
        source_scores: list[dict[str, dict[str, Any]]] = []
        for _label, pred in sources:
            if qid not in pred:
                source_scores.append({})
                continue
            source_scores.append(
                doc_scores_for_qid(pred[qid], top_rows=top_rows, score_mode=score_mode)
            )
        by_qid[qid] = source_scores
    return by_qid


def fuse_docs_for_qid(
    source_doc_scores: list[dict[str, dict[str, Any]]],
    weights: tuple[float, ...],
) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}
    for source_idx, doc_scores in enumerate(source_doc_scores):
        weight = float(weights[source_idx])
        if weight <= 0:
            continue
        for doc_id, source_record in doc_scores.items():
            contribution = weight * float(source_record["source_doc_score"])
            record = fused.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "score": 0.0,
                    "best_page_idx": int(source_record.get("best_page_idx", 0)),
                    "best_source_contribution": -1.0,
                    "best_rank": int(source_record.get("rank", 10**9)),
                    "source_presence_count": 0,
                },
            )
            record["score"] = float(record["score"]) + contribution
            record["source_presence_count"] = int(record["source_presence_count"]) + 1
            record["best_rank"] = min(int(record["best_rank"]), int(source_record.get("rank", 10**9)))
            if contribution > float(record["best_source_contribution"]):
                record["best_source_contribution"] = contribution
                record["best_page_idx"] = int(source_record.get("best_page_idx", 0))

    ranked = sorted(
        fused.values(),
        key=lambda item: (
            -float(item["score"]),
            int(item["best_rank"]),
            -int(item["source_presence_count"]),
            str(item["doc_id"]),
        ),
    )
    return ranked


def evaluate_weights(
    *,
    qids: list[str],
    gold_by_qid: dict[str, dict[str, Any]],
    source_doc_scores_by_qid: dict[str, list[dict[str, dict[str, Any]]]],
    weights: tuple[float, ...],
    recall_ks: list[int],
) -> dict[str, Any]:
    recall_values = {k: [] for k in recall_ks}
    first_ranks: dict[str, int | None] = {}
    missing_qids = 0
    for qid in qids:
        docs = gold_doc_ids(gold_by_qid[qid])
        ranked = fuse_docs_for_qid(source_doc_scores_by_qid[qid], weights)
        ranked_doc_ids = [str(item["doc_id"]) for item in ranked]
        if not ranked_doc_ids:
            missing_qids += 1
        first_rank = None
        for idx, doc_id in enumerate(ranked_doc_ids, start=1):
            if doc_id in docs:
                first_rank = idx
                break
        first_ranks[qid] = first_rank
        denom = len(docs)
        for k in recall_ks:
            top_docs = set(ranked_doc_ids[:k])
            recall_values[k].append(len(top_docs & docs) / denom if denom else 0.0)

    recall = {
        str(k): (sum(values) / len(values) if values else 0.0)
        for k, values in recall_values.items()
    }
    return {
        "n_qids": len(qids),
        "missing_prediction_qids": missing_qids,
        "doc_recall": recall,
        "first_gold_doc_rank_top1_rate": (
            sum(1 for rank in first_ranks.values() if rank == 1) / len(qids) if qids else 0.0
        ),
        "first_gold_doc_rank_top4_rate": (
            sum(1 for rank in first_ranks.values() if rank is not None and rank <= 4) / len(qids)
            if qids
            else 0.0
        ),
    }


def metric_tuple(metrics: dict[str, Any], *, objective_k: int) -> tuple[float, float, float, float]:
    recall = metrics["doc_recall"]
    return (
        float(recall.get(str(objective_k), 0.0)),
        float(recall.get("1", 0.0)),
        float(recall.get("20", 0.0)),
        float(metrics.get("first_gold_doc_rank_top4_rate", 0.0)),
    )


def split_qids(
    qids: list[str],
    *,
    holdout_frac: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    if not 0.0 < holdout_frac < 1.0:
        raise ValueError("--holdout-frac must be in (0, 1) when qid files are omitted.")
    shuffled = list(qids)
    random.Random(seed).shuffle(shuffled)
    eval_count = max(1, int(round(len(shuffled) * holdout_frac)))
    eval_qids = sorted(shuffled[:eval_count])
    train_qids = sorted(shuffled[eval_count:])
    if not train_qids or not eval_qids:
        raise ValueError("QID split produced an empty train or eval fold.")
    return train_qids, eval_qids


def parse_weight_grid(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--weight-grid must contain at least one value.")
    if any(value < 0 for value in values):
        raise ValueError("--weight-grid values must be non-negative.")
    return sorted(set(values))


def markdown_table(rows: list[dict[str, Any]], recall_ks: list[int]) -> str:
    columns = ["label", "split", "n_qids", *[f"doc@{k}" for k in recall_ks], "hit@1", "hit@4"]
    out = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = [
            row["label"],
            row["split"],
            str(row["n_qids"]),
            *[f"{float(row.get(f'doc@{k}', 0.0)):.4f}" for k in recall_ks],
            f"{float(row.get('hit@1', 0.0)):.4f}",
            f"{float(row.get('hit@4', 0.0)):.4f}",
        ]
        out.append("| " + " | ".join(values) + " |")
    return "\n".join(out) + "\n"


def make_metric_row(
    label: str,
    split: str,
    metrics: dict[str, Any],
    recall_ks: list[int],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "label": label,
        "split": split,
        "n_qids": int(metrics["n_qids"]),
        "hit@1": float(metrics["first_gold_doc_rank_top1_rate"]),
        "hit@4": float(metrics["first_gold_doc_rank_top4_rate"]),
    }
    for k in recall_ks:
        row[f"doc@{k}"] = float(metrics["doc_recall"].get(str(k), 0.0))
    return row


def write_prediction(
    path: Path,
    *,
    qids: list[str],
    source_doc_scores_by_qid: dict[str, list[dict[str, dict[str, Any]]]],
    weights: tuple[float, ...],
) -> None:
    predictions: list[dict[str, Any]] = []
    for qid in qids:
        ranked = fuse_docs_for_qid(source_doc_scores_by_qid[qid], weights)
        predictions.append(
            {
                "qid": qid,
                "page_retrieval_results": [
                    [item["doc_id"], int(item["best_page_idx"]), float(item["score"])]
                    for item in ranked
                ],
                "doc_fusion": {
                    "source_weights": list(weights),
                    "ranked_doc_count": len(ranked),
                },
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"predictions": predictions}, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.source:
        raise ValueError("Pass at least one --source LABEL=prediction.json")

    source_specs = [parse_labeled_path(spec) for spec in args.source]
    source_labels = [label for label, _path in source_specs]
    source_predictions = [(label, load_prediction(path)) for label, path in source_specs]
    gold_rows = load_jsonl(Path(args.gold))
    gold_by_qid = {str(row["qid"]): row for row in gold_rows}

    source_qid_sets = [set(pred.keys()) for _label, pred in source_predictions]
    usable_qids = sorted(set(gold_by_qid).intersection(*source_qid_sets))
    if not usable_qids:
        raise ValueError("No qids are common to gold and all source predictions.")

    if args.train_qids or args.eval_qids:
        if not args.train_qids or not args.eval_qids:
            raise ValueError("Pass both --train-qids and --eval-qids, or neither.")
        train_qids = [qid for qid in load_qids(Path(args.train_qids)) if qid in usable_qids]
        eval_qids = [qid for qid in load_qids(Path(args.eval_qids)) if qid in usable_qids]
    else:
        train_qids, eval_qids = split_qids(
            usable_qids,
            holdout_frac=float(args.holdout_frac),
            seed=int(args.seed),
        )
    if not train_qids or not eval_qids:
        raise ValueError("No usable qids remain in train or eval split.")

    source_doc_scores_by_qid = build_source_doc_scores(
        source_predictions,
        qids=usable_qids,
        top_rows=max(0, int(args.top_rows)),
        score_mode=str(args.source_score_mode),
    )

    weight_values = parse_weight_grid(args.weight_grid)
    candidate_weights = [
        tuple(float(value) for value in combo)
        for combo in itertools.product(weight_values, repeat=len(source_predictions))
        if any(float(value) > 0.0 for value in combo)
    ]
    if not candidate_weights:
        raise ValueError("Weight grid produced no non-zero configurations.")

    best_weights = candidate_weights[0]
    best_train_metrics = evaluate_weights(
        qids=train_qids,
        gold_by_qid=gold_by_qid,
        source_doc_scores_by_qid=source_doc_scores_by_qid,
        weights=best_weights,
        recall_ks=args.recall_k,
    )
    best_key = metric_tuple(best_train_metrics, objective_k=int(args.objective_k))
    searched = []
    for weights in candidate_weights:
        metrics = evaluate_weights(
            qids=train_qids,
            gold_by_qid=gold_by_qid,
            source_doc_scores_by_qid=source_doc_scores_by_qid,
            weights=weights,
            recall_ks=args.recall_k,
        )
        key = metric_tuple(metrics, objective_k=int(args.objective_k))
        searched.append({"weights": list(weights), "train_metrics": metrics})
        if key > best_key:
            best_key = key
            best_weights = weights
            best_train_metrics = metrics

    best_eval_metrics = evaluate_weights(
        qids=eval_qids,
        gold_by_qid=gold_by_qid,
        source_doc_scores_by_qid=source_doc_scores_by_qid,
        weights=best_weights,
        recall_ks=args.recall_k,
    )

    source_baselines: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for idx, label in enumerate(source_labels):
        weights = tuple(1.0 if source_idx == idx else 0.0 for source_idx in range(len(source_labels)))
        train_metrics = evaluate_weights(
            qids=train_qids,
            gold_by_qid=gold_by_qid,
            source_doc_scores_by_qid=source_doc_scores_by_qid,
            weights=weights,
            recall_ks=args.recall_k,
        )
        eval_metrics = evaluate_weights(
            qids=eval_qids,
            gold_by_qid=gold_by_qid,
            source_doc_scores_by_qid=source_doc_scores_by_qid,
            weights=weights,
            recall_ks=args.recall_k,
        )
        source_baselines.append(
            {"label": label, "weights": list(weights), "train_metrics": train_metrics, "eval_metrics": eval_metrics}
        )
        metric_rows.append(make_metric_row(label, "train", train_metrics, args.recall_k))
        metric_rows.append(make_metric_row(label, "eval", eval_metrics, args.recall_k))

    metric_rows.append(make_metric_row("tuned_fusion", "train", best_train_metrics, args.recall_k))
    metric_rows.append(make_metric_row("tuned_fusion", "eval", best_eval_metrics, args.recall_k))

    objective_key = f"doc@{int(args.objective_k)}"
    best_source_eval = max(
        (float(row.get(objective_key, 0.0)) for row in metric_rows if row["split"] == "eval" and row["label"] != "tuned_fusion"),
        default=0.0,
    )
    tuned_eval = float(
        next(row for row in metric_rows if row["split"] == "eval" and row["label"] == "tuned_fusion").get(
            objective_key,
            0.0,
        )
    )

    if args.output_train_qids:
        write_qids(Path(args.output_train_qids), train_qids)
    if args.output_eval_qids:
        write_qids(Path(args.output_eval_qids), eval_qids)
    if args.output_prediction_json:
        if args.prediction_scope == "train":
            prediction_qids = train_qids
        elif args.prediction_scope == "all":
            prediction_qids = usable_qids
        else:
            prediction_qids = eval_qids
        write_prediction(
            Path(args.output_prediction_json),
            qids=prediction_qids,
            source_doc_scores_by_qid=source_doc_scores_by_qid,
            weights=best_weights,
        )

    summary = {
        "gold": args.gold,
        "source_labels": source_labels,
        "source_paths": [str(path) for _label, path in source_specs],
        "source_score_mode": args.source_score_mode,
        "top_rows": int(args.top_rows),
        "weight_grid": weight_values,
        "searched_config_count": len(candidate_weights),
        "objective_k": int(args.objective_k),
        "usable_qid_count": len(usable_qids),
        "train_qid_count": len(train_qids),
        "eval_qid_count": len(eval_qids),
        "best_weights": dict(zip(source_labels, best_weights)),
        "best_train_metrics": best_train_metrics,
        "best_eval_metrics": best_eval_metrics,
        "source_baselines": source_baselines,
        "metric_rows": metric_rows,
        "eval_delta_vs_best_source": {
            objective_key: tuned_eval - best_source_eval,
            "tuned_eval": tuned_eval,
            "best_source_eval": best_source_eval,
        },
    }

    output_summary_json = Path(args.output_summary_json)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.output_table_md:
        table_path = Path(args.output_table_md)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table_path.write_text(markdown_table(metric_rows, args.recall_k), encoding="utf-8")

    print(f"saved_summary={output_summary_json}")
    if args.output_table_md:
        print(f"saved_table={args.output_table_md}")
    if args.output_prediction_json:
        print(f"saved_prediction={args.output_prediction_json}")
    print(f"best_weights={summary['best_weights']}")
    print(
        f"eval_{objective_key}={tuned_eval:.4f} "
        f"best_source_eval_{objective_key}={best_source_eval:.4f} "
        f"delta={tuned_eval - best_source_eval:+.4f}"
    )


if __name__ == "__main__":
    main()
