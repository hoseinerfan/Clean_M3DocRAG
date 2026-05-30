#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

from tune_m3docvqa_doc_fusion_split import (
    DEFAULT_RECALL_KS,
    build_source_doc_scores,
    evaluate_weights,
    load_jsonl,
    load_prediction,
    make_metric_row,
    markdown_table,
    metric_tuple,
    parse_labeled_path,
    parse_weight_grid,
    write_prediction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune M3DocVQA document-fusion weights on MMQA train and evaluate the frozen "
            "weights on MMQA dev. This uses only document-level gold labels."
        )
    )
    parser.add_argument("--train-gold", required=True, help="Gold MMQA_train.jsonl")
    parser.add_argument("--eval-gold", required=True, help="Gold MMQA_dev.jsonl")
    parser.add_argument(
        "--train-source",
        action="append",
        default=[],
        help="Train source prediction as LABEL=path/to/train.prediction.json. Repeat.",
    )
    parser.add_argument(
        "--eval-source",
        action="append",
        default=[],
        help="Eval/dev source prediction as LABEL=path/to/dev.prediction.json. Repeat.",
    )
    parser.add_argument("--top-rows", type=int, default=1000)
    parser.add_argument(
        "--source-score-mode",
        choices=["rank", "score", "rank_score"],
        default="rank_score",
    )
    parser.add_argument("--weight-grid", default="0,0.25,0.5,1,2,4")
    parser.add_argument("--objective-k", type=int, default=4)
    parser.add_argument("--recall-k", type=int, nargs="+", default=DEFAULT_RECALL_KS)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument("--output-table-md")
    parser.add_argument("--output-prediction-json")
    return parser.parse_args()


def gold_by_qid(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    return {str(row["qid"]): row for row in rows}


def load_labeled_sources(raw_specs: list[str]) -> list[tuple[str, dict[str, dict[str, Any]], Path]]:
    if not raw_specs:
        raise ValueError("Pass at least one labeled source.")
    out: list[tuple[str, dict[str, dict[str, Any]], Path]] = []
    seen: set[str] = set()
    for spec in raw_specs:
        label, path = parse_labeled_path(spec)
        if label in seen:
            raise ValueError(f"Duplicate source label: {label}")
        seen.add(label)
        out.append((label, load_prediction(path), path))
    return out


def usable_qids(
    gold: dict[str, dict[str, Any]],
    sources: list[tuple[str, dict[str, dict[str, Any]], Path]],
) -> list[str]:
    source_qid_sets = [set(pred.keys()) for _label, pred, _path in sources]
    return sorted(set(gold).intersection(*source_qid_sets))


def source_predictions_for_builder(
    sources: list[tuple[str, dict[str, dict[str, Any]], Path]],
) -> list[tuple[str, dict[str, dict[str, Any]]]]:
    return [(label, pred) for label, pred, _path in sources]


def source_paths(sources: list[tuple[str, dict[str, dict[str, Any]], Path]]) -> list[str]:
    return [str(path) for _label, _pred, path in sources]


def main() -> None:
    args = parse_args()

    train_sources = load_labeled_sources(args.train_source)
    eval_sources = load_labeled_sources(args.eval_source)
    train_labels = [label for label, _pred, _path in train_sources]
    eval_labels = [label for label, _pred, _path in eval_sources]
    if train_labels != eval_labels:
        raise ValueError(
            "Train/eval source labels must match in the same order. "
            f"train={train_labels} eval={eval_labels}"
        )

    train_gold = gold_by_qid(Path(args.train_gold))
    eval_gold = gold_by_qid(Path(args.eval_gold))
    train_qids = usable_qids(train_gold, train_sources)
    eval_qids = usable_qids(eval_gold, eval_sources)
    if not train_qids:
        raise ValueError("No train qids are common to train gold and all train sources.")
    if not eval_qids:
        raise ValueError("No eval qids are common to eval gold and all eval sources.")

    train_source_doc_scores = build_source_doc_scores(
        source_predictions_for_builder(train_sources),
        qids=train_qids,
        top_rows=max(0, int(args.top_rows)),
        score_mode=str(args.source_score_mode),
    )
    eval_source_doc_scores = build_source_doc_scores(
        source_predictions_for_builder(eval_sources),
        qids=eval_qids,
        top_rows=max(0, int(args.top_rows)),
        score_mode=str(args.source_score_mode),
    )

    weight_values = parse_weight_grid(args.weight_grid)
    candidate_weights = [
        tuple(float(value) for value in combo)
        for combo in itertools.product(weight_values, repeat=len(train_sources))
        if any(float(value) > 0.0 for value in combo)
    ]
    if not candidate_weights:
        raise ValueError("Weight grid produced no non-zero configurations.")

    best_weights = candidate_weights[0]
    best_train_metrics = evaluate_weights(
        qids=train_qids,
        gold_by_qid=train_gold,
        source_doc_scores_by_qid=train_source_doc_scores,
        weights=best_weights,
        recall_ks=args.recall_k,
    )
    best_key = metric_tuple(best_train_metrics, objective_k=int(args.objective_k))
    searched: list[dict[str, Any]] = []
    for weights in candidate_weights:
        metrics = evaluate_weights(
            qids=train_qids,
            gold_by_qid=train_gold,
            source_doc_scores_by_qid=train_source_doc_scores,
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
        gold_by_qid=eval_gold,
        source_doc_scores_by_qid=eval_source_doc_scores,
        weights=best_weights,
        recall_ks=args.recall_k,
    )

    metric_rows: list[dict[str, Any]] = []
    source_baselines: list[dict[str, Any]] = []
    for idx, label in enumerate(train_labels):
        weights = tuple(1.0 if source_idx == idx else 0.0 for source_idx in range(len(train_labels)))
        train_metrics = evaluate_weights(
            qids=train_qids,
            gold_by_qid=train_gold,
            source_doc_scores_by_qid=train_source_doc_scores,
            weights=weights,
            recall_ks=args.recall_k,
        )
        eval_metrics = evaluate_weights(
            qids=eval_qids,
            gold_by_qid=eval_gold,
            source_doc_scores_by_qid=eval_source_doc_scores,
            weights=weights,
            recall_ks=args.recall_k,
        )
        source_baselines.append(
            {
                "label": label,
                "weights": list(weights),
                "train_metrics": train_metrics,
                "eval_metrics": eval_metrics,
            }
        )
        metric_rows.append(make_metric_row(label, "train", train_metrics, args.recall_k))
        metric_rows.append(make_metric_row(label, "eval", eval_metrics, args.recall_k))

    metric_rows.append(make_metric_row("train_tuned_fusion", "train", best_train_metrics, args.recall_k))
    metric_rows.append(make_metric_row("train_tuned_fusion", "eval", best_eval_metrics, args.recall_k))

    objective_key = f"doc@{int(args.objective_k)}"
    best_source_eval = max(
        (
            float(row.get(objective_key, 0.0))
            for row in metric_rows
            if row["split"] == "eval" and row["label"] != "train_tuned_fusion"
        ),
        default=0.0,
    )
    tuned_eval = float(
        next(
            row
            for row in metric_rows
            if row["split"] == "eval" and row["label"] == "train_tuned_fusion"
        ).get(objective_key, 0.0)
    )

    if args.output_prediction_json:
        write_prediction(
            Path(args.output_prediction_json),
            qids=eval_qids,
            source_doc_scores_by_qid=eval_source_doc_scores,
            weights=best_weights,
        )

    summary = {
        "train_gold": args.train_gold,
        "eval_gold": args.eval_gold,
        "source_labels": train_labels,
        "train_source_paths": source_paths(train_sources),
        "eval_source_paths": source_paths(eval_sources),
        "source_score_mode": args.source_score_mode,
        "top_rows": int(args.top_rows),
        "weight_grid": weight_values,
        "searched_config_count": len(candidate_weights),
        "objective_k": int(args.objective_k),
        "train_qid_count": len(train_qids),
        "eval_qid_count": len(eval_qids),
        "best_weights": dict(zip(train_labels, best_weights)),
        "best_train_metrics": best_train_metrics,
        "best_eval_metrics": best_eval_metrics,
        "source_baselines": source_baselines,
        "metric_rows": metric_rows,
        "eval_delta_vs_best_source": {
            objective_key: tuned_eval - best_source_eval,
            "tuned_eval": tuned_eval,
            "best_source_eval": best_source_eval,
        },
        "searched": searched,
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
