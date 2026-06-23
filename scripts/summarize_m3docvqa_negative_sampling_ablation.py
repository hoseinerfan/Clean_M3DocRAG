#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


METRICS = ["page_mrr", "doc_mrr", "page@4", "doc@4", "page@10", "page@100"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize CAPP negative-sampling results.")
    parser.add_argument("--eval-json", required=True)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Run and model path as STRATEGY:SEED=path/to/model.json.",
    )
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def parse_model_spec(spec: str) -> tuple[str, int, Path]:
    if "=" not in spec or ":" not in spec.split("=", 1)[0]:
        raise ValueError(f"Expected STRATEGY:SEED=path, got {spec!r}")
    run, raw_path = spec.split("=", 1)
    strategy, raw_seed = run.rsplit(":", 1)
    return strategy, int(raw_seed), Path(raw_path)


def tuning_values(model: dict[str, Any]) -> tuple[float, float, int]:
    train_metadata = model.get("train_metadata", {})
    tuning = train_metadata.get("tuning_summary") or {}
    alpha = tuning.get("selected_blend_alpha", train_metadata.get("selected_blend_alpha"))
    score = tuning.get("optimized_metric_value")
    count = tuning.get("tune_eval_qid_count")
    if alpha is None or score is None or count is None:
        raise ValueError("Model JSON is missing held-out page@4 tuning metadata")
    return float(alpha), float(score), int(count)


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def build_summary(
    eval_rows: list[dict[str, Any]], model_specs: list[tuple[str, int, Path]]
) -> dict[str, Any]:
    eval_by_label = {str(row.get("label")): row for row in eval_rows}
    runs: list[dict[str, Any]] = []
    for strategy, seed, model_path in sorted(model_specs):
        label = f"{strategy}_seed_{seed}"
        if label not in eval_by_label:
            raise ValueError(f"Evaluation JSON is missing {label}")
        alpha, heldout_score, tune_count = tuning_values(
            json.loads(model_path.read_text(encoding="utf-8"))
        )
        eval_row = eval_by_label[label]
        row: dict[str, Any] = {
            "strategy": strategy,
            "seed": seed,
            "selected_blend_alpha": alpha,
            "heldout_train_page@4": heldout_score,
            "heldout_train_qids": tune_count,
            "n_eval": int(eval_row.get("n_eval", 0)),
            "recovered@4": int(eval_row.get("recovered@4", 0)),
            "lost@4": int(eval_row.get("lost@4", 0)),
            "net@4": int(eval_row.get("net@4", 0)),
        }
        for metric in METRICS:
            row[metric] = float(eval_row.get(metric, 0.0))
        runs.append(row)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        grouped[str(row["strategy"])].append(row)
    aggregate: list[dict[str, Any]] = []
    for strategy in sorted(grouped):
        rows = grouped[strategy]
        aggregate_row: dict[str, Any] = {
            "strategy": strategy,
            "seed_count": len(rows),
            "selected_alphas": [float(row["selected_blend_alpha"]) for row in rows],
        }
        for metric in ["heldout_train_page@4", *METRICS]:
            values = [float(row[metric]) for row in rows]
            mean, std = mean_std(values)
            aggregate_row[metric] = {
                "mean": mean,
                "sample_std": std,
                "min": min(values),
                "max": max(values),
            }
        aggregate.append(aggregate_row)
    return {"baseline": eval_by_label.get("GPP"), "runs": runs, "aggregate": aggregate}


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# CAPP Negative-Sampling Ablation",
        "",
        "Each policy uses 50 negatives per labeled question, cap 20, the logistic scorer, and held-out-train page@4 alpha selection.",
        "",
        "| strategy | seed | alpha | held-out train page@4 | dev page MRR | dev page@4 | dev doc@4 | recovered@4 | lost@4 | net@4 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["runs"]:
        lines.append(
            "| {strategy} | {seed} | {selected_blend_alpha:.2f} | {heldout_train_page@4:.4f} | "
            "{page_mrr:.4f} | {page@4:.4f} | {doc@4:.4f} | {recovered@4} | {lost@4} | "
            "{net@4} |".format(**row)
        )
    lines.extend(
        [
            "",
            "| strategy | alphas | held-out page@4 mean +/- std | dev page@4 mean +/- std | dev page MRR mean +/- std | dev page@10 mean +/- std |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["aggregate"]:
        heldout = row["heldout_train_page@4"]
        page4 = row["page@4"]
        page_mrr = row["page_mrr"]
        page10 = row["page@10"]
        alphas = ", ".join(f"{value:.2f}" for value in row["selected_alphas"])
        lines.append(
            f"| {row['strategy']} | {alphas} | {heldout['mean']:.4f} +/- {heldout['sample_std']:.4f} | "
            f"{page4['mean']:.4f} +/- {page4['sample_std']:.4f} | "
            f"{page_mrr['mean']:.4f} +/- {page_mrr['sample_std']:.4f} | "
            f"{page10['mean']:.4f} +/- {page10['sample_std']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    summary = build_summary(
        json.loads(Path(args.eval_json).read_text(encoding="utf-8")),
        [parse_model_spec(spec) for spec in args.model],
    )
    output_md = Path(args.output_md)
    output_json = Path(args.output_json)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(summary), encoding="utf-8")
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"saved_output_md={output_md}")
    print(f"saved_output_json={output_json}")


if __name__ == "__main__":
    main()
