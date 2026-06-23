#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


METRICS = ["page_mrr", "doc_mrr", "page@4", "doc@4", "page@10", "page@100"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize CAPP seed stability from retrieval evaluation and model JSON files."
    )
    parser.add_argument("--eval-json", required=True)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Seed and model path as SEED=path/to/model.json. Repeat once per seed.",
    )
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def parse_seed_path(spec: str) -> tuple[int, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected SEED=path, got {spec!r}")
    raw_seed, raw_path = spec.split("=", 1)
    return int(raw_seed), Path(raw_path)


def model_tuning(model: dict[str, Any]) -> tuple[float, float, int]:
    train_metadata = model.get("train_metadata", {})
    tuning = train_metadata.get("tuning_summary") or {}
    alpha = tuning.get("selected_blend_alpha", train_metadata.get("selected_blend_alpha"))
    value = tuning.get("optimized_metric_value")
    count = tuning.get("tune_eval_qid_count")
    if alpha is None or value is None or count is None:
        raise ValueError("Model JSON is missing held-out alpha-tuning metadata")
    return float(alpha), float(value), int(count)


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def build_summary(
    eval_rows: list[dict[str, Any]], model_specs: list[tuple[int, Path]]
) -> dict[str, Any]:
    eval_by_label = {str(row.get("label")): row for row in eval_rows}
    baseline = eval_by_label.get("GPP")
    seed_rows: list[dict[str, Any]] = []
    for seed, model_path in sorted(model_specs):
        label = f"seed_{seed}"
        if label not in eval_by_label:
            raise ValueError(f"Evaluation JSON is missing {label}")
        model = json.loads(model_path.read_text(encoding="utf-8"))
        alpha, heldout_page4, tune_count = model_tuning(model)
        eval_row = eval_by_label[label]
        row: dict[str, Any] = {
            "seed": seed,
            "selected_blend_alpha": alpha,
            "heldout_train_page@4": heldout_page4,
            "heldout_train_qids": tune_count,
            "n_eval": int(eval_row.get("n_eval", 0)),
            "recovered@4": int(eval_row.get("recovered@4", 0)),
            "lost@4": int(eval_row.get("lost@4", 0)),
            "net@4": int(eval_row.get("net@4", 0)),
        }
        for metric in METRICS:
            row[metric] = float(eval_row.get(metric, 0.0))
        seed_rows.append(row)

    aggregate: dict[str, Any] = {"seed_count": len(seed_rows)}
    for metric in METRICS:
        values = [float(row[metric]) for row in seed_rows]
        mean, std = mean_std(values)
        aggregate[metric] = {
            "mean": mean,
            "sample_std": std,
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values),
        }
    return {"baseline": baseline, "seeds": seed_rows, "aggregate": aggregate}


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# CAPP Seed Stability",
        "",
        "All runs use cap 20 and differ only in the random seed. Alpha is selected on the held-out training split for page@4.",
        "",
        "| seed | selected alpha | held-out train page@4 | tune qids | dev page MRR | dev doc MRR | dev page@4 | dev doc@4 | recovered@4 | lost@4 | net@4 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["seeds"]:
        lines.append(
            "| {seed} | {selected_blend_alpha:.2f} | {heldout_train_page@4:.4f} | "
            "{heldout_train_qids} | {page_mrr:.4f} | {doc_mrr:.4f} | {page@4:.4f} | "
            "{doc@4:.4f} | {recovered@4} | {lost@4} | {net@4} |".format(**row)
        )

    lines.extend(
        [
            "",
            "| metric | mean | sample std | min | max | range |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for metric in METRICS:
        row = summary["aggregate"][metric]
        lines.append(
            f"| {metric} | {row['mean']:.4f} | {row['sample_std']:.4f} | "
            f"{row['min']:.4f} | {row['max']:.4f} | {row['range']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    eval_rows = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))
    summary = build_summary(eval_rows, [parse_seed_path(spec) for spec in args.model])
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
