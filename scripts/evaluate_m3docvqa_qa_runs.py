#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from m3docrag.datasets.m3_docvqa.evaluate import (
    MULTI_HOP_QUESTION_TYPES,
    eval_retrieval,
    evaluate_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate multiple M3DocVQA QA prediction files on the same gold JSONL."
    )
    parser.add_argument("--gold", required=True, help="Gold JSONL, optionally filtered to a subset.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="LABEL=prediction.json. Can be repeated.",
    )
    parser.add_argument("--output-md", default="", help="Optional Markdown table output.")
    parser.add_argument("--output-csv", default="", help="Optional CSV table output.")
    parser.add_argument("--output-json", default="", help="Optional full JSON output.")
    return parser.parse_args()


def parse_run(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        path = Path(spec)
        return path.stem, path
    label, raw_path = spec.split("=", 1)
    return label.strip() or Path(raw_path).stem, Path(raw_path)


def gold_qid_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


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
    if not isinstance(payload, dict):
        raise TypeError(f"Prediction file must be a JSON object: {path}")
    return {str(qid): row for qid, row in payload.items() if isinstance(row, dict)}


def pct(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.2f}"


def recall_at(scores: dict[str, Any], k: int) -> float | None:
    values = scores.get("average_recall_at_k", {})
    return values.get(k, values.get(str(k)))


def summarize(label: str, pred_path: Path, gold_path: Path, n_gold: int) -> dict[str, Any]:
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction for {label}: {pred_path}")

    predictions = load_prediction(pred_path)
    examples = load_jsonl(gold_path)
    qid2predicted_answers: dict[str, str] = {}
    qid2retrieval_results: dict[str, list[Any]] = {}
    for qid, row in predictions.items():
        qid2predicted_answers[qid] = str(row.get("pred_answer", "")).strip()
        qid2retrieval_results[qid] = row.get("page_retrieval_results", [])

    gold_answers: dict[str, list[list[str]]] = {}
    answer_modalities: dict[str, str] = {}
    hop_types: dict[str, str] = {}
    for example in examples:
        qid = str(example["qid"])
        gold_answer = [str(item["answer"]) for item in example["answers"]]
        gold_answers[qid] = [gold_answer]
        answer_modality = {str(item["modality"]) for item in example["answers"]}
        if len(answer_modality) != 1:
            raise ValueError(f"Expected one answer modality for {qid}, got {sorted(answer_modality)}")
        answer_modalities[qid] = answer_modality.pop()
        qtype = str(example["metadata"]["type"])
        hop_types[qid] = "Multi-hop" if qtype in MULTI_HOP_QUESTION_TYPES else "Single-hop"

    retrieval_scores = eval_retrieval(qid2retrieval_results, examples)
    overall, _ = evaluate_predictions(qid2predicted_answers, gold_answers)
    _, _, modalities = evaluate_predictions(qid2predicted_answers, gold_answers, answer_modalities)
    _, _, hops = evaluate_predictions(qid2predicted_answers, gold_answers, hop_types)
    scores = {
        "overall": overall,
        "modalities": modalities,
        "hop_types": hops,
        "average_recall_at_k": retrieval_scores.get("average_recall_at_k", {}),
    }
    return {
        "label": label,
        "prediction_path": str(pred_path),
        "n_gold": n_gold,
        "em": overall.get("list_em"),
        "f1": overall.get("list_f1"),
        "support_doc_recall@1": recall_at(scores, 1),
        "support_doc_recall@2": recall_at(scores, 2),
        "support_doc_recall@4": recall_at(scores, 4),
        "support_doc_recall@5": recall_at(scores, 5),
        "support_doc_recall@10": recall_at(scores, 10),
        "image_f1": modalities.get("image", {}).get("list_f1"),
        "table_f1": modalities.get("table", {}).get("list_f1"),
        "text_f1": modalities.get("text", {}).get("list_f1"),
        "single_hop_f1": hops.get("Single-hop", {}).get("list_f1"),
        "multi_hop_f1": hops.get("Multi-hop", {}).get("list_f1"),
        "raw_scores": scores,
    }


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "| method | n | support-doc recall@4 | EM | F1 | image F1 | table F1 | text F1 | single-hop F1 | multi-hop F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {label} | {n_gold} | {r4:.4f} | {em} | {f1} | {image_f1} | {table_f1} | "
            "{text_f1} | {single_hop_f1} | {multi_hop_f1} |".format(
                label=row["label"],
                n_gold=int(row["n_gold"]),
                r4=float(row.get("support_doc_recall@4") or 0.0),
                em=pct(row.get("em")),
                f1=pct(row.get("f1")),
                image_f1=pct(row.get("image_f1")),
                table_f1=pct(row.get("table_f1")),
                text_f1=pct(row.get("text_f1")),
                single_hop_f1=pct(row.get("single_hop_f1")),
                multi_hop_f1=pct(row.get("multi_hop_f1")),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "label",
        "n_gold",
        "support_doc_recall@1",
        "support_doc_recall@2",
        "support_doc_recall@4",
        "support_doc_recall@5",
        "support_doc_recall@10",
        "em",
        "f1",
        "image_f1",
        "table_f1",
        "text_f1",
        "single_hop_f1",
        "multi_hop_f1",
        "prediction_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def main() -> None:
    args = parse_args()
    gold_path = Path(args.gold)
    n_gold = gold_qid_count(gold_path)
    rows = [summarize(label, path, gold_path, n_gold) for label, path in map(parse_run, args.run)]

    if args.output_md:
        write_md(Path(args.output_md), rows)
        print(f"saved_output_md={args.output_md}")
    if args.output_csv:
        write_csv(Path(args.output_csv), rows)
        print(f"saved_output_csv={args.output_csv}")
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"saved_output_json={args.output_json}")

    if not (args.output_md or args.output_csv or args.output_json):
        print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
