#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_RECALL_KS = [1, 2, 4, 5, 10, 20, 50, 100]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize multiple page-retrieval predictions in one ablation table. "
            "Each run is evaluated on the qids common to all supplied predictions and gold."
        )
    )
    parser.add_argument("--gold", required=True, help="Converted MMQA_dev.jsonl gold file.")
    parser.add_argument(
        "--run",
        nargs=2,
        action="append",
        metavar=("LABEL", "PREDICTION_JSON"),
        required=True,
        help="Run label and prediction JSON. Repeat for each ablation row.",
    )
    parser.add_argument(
        "--baseline-label",
        default="",
        help="Run label to use for recovered/lost deltas. Defaults to first --run label.",
    )
    parser.add_argument("--hit-k", type=int, default=4)
    parser.add_argument("--recall-k", dest="recall_ks", type=int, nargs="+", default=DEFAULT_RECALL_KS)
    parser.add_argument("--format", choices=["markdown", "csv", "tsv", "json"], default="markdown")
    parser.add_argument("--output", default="", help="Optional output path.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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

    if isinstance(payload, dict):
        iterable = payload.items()
    elif isinstance(payload, list):
        iterable = enumerate(payload)
    else:
        raise TypeError(f"Prediction JSON must be an object or list: {path}")

    rows: dict[str, dict[str, Any]] = {}
    for raw_key, row in iterable:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("qid", "")).strip() or str(raw_key).strip()
        if qid:
            rows[qid] = row
    return rows


def page_uid(doc_id: str, page_idx: int) -> str:
    return f"{doc_id}_page{int(page_idx)}"


def gold_page_uids(row: dict[str, Any]) -> set[str]:
    metadata = row.get("metadata", {})
    uids = {
        str(value).strip()
        for value in metadata.get("gold_page_uids", [])
        if str(value).strip()
    }
    for ctx in row.get("supporting_context", []):
        doc_id = str(ctx.get("doc_id", "")).strip()
        page_idx = ctx.get("page_idx", ctx.get("page_id"))
        if doc_id and page_idx is not None:
            uids.add(page_uid(doc_id, int(page_idx)))
    return uids


def gold_doc_ids(row: dict[str, Any]) -> set[str]:
    docs = {
        str(ctx.get("doc_id", "")).strip()
        for ctx in row.get("supporting_context", [])
        if str(ctx.get("doc_id", "")).strip()
    }
    for uid in gold_page_uids(row):
        if "_page" in uid:
            docs.add(uid.rsplit("_page", 1)[0])
    return docs


def parse_page_row(row: Any) -> tuple[str, int] | None:
    if not isinstance(row, (list, tuple)) or len(row) < 2:
        return None
    doc_id = str(row[0]).strip()
    if not doc_id:
        return None
    try:
        page_idx = int(row[1])
    except (TypeError, ValueError):
        return None
    return doc_id, page_idx


def ranked_pages(pred_row: dict[str, Any]) -> list[str]:
    pages = []
    for row in pred_row.get("page_retrieval_results", []):
        parsed = parse_page_row(row)
        if parsed is not None:
            pages.append(page_uid(parsed[0], parsed[1]))
    return pages


def ranked_docs(pred_row: dict[str, Any]) -> list[str]:
    docs = []
    seen = set()
    for row in pred_row.get("page_retrieval_results", []):
        parsed = parse_page_row(row)
        if parsed is None:
            continue
        doc_id = parsed[0]
        if doc_id not in seen:
            seen.add(doc_id)
            docs.append(doc_id)
    return docs


def first_rank(items: list[str], gold: set[str]) -> int | None:
    for idx, item in enumerate(items, start=1):
        if item in gold:
            return idx
    return None


def recall_at_k(items: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(items[:k]) & gold) / len(gold)


def movement_for_hit(base_rank: int | None, cand_rank: int | None, hit_k: int) -> str:
    base_hit = base_rank is not None and base_rank <= hit_k
    cand_hit = cand_rank is not None and cand_rank <= hit_k
    if not base_hit and cand_hit:
        return "recovered"
    if base_hit and not cand_hit:
        return "lost"
    if base_rank is None and cand_rank is None:
        return "missing_in_both"
    if base_rank is not None and cand_rank is not None and cand_rank < base_rank:
        return "improved_rank"
    if base_rank is not None and cand_rank is not None and cand_rank > base_rank:
        return "worsened_rank"
    return "unchanged"


def evaluate_run(
    prediction: dict[str, dict[str, Any]],
    gold_by_qid: dict[str, dict[str, Any]],
    qids: list[str],
    recall_ks: list[int],
) -> dict[str, Any]:
    per_qid = {}
    for qid in qids:
        pred_row = prediction[qid]
        gold_row = gold_by_qid[qid]
        pages = ranked_pages(pred_row)
        docs = ranked_docs(pred_row)
        gold_pages = gold_page_uids(gold_row)
        gold_docs = gold_doc_ids(gold_row)
        per_qid[qid] = {
            "page_rank": first_rank(pages, gold_pages),
            "doc_rank": first_rank(docs, gold_docs),
            "page_recall": {k: recall_at_k(pages, gold_pages, k) for k in recall_ks},
            "doc_recall": {k: recall_at_k(docs, gold_docs, k) for k in recall_ks},
        }
    return {"per_qid": per_qid}


def summarize_run(
    label: str,
    run_eval: dict[str, Any],
    baseline_eval: dict[str, Any],
    qids: list[str],
    hit_k: int,
    recall_ks: list[int],
) -> dict[str, Any]:
    per_qid = run_eval["per_qid"]
    base_per_qid = baseline_eval["per_qid"]
    n = len(qids)
    movement = Counter(
        movement_for_hit(base_per_qid[qid]["page_rank"], per_qid[qid]["page_rank"], hit_k)
        for qid in qids
    )
    row: dict[str, Any] = {
        "label": label,
        "n_qids": n,
        f"page_hit@{hit_k}": sum(
            1 for qid in qids if per_qid[qid]["page_rank"] is not None and per_qid[qid]["page_rank"] <= hit_k
        ),
        f"doc_hit@{hit_k}": sum(
            1 for qid in qids if per_qid[qid]["doc_rank"] is not None and per_qid[qid]["doc_rank"] <= hit_k
        ),
        "recovered": movement.get("recovered", 0),
        "lost": movement.get("lost", 0),
        "improved_rank": movement.get("improved_rank", 0),
        "worsened_rank": movement.get("worsened_rank", 0),
        "unchanged": movement.get("unchanged", 0),
        "missing_in_both": movement.get("missing_in_both", 0),
    }
    row[f"page_hit@{hit_k}_pct"] = row[f"page_hit@{hit_k}"] / n if n else 0.0
    row[f"doc_hit@{hit_k}_pct"] = row[f"doc_hit@{hit_k}"] / n if n else 0.0
    for k in recall_ks:
        row[f"page_recall@{k}"] = sum(per_qid[qid]["page_recall"][k] for qid in qids) / n if n else 0.0
        row[f"doc_recall@{k}"] = sum(per_qid[qid]["doc_recall"][k] for qid in qids) / n if n else 0.0
    return row


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def output_markdown(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def output_delimited(rows: list[dict[str, Any]], columns: list[str], delimiter: str) -> str:
    from io import StringIO

    handle = StringIO()
    writer = csv.DictWriter(handle, fieldnames=columns, delimiter=delimiter)
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return handle.getvalue()


def main() -> None:
    args = parse_args()
    runs = [(label, Path(path)) for label, path in args.run]
    labels = [label for label, _ in runs]
    if len(set(labels)) != len(labels):
        raise ValueError("Run labels must be unique.")
    baseline_label = args.baseline_label or labels[0]
    if baseline_label not in labels:
        raise ValueError(f"--baseline-label must match one of the run labels: {baseline_label}")

    gold_by_qid = {str(row["qid"]): row for row in read_jsonl(Path(args.gold))}
    predictions = {label: load_prediction(path) for label, path in runs}
    common_qids = set(gold_by_qid)
    for pred in predictions.values():
        common_qids &= set(pred)
    qids = sorted(common_qids)
    if not qids:
        raise ValueError("No qids are common to gold and all supplied predictions.")

    recall_ks = sorted(set(int(k) for k in args.recall_ks))
    evals = {
        label: evaluate_run(predictions[label], gold_by_qid, qids, recall_ks)
        for label in labels
    }
    baseline_eval = evals[baseline_label]
    rows = [
        summarize_run(label, evals[label], baseline_eval, qids, int(args.hit_k), recall_ks)
        for label in labels
    ]
    hit_k = int(args.hit_k)
    columns = [
        "label",
        "n_qids",
        f"page_hit@{hit_k}",
        f"page_hit@{hit_k}_pct",
        f"doc_hit@{hit_k}",
        f"doc_hit@{hit_k}_pct",
        "recovered",
        "lost",
        "improved_rank",
        "worsened_rank",
        "unchanged",
        "missing_in_both",
        *[f"page_recall@{k}" for k in recall_ks],
        *[f"doc_recall@{k}" for k in recall_ks],
    ]

    payload = {
        "gold": str(Path(args.gold)),
        "baseline_label": baseline_label,
        "hit_k": hit_k,
        "n_common_qids": len(qids),
        "runs": rows,
    }
    if args.format == "json":
        text = json.dumps(payload, indent=2) + "\n"
    elif args.format == "csv":
        text = output_delimited(rows, columns, ",")
    elif args.format == "tsv":
        text = output_delimited(rows, columns, "\t")
    else:
        text = output_markdown(rows, columns)

    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"saved_output {args.output}")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
