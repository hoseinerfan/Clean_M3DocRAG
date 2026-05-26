#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate retrieval with synthetic page gold labels constructed as "
            "page N of each gold supporting-context document. Useful for "
            "M3DocVQA/ImageListQ diagnostics when page-level gold is unavailable."
        )
    )
    parser.add_argument("--pred", required=True, help="Prediction JSON keyed by qid.")
    parser.add_argument(
        "--baseline-pred",
        default="",
        help=(
            "Optional baseline prediction JSON. When provided, report recovered/lost "
            "synthetic first-page hits relative to this ranking."
        ),
    )
    parser.add_argument("--gold", required=True, help="MMQA-style JSONL with supporting_context.")
    parser.add_argument(
        "--question-type",
        default="",
        help="Optional metadata.type filter, e.g. ImageListQ.",
    )
    parser.add_argument(
        "--first-page-idx",
        type=int,
        default=0,
        help="Synthetic gold page index for every gold doc. Default: 0.",
    )
    parser.add_argument(
        "--recall-k",
        dest="recall_ks",
        type=int,
        nargs="+",
        default=[1, 2, 4, 5, 10, 20, 50, 100, 500, 1000],
    )
    parser.add_argument("--hit-k", type=int, default=4, help="Boundary for hit/recovered/lost reporting.")
    parser.add_argument("--output-json", default="", help="Optional path to save the full JSON payload.")
    parser.add_argument("--json", action="store_true", help="Print full JSON payload.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def gold_doc_ids(row: dict[str, Any]) -> list[str]:
    return sorted(
        {
            str(ctx.get("doc_id", "")).strip()
            for ctx in row.get("supporting_context", [])
            if str(ctx.get("doc_id", "")).strip()
        }
    )


def synthetic_first_page_uids(row: dict[str, Any], first_page_idx: int) -> list[str]:
    return [f"{doc_id}_page{int(first_page_idx)}" for doc_id in gold_doc_ids(row)]


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / len(gold)


def first_rank(ranked: list[str], gold: set[str]) -> int | None:
    for idx, item in enumerate(ranked, start=1):
        if item in gold:
            return idx
    return None


def hit_at(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= int(k)


def movement_at_k(baseline_rank: int | None, output_rank: int | None, k: int) -> str:
    baseline_hit = hit_at(baseline_rank, k)
    output_hit = hit_at(output_rank, k)
    if not baseline_hit and output_hit:
        return "recovered"
    if baseline_hit and not output_hit:
        return "lost"
    if baseline_rank is None and output_rank is None:
        return "missing_in_both"
    if baseline_rank is not None and output_rank is not None:
        if output_rank < baseline_rank:
            return "improved_rank"
        if output_rank > baseline_rank:
            return "worsened_rank"
    return "unchanged"


def metadata_type(row: dict[str, Any]) -> str:
    return str(row.get("metadata", {}).get("type", "")).strip()


def load_prediction(path: str) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("predictions"), (dict, list)):
        payload = payload["predictions"]
    if isinstance(payload, dict):
        return {
            str(row.get("qid", key)).strip(): row
            for key, row in payload.items()
            if isinstance(row, dict)
        }
    if isinstance(payload, list):
        return {
            str(row.get("qid", "")).strip(): row
            for row in payload
            if isinstance(row, dict) and str(row.get("qid", "")).strip()
        }
    raise TypeError(f"Prediction JSON must be a list or object keyed by qid: {path}")


def prediction_metrics(
    pred_row: dict[str, Any],
    doc_gold: set[str],
    page_gold: set[str],
    recall_ks: list[int],
) -> dict[str, Any]:
    retrieval_rows = pred_row.get("page_retrieval_results", [])
    ranked_pages = [f"{row[0]}_page{int(row[1])}" for row in retrieval_rows]
    ranked_docs: list[str] = []
    seen_docs: set[str] = set()
    for row in retrieval_rows:
        doc_id = str(row[0])
        if doc_id not in seen_docs:
            seen_docs.add(doc_id)
            ranked_docs.append(doc_id)
    return {
        "first_gold_doc_rank": first_rank(ranked_docs, doc_gold),
        "first_synthetic_gold_page_rank": first_rank(ranked_pages, page_gold),
        "doc_recall_at_k": {str(k): recall_at_k(ranked_docs, doc_gold, k) for k in recall_ks},
        "synthetic_page_recall_at_k": {
            str(k): recall_at_k(ranked_pages, page_gold, k) for k in recall_ks
        },
    }


def main() -> None:
    args = parse_args()
    pred = load_prediction(args.pred)
    baseline = load_prediction(args.baseline_pred) if args.baseline_pred else {}
    gold_rows = read_jsonl(Path(args.gold))
    wanted_type = str(args.question_type).strip()
    hit_k = int(args.hit_k)

    per_qid: list[dict[str, Any]] = []
    skipped_no_prediction = 0
    skipped_no_baseline_prediction = 0
    skipped_no_gold_doc = 0
    for gold_row in gold_rows:
        if wanted_type and metadata_type(gold_row) != wanted_type:
            continue
        qid = str(gold_row.get("qid", "")).strip()
        if not qid:
            continue
        if qid not in pred:
            skipped_no_prediction += 1
            continue
        if baseline and qid not in baseline:
            skipped_no_baseline_prediction += 1
            continue
        doc_gold = set(gold_doc_ids(gold_row))
        if not doc_gold:
            skipped_no_gold_doc += 1
            continue
        page_gold = set(synthetic_first_page_uids(gold_row, int(args.first_page_idx)))
        item = {
            "qid": qid,
            "question": gold_row.get("question", ""),
            "metadata_type": metadata_type(gold_row),
            "gold_doc_ids": sorted(doc_gold),
            "synthetic_gold_page_uids": sorted(page_gold),
            **prediction_metrics(pred[qid], doc_gold, page_gold, args.recall_ks),
        }
        if baseline:
            baseline_metrics = prediction_metrics(
                baseline[qid], doc_gold, page_gold, args.recall_ks
            )
            item["baseline"] = baseline_metrics
            item["synthetic_page_movement_vs_baseline"] = movement_at_k(
                baseline_metrics["first_synthetic_gold_page_rank"],
                item["first_synthetic_gold_page_rank"],
                hit_k,
            )
            item["doc_movement_vs_baseline"] = movement_at_k(
                baseline_metrics["first_gold_doc_rank"],
                item["first_gold_doc_rank"],
                hit_k,
            )
        per_qid.append(item)

    summary: dict[str, Any] = {
        "prediction": str(args.pred),
        "baseline_prediction": str(args.baseline_pred),
        "n_qids": len(per_qid),
        "question_type": wanted_type,
        "first_page_idx": int(args.first_page_idx),
        "hit_k": hit_k,
        "assumption": "Every supporting document page at first_page_idx is treated as synthetic gold.",
        "skipped_no_prediction": skipped_no_prediction,
        "skipped_no_baseline_prediction": skipped_no_baseline_prediction,
        "skipped_no_gold_doc": skipped_no_gold_doc,
        "synthetic_page_recall_at_k": {},
        "doc_recall_at_k": {},
    }
    for k in args.recall_ks:
        key = str(k)
        summary["synthetic_page_recall_at_k"][key] = (
            sum(item["synthetic_page_recall_at_k"][key] for item in per_qid) / len(per_qid)
            if per_qid
            else 0.0
        )
        summary["doc_recall_at_k"][key] = (
            sum(item["doc_recall_at_k"][key] for item in per_qid) / len(per_qid)
            if per_qid
            else 0.0
        )
    summary["synthetic_page_hit_at_k_count"] = sum(
        1
        for item in per_qid
        if item["first_synthetic_gold_page_rank"] is not None
        and int(item["first_synthetic_gold_page_rank"]) <= hit_k
    )
    summary["doc_hit_at_k_count"] = sum(
        1
        for item in per_qid
        if item["first_gold_doc_rank"] is not None and int(item["first_gold_doc_rank"]) <= hit_k
    )
    summary[f"synthetic_page_hit_at_{hit_k}_count"] = summary["synthetic_page_hit_at_k_count"]
    summary[f"doc_hit_at_{hit_k}_count"] = summary["doc_hit_at_k_count"]

    if baseline:
        summary["baseline_synthetic_page_recall_at_k"] = {}
        summary["baseline_doc_recall_at_k"] = {}
        for k in args.recall_ks:
            key = str(k)
            summary["baseline_synthetic_page_recall_at_k"][key] = (
                sum(item["baseline"]["synthetic_page_recall_at_k"][key] for item in per_qid)
                / len(per_qid)
                if per_qid
                else 0.0
            )
            summary["baseline_doc_recall_at_k"][key] = (
                sum(item["baseline"]["doc_recall_at_k"][key] for item in per_qid)
                / len(per_qid)
                if per_qid
                else 0.0
            )
        summary["baseline_synthetic_page_hit_at_k_count"] = sum(
            1
            for item in per_qid
            if hit_at(item["baseline"]["first_synthetic_gold_page_rank"], hit_k)
        )
        summary["baseline_doc_hit_at_k_count"] = sum(
            1 for item in per_qid if hit_at(item["baseline"]["first_gold_doc_rank"], hit_k)
        )
        page_movement = {
            key: sum(1 for item in per_qid if item["synthetic_page_movement_vs_baseline"] == key)
            for key in ["recovered", "lost", "improved_rank", "worsened_rank", "unchanged", "missing_in_both"]
        }
        doc_movement = {
            key: sum(1 for item in per_qid if item["doc_movement_vs_baseline"] == key)
            for key in ["recovered", "lost", "improved_rank", "worsened_rank", "unchanged", "missing_in_both"]
        }
        summary["synthetic_page_movement_vs_baseline_counts"] = page_movement
        summary["doc_movement_vs_baseline_counts"] = doc_movement
        summary["synthetic_page_recovered"] = page_movement["recovered"]
        summary["synthetic_page_lost"] = page_movement["lost"]
        summary["synthetic_page_net_recovered"] = page_movement["recovered"] - page_movement["lost"]
        summary["doc_recovered"] = doc_movement["recovered"]
        summary["doc_lost"] = doc_movement["lost"]
        summary["doc_net_recovered"] = doc_movement["recovered"] - doc_movement["lost"]

    payload = {"summary": summary, "per_qid": per_qid}
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"saved_json {output_path}")
    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(f"n_qids {summary['n_qids']}")
    print(f"question_type {summary['question_type']}")
    print(f"first_page_idx {summary['first_page_idx']}")
    print(f"assumption {summary['assumption']}")
    print(f"skipped_no_prediction {summary['skipped_no_prediction']}")
    print(f"skipped_no_baseline_prediction {summary['skipped_no_baseline_prediction']}")
    print(f"skipped_no_gold_doc {summary['skipped_no_gold_doc']}")
    print(f"synthetic_page_recall_at_k {summary['synthetic_page_recall_at_k']}")
    print(f"doc_recall_at_k {summary['doc_recall_at_k']}")
    print(f"synthetic_page_hit_at_{hit_k}_count {summary['synthetic_page_hit_at_k_count']}")
    print(f"doc_hit_at_{hit_k}_count {summary['doc_hit_at_k_count']}")
    if baseline:
        print(f"baseline_synthetic_page_recall_at_k {summary['baseline_synthetic_page_recall_at_k']}")
        print(f"baseline_doc_recall_at_k {summary['baseline_doc_recall_at_k']}")
        print(
            f"baseline_synthetic_page_hit_at_{hit_k}_count "
            f"{summary['baseline_synthetic_page_hit_at_k_count']}"
        )
        print(f"baseline_doc_hit_at_{hit_k}_count {summary['baseline_doc_hit_at_k_count']}")
        print(
            f"synthetic_page_movement_vs_baseline_counts "
            f"{summary['synthetic_page_movement_vs_baseline_counts']}"
        )
        print(
            f"synthetic_page_recovered {summary['synthetic_page_recovered']} "
            f"synthetic_page_lost {summary['synthetic_page_lost']} "
            f"synthetic_page_net_recovered {summary['synthetic_page_net_recovered']}"
        )
        print(f"doc_movement_vs_baseline_counts {summary['doc_movement_vs_baseline_counts']}")
        print(
            f"doc_recovered {summary['doc_recovered']} "
            f"doc_lost {summary['doc_lost']} doc_net_recovered {summary['doc_net_recovered']}"
        )


if __name__ == "__main__":
    main()
