#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse multiple page-retrieval prediction JSONs with unweighted reciprocal-rank "
            "fusion. This is intended for label-free graph-view fusion: each input is an "
            "independent graph/PPR view, and no tuned selector threshold is used."
        )
    )
    parser.add_argument(
        "--prediction",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Prediction JSON to fuse. Repeat at least twice. LABEL= is optional.",
    )
    parser.add_argument("--gold", help="Optional MMQA_<split>.jsonl for reporting metrics.")
    parser.add_argument(
        "--baseline",
        help="Optional baseline prediction JSON used only for recovered/lost/worsened reporting.",
    )
    parser.add_argument(
        "--rrf-k",
        type=float,
        default=60.0,
        help="RRF smoothing constant. Default: 60, the standard robust IR setting.",
    )
    parser.add_argument(
        "--input-top-pages",
        type=int,
        default=1000,
        help="Maximum pages read from each input ranking per qid. Use 0 for all pages.",
    )
    parser.add_argument(
        "--output-top-pages",
        type=int,
        default=1000,
        help="Maximum fused pages written per qid. Use 0 for all fused pages.",
    )
    parser.add_argument(
        "--recall-k",
        dest="recall_ks",
        type=int,
        nargs="+",
        default=[1, 2, 4, 5, 10, 20, 50, 100],
        help="Recall cutoffs for optional gold reporting.",
    )
    parser.add_argument(
        "--hit-k",
        type=int,
        default=4,
        help="Hit cutoff for optional recovered/lost/worsened reporting. Default: 4.",
    )
    parser.add_argument("--output-prediction-json", required=True)
    parser.add_argument("--output-summary-json", required=True)
    return parser.parse_args()


def parse_labeled_path(value: str) -> tuple[str, Path]:
    raw = str(value).strip()
    if not raw:
        raise ValueError("Empty --prediction value")
    if "=" in raw:
        label, path = raw.split("=", 1)
        label = label.strip()
        parsed = Path(path.strip())
    else:
        parsed = Path(raw)
        label = parsed.stem
    if not label:
        label = parsed.stem
    return label, parsed


def dedupe_labels(items: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    counts: Counter[str] = Counter()
    seen_paths: set[str] = set()
    deduped: list[tuple[str, Path]] = []
    for label, path in items:
        path_key = str(path.expanduser())
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        counts[label] += 1
        unique_label = label if counts[label] == 1 else f"{label}_{counts[label]}"
        deduped.append((unique_label, path))
    return deduped


def load_prediction(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "predictions" in payload and isinstance(
        payload["predictions"],
        (dict, list),
    ):
        payload = payload["predictions"]

    rows_by_qid: dict[str, dict[str, Any]] = {}
    if isinstance(payload, list):
        iterable: Any = enumerate(payload)
    elif isinstance(payload, dict):
        iterable = payload.items()
    else:
        raise TypeError(f"Prediction JSON must be a list or object: {path}")

    for raw_key, row in iterable:
        if not isinstance(row, dict):
            raise TypeError(f"Prediction row must be an object: {path} key={raw_key!r}")
        qid = str(row.get("qid", "")).strip() or str(raw_key).strip()
        if not qid:
            raise ValueError(f"Prediction row is missing qid: {path} key={raw_key!r}")
        if qid in rows_by_qid:
            raise ValueError(f"Duplicate qid after normalization: {qid} ({path})")
        rows_by_qid[qid] = row
    return rows_by_qid


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def page_uid(doc_id: object, page_idx: object) -> str:
    return f"{doc_id}_page{int(page_idx)}"


def parse_page_uid(uid: str) -> tuple[str, int]:
    marker = "_page"
    if marker not in uid:
        raise ValueError(f"Invalid page uid: {uid}")
    doc_id, page_idx = uid.rsplit(marker, 1)
    return doc_id, int(page_idx)


def prediction_rows(row: dict[str, Any]) -> list[Any]:
    rows = row.get("page_retrieval_results", [])
    return rows if isinstance(rows, list) else []


def ranked_page_uids(row: dict[str, Any], limit: int = 0) -> list[str]:
    pages: list[str] = []
    seen: set[str] = set()
    for item in prediction_rows(row):
        if not isinstance(item, list) or len(item) < 2:
            continue
        try:
            uid = page_uid(item[0], item[1])
        except (TypeError, ValueError):
            continue
        if uid in seen:
            continue
        seen.add(uid)
        pages.append(uid)
        if limit > 0 and len(pages) >= limit:
            break
    return pages


def ranked_doc_ids_from_pages(page_uids: list[str]) -> list[str]:
    docs: list[str] = []
    seen: set[str] = set()
    for uid in page_uids:
        doc_id, _ = parse_page_uid(uid)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        docs.append(doc_id)
    return docs


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
            uids.add(page_uid(doc_id, page_idx))
    return uids


def gold_doc_ids(row: dict[str, Any]) -> set[str]:
    metadata = row.get("metadata", {})
    doc_ids = {
        str(value).strip()
        for value in metadata.get("gold_doc_ids", [])
        if str(value).strip()
    }
    for ctx in row.get("supporting_context", []):
        doc_id = str(ctx.get("doc_id", "")).strip()
        if doc_id:
            doc_ids.add(doc_id)
    return doc_ids


def recall_at_k(ranked: list[str], gold: set[str], topk: int) -> float | None:
    if not gold:
        return None
    return len(set(ranked[:topk]) & gold) / float(len(gold))


def first_rank(ranked: list[str], gold: set[str]) -> int | None:
    for idx, value in enumerate(ranked, start=1):
        if value in gold:
            return idx
    return None


def hit_at(rank: int | None, topk: int) -> bool:
    return rank is not None and int(rank) <= int(topk)


def movement_for_hit(
    baseline_rank: int | None,
    candidate_rank: int | None,
    topk: int,
) -> str:
    baseline_hit = hit_at(baseline_rank, topk)
    candidate_hit = hit_at(candidate_rank, topk)
    if not baseline_hit and candidate_hit:
        return "recovered"
    if baseline_hit and not candidate_hit:
        return "lost"
    if baseline_rank is None and candidate_rank is None:
        return "missing_in_both"
    if baseline_rank is not None and candidate_rank is not None and candidate_rank < baseline_rank:
        return "improved_rank"
    if baseline_rank is not None and candidate_rank is not None and candidate_rank > baseline_rank:
        return "worsened_rank"
    return "unchanged"


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def fuse_qid(
    *,
    qid: str,
    predictions: dict[str, dict[str, dict[str, Any]]],
    rrf_k: float,
    input_top_pages: int,
    output_top_pages: int,
) -> tuple[list[list[Any]], dict[str, Any]]:
    score_by_uid: dict[str, float] = {}
    rank_by_label: dict[str, dict[str, int]] = {}
    best_rank_by_uid: dict[str, int] = {}
    support_by_uid: dict[str, int] = {}

    for label, prediction in predictions.items():
        pages = ranked_page_uids(prediction[qid], input_top_pages)
        rank_by_label[label] = {}
        for rank, uid in enumerate(pages, start=1):
            rank_by_label[label][uid] = rank
            score_by_uid[uid] = score_by_uid.get(uid, 0.0) + 1.0 / (rrf_k + float(rank))
            best_rank_by_uid[uid] = min(best_rank_by_uid.get(uid, 10**9), rank)
            support_by_uid[uid] = support_by_uid.get(uid, 0) + 1

    labels = list(predictions)
    ranked_uids = sorted(
        score_by_uid,
        key=lambda uid: (
            -score_by_uid[uid],
            -support_by_uid[uid],
            best_rank_by_uid[uid],
            [rank_by_label[label].get(uid, 10**9) for label in labels],
            uid,
        ),
    )
    if output_top_pages > 0:
        ranked_uids = ranked_uids[:output_top_pages]

    fused_rows: list[list[Any]] = []
    for uid in ranked_uids:
        doc_id, page_idx = parse_page_uid(uid)
        fused_rows.append([doc_id, page_idx, float(score_by_uid[uid])])

    summary = {
        "qid": qid,
        "fused_page_count": len(fused_rows),
        "mean_top4_support_count": mean(
            [float(support_by_uid[uid]) for uid in ranked_uids[:4]]
        ),
        "top_pages": [
            {
                "page_uid": uid,
                "score": float(score_by_uid[uid]),
                "support_count": int(support_by_uid[uid]),
                "best_rank": int(best_rank_by_uid[uid]),
                "ranks": {
                    label: rank_by_label[label][uid]
                    for label in labels
                    if uid in rank_by_label[label]
                },
            }
            for uid in ranked_uids[:10]
        ],
    }
    return fused_rows, summary


def main() -> None:
    args = parse_args()
    labeled_paths = dedupe_labels([parse_labeled_path(value) for value in args.prediction])
    if len(labeled_paths) < 2:
        raise ValueError("Provide at least two --prediction inputs.")

    predictions = {label: load_prediction(path) for label, path in labeled_paths}
    qid_sets = {label: set(prediction) for label, prediction in predictions.items()}
    common_qids = set.intersection(*qid_sets.values())
    if not common_qids:
        raise ValueError("Prediction inputs have no qids in common.")
    missing_by_label = {
        label: len(set.union(*qid_sets.values()) - qids)
        for label, qids in qid_sets.items()
    }

    gold_rows = {
        str(row.get("qid", "")).strip(): row
        for row in read_jsonl(Path(args.gold))
        if str(row.get("qid", "")).strip()
    } if args.gold else {}
    baseline = load_prediction(Path(args.baseline)) if args.baseline else {}

    fused_payload: dict[str, dict[str, Any]] = {}
    per_qid: list[dict[str, Any]] = []
    page_recall_values: dict[int, list[float]] = {int(k): [] for k in args.recall_ks}
    doc_recall_values: dict[int, list[float]] = {int(k): [] for k in args.recall_ks}
    movement_counts: Counter[str] = Counter()

    first_label = labeled_paths[0][0]
    for qid in sorted(common_qids):
        fused_rows, row_summary = fuse_qid(
            qid=qid,
            predictions=predictions,
            rrf_k=float(args.rrf_k),
            input_top_pages=int(args.input_top_pages),
            output_top_pages=int(args.output_top_pages),
        )
        fused_pages = [page_uid(row[0], row[1]) for row in fused_rows]
        fused_docs = ranked_doc_ids_from_pages(fused_pages)
        template = predictions[first_label][qid]
        fused_payload[qid] = {
            "pred_answer": template.get("pred_answer", ""),
            "page_retrieval_results": fused_rows,
            "qid": qid,
            "question": template.get("question", ""),
            "top_retrieved_docs": fused_docs[:10],
            "reranker_metadata": {
                "fusion_method": "unweighted_page_rrf",
                "rrf_k": float(args.rrf_k),
                "input_top_pages": int(args.input_top_pages),
                "output_top_pages": int(args.output_top_pages),
                "prediction_paths": {
                    label: str(path) for label, path in labeled_paths
                },
            },
        }

        if qid in gold_rows:
            page_gold = gold_page_uids(gold_rows[qid])
            doc_gold = gold_doc_ids(gold_rows[qid])
            fused_page_rank = first_rank(fused_pages, page_gold)
            fused_doc_rank = first_rank(fused_docs, doc_gold)
            row_summary["first_gold_page_rank"] = fused_page_rank
            row_summary["first_gold_doc_rank"] = fused_doc_rank
            row_summary["page_recall_at_k"] = {}
            row_summary["doc_recall_at_k"] = {}
            for cutoff in args.recall_ks:
                cutoff = int(cutoff)
                page_value = recall_at_k(fused_pages, page_gold, cutoff)
                doc_value = recall_at_k(fused_docs, doc_gold, cutoff)
                row_summary["page_recall_at_k"][str(cutoff)] = page_value
                row_summary["doc_recall_at_k"][str(cutoff)] = doc_value
                if page_value is not None:
                    page_recall_values[cutoff].append(float(page_value))
                if doc_value is not None:
                    doc_recall_values[cutoff].append(float(doc_value))
            if baseline and qid in baseline:
                baseline_pages = ranked_page_uids(
                    baseline[qid],
                    max(int(args.output_top_pages), max(args.recall_ks), int(args.hit_k)),
                )
                baseline_rank = first_rank(baseline_pages, page_gold)
                movement = movement_for_hit(baseline_rank, fused_page_rank, int(args.hit_k))
                row_summary["baseline_first_gold_page_rank"] = baseline_rank
                row_summary["movement_vs_baseline"] = movement
                movement_counts[movement] += 1
        per_qid.append(row_summary)

    output_prediction_json = Path(args.output_prediction_json)
    output_prediction_json.parent.mkdir(parents=True, exist_ok=True)
    output_prediction_json.write_text(
        json.dumps(fused_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary: dict[str, Any] = {
        "fusion_method": "unweighted_page_rrf",
        "rrf_k": float(args.rrf_k),
        "input_top_pages": int(args.input_top_pages),
        "output_top_pages": int(args.output_top_pages),
        "prediction_paths": {label: str(path) for label, path in labeled_paths},
        "qid_count": len(per_qid),
        "missing_qid_count_by_prediction": missing_by_label,
        "mean_fused_page_count": mean([float(row["fused_page_count"]) for row in per_qid]),
        "mean_top4_support_count": mean(
            [
                float(row["mean_top4_support_count"])
                for row in per_qid
                if row.get("mean_top4_support_count") is not None
            ]
        ),
        "per_qid": per_qid,
    }
    if gold_rows:
        summary["page_recall_at_k"] = {
            str(k): mean(values) for k, values in page_recall_values.items()
        }
        summary["doc_recall_at_k"] = {
            str(k): mean(values) for k, values in doc_recall_values.items()
        }
        summary["page_hit_at_4_count"] = sum(
            1
            for row in per_qid
            if row.get("first_gold_page_rank") is not None
            and int(row["first_gold_page_rank"]) <= 4
        )
        summary["doc_hit_at_4_count"] = sum(
            1
            for row in per_qid
            if row.get("first_gold_doc_rank") is not None
            and int(row["first_gold_doc_rank"]) <= 4
        )
    if movement_counts:
        summary["movement_vs_baseline_at_k"] = int(args.hit_k)
        summary["movement_vs_baseline_counts"] = dict(sorted(movement_counts.items()))

    output_summary_json = Path(args.output_summary_json)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"saved_prediction: {output_prediction_json}")
    print(f"saved_summary: {output_summary_json}")
    print(f"qid_count: {len(per_qid)}")
    if gold_rows:
        print(f"page_recall_at_k: {summary['page_recall_at_k']}")
        print(f"doc_recall_at_k: {summary['doc_recall_at_k']}")
        print(f"page_hit_at_4_count: {summary['page_hit_at_4_count']}")
        print(f"doc_hit_at_4_count: {summary['doc_hit_at_4_count']}")
    if movement_counts:
        print(f"movement_vs_baseline_counts: {dict(sorted(movement_counts.items()))}")


if __name__ == "__main__":
    main()
