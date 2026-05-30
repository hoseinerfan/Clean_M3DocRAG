#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Expand a page-retrieval prediction JSON with target pages from an external "
            "query-specific page graph. This is intended for FAISS token-neighbor pages "
            "used as candidate expansion before a normal reranker/PPR pass, not as graph edges."
        )
    )
    parser.add_argument("--prediction-json", required=True)
    parser.add_argument("--external-page-graph-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--max-new-pages-per-qid", type=int, default=50)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument(
        "--aggregation",
        choices=["max", "sum", "log_count"],
        default="log_count",
    )
    parser.add_argument(
        "--synthetic-score-mode",
        choices=["below_min", "normalized"],
        default="below_min",
    )
    parser.add_argument(
        "--append-after-top-k",
        type=int,
        default=0,
        help="Insert new pages after this many original pages. Use 0 to append after all original pages.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_prediction_payload(path: Path) -> tuple[Any, dict[str, dict[str, Any]]]:
    payload = read_json(path)
    rows_root = payload["predictions"] if isinstance(payload, dict) and "predictions" in payload else payload
    rows_by_qid: dict[str, dict[str, Any]] = {}
    if isinstance(rows_root, dict):
        iterator = rows_root.items()
    elif isinstance(rows_root, list):
        iterator = enumerate(rows_root)
    else:
        raise TypeError(f"Unsupported prediction JSON root: {path}")
    for raw_key, row in iterator:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("qid") or raw_key).strip()
        if qid:
            rows_by_qid[qid] = dict(row)
    return payload, rows_by_qid


def page_uid(doc_id: str, page_idx: int) -> str:
    return f"{doc_id}_page{int(page_idx)}"


def parse_page_uid(uid: str) -> tuple[str, int] | None:
    if "_page" not in uid:
        return None
    doc_id, raw_page_idx = uid.rsplit("_page", 1)
    if not doc_id:
        return None
    try:
        return doc_id, int(raw_page_idx)
    except ValueError:
        return None


def row_page_uid(raw: object) -> str | None:
    if isinstance(raw, dict):
        uid = str(raw.get("page_uid", "")).strip()
        if uid:
            return uid
        doc_id = str(raw.get("doc_id", raw.get("docid", ""))).strip()
        page_idx = raw.get("page_idx", raw.get("page_index", raw.get("page", None)))
    elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
        doc_id = str(raw[0]).strip()
        page_idx = raw[1]
    else:
        return None
    if not doc_id:
        return None
    try:
        return page_uid(doc_id, int(page_idx))
    except (TypeError, ValueError):
        return None


def row_score(raw: object) -> float:
    if isinstance(raw, dict):
        value = raw.get("score", raw.get("retrieval_score", 0.0))
    elif isinstance(raw, (list, tuple)) and len(raw) >= 3:
        value = raw[2]
    else:
        value = 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def load_external_targets(
    path: Path,
    *,
    min_score: float,
    aggregation: str,
) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("qid", "")).strip()
            target_uid = str(row.get("target_page_uid", "")).strip()
            if not qid or not target_uid:
                continue
            try:
                score = float(row.get("score", row.get("raw_score", 0.0)))
            except (TypeError, ValueError):
                score = 0.0
            if score < min_score:
                continue
            values[qid][target_uid].append(score)

    aggregated: dict[str, dict[str, float]] = {}
    for qid, page_values in values.items():
        aggregated[qid] = {}
        for uid, scores in page_values.items():
            if aggregation == "max":
                score = max(scores)
            elif aggregation == "sum":
                score = sum(scores)
            else:
                score = max(scores) * math.log1p(len(scores))
            aggregated[qid][uid] = float(score)
    return aggregated


def expanded_rows(
    rows: list[Any],
    target_scores: dict[str, float],
    *,
    max_new_pages: int,
    synthetic_score_mode: str,
    append_after_top_k: int,
) -> tuple[list[Any], int, int]:
    existing_uids = {uid for raw in rows if (uid := row_page_uid(raw))}
    candidates = [
        (uid, score)
        for uid, score in target_scores.items()
        if uid not in existing_uids and parse_page_uid(uid) is not None
    ]
    candidates.sort(key=lambda item: (-item[1], item[0]))
    if max_new_pages > 0:
        candidates = candidates[:max_new_pages]

    original_scores = [row_score(raw) for raw in rows]
    min_original_score = min(original_scores) if original_scores else 0.0
    max_candidate_score = max((score for _uid, score in candidates), default=0.0)
    new_rows: list[list[object]] = []
    for idx, (uid, score) in enumerate(candidates, start=1):
        parsed = parse_page_uid(uid)
        if parsed is None:
            continue
        doc_id, page_idx = parsed
        if synthetic_score_mode == "normalized" and max_candidate_score > 0:
            synthetic_score = float(score / max_candidate_score)
        else:
            synthetic_score = float(min_original_score - 1e-6 * idx)
        new_rows.append([doc_id, int(page_idx), synthetic_score])

    if append_after_top_k > 0:
        prefix = rows[:append_after_top_k]
        suffix = rows[append_after_top_k:]
        return [*prefix, *new_rows, *suffix], len(new_rows), len(candidates)
    return [*rows, *new_rows], len(new_rows), len(candidates)


def main() -> None:
    args = parse_args()
    payload, rows_by_qid = load_prediction_payload(Path(args.prediction_json))
    target_scores_by_qid = load_external_targets(
        Path(args.external_page_graph_jsonl),
        min_score=float(args.min_score),
        aggregation=str(args.aggregation),
    )

    output_rows: dict[str, dict[str, Any]] = {}
    added_counts: list[int] = []
    for qid, row in rows_by_qid.items():
        copied = dict(row)
        page_rows = list(copied.get("page_retrieval_results", []))
        expanded, added_count, candidate_count = expanded_rows(
            page_rows,
            target_scores_by_qid.get(qid, {}),
            max_new_pages=int(args.max_new_pages_per_qid),
            synthetic_score_mode=str(args.synthetic_score_mode),
            append_after_top_k=int(args.append_after_top_k),
        )
        copied["page_retrieval_results"] = expanded
        copied["faiss_token_neighbor_candidate_expansion"] = {
            "source_external_page_graph_jsonl": str(args.external_page_graph_jsonl),
            "max_new_pages_per_qid": int(args.max_new_pages_per_qid),
            "min_score": float(args.min_score),
            "aggregation": str(args.aggregation),
            "synthetic_score_mode": str(args.synthetic_score_mode),
            "append_after_top_k": int(args.append_after_top_k),
            "candidate_target_page_count": int(candidate_count),
            "added_page_count": int(added_count),
        }
        output_rows[qid] = copied
        added_counts.append(added_count)

    output_payload: Any
    if isinstance(payload, dict) and "predictions" in payload:
        output_payload = dict(payload)
        output_payload["predictions"] = output_rows
    else:
        output_payload = output_rows

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, indent=2) + "\n", encoding="utf-8")

    summary = {
        "prediction_json": str(args.prediction_json),
        "external_page_graph_jsonl": str(args.external_page_graph_jsonl),
        "output_json": str(output_path),
        "qid_count": len(output_rows),
        "external_qid_count": len(target_scores_by_qid),
        "max_new_pages_per_qid": int(args.max_new_pages_per_qid),
        "total_added_page_count": int(sum(added_counts)),
        "mean_added_page_count": (
            float(sum(added_counts) / len(added_counts)) if added_counts else 0.0
        ),
        "qid_with_added_page_count": int(sum(1 for count in added_counts if count > 0)),
    }
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"saved_expanded_prediction={output_path}")
    if args.summary_json:
        print(f"saved_summary={args.summary_json}")
    print(f"qid_count={summary['qid_count']}")
    print(f"total_added_page_count={summary['total_added_page_count']}")
    print(f"mean_added_page_count={summary['mean_added_page_count']:.3f}")


if __name__ == "__main__":
    main()
