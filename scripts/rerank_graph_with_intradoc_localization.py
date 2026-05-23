#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process a graph-PPR page prediction by preserving the graph document order "
            "while re-localizing pages within each document using dense/sparse/source support."
        )
    )
    parser.add_argument("--graph-prediction-json", required=True)
    parser.add_argument("--dense-prediction-json", required=True)
    parser.add_argument("--sparse-prediction-json", required=True)
    parser.add_argument("--output-prediction-json", required=True)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument(
        "--strategy",
        choices=["preserve_doc_order", "global_fusion"],
        default="preserve_doc_order",
        help=(
            "preserve_doc_order keeps graph document order and reorders pages within each doc. "
            "global_fusion sorts all candidate pages by the fused localization score."
        ),
    )
    parser.add_argument("--dense-top-pages", type=int, default=1000)
    parser.add_argument("--sparse-top-pages", type=int, default=1000)
    parser.add_argument("--graph-top-pages", type=int, default=1000)
    parser.add_argument("--output-top-pages", type=int, default=0, help="0 preserves graph output length per qid.")
    parser.add_argument("--rrf-k", type=float, default=10.0)
    parser.add_argument("--graph-page-weight", type=float, default=0.25)
    parser.add_argument("--dense-page-weight", type=float, default=1.25)
    parser.add_argument("--sparse-page-weight", type=float, default=0.75)
    parser.add_argument("--both-source-bonus", type=float, default=0.10)
    parser.add_argument("--local-window", type=int, default=1)
    parser.add_argument("--local-neighbor-weight", type=float, default=0.25)
    parser.add_argument(
        "--candidate-doc-limit",
        type=int,
        default=0,
        help="Optional limit on graph document ranks eligible for source-page expansion. 0 means all graph docs.",
    )
    return parser.parse_args()


def load_prediction(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("predictions"), (dict, list)):
        payload = payload["predictions"]

    if isinstance(payload, dict):
        items = payload.items()
    elif isinstance(payload, list):
        items = enumerate(payload)
    else:
        raise TypeError(f"Prediction JSON must be a dict or list: {path}")

    rows_by_qid: dict[str, dict[str, Any]] = {}
    for raw_key, row in items:
        if not isinstance(row, dict):
            raise TypeError(f"Prediction row must be an object: {path} key={raw_key!r}")
        qid = str(row.get("qid", "")).strip() or str(raw_key).strip()
        if not qid:
            raise ValueError(f"Prediction row is missing qid: {path} key={raw_key!r}")
        if qid in rows_by_qid:
            raise ValueError(f"Duplicate qid after normalization: {qid} ({path})")
        rows_by_qid[qid] = row
    return rows_by_qid


def page_uid(doc_id: str, page_idx: int) -> str:
    return f"{doc_id}_page{int(page_idx)}"


def parse_row(row: Any) -> tuple[str, int, float] | None:
    if isinstance(row, (list, tuple)) and len(row) >= 2:
        try:
            return str(row[0]), int(row[1]), float(row[2]) if len(row) >= 3 else 0.0
        except (TypeError, ValueError):
            return None
    if isinstance(row, dict):
        doc_id = row.get("doc_id", row.get("docid", row.get("document_id")))
        page_idx = row.get("page_idx", row.get("page_id", row.get("page")))
        if doc_id is None or page_idx is None:
            uid = row.get("page_uid")
            if isinstance(uid, str) and "_page" in uid:
                doc_id, raw_page = uid.rsplit("_page", 1)
                page_idx = raw_page
            else:
                return None
        try:
            return str(doc_id), int(page_idx), float(row.get("score", 0.0))
        except (TypeError, ValueError):
            return None
    return None


def retrieval_rows(pred_row: dict[str, Any], limit: int = 0) -> list[Any]:
    rows = pred_row.get("page_retrieval_results", [])
    if not isinstance(rows, list):
        return []
    return rows[:limit] if limit > 0 else rows


def ranked_unique_rows(pred_row: dict[str, Any], limit: int = 0) -> list[tuple[str, int, float, int]]:
    rows = []
    seen: set[str] = set()
    for rank, row in enumerate(retrieval_rows(pred_row, limit=limit), start=1):
        parsed = parse_row(row)
        if parsed is None:
            continue
        doc_id, page_idx, score = parsed
        uid = page_uid(doc_id, page_idx)
        if uid in seen:
            continue
        seen.add(uid)
        rows.append((doc_id, page_idx, float(score), rank))
    return rows


def minmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if high <= low:
        return {key: 1.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def reciprocal(rank: int | None, rrf_k: float) -> float:
    if rank is None:
        return 0.0
    return (rrf_k + 1.0) / (rrf_k + float(rank))


def doc_order(rows: list[tuple[str, int, float, int]]) -> list[str]:
    docs = []
    seen = set()
    for doc_id, _page_idx, _score, _rank in rows:
        if doc_id not in seen:
            seen.add(doc_id)
            docs.append(doc_id)
    return docs


def build_qid_ranking(
    *,
    qid: str,
    graph_row: dict[str, Any],
    dense_row: dict[str, Any],
    sparse_row: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    graph_rows = ranked_unique_rows(graph_row, int(args.graph_top_pages))
    dense_rows = ranked_unique_rows(dense_row, int(args.dense_top_pages))
    sparse_rows = ranked_unique_rows(sparse_row, int(args.sparse_top_pages))
    graph_output_len = len(retrieval_rows(graph_row))
    output_limit = int(args.output_top_pages) if int(args.output_top_pages) > 0 else graph_output_len

    graph_doc_order = doc_order(graph_rows)
    eligible_docs = set(graph_doc_order)
    if int(args.candidate_doc_limit) > 0:
        eligible_docs = set(graph_doc_order[: int(args.candidate_doc_limit)])

    graph_score_by_uid = {
        page_uid(doc_id, page_idx): float(score) for doc_id, page_idx, score, _rank in graph_rows
    }
    graph_rank_by_uid = {
        page_uid(doc_id, page_idx): int(rank) for doc_id, page_idx, _score, rank in graph_rows
    }
    graph_norm_by_uid = minmax(graph_score_by_uid)
    dense_rank_by_uid = {
        page_uid(doc_id, page_idx): int(rank) for doc_id, page_idx, _score, rank in dense_rows
    }
    sparse_rank_by_uid = {
        page_uid(doc_id, page_idx): int(rank) for doc_id, page_idx, _score, rank in sparse_rows
    }

    candidates: dict[str, dict[str, Any]] = {}
    for source_name, source_rows in [
        ("graph", graph_rows),
        ("dense", dense_rows),
        ("sparse", sparse_rows),
    ]:
        for doc_id, page_idx, _score, _rank in source_rows:
            if source_name != "graph" and doc_id not in eligible_docs:
                continue
            uid = page_uid(doc_id, page_idx)
            item = candidates.get(uid)
            if item is None:
                item = {"doc_id": doc_id, "page_idx": int(page_idx), "page_uid": uid}
                candidates[uid] = item

    source_base_by_uid: dict[str, float] = {}
    for uid in candidates:
        dense_rank = dense_rank_by_uid.get(uid)
        sparse_rank = sparse_rank_by_uid.get(uid)
        dense_component = reciprocal(dense_rank, float(args.rrf_k))
        sparse_component = reciprocal(sparse_rank, float(args.rrf_k))
        both_bonus = float(args.both_source_bonus) if dense_rank is not None and sparse_rank is not None else 0.0
        source_base_by_uid[uid] = (
            float(args.dense_page_weight) * dense_component
            + float(args.sparse_page_weight) * sparse_component
            + both_bonus
        )

    local_support_by_uid = local_neighbor_support(
        candidates=candidates,
        source_base_by_uid=source_base_by_uid,
        window=int(args.local_window),
    )

    scored_pages: list[tuple[str, int, float, dict[str, Any]]] = []
    for uid, item in candidates.items():
        graph_component = graph_norm_by_uid.get(uid, 0.0)
        source_component = source_base_by_uid.get(uid, 0.0)
        local_component = local_support_by_uid.get(uid, 0.0)
        dense_rank = dense_rank_by_uid.get(uid)
        sparse_rank = sparse_rank_by_uid.get(uid)
        score = (
            float(args.graph_page_weight) * graph_component
            + source_component
            + float(args.local_neighbor_weight) * local_component
        )
        trace = {
            "page_uid": uid,
            "doc_id": item["doc_id"],
            "page_idx": int(item["page_idx"]),
            "localization_score": float(score),
            "graph_component": float(graph_component),
            "source_component": float(source_component),
            "local_neighbor_component": float(local_component),
            "graph_rank": graph_rank_by_uid.get(uid),
            "dense_rank": dense_rank,
            "sparse_rank": sparse_rank,
        }
        scored_pages.append((item["doc_id"], int(item["page_idx"]), float(score), trace))

    if args.strategy == "global_fusion":
        final_scored = sorted(
            scored_pages,
            key=lambda row: (
                -row[2],
                row[3]["graph_rank"] if row[3]["graph_rank"] is not None else 10**9,
                row[3]["dense_rank"] if row[3]["dense_rank"] is not None else 10**9,
                row[3]["sparse_rank"] if row[3]["sparse_rank"] is not None else 10**9,
                row[0],
                row[1],
            ),
        )
    else:
        pages_by_doc: dict[str, list[tuple[str, int, float, dict[str, Any]]]] = defaultdict(list)
        for row in scored_pages:
            pages_by_doc[row[0]].append(row)
        for doc_pages in pages_by_doc.values():
            doc_pages.sort(
                key=lambda row: (
                    -row[2],
                    row[3]["graph_rank"] if row[3]["graph_rank"] is not None else 10**9,
                    row[3]["dense_rank"] if row[3]["dense_rank"] is not None else 10**9,
                    row[3]["sparse_rank"] if row[3]["sparse_rank"] is not None else 10**9,
                    row[1],
                )
            )
        final_scored = []
        used_pages: set[str] = set()
        doc_next_index: Counter[str] = Counter()
        for doc_id, _page_idx, _score, _rank in graph_rows:
            doc_pages = pages_by_doc.get(doc_id, [])
            while doc_next_index[doc_id] < len(doc_pages):
                candidate = doc_pages[doc_next_index[doc_id]]
                doc_next_index[doc_id] += 1
                uid = candidate[3]["page_uid"]
                if uid in used_pages:
                    continue
                used_pages.add(uid)
                final_scored.append(candidate)
                break
            if output_limit > 0 and len(final_scored) >= output_limit:
                break

        if output_limit <= 0 or len(final_scored) < output_limit:
            for candidate in sorted(scored_pages, key=lambda row: (-row[2], row[0], row[1])):
                uid = candidate[3]["page_uid"]
                if uid in used_pages:
                    continue
                used_pages.add(uid)
                final_scored.append(candidate)
                if output_limit > 0 and len(final_scored) >= output_limit:
                    break

    if output_limit > 0:
        final_scored = final_scored[:output_limit]

    final_rows = [[doc_id, int(page_idx), float(score)] for doc_id, page_idx, score, _trace in final_scored]
    trace_top = [trace for _doc_id, _page_idx, _score, trace in final_scored[:20]]
    moved_top4 = count_topk_page_changes(graph_rows, final_scored, top_k=4)
    metadata = {
        "qid": qid,
        "strategy": args.strategy,
        "graph_input_page_count": len(graph_rows),
        "dense_input_page_count": len(dense_rows),
        "sparse_input_page_count": len(sparse_rows),
        "candidate_page_count": len(candidates),
        "candidate_doc_count": len({item["doc_id"] for item in candidates.values()}),
        "output_page_count": len(final_rows),
        "top4_page_change_count": moved_top4,
        "top_localized_pages": trace_top,
    }
    output_row = {
        **graph_row,
        "qid": qid,
        "page_retrieval_results": final_rows,
        "reranker_metadata": {
            **graph_row.get("reranker_metadata", {}),
            "intradoc_localization": {
                key: value for key, value in metadata.items() if key != "top_localized_pages"
            },
        },
    }
    return output_row, metadata


def local_neighbor_support(
    *,
    candidates: dict[str, dict[str, Any]],
    source_base_by_uid: dict[str, float],
    window: int,
) -> dict[str, float]:
    if window <= 0:
        return {uid: 0.0 for uid in candidates}
    pages_by_doc: dict[str, set[int]] = defaultdict(set)
    for item in candidates.values():
        pages_by_doc[str(item["doc_id"])].add(int(item["page_idx"]))

    raw: dict[str, float] = {}
    for uid, item in candidates.items():
        doc_id = str(item["doc_id"])
        page_idx = int(item["page_idx"])
        support = 0.0
        for delta in range(1, window + 1):
            for neighbor_idx in (page_idx - delta, page_idx + delta):
                if neighbor_idx not in pages_by_doc[doc_id]:
                    continue
                neighbor_uid = page_uid(doc_id, neighbor_idx)
                support += source_base_by_uid.get(neighbor_uid, 0.0) / float(delta + 1)
        raw[uid] = support
    return minmax(raw)


def count_topk_page_changes(
    graph_rows: list[tuple[str, int, float, int]],
    final_rows: list[tuple[str, int, float, dict[str, Any]]],
    top_k: int,
) -> int:
    old = [page_uid(doc_id, page_idx) for doc_id, page_idx, _score, _rank in graph_rows[:top_k]]
    new = [trace["page_uid"] for _doc_id, _page_idx, _score, trace in final_rows[:top_k]]
    return sum(1 for left, right in zip(old, new) if left != right)


def main() -> None:
    args = parse_args()
    graph_pred = load_prediction(Path(args.graph_prediction_json))
    dense_pred = load_prediction(Path(args.dense_prediction_json))
    sparse_pred = load_prediction(Path(args.sparse_prediction_json))
    qids = sorted(set(graph_pred) & set(dense_pred) & set(sparse_pred))
    if not qids:
        raise ValueError("No qids overlap across graph/dense/sparse predictions.")

    output: dict[str, dict[str, Any]] = {}
    per_qid: list[dict[str, Any]] = []
    for qid in qids:
        row, metadata = build_qid_ranking(
            qid=qid,
            graph_row=graph_pred[qid],
            dense_row=dense_pred[qid],
            sparse_row=sparse_pred[qid],
            args=args,
        )
        output[qid] = row
        per_qid.append(metadata)

    output_prediction_json = Path(args.output_prediction_json)
    output_prediction_json.parent.mkdir(parents=True, exist_ok=True)
    output_prediction_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    summary = {
        "qid_count": len(qids),
        "strategy": args.strategy,
        "graph_prediction_json": args.graph_prediction_json,
        "dense_prediction_json": args.dense_prediction_json,
        "sparse_prediction_json": args.sparse_prediction_json,
        "dense_top_pages": int(args.dense_top_pages),
        "sparse_top_pages": int(args.sparse_top_pages),
        "graph_top_pages": int(args.graph_top_pages),
        "output_top_pages": int(args.output_top_pages),
        "rrf_k": float(args.rrf_k),
        "graph_page_weight": float(args.graph_page_weight),
        "dense_page_weight": float(args.dense_page_weight),
        "sparse_page_weight": float(args.sparse_page_weight),
        "both_source_bonus": float(args.both_source_bonus),
        "local_window": int(args.local_window),
        "local_neighbor_weight": float(args.local_neighbor_weight),
        "candidate_doc_limit": int(args.candidate_doc_limit),
        "mean_candidate_page_count": mean_float(row["candidate_page_count"] for row in per_qid),
        "mean_candidate_doc_count": mean_float(row["candidate_doc_count"] for row in per_qid),
        "mean_top4_page_change_count": mean_float(row["top4_page_change_count"] for row in per_qid),
        "qids_with_top4_page_change": sum(1 for row in per_qid if int(row["top4_page_change_count"]) > 0),
        "per_qid": per_qid,
    }
    output_summary_json = Path(args.output_summary_json)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"saved_prediction: {output_prediction_json}")
    print(f"saved_summary: {output_summary_json}")
    print(f"qid_count: {len(qids)}")
    print(f"mean_candidate_page_count: {summary['mean_candidate_page_count']}")
    print(f"mean_candidate_doc_count: {summary['mean_candidate_doc_count']}")
    print(f"mean_top4_page_change_count: {summary['mean_top4_page_change_count']}")
    print(f"qids_with_top4_page_change: {summary['qids_with_top4_page_change']}")


def mean_float(values: Any) -> float:
    materialized = [float(value) for value in values]
    return float(statistics.fmean(materialized)) if materialized else 0.0


if __name__ == "__main__":
    main()
