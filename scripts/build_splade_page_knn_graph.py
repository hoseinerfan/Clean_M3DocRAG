#!/usr/bin/env python3

from __future__ import annotations

import argparse
import heapq
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build query-independent page-page kNN graph edges from a SPLADE page index. "
            "The output JSONL is compatible with graph_rerank_page_retrieval_predictions.py "
            "--external-page-graph-jsonl."
        )
    )
    parser.add_argument("--splade-index-pt", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument(
        "--source-prediction-json",
        action="append",
        default=[],
        help=(
            "Prediction JSON whose top pages define graph source pages. Repeat to use "
            "dense and sparse predictions. If omitted, all indexed pages are sources."
        ),
    )
    parser.add_argument(
        "--qid-filter-jsonl",
        action="append",
        default=[],
        help=(
            "Optional JSONL with qid fields used to restrict qids read from source predictions. "
            "Use this with subset gold files to avoid building source edges for the full dev set."
        ),
    )
    parser.add_argument(
        "--qid",
        action="append",
        default=[],
        help="Optional explicit qid to include from source predictions. Repeatable.",
    )
    parser.add_argument(
        "--source-top-pages",
        type=int,
        default=1000,
        help="Per-qid top pages to collect from each source prediction JSON.",
    )
    parser.add_argument(
        "--source-page-uid-jsonl",
        action="append",
        default=[],
        help="Optional JSONL with page_uid or doc_id/page_idx rows to add as source pages.",
    )
    parser.add_argument(
        "--source-page-uid",
        action="append",
        default=[],
        help="Optional explicit source page uid. Repeatable.",
    )
    parser.add_argument(
        "--max-source-pages",
        type=int,
        default=0,
        help="Optional cap after sorting source page uids. Use 0 for no cap.",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--source-topk-terms",
        type=int,
        default=64,
        help="Use at most this many highest-weight SPLADE terms from each source page. Use 0 for all.",
    )
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--score-mode", choices=["cosine", "dot"], default="cosine")
    parser.add_argument("--same-doc-only", action="store_true")
    parser.add_argument("--cross-doc-only", action="store_true")
    parser.add_argument(
        "--bidirectional-dedup",
        action="store_true",
        help="Emit only one edge for each unordered pair. PPR can add reverse edges later.",
    )
    return parser.parse_args()


def page_uid(doc_id: str, page_idx: int) -> str:
    return f"{doc_id}_page{int(page_idx)}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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
        payload["predictions"],
        (dict, list),
    ):
        payload = payload["predictions"]

    rows_by_qid: dict[str, dict[str, Any]] = {}
    iterable: Any
    if isinstance(payload, list):
        iterable = enumerate(payload)
    elif isinstance(payload, dict):
        iterable = payload.items()
    else:
        raise TypeError(f"Prediction JSON must be a list or object: {path}")

    for raw_key, row in iterable:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("qid", "")).strip() or str(raw_key).strip()
        if qid:
            rows_by_qid[qid] = row
    return rows_by_qid


def load_qid_filter(args: argparse.Namespace) -> set[str]:
    qids = {str(qid).strip() for qid in args.qid if str(qid).strip()}
    for raw_path in args.qid_filter_jsonl:
        for row in read_jsonl(Path(raw_path)):
            qid = str(row.get("qid", "")).strip()
            if qid:
                qids.add(qid)
    return qids


def collect_prediction_source_pages(
    path: Path,
    *,
    source_top_pages: int,
    qid_filter: set[str],
) -> set[str]:
    source_pages: set[str] = set()
    prediction = load_prediction(path)
    limit = max(0, int(source_top_pages))
    qids = sorted(qid_filter) if qid_filter else sorted(prediction)
    for qid in qids:
        row = prediction.get(qid)
        if row is None:
            continue
        seen_for_qid: set[str] = set()
        for raw in row.get("page_retrieval_results", []):
            if not isinstance(raw, list) or len(raw) < 2:
                continue
            try:
                uid = page_uid(str(raw[0]), int(raw[1]))
            except (TypeError, ValueError):
                continue
            if uid in seen_for_qid:
                continue
            seen_for_qid.add(uid)
            source_pages.add(uid)
            if limit > 0 and len(seen_for_qid) >= limit:
                break
    return source_pages


def collect_source_pages(args: argparse.Namespace) -> set[str]:
    source_pages = {str(uid).strip() for uid in args.source_page_uid if str(uid).strip()}
    qid_filter = load_qid_filter(args)
    for raw_path in args.source_prediction_json:
        source_pages |= collect_prediction_source_pages(
            Path(raw_path),
            source_top_pages=int(args.source_top_pages),
            qid_filter=qid_filter,
        )
    for raw_path in args.source_page_uid_jsonl:
        for row in read_jsonl(Path(raw_path)):
            uid = str(row.get("page_uid", "")).strip()
            if not uid:
                doc_id = str(row.get("doc_id", "")).strip()
                page_idx = row.get("page_idx", row.get("page_id"))
                if doc_id and page_idx is not None:
                    try:
                        uid = page_uid(doc_id, int(page_idx))
                    except (TypeError, ValueError):
                        uid = ""
            if uid:
                source_pages.add(uid)
    return source_pages


def qid_filter_count(args: argparse.Namespace) -> int:
    return len(load_qid_filter(args))


def load_splade_index(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu")
    required = ["page_uids", "doc_ids", "page_indices", "offsets", "term_ids", "term_weights"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"SPLADE index is missing required keys {missing}: {path}")
    page_uids = [str(value) for value in payload["page_uids"]]
    doc_ids = [str(value) for value in payload["doc_ids"]]
    page_indices = [int(value) for value in payload["page_indices"].tolist()]
    offsets = [int(value) for value in payload["offsets"].tolist()]
    term_ids = [int(value) for value in payload["term_ids"].tolist()]
    term_weights = [float(value) for value in payload["term_weights"].tolist()]
    if len(offsets) != len(page_uids) + 1:
        raise ValueError("SPLADE index offsets length must equal page count + 1.")
    return {
        "page_uids": page_uids,
        "doc_ids": doc_ids,
        "page_indices": page_indices,
        "offsets": offsets,
        "term_ids": term_ids,
        "term_weights": term_weights,
    }


def page_terms(index: dict[str, Any], page_idx: int, topk_terms: int) -> list[tuple[int, float]]:
    offsets = index["offsets"]
    term_ids = index["term_ids"]
    term_weights = index["term_weights"]
    start = int(offsets[page_idx])
    end = int(offsets[page_idx + 1])
    pairs = [(int(term_id), float(weight)) for term_id, weight in zip(term_ids[start:end], term_weights[start:end])]
    pairs = [(term_id, weight) for term_id, weight in pairs if weight > 0]
    pairs.sort(key=lambda item: (-item[1], item[0]))
    if topk_terms > 0:
        pairs = pairs[:topk_terms]
    return pairs


def build_postings(index: dict[str, Any]) -> dict[int, list[tuple[int, float]]]:
    postings: dict[int, list[tuple[int, float]]] = defaultdict(list)
    page_count = len(index["page_uids"])
    for page_idx in range(page_count):
        start = int(index["offsets"][page_idx])
        end = int(index["offsets"][page_idx + 1])
        for term_id, weight in zip(index["term_ids"][start:end], index["term_weights"][start:end]):
            weight = float(weight)
            if weight > 0:
                postings[int(term_id)].append((page_idx, weight))
    return dict(postings)


def page_l2_norms(index: dict[str, Any]) -> list[float]:
    norms: list[float] = []
    page_count = len(index["page_uids"])
    for page_idx in range(page_count):
        start = int(index["offsets"][page_idx])
        end = int(index["offsets"][page_idx + 1])
        total = sum(float(weight) * float(weight) for weight in index["term_weights"][start:end])
        norms.append(math.sqrt(total))
    return norms


def top_neighbors_for_source(
    *,
    source_idx: int,
    index: dict[str, Any],
    postings: dict[int, list[tuple[int, float]]],
    norms: list[float],
    top_k: int,
    source_topk_terms: int,
    min_score: float,
    score_mode: str,
    same_doc_only: bool,
    cross_doc_only: bool,
) -> list[tuple[int, float]]:
    source_terms = page_terms(index, source_idx, source_topk_terms)
    if not source_terms:
        return []
    source_doc = index["doc_ids"][source_idx]
    scores: dict[int, float] = defaultdict(float)
    for term_id, source_weight in source_terms:
        for target_idx, target_weight in postings.get(int(term_id), []):
            if target_idx == source_idx:
                continue
            scores[target_idx] += float(source_weight) * float(target_weight)

    source_norm = norms[source_idx]
    candidates: list[tuple[int, float]] = []
    for target_idx, score in scores.items():
        target_doc = index["doc_ids"][target_idx]
        if same_doc_only and source_doc != target_doc:
            continue
        if cross_doc_only and source_doc == target_doc:
            continue
        if score_mode == "cosine":
            denom = source_norm * norms[target_idx]
            if denom <= 0:
                continue
            score = score / denom
        if score < min_score:
            continue
        candidates.append((target_idx, float(score)))
    return heapq.nsmallest(
        max(0, top_k),
        candidates,
        key=lambda item: (-item[1], index["page_uids"][item[0]]),
    )


def main() -> None:
    args = parse_args()
    if bool(args.same_doc_only) and bool(args.cross_doc_only):
        raise ValueError("--same-doc-only and --cross-doc-only are mutually exclusive.")
    top_k = int(args.top_k)
    if top_k <= 0:
        raise ValueError("--top-k must be positive.")

    index = load_splade_index(Path(args.splade_index_pt))
    page_uids = index["page_uids"]
    page_uid_to_idx = {uid: idx for idx, uid in enumerate(page_uids)}
    source_pages = collect_source_pages(args)
    if source_pages:
        missing_source_pages = sorted(uid for uid in source_pages if uid not in page_uid_to_idx)
        source_indices = sorted(page_uid_to_idx[uid] for uid in source_pages if uid in page_uid_to_idx)
    else:
        missing_source_pages = []
        source_indices = list(range(len(page_uids)))
    if int(args.max_source_pages) > 0:
        source_indices = source_indices[: int(args.max_source_pages)]
    if not source_indices:
        raise ValueError("No valid source pages remain after filtering.")

    postings = build_postings(index)
    norms = page_l2_norms(index)
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    edge_count = 0
    source_page_count = 0
    target_pages: set[str] = set()
    target_docs: set[str] = set()
    emitted_pair_keys: set[tuple[str, str]] = set()
    score_values: list[float] = []

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for source_idx in source_indices:
            source_uid = page_uids[source_idx]
            neighbors = top_neighbors_for_source(
                source_idx=source_idx,
                index=index,
                postings=postings,
                norms=norms,
                top_k=top_k,
                source_topk_terms=int(args.source_topk_terms),
                min_score=float(args.min_score),
                score_mode=str(args.score_mode),
                same_doc_only=bool(args.same_doc_only),
                cross_doc_only=bool(args.cross_doc_only),
            )
            emitted_for_source = 0
            for target_idx, score in neighbors:
                target_uid = page_uids[target_idx]
                if bool(args.bidirectional_dedup):
                    pair_key = tuple(sorted((source_uid, target_uid)))
                    if pair_key in emitted_pair_keys:
                        continue
                    emitted_pair_keys.add(pair_key)
                row = {
                    "edge_type": "splade_page_knn",
                    "source_page_uid": source_uid,
                    "target_page_uid": target_uid,
                    "source_doc_id": index["doc_ids"][source_idx],
                    "source_page_idx": int(index["page_indices"][source_idx]),
                    "target_doc_id": index["doc_ids"][target_idx],
                    "target_page_idx": int(index["page_indices"][target_idx]),
                    "score": float(score),
                    "weight": float(score),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                edge_count += 1
                emitted_for_source += 1
                target_pages.add(target_uid)
                target_docs.add(str(index["doc_ids"][target_idx]))
                score_values.append(float(score))
            if emitted_for_source > 0:
                source_page_count += 1

    summary = {
        "splade_index_pt": args.splade_index_pt,
        "page_count": len(page_uids),
        "source_prediction_jsons": list(args.source_prediction_json),
        "qid_filter_count": qid_filter_count(args),
        "requested_source_page_count": len(source_pages) if source_pages else len(page_uids),
        "missing_source_page_count": len(missing_source_pages),
        "source_page_count": source_page_count,
        "target_page_count": len(target_pages),
        "target_doc_count": len(target_docs),
        "edge_count": edge_count,
        "top_k": top_k,
        "source_top_pages": int(args.source_top_pages),
        "source_topk_terms": int(args.source_topk_terms),
        "min_score": float(args.min_score),
        "score_mode": str(args.score_mode),
        "same_doc_only": bool(args.same_doc_only),
        "cross_doc_only": bool(args.cross_doc_only),
        "bidirectional_dedup": bool(args.bidirectional_dedup),
        "score_min": min(score_values) if score_values else None,
        "score_max": max(score_values) if score_values else None,
        "score_mean": (sum(score_values) / len(score_values)) if score_values else None,
        "missing_source_page_sample": missing_source_pages[:20],
    }
    output_summary_json = Path(args.output_summary_json)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"saved_edges: {output_jsonl}")
    print(f"saved_summary: {output_summary_json}")
    for key in [
        "page_count",
        "requested_source_page_count",
        "missing_source_page_count",
        "source_page_count",
        "target_page_count",
        "target_doc_count",
        "edge_count",
        "score_mean",
    ]:
        print(f"{key}: {summary[key]}")


if __name__ == "__main__":
    main()
