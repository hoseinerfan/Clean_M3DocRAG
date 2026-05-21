#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class PageRecord:
    doc_id: str
    page_idx: int
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None

    @property
    def page_uid(self) -> str:
        return page_uid(self.doc_id, self.page_idx)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Graph/PPR rerank over dense and SPLADE page-retrieval candidates. "
            "The output uses the same prediction JSON schema as the existing retrieval helpers."
        )
    )
    parser.add_argument("--dense-prediction-json", required=True)
    parser.add_argument("--sparse-prediction-json", required=True)
    parser.add_argument("--gold", help="Optional MMQA-style JSONL for summary metrics.")
    parser.add_argument(
        "--question-type",
        default="",
        help="Optional metadata.type filter applied when --gold is provided, e.g. ImageListQ.",
    )
    parser.add_argument("--dense-top-pages", type=int, default=1000)
    parser.add_argument("--sparse-top-pages", type=int, default=1000)
    parser.add_argument(
        "--final-top-pages",
        type=int,
        default=1000,
        help="Number of final page rows to write. Use 0 to write every graph candidate page.",
    )
    parser.add_argument(
        "--per-doc-page-limit",
        type=int,
        default=0,
        help=(
            "Optional cap on final page rows per document. Use 1 for doc-shortlist style output; "
            "use 0 for no cap."
        ),
    )
    parser.add_argument("--rrf-k", type=float, default=10.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--sparse-weight", type=float, default=1.0)
    parser.add_argument(
        "--score-seed-weight",
        type=float,
        default=0.0,
        help=(
            "Optional within-source min-max score contribution added to rank/RRF page seeds. "
            "Default 0 keeps the graph seed rank-based and comparable across dense/SPLADE."
        ),
    )
    parser.add_argument(
        "--doc-seed-weight",
        type=float,
        default=1.0,
        help="Weight for doc-node restart mass from dense/SPLADE doc RRF. Default: 1.",
    )
    parser.add_argument("--restart-prob", type=float, default=0.20)
    parser.add_argument("--ppr-iters", type=int, default=30)
    parser.add_argument("--page-doc-edge-weight", type=float, default=1.0)
    parser.add_argument("--adjacent-page-edge-weight", type=float, default=0.25)
    parser.add_argument(
        "--same-doc-window",
        type=int,
        default=1,
        help="Connect candidate pages from the same doc when their page indices differ by <= this value.",
    )
    parser.add_argument(
        "--final-page-seed-weight",
        type=float,
        default=1.0,
        help="Final normalized source page-seed weight.",
    )
    parser.add_argument(
        "--final-ppr-page-weight",
        type=float,
        default=1.0,
        help="Final normalized page-PPR weight.",
    )
    parser.add_argument(
        "--final-ppr-doc-weight",
        type=float,
        default=0.5,
        help="Final normalized owning-doc PPR weight for each page.",
    )
    parser.add_argument("--output-prediction-json", required=True)
    parser.add_argument("--output-summary-json", required=True)
    return parser.parse_args()


def load_prediction(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Prediction JSON must be an object keyed by qid: {path}")
    return {str(qid): row for qid, row in payload.items()}


def load_gold_rows(path: Path, question_type: str = "") -> dict[str, dict]:
    rows: dict[str, dict] = {}
    wanted_type = str(question_type).strip()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("qid", "")).strip()
            if not qid:
                continue
            row_type = str(row.get("metadata", {}).get("type", "")).strip()
            if wanted_type and row_type != wanted_type:
                continue
            rows[qid] = row
    return rows


def page_uid(doc_id: str, page_idx: int) -> str:
    return f"{doc_id}_page{int(page_idx)}"


def parse_page_row(row: list[object]) -> tuple[str, int, float] | None:
    if not isinstance(row, list) or len(row) < 2:
        return None
    doc_id = str(row[0]).strip()
    if not doc_id:
        return None
    try:
        page_idx = int(row[1])
    except (TypeError, ValueError):
        return None
    score = 0.0
    if len(row) >= 3:
        try:
            score = float(row[2])
        except (TypeError, ValueError):
            score = 0.0
    return doc_id, page_idx, score


def ranked_unique_pages(rows: list[list[object]], top_pages: int) -> list[tuple[str, int, float, int]]:
    ranked: list[tuple[str, int, float, int]] = []
    seen: set[str] = set()
    for row in rows:
        parsed = parse_page_row(row)
        if parsed is None:
            continue
        doc_id, page_idx, score = parsed
        uid = page_uid(doc_id, page_idx)
        if uid in seen:
            continue
        seen.add(uid)
        ranked.append((doc_id, page_idx, score, len(ranked) + 1))
        if top_pages > 0 and len(ranked) >= top_pages:
            break
    return ranked


def first_doc_rank_map(page_rows: Iterable[tuple[str, int, float, int]]) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for doc_id, _page_idx, _score, _page_rank in page_rows:
        if doc_id not in ranks:
            ranks[doc_id] = len(ranks) + 1
    return ranks


def minmax_by_uid(rows: list[tuple[str, int, float, int]]) -> dict[str, float]:
    if not rows:
        return {}
    scores = [score for _doc_id, _page_idx, score, _rank in rows]
    lo = min(scores)
    hi = max(scores)
    if hi <= lo:
        return {page_uid(doc_id, page_idx): 1.0 for doc_id, page_idx, _score, _rank in rows}
    return {
        page_uid(doc_id, page_idx): (float(score) - lo) / (hi - lo)
        for doc_id, page_idx, score, _rank in rows
    }


def add_undirected_edge(
    graph: dict[str, dict[str, float]],
    left: str,
    right: str,
    weight: float,
) -> None:
    if weight <= 0 or left == right:
        return
    graph[left][right] = graph[left].get(right, 0.0) + float(weight)
    graph[right][left] = graph[right].get(left, 0.0) + float(weight)


def normalize_nonnegative(values: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, value) for value in values.values())
    if total <= 0:
        if not values:
            return {}
        uniform = 1.0 / len(values)
        return {key: uniform for key in values}
    return {key: max(0.0, value) / total for key, value in values.items()}


def max_scale(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    max_value = max(values.values())
    if max_value <= 0:
        return {key: 0.0 for key in values}
    return {key: float(value) / max_value for key, value in values.items()}


def run_ppr(
    *,
    graph: dict[str, dict[str, float]],
    seed: dict[str, float],
    restart_prob: float,
    iters: int,
) -> dict[str, float]:
    nodes = sorted(set(graph) | set(seed))
    if not nodes:
        return {}
    normalized_seed = normalize_nonnegative({node: seed.get(node, 0.0) for node in nodes})
    rank = dict(normalized_seed)
    outgoing_totals = {
        node: sum(max(0.0, weight) for weight in graph.get(node, {}).values())
        for node in nodes
    }

    for _ in range(max(0, int(iters))):
        next_rank = {node: float(restart_prob) * normalized_seed[node] for node in nodes}
        dangling_mass = 0.0
        for src in nodes:
            src_rank = rank.get(src, 0.0)
            total = outgoing_totals.get(src, 0.0)
            if total <= 0:
                dangling_mass += src_rank
                continue
            share = (1.0 - float(restart_prob)) * src_rank / total
            for dst, weight in graph.get(src, {}).items():
                if weight > 0:
                    next_rank[dst] = next_rank.get(dst, 0.0) + share * float(weight)
        if dangling_mass > 0:
            for node in nodes:
                next_rank[node] += (1.0 - float(restart_prob)) * dangling_mass * normalized_seed[node]
        rank = next_rank
    return rank


def gold_doc_ids(row: dict) -> set[str]:
    return {
        str(item.get("doc_id", "")).strip()
        for item in row.get("supporting_context", [])
        if str(item.get("doc_id", "")).strip()
    }


def gold_page_uids(row: dict) -> set[str]:
    metadata = row.get("metadata", {})
    uids = {
        str(value).strip()
        for value in metadata.get("gold_page_uids", [])
        if str(value).strip()
    }
    for item in row.get("supporting_context", []):
        doc_id = str(item.get("doc_id", "")).strip()
        page_idx = item.get("page_idx", item.get("page_id"))
        if doc_id and page_idx is not None:
            uids.add(page_uid(doc_id, int(page_idx)))
    return uids


def first_rank(items: Iterable[str], gold: set[str]) -> int | None:
    for idx, item in enumerate(items, start=1):
        if item in gold:
            return idx
    return None


def ranked_docs_from_rows(rows: list[list[object]]) -> list[str]:
    docs: list[str] = []
    seen: set[str] = set()
    for row in rows:
        parsed = parse_page_row(row)
        if parsed is None:
            continue
        doc_id, _page_idx, _score = parsed
        if doc_id not in seen:
            seen.add(doc_id)
            docs.append(doc_id)
    return docs


def ranked_pages_from_rows(rows: list[list[object]]) -> list[str]:
    pages: list[str] = []
    for row in rows:
        parsed = parse_page_row(row)
        if parsed is None:
            continue
        doc_id, page_idx, _score = parsed
        pages.append(page_uid(doc_id, page_idx))
    return pages


def page_rows_from_ranked_unique_pages(
    rows: list[list[object]],
    top_pages: int,
) -> list[list[object]]:
    return [
        [doc_id, int(page_idx), float(score)]
        for doc_id, page_idx, score, _rank in ranked_unique_pages(rows, top_pages)
    ]


def median_or_none(values: list[int | None]) -> float | None:
    filtered = sorted(int(value) for value in values if value is not None)
    if not filtered:
        return None
    return float(statistics.median(filtered))


def build_qid_graph_ranking(
    *,
    qid: str,
    dense_row: dict,
    sparse_row: dict,
    args: argparse.Namespace,
) -> tuple[list[list[object]], dict]:
    dense_pages = ranked_unique_pages(
        dense_row.get("page_retrieval_results", []),
        int(args.dense_top_pages),
    )
    sparse_pages = ranked_unique_pages(
        sparse_row.get("page_retrieval_results", []),
        int(args.sparse_top_pages),
    )
    dense_score_norm = minmax_by_uid(dense_pages)
    sparse_score_norm = minmax_by_uid(sparse_pages)
    dense_doc_ranks = first_doc_rank_map(dense_pages)
    sparse_doc_ranks = first_doc_rank_map(sparse_pages)

    records: dict[str, PageRecord] = {}
    page_seed: dict[str, float] = defaultdict(float)
    doc_seed: dict[str, float] = defaultdict(float)

    for source_name, source_weight, source_pages, source_score_norm in [
        ("dense", float(args.dense_weight), dense_pages, dense_score_norm),
        ("sparse", float(args.sparse_weight), sparse_pages, sparse_score_norm),
    ]:
        for doc_id, page_idx, score, rank in source_pages:
            uid = page_uid(doc_id, page_idx)
            record = records.get(uid)
            if record is None:
                record = PageRecord(doc_id=doc_id, page_idx=page_idx)
                records[uid] = record
            if source_name == "dense":
                record.dense_rank = rank
                record.dense_score = score
            else:
                record.sparse_rank = rank
                record.sparse_score = score
            page_seed[uid] += source_weight / (float(args.rrf_k) + float(rank))
            page_seed[uid] += source_weight * float(args.score_seed_weight) * source_score_norm.get(uid, 0.0)

    for doc_id, rank in dense_doc_ranks.items():
        doc_seed[f"doc::{doc_id}"] += (
            float(args.doc_seed_weight) * float(args.dense_weight) / (float(args.rrf_k) + float(rank))
        )
    for doc_id, rank in sparse_doc_ranks.items():
        doc_seed[f"doc::{doc_id}"] += (
            float(args.doc_seed_weight) * float(args.sparse_weight) / (float(args.rrf_k) + float(rank))
        )

    graph: dict[str, dict[str, float]] = defaultdict(dict)
    for uid, record in records.items():
        doc_node = f"doc::{record.doc_id}"
        graph.setdefault(uid, {})
        graph.setdefault(doc_node, {})
        add_undirected_edge(graph, uid, doc_node, float(args.page_doc_edge_weight))

    pages_by_doc: dict[str, list[PageRecord]] = defaultdict(list)
    for record in records.values():
        pages_by_doc[record.doc_id].append(record)
    same_doc_window = int(args.same_doc_window)
    if same_doc_window > 0 and float(args.adjacent_page_edge_weight) > 0:
        for doc_records in pages_by_doc.values():
            doc_records.sort(key=lambda item: item.page_idx)
            for left_idx, left in enumerate(doc_records):
                for right in doc_records[left_idx + 1 :]:
                    if right.page_idx - left.page_idx > same_doc_window:
                        break
                    add_undirected_edge(
                        graph,
                        left.page_uid,
                        right.page_uid,
                        float(args.adjacent_page_edge_weight),
                    )

    seed = dict(page_seed)
    for node, value in doc_seed.items():
        seed[node] = seed.get(node, 0.0) + value
    ppr = run_ppr(
        graph=graph,
        seed=seed,
        restart_prob=float(args.restart_prob),
        iters=int(args.ppr_iters),
    )

    page_seed_scaled = max_scale({uid: float(score) for uid, score in page_seed.items()})
    page_ppr_scaled = max_scale({uid: ppr.get(uid, 0.0) for uid in records})
    doc_ppr_scaled = max_scale({node: ppr.get(node, 0.0) for node in doc_seed})

    ranked_records: list[tuple[PageRecord, float, float, float, float]] = []
    for uid, record in records.items():
        doc_node = f"doc::{record.doc_id}"
        seed_component = page_seed_scaled.get(uid, 0.0)
        page_ppr_component = page_ppr_scaled.get(uid, 0.0)
        doc_ppr_component = doc_ppr_scaled.get(doc_node, 0.0)
        final_score = (
            float(args.final_page_seed_weight) * seed_component
            + float(args.final_ppr_page_weight) * page_ppr_component
            + float(args.final_ppr_doc_weight) * doc_ppr_component
        )
        ranked_records.append(
            (record, float(final_score), seed_component, page_ppr_component, doc_ppr_component)
        )

    ranked_records.sort(
        key=lambda item: (
            -item[1],
            min(item[0].dense_rank or 10**9, item[0].sparse_rank or 10**9),
            item[0].dense_rank or 10**9,
            item[0].sparse_rank or 10**9,
            item[0].doc_id,
            item[0].page_idx,
        )
    )

    final_rows: list[list[object]] = []
    per_doc_counts: dict[str, int] = defaultdict(int)
    for record, final_score, _seed_component, _page_ppr_component, _doc_ppr_component in ranked_records:
        if int(args.per_doc_page_limit) > 0 and per_doc_counts[record.doc_id] >= int(args.per_doc_page_limit):
            continue
        per_doc_counts[record.doc_id] += 1
        final_rows.append([record.doc_id, int(record.page_idx), float(final_score)])
        if int(args.final_top_pages) > 0 and len(final_rows) >= int(args.final_top_pages):
            break

    trace_top = [
        {
            "page_uid": record.page_uid,
            "doc_id": record.doc_id,
            "page_idx": int(record.page_idx),
            "final_score": final_score,
            "page_seed_norm": seed_component,
            "page_ppr_norm": page_ppr_component,
            "doc_ppr_norm": doc_ppr_component,
            "dense_rank": record.dense_rank,
            "sparse_rank": record.sparse_rank,
        }
        for record, final_score, seed_component, page_ppr_component, doc_ppr_component in ranked_records[:20]
    ]
    metadata = {
        "qid": qid,
        "candidate_page_count": len(records),
        "candidate_doc_count": len(pages_by_doc),
        "dense_candidate_page_count": len(dense_pages),
        "sparse_candidate_page_count": len(sparse_pages),
        "output_page_count": len(final_rows),
        "graph_node_count": len(graph),
        "graph_edge_count_undirected": sum(len(neighbors) for neighbors in graph.values()) // 2,
        "top_graph_pages": trace_top,
    }
    return final_rows, metadata


def summarize_prediction_rows(rows: list[list[object]], gold_row: dict | None) -> dict:
    ranked_docs = ranked_docs_from_rows(rows)
    ranked_pages = ranked_pages_from_rows(rows)
    summary: dict[str, object] = {
        "doc_count": len(ranked_docs),
        "page_count": len(ranked_pages),
        "top_doc_ids": ranked_docs[:20],
    }
    if gold_row is not None:
        doc_gold = gold_doc_ids(gold_row)
        page_gold = gold_page_uids(gold_row)
        summary["gold_doc_ids"] = sorted(doc_gold)
        summary["gold_page_uids"] = sorted(page_gold)
        summary["reranked_first_gold_doc_rank"] = first_rank(ranked_docs, doc_gold)
        summary["reranked_first_gold_page_rank"] = first_rank(ranked_pages, page_gold) if page_gold else None
    return summary


def prefixed_gold_ranks(prefix: str, summary: dict) -> dict[str, object]:
    return {
        f"{prefix}_first_gold_doc_rank": summary.get("reranked_first_gold_doc_rank"),
        f"{prefix}_first_gold_page_rank": summary.get("reranked_first_gold_page_rank"),
    }


def main() -> None:
    args = parse_args()

    dense_pred = load_prediction(Path(args.dense_prediction_json))
    sparse_pred = load_prediction(Path(args.sparse_prediction_json))
    common_qids = sorted(set(dense_pred) & set(sparse_pred))
    if not common_qids:
        raise ValueError("Dense and sparse predictions have no qids in common.")

    gold_rows = load_gold_rows(Path(args.gold), args.question_type) if args.gold else {}
    if gold_rows:
        qids = sorted(set(common_qids) & set(gold_rows))
        if not qids:
            raise ValueError("No qids remain after intersecting predictions with gold/filter.")
    else:
        qids = common_qids

    fused_payload: dict[str, dict] = {}
    per_qid: list[dict] = []
    for qid in qids:
        final_rows, graph_metadata = build_qid_graph_ranking(
            qid=qid,
            dense_row=dense_pred[qid],
            sparse_row=sparse_pred[qid],
            args=args,
        )
        question = dense_pred[qid].get("question") or sparse_pred[qid].get("question", "")
        gold_row = gold_rows.get(qid)
        dense_source_summary = summarize_prediction_rows(
            page_rows_from_ranked_unique_pages(
                dense_pred[qid].get("page_retrieval_results", []),
                int(args.dense_top_pages),
            ),
            gold_row,
        )
        sparse_source_summary = summarize_prediction_rows(
            page_rows_from_ranked_unique_pages(
                sparse_pred[qid].get("page_retrieval_results", []),
                int(args.sparse_top_pages),
            ),
            gold_row,
        )
        row_summary = {
            "qid": qid,
            "question": question,
            **summarize_prediction_rows(final_rows, gold_row),
            **prefixed_gold_ranks("dense", dense_source_summary),
            **prefixed_gold_ranks("sparse", sparse_source_summary),
            "graph": {
                key: value
                for key, value in graph_metadata.items()
                if key not in {"top_graph_pages"}
            },
        }
        per_qid.append(row_summary)
        fused_payload[qid] = {
            "pred_answer": dense_pred[qid].get("pred_answer", ""),
            "page_retrieval_results": final_rows,
            "qid": qid,
            "question": question,
            "top_retrieved_docs": ranked_docs_from_rows(final_rows)[:10],
            "reranker_metadata": {
                "fusion_method": "graph_ppr",
                "dense_prediction_json": args.dense_prediction_json,
                "sparse_prediction_json": args.sparse_prediction_json,
                "dense_top_pages": int(args.dense_top_pages),
                "sparse_top_pages": int(args.sparse_top_pages),
                "final_top_pages": int(args.final_top_pages),
                "per_doc_page_limit": int(args.per_doc_page_limit),
                "rrf_k": float(args.rrf_k),
                "dense_weight": float(args.dense_weight),
                "sparse_weight": float(args.sparse_weight),
                "score_seed_weight": float(args.score_seed_weight),
                "doc_seed_weight": float(args.doc_seed_weight),
                "restart_prob": float(args.restart_prob),
                "ppr_iters": int(args.ppr_iters),
                "page_doc_edge_weight": float(args.page_doc_edge_weight),
                "adjacent_page_edge_weight": float(args.adjacent_page_edge_weight),
                "same_doc_window": int(args.same_doc_window),
                "final_page_seed_weight": float(args.final_page_seed_weight),
                "final_ppr_page_weight": float(args.final_ppr_page_weight),
                "final_ppr_doc_weight": float(args.final_ppr_doc_weight),
                "graph": graph_metadata,
            },
        }

    summary: dict[str, object] = {
        "fusion_method": "graph_ppr",
        "qid_count": len(qids),
        "dense_prediction_json": args.dense_prediction_json,
        "sparse_prediction_json": args.sparse_prediction_json,
        "gold": args.gold,
        "question_type": args.question_type,
        "dense_top_pages": int(args.dense_top_pages),
        "sparse_top_pages": int(args.sparse_top_pages),
        "final_top_pages": int(args.final_top_pages),
        "per_doc_page_limit": int(args.per_doc_page_limit),
        "rrf_k": float(args.rrf_k),
        "dense_weight": float(args.dense_weight),
        "sparse_weight": float(args.sparse_weight),
        "score_seed_weight": float(args.score_seed_weight),
        "doc_seed_weight": float(args.doc_seed_weight),
        "restart_prob": float(args.restart_prob),
        "ppr_iters": int(args.ppr_iters),
        "page_doc_edge_weight": float(args.page_doc_edge_weight),
        "adjacent_page_edge_weight": float(args.adjacent_page_edge_weight),
        "same_doc_window": int(args.same_doc_window),
        "final_page_seed_weight": float(args.final_page_seed_weight),
        "final_ppr_page_weight": float(args.final_ppr_page_weight),
        "final_ppr_doc_weight": float(args.final_ppr_doc_weight),
        "mean_candidate_page_count": (
            statistics.fmean(float(row["graph"]["candidate_page_count"]) for row in per_qid)
            if per_qid
            else None
        ),
        "mean_candidate_doc_count": (
            statistics.fmean(float(row["graph"]["candidate_doc_count"]) for row in per_qid)
            if per_qid
            else None
        ),
        "per_qid": per_qid,
    }
    if gold_rows:
        doc_ranks = [row.get("reranked_first_gold_doc_rank") for row in per_qid]
        page_ranks = [row.get("reranked_first_gold_page_rank") for row in per_qid]
        dense_doc_ranks = [row.get("dense_first_gold_doc_rank") for row in per_qid]
        sparse_doc_ranks = [row.get("sparse_first_gold_doc_rank") for row in per_qid]
        dense_page_ranks = [row.get("dense_first_gold_page_rank") for row in per_qid]
        sparse_page_ranks = [row.get("sparse_first_gold_page_rank") for row in per_qid]
        summary["reranked_top4_doc_count"] = sum(
            1 for rank in doc_ranks if rank is not None and int(rank) <= 4
        )
        summary["reranked_top20_doc_count"] = sum(
            1 for rank in doc_ranks if rank is not None and int(rank) <= 20
        )
        summary["reranked_top4_page_count"] = sum(
            1 for rank in page_ranks if rank is not None and int(rank) <= 4
        )
        summary["reranked_top20_page_count"] = sum(
            1 for rank in page_ranks if rank is not None and int(rank) <= 20
        )
        summary["dense_top4_doc_count"] = sum(
            1 for rank in dense_doc_ranks if rank is not None and int(rank) <= 4
        )
        summary["dense_top20_doc_count"] = sum(
            1 for rank in dense_doc_ranks if rank is not None and int(rank) <= 20
        )
        summary["sparse_top4_doc_count"] = sum(
            1 for rank in sparse_doc_ranks if rank is not None and int(rank) <= 4
        )
        summary["sparse_top20_doc_count"] = sum(
            1 for rank in sparse_doc_ranks if rank is not None and int(rank) <= 20
        )
        summary["dense_top4_page_count"] = sum(
            1 for rank in dense_page_ranks if rank is not None and int(rank) <= 4
        )
        summary["dense_top20_page_count"] = sum(
            1 for rank in dense_page_ranks if rank is not None and int(rank) <= 20
        )
        summary["sparse_top4_page_count"] = sum(
            1 for rank in sparse_page_ranks if rank is not None and int(rank) <= 4
        )
        summary["sparse_top20_page_count"] = sum(
            1 for rank in sparse_page_ranks if rank is not None and int(rank) <= 20
        )
        summary["graph_recovers_top4_doc_vs_dense_count"] = sum(
            1
            for dense_rank, graph_rank in zip(dense_doc_ranks, doc_ranks)
            if not (dense_rank is not None and int(dense_rank) <= 4)
            and graph_rank is not None
            and int(graph_rank) <= 4
        )
        summary["graph_loses_top4_doc_vs_dense_count"] = sum(
            1
            for dense_rank, graph_rank in zip(dense_doc_ranks, doc_ranks)
            if dense_rank is not None
            and int(dense_rank) <= 4
            and not (graph_rank is not None and int(graph_rank) <= 4)
        )
        summary["reranked_doc_rank_median"] = median_or_none(doc_ranks)  # type: ignore[arg-type]
        summary["reranked_page_rank_median"] = median_or_none(page_ranks)  # type: ignore[arg-type]
        summary["dense_doc_rank_median"] = median_or_none(dense_doc_ranks)  # type: ignore[arg-type]
        summary["sparse_doc_rank_median"] = median_or_none(sparse_doc_ranks)  # type: ignore[arg-type]

    output_prediction_json = Path(args.output_prediction_json)
    output_prediction_json.parent.mkdir(parents=True, exist_ok=True)
    output_prediction_json.write_text(json.dumps(fused_payload, indent=2) + "\n", encoding="utf-8")

    output_summary_json = Path(args.output_summary_json)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"saved_prediction: {output_prediction_json}")
    print(f"saved_summary: {output_summary_json}")
    print(f"qid_count: {len(qids)}")
    if gold_rows:
        print(f"reranked_top4_doc_count: {summary['reranked_top4_doc_count']}")
        print(f"reranked_top20_doc_count: {summary['reranked_top20_doc_count']}")
        print(f"reranked_top4_page_count: {summary['reranked_top4_page_count']}")
        print(f"reranked_top20_page_count: {summary['reranked_top20_page_count']}")


if __name__ == "__main__":
    main()
