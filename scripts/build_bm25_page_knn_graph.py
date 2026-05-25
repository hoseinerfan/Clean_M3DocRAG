#!/usr/bin/env python3

from __future__ import annotations

import argparse
import heapq
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build query-independent page-page kNN graph edges from BM25 lexical "
            "similarity over page text. The output JSONL is compatible with "
            "graph_rerank_page_retrieval_predictions.py --external-page-graph-jsonl."
        )
    )
    parser.add_argument("--page-text-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument(
        "--text-field",
        action="append",
        default=[],
        help=(
            "Page JSONL field to index. Repeat to concatenate fields. Defaults to "
            "the exported page-text field `text`."
        ),
    )
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
        help="Use at most this many highest BM25-idf source terms per source page. Use 0 for all.",
    )
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument(
        "--score-output-mode",
        choices=["raw", "source_max"],
        default="source_max",
        help=(
            "How to write edge score/weight values. source_max scales each source page's "
            "BM25 neighbor scores to [0, 1], which is safer for graph edge weighting."
        ),
    )
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--min-token-len", type=int, default=2)
    parser.add_argument(
        "--source-term-weight-mode",
        choices=["binary", "tf", "log_tf"],
        default="log_tf",
        help="How repeated source-page terms weight BM25 query terms.",
    )
    parser.add_argument(
        "--max-token-doc-freq",
        type=int,
        default=0,
        help="Drop terms appearing in more than this many pages. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-token-doc-freq-frac",
        type=float,
        default=0.0,
        help="Drop terms appearing in more than this fraction of pages. Use 0 to disable.",
    )
    parser.add_argument(
        "--min-token-doc-freq",
        type=int,
        default=1,
        help="Drop terms appearing in fewer than this many pages.",
    )
    parser.add_argument(
        "--require-nonempty-text",
        action="store_true",
        help="Fail if all indexed pages have empty text.",
    )
    parser.add_argument("--same-doc-only", action="store_true")
    parser.add_argument("--cross-doc-only", action="store_true")
    parser.add_argument(
        "--bidirectional-dedup",
        action="store_true",
        help="Emit only one edge for each unordered pair. PPR can add reverse edges later.",
    )
    parser.add_argument(
        "--mutual-only",
        action="store_true",
        help=(
            "Keep only reciprocal BM25 kNN edges: source->target is emitted only when source "
            "also appears in target's top-k BM25 neighbors under the same scoring setup."
        ),
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


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = " ".join(str(item) for item in value if item is not None)
    elif isinstance(value, dict):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).replace("\x00", " ").replace("\x0c", " ")


def row_text(row: dict[str, Any], fields: list[str]) -> str:
    values = [normalize_text(row.get(field)) for field in fields]
    return " ".join(value for value in values if value.strip())


def tokenize(text: str, min_token_len: int) -> list[str]:
    tokens: list[str] = []
    for raw in TOKEN_RE.findall(text.lower()):
        token = raw.strip("_")
        if not token:
            continue
        if token.isdigit() or len(token) >= min_token_len:
            tokens.append(token)
    return tokens


def load_page_text_index(
    path: Path,
    *,
    text_fields: list[str],
    min_token_len: int,
    require_nonempty_text: bool,
) -> dict[str, Any]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"No rows found in {path}")

    page_uids: list[str] = []
    doc_ids: list[str] = []
    page_indices: list[int] = []
    term_counts: list[Counter[str]] = []
    doc_freq: Counter[str] = Counter()
    doc_lengths: list[int] = []
    nonempty_text_page_count = 0

    fields = text_fields or ["text"]
    for row in rows:
        doc_id = str(row.get("doc_id", "")).strip()
        page_idx = row.get("page_idx", row.get("page_id"))
        uid = str(row.get("page_uid", "")).strip()
        if not uid and doc_id and page_idx is not None:
            uid = page_uid(doc_id, int(page_idx))
        if not uid or not doc_id or page_idx is None:
            raise ValueError(f"Page row is missing page_uid/doc_id/page_idx: {row}")

        text = row_text(row, fields)
        if text.strip():
            nonempty_text_page_count += 1
        counts = Counter(tokenize(text, min_token_len=min_token_len))
        page_uids.append(uid)
        doc_ids.append(doc_id)
        page_indices.append(int(page_idx))
        term_counts.append(counts)
        doc_lengths.append(sum(counts.values()))
        doc_freq.update(counts.keys())

    if require_nonempty_text and nonempty_text_page_count == 0:
        raise ValueError(
            f"No non-empty page text rows found in {path}. Provide a page-text JSONL "
            "with real text or disable --require-nonempty-text only for an explicit ablation."
        )

    return {
        "page_text_jsonl": str(path),
        "text_fields": fields,
        "page_uids": page_uids,
        "doc_ids": doc_ids,
        "page_indices": page_indices,
        "term_counts": term_counts,
        "doc_freq": doc_freq,
        "doc_lengths": doc_lengths,
        "nonempty_text_page_count": nonempty_text_page_count,
    }


def kept_terms(index: dict[str, Any], args: argparse.Namespace) -> set[str]:
    page_count = len(index["page_uids"])
    max_df = int(args.max_token_doc_freq)
    if float(args.max_token_doc_freq_frac) > 0:
        frac_cap = int(math.floor(float(args.max_token_doc_freq_frac) * page_count))
        max_df = frac_cap if max_df <= 0 else min(max_df, frac_cap)
    min_df = max(1, int(args.min_token_doc_freq))
    keep: set[str] = set()
    for term, df in index["doc_freq"].items():
        if df < min_df:
            continue
        if max_df > 0 and df > max_df:
            continue
        keep.add(term)
    return keep


def bm25_idf(df: int, page_count: int) -> float:
    return math.log(1.0 + (float(page_count) - float(df) + 0.5) / (float(df) + 0.5))


def build_postings(
    index: dict[str, Any],
    *,
    keep_terms: set[str],
) -> dict[str, list[tuple[int, int]]]:
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for page_idx, counts in enumerate(index["term_counts"]):
        for term, tf in counts.items():
            if term in keep_terms and tf > 0:
                postings[term].append((page_idx, int(tf)))
    return dict(postings)


def source_term_weight(tf: int, mode: str) -> float:
    if mode == "binary":
        return 1.0
    if mode == "tf":
        return float(tf)
    return 1.0 + math.log(float(tf))


def source_terms(
    index: dict[str, Any],
    page_idx: int,
    *,
    keep_terms: set[str],
    idf_by_term: dict[str, float],
    topk_terms: int,
    weight_mode: str,
) -> list[tuple[str, float]]:
    terms: list[tuple[str, float, float]] = []
    for term, tf in index["term_counts"][page_idx].items():
        if term not in keep_terms:
            continue
        query_weight = source_term_weight(int(tf), weight_mode)
        terms.append((term, query_weight, query_weight * idf_by_term.get(term, 0.0)))
    terms.sort(key=lambda item: (-item[2], item[0]))
    if topk_terms > 0:
        terms = terms[:topk_terms]
    return [(term, query_weight) for term, query_weight, _rank_weight in terms]


def top_neighbors_for_source(
    *,
    source_idx: int,
    index: dict[str, Any],
    postings: dict[str, list[tuple[int, int]]],
    keep_terms: set[str],
    idf_by_term: dict[str, float],
    avgdl: float,
    top_k: int,
    source_topk_terms: int,
    min_score: float,
    source_term_weight_mode: str,
    k1: float,
    b: float,
    same_doc_only: bool,
    cross_doc_only: bool,
) -> list[tuple[int, float]]:
    query_terms = source_terms(
        index,
        source_idx,
        keep_terms=keep_terms,
        idf_by_term=idf_by_term,
        topk_terms=source_topk_terms,
        weight_mode=source_term_weight_mode,
    )
    if not query_terms:
        return []

    source_doc = index["doc_ids"][source_idx]
    scores: dict[int, float] = defaultdict(float)
    doc_lengths = index["doc_lengths"]
    for term, query_weight in query_terms:
        idf = idf_by_term.get(term, 0.0)
        if idf <= 0:
            continue
        for target_idx, tf in postings.get(term, []):
            if target_idx == source_idx:
                continue
            dl = float(doc_lengths[target_idx])
            norm = float(k1) * (1.0 - float(b) + float(b) * dl / max(avgdl, 1e-9))
            term_score = idf * (float(tf) * (float(k1) + 1.0)) / (float(tf) + norm)
            scores[target_idx] += float(query_weight) * term_score

    candidates: list[tuple[int, float]] = []
    for target_idx, score in scores.items():
        target_doc = index["doc_ids"][target_idx]
        if same_doc_only and source_doc != target_doc:
            continue
        if cross_doc_only and source_doc == target_doc:
            continue
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
    if float(args.k1) <= 0:
        raise ValueError("--k1 must be positive.")
    if not 0 <= float(args.b) <= 1:
        raise ValueError("--b must be in [0, 1].")

    index = load_page_text_index(
        Path(args.page_text_jsonl),
        text_fields=list(args.text_field),
        min_token_len=max(1, int(args.min_token_len)),
        require_nonempty_text=bool(args.require_nonempty_text),
    )
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

    keep = kept_terms(index, args)
    postings = build_postings(index, keep_terms=keep)
    page_count = len(page_uids)
    idf_by_term = {
        term: bm25_idf(int(index["doc_freq"][term]), page_count)
        for term in keep
    }
    avgdl = (
        sum(float(length) for length in index["doc_lengths"]) / len(index["doc_lengths"])
        if index["doc_lengths"]
        else 0.0
    )

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    edge_count = 0
    source_page_count = 0
    target_pages: set[str] = set()
    target_docs: set[str] = set()
    emitted_pair_keys: set[tuple[str, str]] = set()
    score_values: list[float] = []
    neighbor_cache: dict[int, list[tuple[int, float]]] = {}
    candidate_edge_count = 0
    mutual_rejected_edge_count = 0
    neighbor_index_cache: dict[int, set[int]] = {}

    def cached_neighbors(source_idx: int) -> list[tuple[int, float]]:
        neighbors = neighbor_cache.get(source_idx)
        if neighbors is None:
            neighbors = top_neighbors_for_source(
                source_idx=source_idx,
                index=index,
                postings=postings,
                keep_terms=keep,
                idf_by_term=idf_by_term,
                avgdl=avgdl,
                top_k=top_k,
                source_topk_terms=int(args.source_topk_terms),
                min_score=float(args.min_score),
                source_term_weight_mode=str(args.source_term_weight_mode),
                k1=float(args.k1),
                b=float(args.b),
                same_doc_only=bool(args.same_doc_only),
                cross_doc_only=bool(args.cross_doc_only),
            )
            neighbor_cache[source_idx] = neighbors
        return neighbors

    def cached_neighbor_indices(source_idx: int) -> set[int]:
        neighbor_indices = neighbor_index_cache.get(source_idx)
        if neighbor_indices is None:
            neighbor_indices = {idx for idx, _score in cached_neighbors(source_idx)}
            neighbor_index_cache[source_idx] = neighbor_indices
        return neighbor_indices

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for source_idx in source_indices:
            source_uid = page_uids[source_idx]
            neighbors = cached_neighbors(source_idx)
            source_max_score = max((float(score) for _idx, score in neighbors), default=0.0)
            emitted_for_source = 0
            for target_idx, score in neighbors:
                candidate_edge_count += 1
                target_uid = page_uids[target_idx]
                if bool(args.mutual_only):
                    if source_idx not in cached_neighbor_indices(target_idx):
                        mutual_rejected_edge_count += 1
                        continue
                if bool(args.bidirectional_dedup):
                    pair_key = tuple(sorted((source_uid, target_uid)))
                    if pair_key in emitted_pair_keys:
                        continue
                    emitted_pair_keys.add(pair_key)
                output_score = float(score)
                if str(args.score_output_mode) == "source_max" and source_max_score > 0:
                    output_score = output_score / source_max_score
                row = {
                    "edge_type": "bm25_page_knn",
                    "source_page_uid": source_uid,
                    "target_page_uid": target_uid,
                    "source_doc_id": index["doc_ids"][source_idx],
                    "source_page_idx": int(index["page_indices"][source_idx]),
                    "target_doc_id": index["doc_ids"][target_idx],
                    "target_page_idx": int(index["page_indices"][target_idx]),
                    "score": float(output_score),
                    "weight": float(output_score),
                    "raw_score": float(score),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                edge_count += 1
                emitted_for_source += 1
                target_pages.add(target_uid)
                target_docs.add(str(index["doc_ids"][target_idx]))
                score_values.append(float(output_score))
            if emitted_for_source > 0:
                source_page_count += 1

    summary = {
        "page_text_jsonl": args.page_text_jsonl,
        "text_fields": index["text_fields"],
        "page_count": page_count,
        "nonempty_text_page_count": int(index["nonempty_text_page_count"]),
        "empty_text_page_count": page_count - int(index["nonempty_text_page_count"]),
        "source_prediction_jsons": list(args.source_prediction_json),
        "qid_filter_count": qid_filter_count(args),
        "requested_source_page_count": len(source_pages) if source_pages else page_count,
        "missing_source_page_count": len(missing_source_pages),
        "source_page_count": source_page_count,
        "target_page_count": len(target_pages),
        "target_doc_count": len(target_docs),
        "edge_count": edge_count,
        "top_k": top_k,
        "source_top_pages": int(args.source_top_pages),
        "source_topk_terms": int(args.source_topk_terms),
        "source_term_weight_mode": str(args.source_term_weight_mode),
        "min_score": float(args.min_score),
        "score_output_mode": str(args.score_output_mode),
        "k1": float(args.k1),
        "b": float(args.b),
        "min_token_len": int(args.min_token_len),
        "min_token_doc_freq": int(args.min_token_doc_freq),
        "max_token_doc_freq": int(args.max_token_doc_freq),
        "max_token_doc_freq_frac": float(args.max_token_doc_freq_frac),
        "kept_term_count": len(keep),
        "posting_term_count": len(postings),
        "posting_count": sum(len(values) for values in postings.values()),
        "avg_doc_length": avgdl,
        "same_doc_only": bool(args.same_doc_only),
        "cross_doc_only": bool(args.cross_doc_only),
        "bidirectional_dedup": bool(args.bidirectional_dedup),
        "mutual_only": bool(args.mutual_only),
        "candidate_edge_count_before_mutual_filter": candidate_edge_count,
        "mutual_rejected_edge_count": mutual_rejected_edge_count,
        "neighbor_cache_size": len(neighbor_cache),
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
        "nonempty_text_page_count",
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
