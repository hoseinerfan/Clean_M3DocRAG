#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class SourceTokenHit:
    token_idx: int
    page_uid: str
    query_token_idx: int
    query_neighbor_rank: int
    source_score: float


@dataclass
class EdgeAccumulator:
    values: list[float] = field(default_factory=list)
    source_token_indices: set[int] = field(default_factory=set)
    max_source_score: float = 0.0
    max_neighbor_score: float = 0.0
    best_query_neighbor_rank: int | None = None
    best_token_neighbor_rank: int | None = None

    def add(
        self,
        *,
        value: float,
        source_token_idx: int,
        source_score: float,
        neighbor_score: float,
        query_neighbor_rank: int,
        token_neighbor_rank: int,
    ) -> None:
        self.values.append(float(value))
        self.source_token_indices.add(int(source_token_idx))
        self.max_source_score = max(self.max_source_score, float(source_score))
        self.max_neighbor_score = max(self.max_neighbor_score, float(neighbor_score))
        if self.best_query_neighbor_rank is None:
            self.best_query_neighbor_rank = int(query_neighbor_rank)
        else:
            self.best_query_neighbor_rank = min(
                self.best_query_neighbor_rank,
                int(query_neighbor_rank),
            )
        if self.best_token_neighbor_rank is None:
            self.best_token_neighbor_rank = int(token_neighbor_rank)
        else:
            self.best_token_neighbor_rank = min(
                self.best_token_neighbor_rank,
                int(token_neighbor_rank),
            )

    def raw_score(self, mode: str) -> float:
        if not self.values:
            return 0.0
        if mode == "max":
            return max(self.values)
        if mode == "mean":
            return sum(self.values) / len(self.values)
        if mode == "sum":
            return sum(self.values)
        if mode == "log_count":
            return max(self.values) * math.log1p(len(self.values))
        raise ValueError(f"Unsupported edge aggregation mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build query-specific external page-page graph edges from FAISS token "
            "neighbors. For each query token's retrieved document tokens, the script "
            "looks up nearby document tokens in the same FAISS index and links the "
            "source page to the neighbor-token pages."
        )
    )
    parser.add_argument("--prediction-json", required=True)
    parser.add_argument("--query-embedding-dir", required=True)
    parser.add_argument("--page-embedding-dir", required=True)
    parser.add_argument("--doc-ids-json", required=True)
    parser.add_argument("--faiss-index", required=True)
    parser.add_argument("--faiss-nprobe", type=int, default=0)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--query-embedding-key", default="embeddings")
    parser.add_argument("--page-embedding-key", default="embeddings")
    parser.add_argument("--source-page-top-k", type=int, default=1000)
    parser.add_argument(
        "--target-page-top-k",
        type=int,
        default=1000,
        help=(
            "Only keep neighbor-token targets that are also in the top-K prediction "
            "candidate pages. Use 0 to allow any page in the FAISS index."
        ),
    )
    parser.add_argument(
        "--query-faiss-hit-k",
        type=int,
        default=224,
        help="How many FAISS token hits to inspect for each query token.",
    )
    parser.add_argument(
        "--source-token-top-k",
        type=int,
        default=128,
        help="Maximum source document-token hits to expand per qid. Use 0 for all.",
    )
    parser.add_argument(
        "--neighbor-token-k",
        type=int,
        default=10,
        help="How many FAISS neighbor tokens to pull from each selected source token.",
    )
    parser.add_argument(
        "--max-edges-per-source-page",
        type=int,
        default=10,
        help="Keep at most this many target pages per source page after aggregation. Use 0 for all.",
    )
    parser.add_argument("--min-source-score", type=float, default=0.0)
    parser.add_argument("--min-neighbor-score", type=float, default=0.0)
    parser.add_argument(
        "--edge-value-mode",
        choices=["neighbor", "source_neighbor_product", "rank_decay"],
        default="neighbor",
    )
    parser.add_argument(
        "--edge-aggregation",
        choices=["max", "mean", "sum", "log_count"],
        default="log_count",
    )
    parser.add_argument(
        "--score-normalization",
        choices=["none", "per_qid", "per_source_page"],
        default="per_source_page",
    )
    parser.add_argument("--include-self-page", action="store_true")
    parser.add_argument("--qid", action="append", default=[], help="Optional qid filter; repeatable.")
    parser.add_argument("--limit-qids", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_prediction(path: Path) -> dict[str, dict]:
    payload = read_json(path)
    if isinstance(payload, dict) and "predictions" in payload:
        payload = payload["predictions"]
    rows_by_qid: dict[str, dict] = {}
    if isinstance(payload, dict):
        iterator = payload.items()
    elif isinstance(payload, list):
        iterator = enumerate(payload)
    else:
        raise TypeError(f"Unsupported prediction JSON root: {path}")
    for raw_key, row in iterator:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("qid") or raw_key).strip()
        if qid:
            rows_by_qid[qid] = row
    return rows_by_qid


def page_uid(doc_id: str, page_idx: int) -> str:
    return f"{doc_id}_page{int(page_idx)}"


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


def prediction_page_uids(row: dict, top_k: int) -> list[str]:
    rows = row.get("page_retrieval_results", [])
    if not isinstance(rows, list):
        return []
    seen: set[str] = set()
    page_uids: list[str] = []
    for raw in rows:
        uid = row_page_uid(raw)
        if uid is None or uid in seen:
            continue
        seen.add(uid)
        page_uids.append(uid)
        if top_k > 0 and len(page_uids) >= top_k:
            break
    return page_uids


def load_safetensor_array(path: Path, key: str) -> np.ndarray:
    import torch
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        tensor_key = key if key in handle.keys() else next(iter(handle.keys()))
        tensor = handle.get_tensor(tensor_key)
    if tensor.dtype in {torch.bfloat16, torch.float16}:
        tensor = tensor.float()
    return tensor.detach().cpu().numpy().astype(np.float32, copy=False)


def load_query_embedding(path: Path, key: str) -> np.ndarray:
    array = load_safetensor_array(path, key)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"Query embedding must be 2-D after squeeze: {path} shape={array.shape}")
    return np.ascontiguousarray(array, dtype=np.float32)


def progress_iter(items: Iterable[str], total: int | None = None):
    try:
        from tqdm.auto import tqdm

        return tqdm(items, total=total)
    except Exception:
        return items


def build_flattened_doc_token_table(
    *,
    page_embedding_dir: Path,
    doc_ids: list[str],
    page_embedding_key: str,
) -> tuple[list[str], np.ndarray]:
    token2pageuid: list[str] = []
    chunks: list[np.ndarray] = []
    for doc_id in progress_iter(doc_ids, total=len(doc_ids)):
        emb_path = page_embedding_dir / f"{doc_id}.safetensors"
        if not emb_path.exists():
            raise FileNotFoundError(f"Missing page embedding file: {emb_path}")
        doc_emb = load_safetensor_array(emb_path, page_embedding_key)
        if doc_emb.ndim != 3:
            raise ValueError(
                "Expected ColPali document embeddings shaped "
                f"[n_pages, n_tokens, dim], got {doc_emb.shape} for {emb_path}"
            )
        for page_idx in range(doc_emb.shape[0]):
            page_emb = np.ascontiguousarray(doc_emb[page_idx].reshape(-1, doc_emb.shape[-1]), dtype=np.float32)
            chunks.append(page_emb)
            token2pageuid.extend([page_uid(doc_id, page_idx)] * page_emb.shape[0])
    if not chunks:
        raise ValueError(f"No embeddings loaded from {page_embedding_dir}")
    all_token_embeddings = np.concatenate(chunks, axis=0)
    return token2pageuid, np.ascontiguousarray(all_token_embeddings, dtype=np.float32)


def select_source_token_hits(
    *,
    query_emb: np.ndarray,
    index,
    token2pageuid: list[str],
    source_page_uids: set[str],
    query_faiss_hit_k: int,
    source_token_top_k: int,
    min_source_score: float,
) -> list[SourceTokenHit]:
    distances, indices = index.search(query_emb, int(query_faiss_hit_k))
    best_by_token: dict[int, SourceTokenHit] = {}
    for query_token_idx in range(indices.shape[0]):
        for nn_rank in range(indices.shape[1]):
            token_idx = int(indices[query_token_idx, nn_rank])
            if token_idx < 0 or token_idx >= len(token2pageuid):
                continue
            source_score = float(distances[query_token_idx, nn_rank])
            if source_score < min_source_score:
                continue
            page = token2pageuid[token_idx]
            if page not in source_page_uids:
                continue
            existing = best_by_token.get(token_idx)
            if existing is None or source_score > existing.source_score:
                best_by_token[token_idx] = SourceTokenHit(
                    token_idx=token_idx,
                    page_uid=page,
                    query_token_idx=query_token_idx,
                    query_neighbor_rank=nn_rank + 1,
                    source_score=source_score,
                )
    hits = sorted(
        best_by_token.values(),
        key=lambda item: (-item.source_score, item.query_neighbor_rank, item.token_idx),
    )
    if source_token_top_k > 0:
        hits = hits[:source_token_top_k]
    return hits


def edge_value(
    *,
    source_score: float,
    neighbor_score: float,
    query_neighbor_rank: int,
    token_neighbor_rank: int,
    mode: str,
) -> float:
    if mode == "neighbor":
        return max(0.0, neighbor_score)
    if mode == "source_neighbor_product":
        return max(0.0, source_score) * max(0.0, neighbor_score)
    if mode == "rank_decay":
        return 1.0 / ((query_neighbor_rank + 1.0) * (token_neighbor_rank + 1.0))
    raise ValueError(f"Unsupported edge value mode: {mode}")


def build_edges_for_qid(
    *,
    qid: str,
    row: dict,
    query_emb: np.ndarray,
    index,
    token2pageuid: list[str],
    all_token_embeddings: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict], dict[str, object]]:
    source_page_list = prediction_page_uids(row, int(args.source_page_top_k))
    target_page_top_k = int(args.target_page_top_k)
    target_page_list = (
        prediction_page_uids(row, target_page_top_k) if target_page_top_k > 0 else []
    )
    source_page_uids = set(source_page_list)
    target_page_uids = set(target_page_list)
    source_hits = select_source_token_hits(
        query_emb=query_emb,
        index=index,
        token2pageuid=token2pageuid,
        source_page_uids=source_page_uids,
        query_faiss_hit_k=int(args.query_faiss_hit_k),
        source_token_top_k=int(args.source_token_top_k),
        min_source_score=float(args.min_source_score),
    )
    if not source_hits:
        return [], {
            "source_page_count": len(source_page_uids),
            "source_token_count": 0,
            "edge_count": 0,
        }

    source_vectors = np.ascontiguousarray(
        all_token_embeddings[[hit.token_idx for hit in source_hits]],
        dtype=np.float32,
    )
    neighbor_k = int(args.neighbor_token_k) + 1
    neighbor_scores, neighbor_indices = index.search(source_vectors, neighbor_k)

    by_pair: dict[tuple[str, str], EdgeAccumulator] = defaultdict(EdgeAccumulator)
    for source_row_idx, hit in enumerate(source_hits):
        for nn_rank in range(neighbor_indices.shape[1]):
            neighbor_token_idx = int(neighbor_indices[source_row_idx, nn_rank])
            if neighbor_token_idx < 0 or neighbor_token_idx >= len(token2pageuid):
                continue
            if neighbor_token_idx == hit.token_idx:
                continue
            neighbor_score = float(neighbor_scores[source_row_idx, nn_rank])
            if neighbor_score < float(args.min_neighbor_score):
                continue
            target_page_uid = token2pageuid[neighbor_token_idx]
            if not bool(args.include_self_page) and target_page_uid == hit.page_uid:
                continue
            if target_page_uids and target_page_uid not in target_page_uids:
                continue
            value = edge_value(
                source_score=hit.source_score,
                neighbor_score=neighbor_score,
                query_neighbor_rank=hit.query_neighbor_rank,
                token_neighbor_rank=nn_rank + 1,
                mode=str(args.edge_value_mode),
            )
            if value <= 0:
                continue
            by_pair[(hit.page_uid, target_page_uid)].add(
                value=value,
                source_token_idx=hit.token_idx,
                source_score=hit.source_score,
                neighbor_score=neighbor_score,
                query_neighbor_rank=hit.query_neighbor_rank,
                token_neighbor_rank=nn_rank + 1,
            )

    raw_edges: list[dict] = []
    for (source_page_uid, target_page_uid), stats in by_pair.items():
        raw_score = stats.raw_score(str(args.edge_aggregation))
        if raw_score <= 0:
            continue
        raw_edges.append(
            {
                "qid": qid,
                "source_page_uid": source_page_uid,
                "target_page_uid": target_page_uid,
                "raw_score": raw_score,
                "raw_token_hit_count": len(stats.values),
                "source_token_count": len(stats.source_token_indices),
                "max_source_score": stats.max_source_score,
                "max_neighbor_score": stats.max_neighbor_score,
                "best_query_neighbor_rank": stats.best_query_neighbor_rank,
                "best_token_neighbor_rank": stats.best_token_neighbor_rank,
            }
        )

    if str(args.score_normalization) == "per_qid":
        max_score = max((edge["raw_score"] for edge in raw_edges), default=0.0)
        for edge in raw_edges:
            edge["score"] = edge["raw_score"] / max_score if max_score > 0 else 0.0
    elif str(args.score_normalization) == "per_source_page":
        max_by_source: dict[str, float] = defaultdict(float)
        for edge in raw_edges:
            max_by_source[edge["source_page_uid"]] = max(
                max_by_source[edge["source_page_uid"]],
                float(edge["raw_score"]),
            )
        for edge in raw_edges:
            max_score = max_by_source[edge["source_page_uid"]]
            edge["score"] = edge["raw_score"] / max_score if max_score > 0 else 0.0
    else:
        for edge in raw_edges:
            edge["score"] = edge["raw_score"]

    grouped: dict[str, list[dict]] = defaultdict(list)
    for edge in raw_edges:
        grouped[edge["source_page_uid"]].append(edge)

    output_edges: list[dict] = []
    for source_page_uid in sorted(grouped):
        source_edges = sorted(
            grouped[source_page_uid],
            key=lambda item: (-float(item["score"]), item["target_page_uid"]),
        )
        if int(args.max_edges_per_source_page) > 0:
            source_edges = source_edges[: int(args.max_edges_per_source_page)]
        output_edges.extend(source_edges)

    for edge in output_edges:
        edge["score"] = float(edge["score"])
        edge["weight"] = float(edge["score"])
        edge["edge_type"] = "faiss_token_neighbor"

    return output_edges, {
        "source_page_count": len(source_page_uids),
        "target_page_pool_count": len(target_page_uids),
        "source_token_count": len(source_hits),
        "edge_count": len(output_edges),
        "raw_edge_count": len(raw_edges),
    }


def main() -> None:
    args = parse_args()
    prediction_path = Path(args.prediction_json)
    query_embedding_dir = Path(args.query_embedding_dir)
    page_embedding_dir = Path(args.page_embedding_dir)
    doc_ids_path = Path(args.doc_ids_json)
    faiss_index_path = Path(args.faiss_index)
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    print(f"loading_prediction={prediction_path}")
    predictions = load_prediction(prediction_path)
    if args.qid:
        keep_qids = set(map(str, args.qid))
        predictions = {qid: row for qid, row in predictions.items() if qid in keep_qids}
    if int(args.limit_qids) > 0:
        predictions = dict(list(predictions.items())[: int(args.limit_qids)])

    print(f"loading_doc_ids={doc_ids_path}")
    doc_ids = [str(doc_id) for doc_id in read_json(doc_ids_path)]

    print(f"loading_faiss_index={faiss_index_path}")
    import faiss

    index = faiss.read_index(str(faiss_index_path))
    if hasattr(index, "nprobe"):
        nprobe = int(getattr(args, "faiss_nprobe", 0) or 0)
        if nprobe > 0:
            index.nprobe = nprobe

    print(f"loading_page_embeddings={page_embedding_dir}")
    token2pageuid, all_token_embeddings = build_flattened_doc_token_table(
        page_embedding_dir=page_embedding_dir,
        doc_ids=doc_ids,
        page_embedding_key=str(args.page_embedding_key),
    )
    if int(index.ntotal) != len(token2pageuid):
        raise ValueError(
            "FAISS index and reconstructed token table disagree: "
            f"index.ntotal={index.ntotal} token2pageuid={len(token2pageuid)}"
        )
    if all_token_embeddings.shape[0] != len(token2pageuid):
        raise ValueError("Token embedding table and token2pageuid length disagree.")

    print(f"building_edges_for_qids={len(predictions)}")
    total_edges = 0
    missing_query_embeddings = 0
    qid_stats: list[dict[str, object]] = []
    with output_jsonl.open("w", encoding="utf-8") as out:
        for offset, (qid, row) in enumerate(predictions.items(), start=1):
            query_path = query_embedding_dir / f"{qid}.safetensors"
            if not query_path.exists():
                missing_query_embeddings += 1
                continue
            query_emb = load_query_embedding(query_path, str(args.query_embedding_key))
            edges, stats = build_edges_for_qid(
                qid=qid,
                row=row,
                query_emb=query_emb,
                index=index,
                token2pageuid=token2pageuid,
                all_token_embeddings=all_token_embeddings,
                args=args,
            )
            for edge in edges:
                out.write(json.dumps(edge, sort_keys=True) + "\n")
            total_edges += len(edges)
            stats = {"qid": qid, **stats}
            qid_stats.append(stats)
            if int(args.log_every) > 0 and offset % int(args.log_every) == 0:
                print(f"processed_qids={offset} total_edges={total_edges}")

    edge_counts = [int(row.get("edge_count", 0)) for row in qid_stats]
    source_token_counts = [int(row.get("source_token_count", 0)) for row in qid_stats]
    summary = {
        "prediction_json": str(prediction_path),
        "query_embedding_dir": str(query_embedding_dir),
        "page_embedding_dir": str(page_embedding_dir),
        "doc_ids_json": str(doc_ids_path),
        "faiss_index": str(faiss_index_path),
        "output_jsonl": str(output_jsonl),
        "qid_count": len(predictions),
        "processed_qid_count": len(qid_stats),
        "missing_query_embedding_count": missing_query_embeddings,
        "edge_count": total_edges,
        "mean_edges_per_processed_qid": (
            sum(edge_counts) / len(edge_counts) if edge_counts else 0.0
        ),
        "mean_source_tokens_per_processed_qid": (
            sum(source_token_counts) / len(source_token_counts) if source_token_counts else 0.0
        ),
        "query_faiss_hit_k": int(args.query_faiss_hit_k),
        "source_token_top_k": int(args.source_token_top_k),
        "neighbor_token_k": int(args.neighbor_token_k),
        "max_edges_per_source_page": int(args.max_edges_per_source_page),
        "source_page_top_k": int(args.source_page_top_k),
        "target_page_top_k": int(args.target_page_top_k),
        "edge_value_mode": str(args.edge_value_mode),
        "edge_aggregation": str(args.edge_aggregation),
        "score_normalization": str(args.score_normalization),
        "elapsed_sec": time.time() - start,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
