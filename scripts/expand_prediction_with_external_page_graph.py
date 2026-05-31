#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


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
    parser.add_argument(
        "--verification-mode",
        choices=["none", "exact_maxsim"],
        default="none",
        help="Optionally rescore candidate pages against the original query before insertion.",
    )
    parser.add_argument("--verify-query-embedding-dir", default="")
    parser.add_argument("--verify-page-embedding-dir", default="")
    parser.add_argument("--verify-query-embedding-key", default="embeddings")
    parser.add_argument("--verify-page-embedding-key", default="embeddings")
    parser.add_argument(
        "--verification-min-score",
        type=float,
        default=-1e30,
        help="Keep verified candidates only when exact verification score is at least this value.",
    )
    parser.add_argument(
        "--verification-candidate-pool",
        type=int,
        default=0,
        help="Verify only the top-N graph candidates per qid before final selection. Use 0 for all.",
    )
    parser.add_argument(
        "--verified-score-mode",
        choices=["verified", "graph", "verified_plus_graph"],
        default="verified",
        help="Score used to sort candidates after verification.",
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


class ExactMaxSimVerifier:
    def __init__(
        self,
        *,
        query_embedding_dir: Path,
        page_embedding_dir: Path,
        query_embedding_key: str,
        page_embedding_key: str,
    ) -> None:
        self.query_embedding_dir = query_embedding_dir
        self.page_embedding_dir = page_embedding_dir
        self.query_embedding_key = query_embedding_key
        self.page_embedding_key = page_embedding_key
        self.query_cache: dict[str, np.ndarray] = {}
        self.doc_cache: dict[str, np.ndarray] = {}
        self.missing_query_count = 0
        self.missing_page_count = 0

    def query_embedding(self, qid: str) -> np.ndarray | None:
        if qid in self.query_cache:
            return self.query_cache[qid]
        path = self.query_embedding_dir / f"{qid}.safetensors"
        if not path.exists():
            self.missing_query_count += 1
            return None
        query_emb = load_query_embedding(path, self.query_embedding_key)
        self.query_cache[qid] = query_emb
        return query_emb

    def doc_embedding(self, doc_id: str) -> np.ndarray | None:
        if doc_id in self.doc_cache:
            return self.doc_cache[doc_id]
        path = self.page_embedding_dir / f"{doc_id}.safetensors"
        if not path.exists():
            self.missing_page_count += 1
            return None
        doc_emb = load_safetensor_array(path, self.page_embedding_key)
        if doc_emb.ndim != 3:
            raise ValueError(
                "Expected ColPali document embeddings shaped "
                f"[n_pages, n_tokens, dim], got {doc_emb.shape} for {path}"
            )
        self.doc_cache[doc_id] = doc_emb
        return doc_emb

    def score_page(self, qid: str, page_uid_value: str) -> float | None:
        parsed = parse_page_uid(page_uid_value)
        if parsed is None:
            return None
        doc_id, page_idx = parsed
        query_emb = self.query_embedding(qid)
        doc_emb = self.doc_embedding(doc_id)
        if query_emb is None or doc_emb is None:
            return None
        if page_idx < 0 or page_idx >= doc_emb.shape[0]:
            self.missing_page_count += 1
            return None
        page_emb = np.ascontiguousarray(doc_emb[page_idx], dtype=np.float32)
        if query_emb.shape[-1] != page_emb.shape[-1]:
            raise ValueError(
                "Query/page embedding dimension mismatch: "
                f"qid={qid} page={page_uid_value} query={query_emb.shape} page={page_emb.shape}"
            )
        sim = np.matmul(query_emb, page_emb.T)
        return float(sim.max(axis=1).sum())


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
    *,
    qid: str,
    rows: list[Any],
    target_scores: dict[str, float],
    max_new_pages: int,
    synthetic_score_mode: str,
    append_after_top_k: int,
    verifier: ExactMaxSimVerifier | None,
    verification_min_score: float,
    verification_candidate_pool: int,
    verified_score_mode: str,
) -> tuple[list[Any], dict[str, Any]]:
    existing_uids = {uid for raw in rows if (uid := row_page_uid(raw))}
    candidates = [
        (uid, score)
        for uid, score in target_scores.items()
        if uid not in existing_uids and parse_page_uid(uid) is not None
    ]
    candidates.sort(key=lambda item: (-item[1], item[0]))
    raw_candidate_count = len(candidates)

    scored_candidates: list[dict[str, float | str]] = []
    verified_scores: list[float] = []
    verification_attempted_count = 0
    verification_rejected_count = 0
    if verifier is not None:
        if verification_candidate_pool > 0:
            candidates = candidates[:verification_candidate_pool]
        for uid, graph_score in candidates:
            verification_attempted_count += 1
            verified_score = verifier.score_page(qid, uid)
            if verified_score is None or verified_score < verification_min_score:
                verification_rejected_count += 1
                continue
            verified_scores.append(float(verified_score))
            if verified_score_mode == "graph":
                final_score = float(graph_score)
            elif verified_score_mode == "verified_plus_graph":
                final_score = float(verified_score) + float(graph_score)
            else:
                final_score = float(verified_score)
            scored_candidates.append(
                {
                    "uid": uid,
                    "graph_score": float(graph_score),
                    "verified_score": float(verified_score),
                    "final_score": float(final_score),
                }
            )
        scored_candidates.sort(
            key=lambda item: (
                -float(item["final_score"]),
                -float(item["graph_score"]),
                str(item["uid"]),
            )
        )
    else:
        scored_candidates = [
            {
                "uid": uid,
                "graph_score": float(score),
                "verified_score": float("nan"),
                "final_score": float(score),
            }
            for uid, score in candidates
        ]

    if max_new_pages > 0:
        scored_candidates = scored_candidates[:max_new_pages]

    original_scores = [row_score(raw) for raw in rows]
    min_original_score = min(original_scores) if original_scores else 0.0
    max_candidate_score = max((float(item["final_score"]) for item in scored_candidates), default=0.0)
    new_rows: list[list[object]] = []
    for idx, item in enumerate(scored_candidates, start=1):
        uid = str(item["uid"])
        parsed = parse_page_uid(uid)
        if parsed is None:
            continue
        doc_id, page_idx = parsed
        if synthetic_score_mode == "normalized" and max_candidate_score > 0:
            synthetic_score = float(float(item["final_score"]) / max_candidate_score)
        else:
            synthetic_score = float(min_original_score - 1e-6 * idx)
        new_rows.append([doc_id, int(page_idx), synthetic_score])

    if append_after_top_k > 0:
        prefix = rows[:append_after_top_k]
        suffix = rows[append_after_top_k:]
        expanded = [*prefix, *new_rows, *suffix]
    else:
        expanded = [*rows, *new_rows]

    stats = {
        "candidate_target_page_count": int(raw_candidate_count),
        "verified_candidate_attempted_count": int(verification_attempted_count),
        "verified_candidate_kept_count": int(len(verified_scores)),
        "verification_rejected_count": int(verification_rejected_count),
        "added_page_count": int(len(new_rows)),
        "mean_verified_score": (
            float(sum(verified_scores) / len(verified_scores)) if verified_scores else None
        ),
        "max_verified_score": max(verified_scores) if verified_scores else None,
    }
    return expanded, stats


def main() -> None:
    args = parse_args()
    payload, rows_by_qid = load_prediction_payload(Path(args.prediction_json))
    target_scores_by_qid = load_external_targets(
        Path(args.external_page_graph_jsonl),
        min_score=float(args.min_score),
        aggregation=str(args.aggregation),
    )
    verifier = None
    if str(args.verification_mode) == "exact_maxsim":
        if not args.verify_query_embedding_dir or not args.verify_page_embedding_dir:
            raise ValueError(
                "--verification-mode=exact_maxsim requires "
                "--verify-query-embedding-dir and --verify-page-embedding-dir."
            )
        verifier = ExactMaxSimVerifier(
            query_embedding_dir=Path(args.verify_query_embedding_dir),
            page_embedding_dir=Path(args.verify_page_embedding_dir),
            query_embedding_key=str(args.verify_query_embedding_key),
            page_embedding_key=str(args.verify_page_embedding_key),
        )

    output_rows: dict[str, dict[str, Any]] = {}
    added_counts: list[int] = []
    candidate_counts: list[int] = []
    verified_attempted_counts: list[int] = []
    verified_kept_counts: list[int] = []
    verified_scores: list[float] = []
    for qid, row in rows_by_qid.items():
        copied = dict(row)
        page_rows = list(copied.get("page_retrieval_results", []))
        expanded, expansion_stats = expanded_rows(
            qid=qid,
            rows=page_rows,
            target_scores=target_scores_by_qid.get(qid, {}),
            max_new_pages=int(args.max_new_pages_per_qid),
            synthetic_score_mode=str(args.synthetic_score_mode),
            append_after_top_k=int(args.append_after_top_k),
            verifier=verifier,
            verification_min_score=float(args.verification_min_score),
            verification_candidate_pool=int(args.verification_candidate_pool),
            verified_score_mode=str(args.verified_score_mode),
        )
        added_count = int(expansion_stats["added_page_count"])
        copied["page_retrieval_results"] = expanded
        copied["faiss_token_neighbor_candidate_expansion"] = {
            "source_external_page_graph_jsonl": str(args.external_page_graph_jsonl),
            "max_new_pages_per_qid": int(args.max_new_pages_per_qid),
            "min_score": float(args.min_score),
            "aggregation": str(args.aggregation),
            "synthetic_score_mode": str(args.synthetic_score_mode),
            "append_after_top_k": int(args.append_after_top_k),
            "verification_mode": str(args.verification_mode),
            "verification_min_score": float(args.verification_min_score),
            "verification_candidate_pool": int(args.verification_candidate_pool),
            "verified_score_mode": str(args.verified_score_mode),
            **expansion_stats,
        }
        output_rows[qid] = copied
        added_counts.append(added_count)
        candidate_counts.append(int(expansion_stats["candidate_target_page_count"]))
        verified_attempted_counts.append(int(expansion_stats["verified_candidate_attempted_count"]))
        verified_kept_counts.append(int(expansion_stats["verified_candidate_kept_count"]))
        if expansion_stats["mean_verified_score"] is not None:
            verified_scores.append(float(expansion_stats["mean_verified_score"]))

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
        "verification_mode": str(args.verification_mode),
        "verification_min_score": float(args.verification_min_score),
        "verification_candidate_pool": int(args.verification_candidate_pool),
        "verified_score_mode": str(args.verified_score_mode),
        "total_added_page_count": int(sum(added_counts)),
        "mean_added_page_count": (
            float(sum(added_counts) / len(added_counts)) if added_counts else 0.0
        ),
        "qid_with_added_page_count": int(sum(1 for count in added_counts if count > 0)),
        "total_candidate_target_page_count": int(sum(candidate_counts)),
        "mean_candidate_target_page_count": (
            float(sum(candidate_counts) / len(candidate_counts)) if candidate_counts else 0.0
        ),
        "total_verified_candidate_attempted_count": int(sum(verified_attempted_counts)),
        "mean_verified_candidate_attempted_count": (
            float(sum(verified_attempted_counts) / len(verified_attempted_counts))
            if verified_attempted_counts
            else 0.0
        ),
        "total_verified_candidate_kept_count": int(sum(verified_kept_counts)),
        "mean_verified_candidate_kept_count": (
            float(sum(verified_kept_counts) / len(verified_kept_counts))
            if verified_kept_counts
            else 0.0
        ),
        "mean_qid_mean_verified_score": (
            float(sum(verified_scores) / len(verified_scores)) if verified_scores else None
        ),
        "missing_query_embedding_count": (
            int(verifier.missing_query_count) if verifier is not None else 0
        ),
        "missing_page_embedding_count": (
            int(verifier.missing_page_count) if verifier is not None else 0
        ),
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
