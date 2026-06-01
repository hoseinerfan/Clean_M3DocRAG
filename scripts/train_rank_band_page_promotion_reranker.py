#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - runtime dependency check
    torch = None
    F = None


DEFAULT_RECALL_KS = [1, 2, 4, 5, 10, 20, 50, 100]
BASE_FEATURE_NAMES = [
    "base_rank_recip",
    "base_rank_log_recip",
    "base_rank_frac",
    "base_norm_score",
    "base_score_gap_to_top",
    "base_score_gap_to_prev",
    "is_anchor_topk",
    "is_candidate_band",
    "rank_band_1_4",
    "rank_band_5_20",
    "rank_band_21_100",
    "rank_band_101_500",
    "rank_band_501_1000",
    "doc_rank_recip",
    "doc_rank_frac",
    "page_rank_in_doc_recip",
    "doc_page_count_log",
    "doc_best_norm_score",
    "doc_in_anchor_topk",
    "page_idx_log",
    "page_idx_recip",
    "is_first_page",
    "source_present_count",
    "source_best_rank_recip",
    "source_best_norm_score",
    "source_mean_rank_recip",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and apply a rank-band page promotion reranker. The model is "
            "trained on gold pages that already appear in a top-k candidate pool and "
            "learns to promote them over hard negatives from top ranks and deeper rank bands."
        )
    )
    parser.add_argument("--train-gold", required=True)
    parser.add_argument("--eval-gold", required=True)
    parser.add_argument("--train-base-pred", required=True, help="Base train prediction, usually dense top-1000.")
    parser.add_argument("--eval-base-pred", required=True, help="Base eval prediction to rerank.")
    parser.add_argument("--train-source", action="append", default=[], help="Optional LABEL=prediction.json train source.")
    parser.add_argument("--eval-source", action="append", default=[], help="Optional LABEL=prediction.json eval source.")
    parser.add_argument("--candidate-top-k", type=int, default=1000)
    parser.add_argument("--anchor-top-k", type=int, default=4)
    parser.add_argument("--promotion-rank-min", type=int, default=5)
    parser.add_argument("--promotion-rank-max", type=int, default=500)
    parser.add_argument(
        "--positive-scope",
        choices=["auto", "page", "doc"],
        default="auto",
        help=(
            "Training target granularity. 'page' marks only exact gold pages positive; "
            "'doc' marks any page from a gold document positive; 'auto' uses page labels "
            "when available and falls back to document labels."
        ),
    )
    parser.add_argument("--negatives-per-band", type=int, default=8)
    parser.add_argument("--max-negatives-per-qid", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pair-batch-size", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--inference-mode",
        choices=["safe_promote", "full_rerank"],
        default="safe_promote",
    )
    parser.add_argument("--max-promotions-per-qid", type=int, default=2)
    parser.add_argument(
        "--promotion-margin",
        type=float,
        default=0.0,
        help="Safe promotion accepts a candidate only when score(candidate)-score(weakest_anchor) >= margin.",
    )
    parser.add_argument("--recall-k", type=int, nargs="+", default=DEFAULT_RECALL_KS)
    parser.add_argument("--output-model-json", required=True)
    parser.add_argument("--output-prediction-json", required=True)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument("--output-table-md", default="")
    parser.add_argument("--output-train-features-jsonl", default="")
    parser.add_argument("--output-eval-features-jsonl", default="")
    parser.add_argument(
        "--output-eval-prior-jsonl",
        default="",
        help=(
            "Optional JSONL page-prior file for graph reranking. Each row contains "
            "qid/page_uid/base_rank plus learned_score and learned_score_norm."
        ),
    )
    return parser.parse_args()


def sanitize_label(label: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", label.strip())
    return text.strip("_") or "source"


def parse_labeled_path(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected LABEL=path, got {spec!r}")
    label, raw_path = spec.split("=", 1)
    label = sanitize_label(label)
    path = Path(raw_path.strip())
    if not label or not str(path):
        raise ValueError(f"Invalid source spec: {spec!r}")
    return label, path


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
    if isinstance(payload, dict) and isinstance(payload.get("predictions"), (dict, list)):
        payload = payload["predictions"]
    if isinstance(payload, dict):
        iterator = payload.items()
    elif isinstance(payload, list):
        iterator = enumerate(payload)
    else:
        raise TypeError(f"Prediction JSON must be object or list: {path}")
    out: dict[str, dict[str, Any]] = {}
    for raw_key, row in iterator:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("qid") or raw_key).strip()
        if qid:
            out[qid] = row
    return out


def page_uid(doc_id: str, page_idx: int) -> str:
    return f"{doc_id}_page{int(page_idx)}"


def parse_page_uid(uid: str) -> tuple[str, int] | None:
    if "_page" not in uid:
        return None
    doc_id, raw_page = uid.rsplit("_page", 1)
    if not doc_id:
        return None
    try:
        return doc_id, int(raw_page)
    except ValueError:
        return None


def parse_prediction_row(row: Any) -> tuple[str, int, float] | None:
    if isinstance(row, (list, tuple)) and len(row) >= 2:
        doc_id = str(row[0]).strip()
        if not doc_id:
            return None
        try:
            page_idx = int(row[1])
        except (TypeError, ValueError):
            return None
        try:
            score = float(row[2]) if len(row) >= 3 and row[2] is not None else 0.0
        except (TypeError, ValueError):
            score = 0.0
        return doc_id, page_idx, score
    if isinstance(row, dict):
        uid = str(row.get("page_uid", "")).strip()
        if uid:
            parsed = parse_page_uid(uid)
            if parsed is None:
                return None
            doc_id, page_idx = parsed
        else:
            doc_id = str(row.get("doc_id", row.get("docid", row.get("document_id", "")))).strip()
            page_value = row.get("page_idx", row.get("page_id", row.get("page")))
            if not doc_id or page_value is None:
                return None
            try:
                page_idx = int(page_value)
            except (TypeError, ValueError):
                return None
        score_value = row.get("score", row.get("retrieval_score", row.get("fused_page_score", 0.0)))
        try:
            score = float(score_value)
        except (TypeError, ValueError):
            score = 0.0
        return doc_id, page_idx, score
    return None


def prediction_rows(row: dict[str, Any] | None) -> list[Any]:
    if not isinstance(row, dict):
        return []
    rows = row.get("page_retrieval_results", row.get("retrieval_results", row.get("results", [])))
    return rows if isinstance(rows, list) else []


def ranked_page_records(row: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for raw in prediction_rows(row):
        parsed = parse_prediction_row(raw)
        if parsed is None:
            continue
        doc_id, page_idx, score = parsed
        uid = page_uid(doc_id, page_idx)
        if uid in seen:
            continue
        seen.add(uid)
        out.append(
            {
                "uid": uid,
                "doc_id": doc_id,
                "page_idx": int(page_idx),
                "raw_score": float(score),
                "base_rank": len(out) + 1,
            }
        )
        if top_k > 0 and len(out) >= top_k:
            break
    return out


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return [1.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def gold_page_uids(row: dict[str, Any]) -> set[str]:
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    uids: set[str] = {
        str(value).strip()
        for value in (metadata.get("gold_page_uids", []) or row.get("gold_page_uids", []) or [])
        if str(value).strip()
    }
    doc_ids = metadata.get("gold_doc_ids") or row.get("gold_doc_ids") or []
    page_ids = metadata.get("gold_page_ids") or row.get("gold_page_ids") or []
    if not isinstance(doc_ids, list):
        doc_ids = [doc_ids]
    if not isinstance(page_ids, list):
        page_ids = [page_ids]
    if len(doc_ids) == 1 and len(page_ids) > 1:
        doc_ids = doc_ids * len(page_ids)
    for doc_id, page_idx in zip(doc_ids, page_ids):
        if doc_id is not None and page_idx is not None:
            uids.add(page_uid(str(doc_id).strip(), int(page_idx)))
    for ctx in row.get("supporting_context", []):
        if not isinstance(ctx, dict):
            continue
        doc_id = str(ctx.get("doc_id", ctx.get("doc_name", ""))).strip()
        page_idx = ctx.get("page_idx", ctx.get("page_id", ctx.get("page")))
        if doc_id and page_idx is not None:
            uids.add(page_uid(doc_id, int(page_idx)))
    return uids


def gold_doc_ids(row: dict[str, Any]) -> set[str]:
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    raw_values = metadata.get("gold_doc_ids") or row.get("gold_doc_ids") or []
    if not isinstance(raw_values, list):
        raw_values = [raw_values]
    docs = {
        str(value).strip()
        for value in raw_values
        if str(value).strip()
    }
    for uid in gold_page_uids(row):
        parsed = parse_page_uid(uid)
        if parsed:
            docs.add(parsed[0])
    for ctx in row.get("supporting_context", []):
        if isinstance(ctx, dict):
            doc_id = str(ctx.get("doc_id", ctx.get("doc_name", ""))).strip()
            if doc_id:
                docs.add(doc_id)
    return docs


def load_gold(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        qid = str(row.get("qid", "")).strip()
        if qid:
            rows[qid] = row
    return rows


def source_stats_for_qid(
    source_rows_by_label: dict[str, dict[str, Any]],
    qid: str,
    *,
    top_k: int,
) -> dict[str, dict[str, dict[str, float]]]:
    stats: dict[str, dict[str, dict[str, float]]] = {}
    for label, pred in source_rows_by_label.items():
        records = ranked_page_records(pred.get(qid), top_k)
        norms = minmax([float(record["raw_score"]) for record in records])
        source_map: dict[str, dict[str, float]] = {}
        for idx, record in enumerate(records):
            source_map[str(record["uid"])] = {
                "rank": float(idx + 1),
                "norm_score": float(norms[idx] if idx < len(norms) else 0.0),
            }
        stats[label] = source_map
    return stats


def rank_band_features(rank: int) -> dict[str, float]:
    return {
        "rank_band_1_4": 1.0 if 1 <= rank <= 4 else 0.0,
        "rank_band_5_20": 1.0 if 5 <= rank <= 20 else 0.0,
        "rank_band_21_100": 1.0 if 21 <= rank <= 100 else 0.0,
        "rank_band_101_500": 1.0 if 101 <= rank <= 500 else 0.0,
        "rank_band_501_1000": 1.0 if rank >= 501 else 0.0,
    }


def build_feature_names(source_labels: list[str]) -> list[str]:
    names = list(BASE_FEATURE_NAMES)
    for label in source_labels:
        names.extend(
            [
                f"{label}_present",
                f"{label}_rank_recip",
                f"{label}_norm_score",
                f"{label}_score_minus_base",
            ]
        )
    return names


def build_candidates_for_qid(
    *,
    base_row: dict[str, Any] | None,
    source_rows_by_label: dict[str, dict[str, Any]],
    source_labels: list[str],
    qid: str,
    candidate_top_k: int,
    anchor_top_k: int,
    promotion_rank_min: int,
) -> list[dict[str, Any]]:
    records = ranked_page_records(base_row, candidate_top_k)
    if not records:
        return []
    norms = minmax([float(record["raw_score"]) for record in records])
    top_norm = norms[0] if norms else 0.0
    source_stats = source_stats_for_qid(source_rows_by_label, qid, top_k=candidate_top_k)

    doc_rank: dict[str, int] = {}
    doc_page_count: Counter[str] = Counter()
    doc_best_norm: dict[str, float] = {}
    page_rank_in_doc_counter: Counter[str] = Counter()
    for idx, record in enumerate(records):
        doc_id = str(record["doc_id"])
        doc_page_count[doc_id] += 1
        doc_best_norm[doc_id] = max(doc_best_norm.get(doc_id, 0.0), float(norms[idx]))
        if doc_id not in doc_rank:
            doc_rank[doc_id] = len(doc_rank) + 1
    anchor_doc_ids = {str(record["doc_id"]) for record in records[:anchor_top_k]}

    candidates = []
    for idx, record in enumerate(records):
        rank = idx + 1
        doc_id = str(record["doc_id"])
        page_rank_in_doc_counter[doc_id] += 1
        norm_score = float(norms[idx])
        prev_norm = float(norms[idx - 1]) if idx > 0 else norm_score
        source_present_count = 0
        source_rank_recips = []
        source_norm_scores = []
        features = {
            "base_rank_recip": 1.0 / float(rank),
            "base_rank_log_recip": 1.0 / math.log2(float(rank) + 1.0),
            "base_rank_frac": float(rank) / float(max(1, candidate_top_k)),
            "base_norm_score": norm_score,
            "base_score_gap_to_top": float(top_norm - norm_score),
            "base_score_gap_to_prev": float(prev_norm - norm_score),
            "is_anchor_topk": 1.0 if rank <= anchor_top_k else 0.0,
            "is_candidate_band": 1.0 if rank >= promotion_rank_min else 0.0,
            "doc_rank_recip": 1.0 / float(doc_rank[doc_id]),
            "doc_rank_frac": float(doc_rank[doc_id]) / float(max(1, len(doc_rank))),
            "page_rank_in_doc_recip": 1.0 / float(page_rank_in_doc_counter[doc_id]),
            "doc_page_count_log": math.log1p(float(doc_page_count[doc_id])),
            "doc_best_norm_score": float(doc_best_norm.get(doc_id, 0.0)),
            "doc_in_anchor_topk": 1.0 if doc_id in anchor_doc_ids else 0.0,
            "page_idx_log": math.log1p(float(record["page_idx"])),
            "page_idx_recip": 1.0 / float(int(record["page_idx"]) + 1),
            "is_first_page": 1.0 if int(record["page_idx"]) == 0 else 0.0,
            **rank_band_features(rank),
        }
        for label in source_labels:
            item = source_stats.get(label, {}).get(str(record["uid"]))
            if item is None:
                present = 0.0
                rank_recip = 0.0
                source_norm = 0.0
            else:
                present = 1.0
                rank_recip = 1.0 / float(item["rank"])
                source_norm = float(item["norm_score"])
                source_present_count += 1
                source_rank_recips.append(rank_recip)
                source_norm_scores.append(source_norm)
            features[f"{label}_present"] = present
            features[f"{label}_rank_recip"] = rank_recip
            features[f"{label}_norm_score"] = source_norm
            features[f"{label}_score_minus_base"] = source_norm - norm_score
        features["source_present_count"] = float(source_present_count)
        features["source_best_rank_recip"] = max(source_rank_recips) if source_rank_recips else 0.0
        features["source_best_norm_score"] = max(source_norm_scores) if source_norm_scores else 0.0
        features["source_mean_rank_recip"] = (
            float(sum(source_rank_recips) / len(source_rank_recips)) if source_rank_recips else 0.0
        )
        candidates.append(
            {
                "qid": qid,
                "uid": str(record["uid"]),
                "doc_id": doc_id,
                "page_idx": int(record["page_idx"]),
                "base_rank": rank,
                "raw_score": float(record["raw_score"]),
                "feature_values": features,
            }
        )
    return candidates


def selected_training_rows(
    candidates: list[dict[str, Any]],
    gold_pages: set[str],
    gold_docs: set[str],
    *,
    positive_scope: str,
    anchor_top_k: int,
    promotion_rank_max: int,
    negatives_per_band: int,
    max_negatives_per_qid: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if positive_scope == "auto":
        resolved_scope = "page" if gold_pages else "doc"
    else:
        resolved_scope = positive_scope
    if resolved_scope == "page":
        positives = [row for row in candidates if row["uid"] in gold_pages]
    elif resolved_scope == "doc":
        positives = [row for row in candidates if str(row["doc_id"]) in gold_docs]
    else:
        raise ValueError(f"Unsupported positive_scope: {positive_scope}")
    if not positives:
        return [], {f"no_positive_in_pool_{resolved_scope}": 1}
    negatives_by_uid: dict[str, dict[str, Any]] = {}

    def add_negatives(lo: int, hi: int, limit: int) -> None:
        count = 0
        for row in candidates:
            rank = int(row["base_rank"])
            is_positive = (
                row["uid"] in gold_pages
                if resolved_scope == "page"
                else str(row["doc_id"]) in gold_docs
            )
            if rank < lo or rank > hi or is_positive:
                continue
            negatives_by_uid.setdefault(str(row["uid"]), row)
            count += 1
            if count >= limit:
                break

    add_negatives(1, anchor_top_k, anchor_top_k)
    add_negatives(anchor_top_k + 1, 20, negatives_per_band)
    add_negatives(21, 100, negatives_per_band)
    add_negatives(101, 500, negatives_per_band)
    add_negatives(501, promotion_rank_max, negatives_per_band)
    negatives = sorted(negatives_by_uid.values(), key=lambda row: int(row["base_rank"]))[:max_negatives_per_qid]
    if not negatives:
        return [], {"no_negative_in_pool": 1}
    rows = []
    for row in positives + negatives:
        copied = dict(row)
        copied["label"] = (
            1.0
            if (row["uid"] in gold_pages if resolved_scope == "page" else str(row["doc_id"]) in gold_docs)
            else 0.0
        )
        copied["positive_scope"] = resolved_scope
        rows.append(copied)
    return rows, {
        "positive_count": len(positives),
        "negative_count": len(negatives),
        f"positive_scope_{resolved_scope}_qid_count": 1,
    }


def feature_matrix(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    means: Any | None = None,
    stds: Any | None = None,
) -> tuple[Any, Any, Any, Any]:
    vectors = [[float(row["feature_values"].get(name, 0.0)) for name in feature_names] for row in rows]
    labels = [float(row.get("label", 0.0)) for row in rows]
    x = torch.tensor(vectors, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    if means is None:
        means = x.mean(dim=0)
    if stds is None:
        stds = x.std(dim=0, unbiased=False)
    stds = torch.where(stds > 1e-6, stds, torch.ones_like(stds))
    return (x - means) / stds, y, means, stds


def model_scores(rows: list[dict[str, Any]], feature_names: list[str], model: dict[str, Any]) -> list[float]:
    means = model["feature_means"]
    stds = model["feature_stds"]
    weights = model["weights"]
    bias = float(model["bias"])
    scores = []
    for row in rows:
        score = bias
        values = row["feature_values"]
        for idx, name in enumerate(feature_names):
            std = float(stds[idx])
            value = float(values.get(name, 0.0))
            centered = value - float(means[idx])
            standardized = centered / std if std > 0 else centered
            score += standardized * float(weights[idx])
        scores.append(float(score))
    return scores


def evaluate_ranking(prediction: dict[str, dict[str, Any]], gold: dict[str, dict[str, Any]], recall_ks: list[int]) -> dict[str, Any]:
    qids = sorted(set(prediction) & set(gold))
    page_hits = {k: 0 for k in recall_ks}
    doc_hits = {k: 0 for k in recall_ks}
    first_page_ranks: dict[str, int | None] = {}
    first_doc_ranks: dict[str, int | None] = {}
    for qid in qids:
        pages = []
        docs = []
        seen_pages = set()
        seen_docs = set()
        for record in ranked_page_records(prediction[qid], max(recall_ks) if recall_ks else 1000):
            uid = str(record["uid"])
            doc_id = str(record["doc_id"])
            if uid not in seen_pages:
                seen_pages.add(uid)
                pages.append(uid)
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                docs.append(doc_id)
        gold_pages = gold_page_uids(gold[qid])
        gold_docs = gold_doc_ids(gold[qid])
        page_rank = first_rank(pages, gold_pages)
        doc_rank = first_rank(docs, gold_docs)
        first_page_ranks[qid] = page_rank
        first_doc_ranks[qid] = doc_rank
        for k in recall_ks:
            page_hits[k] += int(page_rank is not None and page_rank <= k)
            doc_hits[k] += int(doc_rank is not None and doc_rank <= k)
    return {
        "qid_count": len(qids),
        "page_hits": page_hits,
        "doc_hits": doc_hits,
        "first_page_ranks": first_page_ranks,
        "first_doc_ranks": first_doc_ranks,
    }


def first_rank(items: list[str], gold: set[str]) -> int | None:
    for idx, item in enumerate(items, start=1):
        if item in gold:
            return idx
    return None


def compare_hit_movements(base_eval: dict[str, Any], pred_eval: dict[str, Any], hit_k: int) -> dict[str, int]:
    out = Counter()
    for qid, base_rank in base_eval["first_page_ranks"].items():
        new_rank = pred_eval["first_page_ranks"].get(qid)
        base_hit = base_rank is not None and base_rank <= hit_k
        new_hit = new_rank is not None and new_rank <= hit_k
        if not base_hit and new_hit:
            out["recovered"] += 1
        elif base_hit and not new_hit:
            out["lost"] += 1
        elif base_rank is not None and new_rank is not None and new_rank < base_rank:
            out["improved_rank"] += 1
        elif base_rank is not None and new_rank is not None and new_rank > base_rank:
            out["worsened_rank"] += 1
        else:
            out["unchanged"] += 1
    return dict(out)


def rows_to_prediction_rows(rows: list[dict[str, Any]]) -> list[list[Any]]:
    n = len(rows)
    return [[row["doc_id"], int(row["page_idx"]), float(n - idx)] for idx, row in enumerate(rows)]


def apply_reranker_to_qid(
    *,
    qid: str,
    base_row: dict[str, Any],
    source_rows_by_label: dict[str, dict[str, Any]],
    source_labels: list[str],
    feature_names: list[str],
    model: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = build_candidates_for_qid(
        base_row=base_row,
        source_rows_by_label=source_rows_by_label,
        source_labels=source_labels,
        qid=qid,
        candidate_top_k=int(args.candidate_top_k),
        anchor_top_k=int(args.anchor_top_k),
        promotion_rank_min=int(args.promotion_rank_min),
    )
    if not candidates:
        return dict(base_row), {"promotion_count": 0, "candidate_count": 0}
    scores = model_scores(candidates, feature_names, model)
    for row, score in zip(candidates, scores):
        row["promotion_score"] = float(score)

    if args.inference_mode == "full_rerank":
        ranked = sorted(candidates, key=lambda row: (-float(row["promotion_score"]), int(row["base_rank"]), row["uid"]))
        promotion_count = sum(1 for idx, row in enumerate(ranked[: args.anchor_top_k], start=1) if int(row["base_rank"]) > args.anchor_top_k)
    else:
        by_uid = {str(row["uid"]): row for row in candidates}
        selected = [row for row in candidates if int(row["base_rank"]) <= args.anchor_top_k]
        selected_uids = {str(row["uid"]) for row in selected}
        pool = [
            row
            for row in candidates
            if int(row["base_rank"]) >= args.promotion_rank_min
            and int(row["base_rank"]) <= args.promotion_rank_max
            and str(row["uid"]) not in selected_uids
        ]
        pool.sort(key=lambda row: (-float(row["promotion_score"]), int(row["base_rank"]), row["uid"]))
        promotion_count = 0
        for row in pool:
            if promotion_count >= args.max_promotions_per_qid:
                break
            if not selected:
                break
            weakest_idx = min(range(len(selected)), key=lambda idx: float(selected[idx]["promotion_score"]))
            weakest = selected[weakest_idx]
            margin = float(row["promotion_score"]) - float(weakest["promotion_score"])
            if margin < float(args.promotion_margin):
                break
            selected_uids.remove(str(weakest["uid"]))
            selected[weakest_idx] = row
            selected_uids.add(str(row["uid"]))
            promotion_count += 1
        remaining = [by_uid[str(row["uid"])] for row in candidates if str(row["uid"]) not in selected_uids]
        ranked = [*selected, *remaining]

    output = dict(base_row)
    output["qid"] = qid
    output["page_retrieval_results"] = rows_to_prediction_rows(ranked)
    output["rank_band_page_promotion"] = {
        "inference_mode": args.inference_mode,
        "promotion_count": int(promotion_count),
        "candidate_count": int(len(candidates)),
        "anchor_top_k": int(args.anchor_top_k),
        "promotion_rank_min": int(args.promotion_rank_min),
        "promotion_rank_max": int(args.promotion_rank_max),
        "promotion_margin": float(args.promotion_margin),
    }
    return output, output["rank_band_page_promotion"]


def format_metric_row(label: str, metrics: dict[str, Any], recall_ks: list[int]) -> dict[str, Any]:
    n = int(metrics["qid_count"])
    row = {"label": label, "qid_count": n}
    for k in recall_ks:
        row[f"page@{k}"] = float(metrics["page_hits"][k] / n) if n else 0.0
        row[f"doc@{k}"] = float(metrics["doc_hits"][k] / n) if n else 0.0
    return row


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_feature_jsonl(path: str, rows_by_qid: dict[str, list[dict[str, Any]]]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for qid in sorted(rows_by_qid):
            for row in rows_by_qid[qid]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_prior_jsonl(path: str, rows_by_qid: dict[str, list[dict[str, Any]]]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for qid in sorted(rows_by_qid):
            rows = rows_by_qid[qid]
            scores = [float(row["learned_score"]) for row in rows]
            lo = min(scores) if scores else 0.0
            hi = max(scores) if scores else 0.0
            for row in rows:
                score = float(row["learned_score"])
                norm = (score - lo) / (hi - lo) if hi > lo else 0.0
                item = {
                    "qid": qid,
                    "page_uid": row["uid"],
                    "doc_id": row["doc_id"],
                    "page_idx": int(row["page_idx"]),
                    "base_rank": int(row["base_rank"]),
                    "base_score": float(row["raw_score"]),
                    "learned_score": score,
                    "learned_score_norm": float(norm),
                }
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if torch is None or F is None:
        raise ImportError("This script requires torch in the active environment.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_sources = dict(parse_labeled_path(spec) for spec in args.train_source)
    eval_sources = dict(parse_labeled_path(spec) for spec in args.eval_source)
    if set(train_sources) != set(eval_sources):
        raise ValueError(
            f"Train/eval source labels differ: train={sorted(train_sources)} eval={sorted(eval_sources)}"
        )
    source_labels = sorted(train_sources)
    feature_names = build_feature_names(source_labels)

    train_gold = load_gold(Path(args.train_gold))
    eval_gold = load_gold(Path(args.eval_gold))
    train_base = load_prediction(Path(args.train_base_pred))
    eval_base = load_prediction(Path(args.eval_base_pred))
    train_source_preds = {label: load_prediction(path) for label, path in train_sources.items()}
    eval_source_preds = {label: load_prediction(path) for label, path in eval_sources.items()}

    train_rows_by_qid: dict[str, list[dict[str, Any]]] = {}
    train_stats = Counter()
    for qid in sorted(set(train_gold) & set(train_base)):
        candidates = build_candidates_for_qid(
            base_row=train_base.get(qid),
            source_rows_by_label=train_source_preds,
            source_labels=source_labels,
            qid=qid,
            candidate_top_k=int(args.candidate_top_k),
            anchor_top_k=int(args.anchor_top_k),
            promotion_rank_min=int(args.promotion_rank_min),
        )
        rows, stats = selected_training_rows(
            candidates,
            gold_page_uids(train_gold[qid]),
            gold_doc_ids(train_gold[qid]),
            positive_scope=args.positive_scope,
            anchor_top_k=int(args.anchor_top_k),
            promotion_rank_max=int(args.promotion_rank_max),
            negatives_per_band=int(args.negatives_per_band),
            max_negatives_per_qid=int(args.max_negatives_per_qid),
        )
        train_stats.update(stats)
        if rows:
            train_rows_by_qid[qid] = rows
    if not train_rows_by_qid:
        raise ValueError("No train qids produced positive/negative promotion pairs.")
    write_feature_jsonl(args.output_train_features_jsonl, train_rows_by_qid)

    all_train_rows = []
    train_row_offsets: dict[str, tuple[int, int]] = {}
    for qid, rows in train_rows_by_qid.items():
        train_row_offsets[qid] = (len(all_train_rows), len(rows))
        all_train_rows.extend(rows)
    x_all, _y_all, means, stds = feature_matrix(all_train_rows, feature_names)
    pair_pos = []
    pair_neg = []
    for qid, rows in train_rows_by_qid.items():
        offset, _n = train_row_offsets[qid]
        positives = [offset + idx for idx, row in enumerate(rows) if float(row.get("label", 0.0)) > 0.5]
        negatives = [offset + idx for idx, row in enumerate(rows) if float(row.get("label", 0.0)) <= 0.5]
        for pos_idx in positives:
            for neg_idx in negatives:
                pair_pos.append(pos_idx)
                pair_neg.append(neg_idx)
    if not pair_pos:
        raise ValueError("No pairwise train batches were formed.")
    pair_pos_t = torch.tensor(pair_pos, dtype=torch.long)
    pair_neg_t = torch.tensor(pair_neg, dtype=torch.long)
    weights = torch.nn.Parameter(torch.zeros(len(feature_names), dtype=torch.float32))
    bias = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
    optimizer = torch.optim.Adam([weights, bias], lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    history = []
    best_state = None
    best_top4 = -1
    pair_batch_size = max(1, int(args.pair_batch_size))

    for epoch in range(int(args.epochs)):
        order = torch.randperm(pair_pos_t.numel())
        epoch_loss = 0.0
        seen_pairs = 0
        for start in range(0, int(order.numel()), pair_batch_size):
            batch = order[start : start + pair_batch_size]
            optimizer.zero_grad()
            pos_scores = x_all[pair_pos_t[batch]] @ weights + bias
            neg_scores = x_all[pair_neg_t[batch]] @ weights + bias
            loss = F.softplus(-(pos_scores - neg_scores)).mean()
            loss.backward()
            optimizer.step()
            batch_size = int(batch.numel())
            epoch_loss += float(loss.item()) * batch_size
            seen_pairs += batch_size
        loss_value = epoch_loss / float(max(1, seen_pairs))

        with torch.no_grad():
            train_scores = x_all @ weights + bias
            top4_hits = 0
            for qid, rows in train_rows_by_qid.items():
                offset, n = train_row_offsets[qid]
                local_scores = train_scores[offset : offset + n]
                ranked = sorted(range(n), key=lambda idx: (-float(local_scores[idx]), int(rows[idx]["base_rank"])))
                if any(float(rows[idx]["label"]) > 0.5 for idx in ranked[: int(args.anchor_top_k)]):
                    top4_hits += 1
            history.append({"epoch": epoch + 1, "loss": float(loss_value), "train_sample_top4_hits": top4_hits})
            if top4_hits > best_top4:
                best_top4 = top4_hits
                best_state = {
                    "weights": [float(v) for v in weights.detach().tolist()],
                    "bias": float(bias.detach().item()),
                    "best_epoch": epoch + 1,
                }

    if best_state is None:
        best_state = {
            "weights": [float(v) for v in weights.detach().tolist()],
            "bias": float(bias.detach().item()),
            "best_epoch": int(args.epochs),
        }
    model = {
        "model_type": "rank_band_page_promotion_linear_pairwise",
        "feature_names": feature_names,
        "feature_means": [float(v) for v in means.tolist()],
        "feature_stds": [float(v) for v in stds.tolist()],
        "weights": best_state["weights"],
        "bias": best_state["bias"],
        "source_labels": source_labels,
        "training_args": vars(args),
        "train_pair_stats": dict(train_stats),
        "train_pair_count": int(len(pair_pos)),
        "best_epoch": best_state["best_epoch"],
    }

    output_prediction: dict[str, dict[str, Any]] = {}
    promotion_stats = Counter()
    eval_feature_rows: dict[str, list[dict[str, Any]]] = {}
    eval_prior_rows: dict[str, list[dict[str, Any]]] = {}
    for qid in sorted(eval_base):
        reranked, stats = apply_reranker_to_qid(
            qid=qid,
            base_row=eval_base[qid],
            source_rows_by_label=eval_source_preds,
            source_labels=source_labels,
            feature_names=feature_names,
            model=model,
            args=args,
        )
        output_prediction[qid] = reranked
        promotion_stats["qid_count"] += 1
        promotion_stats["promotion_count"] += int(stats.get("promotion_count", 0))
        promotion_stats["qid_with_promotion_count"] += int(int(stats.get("promotion_count", 0)) > 0)
        if args.output_eval_features_jsonl or args.output_eval_prior_jsonl:
            candidates = build_candidates_for_qid(
                base_row=eval_base.get(qid),
                source_rows_by_label=eval_source_preds,
                source_labels=source_labels,
                qid=qid,
                candidate_top_k=int(args.candidate_top_k),
                anchor_top_k=int(args.anchor_top_k),
                promotion_rank_min=int(args.promotion_rank_min),
            )
            scores = model_scores(candidates, feature_names, model)
            for row, score in zip(candidates, scores):
                row["learned_score"] = float(score)
            if args.output_eval_features_jsonl:
                eval_feature_rows[qid] = candidates[: max(20, int(args.anchor_top_k))]
            if args.output_eval_prior_jsonl:
                eval_prior_rows[qid] = candidates
    write_feature_jsonl(args.output_eval_features_jsonl, eval_feature_rows)
    write_prior_jsonl(args.output_eval_prior_jsonl, eval_prior_rows)

    base_eval = evaluate_ranking(eval_base, eval_gold, args.recall_k)
    rerank_eval = evaluate_ranking(output_prediction, eval_gold, args.recall_k)
    movement = compare_hit_movements(base_eval, rerank_eval, int(args.anchor_top_k))
    metric_rows = [
        format_metric_row("base", base_eval, args.recall_k),
        format_metric_row("rank_band_promotion", rerank_eval, args.recall_k),
    ]
    columns = ["label", "qid_count"] + [f"page@{k}" for k in args.recall_k] + [f"doc@{k}" for k in args.recall_k]

    model_path = Path(args.output_model_json)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")

    pred_path = Path(args.output_prediction_json)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.write_text(json.dumps(output_prediction, indent=2) + "\n", encoding="utf-8")

    summary = {
        "model_path": str(model_path),
        "prediction_path": str(pred_path),
        "feature_names": feature_names,
        "source_labels": source_labels,
        "positive_scope": args.positive_scope,
        "train_qid_count": len(train_rows_by_qid),
        "train_row_count": len(all_train_rows),
        "train_pair_count": int(len(pair_pos)),
        "train_pair_stats": dict(train_stats),
        "promotion_stats": dict(promotion_stats),
        "movement_vs_base": movement,
        "base_eval": {k: v for k, v in base_eval.items() if not k.startswith("first_")},
        "rerank_eval": {k: v for k, v in rerank_eval.items() if not k.startswith("first_")},
        "metric_rows": metric_rows,
        "history_tail": history[-10:],
    }
    summary_path = Path(args.output_summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.output_table_md:
        table_path = Path(args.output_table_md)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table_path.write_text(markdown_table(metric_rows, columns), encoding="utf-8")

    print(f"saved_model={model_path}")
    print(f"saved_prediction={pred_path}")
    print(f"saved_summary={summary_path}")
    if args.output_table_md:
        print(f"saved_table={args.output_table_md}")
    if args.output_eval_prior_jsonl:
        print(f"saved_eval_prior={args.output_eval_prior_jsonl}")
    print(f"train_qid_count={len(train_rows_by_qid)}")
    print(f"train_row_count={len(all_train_rows)}")
    print(f"movement_vs_base={movement}")
    for row in metric_rows:
        print(row)


if __name__ == "__main__":
    main()
