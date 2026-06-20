#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_RECALL_KS = [1, 2, 4, 5, 10, 20, 50, 100]

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "how",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "there",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "with",
}

FEATURE_NAMES = [
    "base_rank_recip",
    "base_rank_log_recip",
    "base_rank_frac",
    "base_norm_score",
    "base_score_gap_to_top",
    "doc_rank_recip",
    "page_rank_in_doc_recip",
    "doc_page_count_log",
    "page_idx_log",
    "page_idx_recip",
    "is_first_page",
    "source_present_count",
    "source_best_rank_recip",
    "source_best_norm_score",
    "source_mean_rank_recip",
    "page_token_count_log",
    "question_token_recall",
    "question_token_precision",
    "question_token_jaccard",
    "question_overlap_count_log",
    "anchor_token_recall",
    "anchor_token_count_log",
    "number_token_recall",
    "number_token_count_log",
    "phrase_match_count_log",
    "phrase_match_fraction",
    "bigram_match_fraction",
    "trigram_match_fraction",
    "longest_question_ngram_match",
    "exact_question_substring",
]

RANK_FEATURES = [
    "base_rank_recip",
    "base_rank_log_recip",
    "base_rank_frac",
    "base_norm_score",
    "base_score_gap_to_top",
]

STRUCTURE_FEATURES = [
    "doc_rank_recip",
    "page_rank_in_doc_recip",
    "doc_page_count_log",
    "page_idx_log",
    "page_idx_recip",
    "is_first_page",
]

SOURCE_FEATURES = [
    "source_present_count",
    "source_best_rank_recip",
    "source_best_norm_score",
    "source_mean_rank_recip",
]

CONTENT_FEATURES = [
    "page_token_count_log",
    "question_token_recall",
    "question_token_precision",
    "question_token_jaccard",
    "question_overlap_count_log",
    "anchor_token_recall",
    "anchor_token_count_log",
    "number_token_recall",
    "number_token_count_log",
    "phrase_match_count_log",
    "phrase_match_fraction",
    "bigram_match_fraction",
    "trigram_match_fraction",
    "longest_question_ngram_match",
    "exact_question_substring",
]

FEATURE_SET_NAMES = [
    "all",
    "rank_only",
    "rank_source",
    "rank_structure",
    "rank_source_structure",
    "content_only",
    "no_content",
    "no_source",
    "no_structure",
]

QUERY_ALPHA_FEATURE_NAMES = [
    "base_confidence",
    "candidate_count_log",
    "base_top_score_gap",
    "base_top5_doc_concentration",
    "model_top_score",
    "model_top2_margin",
    "model_top4_mean",
    "model_top_window_mean",
    "model_top_window_std",
    "model_entropy_top_window",
    "model_top_base_rank_recip",
    "model_top_base_rank_frac",
    "model_top_in_base_top4",
    "model_top_in_base_top10",
    "model_base_disagreement",
    "model_top_source_fraction",
    "model_top_window_source_fraction",
]

ALPHA_UTILITY_FEATURE_NAMES = (
    list(QUERY_ALPHA_FEATURE_NAMES)
    + ["candidate_alpha", "candidate_alpha_squared"]
    + [f"{name}_x_alpha" for name in QUERY_ALPHA_FEATURE_NAMES]
)


def dedupe_feature_names(names: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        if name not in FEATURE_NAMES:
            raise ValueError(f"Unknown feature name in feature set: {name}")
        seen.add(name)
        out.append(name)
    return out


def resolve_feature_names(feature_set: str) -> list[str]:
    key = str(feature_set or "all").strip().lower()
    if key == "all":
        return list(FEATURE_NAMES)
    if key == "rank_only":
        return list(RANK_FEATURES)
    if key == "rank_source":
        return dedupe_feature_names(RANK_FEATURES + SOURCE_FEATURES)
    if key == "rank_structure":
        return dedupe_feature_names(RANK_FEATURES + STRUCTURE_FEATURES)
    if key == "rank_source_structure":
        return dedupe_feature_names(RANK_FEATURES + SOURCE_FEATURES + STRUCTURE_FEATURES)
    if key == "content_only":
        return list(CONTENT_FEATURES)
    if key == "no_content":
        return [name for name in FEATURE_NAMES if name not in set(CONTENT_FEATURES)]
    if key == "no_source":
        return [name for name in FEATURE_NAMES if name not in set(SOURCE_FEATURES)]
    if key == "no_structure":
        return [name for name in FEATURE_NAMES if name not in set(STRUCTURE_FEATURES)]
    raise ValueError(f"Unknown feature_set {feature_set!r}; expected one of {', '.join(FEATURE_SET_NAMES)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a lightweight content-aware page reranker from MMQA pseudo page labels. "
            "Features combine retrieval rank/score with question-page lexical evidence from "
            "exported PDF page text."
        )
    )
    parser.add_argument("--train-gold", required=True)
    parser.add_argument("--eval-gold", required=True)
    parser.add_argument("--train-base-pred", required=True)
    parser.add_argument("--eval-base-pred", required=True)
    parser.add_argument("--train-page-text-jsonl", required=True)
    parser.add_argument("--eval-page-text-jsonl", required=True)
    parser.add_argument("--train-source", action="append", default=[], help="Optional LABEL=prediction.json")
    parser.add_argument("--eval-source", action="append", default=[], help="Optional LABEL=prediction.json")
    parser.add_argument(
        "--feature-set",
        choices=FEATURE_SET_NAMES,
        default="all",
        help="Feature group used for final-method ablation.",
    )
    parser.add_argument("--candidate-top-k", type=int, default=1000)
    parser.add_argument("--negatives-per-band", type=int, default=10)
    parser.add_argument("--max-negatives-per-qid", type=int, default=64)
    parser.add_argument(
        "--respect-pseudo-supervision-tiers",
        action="store_true",
        help=(
            "Honor evidence-unit-aware pseudo-label metadata. QIDs marked exclude_unlabeled "
            "are skipped, *_positive_only tiers contribute positives without same-query "
            "negatives, and non_support_docs_only negative scopes avoid support-doc pages."
        ),
    )
    parser.add_argument(
        "--pseudo-supervision-weighting",
        action="store_true",
        help=(
            "Use confidence-aware row weights from pseudo-label metadata. Direct complete "
            "positives keep full weight; visual-proxy, hybrid, and partial positives are "
            "downweighted; optional safe support-doc negatives are low-weight."
        ),
    )
    parser.add_argument("--hybrid-positive-weight", type=float, default=0.80)
    parser.add_argument("--visual-proxy-positive-weight", type=float, default=0.60)
    parser.add_argument("--partial-positive-weight", type=float, default=0.50)
    parser.add_argument(
        "--include-safe-support-doc-negatives",
        action="store_true",
        help=(
            "When respecting pseudo supervision tiers, add a small number of low-weight "
            "support-document negatives that are not pseudo-positive and are not adjacent "
            "to a pseudo-positive page in the same document."
        ),
    )
    parser.add_argument("--safe-support-doc-negative-weight", type=float, default=0.25)
    parser.add_argument("--max-safe-support-doc-negatives-per-qid", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--positive-weight-cap", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--inference-mode",
        choices=["blend_rerank", "full_rerank", "safe_promote", "doc_head_blend", "doc_slot_blend"],
        default="blend_rerank",
    )
    parser.add_argument(
        "--blend-alpha",
        type=float,
        default=0.30,
        help="Model-score weight for blend_rerank. 0 keeps base order; 1 uses model order.",
    )
    parser.add_argument(
        "--auto-tune-blend-alpha",
        action="store_true",
        help=(
            "Select blend_alpha on a held-out slice of the training qids instead of using "
            "--blend-alpha directly. This avoids tuning on dev."
        ),
    )
    parser.add_argument(
        "--query-adaptive-alpha",
        action="store_true",
        help=(
            "After global alpha tuning, also tune one blend alpha per held-out query-confidence "
            "bucket. At inference time each query gets the bucket-specific alpha."
        ),
    )
    parser.add_argument(
        "--learned-query-alpha",
        action="store_true",
        help=(
            "Train a small held-out ridge regressor that predicts the blend alpha per query "
            "from base-ranking confidence, model confidence, source agreement, and model/base "
            "disagreement features. Requires --auto-tune-blend-alpha."
        ),
    )
    parser.add_argument(
        "--learned-alpha-action",
        action="store_true",
        help=(
            "Train a held-out multinomial action selector over the alpha grid. The selector "
            "learns which discrete alpha action gives the best pseudo-page rank for each "
            "query, instead of regressing alpha as a continuous value. Requires "
            "--auto-tune-blend-alpha."
        ),
    )
    parser.add_argument(
        "--learned-alpha-utility-gate",
        action="store_true",
        help=(
            "Train a held-out source-only utility gate over (query confidence features, alpha). "
            "The gate predicts expected hit@k gain/loss and chooses a nonzero alpha only when "
            "the predicted utility is safely positive. Requires --auto-tune-blend-alpha."
        ),
    )
    parser.add_argument(
        "--base-aware-alpha-utility-gate",
        action="store_true",
        help=(
            "Train a safer held-out utility gate that separately predicts recovery and loss "
            "risk for each alpha. It chooses a nonzero alpha only when predicted recovery "
            "outweighs predicted base-ranking damage. Requires --auto-tune-blend-alpha."
        ),
    )
    parser.add_argument(
        "--query-alpha-bins",
        type=int,
        default=3,
        help="Number of query-confidence buckets used by --query-adaptive-alpha.",
    )
    parser.add_argument(
        "--query-alpha-feature-top-k",
        type=int,
        default=20,
        help="Top base-ranked candidates used to aggregate query-level features for --learned-query-alpha.",
    )
    parser.add_argument(
        "--query-alpha-ridge",
        type=float,
        default=1.0,
        help="Ridge regularization strength for --learned-query-alpha.",
    )
    parser.add_argument(
        "--query-alpha-continuous",
        action="store_true",
        help="Use the continuous regressor output instead of snapping --learned-query-alpha to the alpha grid.",
    )
    parser.add_argument(
        "--query-alpha-action-epochs",
        type=int,
        default=200,
        help="Training epochs for --learned-alpha-action multinomial selector.",
    )
    parser.add_argument(
        "--query-alpha-action-learning-rate",
        type=float,
        default=0.05,
        help="Learning rate for --learned-alpha-action multinomial selector.",
    )
    parser.add_argument(
        "--query-alpha-action-weight-decay",
        type=float,
        default=1e-3,
        help="L2 regularization for --learned-alpha-action multinomial selector.",
    )
    parser.add_argument(
        "--alpha-utility-ridge",
        type=float,
        default=1.0,
        help="Ridge regularization strength for --learned-alpha-utility-gate.",
    )
    parser.add_argument(
        "--alpha-utility-threshold-grid",
        default="0.00,0.01,0.02,0.05,0.10",
        help=(
            "Comma-separated predicted-utility thresholds calibrated on source held-out qids "
            "for --learned-alpha-utility-gate. Include 0.00 for a pure positive-utility gate."
        ),
    )
    parser.add_argument(
        "--alpha-utility-risk-penalty-grid",
        default="1.00,1.50,2.00,3.00",
        help=(
            "Comma-separated loss-risk penalties calibrated on source held-out qids for "
            "--base-aware-alpha-utility-gate."
        ),
    )
    parser.add_argument(
        "--tune-fraction",
        type=float,
        default=0.20,
        help="Fraction of training qids held out for auto blend-alpha tuning.",
    )
    parser.add_argument(
        "--tune-blend-alpha-grid",
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50",
        help="Comma-separated alpha candidates used by --auto-tune-blend-alpha.",
    )
    parser.add_argument(
        "--tune-hit-k",
        type=int,
        default=4,
        help="Pseudo-page recall cutoff optimized by --auto-tune-blend-alpha.",
    )
    parser.add_argument(
        "--skip-retrain-after-tuning",
        action="store_true",
        help="Keep the fit-split model after tuning instead of retraining on all train qids.",
    )
    parser.add_argument("--anchor-top-k", type=int, default=4)
    parser.add_argument("--promotion-rank-min", type=int, default=5)
    parser.add_argument("--promotion-rank-max", type=int, default=200)
    parser.add_argument("--max-promotions-per-qid", type=int, default=2)
    parser.add_argument("--promotion-margin", type=float, default=0.05)
    parser.add_argument("--recall-k", type=int, nargs="+", default=DEFAULT_RECALL_KS)
    parser.add_argument("--output-model-json", required=True)
    parser.add_argument("--output-prediction-json", required=True)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument("--output-table-md", default="")
    parser.add_argument("--output-eval-prior-jsonl", default="")
    parser.add_argument(
        "--restrict-eval-to-gold-qids",
        action="store_true",
        help=(
            "Only apply the trained reranker to qids present in --eval-gold. "
            "Useful for out-of-fold train prediction generation."
        ),
    )
    return parser.parse_args()


def sanitize_label(label: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", label.strip())
    return text.strip("_") or "source"


def parse_labeled_path(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected LABEL=path source spec, got {spec!r}")
    label, path = spec.split("=", 1)
    return sanitize_label(label), Path(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_gold(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        qid = str(row.get("qid", "")).strip()
        if qid:
            out[qid] = row
    return out


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
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
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
                "score": float(score),
                "raw": raw,
                "base_rank": len(out) + 1,
            }
        )
        if len(out) >= int(top_k):
            break
    return out


def normalize_scores(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {}
    values = [float(row["score"]) for row in records]
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        return {row["uid"]: 1.0 for row in records}
    return {row["uid"]: (float(row["score"]) - lo) / (hi - lo) for row in records}


def source_maps(pred: dict[str, dict[str, Any]], top_k: int) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for qid, row in pred.items():
        records = ranked_page_records(row, top_k)
        norm_scores = normalize_scores(records)
        out[qid] = {
            record["uid"]: {
                "rank": float(idx),
                "norm_score": float(norm_scores.get(record["uid"], 0.0)),
            }
            for idx, record in enumerate(records, start=1)
        }
    return out


def normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize_text(text))


def page_text(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("text", "page_text", "ocr_text", "markdown", "vlm_text", "content"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)
    return "\n".join(parts)


def parse_page_idx(row: dict[str, Any]) -> int:
    for key in ("page_idx", "page_id", "page"):
        if row.get(key) is not None:
            return int(row[key])
    uid = str(row.get("page_uid", ""))
    if "_page" in uid:
        return int(uid.rsplit("_page", 1)[1])
    raise ValueError(f"Page text row missing page index: {row}")


def load_page_features(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        doc_id = str(row.get("doc_id", "")).strip()
        if not doc_id:
            uid = str(row.get("page_uid", ""))
            if "_page" in uid:
                doc_id = uid.rsplit("_page", 1)[0]
        if not doc_id:
            continue
        page_idx = parse_page_idx(row)
        uid = str(row.get("page_uid") or page_uid(doc_id, page_idx))
        text = page_text(row)
        tokens = set(tokenize(text))
        out[uid] = {
            "page_uid": uid,
            "doc_id": doc_id,
            "page_idx": int(page_idx),
            "norm_text": normalize_text(text),
            "tokens": tokens,
            "token_count": len(tokens),
        }
    return out


def question_profile(row: dict[str, Any]) -> dict[str, Any]:
    question = str(row.get("question", "") or "")
    tokens = [tok for tok in tokenize(question) if tok not in STOPWORDS]
    token_set = set(tokens)
    anchor_tokens = {tok for tok in token_set if len(tok) >= 4 or re.fullmatch(r"\d{2,4}", tok)}
    number_tokens = {tok for tok in token_set if re.fullmatch(r"\d+(?:\.\d+)?", tok)}
    phrases = set()
    for quoted in re.findall(r'"([^"]+)"|\'([^\']+)\'', question):
        phrase = next((part for part in quoted if part), "")
        if phrase:
            phrases.add(normalize_text(phrase))
    for phrase in re.findall(r"\b(?:[A-Z][A-Za-z0-9'.-]+(?:\s+|$)){2,}", question):
        normalized = normalize_text(phrase)
        if normalized and len(normalized.split()) >= 2:
            phrases.add(normalized)
    return {
        "question": question,
        "norm_question": normalize_text(question),
        "tokens": token_set,
        "anchor_tokens": anchor_tokens,
        "number_tokens": number_tokens,
        "phrases": {phrase for phrase in phrases if phrase},
        "bigrams": set(ngrams(tokens, 2)),
        "trigrams": set(ngrams(tokens, 3)),
    }


def ngrams(tokens: list[str], n: int) -> list[str]:
    if len(tokens) < n:
        return []
    return [" ".join(tokens[idx : idx + n]) for idx in range(len(tokens) - n + 1)]


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def doc_rank_maps(records: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    doc_rank: dict[str, int] = {}
    doc_page_rank: dict[str, int] = {}
    doc_page_count: Counter[str] = Counter()
    next_doc_rank = 1
    for record in records:
        doc_id = str(record["doc_id"])
        if doc_id not in doc_rank:
            doc_rank[doc_id] = next_doc_rank
            next_doc_rank += 1
        doc_page_count[doc_id] += 1
        doc_page_rank[record["uid"]] = doc_page_count[doc_id]
    return doc_rank, doc_page_rank, dict(doc_page_count)


def content_features(profile: dict[str, Any], page: dict[str, Any] | None) -> dict[str, float]:
    if page is None:
        return {
            "page_token_count_log": 0.0,
            "question_token_recall": 0.0,
            "question_token_precision": 0.0,
            "question_token_jaccard": 0.0,
            "question_overlap_count_log": 0.0,
            "anchor_token_recall": 0.0,
            "anchor_token_count_log": 0.0,
            "number_token_recall": 0.0,
            "number_token_count_log": 0.0,
            "phrase_match_count_log": 0.0,
            "phrase_match_fraction": 0.0,
            "bigram_match_fraction": 0.0,
            "trigram_match_fraction": 0.0,
            "longest_question_ngram_match": 0.0,
            "exact_question_substring": 0.0,
        }
    page_tokens = page["tokens"]
    norm_text = str(page["norm_text"])
    q_tokens = profile["tokens"]
    anchor_tokens = profile["anchor_tokens"]
    number_tokens = profile["number_tokens"]
    overlap = q_tokens & page_tokens
    anchor_overlap = anchor_tokens & page_tokens
    number_overlap = number_tokens & page_tokens
    phrase_matches = [phrase for phrase in profile["phrases"] if f" {phrase} " in f" {norm_text} "]
    bigram_matches = [gram for gram in profile["bigrams"] if f" {gram} " in f" {norm_text} "]
    trigram_matches = [gram for gram in profile["trigrams"] if f" {gram} " in f" {norm_text} "]
    longest_ngram = 0
    for n in range(5, 1, -1):
        grams = ngrams([tok for tok in tokenize(profile["question"]) if tok not in STOPWORDS], n)
        if any(f" {gram} " in f" {norm_text} " for gram in grams):
            longest_ngram = n
            break
    norm_question = str(profile["norm_question"])
    exact_question = bool(norm_question and len(norm_question) >= 16 and norm_question in norm_text)
    return {
        "page_token_count_log": math.log1p(float(page.get("token_count", 0))),
        "question_token_recall": safe_div(len(overlap), len(q_tokens)),
        "question_token_precision": safe_div(len(overlap), len(page_tokens)),
        "question_token_jaccard": safe_div(len(overlap), len(q_tokens | page_tokens)),
        "question_overlap_count_log": math.log1p(float(len(overlap))),
        "anchor_token_recall": safe_div(len(anchor_overlap), len(anchor_tokens)),
        "anchor_token_count_log": math.log1p(float(len(anchor_overlap))),
        "number_token_recall": safe_div(len(number_overlap), len(number_tokens)),
        "number_token_count_log": math.log1p(float(len(number_overlap))),
        "phrase_match_count_log": math.log1p(float(len(phrase_matches))),
        "phrase_match_fraction": safe_div(len(phrase_matches), len(profile["phrases"])),
        "bigram_match_fraction": safe_div(len(bigram_matches), len(profile["bigrams"])),
        "trigram_match_fraction": safe_div(len(trigram_matches), len(profile["trigrams"])),
        "longest_question_ngram_match": float(longest_ngram) / 5.0,
        "exact_question_substring": 1.0 if exact_question else 0.0,
    }


def feature_vector(
    record: dict[str, Any],
    *,
    records: list[dict[str, Any]],
    base_norm_scores: dict[str, float],
    doc_rank: dict[str, int],
    doc_page_rank: dict[str, int],
    doc_page_count: dict[str, int],
    question: dict[str, Any],
    page_features: dict[str, dict[str, Any]],
    source_maps_by_label: dict[str, dict[str, dict[str, float]]],
    qid: str,
    feature_names: list[str] | None = None,
) -> list[float]:
    rank = int(record["base_rank"])
    top_score = float(records[0]["score"]) if records else 0.0
    score = float(record["score"])
    uid = str(record["uid"])
    doc_id = str(record["doc_id"])
    source_hits = []
    for source_map in source_maps_by_label.values():
        hit = source_map.get(qid, {}).get(uid)
        if hit:
            source_hits.append(hit)
    source_ranks = [float(hit["rank"]) for hit in source_hits]
    source_scores = [float(hit["norm_score"]) for hit in source_hits]
    base = {
        "base_rank_recip": 1.0 / float(rank),
        "base_rank_log_recip": 1.0 / math.log2(float(rank) + 1.0),
        "base_rank_frac": 1.0 - min(float(rank), float(len(records))) / max(float(len(records)), 1.0),
        "base_norm_score": float(base_norm_scores.get(uid, 0.0)),
        "base_score_gap_to_top": max(0.0, top_score - score),
        "doc_rank_recip": 1.0 / float(doc_rank.get(doc_id, len(doc_rank) + 1)),
        "page_rank_in_doc_recip": 1.0 / float(doc_page_rank.get(uid, 1)),
        "doc_page_count_log": math.log1p(float(doc_page_count.get(doc_id, 1))),
        "page_idx_log": math.log1p(float(record["page_idx"])),
        "page_idx_recip": 1.0 / float(int(record["page_idx"]) + 1),
        "is_first_page": 1.0 if int(record["page_idx"]) == 0 else 0.0,
        "source_present_count": float(len(source_hits)),
        "source_best_rank_recip": 0.0 if not source_ranks else 1.0 / min(source_ranks),
        "source_best_norm_score": 0.0 if not source_scores else max(source_scores),
        "source_mean_rank_recip": 0.0
        if not source_ranks
        else float(sum(1.0 / rank_value for rank_value in source_ranks)) / float(len(source_ranks)),
    }
    base.update(content_features(question, page_features.get(uid)))
    selected = feature_names or FEATURE_NAMES
    return [float(base[name]) for name in selected]


def gold_page_uids(row: dict[str, Any]) -> set[str]:
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    values = metadata.get("gold_page_uids") or row.get("gold_page_uids") or []
    return {str(value).strip() for value in values if str(value).strip()}


def gold_doc_ids(row: dict[str, Any]) -> set[str]:
    docs = set()
    for uid in gold_page_uids(row):
        parsed = parse_page_uid(uid)
        if parsed:
            docs.add(parsed[0])
    if docs:
        return docs
    for ctx in row.get("supporting_context", []):
        if isinstance(ctx, dict) and ctx.get("doc_id"):
            docs.add(str(ctx["doc_id"]).strip())
    return docs


def support_doc_ids(row: dict[str, Any]) -> set[str]:
    docs = set()
    for ctx in row.get("supporting_context", []):
        if isinstance(ctx, dict) and ctx.get("doc_id"):
            docs.add(str(ctx["doc_id"]).strip())
    return docs or gold_doc_ids(row)


def pseudo_supervision_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    return metadata


def qid_supervision_tier(row: dict[str, Any]) -> str:
    metadata = pseudo_supervision_metadata(row)
    return str(
        metadata.get("pseudo_gold_qid_supervision_tier")
        or metadata.get("vlm_training_tier")
        or row.get("supervision_tier")
        or ""
    ).strip()


def negative_supervision_scope(row: dict[str, Any]) -> str:
    metadata = pseudo_supervision_metadata(row)
    return str(
        metadata.get("pseudo_gold_negative_supervision_scope")
        or metadata.get("negative_supervision_scope")
        or row.get("negative_supervision_scope")
        or ""
    ).strip()


def is_positive_only_supervision_tier(tier: str) -> bool:
    return "positive_only" in tier


def page_supervision_tiers(row: dict[str, Any]) -> dict[str, str]:
    metadata = pseudo_supervision_metadata(row)
    values = metadata.get("pseudo_gold_page_supervision_tiers", {})
    if not isinstance(values, dict):
        return {}
    return {str(uid): str(tier) for uid, tier in values.items()}


def positive_row_weight(uid: str, row: dict[str, Any], args: argparse.Namespace) -> float:
    if not bool(getattr(args, "pseudo_supervision_weighting", False)):
        return 1.0
    qid_tier = qid_supervision_tier(row)
    page_tier = page_supervision_tiers(row).get(uid, "")
    if "partial" in qid_tier or "support_incomplete" in qid_tier:
        return float(args.partial_positive_weight)
    if page_tier == "visual_proxy":
        return float(args.visual_proxy_positive_weight)
    if page_tier in {"mixed_direct_proxy", "direct_visual_verified"}:
        return float(args.hybrid_positive_weight)
    if "hybrid" in qid_tier:
        return float(args.hybrid_positive_weight)
    return 1.0


def positive_page_locations(positive_uids: set[str]) -> dict[str, set[int]]:
    out: dict[str, set[int]] = defaultdict(set)
    for uid in positive_uids:
        parsed = parse_page_uid(uid)
        if parsed:
            doc_id, page_idx = parsed
            out[doc_id].add(int(page_idx))
    return out


def safe_support_doc_negative_indices(
    records: list[dict[str, Any]],
    positive_uids: set[str],
    support_docs: set[str],
    args: argparse.Namespace,
) -> list[int]:
    if not bool(getattr(args, "include_safe_support_doc_negatives", False)):
        return []
    positive_by_doc = positive_page_locations(positive_uids)
    selected: list[int] = []
    for idx, record in enumerate(records):
        uid = str(record["uid"])
        doc_id = str(record.get("doc_id", "")).strip()
        if uid in positive_uids or doc_id not in support_docs:
            continue
        page_idx = int(record.get("page_idx", 0))
        positive_pages = positive_by_doc.get(doc_id, set())
        if any(abs(page_idx - positive_idx) <= 1 for positive_idx in positive_pages):
            continue
        selected.append(idx)
        if len(selected) >= int(args.max_safe_support_doc_negatives_per_qid):
            break
    return selected


def split_gold_for_tuning(
    gold: dict[str, dict[str, Any]],
    *,
    tune_fraction: float,
    seed: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if not 0.0 < float(tune_fraction) < 1.0:
        raise ValueError(f"tune_fraction must be in (0, 1), got {tune_fraction}")
    qids = sorted(gold)
    rng = random.Random(int(seed))
    rng.shuffle(qids)
    tune_count = int(round(len(qids) * float(tune_fraction)))
    tune_count = max(1, min(len(qids) - 1, tune_count)) if len(qids) >= 2 else 0
    tune_qids = set(qids[:tune_count])
    fit = {qid: row for qid, row in gold.items() if qid not in tune_qids}
    tune = {qid: row for qid, row in gold.items() if qid in tune_qids}
    return fit, tune


def pick_negative_indices(
    records: list[dict[str, Any]],
    positive_uids: set[str],
    args: argparse.Namespace,
    *,
    excluded_doc_ids: set[str] | None = None,
) -> list[int]:
    excluded_doc_ids = excluded_doc_ids or set()
    negative_indices: list[int] = []
    used: set[int] = set()
    bands = [
        (1, 4),
        (5, 20),
        (21, 100),
        (101, 500),
        (501, int(args.candidate_top_k)),
    ]
    for lo, hi in bands:
        count = 0
        for idx, record in enumerate(records):
            rank = int(record["base_rank"])
            doc_id = str(record.get("doc_id", "")).strip()
            if (
                rank < lo
                or rank > hi
                or record["uid"] in positive_uids
                or idx in used
                or doc_id in excluded_doc_ids
            ):
                continue
            negative_indices.append(idx)
            used.add(idx)
            count += 1
            if count >= int(args.negatives_per_band):
                break
            if len(negative_indices) >= int(args.max_negatives_per_qid):
                return negative_indices
    return negative_indices[: int(args.max_negatives_per_qid)]


def build_matrix(
    *,
    gold: dict[str, dict[str, Any]],
    base_pred: dict[str, dict[str, Any]],
    page_features: dict[str, dict[str, Any]],
    source_maps_by_label: dict[str, dict[str, dict[str, float]]],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    vectors: list[list[float]] = []
    labels: list[int] = []
    row_weights: list[float] = []
    qid_count = 0
    qid_with_positive_in_pool = 0
    skipped_no_page_gold = 0
    skipped_no_positive_in_pool = 0
    skipped_exclude_unlabeled_tier = 0
    skipped_no_negative_rows = 0
    positive_only_qid_count = 0
    qid_with_negative_rows = 0
    qid_without_negative_rows = 0
    positive_count = 0
    negative_count = 0
    weighted_positive_count = 0.0
    weighted_negative_count = 0.0
    safe_support_doc_negative_count = 0
    supervision_tier_counts: Counter[str] = Counter()
    negative_scope_counts: Counter[str] = Counter()
    negative_excluded_support_doc_qids = 0

    rng = random.Random(int(args.seed))
    respect_tiers = bool(getattr(args, "respect_pseudo_supervision_tiers", False))
    for qid, gold_row in gold.items():
        tier = qid_supervision_tier(gold_row)
        if tier:
            supervision_tier_counts[tier] += 1
        scope = negative_supervision_scope(gold_row)
        if scope:
            negative_scope_counts[scope] += 1
        if respect_tiers and tier == "exclude_unlabeled":
            skipped_exclude_unlabeled_tier += 1
            continue
        positive_uids = gold_page_uids(gold_row)
        if not positive_uids:
            skipped_no_page_gold += 1
            continue
        records = ranked_page_records(base_pred.get(qid), int(args.candidate_top_k))
        if not records:
            continue
        qid_count += 1
        record_uids = {record["uid"] for record in records}
        positive_indices = [idx for idx, record in enumerate(records) if record["uid"] in positive_uids]
        if not positive_indices:
            skipped_no_positive_in_pool += 1
            continue
        qid_with_positive_in_pool += 1
        positive_only_tier = respect_tiers and is_positive_only_supervision_tier(tier)
        negative_indices: list[int] = []
        safe_support_negative_indices: list[int] = []
        if positive_only_tier:
            positive_only_qid_count += 1
        else:
            excluded_doc_ids: set[str] = set()
            if respect_tiers and scope == "non_support_docs_only":
                excluded_doc_ids = support_doc_ids(gold_row)
                if excluded_doc_ids:
                    negative_excluded_support_doc_qids += 1
            negative_indices = pick_negative_indices(
                records,
                positive_uids,
                args,
                excluded_doc_ids=excluded_doc_ids,
            )
            if respect_tiers:
                safe_support_negative_indices = safe_support_doc_negative_indices(
                    records,
                    positive_uids,
                    support_doc_ids(gold_row),
                    args,
                )
                safe_support_doc_negative_count += len(safe_support_negative_indices)
                seen_negative_indices = set(negative_indices)
                for idx in safe_support_negative_indices:
                    if idx not in seen_negative_indices:
                        negative_indices.append(idx)
                        seen_negative_indices.add(idx)
        if negative_indices:
            qid_with_negative_rows += 1
        else:
            qid_without_negative_rows += 1
        if not negative_indices and not respect_tiers:
            skipped_no_negative_rows += 1
            continue
        rng.shuffle(negative_indices)
        negative_indices = negative_indices[: int(args.max_negatives_per_qid)]

        base_norm_scores = normalize_scores(records)
        doc_rank, doc_page_rank, doc_page_count = doc_rank_maps(records)
        q_profile = question_profile(gold_row)
        for idx in positive_indices:
            vectors.append(
                feature_vector(
                    records[idx],
                    records=records,
                    base_norm_scores=base_norm_scores,
                    doc_rank=doc_rank,
                    doc_page_rank=doc_page_rank,
                    doc_page_count=doc_page_count,
                    question=q_profile,
                    page_features=page_features,
                    source_maps_by_label=source_maps_by_label,
                    qid=qid,
                    feature_names=getattr(args, "active_feature_names", FEATURE_NAMES),
                )
            )
            labels.append(1)
            weight = positive_row_weight(records[idx]["uid"], gold_row, args)
            row_weights.append(float(weight))
            weighted_positive_count += float(weight)
            positive_count += 1
        for idx in negative_indices:
            vectors.append(
                feature_vector(
                    records[idx],
                    records=records,
                    base_norm_scores=base_norm_scores,
                    doc_rank=doc_rank,
                    doc_page_rank=doc_page_rank,
                    doc_page_count=doc_page_count,
                    question=q_profile,
                    page_features=page_features,
                    source_maps_by_label=source_maps_by_label,
                    qid=qid,
                    feature_names=getattr(args, "active_feature_names", FEATURE_NAMES),
                )
            )
            labels.append(0)
            if idx in safe_support_negative_indices:
                weight = float(args.safe_support_doc_negative_weight)
            else:
                weight = 1.0
            row_weights.append(float(weight))
            weighted_negative_count += float(weight)
            negative_count += 1

    if not vectors:
        raise ValueError("No training rows were produced.")
    metadata = {
        "train_qid_with_page_gold_and_base": int(qid_count),
        "train_qid_with_positive_in_pool": int(qid_with_positive_in_pool),
        "skipped_no_page_gold": int(skipped_no_page_gold),
        "skipped_no_positive_in_pool": int(skipped_no_positive_in_pool),
        "skipped_exclude_unlabeled_tier": int(skipped_exclude_unlabeled_tier),
        "skipped_no_negative_rows": int(skipped_no_negative_rows),
        "positive_only_qid_count": int(positive_only_qid_count),
        "qid_with_negative_rows": int(qid_with_negative_rows),
        "qid_without_negative_rows": int(qid_without_negative_rows),
        "positive_count": int(positive_count),
        "negative_count": int(negative_count),
        "weighted_positive_count": float(weighted_positive_count),
        "weighted_negative_count": float(weighted_negative_count),
        "safe_support_doc_negative_count": int(safe_support_doc_negative_count),
        "row_count": int(len(vectors)),
        "respect_pseudo_supervision_tiers": bool(respect_tiers),
        "pseudo_supervision_weighting": bool(getattr(args, "pseudo_supervision_weighting", False)),
        "include_safe_support_doc_negatives": bool(
            getattr(args, "include_safe_support_doc_negatives", False)
        ),
        "negative_excluded_support_doc_qids": int(negative_excluded_support_doc_qids),
        "supervision_tier_counts": dict(sorted(supervision_tier_counts.items())),
        "negative_scope_counts": dict(sorted(negative_scope_counts.items())),
    }
    return (
        np.asarray(vectors, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        np.asarray(row_weights, dtype=np.float32),
        metadata,
    )


def standardize_train(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (X - mean) / std, mean.astype(np.float32), std.astype(np.float32)


def standardize_eval(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def train_logistic(
    X: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
    *,
    row_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, float, list[dict[str, float]]]:
    rng = np.random.default_rng(int(args.seed))
    w = np.zeros(X.shape[1], dtype=np.float32)
    b = np.float32(0.0)
    m_w = np.zeros_like(w)
    v_w = np.zeros_like(w)
    m_b = np.float32(0.0)
    v_b = np.float32(0.0)
    base_weights = (
        np.ones_like(y, dtype=np.float32)
        if row_weights is None
        else np.asarray(row_weights, dtype=np.float32)
    )
    if base_weights.shape != y.shape:
        raise ValueError(f"row_weights shape {base_weights.shape} does not match y shape {y.shape}")
    pos = float(base_weights[y > 0.5].sum())
    neg = float(base_weights[y <= 0.5].sum())
    pos_weight = min(float(args.positive_weight_cap), neg / max(pos, 1.0))
    weights = (base_weights * np.where(y > 0.5, pos_weight, 1.0)).astype(np.float32)
    history: list[dict[str, float]] = []
    step = 0
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    lr = float(args.learning_rate)
    batch_size = int(args.batch_size)

    for epoch in range(1, int(args.epochs) + 1):
        order = rng.permutation(len(y))
        total_loss = 0.0
        total_weight = 0.0
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            xb = X[idx]
            yb = y[idx]
            wb = weights[idx]
            logits = xb @ w + b
            probs = sigmoid(logits)
            error = (probs - yb) * wb
            denom = max(float(wb.sum()), 1.0)
            grad_w = (xb.T @ error) / denom + float(args.weight_decay) * w
            grad_b = np.float32(error.sum() / denom)
            step += 1
            m_w = beta1 * m_w + (1.0 - beta1) * grad_w
            v_w = beta2 * v_w + (1.0 - beta2) * (grad_w * grad_w)
            m_b = np.float32(beta1 * m_b + (1.0 - beta1) * grad_b)
            v_b = np.float32(beta2 * v_b + (1.0 - beta2) * (grad_b * grad_b))
            m_w_hat = m_w / (1.0 - beta1**step)
            v_w_hat = v_w / (1.0 - beta2**step)
            m_b_hat = m_b / (1.0 - beta1**step)
            v_b_hat = v_b / (1.0 - beta2**step)
            w -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)
            b -= np.float32(lr * m_b_hat / (math.sqrt(float(v_b_hat)) + eps))
            loss = -(
                yb * np.log(np.clip(probs, 1e-6, 1.0))
                + (1.0 - yb) * np.log(np.clip(1.0 - probs, 1e-6, 1.0))
            )
            total_loss += float((loss * wb).sum())
            total_weight += float(wb.sum())
        if epoch == 1 or epoch == int(args.epochs) or epoch % max(1, int(args.epochs) // 10) == 0:
            history.append({"epoch": float(epoch), "loss": total_loss / max(total_weight, 1.0)})
    return w.astype(np.float32), float(b), history


def score_records(
    *,
    qid: str,
    gold_row: dict[str, Any],
    records: list[dict[str, Any]],
    page_features: dict[str, dict[str, Any]],
    source_maps_by_label: dict[str, dict[str, dict[str, float]]],
    mean: np.ndarray,
    std: np.ndarray,
    weights: np.ndarray,
    bias: float,
    feature_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not records:
        return []
    base_norm_scores = normalize_scores(records)
    doc_rank, doc_page_rank, doc_page_count = doc_rank_maps(records)
    q_profile = question_profile(gold_row)
    vectors = [
        feature_vector(
            record,
            records=records,
            base_norm_scores=base_norm_scores,
            doc_rank=doc_rank,
            doc_page_rank=doc_page_rank,
            doc_page_count=doc_page_count,
            question=q_profile,
            page_features=page_features,
            source_maps_by_label=source_maps_by_label,
            qid=qid,
            feature_names=feature_names,
        )
        for record in records
    ]
    X = standardize_eval(np.asarray(vectors, dtype=np.float32), mean, std)
    probs = sigmoid(X @ weights + np.float32(bias))
    for record, prob in zip(records, probs):
        record["learned_score"] = float(prob)
    return records


def minmax_by_key(records: list[dict[str, Any]], key: str) -> dict[str, float]:
    if not records:
        return {}
    values = [float(record.get(key, 0.0)) for record in records]
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        return {record["uid"]: 1.0 for record in records}
    return {record["uid"]: (float(record.get(key, 0.0)) - lo) / (hi - lo) for record in records}


def assign_blend_scores(records: list[dict[str, Any]], alpha: float) -> None:
    model_norm = minmax_by_key(records, "learned_score")
    base_rank_scores = {row["uid"]: 1.0 / math.log2(float(row["base_rank"]) + 1.0) for row in records}
    lo = min(base_rank_scores.values())
    hi = max(base_rank_scores.values())
    if not math.isclose(lo, hi):
        base_rank_scores = {uid: (value - lo) / (hi - lo) for uid, value in base_rank_scores.items()}
    for row in records:
        row["rerank_score"] = (1.0 - alpha) * base_rank_scores[row["uid"]] + alpha * model_norm[row["uid"]]


def query_confidence_score(
    records: list[dict[str, Any]],
    source_maps_by_label: dict[str, dict[str, dict[str, float]]],
    qid: str,
) -> float:
    """Estimate how much the base ranking should be trusted for this query."""
    if not records:
        return 0.0

    norm_scores = normalize_scores(records)
    top_uid = str(records[0]["uid"])
    if len(records) >= 2:
        score_gap = float(norm_scores.get(top_uid, 0.0)) - float(norm_scores.get(str(records[1]["uid"]), 0.0))
    else:
        score_gap = 1.0
    score_gap = max(0.0, min(1.0, score_gap))

    top_doc = str(records[0]["doc_id"])
    top_window = records[: min(5, len(records))]
    doc_concentration = (
        sum(1 for row in top_window if str(row["doc_id"]) == top_doc) / float(len(top_window))
        if top_window
        else 0.0
    )

    source_agreement = 0.0
    if source_maps_by_label:
        agree = 0
        total = 0
        for qid_map in source_maps_by_label.values():
            page_map = qid_map.get(qid, {})
            total += 1
            source_row = page_map.get(top_uid)
            if source_row and float(source_row.get("rank", math.inf)) <= 5.0:
                agree += 1
        source_agreement = agree / float(max(total, 1))

    components = [score_gap, doc_concentration]
    if source_maps_by_label:
        components.append(source_agreement)
    return float(sum(components) / float(len(components)))


def fit_confidence_thresholds(scores: list[float], bin_count: int) -> list[float]:
    if bin_count <= 1 or not scores:
        return []
    thresholds: list[float] = []
    values = np.asarray(scores, dtype=np.float32)
    for idx in range(1, int(bin_count)):
        threshold = float(np.quantile(values, idx / float(bin_count)))
        if thresholds and math.isclose(threshold, thresholds[-1]):
            continue
        thresholds.append(threshold)
    return thresholds


def confidence_bin(score: float, thresholds: list[float]) -> int:
    for idx, threshold in enumerate(thresholds):
        if float(score) <= float(threshold):
            return idx
    return len(thresholds)


def alpha_for_query_confidence(score: float, adaptive_config: dict[str, Any] | None, fallback_alpha: float) -> tuple[float, int | None]:
    if not adaptive_config:
        return float(fallback_alpha), None
    thresholds = [float(value) for value in adaptive_config.get("confidence_thresholds", [])]
    bin_idx = confidence_bin(float(score), thresholds)
    alphas = adaptive_config.get("bin_selected_blend_alpha", [])
    if 0 <= bin_idx < len(alphas):
        return float(alphas[bin_idx]), int(bin_idx)
    return float(adaptive_config.get("global_fallback_blend_alpha", fallback_alpha)), int(bin_idx)


def source_fraction_for_uid(
    *,
    uid: str,
    qid: str,
    source_maps_by_label: dict[str, dict[str, dict[str, float]]],
) -> float:
    if not source_maps_by_label:
        return 0.0
    hits = 0
    for qid_map in source_maps_by_label.values():
        if uid in qid_map.get(qid, {}):
            hits += 1
    return float(hits) / float(max(len(source_maps_by_label), 1))


def normalized_entropy(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    arr = np.asarray(values, dtype=np.float32)
    arr = arr - float(arr.max())
    probs = np.exp(arr)
    denom = float(probs.sum())
    if denom <= 0.0:
        return 0.0
    probs = probs / denom
    entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0))))
    return entropy / math.log(float(len(values)))


def query_alpha_feature_vector(
    records: list[dict[str, Any]],
    source_maps_by_label: dict[str, dict[str, dict[str, float]]],
    qid: str,
    top_k: int,
) -> list[float]:
    if not records:
        return [0.0 for _ in QUERY_ALPHA_FEATURE_NAMES]

    n = len(records)
    feature_top_k = max(1, min(int(top_k), n))
    base_norm_scores = normalize_scores(records)
    top_uid = str(records[0]["uid"])
    if len(records) >= 2:
        base_top_score_gap = float(base_norm_scores.get(top_uid, 0.0)) - float(
            base_norm_scores.get(str(records[1]["uid"]), 0.0)
        )
    else:
        base_top_score_gap = 1.0
    base_top_score_gap = max(0.0, min(1.0, base_top_score_gap))

    base_window5 = records[: min(5, n)]
    top_doc = str(records[0]["doc_id"])
    base_top5_doc_concentration = (
        sum(1 for row in base_window5 if str(row["doc_id"]) == top_doc) / float(len(base_window5))
        if base_window5
        else 0.0
    )

    model_sorted = sorted(
        records,
        key=lambda row: (-float(row.get("learned_score", 0.0)), int(row["base_rank"]), str(row["uid"])),
    )
    top_model = model_sorted[0]
    model_top_score = float(top_model.get("learned_score", 0.0))
    if len(model_sorted) >= 2:
        model_top2_margin = model_top_score - float(model_sorted[1].get("learned_score", 0.0))
    else:
        model_top2_margin = 1.0
    model_top2_margin = max(0.0, model_top2_margin)

    base_top4_scores = [float(row.get("learned_score", 0.0)) for row in records[: min(4, n)]]
    base_top_window_scores = [float(row.get("learned_score", 0.0)) for row in records[:feature_top_k]]
    model_top_window_scores = [float(row.get("learned_score", 0.0)) for row in model_sorted[:feature_top_k]]
    model_top4_mean = float(np.mean(base_top4_scores)) if base_top4_scores else 0.0
    model_top_window_mean = float(np.mean(base_top_window_scores)) if base_top_window_scores else 0.0
    model_top_window_std = float(np.std(base_top_window_scores)) if base_top_window_scores else 0.0
    model_entropy_top_window = normalized_entropy(model_top_window_scores)

    top_model_rank = int(top_model["base_rank"])
    model_top_base_rank_recip = 1.0 / float(max(top_model_rank, 1))
    if n <= 1:
        model_top_base_rank_frac = 1.0
        model_base_disagreement = 0.0
    else:
        model_top_base_rank_frac = 1.0 - min(float(top_model_rank), float(n)) / float(n)
        model_base_disagreement = min(max((float(top_model_rank) - 1.0) / float(n - 1), 0.0), 1.0)

    model_top_source_fraction = source_fraction_for_uid(
        uid=str(top_model["uid"]),
        qid=qid,
        source_maps_by_label=source_maps_by_label,
    )
    model_top_window_source_fraction = float(
        np.mean(
            [
                source_fraction_for_uid(
                    uid=str(row["uid"]),
                    qid=qid,
                    source_maps_by_label=source_maps_by_label,
                )
                for row in model_sorted[:feature_top_k]
            ]
        )
    )

    values = {
        "base_confidence": query_confidence_score(records, source_maps_by_label, qid),
        "candidate_count_log": math.log1p(float(n)),
        "base_top_score_gap": base_top_score_gap,
        "base_top5_doc_concentration": base_top5_doc_concentration,
        "model_top_score": model_top_score,
        "model_top2_margin": model_top2_margin,
        "model_top4_mean": model_top4_mean,
        "model_top_window_mean": model_top_window_mean,
        "model_top_window_std": model_top_window_std,
        "model_entropy_top_window": model_entropy_top_window,
        "model_top_base_rank_recip": model_top_base_rank_recip,
        "model_top_base_rank_frac": model_top_base_rank_frac,
        "model_top_in_base_top4": 1.0 if top_model_rank <= 4 else 0.0,
        "model_top_in_base_top10": 1.0 if top_model_rank <= 10 else 0.0,
        "model_base_disagreement": model_base_disagreement,
        "model_top_source_fraction": model_top_source_fraction,
        "model_top_window_source_fraction": model_top_window_source_fraction,
    }
    return [float(values[name]) for name in QUERY_ALPHA_FEATURE_NAMES]


def select_best_alpha_for_case(
    *,
    case: dict[str, Any],
    alpha_grid: list[float],
    candidate_args: argparse.Namespace,
) -> tuple[float, int | None, bool]:
    pages_gold = set(case["pages_gold"])
    best_alpha = float(alpha_grid[0])
    best_rank_value = math.inf
    best_rank: int | None = None
    for alpha in alpha_grid:
        candidate_args.blend_alpha = float(alpha)
        candidate_records = [dict(row) for row in case["scored_records"]]
        ranked = rerank_records(candidate_records, candidate_args)
        rank = first_rank([str(row["uid"]) for row in ranked], pages_gold)
        rank_value = float(rank if rank is not None else math.inf)
        if rank_value < best_rank_value or (math.isclose(rank_value, best_rank_value) and float(alpha) < best_alpha):
            best_alpha = float(alpha)
            best_rank_value = rank_value
            best_rank = rank
    return best_alpha, best_rank, best_rank is not None


def nearest_alpha_grid_value(value: float, alpha_grid: list[float]) -> float:
    return float(min(alpha_grid, key=lambda alpha: (abs(float(alpha) - float(value)), float(alpha))))


def predict_learned_query_alpha(
    *,
    records: list[dict[str, Any]],
    source_maps_by_label: dict[str, dict[str, dict[str, float]]],
    qid: str,
    adaptive_config: dict[str, Any] | None,
    fallback_alpha: float,
) -> tuple[float, dict[str, Any]]:
    if not adaptive_config or adaptive_config.get("mode") != "learned_query_regressor":
        return float(fallback_alpha), {"mode": None}
    feature_top_k = int(adaptive_config.get("query_alpha_feature_top_k", 20))
    features = query_alpha_feature_vector(records, source_maps_by_label, qid, feature_top_k)
    mean = np.asarray(adaptive_config.get("feature_mean", []), dtype=np.float32)
    std = np.asarray(adaptive_config.get("feature_std", []), dtype=np.float32)
    weights = np.asarray(adaptive_config.get("weights", []), dtype=np.float32)
    bias = float(adaptive_config.get("bias", 0.0))
    if len(features) != len(mean) or len(features) != len(std) or len(features) != len(weights):
        return float(adaptive_config.get("global_fallback_blend_alpha", fallback_alpha)), {
            "mode": "learned_query_regressor",
            "error": "feature_shape_mismatch",
        }
    X = (np.asarray(features, dtype=np.float32) - mean) / std
    raw_alpha = float(X @ weights + np.float32(bias))
    min_alpha = float(adaptive_config.get("alpha_min", 0.0))
    max_alpha = float(adaptive_config.get("alpha_max", 1.0))
    clipped_alpha = min(max(raw_alpha, min_alpha), max_alpha)
    alpha_grid = [float(value) for value in adaptive_config.get("alpha_grid", [])]
    snap_to_grid = bool(adaptive_config.get("snap_to_grid", True))
    selected_alpha = nearest_alpha_grid_value(clipped_alpha, alpha_grid) if snap_to_grid and alpha_grid else clipped_alpha
    return float(selected_alpha), {
        "mode": "learned_query_regressor",
        "raw_alpha": raw_alpha,
        "clipped_alpha": clipped_alpha,
        "snap_to_grid": snap_to_grid,
    }


def softmax_rows(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40.0, 40.0)
    shifted = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    denom = np.clip(exp_values.sum(axis=1, keepdims=True), 1e-12, None)
    return exp_values / denom


def train_multiclass_logistic(
    X: np.ndarray,
    y: np.ndarray,
    *,
    class_count: int,
    seed: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    rng = np.random.default_rng(int(seed))
    W = np.zeros((X.shape[1], class_count), dtype=np.float32)
    b = np.zeros(class_count, dtype=np.float32)
    class_hist = np.bincount(y, minlength=class_count).astype(np.float32)
    present = class_hist > 0
    class_weights = np.ones(class_count, dtype=np.float32)
    if present.any():
        class_weights[present] = float(len(y)) / (float(present.sum()) * np.maximum(class_hist[present], 1.0))
    history: list[dict[str, float]] = []
    lr = float(learning_rate)
    wd = float(weight_decay)
    epochs = max(1, int(epochs))

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(y))
        Xb = X[order]
        yb = y[order]
        logits = Xb @ W + b
        probs = softmax_rows(logits)
        onehot = np.zeros_like(probs, dtype=np.float32)
        onehot[np.arange(len(yb)), yb] = 1.0
        row_weights = class_weights[yb].reshape(-1, 1)
        error = (probs - onehot) * row_weights
        denom = max(float(row_weights.sum()), 1.0)
        grad_W = (Xb.T @ error) / denom + wd * W
        grad_b = error.sum(axis=0) / denom
        W -= np.float32(lr) * grad_W.astype(np.float32)
        b -= np.float32(lr) * grad_b.astype(np.float32)
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
            chosen = np.clip(probs[np.arange(len(yb)), yb], 1e-6, 1.0)
            loss = -np.log(chosen) * class_weights[yb]
            history.append({"epoch": float(epoch), "loss": float(loss.mean())})
    return W.astype(np.float32), b.astype(np.float32), history


def predict_learned_alpha_action(
    *,
    records: list[dict[str, Any]],
    source_maps_by_label: dict[str, dict[str, dict[str, float]]],
    qid: str,
    adaptive_config: dict[str, Any] | None,
    fallback_alpha: float,
) -> tuple[float, dict[str, Any]]:
    if not adaptive_config or adaptive_config.get("mode") != "learned_alpha_action":
        return float(fallback_alpha), {"mode": None}
    feature_top_k = int(adaptive_config.get("query_alpha_feature_top_k", 20))
    features = query_alpha_feature_vector(records, source_maps_by_label, qid, feature_top_k)
    mean = np.asarray(adaptive_config.get("feature_mean", []), dtype=np.float32)
    std = np.asarray(adaptive_config.get("feature_std", []), dtype=np.float32)
    weights = np.asarray(adaptive_config.get("weights", []), dtype=np.float32)
    bias = np.asarray(adaptive_config.get("bias", []), dtype=np.float32)
    alpha_grid = [float(value) for value in adaptive_config.get("alpha_grid", [])]
    if (
        len(features) != len(mean)
        or len(features) != len(std)
        or weights.ndim != 2
        or weights.shape[0] != len(features)
        or weights.shape[1] != len(alpha_grid)
        or len(bias) != len(alpha_grid)
    ):
        return float(adaptive_config.get("global_fallback_blend_alpha", fallback_alpha)), {
            "mode": "learned_alpha_action",
            "error": "feature_shape_mismatch",
        }
    X = ((np.asarray(features, dtype=np.float32) - mean) / std).reshape(1, -1)
    probs = softmax_rows(X @ weights + bias)[0]
    action_idx = int(np.argmax(probs))
    selected_alpha = float(alpha_grid[action_idx])
    top_probs = sorted(
        (
            {"alpha": float(alpha_grid[idx]), "probability": float(probs[idx])}
            for idx in range(len(alpha_grid))
        ),
        key=lambda row: (-float(row["probability"]), float(row["alpha"])),
    )[:5]
    return selected_alpha, {
        "mode": "learned_alpha_action",
        "action_idx": int(action_idx),
        "action_probability": float(probs[action_idx]),
        "top_actions": top_probs,
    }


def fit_learned_query_alpha_config(
    *,
    tune_cases: list[dict[str, Any]],
    alpha_grid: list[float],
    hit_k: int,
    candidate_args: argparse.Namespace,
    source_maps_by_label: dict[str, dict[str, dict[str, float]]],
    args: argparse.Namespace,
    global_fallback_alpha: float,
) -> dict[str, Any]:
    vectors: list[list[float]] = []
    targets: list[float] = []
    target_rows: list[dict[str, Any]] = []
    feature_top_k = int(getattr(args, "query_alpha_feature_top_k", 20))
    for case in tune_cases:
        target_alpha, target_rank, target_hit_any = select_best_alpha_for_case(
            case=case,
            alpha_grid=alpha_grid,
            candidate_args=candidate_args,
        )
        vectors.append(
            query_alpha_feature_vector(
                case["scored_records"],
                source_maps_by_label,
                str(case["qid"]),
                feature_top_k,
            )
        )
        targets.append(float(target_alpha))
        target_rows.append(
            {
                "qid": str(case["qid"]),
                "target_alpha": float(target_alpha),
                "target_rank": None if target_rank is None else int(target_rank),
                f"target_hit@{hit_k}": bool(target_rank is not None and target_rank <= int(hit_k)),
                "target_hit_any": bool(target_hit_any),
            }
        )

    if not vectors:
        return {
            "mode": "learned_query_regressor",
            "error": "no_tune_cases",
            "global_fallback_blend_alpha": float(global_fallback_alpha),
        }

    X = np.asarray(vectors, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    X_std, mean, std = standardize_train(X)
    X_aug = np.concatenate([X_std, np.ones((X_std.shape[0], 1), dtype=np.float32)], axis=1)
    ridge = max(0.0, float(getattr(args, "query_alpha_ridge", 1.0)))
    reg = np.eye(X_aug.shape[1], dtype=np.float32) * ridge
    reg[-1, -1] = 0.0
    lhs = X_aug.T @ X_aug + reg
    rhs = X_aug.T @ y
    try:
        coef = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    weights = coef[:-1].astype(np.float32)
    bias = float(coef[-1])

    target_counts = Counter(round(float(value), 8) for value in targets)
    alpha_min = min(float(value) for value in alpha_grid)
    alpha_max = max(float(value) for value in alpha_grid)
    snap_to_grid = not bool(getattr(args, "query_alpha_continuous", False))
    config = {
        "mode": "learned_query_regressor",
        "query_alpha_feature_names": list(QUERY_ALPHA_FEATURE_NAMES),
        "query_alpha_feature_top_k": int(feature_top_k),
        "query_alpha_ridge": float(ridge),
        "snap_to_grid": bool(snap_to_grid),
        "alpha_grid": [float(value) for value in alpha_grid],
        "alpha_min": float(alpha_min),
        "alpha_max": float(alpha_max),
        "weights": [float(value) for value in weights.tolist()],
        "bias": float(bias),
        "feature_mean": [float(value) for value in mean.tolist()],
        "feature_std": [float(value) for value in std.tolist()],
        "global_fallback_blend_alpha": float(global_fallback_alpha),
        "target_alpha_distribution": {
            f"{float(alpha):.8g}": int(count)
            for alpha, count in sorted(target_counts.items(), key=lambda item: float(item[0]))
        },
    }

    predicted_hits = 0
    oracle_hits = 0
    predicted_counts: Counter[float] = Counter()
    squared_error = 0.0
    for case, target_alpha in zip(tune_cases, targets):
        predicted_alpha, info = predict_learned_query_alpha(
            records=case["scored_records"],
            source_maps_by_label=source_maps_by_label,
            qid=str(case["qid"]),
            adaptive_config=config,
            fallback_alpha=float(global_fallback_alpha),
        )
        predicted_counts[round(float(predicted_alpha), 8)] += 1
        squared_error += (float(info.get("clipped_alpha", predicted_alpha)) - float(target_alpha)) ** 2
        for alpha, key in [(predicted_alpha, "predicted"), (target_alpha, "oracle")]:
            candidate_args.blend_alpha = float(alpha)
            candidate_records = [dict(row) for row in case["scored_records"]]
            ranked = rerank_records(candidate_records, candidate_args)
            ranked_uids = [str(row["uid"]) for row in ranked]
            hit = any(uid in case["pages_gold"] for uid in ranked_uids[:hit_k])
            if key == "predicted" and hit:
                predicted_hits += 1
            if key == "oracle" and hit:
                oracle_hits += 1

    evaluated = max(1, len(tune_cases))
    config.update(
        {
            "tune_eval_qid_count": int(len(tune_cases)),
            "predicted_alpha_distribution": {
                f"{float(alpha):.8g}": int(count)
                for alpha, count in sorted(predicted_counts.items(), key=lambda item: float(item[0]))
            },
            f"predicted_page@{hit_k}": float(predicted_hits) / float(evaluated),
            f"oracle_page@{hit_k}": float(oracle_hits) / float(evaluated),
            "target_alpha_mse": float(squared_error) / float(evaluated),
            "target_examples": target_rows[:20],
        }
    )
    return config


def fit_learned_alpha_action_config(
    *,
    tune_cases: list[dict[str, Any]],
    alpha_grid: list[float],
    hit_k: int,
    candidate_args: argparse.Namespace,
    source_maps_by_label: dict[str, dict[str, dict[str, float]]],
    args: argparse.Namespace,
    global_fallback_alpha: float,
) -> dict[str, Any]:
    vectors: list[list[float]] = []
    targets: list[int] = []
    action_cases: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    feature_top_k = int(getattr(args, "query_alpha_feature_top_k", 20))
    alpha_to_idx = {round(float(alpha), 8): idx for idx, alpha in enumerate(alpha_grid)}
    for case in tune_cases:
        target_alpha, target_rank, target_hit_any = select_best_alpha_for_case(
            case=case,
            alpha_grid=alpha_grid,
            candidate_args=candidate_args,
        )
        alpha_key = round(float(target_alpha), 8)
        if alpha_key not in alpha_to_idx:
            continue
        vectors.append(
            query_alpha_feature_vector(
                case["scored_records"],
                source_maps_by_label,
                str(case["qid"]),
                feature_top_k,
            )
        )
        targets.append(int(alpha_to_idx[alpha_key]))
        action_cases.append(case)
        target_rows.append(
            {
                "qid": str(case["qid"]),
                "target_alpha": float(target_alpha),
                "target_action_idx": int(alpha_to_idx[alpha_key]),
                "target_rank": None if target_rank is None else int(target_rank),
                f"target_hit@{hit_k}": bool(target_rank is not None and target_rank <= int(hit_k)),
                "target_hit_any": bool(target_hit_any),
            }
        )

    if not vectors:
        return {
            "mode": "learned_alpha_action",
            "error": "no_tune_cases",
            "global_fallback_blend_alpha": float(global_fallback_alpha),
        }

    X = np.asarray(vectors, dtype=np.float32)
    y = np.asarray(targets, dtype=np.int64)
    X_std, mean, std = standardize_train(X)
    weights, bias, history = train_multiclass_logistic(
        X_std,
        y,
        class_count=len(alpha_grid),
        seed=int(getattr(args, "seed", 13)),
        epochs=int(getattr(args, "query_alpha_action_epochs", 200)),
        learning_rate=float(getattr(args, "query_alpha_action_learning_rate", 0.05)),
        weight_decay=float(getattr(args, "query_alpha_action_weight_decay", 1e-3)),
    )
    target_counts = Counter(int(value) for value in targets)
    config = {
        "mode": "learned_alpha_action",
        "query_alpha_feature_names": list(QUERY_ALPHA_FEATURE_NAMES),
        "query_alpha_feature_top_k": int(feature_top_k),
        "alpha_grid": [float(value) for value in alpha_grid],
        "weights": [[float(value) for value in row] for row in weights.tolist()],
        "bias": [float(value) for value in bias.tolist()],
        "feature_mean": [float(value) for value in mean.tolist()],
        "feature_std": [float(value) for value in std.tolist()],
        "global_fallback_blend_alpha": float(global_fallback_alpha),
        "action_training_history": history,
        "target_action_distribution": {
            f"{float(alpha_grid[idx]):.8g}": int(target_counts.get(idx, 0))
            for idx in range(len(alpha_grid))
        },
        "query_alpha_action_epochs": int(getattr(args, "query_alpha_action_epochs", 200)),
        "query_alpha_action_learning_rate": float(getattr(args, "query_alpha_action_learning_rate", 0.05)),
        "query_alpha_action_weight_decay": float(getattr(args, "query_alpha_action_weight_decay", 1e-3)),
    }

    predicted_hits = 0
    oracle_hits = 0
    predicted_counts: Counter[int] = Counter()
    target_correct = 0
    for case, target_idx in zip(action_cases, targets):
        predicted_alpha, info = predict_learned_alpha_action(
            records=case["scored_records"],
            source_maps_by_label=source_maps_by_label,
            qid=str(case["qid"]),
            adaptive_config=config,
            fallback_alpha=float(global_fallback_alpha),
        )
        predicted_idx = int(info.get("action_idx", -1))
        predicted_counts[predicted_idx] += 1
        if predicted_idx == int(target_idx):
            target_correct += 1
        oracle_alpha = float(alpha_grid[int(target_idx)])
        for alpha, key in [(predicted_alpha, "predicted"), (oracle_alpha, "oracle")]:
            candidate_args.blend_alpha = float(alpha)
            candidate_records = [dict(row) for row in case["scored_records"]]
            ranked = rerank_records(candidate_records, candidate_args)
            ranked_uids = [str(row["uid"]) for row in ranked]
            hit = any(uid in case["pages_gold"] for uid in ranked_uids[:hit_k])
            if key == "predicted" and hit:
                predicted_hits += 1
            if key == "oracle" and hit:
                oracle_hits += 1

    evaluated = max(1, len(targets))
    config.update(
        {
            "tune_eval_qid_count": int(len(targets)),
            "predicted_action_distribution": {
                f"{float(alpha_grid[idx]):.8g}": int(predicted_counts.get(idx, 0))
                for idx in range(len(alpha_grid))
            },
            f"predicted_page@{hit_k}": float(predicted_hits) / float(evaluated),
            f"oracle_page@{hit_k}": float(oracle_hits) / float(evaluated),
            "target_action_accuracy": float(target_correct) / float(evaluated),
            "target_examples": target_rows[:20],
        }
    )
    return config


def parse_utility_threshold_grid(raw: str) -> list[float]:
    values: list[float] = []
    seen: set[float] = set()
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        key = round(value, 8)
        if key in seen:
            continue
        seen.add(key)
        values.append(value)
    if not values:
        raise ValueError("Empty alpha utility threshold grid.")
    return values


def parse_positive_float_grid(raw: str, *, name: str) -> list[float]:
    values: list[float] = []
    seen: set[float] = set()
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value < 0.0:
            raise ValueError(f"{name} values must be nonnegative, got {value}")
        key = round(value, 8)
        if key in seen:
            continue
        seen.add(key)
        values.append(value)
    if not values:
        raise ValueError(f"Empty {name} grid.")
    return values


def alpha_utility_feature_vector(query_features: list[float], alpha: float) -> list[float]:
    alpha = float(alpha)
    return (
        [float(value) for value in query_features]
        + [alpha, alpha * alpha]
        + [float(value) * alpha for value in query_features]
    )


def fit_alpha_utility_ridge(
    X: np.ndarray,
    y: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    X_std, mean, std = standardize_train(X)
    X_aug = np.concatenate([X_std, np.ones((X_std.shape[0], 1), dtype=np.float32)], axis=1)
    reg = np.eye(X_aug.shape[1], dtype=np.float32) * max(0.0, float(ridge))
    reg[-1, -1] = 0.0
    lhs = X_aug.T @ X_aug + reg
    rhs = X_aug.T @ y
    try:
        coef = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return coef[:-1].astype(np.float32), float(coef[-1]), mean, std


def predict_learned_alpha_utility_gate(
    *,
    records: list[dict[str, Any]],
    source_maps_by_label: dict[str, dict[str, dict[str, float]]],
    qid: str,
    adaptive_config: dict[str, Any] | None,
    fallback_alpha: float,
) -> tuple[float, dict[str, Any]]:
    if not adaptive_config or adaptive_config.get("mode") != "learned_alpha_utility_gate":
        return float(fallback_alpha), {"mode": None}

    alpha_grid = [float(value) for value in adaptive_config.get("alpha_grid", [])]
    if not alpha_grid:
        return 0.0, {"mode": "learned_alpha_utility_gate", "error": "empty_alpha_grid"}

    feature_top_k = int(adaptive_config.get("query_alpha_feature_top_k", 20))
    query_features = query_alpha_feature_vector(records, source_maps_by_label, qid, feature_top_k)
    mean = np.asarray(adaptive_config.get("feature_mean", []), dtype=np.float32)
    std = np.asarray(adaptive_config.get("feature_std", []), dtype=np.float32)
    weights = np.asarray(adaptive_config.get("weights", []), dtype=np.float32)
    bias = float(adaptive_config.get("bias", 0.0))
    if len(ALPHA_UTILITY_FEATURE_NAMES) != len(mean) or len(mean) != len(std) or len(weights) != len(mean):
        return 0.0, {"mode": "learned_alpha_utility_gate", "error": "feature_shape_mismatch"}

    threshold = float(adaptive_config.get("utility_threshold", 0.0))
    utilities: list[dict[str, float]] = []
    for alpha in alpha_grid:
        if math.isclose(float(alpha), 0.0):
            utility = 0.0
        else:
            features = alpha_utility_feature_vector(query_features, float(alpha))
            X = (np.asarray(features, dtype=np.float32) - mean) / std
            utility = float(X @ weights + np.float32(bias))
        utilities.append({"alpha": float(alpha), "predicted_utility": float(utility)})

    eligible = [
        row
        for row in utilities
        if not math.isclose(float(row["alpha"]), 0.0) and float(row["predicted_utility"]) > threshold
    ]
    if not eligible:
        selected_alpha = 0.0
        selected_utility = 0.0
        reason = "no_positive_predicted_utility"
    else:
        selected = max(eligible, key=lambda row: (float(row["predicted_utility"]), -float(row["alpha"])))
        selected_alpha = float(selected["alpha"])
        selected_utility = float(selected["predicted_utility"])
        reason = "positive_predicted_utility"

    return selected_alpha, {
        "mode": "learned_alpha_utility_gate",
        "utility_threshold": threshold,
        "selected_predicted_utility": selected_utility,
        "selection_reason": reason,
        "top_utilities": sorted(
            utilities,
            key=lambda row: (-float(row["predicted_utility"]), float(row["alpha"])),
        )[:5],
    }


def predict_base_aware_alpha_utility_gate(
    *,
    records: list[dict[str, Any]],
    source_maps_by_label: dict[str, dict[str, dict[str, float]]],
    qid: str,
    adaptive_config: dict[str, Any] | None,
    fallback_alpha: float,
) -> tuple[float, dict[str, Any]]:
    if not adaptive_config or adaptive_config.get("mode") != "base_aware_alpha_utility_gate":
        return float(fallback_alpha), {"mode": None}

    alpha_grid = [float(value) for value in adaptive_config.get("alpha_grid", [])]
    if not alpha_grid:
        return 0.0, {"mode": "base_aware_alpha_utility_gate", "error": "empty_alpha_grid"}

    feature_top_k = int(adaptive_config.get("query_alpha_feature_top_k", 20))
    query_features = query_alpha_feature_vector(records, source_maps_by_label, qid, feature_top_k)
    recovery_mean = np.asarray(adaptive_config.get("recovery_feature_mean", []), dtype=np.float32)
    recovery_std = np.asarray(adaptive_config.get("recovery_feature_std", []), dtype=np.float32)
    recovery_weights = np.asarray(adaptive_config.get("recovery_weights", []), dtype=np.float32)
    recovery_bias = float(adaptive_config.get("recovery_bias", 0.0))
    loss_mean = np.asarray(adaptive_config.get("loss_feature_mean", []), dtype=np.float32)
    loss_std = np.asarray(adaptive_config.get("loss_feature_std", []), dtype=np.float32)
    loss_weights = np.asarray(adaptive_config.get("loss_weights", []), dtype=np.float32)
    loss_bias = float(adaptive_config.get("loss_bias", 0.0))
    expected_len = len(ALPHA_UTILITY_FEATURE_NAMES)
    if (
        len(recovery_mean) != expected_len
        or len(recovery_std) != expected_len
        or len(recovery_weights) != expected_len
        or len(loss_mean) != expected_len
        or len(loss_std) != expected_len
        or len(loss_weights) != expected_len
    ):
        return 0.0, {"mode": "base_aware_alpha_utility_gate", "error": "feature_shape_mismatch"}

    threshold = float(adaptive_config.get("utility_threshold", 0.0))
    risk_penalty = float(adaptive_config.get("loss_risk_penalty", 1.0))
    utilities: list[dict[str, float]] = []
    for alpha in alpha_grid:
        if math.isclose(float(alpha), 0.0):
            predicted_recovery = 0.0
            predicted_loss = 0.0
            utility = 0.0
        else:
            features = np.asarray(alpha_utility_feature_vector(query_features, float(alpha)), dtype=np.float32)
            recovery_x = (features - recovery_mean) / recovery_std
            loss_x = (features - loss_mean) / loss_std
            predicted_recovery = float(recovery_x @ recovery_weights + np.float32(recovery_bias))
            predicted_loss = float(loss_x @ loss_weights + np.float32(loss_bias))
            utility = predicted_recovery - risk_penalty * predicted_loss
        utilities.append(
            {
                "alpha": float(alpha),
                "predicted_recovery": float(predicted_recovery),
                "predicted_loss": float(predicted_loss),
                "predicted_utility": float(utility),
            }
        )

    eligible = [
        row
        for row in utilities
        if not math.isclose(float(row["alpha"]), 0.0) and float(row["predicted_utility"]) > threshold
    ]
    if not eligible:
        selected_alpha = 0.0
        selected = {"predicted_recovery": 0.0, "predicted_loss": 0.0, "predicted_utility": 0.0}
        reason = "no_safe_positive_predicted_utility"
    else:
        selected = max(eligible, key=lambda row: (float(row["predicted_utility"]), -float(row["alpha"])))
        selected_alpha = float(selected["alpha"])
        reason = "safe_positive_predicted_utility"

    return selected_alpha, {
        "mode": "base_aware_alpha_utility_gate",
        "utility_threshold": threshold,
        "loss_risk_penalty": risk_penalty,
        "selected_predicted_recovery": float(selected["predicted_recovery"]),
        "selected_predicted_loss": float(selected["predicted_loss"]),
        "selected_predicted_utility": float(selected["predicted_utility"]),
        "selection_reason": reason,
        "top_utilities": sorted(
            utilities,
            key=lambda row: (-float(row["predicted_utility"]), float(row["alpha"])),
        )[:5],
    }


def fit_learned_alpha_utility_gate_config(
    *,
    tune_cases: list[dict[str, Any]],
    alpha_grid: list[float],
    hit_k: int,
    candidate_args: argparse.Namespace,
    source_maps_by_label: dict[str, dict[str, dict[str, float]]],
    args: argparse.Namespace,
    global_fallback_alpha: float,
) -> dict[str, Any]:
    if not any(math.isclose(float(alpha), 0.0) for alpha in alpha_grid):
        alpha_grid = [0.0] + list(alpha_grid)
    alpha_grid = sorted({round(float(alpha), 8) for alpha in alpha_grid})
    feature_top_k = int(getattr(args, "query_alpha_feature_top_k", 20))

    vectors: list[list[float]] = []
    targets: list[float] = []
    utility_cases: list[dict[str, Any]] = []
    target_counter: Counter[float] = Counter()
    target_rows: list[dict[str, Any]] = []

    for case in tune_cases:
        query_features = query_alpha_feature_vector(
            case["scored_records"],
            source_maps_by_label,
            str(case["qid"]),
            feature_top_k,
        )
        alpha_hits: dict[float, bool] = {}
        alpha_ranks: dict[float, int | None] = {}
        for alpha in alpha_grid:
            candidate_args.blend_alpha = float(alpha)
            candidate_records = [dict(row) for row in case["scored_records"]]
            ranked = rerank_records(candidate_records, candidate_args)
            ranked_uids = [str(row["uid"]) for row in ranked]
            rank = first_rank(ranked_uids, case["pages_gold"])
            alpha_ranks[float(alpha)] = rank
            alpha_hits[float(alpha)] = bool(rank is not None and rank <= int(hit_k))

        base_hit = bool(alpha_hits.get(0.0, False))
        for alpha in alpha_grid:
            hit = bool(alpha_hits[float(alpha)])
            target = float(int(hit) - int(base_hit))
            vectors.append(alpha_utility_feature_vector(query_features, float(alpha)))
            targets.append(target)
            target_counter[target] += 1
        utility_cases.append(
            {
                "qid": str(case["qid"]),
                "scored_records": case["scored_records"],
                "pages_gold": case["pages_gold"],
                "base_hit": base_hit,
                "query_features": query_features,
            }
        )
        if len(target_rows) < 20:
            best_alpha = max(
                alpha_grid,
                key=lambda alpha: (int(alpha_hits[float(alpha)]), -float(alpha)),
            )
            target_rows.append(
                {
                    "qid": str(case["qid"]),
                    "base_hit": base_hit,
                    "best_alpha_by_hit": float(best_alpha),
                    "best_rank": alpha_ranks[float(best_alpha)],
                    "alpha_hits": {f"{float(alpha):.8g}": bool(alpha_hits[float(alpha)]) for alpha in alpha_grid},
                }
            )

    if not vectors:
        return {
            "mode": "learned_alpha_utility_gate",
            "error": "no_tune_cases",
            "global_fallback_blend_alpha": float(global_fallback_alpha),
        }

    X = np.asarray(vectors, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    ridge = max(0.0, float(getattr(args, "alpha_utility_ridge", 1.0)))
    weights, bias, mean, std = fit_alpha_utility_ridge(X, y, ridge)
    config: dict[str, Any] = {
        "mode": "learned_alpha_utility_gate",
        "query_alpha_feature_names": list(QUERY_ALPHA_FEATURE_NAMES),
        "alpha_utility_feature_names": list(ALPHA_UTILITY_FEATURE_NAMES),
        "query_alpha_feature_top_k": int(feature_top_k),
        "alpha_grid": [float(value) for value in alpha_grid],
        "weights": [float(value) for value in weights.tolist()],
        "bias": float(bias),
        "feature_mean": [float(value) for value in mean.tolist()],
        "feature_std": [float(value) for value in std.tolist()],
        "global_fallback_blend_alpha": float(global_fallback_alpha),
        "alpha_utility_ridge": float(ridge),
        "target_utility_distribution": {
            f"{float(target):.8g}": int(count)
            for target, count in sorted(target_counter.items(), key=lambda item: float(item[0]))
        },
        "target_examples": target_rows,
    }

    threshold_grid = parse_utility_threshold_grid(str(getattr(args, "alpha_utility_threshold_grid", "0.0")))
    threshold_scores: list[dict[str, Any]] = []
    for threshold in threshold_grid:
        config["utility_threshold"] = float(threshold)
        hits = 0
        recovered = 0
        lost = 0
        selected_alpha_sum = 0.0
        selected_counts: Counter[float] = Counter()
        for case in utility_cases:
            selected_alpha, _ = predict_learned_alpha_utility_gate(
                records=case["scored_records"],
                source_maps_by_label=source_maps_by_label,
                qid=str(case["qid"]),
                adaptive_config=config,
                fallback_alpha=float(global_fallback_alpha),
            )
            selected_counts[round(float(selected_alpha), 8)] += 1
            selected_alpha_sum += float(selected_alpha)
            candidate_args.blend_alpha = float(selected_alpha)
            candidate_records = [dict(row) for row in case["scored_records"]]
            ranked = rerank_records(candidate_records, candidate_args)
            ranked_uids = [str(row["uid"]) for row in ranked]
            selected_hit = any(uid in case["pages_gold"] for uid in ranked_uids[:hit_k])
            hits += int(selected_hit)
            if selected_hit and not bool(case["base_hit"]):
                recovered += 1
            if bool(case["base_hit"]) and not selected_hit:
                lost += 1
        evaluated = max(1, len(utility_cases))
        threshold_scores.append(
            {
                "utility_threshold": float(threshold),
                f"page@{hit_k}": float(hits) / float(evaluated),
                "hit_count": int(hits),
                "recovered_hit_count": int(recovered),
                "lost_hit_count": int(lost),
                "net_hit_count": int(recovered - lost),
                "mean_selected_alpha": float(selected_alpha_sum) / float(evaluated),
                "selected_alpha_distribution": {
                    f"{float(alpha):.8g}": int(count)
                    for alpha, count in sorted(selected_counts.items(), key=lambda item: float(item[0]))
                },
            }
        )

    selected_threshold = max(
        threshold_scores,
        key=lambda row: (
            float(row[f"page@{hit_k}"]),
            int(row["net_hit_count"]),
            -int(row["lost_hit_count"]),
            -float(row["mean_selected_alpha"]),
            float(row["utility_threshold"]),
        ),
    )
    config.update(
        {
            "utility_threshold": float(selected_threshold["utility_threshold"]),
            "tune_eval_qid_count": int(len(utility_cases)),
            "threshold_scores": threshold_scores,
            "selected_threshold_score": selected_threshold,
        }
    )
    return config


def fit_base_aware_alpha_utility_gate_config(
    *,
    tune_cases: list[dict[str, Any]],
    alpha_grid: list[float],
    hit_k: int,
    candidate_args: argparse.Namespace,
    source_maps_by_label: dict[str, dict[str, dict[str, float]]],
    args: argparse.Namespace,
    global_fallback_alpha: float,
) -> dict[str, Any]:
    if not any(math.isclose(float(alpha), 0.0) for alpha in alpha_grid):
        alpha_grid = [0.0] + list(alpha_grid)
    alpha_grid = sorted({round(float(alpha), 8) for alpha in alpha_grid})
    feature_top_k = int(getattr(args, "query_alpha_feature_top_k", 20))

    vectors: list[list[float]] = []
    recovery_targets: list[float] = []
    loss_targets: list[float] = []
    utility_cases: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    target_counter: Counter[str] = Counter()

    for case in tune_cases:
        query_features = query_alpha_feature_vector(
            case["scored_records"],
            source_maps_by_label,
            str(case["qid"]),
            feature_top_k,
        )
        alpha_hits: dict[float, bool] = {}
        alpha_ranks: dict[float, int | None] = {}
        for alpha in alpha_grid:
            candidate_args.blend_alpha = float(alpha)
            candidate_records = [dict(row) for row in case["scored_records"]]
            ranked = rerank_records(candidate_records, candidate_args)
            ranked_uids = [str(row["uid"]) for row in ranked]
            rank = first_rank(ranked_uids, case["pages_gold"])
            alpha_ranks[float(alpha)] = rank
            alpha_hits[float(alpha)] = bool(rank is not None and rank <= int(hit_k))

        base_hit = bool(alpha_hits.get(0.0, False))
        for alpha in alpha_grid:
            hit = bool(alpha_hits[float(alpha)])
            recovery = float(int(hit and not base_hit))
            loss = float(int(base_hit and not hit))
            vectors.append(alpha_utility_feature_vector(query_features, float(alpha)))
            recovery_targets.append(recovery)
            loss_targets.append(loss)
            if recovery:
                target_counter["recovery"] += 1
            elif loss:
                target_counter["loss"] += 1
            else:
                target_counter["neutral"] += 1
        utility_cases.append(
            {
                "qid": str(case["qid"]),
                "scored_records": case["scored_records"],
                "pages_gold": case["pages_gold"],
                "base_hit": base_hit,
            }
        )
        if len(target_rows) < 20:
            best_alpha = max(
                alpha_grid,
                key=lambda alpha: (
                    int(alpha_hits[float(alpha)] and not base_hit),
                    int(alpha_hits[float(alpha)]),
                    -float(alpha),
                ),
            )
            target_rows.append(
                {
                    "qid": str(case["qid"]),
                    "base_hit": base_hit,
                    "best_alpha_by_recovery": float(best_alpha),
                    "best_rank": alpha_ranks[float(best_alpha)],
                    "alpha_hits": {f"{float(alpha):.8g}": bool(alpha_hits[float(alpha)]) for alpha in alpha_grid},
                }
            )

    if not vectors:
        return {
            "mode": "base_aware_alpha_utility_gate",
            "error": "no_tune_cases",
            "global_fallback_blend_alpha": float(global_fallback_alpha),
        }

    X = np.asarray(vectors, dtype=np.float32)
    recovery_y = np.asarray(recovery_targets, dtype=np.float32)
    loss_y = np.asarray(loss_targets, dtype=np.float32)
    ridge = max(0.0, float(getattr(args, "alpha_utility_ridge", 1.0)))
    recovery_weights, recovery_bias, recovery_mean, recovery_std = fit_alpha_utility_ridge(X, recovery_y, ridge)
    loss_weights, loss_bias, loss_mean, loss_std = fit_alpha_utility_ridge(X, loss_y, ridge)

    config: dict[str, Any] = {
        "mode": "base_aware_alpha_utility_gate",
        "query_alpha_feature_names": list(QUERY_ALPHA_FEATURE_NAMES),
        "alpha_utility_feature_names": list(ALPHA_UTILITY_FEATURE_NAMES),
        "query_alpha_feature_top_k": int(feature_top_k),
        "alpha_grid": [float(value) for value in alpha_grid],
        "recovery_weights": [float(value) for value in recovery_weights.tolist()],
        "recovery_bias": float(recovery_bias),
        "recovery_feature_mean": [float(value) for value in recovery_mean.tolist()],
        "recovery_feature_std": [float(value) for value in recovery_std.tolist()],
        "loss_weights": [float(value) for value in loss_weights.tolist()],
        "loss_bias": float(loss_bias),
        "loss_feature_mean": [float(value) for value in loss_mean.tolist()],
        "loss_feature_std": [float(value) for value in loss_std.tolist()],
        "global_fallback_blend_alpha": float(global_fallback_alpha),
        "alpha_utility_ridge": float(ridge),
        "target_recovery_loss_distribution": {key: int(value) for key, value in sorted(target_counter.items())},
        "target_examples": target_rows,
    }

    threshold_grid = parse_utility_threshold_grid(str(getattr(args, "alpha_utility_threshold_grid", "0.0")))
    risk_penalty_grid = parse_positive_float_grid(
        str(getattr(args, "alpha_utility_risk_penalty_grid", "1.0")),
        name="alpha utility risk penalty",
    )
    threshold_scores: list[dict[str, Any]] = []
    for risk_penalty in risk_penalty_grid:
        config["loss_risk_penalty"] = float(risk_penalty)
        for threshold in threshold_grid:
            config["utility_threshold"] = float(threshold)
            hits = 0
            recovered = 0
            lost = 0
            selected_alpha_sum = 0.0
            selected_counts: Counter[float] = Counter()
            for case in utility_cases:
                selected_alpha, _ = predict_base_aware_alpha_utility_gate(
                    records=case["scored_records"],
                    source_maps_by_label=source_maps_by_label,
                    qid=str(case["qid"]),
                    adaptive_config=config,
                    fallback_alpha=float(global_fallback_alpha),
                )
                selected_counts[round(float(selected_alpha), 8)] += 1
                selected_alpha_sum += float(selected_alpha)
                candidate_args.blend_alpha = float(selected_alpha)
                candidate_records = [dict(row) for row in case["scored_records"]]
                ranked = rerank_records(candidate_records, candidate_args)
                ranked_uids = [str(row["uid"]) for row in ranked]
                selected_hit = any(uid in case["pages_gold"] for uid in ranked_uids[:hit_k])
                hits += int(selected_hit)
                if selected_hit and not bool(case["base_hit"]):
                    recovered += 1
                if bool(case["base_hit"]) and not selected_hit:
                    lost += 1
            evaluated = max(1, len(utility_cases))
            threshold_scores.append(
                {
                    "utility_threshold": float(threshold),
                    "loss_risk_penalty": float(risk_penalty),
                    f"page@{hit_k}": float(hits) / float(evaluated),
                    "hit_count": int(hits),
                    "recovered_hit_count": int(recovered),
                    "lost_hit_count": int(lost),
                    "net_hit_count": int(recovered - lost),
                    "mean_selected_alpha": float(selected_alpha_sum) / float(evaluated),
                    "selected_alpha_distribution": {
                        f"{float(alpha):.8g}": int(count)
                        for alpha, count in sorted(selected_counts.items(), key=lambda item: float(item[0]))
                    },
                }
            )

    selected_threshold = max(
        threshold_scores,
        key=lambda row: (
            float(row[f"page@{hit_k}"]),
            int(row["net_hit_count"]),
            -int(row["lost_hit_count"]),
            -float(row["mean_selected_alpha"]),
            float(row["loss_risk_penalty"]),
            float(row["utility_threshold"]),
        ),
    )
    config.update(
        {
            "utility_threshold": float(selected_threshold["utility_threshold"]),
            "loss_risk_penalty": float(selected_threshold["loss_risk_penalty"]),
            "tune_eval_qid_count": int(len(utility_cases)),
            "threshold_scores": threshold_scores,
            "selected_threshold_score": selected_threshold,
        }
    )
    return config


def doc_head_rerank_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    doc_order: list[str] = []
    rows_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        doc_id = str(row["doc_id"])
        if doc_id not in rows_by_doc:
            doc_order.append(doc_id)
        rows_by_doc[doc_id].append(row)

    head_by_doc: dict[str, dict[str, Any]] = {}
    for doc_id, rows in rows_by_doc.items():
        head_by_doc[doc_id] = max(
            rows,
            key=lambda row: (
                float(row.get("rerank_score", row.get("learned_score", 0.0))),
                -int(row["base_rank"]),
            ),
        )

    output: list[dict[str, Any]] = []
    selected_uids: set[str] = set()
    for doc_id in doc_order:
        head = head_by_doc[doc_id]
        output.append(head)
        selected_uids.add(str(head["uid"]))
    output.extend(row for row in records if str(row["uid"]) not in selected_uids)
    return output


def doc_slot_rerank_records(records: list[dict[str, Any]], max_rank: int) -> list[dict[str, Any]]:
    eligible_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    eligible_uids: set[str] = set()
    for row in records:
        if int(row["base_rank"]) > int(max_rank):
            continue
        doc_id = str(row["doc_id"])
        eligible_by_doc[doc_id].append(row)
        eligible_uids.add(str(row["uid"]))

    sorted_by_doc: dict[str, list[dict[str, Any]]] = {}
    for doc_id, rows in eligible_by_doc.items():
        sorted_by_doc[doc_id] = sorted(
            rows,
            key=lambda row: (
                -float(row.get("rerank_score", row.get("learned_score", 0.0))),
                int(row["base_rank"]),
                str(row["uid"]),
            ),
        )

    next_idx_by_doc: Counter[str] = Counter()
    output: list[dict[str, Any]] = []
    for row in records:
        uid = str(row["uid"])
        if uid not in eligible_uids:
            output.append(row)
            continue
        doc_id = str(row["doc_id"])
        idx = next_idx_by_doc[doc_id]
        output.append(sorted_by_doc[doc_id][idx])
        next_idx_by_doc[doc_id] += 1
    return output


def rerank_records(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if not records:
        return []
    if args.inference_mode == "full_rerank":
        return sorted(
            records,
            key=lambda row: (-float(row.get("learned_score", 0.0)), int(row["base_rank"]), row["uid"]),
        )
    if args.inference_mode == "safe_promote":
        anchors = records[: int(args.anchor_top_k)]
        rest = records[int(args.anchor_top_k) :]
        weakest_anchor_score = min((float(row.get("learned_score", 0.0)) for row in anchors), default=1.0)
        eligible = [
            row
            for row in rest
            if int(args.promotion_rank_min) <= int(row["base_rank"]) <= int(args.promotion_rank_max)
            and float(row.get("learned_score", 0.0)) >= weakest_anchor_score + float(args.promotion_margin)
        ]
        eligible.sort(key=lambda row: (-float(row.get("learned_score", 0.0)), int(row["base_rank"]), row["uid"]))
        promotions = eligible[: int(args.max_promotions_per_qid)]
        promoted_uids = {row["uid"] for row in promotions}
        return promotions + [row for row in records if row["uid"] not in promoted_uids]

    assign_blend_scores(records, float(args.blend_alpha))
    if args.inference_mode == "doc_head_blend":
        return doc_head_rerank_records(records)
    if args.inference_mode == "doc_slot_blend":
        return doc_slot_rerank_records(records, int(args.promotion_rank_max))
    return sorted(
        records,
        key=lambda row: (-float(row["rerank_score"]), int(row["base_rank"]), row["uid"]),
    )


def parse_alpha_grid(raw: str) -> list[float]:
    values: list[float] = []
    seen: set[float] = set()
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Blend alpha must be in [0, 1], got {value}")
        key = round(value, 8)
        if key in seen:
            continue
        seen.add(key)
        values.append(value)
    if not values:
        raise ValueError("Empty tune blend-alpha grid.")
    return values


def tune_blend_alpha(
    *,
    tune_gold: dict[str, dict[str, Any]],
    base_pred: dict[str, dict[str, Any]],
    page_features: dict[str, dict[str, Any]],
    source_maps_by_label: dict[str, dict[str, dict[str, float]]],
    mean: np.ndarray,
    std: np.ndarray,
    weights: np.ndarray,
    bias: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    alpha_grid = parse_alpha_grid(str(args.tune_blend_alpha_grid))
    if (
        bool(getattr(args, "learned_alpha_utility_gate", False))
        or bool(getattr(args, "base_aware_alpha_utility_gate", False))
    ) and not any(
        math.isclose(float(alpha), 0.0) for alpha in alpha_grid
    ):
        alpha_grid = [0.0] + alpha_grid
    hit_k = int(args.tune_hit_k)
    if hit_k <= 0:
        raise ValueError(f"tune_hit_k must be positive, got {hit_k}")
    candidate_args = copy.copy(args)
    candidate_args.inference_mode = "blend_rerank"

    hits = {alpha: 0 for alpha in alpha_grid}
    tune_cases: list[dict[str, Any]] = []
    evaluated = 0
    skipped_no_page_gold = 0
    skipped_missing_prediction = 0
    for qid, gold_row in tune_gold.items():
        pages_gold = gold_page_uids(gold_row)
        if not pages_gold:
            skipped_no_page_gold += 1
            continue
        records = ranked_page_records(base_pred.get(qid), int(args.candidate_top_k))
        if not records:
            skipped_missing_prediction += 1
            continue
        scored_records = score_records(
            qid=qid,
            gold_row=gold_row,
            records=records,
            page_features=page_features,
            source_maps_by_label=source_maps_by_label,
            mean=mean,
            std=std,
            weights=weights,
            bias=bias,
            feature_names=getattr(args, "active_feature_names", FEATURE_NAMES),
        )
        evaluated += 1
        confidence = query_confidence_score(records, source_maps_by_label, qid)
        tune_cases.append(
            {
                "qid": qid,
                "pages_gold": pages_gold,
                "scored_records": scored_records,
                "confidence": float(confidence),
            }
        )
        for alpha in alpha_grid:
            candidate_args.blend_alpha = float(alpha)
            candidate_records = [dict(row) for row in scored_records]
            ranked = rerank_records(candidate_records, candidate_args)
            ranked_uids = [str(row["uid"]) for row in ranked]
            if any(uid in pages_gold for uid in ranked_uids[:hit_k]):
                hits[alpha] += 1

    if evaluated <= 0:
        raise ValueError("No held-out train qids with pseudo-page labels were available for blend-alpha tuning.")

    scores = [
        {
            "blend_alpha": float(alpha),
            f"page@{hit_k}": float(hits[alpha]) / float(evaluated) if evaluated else 0.0,
            "hit_count": int(hits[alpha]),
        }
        for alpha in alpha_grid
    ]
    best = max(scores, key=lambda row: (float(row[f"page@{hit_k}"]), -float(row["blend_alpha"])))
    adaptive_config: dict[str, Any] | None = None
    if bool(getattr(args, "learned_alpha_utility_gate", False)):
        adaptive_config = fit_learned_alpha_utility_gate_config(
            tune_cases=tune_cases,
            alpha_grid=alpha_grid,
            hit_k=hit_k,
            candidate_args=candidate_args,
            source_maps_by_label=source_maps_by_label,
            args=args,
            global_fallback_alpha=float(best["blend_alpha"]),
        )
    elif bool(getattr(args, "base_aware_alpha_utility_gate", False)):
        adaptive_config = fit_base_aware_alpha_utility_gate_config(
            tune_cases=tune_cases,
            alpha_grid=alpha_grid,
            hit_k=hit_k,
            candidate_args=candidate_args,
            source_maps_by_label=source_maps_by_label,
            args=args,
            global_fallback_alpha=float(best["blend_alpha"]),
        )
    elif bool(getattr(args, "learned_alpha_action", False)):
        adaptive_config = fit_learned_alpha_action_config(
            tune_cases=tune_cases,
            alpha_grid=alpha_grid,
            hit_k=hit_k,
            candidate_args=candidate_args,
            source_maps_by_label=source_maps_by_label,
            args=args,
            global_fallback_alpha=float(best["blend_alpha"]),
        )
    elif bool(getattr(args, "learned_query_alpha", False)):
        adaptive_config = fit_learned_query_alpha_config(
            tune_cases=tune_cases,
            alpha_grid=alpha_grid,
            hit_k=hit_k,
            candidate_args=candidate_args,
            source_maps_by_label=source_maps_by_label,
            args=args,
            global_fallback_alpha=float(best["blend_alpha"]),
        )
    elif bool(getattr(args, "query_adaptive_alpha", False)):
        bin_count = max(1, int(getattr(args, "query_alpha_bins", 3)))
        confidence_scores = [float(case["confidence"]) for case in tune_cases]
        thresholds = fit_confidence_thresholds(confidence_scores, bin_count)
        effective_bin_count = len(thresholds) + 1
        bin_hits = {
            bin_idx: {alpha: 0 for alpha in alpha_grid}
            for bin_idx in range(effective_bin_count)
        }
        bin_eval = Counter()
        for case in tune_cases:
            bin_idx = confidence_bin(float(case["confidence"]), thresholds)
            bin_eval[bin_idx] += 1
            for alpha in alpha_grid:
                candidate_args.blend_alpha = float(alpha)
                candidate_records = [dict(row) for row in case["scored_records"]]
                ranked = rerank_records(candidate_records, candidate_args)
                ranked_uids = [str(row["uid"]) for row in ranked]
                if any(uid in case["pages_gold"] for uid in ranked_uids[:hit_k]):
                    bin_hits[bin_idx][alpha] += 1

        bin_scores: list[dict[str, Any]] = []
        bin_selected_alphas: list[float] = []
        for bin_idx in range(effective_bin_count):
            evaluated_in_bin = int(bin_eval.get(bin_idx, 0))
            if evaluated_in_bin <= 0:
                selected_alpha = float(best["blend_alpha"])
                alpha_scores = []
            else:
                alpha_scores = [
                    {
                        "blend_alpha": float(alpha),
                        f"page@{hit_k}": float(bin_hits[bin_idx][alpha]) / float(evaluated_in_bin),
                        "hit_count": int(bin_hits[bin_idx][alpha]),
                    }
                    for alpha in alpha_grid
                ]
                selected = max(
                    alpha_scores,
                    key=lambda row: (float(row[f"page@{hit_k}"]), -float(row["blend_alpha"])),
                )
                selected_alpha = float(selected["blend_alpha"])
            bin_selected_alphas.append(selected_alpha)
            bin_scores.append(
                {
                    "bin": int(bin_idx),
                    "confidence_lower": None if bin_idx == 0 else float(thresholds[bin_idx - 1]),
                    "confidence_upper": None if bin_idx >= len(thresholds) else float(thresholds[bin_idx]),
                    "selected_blend_alpha": selected_alpha,
                    "tune_eval_qid_count": evaluated_in_bin,
                    "alpha_scores": alpha_scores,
                }
            )
        adaptive_config = {
            "mode": "confidence_bins",
            "query_alpha_bins_requested": int(getattr(args, "query_alpha_bins", 3)),
            "query_alpha_bins_effective": int(effective_bin_count),
            "confidence_thresholds": [float(value) for value in thresholds],
            "bin_selected_blend_alpha": bin_selected_alphas,
            "global_fallback_blend_alpha": float(best["blend_alpha"]),
            "bin_scores": bin_scores,
        }

    return {
        "selected_blend_alpha": float(best["blend_alpha"]),
        "optimized_metric": f"page@{hit_k}",
        "optimized_metric_value": float(best[f"page@{hit_k}"]),
        "tune_eval_qid_count": int(evaluated),
        "skipped_no_page_gold": int(skipped_no_page_gold),
        "skipped_missing_prediction": int(skipped_missing_prediction),
        "alpha_scores": scores,
        "query_adaptive_alpha": bool(getattr(args, "query_adaptive_alpha", False)),
        "learned_query_alpha": bool(getattr(args, "learned_query_alpha", False)),
        "learned_alpha_action": bool(getattr(args, "learned_alpha_action", False)),
        "learned_alpha_utility_gate": bool(getattr(args, "learned_alpha_utility_gate", False)),
        "base_aware_alpha_utility_gate": bool(getattr(args, "base_aware_alpha_utility_gate", False)),
        "adaptive_alpha_config": adaptive_config,
    }


def apply_reranker(
    *,
    gold: dict[str, dict[str, Any]],
    base_pred: dict[str, dict[str, Any]],
    page_features: dict[str, dict[str, Any]],
    source_maps_by_label: dict[str, dict[str, dict[str, float]]],
    mean: np.ndarray,
    std: np.ndarray,
    weights: np.ndarray,
    bias: float,
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    output: dict[str, dict[str, Any]] = {}
    prior_rows: list[dict[str, Any]] = []
    for qid, base_row in base_pred.items():
        records = ranked_page_records(base_row, int(args.candidate_top_k))
        gold_row = gold.get(qid, {"qid": qid, "question": base_row.get("question", "")})
        scored_records = score_records(
            qid=qid,
            gold_row=gold_row,
            records=records,
            page_features=page_features,
            source_maps_by_label=source_maps_by_label,
            mean=mean,
            std=std,
            weights=weights,
            bias=bias,
            feature_names=getattr(args, "active_feature_names", FEATURE_NAMES),
        )
        apply_args = args
        query_alpha = float(args.blend_alpha)
        query_alpha_confidence: float | None = None
        query_alpha_bin: int | None = None
        query_alpha_info: dict[str, Any] = {}
        adaptive_config = getattr(args, "adaptive_alpha_config", None)
        if bool(getattr(args, "learned_alpha_utility_gate", False)):
            query_alpha_confidence = query_confidence_score(scored_records, source_maps_by_label, qid)
            query_alpha, query_alpha_info = predict_learned_alpha_utility_gate(
                records=scored_records,
                source_maps_by_label=source_maps_by_label,
                qid=qid,
                adaptive_config=adaptive_config,
                fallback_alpha=float(args.blend_alpha),
            )
            apply_args = copy.copy(args)
            apply_args.blend_alpha = float(query_alpha)
        elif bool(getattr(args, "base_aware_alpha_utility_gate", False)):
            query_alpha_confidence = query_confidence_score(scored_records, source_maps_by_label, qid)
            query_alpha, query_alpha_info = predict_base_aware_alpha_utility_gate(
                records=scored_records,
                source_maps_by_label=source_maps_by_label,
                qid=qid,
                adaptive_config=adaptive_config,
                fallback_alpha=float(args.blend_alpha),
            )
            apply_args = copy.copy(args)
            apply_args.blend_alpha = float(query_alpha)
        elif bool(getattr(args, "learned_alpha_action", False)):
            query_alpha_confidence = query_confidence_score(scored_records, source_maps_by_label, qid)
            query_alpha, query_alpha_info = predict_learned_alpha_action(
                records=scored_records,
                source_maps_by_label=source_maps_by_label,
                qid=qid,
                adaptive_config=adaptive_config,
                fallback_alpha=float(args.blend_alpha),
            )
            apply_args = copy.copy(args)
            apply_args.blend_alpha = float(query_alpha)
        elif bool(getattr(args, "learned_query_alpha", False)):
            query_alpha_confidence = query_confidence_score(scored_records, source_maps_by_label, qid)
            query_alpha, query_alpha_info = predict_learned_query_alpha(
                records=scored_records,
                source_maps_by_label=source_maps_by_label,
                qid=qid,
                adaptive_config=adaptive_config,
                fallback_alpha=float(args.blend_alpha),
            )
            apply_args = copy.copy(args)
            apply_args.blend_alpha = float(query_alpha)
        elif bool(getattr(args, "query_adaptive_alpha", False)):
            query_alpha_confidence = query_confidence_score(scored_records, source_maps_by_label, qid)
            query_alpha, query_alpha_bin = alpha_for_query_confidence(
                query_alpha_confidence,
                adaptive_config,
                float(args.blend_alpha),
            )
            query_alpha_info = {"mode": "confidence_bins"}
            apply_args = copy.copy(args)
            apply_args.blend_alpha = float(query_alpha)

        reranked = rerank_records(scored_records, apply_args)
        reranked_uids = {row["uid"] for row in reranked}
        raw_by_uid = {row["uid"]: row["raw"] for row in records}
        output_rows = [raw_by_uid[row["uid"]] for row in reranked if row["uid"] in raw_by_uid]
        output_rows.extend(
            raw_by_uid[row["uid"]]
            for row in records
            if row["uid"] not in reranked_uids and row["uid"] in raw_by_uid
        )
        out_row = dict(base_row)
        if "page_retrieval_results" in out_row:
            out_row["page_retrieval_results"] = output_rows
        elif "retrieval_results" in out_row:
            out_row["retrieval_results"] = output_rows
        else:
            out_row["page_retrieval_results"] = output_rows
        out_row["reranker_metadata"] = {
            **(out_row.get("reranker_metadata", {}) if isinstance(out_row.get("reranker_metadata"), dict) else {}),
            "content_aware_pseudo_page_reranker": {
                "inference_mode": args.inference_mode,
                "blend_alpha": float(query_alpha),
                "global_blend_alpha": float(args.blend_alpha),
                "query_adaptive_alpha": bool(getattr(args, "query_adaptive_alpha", False)),
                "learned_query_alpha": bool(getattr(args, "learned_query_alpha", False)),
                "learned_alpha_action": bool(getattr(args, "learned_alpha_action", False)),
                "learned_alpha_utility_gate": bool(getattr(args, "learned_alpha_utility_gate", False)),
                "base_aware_alpha_utility_gate": bool(getattr(args, "base_aware_alpha_utility_gate", False)),
                "query_alpha_mode": query_alpha_info.get("mode"),
                "query_alpha_confidence": query_alpha_confidence,
                "query_alpha_bin": query_alpha_bin,
                "query_alpha_info": query_alpha_info,
            },
        }
        output[qid] = out_row
        for row in scored_records:
            prior_rows.append(
                {
                    "qid": qid,
                    "page_uid": row["uid"],
                    "doc_id": row["doc_id"],
                    "page_idx": int(row["page_idx"]),
                    "base_rank": int(row["base_rank"]),
                    "learned_score": float(row.get("learned_score", 0.0)),
                }
            )
    return output, prior_rows


def ranked_pages(pred_row: dict[str, Any] | None) -> list[str]:
    pages = []
    seen = set()
    for row in prediction_rows(pred_row):
        parsed = parse_prediction_row(row)
        if parsed is None:
            continue
        uid = page_uid(parsed[0], parsed[1])
        if uid not in seen:
            seen.add(uid)
            pages.append(uid)
    return pages


def ranked_docs_from_pages(pages: list[str]) -> list[str]:
    docs = []
    seen = set()
    for uid in pages:
        parsed = parse_page_uid(uid)
        doc_id = parsed[0] if parsed else uid
        if doc_id not in seen:
            seen.add(doc_id)
            docs.append(doc_id)
    return docs


def recall_at(ranked: list[str], gold: set[str], k: int) -> float:
    return 1.0 if gold and any(item in gold for item in ranked[: int(k)]) else 0.0


def first_rank(ranked: list[str], gold: set[str]) -> int | None:
    for idx, item in enumerate(ranked, start=1):
        if item in gold:
            return idx
    return None


def evaluate_run(
    *,
    label: str,
    pred: dict[str, dict[str, Any]],
    gold: dict[str, dict[str, Any]],
    recall_ks: list[int],
) -> dict[str, Any]:
    page_hits = {k: 0.0 for k in recall_ks}
    doc_hits = {k: 0.0 for k in recall_ks}
    n = 0
    skipped_no_page_gold = 0
    for qid, gold_row in gold.items():
        pages_gold = gold_page_uids(gold_row)
        if not pages_gold:
            skipped_no_page_gold += 1
            continue
        docs_gold = gold_doc_ids(gold_row)
        pages = ranked_pages(pred.get(qid))
        docs = ranked_docs_from_pages(pages)
        n += 1
        for k in recall_ks:
            page_hits[k] += recall_at(pages, pages_gold, k)
            doc_hits[k] += recall_at(docs, docs_gold, k)
    result = {
        "label": label,
        "qid_count": int(len(gold)),
        "eval_qid_count": int(n),
        "skipped_no_page_gold": int(skipped_no_page_gold),
    }
    for k in recall_ks:
        result[f"page@{k}"] = page_hits[k] / float(n) if n else 0.0
        result[f"doc@{k}"] = doc_hits[k] / float(n) if n else 0.0
    return result


def movement_vs_base(
    *,
    base_pred: dict[str, dict[str, Any]],
    candidate_pred: dict[str, dict[str, Any]],
    gold: dict[str, dict[str, Any]],
    hit_k: int = 4,
) -> dict[str, Any]:
    counts = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for qid, gold_row in gold.items():
        pages_gold = gold_page_uids(gold_row)
        if not pages_gold:
            continue
        base_rank = first_rank(ranked_pages(base_pred.get(qid)), pages_gold)
        cand_rank = first_rank(ranked_pages(candidate_pred.get(qid)), pages_gold)
        base_hit = base_rank is not None and base_rank <= hit_k
        cand_hit = cand_rank is not None and cand_rank <= hit_k
        if base_hit and not cand_hit:
            key = "lost"
        elif not base_hit and cand_hit:
            key = "recovered"
        elif base_rank == cand_rank:
            key = "unchanged"
        elif cand_rank is None or (base_rank is not None and cand_rank > base_rank):
            key = "worsened_rank"
        else:
            key = "improved_rank"
        counts[key] += 1
        if len(examples[key]) < 5:
            examples[key].append({"qid": qid, "base_rank": base_rank, "candidate_rank": cand_rank})
    return {"counts": dict(counts), "examples": dict(examples)}


def write_table(path: Path, rows: list[dict[str, Any]], recall_ks: list[int]) -> None:
    headers = ["label", "eval_qid_count"]
    for k in recall_ks:
        headers.extend([f"page@{k}", f"doc@{k}"])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [str(row.get("label", "")), str(row.get("eval_qid_count", ""))]
        for k in recall_ks:
            values.append(f"{float(row.get(f'page@{k}', 0.0)):.4f}")
            values.append(f"{float(row.get(f'doc@{k}', 0.0)):.4f}")
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.active_feature_names = resolve_feature_names(args.feature_set)
    if bool(args.query_adaptive_alpha) and not bool(args.auto_tune_blend_alpha):
        raise ValueError("--query-adaptive-alpha requires --auto-tune-blend-alpha.")
    if bool(args.learned_query_alpha) and not bool(args.auto_tune_blend_alpha):
        raise ValueError("--learned-query-alpha requires --auto-tune-blend-alpha.")
    if bool(args.learned_alpha_action) and not bool(args.auto_tune_blend_alpha):
        raise ValueError("--learned-alpha-action requires --auto-tune-blend-alpha.")
    if bool(args.learned_alpha_utility_gate) and not bool(args.auto_tune_blend_alpha):
        raise ValueError("--learned-alpha-utility-gate requires --auto-tune-blend-alpha.")
    if bool(args.base_aware_alpha_utility_gate) and not bool(args.auto_tune_blend_alpha):
        raise ValueError("--base-aware-alpha-utility-gate requires --auto-tune-blend-alpha.")
    adaptive_mode_count = sum(
        bool(value)
        for value in [
            args.query_adaptive_alpha,
            args.learned_query_alpha,
            args.learned_alpha_action,
            args.learned_alpha_utility_gate,
            args.base_aware_alpha_utility_gate,
        ]
    )
    if adaptive_mode_count > 1:
        raise ValueError(
            "Use only one adaptive alpha mode: --query-adaptive-alpha, "
            "--learned-query-alpha, --learned-alpha-action, --learned-alpha-utility-gate, "
            "or --base-aware-alpha-utility-gate."
        )
    if int(args.query_alpha_bins) <= 0:
        raise ValueError(f"--query-alpha-bins must be positive, got {args.query_alpha_bins}")
    if int(args.query_alpha_feature_top_k) <= 0:
        raise ValueError(f"--query-alpha-feature-top-k must be positive, got {args.query_alpha_feature_top_k}")
    if int(args.query_alpha_action_epochs) <= 0:
        raise ValueError(f"--query-alpha-action-epochs must be positive, got {args.query_alpha_action_epochs}")
    if float(args.alpha_utility_ridge) < 0.0:
        raise ValueError(f"--alpha-utility-ridge must be nonnegative, got {args.alpha_utility_ridge}")
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    train_gold = load_gold(Path(args.train_gold))
    eval_gold = load_gold(Path(args.eval_gold))
    train_base = load_prediction(Path(args.train_base_pred))
    eval_base = load_prediction(Path(args.eval_base_pred))
    train_page_features = load_page_features(Path(args.train_page_text_jsonl))
    eval_page_features = load_page_features(Path(args.eval_page_text_jsonl))

    train_source_maps: dict[str, dict[str, dict[str, float]]] = {}
    eval_source_maps: dict[str, dict[str, dict[str, float]]] = {}
    for spec in args.train_source:
        label, path = parse_labeled_path(spec)
        train_source_maps[label] = source_maps(load_prediction(path), int(args.candidate_top_k))
    for spec in args.eval_source:
        label, path = parse_labeled_path(spec)
        eval_source_maps[label] = source_maps(load_prediction(path), int(args.candidate_top_k))

    fit_gold = train_gold
    tune_gold: dict[str, dict[str, Any]] = {}
    if bool(args.auto_tune_blend_alpha):
        fit_gold, tune_gold = split_gold_for_tuning(
            train_gold,
            tune_fraction=float(args.tune_fraction),
            seed=int(args.seed),
        )

    X, y, row_weights, train_meta = build_matrix(
        gold=fit_gold,
        base_pred=train_base,
        page_features=train_page_features,
        source_maps_by_label=train_source_maps,
        args=args,
    )
    X_train, mean, std = standardize_train(X)
    weights, bias, history = train_logistic(X_train, y, args, row_weights=row_weights)

    tuning_summary: dict[str, Any] | None = None
    if bool(args.auto_tune_blend_alpha):
        fit_train_meta = dict(train_meta)
        tuning_summary = tune_blend_alpha(
            tune_gold=tune_gold,
            base_pred=train_base,
            page_features=train_page_features,
            source_maps_by_label=train_source_maps,
            mean=mean,
            std=std,
            weights=weights,
            bias=bias,
            args=args,
        )
        args.blend_alpha = float(tuning_summary["selected_blend_alpha"])
        if (
            bool(args.query_adaptive_alpha)
            or bool(args.learned_query_alpha)
            or bool(args.learned_alpha_action)
            or bool(args.learned_alpha_utility_gate)
            or bool(args.base_aware_alpha_utility_gate)
        ):
            args.adaptive_alpha_config = tuning_summary.get("adaptive_alpha_config")
        if not bool(args.skip_retrain_after_tuning):
            X, y, row_weights, train_meta = build_matrix(
                gold=train_gold,
                base_pred=train_base,
                page_features=train_page_features,
                source_maps_by_label=train_source_maps,
                args=args,
            )
            X_train, mean, std = standardize_train(X)
            weights, bias, history = train_logistic(X_train, y, args, row_weights=row_weights)
        train_meta = {
            **train_meta,
            "auto_tune_blend_alpha": True,
            "retrained_after_tuning": not bool(args.skip_retrain_after_tuning),
            "fit_qid_count": int(len(fit_gold)),
            "tune_qid_count": int(len(tune_gold)),
            "selected_blend_alpha": float(args.blend_alpha),
            "query_adaptive_alpha": bool(args.query_adaptive_alpha),
            "learned_query_alpha": bool(args.learned_query_alpha),
            "learned_alpha_action": bool(args.learned_alpha_action),
            "learned_alpha_utility_gate": bool(args.learned_alpha_utility_gate),
            "base_aware_alpha_utility_gate": bool(args.base_aware_alpha_utility_gate),
            "adaptive_alpha_config": tuning_summary.get("adaptive_alpha_config") if tuning_summary else None,
            "fit_train_metadata": fit_train_meta,
            "tuning_summary": tuning_summary,
        }

    eval_base_for_apply = eval_base
    if bool(args.restrict_eval_to_gold_qids):
        eval_base_for_apply = {
            qid: eval_base[qid]
            for qid in eval_gold
            if qid in eval_base
        }

    output_pred, prior_rows = apply_reranker(
        gold=eval_gold,
        base_pred=eval_base_for_apply,
        page_features=eval_page_features,
        source_maps_by_label=eval_source_maps,
        mean=mean,
        std=std,
        weights=weights,
        bias=bias,
        args=args,
    )

    output_prediction_json = Path(args.output_prediction_json)
    output_prediction_json.parent.mkdir(parents=True, exist_ok=True)
    output_prediction_json.write_text(json.dumps(output_pred) + "\n", encoding="utf-8")

    model = {
        "feature_names": list(args.active_feature_names),
        "all_feature_names": FEATURE_NAMES,
        "feature_set": args.feature_set,
        "weights": [float(value) for value in weights.tolist()],
        "bias": float(bias),
        "mean": [float(value) for value in mean.tolist()],
        "std": [float(value) for value in std.tolist()],
        "history": history,
        "train_metadata": train_meta,
        "adaptive_alpha_config": train_meta.get("adaptive_alpha_config") if isinstance(train_meta, dict) else None,
        "args": {
            "candidate_top_k": int(args.candidate_top_k),
            "inference_mode": args.inference_mode,
            "blend_alpha": float(args.blend_alpha),
            "auto_tune_blend_alpha": bool(args.auto_tune_blend_alpha),
            "query_adaptive_alpha": bool(args.query_adaptive_alpha),
            "learned_query_alpha": bool(args.learned_query_alpha),
            "learned_alpha_action": bool(args.learned_alpha_action),
            "learned_alpha_utility_gate": bool(args.learned_alpha_utility_gate),
            "base_aware_alpha_utility_gate": bool(args.base_aware_alpha_utility_gate),
            "query_alpha_bins": int(args.query_alpha_bins),
            "query_alpha_feature_top_k": int(args.query_alpha_feature_top_k),
            "query_alpha_ridge": float(args.query_alpha_ridge),
            "query_alpha_continuous": bool(args.query_alpha_continuous),
            "query_alpha_action_epochs": int(args.query_alpha_action_epochs),
            "query_alpha_action_learning_rate": float(args.query_alpha_action_learning_rate),
            "query_alpha_action_weight_decay": float(args.query_alpha_action_weight_decay),
            "alpha_utility_ridge": float(args.alpha_utility_ridge),
            "alpha_utility_threshold_grid": str(args.alpha_utility_threshold_grid),
            "alpha_utility_risk_penalty_grid": str(args.alpha_utility_risk_penalty_grid),
            "feature_set": args.feature_set,
            "respect_pseudo_supervision_tiers": bool(args.respect_pseudo_supervision_tiers),
            "pseudo_supervision_weighting": bool(args.pseudo_supervision_weighting),
            "hybrid_positive_weight": float(args.hybrid_positive_weight),
            "visual_proxy_positive_weight": float(args.visual_proxy_positive_weight),
            "partial_positive_weight": float(args.partial_positive_weight),
            "include_safe_support_doc_negatives": bool(args.include_safe_support_doc_negatives),
            "safe_support_doc_negative_weight": float(args.safe_support_doc_negative_weight),
            "max_safe_support_doc_negatives_per_qid": int(args.max_safe_support_doc_negatives_per_qid),
        },
    }
    output_model_json = Path(args.output_model_json)
    output_model_json.parent.mkdir(parents=True, exist_ok=True)
    output_model_json.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")

    metrics = [
        evaluate_run(label="base", pred=eval_base, gold=eval_gold, recall_ks=list(args.recall_k)),
        evaluate_run(
            label="content_aware_pseudo_page_reranker",
            pred=output_pred,
            gold=eval_gold,
            recall_ks=list(args.recall_k),
        ),
    ]
    summary = {
        "train_gold": args.train_gold,
        "eval_gold": args.eval_gold,
        "train_base_pred": args.train_base_pred,
        "eval_base_pred": args.eval_base_pred,
        "train_page_text_jsonl": args.train_page_text_jsonl,
        "eval_page_text_jsonl": args.eval_page_text_jsonl,
        "feature_set": args.feature_set,
        "restrict_eval_to_gold_qids": bool(args.restrict_eval_to_gold_qids),
        "eval_base_apply_qid_count": int(len(eval_base_for_apply)),
        "feature_names": list(args.active_feature_names),
        "all_feature_names": FEATURE_NAMES,
        "train_metadata": train_meta,
        "tuning_summary": tuning_summary,
        "metrics": metrics,
        "movement_vs_base": movement_vs_base(base_pred=eval_base, candidate_pred=output_pred, gold=eval_gold),
    }
    output_summary_json = Path(args.output_summary_json)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.output_table_md:
        write_table(Path(args.output_table_md), metrics, list(args.recall_k))
    if args.output_eval_prior_jsonl:
        prior_path = Path(args.output_eval_prior_jsonl)
        prior_path.parent.mkdir(parents=True, exist_ok=True)
        with prior_path.open("w", encoding="utf-8") as handle:
            for row in prior_rows:
                handle.write(json.dumps(row) + "\n")

    print(f"saved_model={output_model_json}")
    print(f"saved_prediction={output_prediction_json}")
    print(f"saved_summary={output_summary_json}")
    if args.output_table_md:
        print(f"saved_table={args.output_table_md}")
    if args.output_eval_prior_jsonl:
        print(f"saved_eval_prior={args.output_eval_prior_jsonl}")
    print(f"train_metadata={train_meta}")
    print(f"movement_vs_base={summary['movement_vs_base']['counts']}")
    for row in metrics:
        print(row)


if __name__ == "__main__":
    main()
