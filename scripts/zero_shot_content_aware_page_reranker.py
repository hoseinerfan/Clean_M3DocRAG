#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a conservative zero-shot content-aware page reranker. It uses no "
            "training labels: scores are fixed lexical/question-page evidence features "
            "blended with the original retrieval rank."
        )
    )
    parser.add_argument("--base-pred", required=True)
    parser.add_argument("--page-text-jsonl", required=True)
    parser.add_argument("--gold", default="", help="Optional gold JSONL for summary metrics.")
    parser.add_argument("--source", action="append", default=[], help="Optional LABEL=prediction.json source.")
    parser.add_argument("--candidate-top-k", type=int, default=1000)
    parser.add_argument("--inference-mode", choices=["blend_rerank", "safe_insert"], default="blend_rerank")
    parser.add_argument("--blend-alpha", type=float, default=0.20)
    parser.add_argument("--protect-top-k", type=int, default=1)
    parser.add_argument("--promotion-rank-min", type=int, default=5)
    parser.add_argument("--promotion-rank-max", type=int, default=100)
    parser.add_argument("--max-promotions-per-qid", type=int, default=2)
    parser.add_argument("--min-content-score", type=float, default=0.35)
    parser.add_argument("--source-bonus", type=float, default=0.08)
    parser.add_argument("--first-page-penalty", type=float, default=0.00)
    parser.add_argument("--recall-k", type=int, nargs="+", default=DEFAULT_RECALL_KS)
    parser.add_argument("--output-prediction-json", required=True)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument("--output-table-md", default="")
    parser.add_argument("--output-prior-jsonl", default="")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_gold(path: Path) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
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


def parse_labeled_path(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected LABEL=path, got {spec!r}")
    label, raw_path = spec.split("=", 1)
    return re.sub(r"[^A-Za-z0-9_]+", "_", label.strip()).strip("_"), Path(raw_path.strip())


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


def ngrams(tokens: list[str], n: int) -> list[str]:
    if len(tokens) < n:
        return []
    return [" ".join(tokens[idx : idx + n]) for idx in range(len(tokens) - n + 1)]


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


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def content_features(profile: dict[str, Any], page: dict[str, Any] | None) -> dict[str, float]:
    if page is None:
        return {key: 0.0 for key in (
            "question_token_recall",
            "anchor_token_recall",
            "number_token_recall",
            "phrase_match_fraction",
            "bigram_match_fraction",
            "trigram_match_fraction",
            "longest_question_ngram_match",
            "exact_question_substring",
        )}
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
    non_stop = [tok for tok in tokenize(profile["question"]) if tok not in STOPWORDS]
    for n in range(5, 1, -1):
        if any(f" {gram} " in f" {norm_text} " for gram in ngrams(non_stop, n)):
            longest_ngram = n
            break
    norm_question = str(profile["norm_question"])
    exact_question = bool(norm_question and len(norm_question) >= 16 and norm_question in norm_text)
    return {
        "question_token_recall": safe_div(len(overlap), len(q_tokens)),
        "anchor_token_recall": safe_div(len(anchor_overlap), len(anchor_tokens)),
        "number_token_recall": safe_div(len(number_overlap), len(number_tokens)),
        "phrase_match_fraction": safe_div(len(phrase_matches), len(profile["phrases"])),
        "bigram_match_fraction": safe_div(len(bigram_matches), len(profile["bigrams"])),
        "trigram_match_fraction": safe_div(len(trigram_matches), len(profile["trigrams"])),
        "longest_question_ngram_match": float(longest_ngram) / 5.0,
        "exact_question_substring": 1.0 if exact_question else 0.0,
    }


def minmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if math.isclose(lo, hi):
        return {key: 1.0 for key in values}
    return {key: (value - lo) / (hi - lo) for key, value in values.items()}


def source_bonus(qid: str, uid: str, source_maps_by_label: dict[str, dict[str, dict[str, float]]]) -> float:
    hits = []
    for source_map in source_maps_by_label.values():
        hit = source_map.get(qid, {}).get(uid)
        if hit:
            hits.append(hit)
    if not hits:
        return 0.0
    rank_part = max(1.0 / float(hit["rank"]) for hit in hits)
    score_part = max(float(hit.get("norm_score", 0.0)) for hit in hits)
    return min(1.0, 0.5 * float(len(hits)) + 0.25 * rank_part + 0.25 * score_part)


def score_content(
    qid: str,
    record: dict[str, Any],
    profile: dict[str, Any],
    page_features: dict[str, dict[str, Any]],
    source_maps_by_label: dict[str, dict[str, dict[str, float]]],
    args: argparse.Namespace,
) -> float:
    feats = content_features(profile, page_features.get(record["uid"]))
    score = (
        0.25 * feats["anchor_token_recall"]
        + 0.18 * feats["question_token_recall"]
        + 0.15 * feats["phrase_match_fraction"]
        + 0.12 * feats["bigram_match_fraction"]
        + 0.10 * feats["trigram_match_fraction"]
        + 0.10 * feats["number_token_recall"]
        + 0.07 * feats["longest_question_ngram_match"]
        + 0.08 * feats["exact_question_substring"]
    )
    score += float(args.source_bonus) * source_bonus(qid, record["uid"], source_maps_by_label)
    if int(record["page_idx"]) == 0:
        score -= float(args.first_page_penalty)
    return max(0.0, min(1.0, score))


def rerank_records(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if not records:
        return []
    if args.inference_mode == "safe_insert":
        protected = records[: max(0, int(args.protect_top_k))]
        rest = records[max(0, int(args.protect_top_k)) :]
        eligible = [
            row
            for row in rest
            if int(args.promotion_rank_min) <= int(row["base_rank"]) <= int(args.promotion_rank_max)
            and float(row.get("content_score", 0.0)) >= float(args.min_content_score)
        ]
        eligible.sort(key=lambda row: (-float(row.get("content_score", 0.0)), int(row["base_rank"]), row["uid"]))
        promotions = eligible[: int(args.max_promotions_per_qid)]
        promoted = {row["uid"] for row in promotions}
        return protected + promotions + [row for row in rest if row["uid"] not in promoted]

    base_rank_scores = {row["uid"]: 1.0 / math.log2(float(row["base_rank"]) + 1.0) for row in records}
    base_rank_scores = minmax(base_rank_scores)
    content_scores = minmax({row["uid"]: float(row.get("content_score", 0.0)) for row in records})
    alpha = float(args.blend_alpha)
    for row in records:
        row["rerank_score"] = (1.0 - alpha) * base_rank_scores[row["uid"]] + alpha * content_scores[row["uid"]]
    return sorted(records, key=lambda row: (-float(row["rerank_score"]), int(row["base_rank"]), row["uid"]))


def apply_reranker(
    *,
    base_pred: dict[str, dict[str, Any]],
    gold: dict[str, dict[str, Any]],
    page_features: dict[str, dict[str, Any]],
    source_maps_by_label: dict[str, dict[str, dict[str, float]]],
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    prior_rows: list[dict[str, Any]] = []
    total_records = 0
    scored_records = 0
    for qid, base_row in base_pred.items():
        records = ranked_page_records(base_row, int(args.candidate_top_k))
        if not records:
            output[qid] = dict(base_row)
            continue
        gold_row = gold.get(qid, {"qid": qid, "question": base_row.get("question", "")})
        if not str(gold_row.get("question", "")).strip() and str(base_row.get("question", "")).strip():
            gold_row = dict(gold_row)
            gold_row["question"] = base_row.get("question", "")
        profile = question_profile(gold_row)
        for record in records:
            total_records += 1
            if record["uid"] in page_features:
                scored_records += 1
            record["content_score"] = score_content(
                qid, record, profile, page_features, source_maps_by_label, args
            )
        reranked = rerank_records(records, args)
        raw_by_uid = {row["uid"]: row["raw"] for row in records}
        output_rows = [raw_by_uid[row["uid"]] for row in reranked if row["uid"] in raw_by_uid]
        output_rows.extend(
            raw
            for raw in prediction_rows(base_row)
            if parse_prediction_row(raw) is None
            or page_uid(parse_prediction_row(raw)[0], parse_prediction_row(raw)[1]) not in raw_by_uid  # type: ignore[index]
        )
        out_row = dict(base_row)
        if "page_retrieval_results" in out_row:
            out_row["page_retrieval_results"] = output_rows
        elif "retrieval_results" in out_row:
            out_row["retrieval_results"] = output_rows
        else:
            out_row["page_retrieval_results"] = output_rows
        out_row["zero_shot_content_aware_metadata"] = {
            "inference_mode": args.inference_mode,
            "blend_alpha": float(args.blend_alpha),
            "candidate_top_k": int(args.candidate_top_k),
        }
        output[qid] = out_row
        for row in reranked[:50]:
            prior_rows.append(
                {
                    "qid": qid,
                    "page_uid": row["uid"],
                    "doc_id": row["doc_id"],
                    "page_idx": int(row["page_idx"]),
                    "base_rank": int(row["base_rank"]),
                    "content_score": float(row.get("content_score", 0.0)),
                    "rerank_score": float(row.get("rerank_score", row.get("content_score", 0.0))),
                }
            )
    return output, prior_rows, {
        "total_candidate_records": total_records,
        "candidate_records_with_page_text": scored_records,
        "candidate_page_text_coverage": safe_div(scored_records, total_records),
    }


def gold_page_uids(row: dict[str, Any]) -> set[str]:
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    values = metadata.get("gold_page_uids") or row.get("gold_page_uids") or []
    uids = {str(value).strip() for value in values if str(value).strip()}
    for item in row.get("supporting_context", []):
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("doc_id", "")).strip()
        page_idx = item.get("page_idx", item.get("page_id"))
        if doc_id and page_idx is not None:
            uids.add(page_uid(doc_id, int(page_idx)))
    return uids


def gold_doc_ids(row: dict[str, Any]) -> set[str]:
    docs = set()
    for uid in gold_page_uids(row):
        parsed = parse_page_uid(uid)
        if parsed:
            docs.add(parsed[0])
    for item in row.get("supporting_context", []):
        if isinstance(item, dict) and item.get("doc_id"):
            docs.add(str(item["doc_id"]).strip())
    return docs


def ranked_pages(pred_row: dict[str, Any] | None) -> list[str]:
    return [record["uid"] for record in ranked_page_records(pred_row, 1000000)]


def ranked_docs_from_pages(pages: list[str]) -> list[str]:
    docs: list[str] = []
    seen: set[str] = set()
    for uid in pages:
        parsed = parse_page_uid(uid)
        doc_id = parsed[0] if parsed else uid
        if doc_id not in seen:
            seen.add(doc_id)
            docs.append(doc_id)
    return docs


def first_rank(items: list[str], gold: set[str]) -> int | None:
    for idx, item in enumerate(items, start=1):
        if item in gold:
            return idx
    return None


def hit_at(items: list[str], gold: set[str], k: int) -> bool:
    return bool(gold and any(item in gold for item in items[: int(k)]))


def evaluate_run(
    label: str,
    pred: dict[str, dict[str, Any]],
    gold: dict[str, dict[str, Any]],
    recall_ks: list[int],
) -> dict[str, Any]:
    page_hits = {k: 0 for k in recall_ks}
    doc_hits = {k: 0 for k in recall_ks}
    page_mrr = 0.0
    doc_mrr = 0.0
    n_doc = 0
    n_page = 0
    for qid, gold_row in gold.items():
        pred_row = pred.get(qid)
        if pred_row is None:
            continue
        pages = ranked_pages(pred_row)
        docs = ranked_docs_from_pages(pages)
        gd = gold_doc_ids(gold_row)
        gp = gold_page_uids(gold_row)
        if gd:
            n_doc += 1
            doc_rank = first_rank(docs, gd)
            doc_mrr += 0.0 if doc_rank is None else 1.0 / float(doc_rank)
            for k in recall_ks:
                if hit_at(docs, gd, k):
                    doc_hits[k] += 1
        if gp:
            n_page += 1
            page_rank = first_rank(pages, gp)
            page_mrr += 0.0 if page_rank is None else 1.0 / float(page_rank)
            for k in recall_ks:
                if hit_at(pages, gp, k):
                    page_hits[k] += 1
    row: dict[str, Any] = {
        "label": label,
        "qid_count": len(gold),
        "doc_eval_count": n_doc,
        "page_eval_count": n_page,
        "doc_mrr": doc_mrr / float(n_doc) if n_doc else None,
        "page_mrr": page_mrr / float(n_page) if n_page else None,
    }
    for k in recall_ks:
        row[f"doc@{k}"] = doc_hits[k] / float(n_doc) if n_doc else None
        row[f"page@{k}"] = page_hits[k] / float(n_page) if n_page else None
    return row


def movement_vs_base(
    base_pred: dict[str, dict[str, Any]],
    candidate_pred: dict[str, dict[str, Any]],
    gold: dict[str, dict[str, Any]],
    hit_k: int = 4,
) -> dict[str, Any]:
    counts = {"unchanged": 0, "improved_rank": 0, "worsened_rank": 0, "recovered": 0, "lost": 0}
    for qid, gold_row in gold.items():
        gp = gold_page_uids(gold_row)
        if not gp:
            continue
        base_rank = first_rank(ranked_pages(base_pred.get(qid)), gp)
        cand_rank = first_rank(ranked_pages(candidate_pred.get(qid)), gp)
        base_hit = base_rank is not None and base_rank <= hit_k
        cand_hit = cand_rank is not None and cand_rank <= hit_k
        if base_hit and not cand_hit:
            counts["lost"] += 1
        elif not base_hit and cand_hit:
            counts["recovered"] += 1
        if base_rank == cand_rank:
            counts["unchanged"] += 1
        elif cand_rank is None or (base_rank is not None and cand_rank > base_rank):
            counts["worsened_rank"] += 1
        else:
            counts["improved_rank"] += 1
    counts["net"] = counts["recovered"] - counts["lost"]
    return counts


def write_table(path: Path, metrics: list[dict[str, Any]], recall_ks: list[int]) -> None:
    columns = ["label", "qid_count", "doc_eval_count", "page_eval_count", "doc_mrr", "page_mrr"]
    for k in recall_ks:
        columns.extend([f"doc@{k}", f"page@{k}"])
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in metrics:
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            elif value is None:
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_pred = load_prediction(Path(args.base_pred))
    gold = load_gold(Path(args.gold)) if args.gold else {}
    page_features = load_page_features(Path(args.page_text_jsonl))
    source_maps_by_label = {
        label: source_maps(load_prediction(path), int(args.candidate_top_k))
        for label, path in map(parse_labeled_path, args.source)
    }
    output_pred, prior_rows, metadata = apply_reranker(
        base_pred=base_pred,
        gold=gold,
        page_features=page_features,
        source_maps_by_label=source_maps_by_label,
        args=args,
    )
    out_pred = Path(args.output_prediction_json)
    out_pred.parent.mkdir(parents=True, exist_ok=True)
    out_pred.write_text(json.dumps(output_pred) + "\n", encoding="utf-8")

    metrics = []
    if gold:
        metrics = [
            evaluate_run("base", base_pred, gold, list(args.recall_k)),
            evaluate_run("zero_shot_content_aware", output_pred, gold, list(args.recall_k)),
        ]
    summary = {
        "base_pred": args.base_pred,
        "page_text_jsonl": args.page_text_jsonl,
        "source_count": len(source_maps_by_label),
        "candidate_top_k": int(args.candidate_top_k),
        "inference_mode": args.inference_mode,
        "blend_alpha": float(args.blend_alpha),
        "protect_top_k": int(args.protect_top_k),
        "promotion_rank_min": int(args.promotion_rank_min),
        "promotion_rank_max": int(args.promotion_rank_max),
        "max_promotions_per_qid": int(args.max_promotions_per_qid),
        "min_content_score": float(args.min_content_score),
        **metadata,
        "metrics": metrics,
        "movement_vs_base": movement_vs_base(base_pred, output_pred, gold) if gold else {},
    }
    out_summary = Path(args.output_summary_json)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.output_table_md and metrics:
        write_table(Path(args.output_table_md), metrics, list(args.recall_k))
    if args.output_prior_jsonl:
        out_prior = Path(args.output_prior_jsonl)
        out_prior.parent.mkdir(parents=True, exist_ok=True)
        with out_prior.open("w", encoding="utf-8") as handle:
            for row in prior_rows:
                handle.write(json.dumps(row) + "\n")
    print(f"saved_prediction={out_pred}")
    print(f"saved_summary={out_summary}")
    if args.output_table_md and metrics:
        print(f"saved_table={args.output_table_md}")
    if args.output_prior_jsonl:
        print(f"saved_prior={args.output_prior_jsonl}")
    print(f"candidate_page_text_coverage={metadata['candidate_page_text_coverage']:.6f}")
    if metrics:
        for row in metrics:
            print(row)
        print(f"movement_vs_base={summary['movement_vs_base']}")


if __name__ == "__main__":
    main()
