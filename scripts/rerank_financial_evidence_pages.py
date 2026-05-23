#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_TEXT_FIELDS = ["ocr_text", "vlm_text", "markdown", "text", "page_text", "content"]


def load_graph_helpers() -> Any:
    module_path = Path(__file__).with_name("graph_rerank_page_retrieval_predictions.py")
    spec = importlib.util.spec_from_file_location("_graph_rerank_helpers", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load graph helper module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_graph_rerank_helpers"] = module
    spec.loader.exec_module(module)
    return module


GRAPH = load_graph_helpers()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-rerank financial-style queries by verifying metric/year/entity evidence "
            "inside candidate page text lines and local windows."
        )
    )
    parser.add_argument("--prediction-json", required=True)
    parser.add_argument("--gold", required=True, help="Converted MMQA_dev.jsonl. Used for questions and optional metrics.")
    parser.add_argument("--doc-pages-jsonl", required=True)
    parser.add_argument("--text-field", nargs="*", default=DEFAULT_TEXT_FIELDS)
    parser.add_argument("--top-pages-to-rerank", type=int, default=1000)
    parser.add_argument("--rrf-k", type=float, default=10.0)
    parser.add_argument(
        "--evidence-weight",
        type=float,
        default=0.020,
        help="Soft rerank weight added to rank-based base score.",
    )
    parser.add_argument("--line-window", type=int, default=3)
    parser.add_argument(
        "--scoring-mode",
        choices=["soft", "strict"],
        default="soft",
        help=(
            "soft keeps the original page/window/table blend. strict requires metric "
            "and numeric evidence, plus available year/entity evidence, in a local "
            "line window before a page receives a bonus."
        ),
    )
    parser.add_argument(
        "--doc-prior-mode",
        choices=["none", "rank"],
        default="none",
        help=(
            "Optionally multiply evidence by a document prior estimated from the "
            "input ranking. rank uses first document occurrence in the candidate list."
        ),
    )
    parser.add_argument(
        "--doc-prior-rrf-k",
        type=float,
        default=10.0,
        help="RRF-style damping constant for --doc-prior-mode rank.",
    )
    parser.add_argument(
        "--min-evidence-score",
        type=float,
        default=0.0,
        help="Treat pages below this score as having no evidence bonus.",
    )
    parser.add_argument("--hit-k", type=int, default=4)
    parser.add_argument("--output-prediction-json", required=True)
    parser.add_argument("--output-summary-json", required=True)
    return parser.parse_args()


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
    if isinstance(payload, dict) and "predictions" in payload and isinstance(
        payload["predictions"], (dict, list)
    ):
        payload = payload["predictions"]
    if isinstance(payload, dict):
        iterable = payload.items()
    elif isinstance(payload, list):
        iterable = enumerate(payload)
    else:
        raise TypeError(f"Unsupported prediction payload: {path}")
    rows: dict[str, dict[str, Any]] = {}
    for raw_key, row in iterable:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("qid", "")).strip() or str(raw_key).strip()
        if qid:
            rows[qid] = row
    return rows


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.replace("\x0c", "\n").replace("\u0000", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(part for part in (normalize_text(item) for item in value) if part)
    if isinstance(value, dict):
        return "\n".join(part for part in (normalize_text(item) for item in value.values()) if part)
    return str(value).strip()


def page_uid(doc_id: str, page_idx: int) -> str:
    return f"{doc_id}_page{int(page_idx)}"


def load_page_texts(path: Path, text_fields: list[str]) -> dict[str, str]:
    page_texts = {}
    for row in read_jsonl(path):
        doc_id = str(row.get("doc_id", "")).strip()
        raw_page_idx = row.get("page_idx", row.get("page_id", row.get("page_number")))
        if not doc_id or raw_page_idx is None:
            continue
        try:
            page_idx = int(raw_page_idx)
        except (TypeError, ValueError):
            continue
        parts = []
        seen = set()
        for field in text_fields:
            text = normalize_text(row.get(field))
            if text and text not in seen:
                parts.append(text)
                seen.add(text)
        if parts:
            page_texts[page_uid(doc_id, page_idx)] = "\n".join(parts)
    return page_texts


def parse_page_row(row: Any) -> tuple[str, int, float | None] | None:
    if not isinstance(row, (list, tuple)) or len(row) < 2:
        return None
    doc_id = str(row[0]).strip()
    if not doc_id:
        return None
    try:
        page_idx = int(row[1])
    except (TypeError, ValueError):
        return None
    score = None
    if len(row) >= 3:
        try:
            score = float(row[2])
        except (TypeError, ValueError):
            score = None
    return doc_id, page_idx, score


def phrase_tokens(value: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9&./+-]+", value.lower()) if token]


def phrase_match_score(phrase: str, text: str) -> float:
    normalized = GRAPH.normalize_query_anchor(phrase).lower()
    text_lower = text.lower()
    if not normalized or not text_lower:
        return 0.0
    if re.search(rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])", text_lower):
        return 1.0
    tokens = phrase_tokens(normalized)
    if len(tokens) <= 1:
        return 0.0
    hits = sum(
        1
        for token in tokens
        if re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", text_lower)
    )
    coverage = hits / len(tokens)
    return coverage if coverage >= 0.75 else 0.0


def best_slot_match(values: list[str], text: str) -> float:
    if not values:
        return 0.0
    return max(phrase_match_score(value, text) for value in values)


def numeric_density_score(text: str) -> float:
    hits = len(
        re.findall(
            r"(?:\$|€|£)?\(?\d[\d,]*(?:\.\d+)?\)?%?|\b20\d{2}\b|\bFY\s*'?[\d]{2,4}\b",
            text,
            flags=re.IGNORECASE,
        )
    )
    return GRAPH.clamp(hits / 12.0, 0.0, 1.0)


def financial_value_density_score(text: str) -> float:
    hits = 0
    for match in re.finditer(
        r"(?:\$|€|£)?\(?\d[\d,]*(?:\.\d+)?\)?%?",
        text,
        flags=re.IGNORECASE,
    ):
        token = match.group(0).strip()
        bare = token.strip("()$€£%").replace(",", "")
        if re.fullmatch(r"20\d{2}", bare):
            continue
        if re.fullmatch(r"\d{1,2}", bare) and re.search(r"\bFY\s*'?\s*" + re.escape(bare) + r"\b", text, re.IGNORECASE):
            continue
        hits += 1
    return GRAPH.clamp(hits / 8.0, 0.0, 1.0)


def split_lines(text: str) -> list[str]:
    lines = []
    for raw_line in re.split(r"[\n\r]+", text):
        line = re.sub(r"\s+", " ", raw_line).strip()
        if line:
            lines.append(line)
    if not lines and text.strip():
        lines = [re.sub(r"\s+", " ", text).strip()]
    return lines


def evidence_window_features(slots: dict[str, list[str]], window_text: str) -> dict[str, Any]:
    available_slots = [slot for slot in ("metric", "year", "entity") if slots.get(slot)]
    if not available_slots:
        return {
            "score": 0.0,
            "slot_scores": {},
            "slot_coverage": 0.0,
            "slot_strength": 0.0,
            "numeric_score": 0.0,
            "value_numeric_score": 0.0,
            "matched_slot_count": 0,
        }
    slot_scores = {
        slot: best_slot_match(slots.get(slot, []), window_text)
        for slot in available_slots
    }
    matched_slot_count = sum(1 for value in slot_scores.values() if value > 0)
    slot_coverage = matched_slot_count / len(available_slots)
    slot_strength = sum(slot_scores.values()) / len(available_slots)
    numeric_score = numeric_density_score(window_text)
    value_numeric_score = financial_value_density_score(window_text)
    combo_bonus = 0.0
    if slot_scores.get("metric", 0.0) > 0 and slot_scores.get("year", 0.0) > 0:
        combo_bonus += 0.20
    if slot_scores.get("metric", 0.0) > 0 and numeric_score > 0:
        combo_bonus += 0.15
    if slot_scores.get("year", 0.0) > 0 and numeric_score > 0:
        combo_bonus += 0.10
    score = GRAPH.clamp(
        0.45 * slot_coverage + 0.25 * slot_strength + 0.20 * numeric_score + combo_bonus,
        0.0,
        1.0,
    )
    return {
        "score": score,
        "slot_scores": slot_scores,
        "slot_coverage": slot_coverage,
        "slot_strength": slot_strength,
        "numeric_score": numeric_score,
        "value_numeric_score": value_numeric_score,
        "matched_slot_count": matched_slot_count,
    }


def evidence_window_score(slots: dict[str, list[str]], window_text: str) -> float:
    return float(evidence_window_features(slots, window_text)["score"])


def strict_financial_window_score(slots: dict[str, list[str]], window_text: str) -> float:
    features = evidence_window_features(slots, window_text)
    slot_scores = features["slot_scores"]
    if slot_scores.get("metric", 0.0) <= 0:
        return 0.0
    if features["value_numeric_score"] <= 0:
        return 0.0
    has_context_slot = bool(slots.get("year") or slots.get("entity"))
    has_context_match = slot_scores.get("year", 0.0) > 0 or slot_scores.get("entity", 0.0) > 0
    if has_context_slot and not has_context_match:
        return 0.0
    return float(features["score"])


def financial_evidence_score(
    question: str,
    page_text: str,
    line_window: int,
    scoring_mode: str = "soft",
) -> tuple[float, dict[str, Any]]:
    anchors = GRAPH.extract_query_anchors(question, max_anchors=16, min_entity_len=3)
    slots = GRAPH.financial_anchor_slots(question, anchors)
    active_slots = {slot: values for slot, values in slots.items() if values}
    if not active_slots or "metric" not in active_slots:
        return 0.0, {
            "active": False,
            "reason": "no_metric_slot",
            "slots": slots,
            "anchor_count": len(anchors),
        }
    text = page_text or ""
    page_slot_score = evidence_window_score(slots, text)
    table_score = GRAPH.financial_table_likeness(text.lower())
    lines = split_lines(text)
    best_line_score = max((evidence_window_score(slots, line) for line in lines), default=0.0)
    best_strict_line_score = max(
        (strict_financial_window_score(slots, line) for line in lines),
        default=0.0,
    )
    window = max(1, int(line_window))
    best_window_score = 0.0
    best_strict_window_score = 0.0
    for start in range(len(lines)):
        chunk = "\n".join(lines[start : start + window])
        best_window_score = max(best_window_score, evidence_window_score(slots, chunk))
        best_strict_window_score = max(
            best_strict_window_score,
            strict_financial_window_score(slots, chunk),
        )
    if str(scoring_mode) == "strict":
        if best_strict_window_score <= 0:
            score = 0.0
        else:
            score = GRAPH.clamp(
                0.70 * best_strict_window_score
                + 0.20 * best_strict_line_score
                + 0.10 * table_score,
                0.0,
                1.0,
            )
    else:
        score = GRAPH.clamp(
            0.25 * page_slot_score
            + 0.35 * best_window_score
            + 0.25 * best_line_score
            + 0.15 * table_score,
            0.0,
            1.0,
        )
    return score, {
        "active": True,
        "reason": "scored",
        "slots": slots,
        "anchor_count": len(anchors),
        "scoring_mode": str(scoring_mode),
        "page_slot_score": page_slot_score,
        "best_line_score": best_line_score,
        "best_window_score": best_window_score,
        "best_strict_line_score": best_strict_line_score,
        "best_strict_window_score": best_strict_window_score,
        "table_score": table_score,
    }


def gold_page_uids(row: dict[str, Any]) -> set[str]:
    metadata = row.get("metadata", {})
    uids = {
        str(value).strip()
        for value in metadata.get("gold_page_uids", [])
        if str(value).strip()
    }
    for ctx in row.get("supporting_context", []):
        doc_id = str(ctx.get("doc_id", "")).strip()
        page_idx = ctx.get("page_idx", ctx.get("page_id"))
        if doc_id and page_idx is not None:
            uids.add(page_uid(doc_id, int(page_idx)))
    return uids


def gold_doc_ids(row: dict[str, Any]) -> set[str]:
    docs = {
        str(ctx.get("doc_id", "")).strip()
        for ctx in row.get("supporting_context", [])
        if str(ctx.get("doc_id", "")).strip()
    }
    for uid in gold_page_uids(row):
        if "_page" in uid:
            docs.add(uid.rsplit("_page", 1)[0])
    return docs


def ranked_pages(rows: list[Any]) -> list[str]:
    out = []
    for row in rows:
        parsed = parse_page_row(row)
        if parsed is not None:
            out.append(page_uid(parsed[0], parsed[1]))
    return out


def ranked_docs(rows: list[Any]) -> list[str]:
    docs = []
    seen = set()
    for row in rows:
        parsed = parse_page_row(row)
        if parsed is None:
            continue
        doc_id = parsed[0]
        if doc_id not in seen:
            seen.add(doc_id)
            docs.append(doc_id)
    return docs


def first_rank(items: list[str], gold: set[str]) -> int | None:
    for idx, item in enumerate(items, start=1):
        if item in gold:
            return idx
    return None


def movement_for_hit(base_rank: int | None, cand_rank: int | None, hit_k: int) -> str:
    base_hit = base_rank is not None and base_rank <= hit_k
    cand_hit = cand_rank is not None and cand_rank <= hit_k
    if not base_hit and cand_hit:
        return "recovered"
    if base_hit and not cand_hit:
        return "lost"
    if base_rank is None and cand_rank is None:
        return "missing_in_both"
    if base_rank is not None and cand_rank is not None and cand_rank < base_rank:
        return "improved_rank"
    if base_rank is not None and cand_rank is not None and cand_rank > base_rank:
        return "worsened_rank"
    return "unchanged"


def rerank_row(
    row: dict[str, Any],
    question: str,
    page_texts: dict[str, str],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    retrieval_rows = list(row.get("page_retrieval_results", []))
    limit = int(args.top_pages_to_rerank)
    if limit <= 0:
        limit = len(retrieval_rows)
    head = retrieval_rows[:limit]
    tail = retrieval_rows[limit:]
    doc_first_ranks: dict[str, int] = {}
    for rank, page_row in enumerate(head, start=1):
        parsed = parse_page_row(page_row)
        if parsed is None:
            continue
        if parsed[0] not in doc_first_ranks:
            doc_first_ranks[parsed[0]] = len(doc_first_ranks) + 1
    scored_rows = []
    evidence_scores = []
    active = False
    top_evidence = []
    for rank, page_row in enumerate(head, start=1):
        parsed = parse_page_row(page_row)
        if parsed is None:
            scored_rows.append((0.0, rank, page_row, 0.0, {}))
            continue
        uid = page_uid(parsed[0], parsed[1])
        evidence_score, evidence_meta = financial_evidence_score(
            question,
            page_texts.get(uid, ""),
            int(args.line_window),
            str(args.scoring_mode),
        )
        if evidence_score < float(args.min_evidence_score):
            evidence_score = 0.0
        doc_prior = 1.0
        if str(args.doc_prior_mode) == "rank":
            doc_rank = doc_first_ranks.get(parsed[0])
            if doc_rank is None:
                doc_prior = 0.0
            else:
                k = max(0.0, float(args.doc_prior_rrf_k))
                doc_prior = (k + 1.0) / (k + float(doc_rank))
        evidence_meta["doc_prior"] = doc_prior
        active = active or bool(evidence_meta.get("active", False))
        evidence_scores.append(evidence_score)
        base_score = 1.0 / (float(args.rrf_k) + float(rank))
        combined_score = base_score + float(args.evidence_weight) * evidence_score * doc_prior
        scored_rows.append((combined_score, rank, page_row, evidence_score, evidence_meta))
        if evidence_score > 0:
            top_evidence.append((evidence_score, uid, evidence_meta))

    reranked_head = []
    for combined_score, _rank, page_row, _evidence_score, _evidence_meta in sorted(
        scored_rows,
        key=lambda item: (-item[0], item[1]),
    ):
        new_row = list(page_row)
        if len(new_row) >= 3:
            new_row[2] = float(combined_score)
        else:
            new_row.append(float(combined_score))
        reranked_head.append(new_row)

    out_row = dict(row)
    out_row["page_retrieval_results"] = reranked_head + tail
    top_evidence_rows = [
        {
            "page_uid": uid,
            "score": score,
            "slots": meta.get("slots", {}),
            "best_line_score": meta.get("best_line_score"),
            "best_window_score": meta.get("best_window_score"),
            "best_strict_line_score": meta.get("best_strict_line_score"),
            "best_strict_window_score": meta.get("best_strict_window_score"),
            "table_score": meta.get("table_score"),
            "doc_prior": meta.get("doc_prior"),
        }
        for score, uid, meta in sorted(top_evidence, key=lambda item: (-item[0], item[1]))[:10]
    ]
    summary = {
        "active": active,
        "mean_evidence_score": statistics.fmean(evidence_scores) if evidence_scores else 0.0,
        "max_evidence_score": max(evidence_scores) if evidence_scores else 0.0,
        "positive_page_count": sum(1 for value in evidence_scores if value > 0),
        "top_evidence_pages": top_evidence_rows,
    }
    return out_row, summary


def main() -> None:
    args = parse_args()
    prediction = load_prediction(Path(args.prediction_json))
    gold_rows = {str(row["qid"]): row for row in read_jsonl(Path(args.gold))}
    page_texts = load_page_texts(Path(args.doc_pages_jsonl), list(args.text_field))

    output: dict[str, dict[str, Any]] = {}
    per_qid = []
    for qid, row in prediction.items():
        gold_row = gold_rows.get(qid, {})
        question = str(row.get("question") or gold_row.get("question") or "")
        reranked_row, evidence_summary = rerank_row(row, question, page_texts, args)
        output[qid] = reranked_row

        if gold_row:
            base_page_rank = first_rank(ranked_pages(row.get("page_retrieval_results", [])), gold_page_uids(gold_row))
            cand_page_rank = first_rank(ranked_pages(reranked_row.get("page_retrieval_results", [])), gold_page_uids(gold_row))
            base_doc_rank = first_rank(ranked_docs(row.get("page_retrieval_results", [])), gold_doc_ids(gold_row))
            cand_doc_rank = first_rank(ranked_docs(reranked_row.get("page_retrieval_results", [])), gold_doc_ids(gold_row))
        else:
            base_page_rank = cand_page_rank = base_doc_rank = cand_doc_rank = None
        per_qid.append(
            {
                "qid": qid,
                "question": question,
                "baseline_first_gold_page_rank": base_page_rank,
                "reranked_first_gold_page_rank": cand_page_rank,
                "baseline_first_gold_doc_rank": base_doc_rank,
                "reranked_first_gold_doc_rank": cand_doc_rank,
                "movement": movement_for_hit(base_page_rank, cand_page_rank, int(args.hit_k)),
                **evidence_summary,
            }
        )

    movement = Counter(row["movement"] for row in per_qid)
    active_rows = [row for row in per_qid if row["active"]]
    summary = {
        "qid_count": len(per_qid),
        "active_qid_count": len(active_rows),
        "movement_counts": dict(movement),
        "reranked_top4_page_count": sum(
            1
            for row in per_qid
            if row["reranked_first_gold_page_rank"] is not None
            and row["reranked_first_gold_page_rank"] <= int(args.hit_k)
        ),
        "reranked_top4_doc_count": sum(
            1
            for row in per_qid
            if row["reranked_first_gold_doc_rank"] is not None
            and row["reranked_first_gold_doc_rank"] <= int(args.hit_k)
        ),
        "mean_active_evidence_score": (
            statistics.fmean(row["mean_evidence_score"] for row in active_rows)
            if active_rows
            else 0.0
        ),
        "mean_active_positive_page_count": (
            statistics.fmean(row["positive_page_count"] for row in active_rows)
            if active_rows
            else 0.0
        ),
        "config": {
            "prediction_json": args.prediction_json,
            "doc_pages_jsonl": args.doc_pages_jsonl,
            "top_pages_to_rerank": int(args.top_pages_to_rerank),
            "rrf_k": float(args.rrf_k),
            "evidence_weight": float(args.evidence_weight),
            "line_window": int(args.line_window),
            "scoring_mode": str(args.scoring_mode),
            "doc_prior_mode": str(args.doc_prior_mode),
            "doc_prior_rrf_k": float(args.doc_prior_rrf_k),
            "min_evidence_score": float(args.min_evidence_score),
        },
        "per_qid": per_qid,
    }

    Path(args.output_prediction_json).write_text(json.dumps(output, indent=2), encoding="utf-8")
    Path(args.output_summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved_prediction: {args.output_prediction_json}")
    print(f"saved_summary: {args.output_summary_json}")
    print("qid_count:", summary["qid_count"])
    print("active_qid_count:", summary["active_qid_count"])
    print("reranked_top4_page_count:", summary["reranked_top4_page_count"])
    print("reranked_top4_doc_count:", summary["reranked_top4_doc_count"])
    print("movement_counts:", summary["movement_counts"])
    print("mean_active_evidence_score:", summary["mean_active_evidence_score"])
    print("mean_active_positive_page_count:", summary["mean_active_positive_page_count"])


if __name__ == "__main__":
    main()
