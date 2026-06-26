#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any


DEFAULT_EVIDENCE_WEIGHTS: dict[str, float] = {
    "answer_text": 5.0,
    "text_instance": 9.0,
    "text_instance_context": 2.0,
    "image_title": 7.0,
    "image_doc_title": 4.0,
    "table_title": 4.0,
    "table_answer_cell": 10.0,
    "table_row_cell": 2.0,
    "table_row_link_text": 1.5,
    "table_row_link_title": 1.5,
    "supporting_doc_title": 3.0,
    "answer_entity": 4.0,
    "question_entity": 1.2,
    "pseudo_question_slot": 1.0,
}

DIRECT_EVIDENCE_CAP_SOURCES = frozenset(
    {
        "text_instance",
        "table_answer_cell",
        "image_title",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Infer pseudo page labels for M3DocVQA/MMQA examples by matching MMQA "
            "answer/table/image evidence against exported PDF page text. These labels "
            "are not official gold pages; they are intended for audits and supervised "
            "page-promotion experiments."
        )
    )
    parser.add_argument("--gold", required=True, help="MMQA_train/dev.jsonl")
    parser.add_argument("--doc-pages-jsonl", required=True, help="Exported page text JSONL")
    parser.add_argument("--mmqa-texts-jsonl", default="", help="Optional MMQA_texts.jsonl")
    parser.add_argument("--mmqa-tables-jsonl", default="", help="Optional MMQA_tables.jsonl")
    parser.add_argument("--mmqa-images-jsonl", default="", help="Optional MMQA_images.jsonl")
    parser.add_argument("--id-url-mapping-jsonl", default="", help="Optional id_url_mapping.jsonl")
    parser.add_argument("--min-score", type=float, default=4.0)
    parser.add_argument("--top-pages-per-doc", type=int, default=2)
    parser.add_argument("--top-pages-per-qid", type=int, default=8)
    parser.add_argument(
        "--adaptive-page-caps",
        action="store_true",
        help=(
            "Interpret --top-pages-per-doc and --top-pages-per-qid as maximum caps, "
            "then set the effective per-document and per-question caps from MMQA "
            "direct evidence-unit counts. Direct evidence units are text_instance, "
            "table_answer_cell, and image_title."
        ),
    )
    parser.add_argument("--min-token-overlap", type=float, default=0.72)
    parser.add_argument(
        "--selection-policy",
        choices=["score", "evidence_coverage"],
        default="score",
        help=(
            "score keeps the original score-ranked selection. evidence_coverage greedily "
            "selects pages that cover new evidence signals within each support document."
        ),
    )
    parser.add_argument(
        "--coverage-min-match-weight",
        type=float,
        default=3.0,
        help=(
            "Minimum matched evidence weight counted by evidence_coverage selection. "
            "This prevents weak question/entity overlaps from forcing extra pages."
        ),
    )
    parser.add_argument(
        "--evidence-weight-overrides",
        default="",
        help=(
            "Optional comma-separated evidence weights, for example "
            "'question_entity=0,pseudo_question_slot=0,supporting_doc_title=2'. "
            "Unspecified evidence sources keep their default weights."
        ),
    )
    parser.add_argument(
        "--text-instance-context-window-chars",
        type=int,
        default=0,
        help=(
            "If positive, add weak contextual evidence phrases from MMQA_texts around "
            "text_instances[].start_byte. The default keeps the original strict labels "
            "unchanged."
        ),
    )
    parser.add_argument(
        "--text-instance-context-max-phrases",
        type=int,
        default=6,
        help="Maximum context phrases to add per text instance when start_byte context is enabled.",
    )
    parser.add_argument(
        "--text-instance-context-phrase-token-count",
        type=int,
        default=4,
        help="Number of content tokens per text-instance context phrase.",
    )
    parser.add_argument(
        "--text-instance-context-min-token-len",
        type=int,
        default=4,
        help="Minimum normalized token length used in text-instance context phrases.",
    )
    parser.add_argument(
        "--text-instance-context-verification-bonus",
        type=float,
        default=0.0,
        help=(
            "Optional non-standalone bonus applied when a page matches both a "
            "text_instance and start_byte-derived text_instance_context. This lets "
            "start_byte context verify/disambiguate text evidence without allowing "
            "context-only matches to create labels."
        ),
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument(
        "--output-augmented-gold-jsonl",
        default="",
        help=(
            "Optional MMQA JSONL copy with metadata.gold_page_uids populated from "
            "high-scoring pseudo labels."
        ),
    )
    return parser.parse_args()


def parse_evidence_weight_overrides(raw: str) -> dict[str, float]:
    weights = dict(DEFAULT_EVIDENCE_WEIGHTS)
    value = str(raw or "").strip()
    if not value:
        return weights

    if value.startswith("{"):
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("--evidence-weight-overrides JSON value must be an object")
        items = parsed.items()
    else:
        pairs = [item.strip() for item in value.split(",") if item.strip()]
        items = []
        for pair in pairs:
            if "=" not in pair:
                raise ValueError(
                    "Evidence weight overrides must use source=value pairs; "
                    f"got {pair!r}"
                )
            source, weight = pair.split("=", 1)
            items.append((source.strip(), weight.strip()))

    for source, weight in items:
        source = str(source).strip()
        if source not in weights:
            valid = ", ".join(sorted(weights))
            raise ValueError(f"Unknown evidence source {source!r}. Valid sources: {valid}")
        numeric_weight = float(weight)
        if numeric_weight < 0:
            raise ValueError(f"Evidence weight for {source!r} must be non-negative")
        weights[source] = numeric_weight
    return weights


@dataclass(frozen=True)
class Evidence:
    text: str
    source: str
    weight: float
    doc_id: str = ""


@dataclass
class PageScore:
    page_uid: str
    doc_id: str
    page_idx: int
    score: float = 0.0
    exact_matches: list[dict[str, Any]] = field(default_factory=list)
    fuzzy_matches: list[dict[str, Any]] = field(default_factory=list)
    verification_matches: list[dict[str, Any]] = field(default_factory=list)

    def add_exact(self, evidence: Evidence) -> None:
        self.score += float(evidence.weight)
        self.exact_matches.append(
            {
                "source": evidence.source,
                "text": evidence.text,
                "weight": float(evidence.weight),
            }
        )

    def add_fuzzy(self, evidence: Evidence, overlap: float) -> None:
        gain = float(evidence.weight) * 0.45 * float(overlap)
        self.score += gain
        self.fuzzy_matches.append(
            {
                "source": evidence.source,
                "text": evidence.text,
                "overlap": float(overlap),
                "weight": float(gain),
            }
        )

    def add_verification_bonus(self, *, source: str, reason: str, weight: float) -> None:
        self.score += float(weight)
        self.verification_matches.append(
            {
                "source": source,
                "reason": reason,
                "weight": float(weight),
            }
        )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_id_map(path: str) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    return {str(row["id"]): row for row in load_jsonl(Path(path)) if row.get("id")}


def load_url_to_id(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for doc_id, row in load_id_map(path).items():
        url = normalize_url(str(row.get("url", "")))
        if url:
            out[url] = doc_id
    return out


def load_by_id(path: str) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    return {str(row["id"]): row for row in load_jsonl(Path(path)) if row.get("id")}


def normalize_url(url: str) -> str:
    value = str(url or "").strip()
    if not value:
        return ""
    return value.replace("http://", "https://").rstrip("/")


def page_uid(doc_id: str, page_idx: int) -> str:
    return f"{doc_id}_page{int(page_idx)}"


def parse_page_idx(row: dict[str, Any]) -> int:
    for key in ("page_idx", "page_id", "page"):
        if row.get(key) is not None:
            return int(row[key])
    uid = str(row.get("page_uid", ""))
    if "_page" in uid:
        return int(uid.rsplit("_page", 1)[1])
    raise ValueError(f"Page row missing page index: {row}")


def load_page_texts(path: Path) -> dict[str, list[dict[str, Any]]]:
    pages_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(path):
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
        pages_by_doc[doc_id].append(
            {
                "page_uid": uid,
                "doc_id": doc_id,
                "page_idx": page_idx,
                "text": text,
                "norm_text": normalize_text(text),
                "tokens": set(tokenize(text)),
            }
        )
    for rows in pages_by_doc.values():
        rows.sort(key=lambda item: int(item["page_idx"]))
    return dict(pages_by_doc)


def page_text(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("text", "page_text", "ocr_text", "markdown", "vlm_text", "content"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)
    return "\n".join(parts)


def normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize_text(text))


CONTEXT_STOPWORDS = {
    "about",
    "after",
    "also",
    "among",
    "because",
    "before",
    "being",
    "between",
    "during",
    "from",
    "have",
    "into",
    "that",
    "their",
    "there",
    "these",
    "they",
    "this",
    "through",
    "where",
    "which",
    "while",
    "with",
    "would",
}


def text_document_body(row: dict[str, Any]) -> str:
    for key in ("text", "content", "paragraphs"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, list):
            parts = [str(item).strip() for item in value if str(item).strip()]
            if parts:
                return "\n".join(parts)
    return ""


def byte_offset_to_char_index(text: str, start_byte: Any) -> int | None:
    try:
        offset = int(start_byte)
    except (TypeError, ValueError):
        return None
    if offset < 0:
        return None
    encoded = text.encode("utf-8")
    offset = min(offset, len(encoded))
    return len(encoded[:offset].decode("utf-8", errors="ignore"))


def text_instance_context_window(
    *,
    text_row: dict[str, Any],
    start_byte: Any,
    instance_text: str,
    window_chars: int,
) -> str:
    full_text = text_document_body(text_row)
    if not full_text or int(window_chars) <= 0:
        return ""
    char_idx = byte_offset_to_char_index(full_text, start_byte)
    if char_idx is None:
        normalized_instance = normalize_text(instance_text)
        normalized_full = normalize_text(full_text)
        if not normalized_instance:
            return ""
        approx = normalized_full.find(normalized_instance)
        if approx < 0:
            return ""
        char_idx = approx
    start = max(0, int(char_idx) - int(window_chars))
    end = min(len(full_text), int(char_idx) + len(str(instance_text or "")) + int(window_chars))
    return full_text[start:end]


def text_instance_context_phrases(
    *,
    text_row: dict[str, Any],
    start_byte: Any,
    instance_text: str,
    window_chars: int,
    max_phrases: int,
    phrase_token_count: int,
    min_token_len: int,
) -> list[str]:
    context = text_instance_context_window(
        text_row=text_row,
        start_byte=start_byte,
        instance_text=instance_text,
        window_chars=window_chars,
    )
    if not context or int(max_phrases) <= 0:
        return []
    answer_tokens = set(tokenize(instance_text))
    tokens: list[str] = []
    seen: set[str] = set()
    for token in tokenize(context):
        if len(token) < int(min_token_len):
            continue
        if token in CONTEXT_STOPWORDS or token in answer_tokens:
            continue
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    phrase_len = max(2, int(phrase_token_count))
    phrases: list[str] = []
    for idx in range(0, len(tokens), phrase_len):
        phrase_tokens = tokens[idx : idx + phrase_len]
        if len(phrase_tokens) < min(phrase_len, 2):
            continue
        phrases.append(" ".join(phrase_tokens))
        if len(phrases) >= int(max_phrases):
            break
    return phrases


def clean_phrase(text: Any) -> str:
    value = str(text or "").strip()
    value = re.sub(r"\s+", " ", value)
    return value


def phrase_is_useful(text: str) -> bool:
    tokens = tokenize(text)
    if not tokens:
        return False
    if len(tokens) >= 2:
        return True
    token = tokens[0]
    return bool(re.fullmatch(r"\d{2,4}", token) or len(token) >= 4)


def add_evidence(
    out: list[Evidence],
    seen: set[tuple[str, str]],
    *,
    text: Any,
    source: str,
    weight: float,
    doc_id: str = "",
) -> None:
    phrase = clean_phrase(text)
    if not phrase_is_useful(phrase):
        return
    key = (source, normalize_text(phrase))
    if key in seen:
        return
    seen.add(key)
    out.append(Evidence(text=phrase, source=source, weight=float(weight), doc_id=str(doc_id or "")))


def supporting_doc_ids(row: dict[str, Any]) -> list[str]:
    docs: list[str] = []
    seen: set[str] = set()
    for ctx in row.get("supporting_context", []):
        if not isinstance(ctx, dict):
            continue
        doc_id = str(ctx.get("doc_id", "")).strip()
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            docs.append(doc_id)
    return docs


def supporting_doc_parts(row: dict[str, Any]) -> dict[str, set[str]]:
    parts: dict[str, set[str]] = defaultdict(set)
    for ctx in row.get("supporting_context", []):
        if not isinstance(ctx, dict):
            continue
        doc_id = str(ctx.get("doc_id", "")).strip()
        part = str(ctx.get("doc_part", "")).strip()
        if doc_id and part:
            parts[doc_id].add(part)
    return parts


def iter_answer_objects(row: dict[str, Any]) -> list[dict[str, Any]]:
    answers: list[dict[str, Any]] = []
    for answer in row.get("answers", []):
        if isinstance(answer, dict):
            answers.append(answer)
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    for group in metadata.get("intermediate_answers", []) or []:
        if isinstance(group, list):
            answers.extend(answer for answer in group if isinstance(answer, dict))
    return answers


def table_cell_text(table: dict[str, Any], row_idx: int, col_idx: int) -> str:
    rows = table.get("table", {}).get("table_rows", [])
    if row_idx < 0 or row_idx >= len(rows):
        return ""
    cells = rows[row_idx]
    if col_idx < 0 or col_idx >= len(cells):
        return ""
    return clean_phrase(cells[col_idx].get("text", ""))


def table_row_cells(table: dict[str, Any], row_idx: int) -> list[dict[str, Any]]:
    rows = table.get("table", {}).get("table_rows", [])
    if row_idx < 0 or row_idx >= len(rows):
        return []
    return [cell for cell in rows[row_idx] if isinstance(cell, dict)]


def bracketed_pseudo_question_terms(row: dict[str, Any]) -> list[str]:
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    text = str(metadata.get("pseudo_language_question", "") or "")
    return [clean_phrase(value) for value in re.findall(r"\[([^\]]+)\]", text)]


def collect_question_entity_terms(row: dict[str, Any]) -> list[str]:
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    terms: list[str] = []
    for field in ("wiki_entities_in_question",):
        for entity in metadata.get(field, []) or []:
            if not isinstance(entity, dict):
                continue
            for key in ("text", "wiki_title"):
                value = clean_phrase(entity.get(key))
                if value:
                    terms.append(value)
    return terms


def collect_answer_entity_terms(row: dict[str, Any]) -> list[str]:
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    terms: list[str] = []
    for entity in metadata.get("wiki_entities_in_answers", []) or []:
        if not isinstance(entity, dict):
            continue
        for key in ("text", "wiki_title"):
            value = clean_phrase(entity.get(key))
            if value:
                terms.append(value)
    return terms


def evidence_for_row(
    row: dict[str, Any],
    *,
    tables_by_id: dict[str, dict[str, Any]],
    texts_by_id: dict[str, dict[str, Any]],
    images_by_id: dict[str, dict[str, Any]],
    id_map: dict[str, dict[str, Any]],
    url_to_id: dict[str, str],
    evidence_weights: dict[str, float],
    text_instance_context_window_chars: int = 0,
    text_instance_context_max_phrases: int = 6,
    text_instance_context_phrase_token_count: int = 4,
    text_instance_context_min_token_len: int = 4,
) -> tuple[list[Evidence], dict[str, Any]]:
    evidence: list[Evidence] = []
    seen: set[tuple[str, str]] = set()
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}

    for answer in iter_answer_objects(row):
        add_evidence(
            evidence,
            seen,
            text=answer.get("answer"),
            source="answer_text",
            weight=evidence_weights["answer_text"],
        )

        for instance in answer.get("text_instances", []) or []:
            if not isinstance(instance, dict):
                continue
            add_evidence(
                evidence,
                seen,
                text=instance.get("text"),
                source="text_instance",
                weight=evidence_weights["text_instance"],
                doc_id=instance.get("doc_id", ""),
            )
            if int(text_instance_context_window_chars) > 0:
                doc_id = str(instance.get("doc_id", "") or "").strip()
                text_row = texts_by_id.get(doc_id, {})
                for phrase in text_instance_context_phrases(
                    text_row=text_row,
                    start_byte=instance.get("start_byte"),
                    instance_text=str(instance.get("text", "") or ""),
                    window_chars=int(text_instance_context_window_chars),
                    max_phrases=int(text_instance_context_max_phrases),
                    phrase_token_count=int(text_instance_context_phrase_token_count),
                    min_token_len=int(text_instance_context_min_token_len),
                ):
                    add_evidence(
                        evidence,
                        seen,
                        text=phrase,
                        source="text_instance_context",
                        weight=evidence_weights["text_instance_context"],
                        doc_id=doc_id,
                    )

        for instance in answer.get("image_instances", []) or []:
            if not isinstance(instance, dict):
                continue
            doc_id = str(instance.get("doc_id", "")).strip()
            image_row = images_by_id.get(doc_id, {})
            add_evidence(
                evidence,
                seen,
                text=image_row.get("title"),
                source="image_title",
                weight=evidence_weights["image_title"],
                doc_id=doc_id,
            )
            add_evidence(
                evidence,
                seen,
                text=doc_title_from_map(doc_id, image_row, id_map),
                source="image_doc_title",
                weight=evidence_weights["image_doc_title"],
                doc_id=doc_id,
            )

        for row_idx, col_idx in answer.get("table_indices", []) or []:
            table_id = str(metadata.get("table_id", "")).strip()
            table = tables_by_id.get(table_id, {})
            if not table:
                continue
            add_evidence(
                evidence,
                seen,
                text=table.get("title"),
                source="table_title",
                weight=evidence_weights["table_title"],
                doc_id=table_id,
            )
            cell_text = table_cell_text(table, int(row_idx), int(col_idx))
            add_evidence(
                evidence,
                seen,
                text=cell_text,
                source="table_answer_cell",
                weight=evidence_weights["table_answer_cell"],
                doc_id=table_id,
            )
            for cell in table_row_cells(table, int(row_idx)):
                add_evidence(
                    evidence,
                    seen,
                    text=cell.get("text"),
                    source="table_row_cell",
                    weight=evidence_weights["table_row_cell"],
                    doc_id=table_id,
                )
                for link in cell.get("links", []) or []:
                    if not isinstance(link, dict):
                        continue
                    linked_doc_id = url_to_id.get(normalize_url(str(link.get("url", ""))), "")
                    add_evidence(
                        evidence,
                        seen,
                        text=link.get("text"),
                        source="table_row_link_text",
                        weight=evidence_weights["table_row_link_text"],
                        doc_id=linked_doc_id,
                    )
                    add_evidence(
                        evidence,
                        seen,
                        text=link.get("wiki_title"),
                        source="table_row_link_title",
                        weight=evidence_weights["table_row_link_title"],
                        doc_id=linked_doc_id,
                    )

    for doc_id in supporting_doc_ids(row):
        side_row = texts_by_id.get(doc_id) or images_by_id.get(doc_id) or tables_by_id.get(doc_id) or {}
        add_evidence(
            evidence,
            seen,
            text=doc_title_from_map(doc_id, side_row, id_map),
            source="supporting_doc_title",
            weight=evidence_weights["supporting_doc_title"],
            doc_id=doc_id,
        )

    for term in collect_answer_entity_terms(row):
        add_evidence(
            evidence,
            seen,
            text=term,
            source="answer_entity",
            weight=evidence_weights["answer_entity"],
        )
    for term in collect_question_entity_terms(row):
        add_evidence(
            evidence,
            seen,
            text=term,
            source="question_entity",
            weight=evidence_weights["question_entity"],
        )
    for term in bracketed_pseudo_question_terms(row):
        add_evidence(
            evidence,
            seen,
            text=term,
            source="pseudo_question_slot",
            weight=evidence_weights["pseudo_question_slot"],
        )

    source_counts = Counter(item.source for item in evidence)
    positive_source_counts = Counter(item.source for item in evidence if float(item.weight) > 0.0)
    return evidence, {
        "evidence_count": len(evidence),
        "evidence_source_counts": dict(sorted(source_counts.items())),
        "positive_evidence_count": int(sum(positive_source_counts.values())),
        "positive_evidence_source_counts": dict(sorted(positive_source_counts.items())),
    }


def direct_evidence_cap_counts_for_row(
    row: dict[str, Any],
    *,
    tables_by_id: dict[str, dict[str, Any]],
    images_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, int], dict[str, Any]]:
    """Count MMQA direct evidence units per source document.

    This is intentionally separate from evidence_for_row(). The label score can
    use weighted phrase evidence, but adaptive caps should reflect how many
    document-grounded evidence units MMQA says the question needs.
    """

    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    table_id = str(metadata.get("table_id", "") or "").strip()
    table = tables_by_id.get(table_id, {}) if table_id else {}

    per_doc: Counter[str] = Counter()
    per_source: Counter[str] = Counter()

    def add_unit(*, doc_id: Any, source: str, text: Any) -> None:
        clean_doc_id = str(doc_id or "").strip()
        if not clean_doc_id:
            return
        phrase = clean_phrase(text)
        if not phrase_is_useful(phrase):
            return
        per_doc[clean_doc_id] += 1
        per_source[source] += 1

    for answer in iter_answer_objects(row):
        for instance in answer.get("text_instances", []) or []:
            if not isinstance(instance, dict):
                continue
            add_unit(
                doc_id=instance.get("doc_id", ""),
                source="text_instance",
                text=instance.get("text"),
            )

        for instance in answer.get("image_instances", []) or []:
            if not isinstance(instance, dict):
                continue
            doc_id = str(instance.get("doc_id", "") or "").strip()
            image_row = images_by_id.get(doc_id, {})
            add_unit(
                doc_id=doc_id,
                source="image_title",
                text=image_row.get("title"),
            )

        for row_idx, col_idx in answer.get("table_indices", []) or []:
            if not table_id or not table:
                continue
            cell_text = table_cell_text(table, int(row_idx), int(col_idx))
            add_unit(
                doc_id=table_id,
                source="table_answer_cell",
                text=cell_text,
            )

    return dict(per_doc), {
        "adaptive_cap_direct_evidence_unit_count": int(sum(per_doc.values())),
        "adaptive_cap_source_counts": dict(sorted(per_source.items())),
        "adaptive_cap_doc_unit_counts": dict(sorted(per_doc.items())),
    }


def bounded_adaptive_caps(
    raw_doc_unit_counts: dict[str, int],
    *,
    max_pages_per_doc: int,
    max_pages_per_qid: int,
) -> tuple[dict[str, int], int]:
    doc_caps: dict[str, int] = {}
    for doc_id, count in raw_doc_unit_counts.items():
        numeric_count = max(0, int(count))
        if numeric_count <= 0:
            continue
        if int(max_pages_per_doc) > 0:
            numeric_count = min(numeric_count, int(max_pages_per_doc))
        doc_caps[str(doc_id)] = numeric_count

    qid_cap = int(sum(doc_caps.values()))
    if int(max_pages_per_qid) > 0:
        qid_cap = min(qid_cap, int(max_pages_per_qid))
    return doc_caps, qid_cap


def doc_title_from_map(doc_id: str, side_row: dict[str, Any], id_map: dict[str, dict[str, Any]]) -> str:
    if side_row.get("title"):
        return clean_phrase(side_row.get("title"))
    url = str((side_row or {}).get("url") or id_map.get(doc_id, {}).get("url") or "")
    if "/wiki/" not in url:
        return ""
    title = url.rsplit("/wiki/", 1)[1].replace("_", " ")
    return clean_phrase(title)


def phrase_match(norm_phrase: str, page: dict[str, Any], *, min_token_overlap: float) -> tuple[str, float]:
    if not norm_phrase:
        return "", 0.0
    norm_text = str(page["norm_text"])
    if f" {norm_phrase} " in f" {norm_text} ":
        return "exact", 1.0
    phrase_tokens = set(tokenize(norm_phrase))
    if not phrase_tokens:
        return "", 0.0
    overlap = len(phrase_tokens & page["tokens"]) / float(len(phrase_tokens))
    if overlap >= float(min_token_overlap):
        return "fuzzy", overlap
    return "", overlap


def score_pages(
    row: dict[str, Any],
    evidence: list[Evidence],
    pages_by_doc: dict[str, list[dict[str, Any]]],
    *,
    min_token_overlap: float,
    text_instance_context_verification_bonus: float = 0.0,
) -> tuple[list[PageScore], list[str]]:
    gold_docs = supporting_doc_ids(row)
    missing_docs = [doc_id for doc_id in gold_docs if doc_id not in pages_by_doc]
    scores: list[PageScore] = []

    for doc_id in gold_docs:
        for page in pages_by_doc.get(doc_id, []):
            page_score = PageScore(
                page_uid=str(page["page_uid"]),
                doc_id=doc_id,
                page_idx=int(page["page_idx"]),
            )
            for item in evidence:
                if item.doc_id and item.doc_id != doc_id:
                    continue
                norm_phrase = normalize_text(item.text)
                mode, overlap = phrase_match(norm_phrase, page, min_token_overlap=min_token_overlap)
                if mode == "exact":
                    page_score.add_exact(item)
                elif mode == "fuzzy":
                    page_score.add_fuzzy(item, overlap)
            if page_score.score > 0:
                apply_text_instance_context_verification(
                    page_score,
                    bonus=float(text_instance_context_verification_bonus),
                )
                scores.append(page_score)

    scores.sort(key=lambda item: (-item.score, item.doc_id, item.page_idx))
    return scores, missing_docs


def page_matched_sources(item: PageScore) -> set[str]:
    return {
        str(match.get("source", ""))
        for match in item.exact_matches + item.fuzzy_matches
        if str(match.get("source", "")).strip()
    }


def apply_text_instance_context_verification(item: PageScore, *, bonus: float) -> None:
    if float(bonus) <= 0.0:
        return
    sources = page_matched_sources(item)
    if "text_instance" not in sources or "text_instance_context" not in sources:
        return
    item.add_verification_bonus(
        source="text_instance_context_verification",
        reason="page matches text_instance and start_byte-derived local context",
        weight=float(bonus),
    )


def select_labels(
    scores: list[PageScore],
    *,
    min_score: float,
    top_pages_per_doc: int,
    top_pages_per_qid: int,
    doc_caps: dict[str, int] | None = None,
) -> list[PageScore]:
    selected: list[PageScore] = []
    per_doc_count: Counter[str] = Counter()
    for item in scores:
        if item.score < float(min_score):
            continue
        if doc_caps is not None:
            doc_limit = int(doc_caps.get(item.doc_id, 0))
            if doc_limit <= 0 or per_doc_count[item.doc_id] >= doc_limit:
                continue
        elif int(top_pages_per_doc) > 0 and per_doc_count[item.doc_id] >= int(top_pages_per_doc):
            continue
        selected.append(item)
        per_doc_count[item.doc_id] += 1
        if int(top_pages_per_qid) > 0 and len(selected) >= int(top_pages_per_qid):
            break
    return selected


def evidence_match_key(match: dict[str, Any]) -> str:
    return "\t".join(
        [
            str(match.get("source", "")).strip(),
            normalize_text(str(match.get("text", ""))),
        ]
    )


def evidence_match_weight(match: dict[str, Any]) -> float:
    try:
        return float(match.get("weight", 0.0))
    except (TypeError, ValueError):
        return 0.0


def page_evidence_keys(item: PageScore, *, min_match_weight: float = 0.0) -> set[str]:
    out = set()
    for match in item.exact_matches + item.fuzzy_matches:
        if evidence_match_weight(match) < float(min_match_weight):
            continue
        key = evidence_match_key(match)
        if key.strip():
            out.add(key)
    return out


def page_evidence_weight_by_key(item: PageScore, *, min_match_weight: float = 0.0) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for match in item.exact_matches + item.fuzzy_matches:
        weight = evidence_match_weight(match)
        if weight < float(min_match_weight):
            continue
        key = evidence_match_key(match)
        if key.strip():
            out[key] += weight
    return dict(out)


def select_labels_by_evidence_coverage(
    scores: list[PageScore],
    *,
    min_score: float,
    top_pages_per_doc: int,
    top_pages_per_qid: int,
    coverage_min_match_weight: float = 3.0,
    doc_caps: dict[str, int] | None = None,
) -> list[PageScore]:
    eligible = [item for item in scores if item.score >= float(min_score)]
    selected: list[PageScore] = []
    selected_uids: set[str] = set()
    per_doc_count: Counter[str] = Counter()
    covered_by_doc: dict[str, set[str]] = defaultdict(set)

    while eligible:
        if int(top_pages_per_qid) > 0 and len(selected) >= int(top_pages_per_qid):
            break
        best: PageScore | None = None
        best_key: tuple[int, float, float, int, str] | None = None
        for item in eligible:
            if item.page_uid in selected_uids:
                continue
            if doc_caps is not None:
                doc_limit = int(doc_caps.get(item.doc_id, 0))
                if doc_limit <= 0 or per_doc_count[item.doc_id] >= doc_limit:
                    continue
            elif int(top_pages_per_doc) > 0 and per_doc_count[item.doc_id] >= int(top_pages_per_doc):
                continue
            evidence_keys = page_evidence_keys(
                item,
                min_match_weight=float(coverage_min_match_weight),
            )
            new_keys = evidence_keys - covered_by_doc[item.doc_id]
            if not new_keys:
                continue
            weight_by_key = page_evidence_weight_by_key(
                item,
                min_match_weight=float(coverage_min_match_weight),
            )
            new_weight = sum(weight_by_key.get(key, 0.0) for key in new_keys)
            candidate_key = (
                len(new_keys),
                float(new_weight),
                float(item.score),
                -int(item.page_idx),
                item.page_uid,
            )
            if best_key is None or candidate_key > best_key:
                best = item
                best_key = candidate_key
        if best is None:
            break
        selected.append(best)
        selected_uids.add(best.page_uid)
        per_doc_count[best.doc_id] += 1
        covered_by_doc[best.doc_id].update(
            page_evidence_keys(best, min_match_weight=float(coverage_min_match_weight))
        )

    selected.sort(key=lambda item: (-float(item.score), item.doc_id, int(item.page_idx)))
    return selected


def confidence(score: float) -> str:
    if score >= 14.0:
        return "high"
    if score >= 8.0:
        return "medium"
    if score > 0.0:
        return "low"
    return "none"


def positive_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [match for match in matches if evidence_match_weight(match) > 0.0]


def diagnostic_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [match for match in matches if evidence_match_weight(match) <= 0.0]


def page_score_record(item: PageScore) -> dict[str, Any]:
    return {
        "page_uid": item.page_uid,
        "doc_id": item.doc_id,
        "page_idx": int(item.page_idx),
        "score": round(float(item.score), 6),
        "confidence": confidence(item.score),
        "exact_matches": item.exact_matches[:20],
        "fuzzy_matches": item.fuzzy_matches[:20],
        "positive_exact_matches": positive_matches(item.exact_matches)[:20],
        "positive_fuzzy_matches": positive_matches(item.fuzzy_matches)[:20],
        "diagnostic_exact_matches": diagnostic_matches(item.exact_matches)[:20],
        "diagnostic_fuzzy_matches": diagnostic_matches(item.fuzzy_matches)[:20],
        "verification_matches": item.verification_matches[:20],
    }


def label_status(selected: list[PageScore], missing_docs: list[str], gold_docs: list[str]) -> str:
    if selected:
        return "matched"
    if missing_docs and len(missing_docs) == len(gold_docs):
        return "missing_all_supporting_doc_page_text"
    if missing_docs:
        return "missing_some_supporting_doc_page_text"
    return "no_evidence_match"


def augmented_gold_row(row: dict[str, Any], selected: list[PageScore], *, selection_policy: str) -> dict[str, Any]:
    out = json.loads(json.dumps(row))
    metadata = out.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        out["metadata"] = metadata
    uids = [item.page_uid for item in selected]
    metadata["pseudo_gold_page_uids"] = uids
    metadata["gold_page_uids"] = uids
    metadata["pseudo_gold_page_label_source"] = "mmqa_evidence_page_text"
    metadata["pseudo_gold_page_label_scores"] = {
        item.page_uid: round(float(item.score), 6) for item in selected
    }
    metadata["pseudo_gold_page_label_selection_policy"] = selection_policy
    return out


def write_markdown_summary(path: Path, summary: dict[str, Any]) -> None:
    md_path = path.with_suffix(".md")
    lines = [
        "# MMQA Pseudo Page Label Summary",
        "",
        f"- qid_count: `{summary['qid_count']}`",
        f"- matched_qid_count: `{summary['matched_qid_count']}`",
        f"- matched_qid_fraction: `{summary['matched_qid_fraction']}`",
        f"- pseudo_page_label_count: `{summary['pseudo_page_label_count']}`",
        f"- mean_labels_per_matched_qid: `{summary['mean_labels_per_matched_qid']}`",
        "",
        "## Status Counts",
        "",
    ]
    for key, value in summary["status_counts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Question Type Counts", ""])
    for key, value in summary["matched_by_question_type"].items():
        lines.append(f"- `{key}`: `{value}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    gold_rows = load_jsonl(Path(args.gold))
    pages_by_doc = load_page_texts(Path(args.doc_pages_jsonl))
    texts_by_id = load_by_id(args.mmqa_texts_jsonl)
    tables_by_id = load_by_id(args.mmqa_tables_jsonl)
    images_by_id = load_by_id(args.mmqa_images_jsonl)
    id_map = load_id_map(args.id_url_mapping_jsonl)
    url_to_id = load_url_to_id(args.id_url_mapping_jsonl)
    evidence_weights = parse_evidence_weight_overrides(args.evidence_weight_overrides)

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json = Path(args.output_summary_json)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    augmented_path = Path(args.output_augmented_gold_jsonl) if args.output_augmented_gold_jsonl else None
    if augmented_path:
        augmented_path.parent.mkdir(parents=True, exist_ok=True)

    status_counts: Counter[str] = Counter()
    qtype_counts: Counter[str] = Counter()
    matched_qtype_counts: Counter[str] = Counter()
    label_counts: list[int] = []
    score_values: list[float] = []
    source_counts: Counter[str] = Counter()
    positive_source_counts: Counter[str] = Counter()
    verification_source_counts: Counter[str] = Counter()
    missing_doc_count = 0
    matched_qids = 0
    adaptive_qid_caps: list[int] = []
    adaptive_direct_unit_counts: list[int] = []
    adaptive_doc_cap_values: list[int] = []
    adaptive_doc_unit_values: list[int] = []

    with output_jsonl.open("w", encoding="utf-8") as out_handle:
        aug_handle = augmented_path.open("w", encoding="utf-8") if augmented_path else None
        try:
            for row in gold_rows:
                qid = str(row.get("qid", "")).strip()
                metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
                question_type = str(metadata.get("type", "UNKNOWN")).strip() or "UNKNOWN"
                qtype_counts[question_type] += 1

                evidence, evidence_meta = evidence_for_row(
                    row,
                    tables_by_id=tables_by_id,
                    texts_by_id=texts_by_id,
                    images_by_id=images_by_id,
                    id_map=id_map,
                    url_to_id=url_to_id,
                    evidence_weights=evidence_weights,
                    text_instance_context_window_chars=int(args.text_instance_context_window_chars),
                    text_instance_context_max_phrases=int(args.text_instance_context_max_phrases),
                    text_instance_context_phrase_token_count=int(args.text_instance_context_phrase_token_count),
                    text_instance_context_min_token_len=int(args.text_instance_context_min_token_len),
                )
                source_counts.update(evidence_meta["evidence_source_counts"])
                positive_source_counts.update(evidence_meta["positive_evidence_source_counts"])
                scored, missing_docs = score_pages(
                    row,
                    evidence,
                    pages_by_doc,
                    min_token_overlap=float(args.min_token_overlap),
                    text_instance_context_verification_bonus=float(
                        args.text_instance_context_verification_bonus
                    ),
                )
                adaptive_meta: dict[str, Any] = {}
                adaptive_doc_caps: dict[str, int] | None = None
                effective_top_pages_per_qid = int(args.top_pages_per_qid)
                if bool(args.adaptive_page_caps):
                    raw_doc_unit_counts, cap_meta = direct_evidence_cap_counts_for_row(
                        row,
                        tables_by_id=tables_by_id,
                        images_by_id=images_by_id,
                    )
                    bounded_doc_caps, bounded_qid_cap = bounded_adaptive_caps(
                        raw_doc_unit_counts,
                        max_pages_per_doc=int(args.top_pages_per_doc),
                        max_pages_per_qid=int(args.top_pages_per_qid),
                    )
                    adaptive_meta.update(cap_meta)
                    adaptive_meta["adaptive_doc_caps"] = dict(sorted(bounded_doc_caps.items()))
                    adaptive_meta["adaptive_qid_cap"] = int(bounded_qid_cap)
                    adaptive_direct_unit_counts.append(
                        int(cap_meta["adaptive_cap_direct_evidence_unit_count"])
                    )
                    adaptive_doc_unit_values.extend(int(v) for v in raw_doc_unit_counts.values())
                    adaptive_doc_cap_values.extend(int(v) for v in bounded_doc_caps.values())
                    adaptive_qid_caps.append(int(bounded_qid_cap))
                    if bounded_doc_caps and bounded_qid_cap > 0:
                        adaptive_doc_caps = bounded_doc_caps
                        effective_top_pages_per_qid = int(bounded_qid_cap)
                if args.selection_policy == "evidence_coverage":
                    selected = select_labels_by_evidence_coverage(
                        scored,
                        min_score=float(args.min_score),
                        top_pages_per_doc=int(args.top_pages_per_doc),
                        top_pages_per_qid=int(effective_top_pages_per_qid),
                        coverage_min_match_weight=float(args.coverage_min_match_weight),
                        doc_caps=adaptive_doc_caps,
                    )
                else:
                    selected = select_labels(
                        scored,
                        min_score=float(args.min_score),
                        top_pages_per_doc=int(args.top_pages_per_doc),
                        top_pages_per_qid=int(effective_top_pages_per_qid),
                        doc_caps=adaptive_doc_caps,
                    )
                gold_docs = supporting_doc_ids(row)
                status = label_status(selected, missing_docs, gold_docs)
                status_counts[status] += 1
                missing_doc_count += len(missing_docs)
                if selected:
                    matched_qids += 1
                    matched_qtype_counts[question_type] += 1
                label_counts.append(len(selected))
                score_values.extend(float(item.score) for item in selected)
                for item in selected:
                    verification_source_counts.update(
                        str(match.get("source", ""))
                        for match in item.verification_matches
                        if str(match.get("source", "")).strip()
                    )

                output_row = {
                    "qid": qid,
                    "question": row.get("question", ""),
                    "question_type": question_type,
                    "modalities": metadata.get("modalities", []),
                    "gold_doc_ids": gold_docs,
                    "supporting_doc_parts": {
                        doc_id: sorted(parts) for doc_id, parts in supporting_doc_parts(row).items()
                    },
                    "status": status,
                    "pseudo_gold_page_uids": [item.page_uid for item in selected],
                    "pseudo_gold_pages": [page_score_record(item) for item in selected],
                    "candidate_pages_scored": len(scored),
                    "top_scored_pages": [page_score_record(item) for item in scored[:10]],
                    "missing_page_text_doc_ids": missing_docs,
                    "adaptive_page_caps": bool(args.adaptive_page_caps),
                    "effective_top_pages_per_qid": int(effective_top_pages_per_qid),
                    **evidence_meta,
                    **adaptive_meta,
                }
                out_handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                if aug_handle is not None:
                    aug_handle.write(
                        json.dumps(
                            augmented_gold_row(
                                row,
                                selected,
                                selection_policy=str(args.selection_policy),
                            ),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        finally:
            if aug_handle is not None:
                aug_handle.close()

    summary = {
        "gold": str(args.gold),
        "doc_pages_jsonl": str(args.doc_pages_jsonl),
        "mmqa_texts_jsonl": args.mmqa_texts_jsonl,
        "mmqa_tables_jsonl": args.mmqa_tables_jsonl,
        "mmqa_images_jsonl": args.mmqa_images_jsonl,
        "id_url_mapping_jsonl": args.id_url_mapping_jsonl,
        "qid_count": len(gold_rows),
        "page_text_doc_count": len(pages_by_doc),
        "matched_qid_count": int(matched_qids),
        "matched_qid_fraction": (
            round(float(matched_qids) / float(len(gold_rows)), 6) if gold_rows else None
        ),
        "pseudo_page_label_count": int(sum(label_counts)),
        "mean_labels_per_qid": round(float(fmean(label_counts)), 6) if label_counts else None,
        "mean_labels_per_matched_qid": (
            round(float(sum(label_counts)) / float(matched_qids), 6) if matched_qids else None
        ),
        "mean_selected_score": round(float(fmean(score_values)), 6) if score_values else None,
        "status_counts": dict(sorted(status_counts.items())),
        "question_type_counts": dict(sorted(qtype_counts.items())),
        "matched_by_question_type": dict(sorted(matched_qtype_counts.items())),
        "evidence_source_counts": dict(sorted(source_counts.items())),
        "positive_evidence_source_counts": dict(sorted(positive_source_counts.items())),
        "selected_verification_source_counts": dict(sorted(verification_source_counts.items())),
        "missing_page_text_doc_ref_count": int(missing_doc_count),
        "min_score": float(args.min_score),
        "top_pages_per_doc": int(args.top_pages_per_doc),
        "top_pages_per_qid": int(args.top_pages_per_qid),
        "adaptive_page_caps": bool(args.adaptive_page_caps),
        "adaptive_page_cap_sources": sorted(DIRECT_EVIDENCE_CAP_SOURCES),
        "mean_adaptive_qid_cap": (
            round(float(fmean(adaptive_qid_caps)), 6) if adaptive_qid_caps else None
        ),
        "adaptive_qid_cap_hist": dict(
            sorted(Counter(str(value) for value in adaptive_qid_caps).items())
        ),
        "mean_adaptive_direct_evidence_units_per_qid": (
            round(float(fmean(adaptive_direct_unit_counts)), 6)
            if adaptive_direct_unit_counts
            else None
        ),
        "adaptive_direct_evidence_unit_hist": dict(
            sorted(Counter(str(value) for value in adaptive_direct_unit_counts).items())
        ),
        "adaptive_raw_doc_unit_count_hist": dict(
            sorted(Counter(str(value) for value in adaptive_doc_unit_values).items())
        ),
        "adaptive_doc_cap_hist": dict(
            sorted(Counter(str(value) for value in adaptive_doc_cap_values).items())
        ),
        "min_token_overlap": float(args.min_token_overlap),
        "selection_policy": str(args.selection_policy),
        "coverage_min_match_weight": float(args.coverage_min_match_weight),
        "evidence_weight_overrides": str(args.evidence_weight_overrides),
        "evidence_weights": dict(sorted(evidence_weights.items())),
        "text_instance_context_window_chars": int(args.text_instance_context_window_chars),
        "text_instance_context_max_phrases": int(args.text_instance_context_max_phrases),
        "text_instance_context_phrase_token_count": int(args.text_instance_context_phrase_token_count),
        "text_instance_context_min_token_len": int(args.text_instance_context_min_token_len),
        "text_instance_context_verification_bonus": float(
            args.text_instance_context_verification_bonus
        ),
        "output_jsonl": str(output_jsonl),
        "output_augmented_gold_jsonl": str(augmented_path) if augmented_path else "",
    }
    output_summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown_summary(output_summary_json, summary)

    print(f"saved_pseudo_labels={output_jsonl}")
    if augmented_path:
        print(f"saved_augmented_gold={augmented_path}")
    print(f"saved_summary={output_summary_json}")
    print(f"qid_count={summary['qid_count']}")
    print(f"matched_qid_count={summary['matched_qid_count']}")
    print(f"matched_qid_fraction={summary['matched_qid_fraction']}")
    print(f"pseudo_page_label_count={summary['pseudo_page_label_count']}")


if __name__ == "__main__":
    main()
