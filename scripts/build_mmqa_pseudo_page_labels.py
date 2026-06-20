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
            weight=5.0,
        )

        for instance in answer.get("text_instances", []) or []:
            if not isinstance(instance, dict):
                continue
            add_evidence(
                evidence,
                seen,
                text=instance.get("text"),
                source="text_instance",
                weight=9.0,
                doc_id=instance.get("doc_id", ""),
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
                weight=7.0,
                doc_id=doc_id,
            )
            add_evidence(
                evidence,
                seen,
                text=doc_title_from_map(doc_id, image_row, id_map),
                source="image_doc_title",
                weight=4.0,
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
                weight=4.0,
                doc_id=table_id,
            )
            cell_text = table_cell_text(table, int(row_idx), int(col_idx))
            add_evidence(
                evidence,
                seen,
                text=cell_text,
                source="table_answer_cell",
                weight=10.0,
                doc_id=table_id,
            )
            for cell in table_row_cells(table, int(row_idx)):
                add_evidence(
                    evidence,
                    seen,
                    text=cell.get("text"),
                    source="table_row_cell",
                    weight=2.0,
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
                        weight=1.5,
                        doc_id=linked_doc_id,
                    )
                    add_evidence(
                        evidence,
                        seen,
                        text=link.get("wiki_title"),
                        source="table_row_link_title",
                        weight=1.5,
                        doc_id=linked_doc_id,
                    )

    for doc_id in supporting_doc_ids(row):
        side_row = texts_by_id.get(doc_id) or images_by_id.get(doc_id) or tables_by_id.get(doc_id) or {}
        add_evidence(
            evidence,
            seen,
            text=doc_title_from_map(doc_id, side_row, id_map),
            source="supporting_doc_title",
            weight=3.0,
            doc_id=doc_id,
        )

    for term in collect_answer_entity_terms(row):
        add_evidence(evidence, seen, text=term, source="answer_entity", weight=4.0)
    for term in collect_question_entity_terms(row):
        add_evidence(evidence, seen, text=term, source="question_entity", weight=1.2)
    for term in bracketed_pseudo_question_terms(row):
        add_evidence(evidence, seen, text=term, source="pseudo_question_slot", weight=1.0)

    source_counts = Counter(item.source for item in evidence)
    return evidence, {
        "evidence_count": len(evidence),
        "evidence_source_counts": dict(sorted(source_counts.items())),
    }


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
                scores.append(page_score)

    scores.sort(key=lambda item: (-item.score, item.doc_id, item.page_idx))
    return scores, missing_docs


def select_labels(
    scores: list[PageScore],
    *,
    min_score: float,
    top_pages_per_doc: int,
    top_pages_per_qid: int,
) -> list[PageScore]:
    selected: list[PageScore] = []
    per_doc_count: Counter[str] = Counter()
    for item in scores:
        if item.score < float(min_score):
            continue
        if int(top_pages_per_doc) > 0 and per_doc_count[item.doc_id] >= int(top_pages_per_doc):
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
            if int(top_pages_per_doc) > 0 and per_doc_count[item.doc_id] >= int(top_pages_per_doc):
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


def page_score_record(item: PageScore) -> dict[str, Any]:
    return {
        "page_uid": item.page_uid,
        "doc_id": item.doc_id,
        "page_idx": int(item.page_idx),
        "score": round(float(item.score), 6),
        "confidence": confidence(item.score),
        "exact_matches": item.exact_matches[:20],
        "fuzzy_matches": item.fuzzy_matches[:20],
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
    missing_doc_count = 0
    matched_qids = 0

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
                )
                source_counts.update(evidence_meta["evidence_source_counts"])
                scored, missing_docs = score_pages(
                    row,
                    evidence,
                    pages_by_doc,
                    min_token_overlap=float(args.min_token_overlap),
                )
                if args.selection_policy == "evidence_coverage":
                    selected = select_labels_by_evidence_coverage(
                        scored,
                        min_score=float(args.min_score),
                        top_pages_per_doc=int(args.top_pages_per_doc),
                        top_pages_per_qid=int(args.top_pages_per_qid),
                        coverage_min_match_weight=float(args.coverage_min_match_weight),
                    )
                else:
                    selected = select_labels(
                        scored,
                        min_score=float(args.min_score),
                        top_pages_per_doc=int(args.top_pages_per_doc),
                        top_pages_per_qid=int(args.top_pages_per_qid),
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
                    **evidence_meta,
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
        "missing_page_text_doc_ref_count": int(missing_doc_count),
        "min_score": float(args.min_score),
        "top_pages_per_doc": int(args.top_pages_per_doc),
        "top_pages_per_qid": int(args.top_pages_per_qid),
        "min_token_overlap": float(args.min_token_overlap),
        "selection_policy": str(args.selection_policy),
        "coverage_min_match_weight": float(args.coverage_min_match_weight),
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
