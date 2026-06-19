#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_mmqa_pseudo_page_labels as base


@dataclass(frozen=True)
class EvidenceUnit:
    unit_id: str
    unit_type: str
    doc_id: str
    evidence: tuple[base.Evidence, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UnitPageMatch:
    unit: EvidenceUnit
    page_uid: str
    doc_id: str
    page_idx: int
    score: float
    exact_matches: list[dict[str, Any]] = field(default_factory=list)
    fuzzy_matches: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PageLabel:
    page_uid: str
    doc_id: str
    page_idx: int
    score: float = 0.0
    unit_matches: list[UnitPageMatch] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build evidence-unit-aware pseudo-page labels for M3DocVQA/MMQA. "
            "Unlike the original page labeler, this script first maps each "
            "document-grounded MMQA evidence unit to pages in its own support "
            "document, then merges page labels with provenance."
        )
    )
    parser.add_argument("--gold", required=True, help="MMQA_train/dev.jsonl")
    parser.add_argument("--doc-pages-jsonl", required=True, help="Exported page text JSONL")
    parser.add_argument("--mmqa-texts-jsonl", default="", help="Optional MMQA_texts.jsonl")
    parser.add_argument("--mmqa-tables-jsonl", default="", help="Optional MMQA_tables.jsonl")
    parser.add_argument("--mmqa-images-jsonl", default="", help="Optional MMQA_images.jsonl")
    parser.add_argument("--id-url-mapping-jsonl", default="", help="Optional id_url_mapping.jsonl")
    parser.add_argument("--min-score", type=float, default=8.0)
    parser.add_argument("--high-confidence-score", type=float, default=14.0)
    parser.add_argument("--top-pages-per-qid", type=int, default=4)
    parser.add_argument("--max-pages-per-doc", type=int, default=3)
    parser.add_argument("--min-token-overlap", type=float, default=0.72)
    parser.add_argument(
        "--require-exact-or-high-confidence-fuzzy",
        action="store_true",
        default=True,
        help=(
            "Require a selected unit-page match to have at least one exact match, "
            "unless its score is at least --high-confidence-score."
        ),
    )
    parser.add_argument(
        "--allow-fuzzy-only-medium",
        action="store_false",
        dest="require_exact_or_high_confidence_fuzzy",
        help="Allow fuzzy-only unit-page matches when they pass --min-score.",
    )
    parser.add_argument(
        "--include-question-context-signals",
        action="store_true",
        help=(
            "Add weak question/entity/pseudo-question context terms to every "
            "evidence unit. Disabled by default for stricter labels."
        ),
    )
    parser.add_argument(
        "--require-all-units-mapped",
        action="store_true",
        help=(
            "Leave the QID unlabeled unless every document-grounded evidence unit "
            "maps to at least one page above threshold."
        ),
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument("--output-augmented-gold-jsonl", default="")
    return parser.parse_args()


def add_unit_evidence(
    out: list[base.Evidence],
    seen: set[tuple[str, str]],
    *,
    text: Any,
    source: str,
    weight: float,
    doc_id: str,
) -> None:
    base.add_evidence(out, seen, text=text, source=source, weight=weight, doc_id=doc_id)


def table_id_for_row(row: dict[str, Any]) -> str:
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    return str(metadata.get("table_id", "") or "").strip()


def answer_text(answer: dict[str, Any]) -> str:
    return base.clean_phrase(answer.get("answer", ""))


def context_evidence_for_doc(
    row: dict[str, Any],
    *,
    doc_id: str,
    texts_by_id: dict[str, dict[str, Any]],
    tables_by_id: dict[str, dict[str, Any]],
    images_by_id: dict[str, dict[str, Any]],
    id_map: dict[str, dict[str, Any]],
    include_question_context_signals: bool,
) -> list[base.Evidence]:
    evidence: list[base.Evidence] = []
    seen: set[tuple[str, str]] = set()
    side_row = texts_by_id.get(doc_id) or images_by_id.get(doc_id) or tables_by_id.get(doc_id) or {}
    add_unit_evidence(
        evidence,
        seen,
        text=base.doc_title_from_map(doc_id, side_row, id_map),
        source="supporting_doc_title",
        weight=3.0,
        doc_id=doc_id,
    )
    if include_question_context_signals:
        for term in base.collect_answer_entity_terms(row):
            add_unit_evidence(evidence, seen, text=term, source="answer_entity", weight=4.0, doc_id=doc_id)
        for term in base.collect_question_entity_terms(row):
            add_unit_evidence(evidence, seen, text=term, source="question_entity", weight=1.2, doc_id=doc_id)
        for term in base.bracketed_pseudo_question_terms(row):
            add_unit_evidence(evidence, seen, text=term, source="pseudo_question_slot", weight=1.0, doc_id=doc_id)
    return evidence


def evidence_units_for_row(
    row: dict[str, Any],
    *,
    tables_by_id: dict[str, dict[str, Any]],
    texts_by_id: dict[str, dict[str, Any]],
    images_by_id: dict[str, dict[str, Any]],
    id_map: dict[str, dict[str, Any]],
    url_to_id: dict[str, str],
    include_question_context_signals: bool,
) -> list[EvidenceUnit]:
    qid = str(row.get("qid", "")).strip()
    table_id = table_id_for_row(row)
    table = tables_by_id.get(table_id, {}) if table_id else {}
    units: list[EvidenceUnit] = []

    def append_unit(
        *,
        unit_type: str,
        doc_id: str,
        unit_evidence: list[base.Evidence],
        metadata: dict[str, Any],
    ) -> None:
        clean_doc_id = str(doc_id or "").strip()
        if not clean_doc_id or not unit_evidence:
            return
        context = context_evidence_for_doc(
            row,
            doc_id=clean_doc_id,
            texts_by_id=texts_by_id,
            tables_by_id=tables_by_id,
            images_by_id=images_by_id,
            id_map=id_map,
            include_question_context_signals=include_question_context_signals,
        )
        unit_id = f"{qid}::unit{len(units)}"
        units.append(
            EvidenceUnit(
                unit_id=unit_id,
                unit_type=unit_type,
                doc_id=clean_doc_id,
                evidence=tuple(unit_evidence + context),
                metadata=metadata,
            )
        )

    for answer_idx, answer in enumerate(base.iter_answer_objects(row)):
        answer_value = answer_text(answer)

        for instance_idx, instance in enumerate(answer.get("text_instances", []) or []):
            if not isinstance(instance, dict):
                continue
            doc_id = str(instance.get("doc_id", "") or "").strip()
            seen: set[tuple[str, str]] = set()
            evidence: list[base.Evidence] = []
            add_unit_evidence(
                evidence,
                seen,
                text=instance.get("text"),
                source="text_instance",
                weight=9.0,
                doc_id=doc_id,
            )
            add_unit_evidence(evidence, seen, text=answer_value, source="answer_text", weight=5.0, doc_id=doc_id)
            append_unit(
                unit_type="text_instance",
                doc_id=doc_id,
                unit_evidence=evidence,
                metadata={"answer_index": answer_idx, "instance_index": instance_idx},
            )

        for instance_idx, instance in enumerate(answer.get("image_instances", []) or []):
            if not isinstance(instance, dict):
                continue
            doc_id = str(instance.get("doc_id", "") or "").strip()
            image_row = images_by_id.get(doc_id, {})
            seen = set()
            evidence = []
            add_unit_evidence(evidence, seen, text=image_row.get("title"), source="image_title", weight=7.0, doc_id=doc_id)
            add_unit_evidence(
                evidence,
                seen,
                text=base.doc_title_from_map(doc_id, image_row, id_map),
                source="image_doc_title",
                weight=4.0,
                doc_id=doc_id,
            )
            add_unit_evidence(evidence, seen, text=answer_value, source="answer_text", weight=5.0, doc_id=doc_id)
            append_unit(
                unit_type="image_instance",
                doc_id=doc_id,
                unit_evidence=evidence,
                metadata={"answer_index": answer_idx, "instance_index": instance_idx},
            )

        for table_pair_idx, pair in enumerate(answer.get("table_indices", []) or []):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2 or not table_id or not table:
                continue
            row_idx, col_idx = int(pair[0]), int(pair[1])
            seen = set()
            evidence = []
            add_unit_evidence(evidence, seen, text=table.get("title"), source="table_title", weight=4.0, doc_id=table_id)
            add_unit_evidence(
                evidence,
                seen,
                text=base.table_cell_text(table, row_idx, col_idx),
                source="table_answer_cell",
                weight=10.0,
                doc_id=table_id,
            )
            add_unit_evidence(evidence, seen, text=answer_value, source="answer_text", weight=5.0, doc_id=table_id)
            for cell in base.table_row_cells(table, row_idx):
                add_unit_evidence(
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
                    linked_doc_id = url_to_id.get(base.normalize_url(str(link.get("url", ""))), "")
                    if linked_doc_id != table_id:
                        continue
                    add_unit_evidence(
                        evidence,
                        seen,
                        text=link.get("text"),
                        source="table_row_link_text",
                        weight=1.5,
                        doc_id=table_id,
                    )
                    add_unit_evidence(
                        evidence,
                        seen,
                        text=link.get("wiki_title"),
                        source="table_row_link_title",
                        weight=1.5,
                        doc_id=table_id,
                    )
            append_unit(
                unit_type="table_cell",
                doc_id=table_id,
                unit_evidence=evidence,
                metadata={"answer_index": answer_idx, "row_idx": row_idx, "col_idx": col_idx, "pair_index": table_pair_idx},
            )

    return units


def score_unit_pages(
    unit: EvidenceUnit,
    pages_by_doc: dict[str, list[dict[str, Any]]],
    *,
    min_score: float,
    high_confidence_score: float,
    min_token_overlap: float,
    require_exact_or_high_confidence_fuzzy: bool,
) -> list[UnitPageMatch]:
    matches: list[UnitPageMatch] = []
    for page in pages_by_doc.get(unit.doc_id, []):
        score = 0.0
        exact_matches: list[dict[str, Any]] = []
        fuzzy_matches: list[dict[str, Any]] = []
        for item in unit.evidence:
            norm_phrase = base.normalize_text(item.text)
            mode, overlap = base.phrase_match(norm_phrase, page, min_token_overlap=min_token_overlap)
            if mode == "exact":
                score += float(item.weight)
                exact_matches.append({"source": item.source, "text": item.text, "weight": float(item.weight)})
            elif mode == "fuzzy":
                gain = float(item.weight) * 0.45 * float(overlap)
                score += gain
                fuzzy_matches.append(
                    {"source": item.source, "text": item.text, "overlap": float(overlap), "weight": float(gain)}
                )
        if score < float(min_score):
            continue
        if require_exact_or_high_confidence_fuzzy and not exact_matches and score < float(high_confidence_score):
            continue
        matches.append(
            UnitPageMatch(
                unit=unit,
                page_uid=str(page["page_uid"]),
                doc_id=unit.doc_id,
                page_idx=int(page["page_idx"]),
                score=float(score),
                exact_matches=exact_matches,
                fuzzy_matches=fuzzy_matches,
            )
        )
    matches.sort(key=lambda item: (-item.score, item.page_idx, item.page_uid))
    return matches


def adaptive_doc_caps(units: list[EvidenceUnit], *, max_pages_per_doc: int) -> dict[str, int]:
    counts = Counter(unit.doc_id for unit in units)
    cap = max(1, int(max_pages_per_doc))
    return {doc_id: min(cap, max(1, count)) for doc_id, count in counts.items()}


def merge_unit_matches(
    unit_matches: list[UnitPageMatch],
    *,
    doc_caps: dict[str, int],
    top_pages_per_qid: int,
) -> list[PageLabel]:
    labels_by_uid: dict[str, PageLabel] = {}
    for item in unit_matches:
        label = labels_by_uid.setdefault(
            item.page_uid,
            PageLabel(page_uid=item.page_uid, doc_id=item.doc_id, page_idx=item.page_idx),
        )
        label.score += float(item.score)
        label.unit_matches.append(item)

    candidates = sorted(
        labels_by_uid.values(),
        key=lambda item: (-item.score, -len(item.unit_matches), item.doc_id, item.page_idx),
    )
    selected: list[PageLabel] = []
    per_doc_count: Counter[str] = Counter()
    for item in candidates:
        if per_doc_count[item.doc_id] >= int(doc_caps.get(item.doc_id, 1)):
            continue
        selected.append(item)
        per_doc_count[item.doc_id] += 1
        if int(top_pages_per_qid) > 0 and len(selected) >= int(top_pages_per_qid):
            break
    return selected


def confidence(score: float, *, high_confidence_score: float, min_score: float) -> str:
    if score >= float(high_confidence_score):
        return "high"
    if score >= float(min_score):
        return "medium"
    if score > 0:
        return "low"
    return "none"


def page_label_record(
    label: PageLabel,
    *,
    high_confidence_score: float,
    min_score: float,
) -> dict[str, Any]:
    return {
        "page_uid": label.page_uid,
        "doc_id": label.doc_id,
        "page_idx": int(label.page_idx),
        "score": round(float(label.score), 6),
        "confidence": confidence(label.score, high_confidence_score=high_confidence_score, min_score=min_score),
        "evidence_units": [
            {
                "unit_id": item.unit.unit_id,
                "unit_type": item.unit.unit_type,
                "doc_id": item.unit.doc_id,
                "score": round(float(item.score), 6),
                "metadata": item.unit.metadata,
                "exact_matches": item.exact_matches[:20],
                "fuzzy_matches": item.fuzzy_matches[:20],
            }
            for item in label.unit_matches
        ],
    }


def augmented_gold_row(row: dict[str, Any], selected: list[PageLabel]) -> dict[str, Any]:
    out = json.loads(json.dumps(row))
    metadata = out.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        out["metadata"] = metadata
    uids = [item.page_uid for item in selected]
    metadata["pseudo_gold_page_uids"] = uids
    metadata["gold_page_uids"] = uids
    metadata["pseudo_gold_page_label_source"] = "mmqa_evidence_unit_aware_page_text"
    metadata["pseudo_gold_page_label_scores"] = {item.page_uid: round(float(item.score), 6) for item in selected}
    return out


def status_for(selected: list[PageLabel], units: list[EvidenceUnit], mapped_unit_ids: set[str], missing_docs: list[str]) -> str:
    if selected and len(mapped_unit_ids) == len(units):
        return "matched_all_units"
    if selected:
        return "matched_partial_units"
    if missing_docs:
        return "missing_supporting_doc_page_text"
    if units:
        return "no_unit_page_match"
    return "no_document_grounded_evidence_units"


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Evidence-Unit-Aware Pseudo Page Label Summary",
        "",
        f"- qid_count: `{summary['qid_count']}`",
        f"- matched_qid_count: `{summary['matched_qid_count']}`",
        f"- matched_qid_fraction: `{summary['matched_qid_fraction']}`",
        f"- pseudo_page_label_count: `{summary['pseudo_page_label_count']}`",
        f"- mean_labels_per_matched_qid: `{summary['mean_labels_per_matched_qid']}`",
        f"- evidence_unit_count: `{summary['evidence_unit_count']}`",
        f"- mapped_evidence_unit_count: `{summary['mapped_evidence_unit_count']}`",
        f"- mapped_evidence_unit_fraction: `{summary['mapped_evidence_unit_fraction']}`",
        "",
        "## Status Counts",
        "",
    ]
    for key, value in summary["status_counts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Confidence Counts", ""])
    for key, value in summary["confidence_counts"].items():
        lines.append(f"- `{key}`: `{value}`")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    gold_rows = base.load_jsonl(Path(args.gold))
    pages_by_doc = base.load_page_texts(Path(args.doc_pages_jsonl))
    texts_by_id = base.load_by_id(args.mmqa_texts_jsonl)
    tables_by_id = base.load_by_id(args.mmqa_tables_jsonl)
    images_by_id = base.load_by_id(args.mmqa_images_jsonl)
    id_map = base.load_id_map(args.id_url_mapping_jsonl)
    url_to_id = base.load_url_to_id(args.id_url_mapping_jsonl)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.output_summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    augmented_path = Path(args.output_augmented_gold_jsonl) if args.output_augmented_gold_jsonl else None
    if augmented_path:
        augmented_path.parent.mkdir(parents=True, exist_ok=True)

    status_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    unit_type_counts: Counter[str] = Counter()
    matched_qtype_counts: Counter[str] = Counter()
    qtype_counts: Counter[str] = Counter()
    label_counts: list[int] = []
    score_values: list[float] = []
    evidence_unit_count = 0
    mapped_evidence_unit_count = 0
    matched_qids = 0
    missing_doc_count = 0

    with output_path.open("w", encoding="utf-8") as out_handle:
        aug_handle = augmented_path.open("w", encoding="utf-8") if augmented_path else None
        try:
            for row in gold_rows:
                qid = str(row.get("qid", "")).strip()
                metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
                question_type = str(metadata.get("type", "UNKNOWN")).strip() or "UNKNOWN"
                qtype_counts[question_type] += 1
                gold_docs = base.supporting_doc_ids(row)
                missing_docs = [doc_id for doc_id in gold_docs if doc_id not in pages_by_doc]
                missing_doc_count += len(missing_docs)

                units = evidence_units_for_row(
                    row,
                    tables_by_id=tables_by_id,
                    texts_by_id=texts_by_id,
                    images_by_id=images_by_id,
                    id_map=id_map,
                    url_to_id=url_to_id,
                    include_question_context_signals=bool(args.include_question_context_signals),
                )
                unit_type_counts.update(unit.unit_type for unit in units)
                evidence_unit_count += len(units)

                best_unit_matches: list[UnitPageMatch] = []
                mapped_unit_ids: set[str] = set()
                for unit in units:
                    unit_matches = score_unit_pages(
                        unit,
                        pages_by_doc,
                        min_score=float(args.min_score),
                        high_confidence_score=float(args.high_confidence_score),
                        min_token_overlap=float(args.min_token_overlap),
                        require_exact_or_high_confidence_fuzzy=bool(args.require_exact_or_high_confidence_fuzzy),
                    )
                    if unit_matches:
                        mapped_unit_ids.add(unit.unit_id)
                        best_unit_matches.append(unit_matches[0])
                mapped_evidence_unit_count += len(mapped_unit_ids)

                if args.require_all_units_mapped and len(mapped_unit_ids) != len(units):
                    selected: list[PageLabel] = []
                else:
                    selected = merge_unit_matches(
                        best_unit_matches,
                        doc_caps=adaptive_doc_caps(units, max_pages_per_doc=int(args.max_pages_per_doc)),
                        top_pages_per_qid=int(args.top_pages_per_qid),
                    )

                status = status_for(selected, units, mapped_unit_ids, missing_docs)
                status_counts[status] += 1
                label_counts.append(len(selected))
                if selected:
                    matched_qids += 1
                    matched_qtype_counts[question_type] += 1
                    for label in selected:
                        score_values.append(float(label.score))
                        confidence_counts[
                            confidence(
                                label.score,
                                high_confidence_score=float(args.high_confidence_score),
                                min_score=float(args.min_score),
                            )
                        ] += 1

                output_row = {
                    "qid": qid,
                    "question": row.get("question", ""),
                    "question_type": question_type,
                    "gold_doc_ids": gold_docs,
                    "status": status,
                    "evidence_unit_count": len(units),
                    "mapped_evidence_unit_count": len(mapped_unit_ids),
                    "adaptive_doc_caps": adaptive_doc_caps(units, max_pages_per_doc=int(args.max_pages_per_doc)),
                    "pseudo_gold_page_uids": [item.page_uid for item in selected],
                    "pseudo_gold_pages": [
                        page_label_record(
                            item,
                            high_confidence_score=float(args.high_confidence_score),
                            min_score=float(args.min_score),
                        )
                        for item in selected
                    ],
                    "missing_page_text_doc_ids": missing_docs,
                    "config": {
                        "min_score": float(args.min_score),
                        "high_confidence_score": float(args.high_confidence_score),
                        "top_pages_per_qid": int(args.top_pages_per_qid),
                        "max_pages_per_doc": int(args.max_pages_per_doc),
                        "min_token_overlap": float(args.min_token_overlap),
                        "require_exact_or_high_confidence_fuzzy": bool(args.require_exact_or_high_confidence_fuzzy),
                        "include_question_context_signals": bool(args.include_question_context_signals),
                        "require_all_units_mapped": bool(args.require_all_units_mapped),
                    },
                }
                out_handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                if aug_handle is not None:
                    aug_handle.write(json.dumps(augmented_gold_row(row, selected), ensure_ascii=False) + "\n")
        finally:
            if aug_handle is not None:
                aug_handle.close()

    summary = {
        "gold": str(args.gold),
        "doc_pages_jsonl": str(args.doc_pages_jsonl),
        "qid_count": len(gold_rows),
        "matched_qid_count": int(matched_qids),
        "matched_qid_fraction": round(float(matched_qids) / len(gold_rows), 6) if gold_rows else None,
        "pseudo_page_label_count": int(sum(label_counts)),
        "mean_labels_per_qid": round(float(fmean(label_counts)), 6) if label_counts else None,
        "mean_labels_per_matched_qid": round(float(sum(label_counts)) / matched_qids, 6) if matched_qids else None,
        "mean_selected_score": round(float(fmean(score_values)), 6) if score_values else None,
        "evidence_unit_count": int(evidence_unit_count),
        "mapped_evidence_unit_count": int(mapped_evidence_unit_count),
        "mapped_evidence_unit_fraction": (
            round(float(mapped_evidence_unit_count) / evidence_unit_count, 6) if evidence_unit_count else None
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "unit_type_counts": dict(sorted(unit_type_counts.items())),
        "question_type_counts": dict(sorted(qtype_counts.items())),
        "matched_by_question_type": dict(sorted(matched_qtype_counts.items())),
        "missing_page_text_doc_ref_count": int(missing_doc_count),
        "min_score": float(args.min_score),
        "high_confidence_score": float(args.high_confidence_score),
        "top_pages_per_qid": int(args.top_pages_per_qid),
        "max_pages_per_doc": int(args.max_pages_per_doc),
        "min_token_overlap": float(args.min_token_overlap),
        "require_exact_or_high_confidence_fuzzy": bool(args.require_exact_or_high_confidence_fuzzy),
        "include_question_context_signals": bool(args.include_question_context_signals),
        "require_all_units_mapped": bool(args.require_all_units_mapped),
        "output_jsonl": str(output_path),
        "output_augmented_gold_jsonl": str(augmented_path) if augmented_path else "",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_summary_md(summary_path, summary)

    print(f"saved_pseudo_labels={output_path}")
    if augmented_path:
        print(f"saved_augmented_gold={augmented_path}")
    print(f"saved_summary={summary_path}")
    print(f"qid_count={summary['qid_count']}")
    print(f"matched_qid_count={summary['matched_qid_count']}")
    print(f"matched_qid_fraction={summary['matched_qid_fraction']}")
    print(f"pseudo_page_label_count={summary['pseudo_page_label_count']}")
    print(f"mapped_evidence_unit_fraction={summary['mapped_evidence_unit_fraction']}")


if __name__ == "__main__":
    main()
