#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


PRIMARY_DIRECT_SOURCES = {"text_instance", "table_answer_cell", "image_title"}
CORROBORATING_SOURCES = {
    "answer_text",
    "image_doc_title",
    "table_title",
    "supporting_doc_title",
}
WEAK_CONTEXT_SOURCES = {
    "table_row_cell",
    "table_row_link_text",
    "table_row_link_title",
    "answer_entity",
    "question_entity",
    "pseudo_question_slot",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit every evidence-unit-aware MMQA pseudo-page label and record "
            "the evidence sources, match modes, support-document coverage, and "
            "quality-risk flags that produced it."
        )
    )
    parser.add_argument("--labels-jsonl", required=True)
    parser.add_argument(
        "--current-labels-jsonl",
        default="",
        help="Optional original strict pseudo-label JSONL for page-set overlap comparison.",
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--max-example-qids", type=int, default=20)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_by_qid(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("qid", "")).strip(): row
        for row in load_jsonl(path)
        if str(row.get("qid", "")).strip()
    }


def page_uids(row: dict[str, Any] | None) -> list[str]:
    if not row:
        return []
    values = row.get("pseudo_gold_page_uids") or row.get("gold_page_uids") or []
    return [str(value) for value in values if str(value).strip()]


def flatten_page_matches(page: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    exact: list[dict[str, Any]] = []
    fuzzy: list[dict[str, Any]] = []
    unit_types: list[str] = []
    for unit in page.get("evidence_units", []) or []:
        if not isinstance(unit, dict):
            continue
        unit_type = str(unit.get("unit_type", "")).strip()
        if unit_type:
            unit_types.append(unit_type)
        exact.extend(item for item in unit.get("exact_matches", []) or [] if isinstance(item, dict))
        fuzzy.extend(item for item in unit.get("fuzzy_matches", []) or [] if isinstance(item, dict))
    return exact, fuzzy, unit_types


def source_set(matches: list[dict[str, Any]]) -> set[str]:
    return {str(item.get("source", "")).strip() for item in matches if str(item.get("source", "")).strip()}


def page_audit(page: dict[str, Any]) -> dict[str, Any]:
    exact, fuzzy, unit_types = flatten_page_matches(page)
    exact_sources = source_set(exact)
    fuzzy_sources = source_set(fuzzy)
    all_sources = exact_sources | fuzzy_sources
    primary_exact = bool(exact_sources & PRIMARY_DIRECT_SOURCES)
    primary_any = bool(all_sources & PRIMARY_DIRECT_SOURCES)
    any_exact = bool(exact)
    fuzzy_only = bool(fuzzy) and not any_exact
    contextual_only = bool(all_sources) and not primary_any
    weak_only = bool(all_sources) and all_sources.issubset(WEAK_CONTEXT_SOURCES)
    title_only = bool(all_sources) and all_sources.issubset(
        {"image_doc_title", "table_title", "supporting_doc_title", "image_title"}
    )
    return {
        "page_uid": str(page.get("page_uid", "")),
        "doc_id": str(page.get("doc_id", "")),
        "page_idx": page.get("page_idx"),
        "score": page.get("score"),
        "confidence": str(page.get("confidence", "")),
        "unit_types": sorted(set(unit_types)),
        "unit_count": len(page.get("evidence_units", []) or []),
        "exact_match_count": len(exact),
        "fuzzy_match_count": len(fuzzy),
        "exact_sources": sorted(exact_sources),
        "fuzzy_sources": sorted(fuzzy_sources),
        "all_sources": sorted(all_sources),
        "has_primary_direct_exact": primary_exact,
        "has_primary_direct_any": primary_any,
        "has_any_exact": any_exact,
        "fuzzy_only": fuzzy_only,
        "contextual_only": contextual_only,
        "weak_only": weak_only,
        "title_only": title_only,
        "evidence_units": page.get("evidence_units", []) or [],
    }


def audit_tier(
    *,
    label_count: int,
    status: str,
    page_audits: list[dict[str, Any]],
) -> str:
    if label_count == 0:
        return "unlabeled"
    all_units_mapped = status == "matched_all_units"
    if all_units_mapped and all(page["has_primary_direct_exact"] for page in page_audits):
        return "strong_direct"
    if all_units_mapped and all(page["has_any_exact"] for page in page_audits):
        return "strong_corroborated"
    if any(page["fuzzy_only"] or page["contextual_only"] for page in page_audits):
        return "review_recommended"
    return "supported_partial"


def summarize_row(row: dict[str, Any], current_row: dict[str, Any] | None) -> dict[str, Any]:
    qid = str(row.get("qid", "")).strip()
    selected_pages = [page for page in row.get("pseudo_gold_pages", []) or [] if isinstance(page, dict)]
    audits = [page_audit(page) for page in selected_pages]
    selected_uids = [page["page_uid"] for page in audits]
    current_uids = page_uids(current_row)
    gold_docs = {str(value) for value in row.get("gold_doc_ids", []) or [] if str(value).strip()}
    selected_docs = {page["doc_id"] for page in audits if page["doc_id"]}
    covered_docs = gold_docs & selected_docs
    status = str(row.get("status", ""))
    label_count = len(selected_uids)
    tier = audit_tier(label_count=label_count, status=status, page_audits=audits)

    risk_flags: list[str] = []
    if label_count == 0:
        risk_flags.append("unlabeled")
    if status == "matched_partial_units":
        risk_flags.append("partial_evidence_units")
    if gold_docs and covered_docs != gold_docs:
        risk_flags.append("support_docs_not_fully_covered")
    if any(page["fuzzy_only"] for page in audits):
        risk_flags.append("fuzzy_only_page")
    if any(page["contextual_only"] for page in audits):
        risk_flags.append("contextual_only_page")
    if any(page["weak_only"] for page in audits):
        risk_flags.append("weak_only_page")
    if any(page["title_only"] for page in audits):
        risk_flags.append("title_only_page")
    doc_counts = Counter(page["doc_id"] for page in audits if page["doc_id"])
    if any(count > 1 for count in doc_counts.values()):
        risk_flags.append("multi_page_same_doc")
    if current_row is not None and set(selected_uids) != set(current_uids):
        risk_flags.append("changed_vs_current_strict")

    return {
        "qid": qid,
        "question": str(row.get("question", "")),
        "question_type": str(row.get("question_type", "")),
        "status": status,
        "audit_tier": tier,
        "risk_flags": risk_flags,
        "support_doc_count": len(gold_docs),
        "covered_support_doc_count": len(covered_docs),
        "support_doc_coverage": (len(covered_docs) / len(gold_docs)) if gold_docs else None,
        "evidence_unit_count": int(row.get("evidence_unit_count", 0) or 0),
        "mapped_evidence_unit_count": int(row.get("mapped_evidence_unit_count", 0) or 0),
        "label_count": label_count,
        "high_label_count": sum(page["confidence"] == "high" for page in audits),
        "medium_label_count": sum(page["confidence"] == "medium" for page in audits),
        "primary_direct_exact_page_count": sum(page["has_primary_direct_exact"] for page in audits),
        "any_exact_page_count": sum(page["has_any_exact"] for page in audits),
        "fuzzy_only_page_count": sum(page["fuzzy_only"] for page in audits),
        "contextual_only_page_count": sum(page["contextual_only"] for page in audits),
        "selected_page_uids": selected_uids,
        "current_strict_page_uids": current_uids,
        "page_overlap_with_current": len(set(selected_uids) & set(current_uids)),
        "page_union_with_current": len(set(selected_uids) | set(current_uids)),
        "page_audits": audits,
        "review_decision": "",
        "review_notes": "",
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def csv_value(value: Any) -> Any:
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "qid",
        "question_type",
        "status",
        "audit_tier",
        "risk_flags",
        "support_doc_count",
        "covered_support_doc_count",
        "support_doc_coverage",
        "evidence_unit_count",
        "mapped_evidence_unit_count",
        "label_count",
        "high_label_count",
        "medium_label_count",
        "primary_direct_exact_page_count",
        "any_exact_page_count",
        "fuzzy_only_page_count",
        "contextual_only_page_count",
        "selected_page_uids",
        "current_strict_page_uids",
        "page_overlap_with_current",
        "page_union_with_current",
        "question",
        "review_decision",
        "review_notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})


def write_md(path: Path, rows: list[dict[str, Any]], max_examples: int) -> None:
    tier_counts = Counter(row["audit_tier"] for row in rows)
    risk_counts = Counter(flag for row in rows for flag in row["risk_flags"])
    source_counts: Counter[str] = Counter()
    exact_source_counts: Counter[str] = Counter()
    for row in rows:
        for page in row["page_audits"]:
            source_counts.update(page["all_sources"])
            exact_source_counts.update(page["exact_sources"])

    lines = [
        "# Evidence-Unit-Aware Pseudo-Page Provenance Audit",
        "",
        f"- qids: `{len(rows)}`",
        f"- labeled_qids: `{sum(row['label_count'] > 0 for row in rows)}`",
        f"- labels: `{sum(row['label_count'] for row in rows)}`",
        "",
        "## Audit Tiers",
        "",
        "| tier | qids |",
        "| --- | ---: |",
    ]
    for tier, count in tier_counts.most_common():
        lines.append(f"| {tier} | {count} |")

    lines.extend(["", "## Risk Flags", "", "| flag | qids |", "| --- | ---: |"])
    for flag, count in risk_counts.most_common():
        lines.append(f"| {flag} | {count} |")

    lines.extend(["", "## Evidence Sources", "", "| source | pages using source | pages with exact source |", "| --- | ---: | ---: |"])
    for source, count in source_counts.most_common():
        lines.append(f"| {source} | {count} | {exact_source_counts.get(source, 0)} |")

    review_rows = [row for row in rows if row["audit_tier"] == "review_recommended" or "partial_evidence_units" in row["risk_flags"]]
    lines.extend(
        [
            "",
            "## Review-Priority QIDs",
            "",
            "| qid | type | tier | labels | mapped units | total units | risk flags |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in review_rows[: max(0, max_examples)]:
        lines.append(
            f"| {row['qid']} | {row['question_type']} | {row['audit_tier']} | "
            f"{row['label_count']} | {row['mapped_evidence_unit_count']} | "
            f"{row['evidence_unit_count']} | {', '.join(row['risk_flags'])} |"
        )

    lines.extend(
        [
            "",
            "## Per-QID Index",
            "",
            "| qid | type | status | tier | labels | support coverage | risk flags |",
            "| --- | --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        coverage = row["support_doc_coverage"]
        coverage_text = "" if coverage is None else f"{coverage:.3f}"
        lines.append(
            f"| {row['qid']} | {row['question_type']} | {row['status']} | "
            f"{row['audit_tier']} | {row['label_count']} | {coverage_text} | "
            f"{', '.join(row['risk_flags'])} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    labels = load_jsonl(Path(args.labels_jsonl))
    current = load_by_qid(Path(args.current_labels_jsonl)) if args.current_labels_jsonl else {}
    rows = [summarize_row(row, current.get(str(row.get("qid", "")).strip())) for row in labels]

    write_jsonl(Path(args.output_jsonl), rows)
    write_csv(Path(args.output_csv), rows)
    write_md(Path(args.output_md), rows, args.max_example_qids)

    print(f"saved_output_jsonl={args.output_jsonl}")
    print(f"saved_output_csv={args.output_csv}")
    print(f"saved_output_md={args.output_md}")
    print(f"qid_count={len(rows)}")
    print(f"labeled_qid_count={sum(row['label_count'] > 0 for row in rows)}")
    print(f"label_count={sum(row['label_count'] for row in rows)}")
    print(f"audit_tier_counts={dict(Counter(row['audit_tier'] for row in rows))}")


if __name__ == "__main__":
    main()
