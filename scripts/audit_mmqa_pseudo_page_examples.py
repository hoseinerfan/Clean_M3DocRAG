#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_mmqa_pseudo_page_labels as ppl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a qualitative audit report comparing MMQA evidence metadata "
            "with derived strict/loose pseudo-page labels for sample qids."
        )
    )
    parser.add_argument("--gold", required=True, help="Original MMQA_train/dev.jsonl")
    parser.add_argument("--strict-augmented-gold", required=True)
    parser.add_argument("--loose-augmented-gold", default="")
    parser.add_argument("--doc-pages-jsonl", required=True)
    parser.add_argument("--mmqa-texts-jsonl", required=True)
    parser.add_argument("--mmqa-tables-jsonl", required=True)
    parser.add_argument("--mmqa-images-jsonl", required=True)
    parser.add_argument("--id-url-mapping-jsonl", required=True)
    parser.add_argument("--qid", action="append", default=[], help="Specific qid to include.")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--min-score", type=float, default=8.0)
    parser.add_argument("--top-pages-per-doc", type=int, default=1)
    parser.add_argument("--top-pages-per-qid", type=int, default=4)
    parser.add_argument("--min-token-overlap", type=float, default=0.72)
    parser.add_argument("--max-evidence", type=int, default=12)
    parser.add_argument("--max-matches-per-page", type=int, default=8)
    parser.add_argument("--max-top-pages", type=int, default=5)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-jsonl", default="")
    return parser.parse_args()


def load_jsonl_by_qid(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in ppl.load_jsonl(path):
        qid = str(row.get("qid") or row.get("id") or "").strip()
        if qid:
            rows[qid] = row
    return rows


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}


def gold_page_uids(row: dict[str, Any] | None) -> list[str]:
    if not row:
        return []
    meta = metadata(row)
    values = meta.get("gold_page_uids") or meta.get("pseudo_gold_page_uids") or row.get("gold_page_uids") or []
    return [str(value).strip() for value in values if str(value).strip()]


def page_doc(page_uid: str) -> str:
    return str(page_uid).rsplit("_page", 1)[0] if "_page" in str(page_uid) else str(page_uid)


def doc_title(doc_id: str, *, texts: dict[str, dict[str, Any]], tables: dict[str, dict[str, Any]], images: dict[str, dict[str, Any]], id_map: dict[str, dict[str, Any]]) -> str:
    row = texts.get(doc_id) or tables.get(doc_id) or images.get(doc_id) or {}
    return ppl.doc_title_from_map(doc_id, row, id_map)


def doc_url(doc_id: str, id_map: dict[str, dict[str, Any]]) -> str:
    return str(id_map.get(doc_id, {}).get("url", "") or "")


def answer_summary(row: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ans in row.get("answers", []) or []:
        if not isinstance(ans, dict):
            continue
        out.append(
            {
                "answer": ans.get("answer"),
                "type": ans.get("type"),
                "modality": ans.get("modality"),
                "text_instances": ans.get("text_instances", [])[:3],
                "table_indices": ans.get("table_indices", [])[:6],
                "image_instances": ans.get("image_instances", [])[:3],
            }
        )
    return out


def support_rows(
    row: dict[str, Any],
    *,
    texts: dict[str, dict[str, Any]],
    tables: dict[str, dict[str, Any]],
    images: dict[str, dict[str, Any]],
    id_map: dict[str, dict[str, Any]],
    strict_pages: list[str],
    loose_pages: list[str],
) -> list[dict[str, Any]]:
    strict_by_doc: dict[str, list[str]] = defaultdict(list)
    loose_by_doc: dict[str, list[str]] = defaultdict(list)
    for uid in strict_pages:
        strict_by_doc[page_doc(uid)].append(uid)
    for uid in loose_pages:
        loose_by_doc[page_doc(uid)].append(uid)

    rows: list[dict[str, Any]] = []
    for ctx in row.get("supporting_context", []) or []:
        if not isinstance(ctx, dict):
            continue
        doc_id = str(ctx.get("doc_id", "")).strip()
        if not doc_id:
            continue
        rows.append(
            {
                "doc_id": doc_id,
                "doc_part": ctx.get("doc_part", ""),
                "title": doc_title(doc_id, texts=texts, tables=tables, images=images, id_map=id_map),
                "url": doc_url(doc_id, id_map),
                "strict_pages": strict_by_doc.get(doc_id, []),
                "loose_pages": loose_by_doc.get(doc_id, []),
            }
        )
    return rows


def table_context(row: dict[str, Any], tables: dict[str, dict[str, Any]]) -> dict[str, Any]:
    meta = metadata(row)
    table_id = str(meta.get("table_id", "") or "").strip()
    if not table_id or table_id not in tables:
        return {}
    table = tables[table_id]
    cells: list[dict[str, Any]] = []
    table_rows = table.get("table", {}).get("table_rows", []) or []
    for ans in row.get("answers", []) or []:
        if not isinstance(ans, dict):
            continue
        for pair in ans.get("table_indices", []) or []:
            if not isinstance(pair, list | tuple) or len(pair) != 2:
                continue
            r, c = int(pair[0]), int(pair[1])
            cell_text = ""
            row_texts: list[str] = []
            if 0 <= r < len(table_rows):
                row_cells = table_rows[r]
                row_texts = [str(cell.get("text", "")) for cell in row_cells if isinstance(cell, dict)]
                if 0 <= c < len(row_cells) and isinstance(row_cells[c], dict):
                    cell_text = str(row_cells[c].get("text", ""))
            cells.append({"row": r, "col": c, "cell": cell_text, "row_cells": row_texts})
    return {"table_id": table_id, "title": table.get("title", ""), "answer_cells": cells}


def page_text_snippet(page_uid: str, pages_by_doc: dict[str, list[dict[str, Any]]], phrases: list[str], *, max_len: int = 260) -> str:
    doc_id = page_doc(page_uid)
    page = next((item for item in pages_by_doc.get(doc_id, []) if str(item.get("page_uid")) == page_uid), None)
    if not page:
        return ""
    text = str(page.get("text", "") or "")
    if not text:
        return ""
    lower = text.lower()
    for phrase in phrases:
        norm = str(phrase or "").strip().lower()
        if not norm:
            continue
        idx = lower.find(norm)
        if idx >= 0:
            start = max(0, idx - 80)
            end = min(len(text), idx + len(norm) + 180)
            return text[start:end].replace("\n", " ").strip()
    return text[:max_len].replace("\n", " ").strip()


def page_audit_record(
    uid: str,
    scored_by_uid: dict[str, ppl.PageScore],
    pages_by_doc: dict[str, list[dict[str, Any]]],
    *,
    max_matches: int,
) -> dict[str, Any]:
    score = scored_by_uid.get(uid)
    if not score:
        return {"page_uid": uid, "score": None, "exact_matches": [], "fuzzy_matches": [], "snippet": ""}
    exact = score.exact_matches[:max_matches]
    fuzzy = score.fuzzy_matches[: max(0, max_matches - len(exact))]
    phrases = [str(item.get("text", "")) for item in exact + fuzzy]
    return {
        "page_uid": uid,
        "score": round(float(score.score), 3),
        "confidence": ppl.confidence(float(score.score)),
        "exact_matches": exact,
        "fuzzy_matches": fuzzy,
        "snippet": page_text_snippet(uid, pages_by_doc, phrases),
    }


def evidence_rows(evidence: list[ppl.Evidence], *, max_evidence: int) -> list[dict[str, Any]]:
    return [
        {
            "source": item.source,
            "doc_id": item.doc_id,
            "weight": item.weight,
            "text": item.text,
        }
        for item in evidence[:max_evidence]
    ]


def choose_qids(
    gold_rows: dict[str, dict[str, Any]],
    strict_rows: dict[str, dict[str, Any]],
    loose_rows: dict[str, dict[str, Any]],
    explicit_qids: list[str],
    limit: int,
) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()

    def add(qid: str) -> None:
        if qid and qid in gold_rows and qid not in seen and len(selected) < limit:
            seen.add(qid)
            selected.append(qid)

    for qid in explicit_qids:
        add(qid)

    buckets: dict[str, list[str]] = defaultdict(list)
    for qid, row in gold_rows.items():
        answers = answer_summary(row)
        modalities = {str(item.get("modality", "")).lower() for item in answers}
        support_count = len(ppl.supporting_doc_ids(row))
        strict_pages = gold_page_uids(strict_rows.get(qid))
        loose_pages = gold_page_uids(loose_rows.get(qid)) if loose_rows else []
        qtype = str(metadata(row).get("type", ""))

        if strict_pages and support_count >= 2:
            buckets["multi_doc_matched"].append(qid)
        if strict_pages and support_count == 1:
            buckets["single_doc_matched"].append(qid)
        if strict_pages and "text" in modalities:
            buckets["text_answer"].append(qid)
        if strict_pages and "table" in modalities:
            buckets["table_answer"].append(qid)
        if strict_pages and ("image" in modalities or "Image" in qtype):
            buckets["image_related"].append(qid)
        if not strict_pages:
            buckets["no_strict_label"].append(qid)
        if loose_pages and not strict_pages:
            buckets["loose_only"].append(qid)

    order = [
        "multi_doc_matched",
        "single_doc_matched",
        "text_answer",
        "table_answer",
        "image_related",
        "loose_only",
        "no_strict_label",
    ]
    while len(selected) < limit:
        changed = False
        for bucket in order:
            if buckets[bucket]:
                add(buckets[bucket].pop(0))
                changed = True
                if len(selected) >= limit:
                    break
        if not changed:
            break
    return selected


def build_case(
    qid: str,
    *,
    gold_rows: dict[str, dict[str, Any]],
    strict_rows: dict[str, dict[str, Any]],
    loose_rows: dict[str, dict[str, Any]],
    pages_by_doc: dict[str, list[dict[str, Any]]],
    texts: dict[str, dict[str, Any]],
    tables: dict[str, dict[str, Any]],
    images: dict[str, dict[str, Any]],
    id_map: dict[str, dict[str, Any]],
    url_to_id: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    row = gold_rows[qid]
    strict_pages = gold_page_uids(strict_rows.get(qid))
    loose_pages = gold_page_uids(loose_rows.get(qid)) if loose_rows else []
    evidence, evidence_meta = ppl.evidence_for_row(
        row,
        tables_by_id=tables,
        texts_by_id=texts,
        images_by_id=images,
        id_map=id_map,
        url_to_id=url_to_id,
    )
    scored, missing_docs = ppl.score_pages(
        row,
        evidence,
        pages_by_doc,
        min_token_overlap=float(args.min_token_overlap),
    )
    scored_by_uid = {item.page_uid: item for item in scored}
    recomputed_selected = ppl.select_labels(
        scored,
        min_score=float(args.min_score),
        top_pages_per_doc=int(args.top_pages_per_doc),
        top_pages_per_qid=int(args.top_pages_per_qid),
    )

    support = support_rows(
        row,
        texts=texts,
        tables=tables,
        images=images,
        id_map=id_map,
        strict_pages=strict_pages,
        loose_pages=loose_pages,
    )
    strict_page_records = [
        page_audit_record(uid, scored_by_uid, pages_by_doc, max_matches=int(args.max_matches_per_page))
        for uid in strict_pages
    ]
    loose_page_records = [
        page_audit_record(uid, scored_by_uid, pages_by_doc, max_matches=int(args.max_matches_per_page))
        for uid in loose_pages
        if uid not in set(strict_pages)
    ]
    top_scored = [
        page_audit_record(item.page_uid, scored_by_uid, pages_by_doc, max_matches=4)
        for item in scored[: int(args.max_top_pages)]
    ]
    strict_doc_set = {page_doc(uid) for uid in strict_pages}
    support_doc_set = set(ppl.supporting_doc_ids(row))

    return {
        "qid": qid,
        "question": row.get("question", ""),
        "question_type": metadata(row).get("type") or row.get("question_type", ""),
        "answers": answer_summary(row),
        "supporting_docs": support,
        "table_context": table_context(row, tables),
        "evidence_meta": evidence_meta,
        "evidence": evidence_rows(evidence, max_evidence=int(args.max_evidence)),
        "strict_pages": strict_page_records,
        "loose_extra_pages": loose_page_records,
        "top_scored_pages": top_scored,
        "recomputed_strict_pages": [item.page_uid for item in recomputed_selected],
        "missing_page_text_doc_ids": missing_docs,
        "quality_checks": {
            "strict_label_count": len(strict_pages),
            "loose_label_count": len(loose_pages),
            "strict_labels_subset_of_support_docs": strict_doc_set.issubset(support_doc_set),
            "support_doc_count": len(support_doc_set),
            "strict_doc_count": len(strict_doc_set),
            "has_strict_label": bool(strict_pages),
            "has_loose_only_label": bool(loose_pages and not strict_pages),
        },
    }


def md_escape(value: Any) -> str:
    text = str(value if value is not None else "")
    return text.replace("|", "\\|").replace("\n", " ")


def write_table(lines: list[str], headers: list[str], rows: list[list[Any]]) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(md_escape(item) for item in row) + " |")


def write_markdown(cases: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# MMQA Evidence vs Pseudo-Page Label Examples",
        "",
        "This report compares original MMQA/M3DocVQA evidence metadata with derived pseudo-page labels.",
        "",
    ]
    for idx, case in enumerate(cases, start=1):
        lines.extend(
            [
                f"## {idx}. `{case['qid']}`",
                "",
                f"**Question:** {md_escape(case['question'])}",
                "",
                f"**Question type:** `{md_escape(case['question_type'])}`",
                "",
                "### Answers",
                "",
            ]
        )
        write_table(
            lines,
            ["answer", "modality", "text instances", "table indices", "image instances"],
            [
                [
                    item.get("answer", ""),
                    item.get("modality", ""),
                    len(item.get("text_instances", [])),
                    item.get("table_indices", []),
                    len(item.get("image_instances", [])),
                ]
                for item in case["answers"]
            ],
        )
        lines.extend(["", "### Supporting Documents", ""])
        write_table(
            lines,
            ["doc_id", "part", "title", "strict pages", "loose pages"],
            [
                [
                    row["doc_id"],
                    row["doc_part"],
                    row["title"],
                    ", ".join(row["strict_pages"]),
                    ", ".join(row["loose_pages"]),
                ]
                for row in case["supporting_docs"]
            ],
        )
        if case.get("table_context"):
            lines.extend(["", "### Table Context", ""])
            table = case["table_context"]
            lines.append(f"- table_id: `{table.get('table_id', '')}`")
            lines.append(f"- title: `{md_escape(table.get('title', ''))}`")
            if table.get("answer_cells"):
                write_table(
                    lines,
                    ["row", "col", "cell", "row cells"],
                    [
                        [
                            item.get("row", ""),
                            item.get("col", ""),
                            item.get("cell", ""),
                            "; ".join(item.get("row_cells", [])[:8]),
                        ]
                        for item in table["answer_cells"]
                    ],
                )
        lines.extend(["", "### Evidence Phrases Used for Matching", ""])
        write_table(
            lines,
            ["source", "doc_id", "weight", "text"],
            [
                [item["source"], item["doc_id"], item["weight"], item["text"]]
                for item in case["evidence"]
            ],
        )
        lines.extend(["", "### Strict Pseudo-Page Labels", ""])
        if case["strict_pages"]:
            write_table(
                lines,
                ["page_uid", "score", "confidence", "exact/fuzzy evidence", "snippet"],
                [
                    [
                        item["page_uid"],
                        item["score"],
                        item["confidence"],
                        "; ".join(
                            f"{m.get('source')}={m.get('text')}"
                            for m in (item.get("exact_matches", []) + item.get("fuzzy_matches", []))[:5]
                        ),
                        item.get("snippet", ""),
                    ]
                    for item in case["strict_pages"]
                ],
            )
        else:
            lines.append("_No strict pseudo-page label._")
        if case["loose_extra_pages"]:
            lines.extend(["", "### Loose-Only Extra Labels", ""])
            write_table(
                lines,
                ["page_uid", "score", "confidence", "evidence", "snippet"],
                [
                    [
                        item["page_uid"],
                        item["score"],
                        item["confidence"],
                        "; ".join(
                            f"{m.get('source')}={m.get('text')}"
                            for m in (item.get("exact_matches", []) + item.get("fuzzy_matches", []))[:5]
                        ),
                        item.get("snippet", ""),
                    ]
                    for item in case["loose_extra_pages"]
                ],
            )
        lines.extend(["", "### Top Scored Candidate Pages", ""])
        write_table(
            lines,
            ["page_uid", "score", "confidence", "evidence"],
            [
                [
                    item["page_uid"],
                    item["score"],
                    item["confidence"],
                    "; ".join(
                        f"{m.get('source')}={m.get('text')}"
                        for m in (item.get("exact_matches", []) + item.get("fuzzy_matches", []))[:4]
                    ),
                ]
                for item in case["top_scored_pages"]
            ],
        )
        lines.extend(["", "### Quality Checks", ""])
        for key, value in case["quality_checks"].items():
            lines.append(f"- `{key}`: `{value}`")
        if case["missing_page_text_doc_ids"]:
            lines.append(f"- `missing_page_text_doc_ids`: `{case['missing_page_text_doc_ids']}`")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    gold_rows = load_jsonl_by_qid(Path(args.gold))
    strict_rows = load_jsonl_by_qid(Path(args.strict_augmented_gold))
    loose_rows = load_jsonl_by_qid(Path(args.loose_augmented_gold)) if args.loose_augmented_gold else {}
    pages_by_doc = ppl.load_page_texts(Path(args.doc_pages_jsonl))
    texts = ppl.load_by_id(args.mmqa_texts_jsonl)
    tables = ppl.load_by_id(args.mmqa_tables_jsonl)
    images = ppl.load_by_id(args.mmqa_images_jsonl)
    id_map = ppl.load_id_map(args.id_url_mapping_jsonl)
    url_to_id = ppl.load_url_to_id(args.id_url_mapping_jsonl)

    qids = choose_qids(
        gold_rows,
        strict_rows,
        loose_rows,
        explicit_qids=args.qid,
        limit=int(args.limit),
    )
    cases = [
        build_case(
            qid,
            gold_rows=gold_rows,
            strict_rows=strict_rows,
            loose_rows=loose_rows,
            pages_by_doc=pages_by_doc,
            texts=texts,
            tables=tables,
            images=images,
            id_map=id_map,
            url_to_id=url_to_id,
            args=args,
        )
        for qid in qids
    ]

    output_md = Path(args.output_md)
    write_markdown(cases, output_md)
    print(f"saved_output_md={output_md}")
    print(f"case_count={len(cases)}")
    print("qids=" + ",".join(qids))

    if args.output_jsonl:
        output_jsonl = Path(args.output_jsonl)
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with output_jsonl.open("w", encoding="utf-8") as handle:
            for case in cases:
                handle.write(json.dumps(case, ensure_ascii=False) + "\n")
        print(f"saved_output_jsonl={output_jsonl}")


if __name__ == "__main__":
    main()
