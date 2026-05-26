#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+.+?\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit heading-rescue cases by joining rescue gate diagnostics with "
            "PDF-derived markdown pages. Supports comparing full, heuristic-only, "
            "strict, and outline-only doc_pages JSONL variants."
        )
    )
    parser.add_argument("--cases-json", required=True)
    parser.add_argument(
        "--doc-pages-jsonl",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Doc pages JSONL with markdown fields. Repeatable.",
    )
    parser.add_argument("--gold", default="")
    parser.add_argument("--qid", action="append", default=[], help="Specific qid to print.")
    parser.add_argument("--page-uid", action="append", default=[], help="Specific page uid to print.")
    parser.add_argument("--movement", action="append", default=[], help="Filter by movement_vs_base.")
    parser.add_argument("--selection-reason", action="append", default=[])
    parser.add_argument("--accepted-only", action="store_true")
    parser.add_argument("--rejected-only", action="store_true")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--base-top-n", type=int, default=4)
    parser.add_argument("--candidate-top-n", type=int, default=4)
    parser.add_argument("--markdown-chars", type=int, default=1800)
    parser.add_argument("--heading-lines", type=int, default=12)
    parser.add_argument("--output-md", default="")
    return parser.parse_args()


def parse_labeled_path(value: str) -> tuple[str, Path]:
    raw = str(value).strip()
    if not raw:
        raise ValueError("Empty labeled path")
    if "=" in raw:
        label, path = raw.split("=", 1)
        label = label.strip()
        parsed = Path(path.strip())
    else:
        parsed = Path(raw)
        label = parsed.stem
    if not label:
        label = parsed.stem
    return label, parsed


def read_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
        payload = payload["cases"]
    if not isinstance(payload, list):
        raise TypeError(f"Expected cases JSON array or object with cases array: {path}")
    return [row for row in payload if isinstance(row, dict)]


def page_uid(doc_id: object, page_idx: object) -> str:
    return f"{doc_id}_page{int(page_idx)}"


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
            uids.add(page_uid(doc_id, page_idx))
    return uids


def load_gold(path: str) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("qid", "")).strip()
            if qid:
                rows[qid] = row
    return rows


def row_page_uid(row: dict[str, Any]) -> str | None:
    doc_id = str(row.get("doc_id", "") or row.get("doc_name", "")).strip()
    if not doc_id:
        return None
    page_idx = row.get("page_idx", row.get("page_id", row.get("page")))
    if page_idx is None:
        return None
    try:
        return page_uid(doc_id, page_idx)
    except (TypeError, ValueError):
        return None


def load_doc_pages(paths: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
    catalogs: dict[str, dict[str, dict[str, Any]]] = {}
    for value in paths:
        label, path = parse_labeled_path(value)
        pages: dict[str, dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                uid = row_page_uid(row)
                if uid:
                    pages[uid] = row
        catalogs[label] = pages
    return catalogs


def list_field(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def rejected_page_uids(case: dict[str, Any]) -> list[str]:
    uids: list[str] = []
    for item in case.get("rejected_promoted_pages", []):
        if isinstance(item, dict):
            uid = str(item.get("page_uid", "")).strip()
            if uid:
                uids.append(uid)
    return uids


def unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def case_pages(case: dict[str, Any], gold_rows: dict[str, dict[str, Any]], args: argparse.Namespace) -> list[str]:
    qid = str(case.get("qid", "")).strip()
    gold = list_field(case.get("gold_page_uids"))
    if not gold and qid in gold_rows:
        gold = sorted(gold_page_uids(gold_rows[qid]))
    promoted = list_field(case.get("promoted_pages"))
    candidate_promoted = list_field(case.get("candidate_promoted_pages"))
    rejected = rejected_page_uids(case)
    base_top = list_field(case.get("base_top_pages"))[: max(0, int(args.base_top_n))]
    candidate_top = list_field(case.get("candidate_top_pages"))[: max(0, int(args.candidate_top_n))]
    explicit = [str(uid).strip() for uid in args.page_uid if str(uid).strip()]
    return unique_in_order([*explicit, *gold, *promoted, *candidate_promoted, *rejected, *base_top, *candidate_top])


def heading_excerpt(markdown: str, max_lines: int) -> list[str]:
    lines: list[str] = []
    for line in str(markdown or "").splitlines():
        if MARKDOWN_HEADING_RE.match(line):
            lines.append(line.rstrip())
            if len(lines) >= max_lines:
                break
    return lines


def text_excerpt(markdown: str, max_chars: int) -> str:
    text = str(markdown or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n[...]"


def page_roles(uid: str, case: dict[str, Any], gold_pages: set[str], args: argparse.Namespace) -> list[str]:
    roles: list[str] = []
    if uid in gold_pages:
        roles.append("gold")
    if uid in set(list_field(case.get("promoted_pages"))):
        roles.append("accepted_promoted")
    if uid in set(list_field(case.get("candidate_promoted_pages"))):
        roles.append("candidate_promoted")
    if uid in set(rejected_page_uids(case)):
        roles.append("rejected_promoted")
    base_top = list_field(case.get("base_top_pages"))[: max(0, int(args.base_top_n))]
    cand_top = list_field(case.get("candidate_top_pages"))[: max(0, int(args.candidate_top_n))]
    if uid in base_top:
        roles.append(f"base_top{base_top.index(uid) + 1}")
    if uid in cand_top:
        roles.append(f"candidate_top{cand_top.index(uid) + 1}")
    return roles


def case_matches(case: dict[str, Any], args: argparse.Namespace) -> bool:
    qid_filters = {str(qid).strip() for qid in args.qid if str(qid).strip()}
    if qid_filters and str(case.get("qid", "")).strip() not in qid_filters:
        return False
    movement_filters = {str(value).strip() for value in args.movement if str(value).strip()}
    if movement_filters and str(case.get("movement_vs_base", "")).strip() not in movement_filters:
        return False
    reason_filters = {str(value).strip() for value in args.selection_reason if str(value).strip()}
    if reason_filters and str(case.get("selection_reason", "")).strip() not in reason_filters:
        return False
    if bool(args.accepted_only) and not bool(case.get("accepted")):
        return False
    if bool(args.rejected_only) and not case.get("rejected_promoted_pages"):
        return False
    return True


def render_case(
    case: dict[str, Any],
    *,
    catalogs: dict[str, dict[str, dict[str, Any]]],
    gold_rows: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    qid = str(case.get("qid", "")).strip()
    gold_pages = set(list_field(case.get("gold_page_uids")))
    if not gold_pages and qid in gold_rows:
        gold_pages = gold_page_uids(gold_rows[qid])
    lines: list[str] = []
    lines.append(f"## {qid}")
    lines.append("")
    if case.get("question"):
        lines.append(f"Question: {case.get('question')}")
        lines.append("")
    lines.append(
        "Case: "
        f"accepted={case.get('accepted')} "
        f"selected_source={case.get('selected_source')} "
        f"reason={case.get('selection_reason')} "
        f"movement={case.get('movement_vs_base')}"
    )
    rank_fields = [
        "base_first_gold_page_rank",
        "candidate_first_gold_page_rank",
        "output_first_gold_page_rank",
        "base_first_gold_doc_rank",
        "candidate_first_gold_doc_rank",
        "output_first_gold_doc_rank",
    ]
    rank_summary = {key: case.get(key) for key in rank_fields if key in case}
    if rank_summary:
        lines.append(f"Ranks: {json.dumps(rank_summary, sort_keys=True)}")
    lines.append(f"Gold pages: {sorted(gold_pages)}")
    lines.append(f"Promoted pages: {list_field(case.get('promoted_pages'))}")
    lines.append(f"Candidate promoted pages: {list_field(case.get('candidate_promoted_pages'))}")
    if case.get("rejected_promoted_pages"):
        lines.append("Rejected promoted pages:")
        for item in case.get("rejected_promoted_pages", []):
            if isinstance(item, dict):
                brief = {
                    key: item.get(key)
                    for key in (
                        "page_uid",
                        "reject_reason",
                        "candidate_rank",
                        "base_rank",
                        "base_doc_rank",
                        "support_page_vote_count",
                        "support_doc_vote_count",
                    )
                }
                lines.append(f"- {json.dumps(brief, sort_keys=True)}")
    lines.append(f"Base top pages: {list_field(case.get('base_top_pages'))[:max(0, int(args.base_top_n))]}")
    lines.append(
        f"Candidate top pages: {list_field(case.get('candidate_top_pages'))[:max(0, int(args.candidate_top_n))]}"
    )
    lines.append("")

    for uid in case_pages(case, gold_rows, args):
        roles = page_roles(uid, case, gold_pages, args)
        lines.append(f"### {uid} [{', '.join(roles) if roles else 'context'}]")
        for label, pages in catalogs.items():
            row = pages.get(uid)
            if row is None:
                lines.append(f"- {label}: MISSING")
                continue
            source = row.get("markdown_source", "")
            original_source = row.get("markdown_original_source", "")
            outline_count = row.get("pdf_outline_heading_count", row.get("pdf_variant_outline_heading_count", ""))
            heuristic_count = row.get(
                "pdf_heuristic_heading_count",
                row.get("pdf_variant_heuristic_heading_count", ""),
            )
            md = str(row.get("markdown", "") or "")
            heads = heading_excerpt(md, max(0, int(args.heading_lines)))
            lines.append(
                f"- {label}: source={source} original_source={original_source} "
                f"outline={outline_count} heuristic={heuristic_count} chars={len(md)}"
            )
            if heads:
                lines.append("  headings:")
                for head in heads:
                    lines.append(f"  - {head}")
            excerpt = text_excerpt(md, max(0, int(args.markdown_chars)))
            if excerpt:
                lines.append("")
                lines.append(f"```markdown\n{excerpt}\n```")
            else:
                lines.append("")
                lines.append("```markdown\n[EMPTY MARKDOWN]\n```")
        lines.append("")
    return "\n".join(lines).rstrip()


def main() -> None:
    args = parse_args()
    cases = read_cases(Path(args.cases_json))
    gold_rows = load_gold(args.gold)
    catalogs = load_doc_pages(args.doc_pages_jsonl)
    if not catalogs:
        raise ValueError("Provide at least one --doc-pages-jsonl LABEL=PATH")

    selected = [case for case in cases if case_matches(case, args)]
    if int(args.limit) > 0:
        selected = selected[: int(args.limit)]

    parts = [
        "# Heading Rescue Case Audit",
        "",
        f"cases_json: {args.cases_json}",
        f"selected_case_count: {len(selected)}",
        "",
    ]
    for case in selected:
        parts.append(render_case(case, catalogs=catalogs, gold_rows=gold_rows, args=args))
        parts.append("")
    output = "\n".join(parts).rstrip() + "\n"

    if args.output_md:
        path = Path(args.output_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output, encoding="utf-8")
        print(f"saved_md: {path}")
    else:
        print(output)


if __name__ == "__main__":
    main()
