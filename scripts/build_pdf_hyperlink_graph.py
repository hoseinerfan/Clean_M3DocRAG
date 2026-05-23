#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


WIKI_NAMESPACE_PREFIXES = {
    "category",
    "category talk",
    "draft",
    "file",
    "file talk",
    "help",
    "help talk",
    "media",
    "mediawiki",
    "module",
    "portal",
    "special",
    "talk",
    "template",
    "template talk",
    "user",
    "user talk",
    "wikipedia",
    "wikipedia talk",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a filtered, deduplicated PDF hyperlink graph from "
            "scripts/audit_pdf_hyperlinks.py JSON/JSONL output."
        )
    )
    parser.add_argument("--audit-jsonl", default="", help="Hyperlink audit JSONL records.")
    parser.add_argument("--audit-json", default="", help="Hyperlink audit JSON with a records array.")
    parser.add_argument(
        "--pdf-root",
        default="",
        help="Optional PDF root. Its PDF stems define the in-split doc-id universe.",
    )
    parser.add_argument(
        "--valid-doc-ids-json",
        default="",
        help="Optional JSON list/dict of valid doc ids used to filter target docs.",
    )
    parser.add_argument(
        "--valid-doc-ids-jsonl",
        default="",
        help="Optional JSONL with id/doc_id fields used to filter target docs.",
    )
    parser.add_argument("--keep-self-links", action="store_true")
    parser.add_argument("--output-edges-jsonl", required=True)
    parser.add_argument("--output-summary-json", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--sample", type=int, default=20)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_audit_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if args.audit_jsonl:
        records.extend(read_jsonl(Path(args.audit_jsonl)))
    if args.audit_json:
        payload = json.loads(Path(args.audit_json).read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("records"), list):
            records.extend(payload["records"])
        elif isinstance(payload, list):
            records.extend(payload)
    if not records:
        raise ValueError("Provide --audit-jsonl or --audit-json with at least one record.")
    return records


def load_valid_doc_ids(args: argparse.Namespace) -> set[str]:
    valid: set[str] = set()
    if args.pdf_root:
        root = Path(args.pdf_root)
        if root.exists():
            valid |= {path.stem for path in root.rglob("*.pdf")}
    if args.valid_doc_ids_json:
        payload = json.loads(Path(args.valid_doc_ids_json).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            valid |= {str(key) for key in payload.keys()}
        elif isinstance(payload, list):
            valid |= {str(item) for item in payload}
    if args.valid_doc_ids_jsonl:
        for row in read_jsonl(Path(args.valid_doc_ids_jsonl)):
            doc_id = str(row.get("doc_id", row.get("id", ""))).strip()
            if doc_id:
                valid.add(doc_id)
    return valid


def parse_wikipedia_article_url(url: str) -> tuple[str, str, str] | None:
    raw = str(url or "").strip()
    if not raw:
        return None
    parsed = urlparse(raw)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if not host.endswith("wikipedia.org"):
        return None
    if not parsed.path.startswith("/wiki/"):
        return None
    raw_title = unquote(parsed.path[len("/wiki/") :]).strip()
    if not raw_title:
        return None
    title_without_fragment = raw_title.split("#", 1)[0].strip()
    if not title_without_fragment:
        return None
    namespace = title_without_fragment.split(":", 1)[0].replace("_", " ").strip().lower()
    if ":" in title_without_fragment and namespace in WIKI_NAMESPACE_PREFIXES:
        return None
    display_title = title_without_fragment.replace("_", " ")
    canonical_title = display_title.lower()
    canonical_url = f"https://{host}/wiki/{title_without_fragment.replace(' ', '_')}"
    return canonical_title, display_title, canonical_url


def source_page_uid(source_doc_id: str, page_idx: int | None) -> str:
    if page_idx is None:
        return source_doc_id
    return f"{source_doc_id}_page{int(page_idx)}"


def build_edges(
    records: list[dict[str, Any]],
    valid_doc_ids: set[str],
    *,
    keep_self_links: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    skipped = Counter()
    raw_article_records = 0
    raw_mapped_article_records = 0
    raw_valid_target_records = 0
    aggregate: dict[tuple[str, int | None, str, str], dict[str, Any]] = {}

    for record in records:
        source_doc_id = str(record.get("doc_id", "")).strip()
        if not source_doc_id:
            skipped["missing_source_doc_id"] += 1
            continue
        page_idx_raw = record.get("page_idx")
        try:
            page_idx = int(page_idx_raw) if page_idx_raw is not None else None
        except (TypeError, ValueError):
            page_idx = None

        parsed_url = parse_wikipedia_article_url(str(record.get("url", "") or ""))
        if parsed_url is None:
            skipped["not_article_wikipedia_uri"] += 1
            continue
        raw_article_records += 1
        target_doc_ids = [str(x).strip() for x in record.get("target_doc_ids", []) if str(x).strip()]
        if not target_doc_ids:
            skipped["unmapped_article_uri"] += 1
            continue
        raw_mapped_article_records += 1

        canonical_title, display_title, canonical_url = parsed_url
        for target_doc_id in target_doc_ids:
            if not keep_self_links and target_doc_id == source_doc_id:
                skipped["self_link"] += 1
                continue
            target_in_valid_docs = not valid_doc_ids or target_doc_id in valid_doc_ids
            if valid_doc_ids and not target_in_valid_docs:
                skipped["target_not_in_valid_doc_ids"] += 1
                continue
            raw_valid_target_records += 1
            key = (source_doc_id, page_idx, target_doc_id, canonical_title)
            if key not in aggregate:
                aggregate[key] = {
                    "edge_type": "pdf_wikipedia_hyperlink",
                    "source_doc_id": source_doc_id,
                    "source_page_idx": page_idx,
                    "source_page_uid": source_page_uid(source_doc_id, page_idx),
                    "target_doc_id": target_doc_id,
                    "target_wiki_title": display_title,
                    "canonical_url": canonical_url,
                    "target_in_valid_doc_ids": target_in_valid_docs,
                    "raw_link_count": 0,
                }
            aggregate[key]["raw_link_count"] += 1

    edges = sorted(
        aggregate.values(),
        key=lambda row: (
            str(row["source_doc_id"]),
            -int(row["raw_link_count"]),
            str(row["target_doc_id"]),
            str(row["target_wiki_title"]).lower(),
        ),
    )
    for index, edge in enumerate(edges):
        edge["edge_id"] = f"pdf_wikilink_{index:08d}"

    source_docs = {edge["source_doc_id"] for edge in edges}
    source_pages = {edge["source_page_uid"] for edge in edges}
    target_docs = {edge["target_doc_id"] for edge in edges}
    out_degree_doc = Counter(edge["source_doc_id"] for edge in edges)
    in_degree_doc = Counter(edge["target_doc_id"] for edge in edges)
    out_degree_page = Counter(edge["source_page_uid"] for edge in edges)
    summary = {
        "input_record_count": len(records),
        "raw_article_uri_records": raw_article_records,
        "raw_mapped_article_uri_records": raw_mapped_article_records,
        "raw_valid_target_records": raw_valid_target_records,
        "deduped_edge_count": len(edges),
        "source_doc_count": len(source_docs),
        "source_page_count": len(source_pages),
        "target_doc_count": len(target_docs),
        "valid_doc_id_count": len(valid_doc_ids),
        "source_doc_coverage_of_valid_docs": (
            len(source_docs & valid_doc_ids) / len(valid_doc_ids) if valid_doc_ids else None
        ),
        "target_doc_coverage_of_valid_docs": (
            len(target_docs & valid_doc_ids) / len(valid_doc_ids) if valid_doc_ids else None
        ),
        "skipped": dict(skipped),
        "top_source_docs_by_out_degree": out_degree_doc.most_common(20),
        "top_target_docs_by_in_degree": in_degree_doc.most_common(20),
        "top_source_pages_by_out_degree": out_degree_page.most_common(20),
        "sample_edges": edges[:20],
    }
    return edges, summary


def write_markdown(path: Path, summary: dict[str, Any], sample_count: int) -> None:
    lines = [
        "# PDF Hyperlink Graph",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in [
        "input_record_count",
        "raw_article_uri_records",
        "raw_mapped_article_uri_records",
        "raw_valid_target_records",
        "deduped_edge_count",
        "source_doc_count",
        "source_page_count",
        "target_doc_count",
        "valid_doc_id_count",
        "source_doc_coverage_of_valid_docs",
        "target_doc_coverage_of_valid_docs",
    ]:
        lines.append(f"| {key} | {summary.get(key)} |")
    lines.extend(["", "## Skipped", "", "| reason | count |", "| --- | --- |"])
    for reason, count in sorted(summary.get("skipped", {}).items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {reason} | {count} |")
    lines.extend(["", "## Sample Edges", ""])
    for edge in summary.get("sample_edges", [])[:sample_count]:
        lines.append(
            "- "
            f"`{edge['source_page_uid']}` -> `{edge['target_doc_id']}` "
            f"({edge['target_wiki_title']}, raw_count={edge['raw_link_count']})"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    records = load_audit_records(args)
    valid_doc_ids = load_valid_doc_ids(args)
    edges, summary = build_edges(
        records,
        valid_doc_ids,
        keep_self_links=bool(args.keep_self_links),
    )

    output_edges = Path(args.output_edges_jsonl)
    output_edges.parent.mkdir(parents=True, exist_ok=True)
    with output_edges.open("w", encoding="utf-8") as handle:
        for edge in edges:
            handle.write(json.dumps(edge, ensure_ascii=False) + "\n")

    if args.output_summary_json:
        Path(args.output_summary_json).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if args.output_md:
        write_markdown(Path(args.output_md), summary, int(args.sample))

    print("saved_edges", output_edges)
    if args.output_summary_json:
        print("saved_summary", args.output_summary_json)
    if args.output_md:
        print("saved_md", args.output_md)
    for key in [
        "input_record_count",
        "raw_article_uri_records",
        "raw_mapped_article_uri_records",
        "raw_valid_target_records",
        "deduped_edge_count",
        "source_doc_count",
        "source_page_count",
        "target_doc_count",
        "valid_doc_id_count",
        "source_doc_coverage_of_valid_docs",
        "target_doc_coverage_of_valid_docs",
    ]:
        print(key, summary.get(key))
    print("skipped", summary["skipped"])


if __name__ == "__main__":
    main()
