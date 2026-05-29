#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


URL_RE = re.compile(r"https?://[^\s<>()\[\]{}\"']+", re.IGNORECASE)
LINK_KEY_RE = re.compile(r"(?:url|href|link|uri|target|reference)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether converted pages and/or source PDFs expose hyperlinks. "
            "For M3DocVQA, pass --id-url-jsonl to estimate links between dataset docs."
        )
    )
    parser.add_argument("--doc-pages-jsonl", default="", help="Optional converted doc_pages JSONL.")
    parser.add_argument("--pdf-root", default="", help="Optional directory containing source PDFs.")
    parser.add_argument(
        "--id-url-jsonl",
        action="append",
        default=[],
        help="Optional JSONL with id/doc_id and url fields. Repeat to merge mappings.",
    )
    parser.add_argument("--max-docs", type=int, default=0, help="Limit PDF docs for a quick audit.")
    parser.add_argument("--sample", type=int, default=20)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-jsonl", default="")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_url(url: str) -> str:
    value = str(url or "").strip().strip(".,);]}'\"")
    if not value:
        return ""
    value = unquote(value)
    parsed = urlparse(value)
    if not parsed.netloc:
        return value.lower()
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path.rstrip("/")
    return f"{host}{path}".lower()


def canonical_wiki_title(url: str) -> str:
    normalized = normalize_url(url)
    marker = "wikipedia.org/wiki/"
    if marker not in normalized:
        return ""
    title = normalized.split(marker, 1)[1]
    title = title.split("#", 1)[0].split("?", 1)[0]
    return title.replace("_", " ").strip().lower()


def load_id_url_mapping(paths: list[str]) -> tuple[dict[str, str], dict[str, set[str]]]:
    id_to_url: dict[str, str] = {}
    key_to_doc_ids: dict[str, set[str]] = defaultdict(set)
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        for row in read_jsonl(path):
            doc_id = str(row.get("id", row.get("doc_id", ""))).strip()
            url = str(row.get("url", "")).strip()
            if not doc_id or not url:
                continue
            id_to_url[doc_id] = url
            normalized = normalize_url(url)
            if normalized:
                key_to_doc_ids[normalized].add(doc_id)
            title = canonical_wiki_title(url)
            if title:
                key_to_doc_ids[f"wiki_title::{title}"].add(doc_id)
    return id_to_url, key_to_doc_ids


def target_doc_ids_for_url(url: str, key_to_doc_ids: dict[str, set[str]]) -> list[str]:
    hits = set()
    normalized = normalize_url(url)
    if normalized:
        hits |= key_to_doc_ids.get(normalized, set())
    title = canonical_wiki_title(url)
    if title:
        hits |= key_to_doc_ids.get(f"wiki_title::{title}", set())
    return sorted(hits)


def parse_page_uid(page_uid: str) -> tuple[str, int | None]:
    if "_page" not in page_uid:
        return page_uid, None
    doc_id, raw_page_idx = page_uid.rsplit("_page", 1)
    try:
        return doc_id, int(raw_page_idx)
    except ValueError:
        return doc_id, None


def row_doc_page_identity(row: dict[str, Any]) -> tuple[str, int | None]:
    doc_id = str(row.get("doc_id", "")).strip()
    raw_page_idx = row.get("page_idx", row.get("page_id", row.get("page_number")))
    try:
        page_idx = int(raw_page_idx)
    except (TypeError, ValueError):
        page_idx = None
    page_uid = str(row.get("page_uid", "")).strip()
    if page_uid and (not doc_id or page_idx is None):
        uid_doc_id, uid_page_idx = parse_page_uid(page_uid)
        doc_id = doc_id or uid_doc_id
        page_idx = page_idx if page_idx is not None else uid_page_idx
    return doc_id, page_idx


def iter_json_paths(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        out = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.extend(iter_json_paths(child, child_prefix))
        return out
    if isinstance(value, list):
        out = []
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            out.extend(iter_json_paths(child, child_prefix))
        return out
    return [(prefix, value)]


def doc_pages_link_records(
    path: Path,
    key_to_doc_ids: dict[str, set[str]],
) -> list[dict[str, Any]]:
    records = []
    for row in read_jsonl(path):
        doc_id, page_idx = row_doc_page_identity(row)
        for field_path, value in iter_json_paths(row):
            urls = []
            if LINK_KEY_RE.search(field_path) and isinstance(value, (str, int, float)):
                text = str(value)
                if text.startswith(("http://", "https://")):
                    urls.append(text)
            if isinstance(value, str):
                urls.extend(URL_RE.findall(value))
            for url in sorted(set(urls)):
                records.append(
                    {
                        "source": "doc_pages",
                        "doc_id": doc_id,
                        "page_idx": page_idx,
                        "field": field_path,
                        "url": url,
                        "target_doc_ids": target_doc_ids_for_url(url, key_to_doc_ids),
                    }
                )
    return records


def resolve_pdf_from_doc_rows(
    pdf_root: Path,
    doc_id: str,
    rows: list[dict[str, Any]],
    pdfs_by_stem: dict[str, Path],
) -> Path | None:
    if doc_id in pdfs_by_stem:
        return pdfs_by_stem[doc_id]

    seen: set[Path] = set()
    candidates: list[Path] = []
    for row in rows:
        for key in ["pdf_path", "source_pdf_path"]:
            value = str(row.get(key, "") or "").strip()
            if not value:
                continue
            path = Path(value)
            candidates.append(path if path.is_absolute() else pdf_root / path)
        file_name = str(row.get("file_name", "") or "").strip()
        if file_name:
            candidates.extend([pdf_root / file_name, pdf_root / "PDF" / file_name])
        doc_name = str(row.get("doc_name", "") or "").strip()
        category = str(row.get("category", "") or "").strip()
        if doc_name:
            candidates.append(pdf_root / f"{doc_name}.pdf")
            if category:
                candidates.extend(
                    [
                        pdf_root / "PDF" / category / f"{doc_name}.pdf",
                        pdf_root / category / f"{doc_name}.pdf",
                    ]
                )
            if doc_name in pdfs_by_stem:
                candidates.append(pdfs_by_stem[doc_name])

    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists() and candidate.is_file():
            return candidate

    for row in rows[:1]:
        file_name = str(row.get("file_name", "") or "").strip()
        if file_name:
            matches = sorted(pdf_root.rglob(file_name))
            if matches:
                return matches[0]
        doc_name = str(row.get("doc_name", "") or "").strip()
        if doc_name:
            matches = sorted(pdf_root.rglob(f"{doc_name}*.pdf"))
            if matches:
                return matches[0]
    return None


def discover_pdfs(
    pdf_root: Path,
    doc_ids: set[str],
    max_docs: int,
    doc_rows_by_id: dict[str, list[dict[str, Any]]] | None = None,
) -> list[tuple[str, Path]]:
    pdfs_by_stem = {path.stem: path for path in pdf_root.rglob("*.pdf")}
    selected_doc_ids = sorted(doc_ids) if doc_ids else sorted(pdfs_by_stem)
    if max_docs > 0:
        selected_doc_ids = selected_doc_ids[:max_docs]
    out = []
    rows_by_id = doc_rows_by_id or {}
    for doc_id in selected_doc_ids:
        pdf_path = resolve_pdf_from_doc_rows(pdf_root, doc_id, rows_by_id.get(doc_id, []), pdfs_by_stem)
        if pdf_path is not None:
            out.append((doc_id, pdf_path))
    return out


def pdf_link_records_with_fitz(
    pdfs: list[tuple[str, Path]],
    key_to_doc_ids: dict[str, set[str]],
) -> tuple[list[dict[str, Any]], str]:
    try:
        import fitz  # type: ignore
    except Exception:
        return [], "fitz_unavailable"

    records = []
    for doc_id, pdf_path in pdfs:
        try:
            pdf = fitz.open(str(pdf_path))
        except Exception as exc:
            records.append(
                {
                    "source": "pdf_error",
                    "doc_id": doc_id,
                    "page_idx": None,
                    "pdf_path": str(pdf_path),
                    "error": str(exc),
                    "url": "",
                    "target_doc_ids": [],
                }
            )
            continue
        try:
            for page_idx in range(len(pdf)):
                try:
                    links = pdf[page_idx].get_links()
                except Exception:
                    links = []
                for link in links:
                    url = str(link.get("uri", "") or "")
                    target_page = link.get("page")
                    kind = str(link.get("kind", ""))
                    records.append(
                        {
                            "source": "pdf_annotation",
                            "doc_id": doc_id,
                            "page_idx": page_idx,
                            "pdf_path": str(pdf_path),
                            "kind": kind,
                            "url": url,
                            "target_page": target_page if target_page is not None else None,
                            "target_doc_ids": target_doc_ids_for_url(url, key_to_doc_ids)
                            if url
                            else [],
                        }
                    )
        finally:
            pdf.close()
    return records, "fitz"


def pdf_obj_text(value: Any) -> str:
    if value is None:
        return ""
    to_unicode = getattr(value, "to_unicode", None)
    if callable(to_unicode):
        try:
            return str(to_unicode())
        except Exception:
            pass
    text = str(value)
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    return text


def pdf_link_records_with_pypdf(
    pdfs: list[tuple[str, Path]],
    key_to_doc_ids: dict[str, set[str]],
) -> tuple[list[dict[str, Any]], str]:
    try:
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception:
            from PyPDF2 import PdfReader  # type: ignore
    except Exception:
        return [], "pypdf_unavailable"

    records = []
    for doc_id, pdf_path in pdfs:
        try:
            reader = PdfReader(str(pdf_path))
            pages = list(reader.pages)
        except Exception as exc:
            records.append(
                {
                    "source": "pdf_error",
                    "doc_id": doc_id,
                    "page_idx": None,
                    "pdf_path": str(pdf_path),
                    "error": str(exc),
                    "url": "",
                    "target_doc_ids": [],
                }
            )
            continue
        for page_idx, page in enumerate(pages):
            try:
                annots = page.get("/Annots") or []
            except Exception:
                annots = []
            for annot_ref in annots:
                try:
                    annot = annot_ref.get_object()
                except Exception:
                    annot = annot_ref
                try:
                    action = annot.get("/A") or {}
                    url = str(action.get("/URI") or "")
                    kind = str(annot.get("/Subtype") or "")
                    target_page = annot.get("/Dest") or action.get("/D")
                except Exception:
                    url = ""
                    kind = ""
                    target_page = None
                records.append(
                    {
                        "source": "pdf_annotation",
                        "doc_id": doc_id,
                        "page_idx": page_idx,
                        "pdf_path": str(pdf_path),
                        "kind": kind,
                        "url": url,
                        "target_page": pdf_obj_text(target_page) if target_page is not None else None,
                        "target_doc_ids": target_doc_ids_for_url(url, key_to_doc_ids) if url else [],
                    }
                )
    return records, "pypdf"


def pdf_link_records_with_pdfrw(
    pdfs: list[tuple[str, Path]],
    key_to_doc_ids: dict[str, set[str]],
) -> tuple[list[dict[str, Any]], str]:
    try:
        from pdfrw import PdfReader  # type: ignore
    except Exception:
        return [], "pdfrw_unavailable"

    records = []
    for doc_id, pdf_path in pdfs:
        try:
            reader = PdfReader(str(pdf_path))
            pages = list(reader.pages or [])
        except Exception as exc:
            records.append(
                {
                    "source": "pdf_error",
                    "doc_id": doc_id,
                    "page_idx": None,
                    "pdf_path": str(pdf_path),
                    "error": str(exc),
                    "url": "",
                    "target_doc_ids": [],
                }
            )
            continue
        for page_idx, page in enumerate(pages):
            annots = page.Annots or []
            for annot in annots:
                action = getattr(annot, "A", None)
                url = pdf_obj_text(getattr(action, "URI", None)) if action is not None else ""
                target_page = getattr(annot, "Dest", None)
                if target_page is None and action is not None:
                    target_page = getattr(action, "D", None)
                records.append(
                    {
                        "source": "pdf_annotation",
                        "doc_id": doc_id,
                        "page_idx": page_idx,
                        "pdf_path": str(pdf_path),
                        "kind": pdf_obj_text(getattr(annot, "Subtype", "")),
                        "url": url,
                        "target_page": pdf_obj_text(target_page) if target_page is not None else None,
                        "target_doc_ids": target_doc_ids_for_url(url, key_to_doc_ids) if url else [],
                    }
                )
    return records, "pdfrw"


def pdf_link_records(
    pdfs: list[tuple[str, Path]],
    key_to_doc_ids: dict[str, set[str]],
) -> tuple[list[dict[str, Any]], str]:
    records, backend = pdf_link_records_with_fitz(pdfs, key_to_doc_ids)
    if backend == "fitz":
        return records, backend
    records, backend = pdf_link_records_with_pypdf(pdfs, key_to_doc_ids)
    if backend == "pypdf":
        return records, backend
    records, backend = pdf_link_records_with_pdfrw(pdfs, key_to_doc_ids)
    if backend == "pdfrw":
        return records, backend
    return [], "no_pdf_link_backend"


def summarize(records: list[dict[str, Any]], *, pdf_count: int, doc_page_count: int) -> dict[str, Any]:
    records_with_links = [row for row in records if row.get("source") != "pdf_error"]
    by_source = Counter(row.get("source", "") for row in records)
    docs_with_links = {row.get("doc_id") for row in records_with_links if row.get("doc_id")}
    pages_with_links = {
        (row.get("doc_id"), row.get("page_idx"))
        for row in records_with_links
        if row.get("doc_id") and row.get("page_idx") is not None
    }
    uri_records = [row for row in records_with_links if row.get("url")]
    internal_records = [row for row in uri_records if row.get("target_doc_ids")]
    return {
        "record_count": len(records_with_links),
        "by_source": dict(by_source),
        "doc_pages_row_count": doc_page_count,
        "pdf_count_checked": pdf_count,
        "docs_with_links": len(docs_with_links),
        "pages_with_links": len(pages_with_links),
        "uri_link_count": len(uri_records),
        "internal_dataset_link_count": len(internal_records),
        "unique_url_count": len({normalize_url(row.get("url", "")) for row in uri_records if row.get("url")}),
        "unique_target_doc_count": len(
            {
                target
                for row in internal_records
                for target in row.get("target_doc_ids", [])
            }
        ),
    }


def main() -> None:
    args = parse_args()
    _id_to_url, key_to_doc_ids = load_id_url_mapping(args.id_url_jsonl)

    records: list[dict[str, Any]] = []
    doc_ids: set[str] = set()
    doc_rows_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    doc_page_count = 0
    if args.doc_pages_jsonl:
        doc_pages_path = Path(args.doc_pages_jsonl)
        if doc_pages_path.exists():
            rows = read_jsonl(doc_pages_path)
            doc_page_count = len(rows)
            for row in rows:
                doc_id, _page_idx = row_doc_page_identity(row)
                if doc_id:
                    doc_ids.add(doc_id)
                    doc_rows_by_id[doc_id].append(row)
            records.extend(doc_pages_link_records(doc_pages_path, key_to_doc_ids))

    pdf_count = 0
    pdf_backend = ""
    if args.pdf_root:
        pdf_root = Path(args.pdf_root)
        if pdf_root.exists():
            pdfs = discover_pdfs(pdf_root, doc_ids, int(args.max_docs), doc_rows_by_id)
            pdf_count = len(pdfs)
            pdf_records, pdf_backend = pdf_link_records(pdfs, key_to_doc_ids)
            records.extend(pdf_records)

    payload = {
        "summary": summarize(records, pdf_count=pdf_count, doc_page_count=doc_page_count),
        "pdf_backend": pdf_backend,
        "sample_records": records[: max(0, int(args.sample))],
        "records": records,
    }
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.output_jsonl:
        with Path(args.output_jsonl).open("w", encoding="utf-8") as handle:
            for row in records:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("summary", json.dumps(payload["summary"], sort_keys=True))
    print("pdf_backend", pdf_backend or "none")
    print("sample_records")
    for row in payload["sample_records"]:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
