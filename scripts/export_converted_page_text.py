#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


DEFAULT_TEXT_FIELDS = ["ocr_text", "vlm_text", "markdown", "text"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export one text row per page from a converted dataset doc_pages JSONL. "
            "The output format matches scripts/build_splade_page_index.py."
        )
    )
    parser.add_argument("--doc-pages-jsonl", required=True)
    parser.add_argument(
        "--text-field",
        action="append",
        default=[],
        help=(
            "Page manifest field to use as text. Repeat to concatenate fields. "
            f"Defaults to {DEFAULT_TEXT_FIELDS}."
        ),
    )
    parser.add_argument(
        "--extra-field",
        action="append",
        default=[],
        help="Optional non-text metadata field to append to the indexed text.",
    )
    parser.add_argument(
        "--pdf-root",
        default="",
        help=(
            "Optional root containing source PDFs. Used only when manifest text fields "
            "are empty."
        ),
    )
    parser.add_argument("--pdftotext-bin", default="pdftotext")
    parser.add_argument("--max-pages", type=int, default=0)
    parser.add_argument(
        "--require-nonempty",
        action="store_true",
        help="Fail if no page has non-empty extracted text.",
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-summary-json", required=True)
    return parser.parse_args()


def read_jsonl(path: Path, max_pages: int) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_pages > 0 and len(rows) >= max_pages:
                break
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = " ".join(str(item) for item in value if item is not None)
    elif isinstance(value, dict):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = str(value).replace("\x0c", " ").replace("\u0000", " ")
    return re.sub(r"\s+", " ", text).strip()


def lexical_token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", text))


def candidate_pdf_paths(row: dict, pdf_root: Path) -> list[Path]:
    candidates: list[Path] = []

    def add(value: object) -> None:
        text = str(value or "").strip()
        if not text:
            return
        path = Path(text)
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.append(pdf_root / path)

    add(row.get("pdf_path"))
    add(row.get("file_name"))

    doc_name = str(row.get("doc_name", "") or "").strip()
    category = str(row.get("category", "") or "").strip()
    if doc_name:
        candidates.append(pdf_root / f"{doc_name}.pdf")
        if category:
            candidates.append(pdf_root / category / f"{doc_name}.pdf")
            candidates.append(pdf_root / "PDF" / category / f"{doc_name}.pdf")
    file_name = str(row.get("file_name", "") or "").strip()
    if file_name:
        candidates.append(pdf_root / "vidoseek_pdf_document" / file_name)

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def resolve_pdf_path(row: dict, pdf_root: Path, cache: dict[str, Path | None]) -> Path | None:
    doc_id = str(row.get("doc_id", "") or "")
    if doc_id in cache:
        return cache[doc_id]

    for candidate in candidate_pdf_paths(row, pdf_root):
        if candidate.exists():
            cache[doc_id] = candidate
            return candidate

    file_name = str(row.get("file_name", "") or "").strip()
    doc_name = str(row.get("doc_name", "") or "").strip()
    patterns = []
    if file_name:
        patterns.append(file_name)
    if doc_name:
        patterns.append(f"{doc_name}.pdf")
        patterns.append(f"{doc_name}*.pdf")
    for pattern in patterns:
        matches = sorted(pdf_root.rglob(pattern))
        if matches:
            cache[doc_id] = matches[0]
            return matches[0]

    cache[doc_id] = None
    return None


def extract_pdf_page_text(*, pdftotext_bin: str, pdf_path: Path, page_idx: int) -> str:
    page_num = int(page_idx) + 1
    result = subprocess.run(
        [
            pdftotext_bin,
            "-f",
            str(page_num),
            "-l",
            str(page_num),
            str(pdf_path),
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return normalize_text(result.stdout)


def build_page_text(
    *,
    row: dict,
    text_fields: list[str],
    extra_fields: list[str],
    pdf_root: Path | None,
    pdftotext_bin: str,
    pdf_cache: dict[str, Path | None],
) -> tuple[str, str]:
    parts: list[str] = []
    seen_parts: set[str] = set()
    source = "manifest"

    for field in text_fields:
        text = normalize_text(row.get(field))
        if text and text not in seen_parts:
            seen_parts.add(text)
            parts.append(text)

    if not parts and pdf_root is not None:
        pdf_path = resolve_pdf_path(row, pdf_root, pdf_cache)
        if pdf_path is not None:
            text = extract_pdf_page_text(
                pdftotext_bin=pdftotext_bin,
                pdf_path=pdf_path,
                page_idx=int(row.get("page_idx", 0)),
            )
            if text:
                parts.append(text)
                source = "pdf"

    for field in extra_fields:
        text = normalize_text(row.get(field))
        if text and text not in seen_parts:
            seen_parts.add(text)
            parts.append(text)

    return "\n".join(parts).strip(), source


def main() -> None:
    args = parse_args()
    text_fields = args.text_field or DEFAULT_TEXT_FIELDS
    extra_fields = args.extra_field or []
    pdf_root = Path(args.pdf_root) if args.pdf_root else None

    rows = read_jsonl(Path(args.doc_pages_jsonl), int(args.max_pages))
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json = Path(args.output_summary_json)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)

    pdf_cache: dict[str, Path | None] = {}
    page_count = 0
    nonempty_text_page_count = 0
    empty_text_page_count = 0
    source_counts = {"manifest": 0, "pdf": 0, "empty": 0}
    char_counts: list[int] = []
    token_counts: list[int] = []

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            doc_id = str(row.get("doc_id", "") or "").strip()
            page_idx = int(row.get("page_idx", row.get("page_id", 0)))
            page_uid = str(row.get("page_uid", "") or "").strip() or f"{doc_id}_page{page_idx}"
            text, source = build_page_text(
                row=row,
                text_fields=text_fields,
                extra_fields=extra_fields,
                pdf_root=pdf_root,
                pdftotext_bin=args.pdftotext_bin,
                pdf_cache=pdf_cache,
            )
            if text:
                nonempty_text_page_count += 1
                source_counts[source] += 1
            else:
                empty_text_page_count += 1
                source_counts["empty"] += 1
            char_count = len(text)
            token_count = lexical_token_count(text)
            char_counts.append(char_count)
            token_counts.append(token_count)
            page_count += 1
            handle.write(
                json.dumps(
                    {
                        "page_uid": page_uid,
                        "doc_id": doc_id,
                        "page_idx": page_idx,
                        "text": text,
                        "char_count": char_count,
                        "token_count": token_count,
                        "text_source": source if text else "empty",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    if args.require_nonempty and nonempty_text_page_count == 0:
        raise ValueError(
            "No non-empty page text was exported. Provide OCR/markdown fields or --pdf-root."
        )

    summary = {
        "doc_pages_jsonl": args.doc_pages_jsonl,
        "text_fields": text_fields,
        "extra_fields": extra_fields,
        "pdf_root": args.pdf_root,
        "page_count": page_count,
        "nonempty_text_page_count": nonempty_text_page_count,
        "empty_text_page_count": empty_text_page_count,
        "empty_text_page_fraction": (
            float(empty_text_page_count) / float(page_count) if page_count else None
        ),
        "text_source_counts": source_counts,
        "mean_char_count": sum(char_counts) / len(char_counts) if char_counts else None,
        "mean_token_count": sum(token_counts) / len(token_counts) if token_counts else None,
    }
    output_summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"saved_jsonl: {output_jsonl}")
    print(f"saved_summary: {output_summary_json}")
    print(f"page_count: {page_count}")
    print(f"nonempty_text_page_count: {nonempty_text_page_count}")
    print(f"empty_text_page_count: {empty_text_page_count}")
    print(f"empty_text_page_fraction: {summary['empty_text_page_fraction']}")
    print(f"text_source_counts: {source_counts}")


if __name__ == "__main__":
    main()
