#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
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
    parser.add_argument(
        "--image-root",
        default="",
        help=(
            "Optional root containing page images. Used with --ocr-image when manifest "
            "and PDF text are empty."
        ),
    )
    parser.add_argument(
        "--ocr-image",
        action="store_true",
        help="Run OCR on page images when no manifest/PDF text is available.",
    )
    parser.add_argument("--ocr-bin", default="tesseract")
    parser.add_argument("--ocr-lang", default="eng")
    parser.add_argument("--ocr-psm", default="")
    parser.add_argument("--ocr-timeout", type=int, default=120)
    parser.add_argument(
        "--ocr-continue-on-error",
        action="store_true",
        help="Keep exporting if OCR fails for a page; failed pages get empty text.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--max-pages", type=int, default=0)
    parser.add_argument(
        "--require-nonempty",
        action="store_true",
        help="Fail if no page has non-empty extracted text.",
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-summary-json", required=True)
    return parser.parse_args()


def read_jsonl(path: Path, max_pages: int, num_shards: int, shard_index: int) -> list[dict]:
    if num_shards <= 0:
        raise ValueError(f"--num-shards must be positive, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"--shard-index must be in [0, {num_shards}), got {shard_index}"
        )
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            if row_index % num_shards != shard_index:
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


def candidate_image_paths(row: dict, image_root: Path) -> list[Path]:
    candidates: list[Path] = []

    def add(value: object) -> None:
        text = str(value or "").strip()
        if not text:
            return
        path = Path(text)
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.append(image_root / path)

    add(row.get("image_path"))
    add(row.get("page_image_path"))
    add(row.get("image"))
    add(row.get("file_name"))

    page_uid = str(row.get("page_uid", "") or "").strip()
    doc_id = str(row.get("doc_id", "") or "").strip()
    page_idx = str(row.get("page_idx", row.get("page_id", "")) or "").strip()
    if doc_id and page_idx:
        candidates.append(image_root / "pages_dev" / doc_id / f"{page_idx}.jpg")
        candidates.append(image_root / "pages" / doc_id / f"{page_idx}.jpg")
    if page_uid:
        candidates.append(image_root / f"{page_uid}.jpg")

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def resolve_image_path(row: dict, image_root: Path) -> Path | None:
    for candidate in candidate_image_paths(row, image_root):
        if candidate.exists():
            return candidate
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


def extract_image_ocr_text(
    *,
    ocr_bin: str,
    image_path: Path,
    ocr_lang: str,
    ocr_psm: str,
    ocr_timeout: int,
) -> str:
    command = [ocr_bin, str(image_path), "stdout"]
    if ocr_lang:
        command.extend(["-l", ocr_lang])
    if ocr_psm:
        command.extend(["--psm", str(ocr_psm)])
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=int(ocr_timeout),
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
    image_root: Path | None,
    ocr_image: bool,
    ocr_bin: str,
    ocr_lang: str,
    ocr_psm: str,
    ocr_timeout: int,
    ocr_continue_on_error: bool,
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

    if not parts and ocr_image and image_root is not None:
        image_path = resolve_image_path(row, image_root)
        if image_path is not None:
            try:
                text = extract_image_ocr_text(
                    ocr_bin=ocr_bin,
                    image_path=image_path,
                    ocr_lang=ocr_lang,
                    ocr_psm=ocr_psm,
                    ocr_timeout=ocr_timeout,
                )
            except Exception as exc:
                if not ocr_continue_on_error:
                    raise RuntimeError(f"OCR failed for {image_path}") from exc
                text = ""
            if text:
                parts.append(text)
                source = "ocr"

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
    image_root = Path(args.image_root) if args.image_root else None

    rows = read_jsonl(
        Path(args.doc_pages_jsonl),
        int(args.max_pages),
        int(args.num_shards),
        int(args.shard_index),
    )
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json = Path(args.output_summary_json)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)

    pdf_cache: dict[str, Path | None] = {}
    page_count = 0
    nonempty_text_page_count = 0
    empty_text_page_count = 0
    source_counts = {"manifest": 0, "pdf": 0, "ocr": 0, "empty": 0}
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
                image_root=image_root,
                ocr_image=bool(args.ocr_image),
                ocr_bin=args.ocr_bin,
                ocr_lang=args.ocr_lang,
                ocr_psm=args.ocr_psm,
                ocr_timeout=int(args.ocr_timeout),
                ocr_continue_on_error=bool(args.ocr_continue_on_error),
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
            if args.progress_every > 0 and page_count % int(args.progress_every) == 0:
                print(
                    f"exported_pages={page_count} nonempty={nonempty_text_page_count} "
                    f"empty={empty_text_page_count}",
                    file=sys.stderr,
                    flush=True,
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
        "image_root": args.image_root,
        "ocr_image": bool(args.ocr_image),
        "ocr_bin": args.ocr_bin,
        "ocr_lang": args.ocr_lang,
        "ocr_psm": args.ocr_psm,
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
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
