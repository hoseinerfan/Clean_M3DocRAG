#!/usr/bin/env python3

from __future__ import annotations

import argparse
import inspect
import json
import re
import statistics
from collections import Counter, defaultdict
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any


MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a doc_pages JSONL with a PDF-derived markdown field. The exporter "
            "uses original PDF text without OCR or VLM text. The native backend uses "
            "outlines/bookmarks and font-size heading cues; pymupdf4llm uses structured "
            "Markdown conversion with OCR explicitly disabled."
        )
    )
    parser.add_argument("--doc-pages-jsonl", required=True)
    parser.add_argument(
        "--pdf-root",
        action="append",
        default=[],
        help="Root containing original PDFs. Repeatable.",
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument(
        "--backend",
        choices=["native", "pymupdf4llm"],
        default="native",
        help=(
            "PDF-to-Markdown backend. native preserves the existing font/outline exporter; "
            "pymupdf4llm uses page Markdown with OCR controls disabled."
        ),
    )
    parser.add_argument("--max-pages", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument(
        "--body-char-limit",
        type=int,
        default=6000,
        help="Maximum plain page text characters appended after heading lines. Use 0 for headings only.",
    )
    parser.add_argument("--max-heading-lines-per-page", type=int, default=8)
    parser.add_argument("--min-heading-font-ratio", type=float, default=1.15)
    parser.add_argument("--min-heading-font-size", type=float, default=10.5)
    parser.add_argument("--max-heading-words", type=int, default=18)
    parser.add_argument(
        "--pymupdf4llm-keep-header",
        action="store_true",
        help="Retain headers in pymupdf4llm output. Disabled by default to reduce repeated page furniture.",
    )
    parser.add_argument(
        "--pymupdf4llm-keep-footer",
        action="store_true",
        help="Retain footers in pymupdf4llm output. Disabled by default to reduce repeated page furniture.",
    )
    parser.add_argument(
        "--allow-pymupdf4llm-auto-layout",
        action="store_true",
        help=(
            "Allow a PyMuPDF4LLM version other than 0.3.4. Newer releases can initialize "
            "an ONNX layout model on import; this option is intended only for a separately "
            "controlled neural-layout experiment."
        ),
    )
    parser.add_argument(
        "--require-heading-pages",
        action="store_true",
        help="Fail if no page receives a markdown heading line.",
    )
    return parser.parse_args()


def read_jsonl(path: Path, max_pages: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def normalize_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.endswith(".pdf"):
        text = text[:-4]
    return re.sub(r"[^a-z0-9]+", "", text)


def clean_heading(value: str) -> str:
    value = re.sub(r"\s+", " ", value or "").strip()
    value = value.strip(" -_:;.,\t")
    return value


def heading_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def pdf_candidates_for_doc(rows: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    first = rows[0]
    for key in ("doc_id", "doc_name"):
        value = str(first.get(key, "") or "").strip()
        if value:
            values.append(value)
    page_uid = str(first.get("page_uid", "") or "").strip()
    if "_page" in page_uid:
        values.append(page_uid.rsplit("_page", 1)[0])
    return values


def index_pdfs(pdf_roots: list[str]) -> tuple[dict[str, list[Path]], list[Path]]:
    pdf_index: dict[str, list[Path]] = defaultdict(list)
    pdf_paths: list[Path] = []
    for raw_root in pdf_roots:
        root = Path(raw_root)
        if not root.exists():
            raise FileNotFoundError(f"PDF root does not exist: {root}")
        for path in root.rglob("*.pdf"):
            if path.is_file():
                pdf_paths.append(path)
                pdf_index[normalize_key(path.stem)].append(path)
    if not pdf_paths:
        raise ValueError(f"No PDF files found under: {pdf_roots}")
    return pdf_index, pdf_paths


def choose_pdf_for_doc(
    doc_rows: list[dict[str, Any]],
    pdf_index: dict[str, list[Path]],
    pdf_paths: list[Path],
) -> tuple[Path | None, str]:
    candidates = [normalize_key(value) for value in pdf_candidates_for_doc(doc_rows)]
    candidates = [value for value in candidates if value]
    for candidate in candidates:
        matches = pdf_index.get(candidate, [])
        if len(matches) == 1:
            return matches[0], "exact"
        if len(matches) > 1:
            return sorted(matches, key=lambda path: (len(str(path)), str(path)))[0], "exact_ambiguous"

    fuzzy_matches: list[Path] = []
    for candidate in candidates:
        if len(candidate) < 8:
            continue
        for path in pdf_paths:
            key = normalize_key(path.stem)
            if candidate in key or key in candidate:
                fuzzy_matches.append(path)
    unique = sorted(set(fuzzy_matches), key=lambda path: (len(str(path)), str(path)))
    if len(unique) == 1:
        return unique[0], "fuzzy"
    if len(unique) > 1:
        return unique[0], "fuzzy_ambiguous"
    return None, "missing_pdf"


def page_idx(row: dict[str, Any]) -> int:
    return int(row.get("page_idx", row.get("page_id", row.get("page", 0))))


def build_outline_by_page(doc: Any) -> list[list[tuple[int, str]]]:
    page_count = int(doc.page_count)
    outline_by_page: list[list[tuple[int, str]]] = [[] for _ in range(page_count)]
    try:
        toc = doc.get_toc(simple=True)
    except Exception:
        toc = []
    if not toc:
        return outline_by_page

    toc_entries: list[tuple[int, str, int]] = []
    for item in toc:
        if len(item) < 3:
            continue
        level = max(1, int(item[0]))
        title = clean_heading(str(item[1]))
        page = max(0, int(item[2]) - 1)
        if title and page < page_count:
            toc_entries.append((level, title, page))
    toc_entries.sort(key=lambda item: (item[2], item[0]))

    active: dict[int, str] = {}
    cursor = 0
    for current_page in range(page_count):
        while cursor < len(toc_entries) and toc_entries[cursor][2] <= current_page:
            level, title, _ = toc_entries[cursor]
            active[level] = title
            for deeper in [key for key in active if key > level]:
                active.pop(deeper, None)
            cursor += 1
        outline_by_page[current_page] = [
            (level, active[level]) for level in sorted(active) if active.get(level)
        ]
    return outline_by_page


def collect_page_lines(page: Any) -> tuple[list[dict[str, Any]], list[float]]:
    try:
        text_dict = page.get_text("dict")
    except Exception:
        return [], []
    lines: list[dict[str, Any]] = []
    sizes: list[float] = []
    for block in text_dict.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = clean_heading("".join(str(span.get("text", "")) for span in spans))
            if not text:
                continue
            span_sizes = [float(span.get("size", 0.0) or 0.0) for span in spans]
            span_sizes = [size for size in span_sizes if size > 0]
            if not span_sizes:
                continue
            sizes.extend(span_sizes)
            bbox = line.get("bbox", [0, 0, 0, 0])
            lines.append(
                {
                    "text": text,
                    "size": max(span_sizes),
                    "y0": float(bbox[1]) if len(bbox) > 1 else 0.0,
                }
            )
    return lines, sizes


def looks_like_heading(
    text: str,
    size: float,
    median_size: float,
    min_font_ratio: float,
    min_font_size: float,
    max_words: int,
) -> bool:
    words = re.findall(r"[A-Za-z0-9]+", text)
    if not words or len(words) > max_words:
        return False
    if len(text) < 3 or len(text) > 180:
        return False
    if re.fullmatch(r"[\d\s.,:/-]+", text):
        return False
    alpha_chars = [char for char in text if char.isalpha()]
    upper_ratio = (
        sum(1 for char in alpha_chars if char.isupper()) / float(len(alpha_chars))
        if alpha_chars
        else 0.0
    )
    size_pass = size >= max(min_font_size, median_size * min_font_ratio)
    all_caps_pass = upper_ratio >= 0.65 and size >= median_size * 1.05 and len(words) <= 10
    title_case_pass = (
        len(words) <= 8
        and size >= median_size * 1.08
        and sum(1 for word in words if word[:1].isupper()) >= max(1, len(words) // 2)
    )
    return bool(size_pass or all_caps_pass or title_case_pass)


def heading_level(size: float, median_size: float, base_level: int) -> int:
    if size >= median_size * 1.55:
        offset = 0
    elif size >= median_size * 1.30:
        offset = 1
    else:
        offset = 2
    return min(6, max(1, base_level + offset))


def extract_heuristic_headings(
    page: Any,
    outline: list[tuple[int, str]],
    max_heading_lines: int,
    min_font_ratio: float,
    min_font_size: float,
    max_words: int,
) -> list[tuple[int, str]]:
    lines, sizes = collect_page_lines(page)
    if not lines or not sizes:
        return []
    median_size = statistics.median(sizes)
    outline_keys = {heading_key(title) for _, title in outline}
    seen: set[str] = set()
    candidates: list[tuple[float, float, str]] = []
    for line in lines:
        text = str(line["text"])
        key = heading_key(text)
        if not key or key in seen or key in outline_keys:
            continue
        if not looks_like_heading(
            text=text,
            size=float(line["size"]),
            median_size=float(median_size),
            min_font_ratio=min_font_ratio,
            min_font_size=min_font_size,
            max_words=max_words,
        ):
            continue
        seen.add(key)
        candidates.append((float(line["y0"]), float(line["size"]), text))
    candidates.sort(key=lambda item: (item[0], -item[1]))
    base_level = min(6, len(outline) + 1) if outline else 1
    return [
        (heading_level(size, float(median_size), base_level), text)
        for _, size, text in candidates[:max_heading_lines]
    ]


def page_plain_text(page: Any, body_char_limit: int) -> str:
    if body_char_limit <= 0:
        return ""
    try:
        text = page.get_text("text") or ""
    except Exception:
        text = ""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > body_char_limit:
        text = text[:body_char_limit].rstrip()
    return text


def markdown_from_page(
    page: Any,
    outline: list[tuple[int, str]],
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    heuristic = extract_heuristic_headings(
        page=page,
        outline=outline,
        max_heading_lines=int(args.max_heading_lines_per_page),
        min_font_ratio=float(args.min_heading_font_ratio),
        min_font_size=float(args.min_heading_font_size),
        max_words=int(args.max_heading_words),
    )
    lines: list[str] = []
    for level, title in outline:
        lines.append(f"{'#' * min(6, max(1, level))} {title}")
    for level, title in heuristic:
        lines.append(f"{'#' * min(6, max(1, level))} {title}")
    body = page_plain_text(page, int(args.body_char_limit))
    if body:
        if lines:
            lines.append("")
        lines.append(body)
    markdown = "\n".join(lines).strip()
    return markdown, {
        "outline_heading_count": len(outline),
        "heuristic_heading_count": len(heuristic),
        "body_char_count": len(body),
    }


def markdown_heading_count(markdown: str) -> int:
    return sum(
        1 for line in str(markdown or "").splitlines() if MARKDOWN_HEADING_RE.match(line)
    )


def pymupdf4llm_pages(
    pdf_path: Path,
    page_count: int,
    args: argparse.Namespace,
    pymupdf4llm: Any,
) -> dict[int, tuple[str, dict[str, int]]]:
    supported = inspect.signature(pymupdf4llm.to_markdown).parameters
    converter_args: dict[str, Any] = {
        "page_chunks": True,
        "header": bool(args.pymupdf4llm_keep_header),
        "footer": bool(args.pymupdf4llm_keep_footer),
        "show_progress": False,
    }
    converter_args = {key: value for key, value in converter_args.items() if key in supported}
    if "use_ocr" in supported:
        converter_args["use_ocr"] = False
    if "force_ocr" in supported:
        converter_args["force_ocr"] = False
    chunks = pymupdf4llm.to_markdown(str(pdf_path), **converter_args)
    if not isinstance(chunks, list):
        raise TypeError("pymupdf4llm page_chunks=True did not return a page chunk list")
    converted: dict[int, tuple[str, dict[str, int]]] = {}
    for idx, chunk in enumerate(chunks[:page_count]):
        raw_markdown = chunk.get("text", "") if isinstance(chunk, dict) else ""
        markdown = str(raw_markdown or "").strip()
        if int(args.body_char_limit) > 0 and len(markdown) > int(args.body_char_limit):
            markdown = markdown[: int(args.body_char_limit)].rstrip()
        converted[idx] = (
            markdown,
            {
                "outline_heading_count": 0,
                "heuristic_heading_count": markdown_heading_count(markdown),
                "body_char_count": len(markdown),
            },
        )
    return converted


def main() -> None:
    args = parse_args()
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "PyMuPDF is required for PDF markdown export. Install/use an environment "
            "with `import fitz` available."
        ) from exc
    pymupdf4llm = None
    pymupdf4llm_version = ""
    pymupdf4llm_has_use_ocr = False
    if args.backend == "pymupdf4llm":
        try:
            pymupdf4llm_version = importlib_metadata.version("pymupdf4llm")
        except importlib_metadata.PackageNotFoundError as exc:
            raise SystemExit(
                "The pymupdf4llm backend requires `pymupdf4llm`. For the non-OCR "
                "HPC-safe experiment install `pip install --force-reinstall "
                "\"pymupdf4llm==0.3.4\"`."
            ) from exc
        if (
            pymupdf4llm_version != "0.3.4"
            and not bool(args.allow_pymupdf4llm_auto_layout)
        ):
            raise SystemExit(
                "The HPC-safe PyMuPDF4LLM experiment requires `pymupdf4llm==0.3.4`; "
                f"found {pymupdf4llm_version}. Recent releases auto-activate ONNX layout "
                "on import and may fail CPU-affinity setup under SLURM. Install with "
                "`pip install --force-reinstall \"pymupdf4llm==0.3.4\"`, or pass "
                "`--allow-pymupdf4llm-auto-layout` only for a controlled layout-model run."
            )
        try:
            import pymupdf4llm as pymupdf4llm_module  # type: ignore
        except ImportError as exc:
            raise SystemExit(
                "The installed `pymupdf4llm` package could not be imported. Reinstall the "
                "HPC-safe experiment dependency with `pip install --force-reinstall "
                "\"pymupdf4llm==0.3.4\"`."
            ) from exc
        pymupdf4llm = pymupdf4llm_module
        pymupdf4llm_has_use_ocr = "use_ocr" in inspect.signature(
            pymupdf4llm_module.to_markdown
        ).parameters

    if not args.pdf_root:
        raise ValueError("Provide at least one --pdf-root.")

    rows = read_jsonl(Path(args.doc_pages_jsonl), int(args.max_pages))
    pdf_index, pdf_paths = index_pdfs([str(root) for root in args.pdf_root])
    rows_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        doc_id = str(row.get("doc_id", "") or "").strip()
        if not doc_id:
            doc_id = str(row.get("doc_name", "") or "").strip()
        rows_by_doc[doc_id].append(row)

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl_tmp = output_jsonl.with_suffix(output_jsonl.suffix + ".tmp")
    output_summary_json = Path(args.output_summary_json)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json_tmp = output_summary_json.with_suffix(output_summary_json.suffix + ".tmp")

    page_count = 0
    nonempty_markdown_page_count = 0
    heading_page_count = 0
    outline_heading_page_count = 0
    heuristic_heading_page_count = 0
    source_counts: Counter[str] = Counter()
    pdf_match_counts: Counter[str] = Counter()
    unmatched_docs: list[str] = []
    page_out_of_range = 0
    heading_counts: list[int] = []
    backend_errors: list[dict[str, str]] = []

    with output_jsonl_tmp.open("w", encoding="utf-8") as handle:
        for doc_index, (doc_id, doc_rows) in enumerate(sorted(rows_by_doc.items())):
            pdf_path, match_reason = choose_pdf_for_doc(doc_rows, pdf_index, pdf_paths)
            pdf_match_counts[match_reason] += 1
            doc = None
            outline_by_page: list[list[tuple[int, str]]] = []
            alternate_markdown_by_page: dict[int, tuple[str, dict[str, int]]] = {}
            backend_failed = False
            if pdf_path is not None:
                try:
                    doc = fitz.open(str(pdf_path))
                    outline_by_page = build_outline_by_page(doc)
                except Exception:
                    doc = None
                    pdf_match_counts["open_error"] += 1
                if doc is not None and args.backend == "pymupdf4llm":
                    try:
                        alternate_markdown_by_page = pymupdf4llm_pages(
                            pdf_path,
                            int(doc.page_count),
                            args,
                            pymupdf4llm,
                        )
                    except Exception as exc:
                        backend_failed = True
                        backend_errors.append(
                            {
                                "doc_id": doc_id,
                                "error": f"{type(exc).__name__}: {exc}"[:500],
                            }
                        )
            else:
                unmatched_docs.append(doc_id)

            for row in sorted(doc_rows, key=page_idx):
                page_count += 1
                output_row = dict(row)
                source = match_reason
                stats = {
                    "outline_heading_count": 0,
                    "heuristic_heading_count": 0,
                    "body_char_count": 0,
                }
                markdown = ""
                idx = page_idx(row)
                if backend_failed:
                    source = "pymupdf4llm_error"
                elif doc is None:
                    source = "missing_pdf"
                elif idx < 0 or idx >= int(doc.page_count):
                    source = "page_out_of_range"
                    page_out_of_range += 1
                elif args.backend == "pymupdf4llm":
                    markdown, stats = alternate_markdown_by_page.get(idx, ("", stats))
                    if stats["heuristic_heading_count"]:
                        source = "pymupdf4llm_heading"
                    elif markdown:
                        source = "pymupdf4llm_text_only"
                    else:
                        source = "empty"
                else:
                    markdown, stats = markdown_from_page(
                        page=doc[idx],
                        outline=outline_by_page[idx] if idx < len(outline_by_page) else [],
                        args=args,
                    )
                    if stats["outline_heading_count"]:
                        source = "pdf_outline"
                    elif stats["heuristic_heading_count"]:
                        source = "pdf_heading"
                    elif markdown:
                        source = "pdf_text_only"
                    else:
                        source = "empty"

                output_row["markdown"] = markdown
                output_row["markdown_source"] = source
                output_row["markdown_backend"] = str(args.backend)
                output_row["pdf_path"] = str(pdf_path) if pdf_path is not None else ""
                output_row["pdf_outline_heading_count"] = stats["outline_heading_count"]
                output_row["pdf_heuristic_heading_count"] = stats["heuristic_heading_count"]
                source_counts[source] += 1
                heading_count = int(stats["outline_heading_count"]) + int(
                    stats["heuristic_heading_count"]
                )
                heading_counts.append(heading_count)
                if markdown.strip():
                    nonempty_markdown_page_count += 1
                if heading_count > 0:
                    heading_page_count += 1
                if int(stats["outline_heading_count"]) > 0:
                    outline_heading_page_count += 1
                if int(stats["heuristic_heading_count"]) > 0:
                    heuristic_heading_page_count += 1
                handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                if args.progress_every > 0 and page_count % int(args.progress_every) == 0:
                    print(
                        f"exported_pages={page_count} heading_pages={heading_page_count}",
                        flush=True,
                    )

            if doc is not None:
                doc.close()
            if args.progress_every > 0 and (doc_index + 1) % 25 == 0:
                print(f"processed_docs={doc_index + 1}/{len(rows_by_doc)}", flush=True)

    if bool(args.require_heading_pages) and heading_page_count == 0:
        raise ValueError("No PDF-derived markdown heading lines were exported.")

    summary = {
        "doc_pages_jsonl": str(args.doc_pages_jsonl),
        "backend": str(args.backend),
        "pdf_roots": [str(root) for root in args.pdf_root],
        "output_jsonl": str(output_jsonl),
        "doc_count": len(rows_by_doc),
        "page_count": page_count,
        "pdf_file_count": len(pdf_paths),
        "pdf_match_counts": dict(sorted(pdf_match_counts.items())),
        "unmatched_doc_count": len(unmatched_docs),
        "unmatched_doc_sample": unmatched_docs[:25],
        "page_out_of_range_count": page_out_of_range,
        "backend_error_doc_count": len(backend_errors),
        "backend_error_doc_sample": backend_errors[:25],
        "nonempty_markdown_page_count": nonempty_markdown_page_count,
        "heading_page_count": heading_page_count,
        "outline_heading_page_count": outline_heading_page_count,
        "heuristic_heading_page_count": heuristic_heading_page_count,
        "source_counts": dict(sorted(source_counts.items())),
        "mean_heading_count": (
            sum(heading_counts) / float(len(heading_counts)) if heading_counts else 0.0
        ),
        "body_char_limit": int(args.body_char_limit),
        "max_heading_lines_per_page": int(args.max_heading_lines_per_page),
        "min_heading_font_ratio": float(args.min_heading_font_ratio),
        "min_heading_font_size": float(args.min_heading_font_size),
        "max_heading_words": int(args.max_heading_words),
        "pymupdf4llm_use_ocr": False if args.backend == "pymupdf4llm" else None,
        "pymupdf4llm_has_use_ocr_argument": (
            pymupdf4llm_has_use_ocr if args.backend == "pymupdf4llm" else None
        ),
        "pymupdf4llm_version": pymupdf4llm_version if args.backend == "pymupdf4llm" else None,
        "pymupdf4llm_keep_header": bool(args.pymupdf4llm_keep_header),
        "pymupdf4llm_keep_footer": bool(args.pymupdf4llm_keep_footer),
    }
    output_jsonl_tmp.replace(output_jsonl)
    output_summary_json_tmp.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    output_summary_json_tmp.replace(output_summary_json)
    print(f"saved_jsonl: {output_jsonl}")
    print(f"saved_summary: {output_summary_json}")
    print(f"page_count: {page_count}")
    print(f"heading_page_count: {heading_page_count}")
    print(f"unmatched_doc_count: {len(unmatched_docs)}")


if __name__ == "__main__":
    main()
