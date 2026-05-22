#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import ssl
import tarfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote


DEFAULT_ANNOTATIONS_URL = (
    "https://zenodo.org/record/7763635/files/"
    "2023-03-23_DUDE_gt_test_PUBLIC.json?download=1"
)
DEFAULT_BINARIES_URL = (
    "https://huggingface.co/datasets/jordyvl/DUDE_loader/resolve/main/data/"
    "DUDE_train-val-test_binaries.tar.gz"
)

SKIP_DOC_IDS = {
    "nan",
    "ef03364aa27a0987c9870472e312aceb",
    "5c5a5880e6a73b4be2315d506ab0b15b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare DUDE for M3DocRAG-style page retrieval. The converter uses "
            "the public DUDE annotations/PDFs/OCR, renders source PDFs into page "
            "images, and emits MMQA_<split>.jsonl, qids_<split>.jsonl, "
            "gold_pages_<split>.jsonl, and doc_pages_<split>.jsonl."
        )
    )
    parser.add_argument("--hf-repo", default="jordyvl/DUDE_loader")
    parser.add_argument("--hf-config", default="Amazon_original")
    parser.add_argument(
        "--annotations-json",
        default="",
        help=(
            "Optional local DUDE annotation JSON. If omitted, the public Zenodo "
            "annotation file is downloaded into <output-root>/downloads."
        ),
    )
    parser.add_argument(
        "--binaries-url",
        default=DEFAULT_BINARIES_URL,
        help="URL for DUDE_train-val-test_binaries.tar.gz when --data-dir is omitted.",
    )
    parser.add_argument(
        "--data-dir",
        default="",
        help=(
            "Optional extracted DUDE_train-val-test_binaries directory. If omitted, "
            "the tarball is downloaded into <output-root>/downloads and extracted "
            "under <output-root>/raw, avoiding the Hugging Face datasets cache."
        ),
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--source-split",
        default="val",
        help="DUDE source split to read from the loader. The public loader exposes train/val/test.",
    )
    parser.add_argument("--split", default="dev", help="Output split name.")
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--pdf-dpi", type=int, default=144)
    parser.add_argument("--overwrite-rendered-pages", action="store_true")
    parser.add_argument(
        "--answer-page-base",
        default="auto",
        choices=["auto", "0", "1"],
        help=(
            "Base of DUDE answer bbox page numbers. auto picks the base with fewer "
            "out-of-range gold pages after PDF rendering."
        ),
    )
    parser.add_argument(
        "--include-no-gold-page",
        action="store_true",
        help=(
            "Keep rows without answer page boxes, such as not-answerable examples. "
            "By default they are skipped because exact page retrieval has no target."
        ),
    )
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Write manifests even if some PDFs cannot be rendered.",
    )
    return parser.parse_args()


def ssl_context() -> ssl.SSLContext | None:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None


def download_url(url: str, output_path: Path, *, force: bool = False) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if force and output_path.exists():
        output_path.unlink()
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    request = urllib.request.Request(url, headers={"User-Agent": "Clean_M3DocRAG DUDE prepare"})
    bytes_written = 0
    with urllib.request.urlopen(request, context=ssl_context()) as response, tmp_path.open("wb") as out:
        expected_size = response.headers.get("Content-Length")
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            bytes_written += len(chunk)
    if expected_size and bytes_written != int(expected_size):
        tmp_path.unlink(missing_ok=True)
        raise EOFError(
            f"Downloaded {bytes_written} bytes from {url}, expected {expected_size} bytes."
        )
    tmp_path.replace(output_path)
    return output_path


def tar_archive_is_complete(tar_path: Path) -> bool:
    if not tar_path.exists() or tar_path.stat().st_size <= 0:
        return False
    try:
        with tarfile.open(tar_path, "r:*") as tar:
            for _member in tar:
                pass
        return True
    except (tarfile.TarError, EOFError, OSError):
        return False


def safe_extract_tar(tar_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir.resolve()
    with tarfile.open(tar_path, "r:*") as tar:
        for member in tar.getmembers():
            member_path = (output_dir / member.name).resolve()
            if root not in member_path.parents and member_path != root:
                raise ValueError(f"Refusing to extract path outside output dir: {member.name}")
        tar.extractall(output_dir)


def resolve_extracted_binaries_dir(path: Path) -> Path:
    candidates = [
        path,
        path / "DUDE_train-val-test_binaries",
    ]
    candidates.extend(child for child in path.glob("*") if child.is_dir())
    for candidate in candidates:
        if (candidate / "PDF").exists() and (candidate / "OCR").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find DUDE extracted binaries under {path}. "
        "Expected a directory containing PDF/ and OCR/."
    )


def ensure_binaries_dir(args: argparse.Namespace, output_root: Path) -> Path:
    if args.data_dir:
        return resolve_extracted_binaries_dir(Path(args.data_dir))

    download_dir = output_root / "downloads"
    raw_dir = output_root / "raw"
    existing = raw_dir / "DUDE_train-val-test_binaries"
    if existing.exists():
        return resolve_extracted_binaries_dir(existing)

    tar_path = download_dir / "DUDE_train-val-test_binaries.tar.gz"
    print(f"downloading_dude_binaries={args.binaries_url}")
    print(f"download_target={tar_path}")
    force_download = False
    if tar_path.exists() and not tar_archive_is_complete(tar_path):
        print(f"existing_dude_binaries_corrupt={tar_path}")
        force_download = True
    download_url(args.binaries_url, tar_path, force=force_download)
    if not tar_archive_is_complete(tar_path):
        raise EOFError(
            f"Downloaded DUDE binaries archive is incomplete or corrupt: {tar_path}. "
            "Delete it and rerun, or pass --data-dir pointing to an already extracted copy."
        )
    print(f"extracting_dude_binaries={tar_path}")
    safe_extract_tar(tar_path, raw_dir)
    return resolve_extracted_binaries_dir(raw_dir)


def ensure_annotations_json(args: argparse.Namespace, output_root: Path) -> Path:
    if args.annotations_json:
        return Path(args.annotations_json)
    output_path = output_root / "downloads" / "2023-03-23_DUDE_gt_test_PUBLIC.json"
    print(f"downloading_dude_annotations={DEFAULT_ANNOTATIONS_URL}")
    print(f"download_target={output_path}")
    return download_url(DEFAULT_ANNOTATIONS_URL, output_path)


def parse_hf_config(config_name: str) -> tuple[str, str]:
    config = config_name.strip()
    if config.startswith("bin_"):
        config = config[4:]
    parts = config.split("_", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Unsupported DUDE hf-config {config_name!r}. Expected e.g. Amazon_original."
        )
    ocr_engine, ocr_format = parts
    if ocr_engine not in {"Azure", "Amazon", "Tesseract"}:
        raise ValueError(f"Unsupported DUDE OCR engine in hf-config: {ocr_engine}")
    if ocr_format not in {"original", "due"}:
        raise ValueError(f"Unsupported DUDE OCR format in hf-config: {ocr_format}")
    return ocr_engine, ocr_format


def parse_dude_bbox(value: object) -> list[dict[str, int]]:
    if value in (None, [], [[]]):
        return []
    boxes = value
    if isinstance(boxes, list) and boxes and isinstance(boxes[0], list):
        boxes = boxes[0]
    if isinstance(boxes, dict):
        boxes = [boxes]
    if not isinstance(boxes, list):
        return []

    out: list[dict[str, int]] = []
    for box in boxes:
        if not isinstance(box, dict):
            continue
        parsed: dict[str, int] = {}
        for key in ("left", "top", "width", "height", "page"):
            try:
                parsed[key] = int(box[key])
            except (KeyError, TypeError, ValueError):
                parsed = {}
                break
        if parsed:
            out.append(parsed)
    return out


def load_dude_rows(args: argparse.Namespace, output_root: Path) -> list[dict]:
    binaries_dir = ensure_binaries_dir(args, output_root)
    annotations_path = ensure_annotations_json(args, output_root)
    ocr_engine, ocr_format = parse_hf_config(args.hf_config)

    with annotations_path.open("r", encoding="utf-8") as handle:
        annotations = json.load(handle)
    if isinstance(annotations, dict):
        annotations = annotations.get("data", annotations)
    if not isinstance(annotations, list):
        raise TypeError(f"DUDE annotations must be a list or dict with data list: {annotations_path}")

    rows: list[dict] = []
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        row = dict(annotation)
        doc_id = str(row.get("docId", "") or "").strip()
        if doc_id in SKIP_DOC_IDS:
            continue
        data_split = str(row.get("data_split", "") or "")
        if args.source_split in {"train", "val"}:
            if data_split != args.source_split:
                continue
        elif args.source_split not in data_split:
            continue

        row["data_split"] = args.source_split
        row["answers_page_bounding_boxes"] = parse_dude_bbox(
            row.get("answers_page_bounding_boxes", [])
        )
        if "answers" not in row:
            row["answers"] = None
        if "answers_variants" not in row:
            row["answers_variants"] = None
        if "answer_type" not in row:
            row["answer_type"] = None

        row["document"] = str(binaries_dir / "PDF" / args.source_split / f"{doc_id}.pdf")
        row["OCR"] = str(binaries_dir / "OCR" / ocr_engine / f"{doc_id}_{ocr_format}.json")
        rows.append(row)
    return rows


def get_source_rows(dataset, source_split: str) -> list[dict]:
    if hasattr(dataset, "keys"):
        split_names = list(dataset.keys())
        split = source_split
        if split not in dataset:
            aliases = {"dev": "val", "validation": "val", "val": "dev"}
            split = aliases.get(source_split, source_split)
        if split not in dataset:
            raise KeyError(f"Split {source_split!r} not found. Available splits: {split_names}")
        rows = list(dataset[split])
    else:
        rows = list(dataset)
    return [dict(row) for row in rows if str(row.get("docId", "")).strip() not in SKIP_DOC_IDS]


def filter_rows(rows: list[dict], max_docs: int, max_questions: int) -> list[dict]:
    kept_rows: list[dict] = []
    kept_doc_ids: list[str] = []
    kept_doc_set: set[str] = set()
    for row in rows:
        doc_id = safe_doc_id(row.get("docId", ""))
        if not doc_id:
            continue
        if max_docs > 0 and doc_id not in kept_doc_set and len(kept_doc_ids) >= max_docs:
            continue
        if doc_id not in kept_doc_set:
            kept_doc_ids.append(doc_id)
            kept_doc_set.add(doc_id)
        kept_rows.append(row)
        if max_questions > 0 and len(kept_rows) >= max_questions:
            break
    return kept_rows


def safe_doc_id(value: object) -> str:
    return quote(str(value or "").strip(), safe="._-")


def safe_qid(row: dict, row_index: int) -> str:
    raw = str(row.get("questionId", "") or "").strip()
    if not raw:
        raw = f"{safe_doc_id(row.get('docId', ''))}__q{row_index:06d}"
    return quote(raw, safe="._-")


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = " ".join(str(item) for item in value if item is not None)
    elif isinstance(value, dict):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = str(value).replace("\x0c", " ").replace("\u0000", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_answer(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [normalize_text(item) for item in value if normalize_text(item)]
    return [normalize_text(value)]


def rel_or_abs(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def path_from_row(value: object, field_name: str) -> Path:
    if isinstance(value, (bytes, bytearray)):
        raise TypeError(
            f"DUDE field {field_name} is binary. Use a non-binary loader config such as Amazon_original."
        )
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Missing DUDE {field_name} path.")
    return Path(text)


def unique_doc_paths(rows: list[dict]) -> dict[str, dict[str, Path]]:
    docs: dict[str, dict[str, Path]] = {}
    for row in rows:
        doc_id = safe_doc_id(row.get("docId", ""))
        if not doc_id or doc_id in docs:
            continue
        docs[doc_id] = {
            "pdf_path": path_from_row(row.get("document"), "document"),
            "ocr_path": path_from_row(row.get("OCR"), "OCR"),
        }
    return docs


def existing_rendered_pages(page_dir: Path) -> list[tuple[int, Path]]:
    records = []
    if not page_dir.exists():
        return records
    for image_path in sorted(page_dir.glob("*.jpg")):
        try:
            page_idx = int(image_path.stem)
        except ValueError:
            continue
        records.append((page_idx, image_path))
    return sorted(records, key=lambda item: item[0])


def render_pdf_pages(
    *,
    pdf_path: Path,
    page_dir: Path,
    overwrite: bool,
    dpi: int,
) -> list[tuple[int, Path]]:
    existing = existing_rendered_pages(page_dir)
    if existing and not overwrite:
        return existing

    from pdf2image import convert_from_path, pdfinfo_from_path

    page_dir.mkdir(parents=True, exist_ok=True)
    page_count = int(pdfinfo_from_path(pdf_path).get("Pages", 0))
    rendered = []
    for page_number in range(1, page_count + 1):
        page_idx = page_number - 1
        out_path = page_dir / f"{page_idx}.jpg"
        if out_path.exists() and not overwrite:
            rendered.append((page_idx, out_path))
            continue
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=page_number,
            last_page=page_number,
        )
        if not images:
            continue
        images[0].convert("RGB").save(out_path, quality=95)
        rendered.append((page_idx, out_path))
    return rendered


def collect_text_strings(obj: object) -> list[str]:
    strings: list[str] = []
    if isinstance(obj, dict):
        direct_values = []
        for key in ("text", "Text", "content", "Content", "value", "Value", "DetectedText"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                direct_values.append(value)
        if direct_values:
            return [normalize_text(value) for value in direct_values if normalize_text(value)]
        for value in obj.values():
            strings.extend(collect_text_strings(value))
    elif isinstance(obj, list):
        for item in obj:
            strings.extend(collect_text_strings(item))
    elif isinstance(obj, str) and obj.strip():
        strings.append(normalize_text(obj))
    return [text for text in strings if text]


def page_idx_from_value(value: object, *, page_count: int, fallback_idx: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return fallback_idx
    if number == 0:
        return 0
    if 1 <= number <= page_count:
        return number - 1
    return number


def infer_page_idx(page_obj: object, fallback_idx: int, page_count: int) -> int:
    if isinstance(page_obj, dict):
        for key in ("page", "Page", "page_number", "pageNumber", "page_num"):
            if key in page_obj:
                return page_idx_from_value(page_obj[key], page_count=page_count, fallback_idx=fallback_idx)
    return fallback_idx


def text_from_page_object(page_obj: object) -> str:
    if isinstance(page_obj, dict):
        for key in ("text", "Text", "content", "Content"):
            value = page_obj.get(key)
            if isinstance(value, str) and value.strip():
                return normalize_text(value)
        for key in ("lines", "Lines", "paragraphs", "Paragraphs", "words", "Words"):
            if key in page_obj:
                text = " ".join(collect_text_strings(page_obj[key]))
                if text.strip():
                    return normalize_text(text)
    return normalize_text(" ".join(collect_text_strings(page_obj)))


def group_texts_by_page(obj: object, by_page: dict[int, list[str]], page_count: int) -> None:
    if isinstance(obj, dict):
        page_value = None
        for key in ("page", "Page", "page_number", "pageNumber", "page_num"):
            if key in obj:
                page_value = obj[key]
                break
        if page_value is not None:
            page_idx = page_idx_from_value(page_value, page_count=page_count, fallback_idx=0)
            text = text_from_page_object(obj)
            if text:
                by_page[page_idx].append(text)
            return
        for value in obj.values():
            group_texts_by_page(value, by_page, page_count)
    elif isinstance(obj, list):
        for item in obj:
            group_texts_by_page(item, by_page, page_count)


def extract_ocr_page_texts(ocr_path: Path, page_count: int) -> dict[int, str]:
    if not ocr_path.exists():
        return {}
    with ocr_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict):
        blocks = data.get("Blocks") or data.get("blocks")
        if isinstance(blocks, list):
            line_texts: dict[int, list[str]] = defaultdict(list)
            word_texts: dict[int, list[str]] = defaultdict(list)
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                text = normalize_text(block.get("Text") or block.get("text"))
                if not text:
                    continue
                page_idx = page_idx_from_value(block.get("Page", 1), page_count=page_count, fallback_idx=0)
                block_type = str(block.get("BlockType", "")).upper()
                if block_type == "LINE":
                    line_texts[page_idx].append(text)
                elif block_type == "WORD":
                    word_texts[page_idx].append(text)
            chosen = line_texts or word_texts
            if chosen:
                return {idx: normalize_text(" ".join(parts)) for idx, parts in chosen.items()}

        for key in ("pages", "Pages", "readResults", "recognitionResults"):
            pages = data.get(key)
            if isinstance(pages, list):
                out = {}
                for fallback_idx, page_obj in enumerate(pages):
                    page_idx = infer_page_idx(page_obj, fallback_idx, page_count)
                    text = text_from_page_object(page_obj)
                    if text:
                        out[page_idx] = text
                if out:
                    return out

        analyze_result = data.get("analyzeResult")
        if isinstance(analyze_result, dict):
            for key in ("readResults", "pages"):
                pages = analyze_result.get(key)
                if isinstance(pages, list):
                    out = {}
                    for fallback_idx, page_obj in enumerate(pages):
                        page_idx = infer_page_idx(page_obj, fallback_idx, page_count)
                        text = text_from_page_object(page_obj)
                        if text:
                            out[page_idx] = text
                    if out:
                        return out

    if isinstance(data, list) and page_count and len(data) <= page_count + 2:
        out = {}
        for fallback_idx, page_obj in enumerate(data):
            page_idx = infer_page_idx(page_obj, fallback_idx, page_count)
            text = text_from_page_object(page_obj)
            if text:
                out[page_idx] = text
        if out:
            return out

    grouped: dict[int, list[str]] = defaultdict(list)
    group_texts_by_page(data, grouped, page_count)
    return {idx: normalize_text(" ".join(parts)) for idx, parts in grouped.items() if parts}


def collect_answer_pages(value: object) -> list[int]:
    pages: list[int] = []
    if value is None:
        return pages
    if isinstance(value, dict):
        if "page" in value:
            try:
                pages.append(int(value["page"]))
            except (TypeError, ValueError):
                pass
        for nested in value.values():
            pages.extend(collect_answer_pages(nested))
    elif isinstance(value, list):
        for item in value:
            pages.extend(collect_answer_pages(item))
    return pages


def unique_preserve_order(values: list[int]) -> list[int]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def choose_answer_page_base(rows: list[dict], page_counts: dict[str, int], requested: str) -> tuple[int, dict[str, int]]:
    raw_pages_by_doc = []
    for row in rows:
        doc_id = safe_doc_id(row.get("docId", ""))
        for page_number in unique_preserve_order(collect_answer_pages(row.get("answers_page_bounding_boxes"))):
            raw_pages_by_doc.append((doc_id, page_number))

    missing_by_base: dict[str, int] = {}
    for base in (0, 1):
        missing = 0
        for doc_id, page_number in raw_pages_by_doc:
            page_idx = page_number - base
            page_count = page_counts.get(doc_id, 0)
            if page_idx < 0 or page_idx >= page_count:
                missing += 1
        missing_by_base[str(base)] = missing

    if requested in {"0", "1"}:
        return int(requested), missing_by_base
    if missing_by_base["1"] < missing_by_base["0"]:
        return 1, missing_by_base
    return 0, missing_by_base


def build_doc_pages(
    *,
    docs: dict[str, dict[str, Path]],
    output_root: Path,
    split: str,
    overwrite_rendered_pages: bool,
    pdf_dpi: int,
    allow_missing_images: bool,
) -> tuple[dict[str, int], list[str]]:
    doc_pages_path = output_root / f"doc_pages_{split}.jsonl"
    page_counts: dict[str, int] = {}
    missing_docs: list[str] = []
    with doc_pages_path.open("w", encoding="utf-8") as out:
        for doc_id, paths in sorted(docs.items()):
            pdf_path = paths["pdf_path"]
            ocr_path = paths["ocr_path"]
            if not pdf_path.exists():
                missing_docs.append(doc_id)
                if not allow_missing_images:
                    continue
            rendered_dir = output_root / f"pages_{split}" / doc_id
            page_records = (
                render_pdf_pages(
                    pdf_path=pdf_path,
                    page_dir=rendered_dir,
                    overwrite=overwrite_rendered_pages,
                    dpi=pdf_dpi,
                )
                if pdf_path.exists()
                else []
            )
            if not page_records:
                missing_docs.append(doc_id)
            page_counts[doc_id] = len(page_records)
            ocr_texts = extract_ocr_page_texts(ocr_path, len(page_records))
            for page_idx, image_path in page_records:
                text = ocr_texts.get(page_idx, "")
                out.write(
                    json.dumps(
                        {
                            "doc_id": doc_id,
                            "doc_name": doc_id,
                            "file_name": pdf_path.name,
                            "pdf_path": str(pdf_path),
                            "ocr_path": str(ocr_path),
                            "page_idx": page_idx,
                            "page_number": page_idx + 1,
                            "page_uid": f"{doc_id}_page{page_idx}",
                            "image_path": rel_or_abs(image_path, output_root),
                            "text": text,
                            "ocr_text": text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    if missing_docs and not allow_missing_images:
        raise FileNotFoundError(f"Missing/unrendered {len(missing_docs)} DUDE PDFs; first missing: {missing_docs[:10]}")
    return page_counts, missing_docs


def write_converted_rows(
    *,
    rows: list[dict],
    output_root: Path,
    split: str,
    source_split: str,
    hf_config: str,
    page_counts: dict[str, int],
    answer_page_base: int,
    include_no_gold_page: bool,
) -> dict[str, object]:
    mmqa_path = output_root / f"MMQA_{split}.jsonl"
    qids_path = output_root / f"qids_{split}.jsonl"
    gold_pages_path = output_root / f"gold_pages_{split}.jsonl"
    doc_ids_path = output_root / f"{split}_doc_ids.json"
    doc_id_map_path = output_root / "doc_id_map.json"

    used_qids: set[str] = set()
    kept_doc_ids: list[str] = []
    doc_id_map: dict[str, str] = {}
    kept_count = 0
    skipped_no_gold_page_count = 0
    missing_gold_pages: list[dict] = []
    answer_type_counts_all: Counter[str] = Counter()
    answer_type_counts_kept: Counter[str] = Counter()

    with mmqa_path.open("w", encoding="utf-8") as mmqa_out, qids_path.open(
        "w", encoding="utf-8"
    ) as qids_out, gold_pages_path.open("w", encoding="utf-8") as gold_out:
        for row_index, row in enumerate(rows):
            doc_id = safe_doc_id(row.get("docId", ""))
            if not doc_id:
                continue
            answer_type = str(row.get("answer_type", "") or "").strip() or "UNKNOWN"
            answer_type_counts_all[answer_type] += 1
            raw_answer_pages = unique_preserve_order(collect_answer_pages(row.get("answers_page_bounding_boxes")))
            gold_page_ids = [page_number - answer_page_base for page_number in raw_answer_pages]
            valid_gold_page_ids = [
                page_idx
                for page_idx in gold_page_ids
                if page_idx >= 0 and page_idx < page_counts.get(doc_id, 0)
            ]
            if not valid_gold_page_ids and not include_no_gold_page:
                skipped_no_gold_page_count += 1
                continue

            if doc_id not in doc_id_map:
                kept_doc_ids.append(doc_id)
                doc_id_map[doc_id] = str(row.get("document", ""))

            qid_base = safe_qid(row, row_index)
            qid = qid_base
            suffix = 1
            while qid in used_qids:
                suffix += 1
                qid = f"{qid_base}__dup{suffix}"
            used_qids.add(qid)

            gold_page_uids = [f"{doc_id}_page{page_idx}" for page_idx in valid_gold_page_ids]
            for page_idx in gold_page_ids:
                if page_idx < 0 or page_idx >= page_counts.get(doc_id, 0):
                    missing_gold_pages.append({"qid": qid, "doc_id": doc_id, "page_idx": page_idx})

            supporting_context = [
                {
                    "doc_id": doc_id,
                    "doc_part": answer_type,
                    "page_idx": page_idx,
                    "page_id": page_idx,
                    "source_page_number": page_idx + 1,
                }
                for page_idx in valid_gold_page_ids
            ]
            if not supporting_context:
                supporting_context = [{"doc_id": doc_id, "doc_part": answer_type}]

            answers = normalize_answer(row.get("answers"))
            if not answers:
                answers = [""]
            mmqa_row = {
                "qid": qid,
                "question": str(row.get("question", "") or "").strip(),
                "answers": [{"answer": answer, "modality": "document"} for answer in answers],
                "metadata": {
                    "type": "DUDE",
                    "source": "DUDE",
                    "source_split": source_split,
                    "hf_config": hf_config,
                    "source_question_id": str(row.get("questionId", "") or ""),
                    "doc_id": doc_id,
                    "answer_type": answer_type,
                    "answers_variants": normalize_answer(row.get("answers_variants")),
                    "raw_answer_page_numbers": raw_answer_pages,
                    "answer_page_base": answer_page_base,
                    "gold_page_ids": valid_gold_page_ids,
                    "gold_page_uids": gold_page_uids,
                    "page_count": page_counts.get(doc_id, 0),
                },
                "supporting_context": supporting_context,
            }
            mmqa_out.write(json.dumps(mmqa_row, ensure_ascii=False) + "\n")
            qids_out.write(json.dumps({"qid": qid}, ensure_ascii=False) + "\n")
            gold_out.write(
                json.dumps(
                    {
                        "qid": qid,
                        "doc_id": doc_id,
                        "answer_type": answer_type,
                        "raw_answer_page_numbers": raw_answer_pages,
                        "answer_page_base": answer_page_base,
                        "gold_page_ids": valid_gold_page_ids,
                        "gold_page_uids": gold_page_uids,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            answer_type_counts_kept[answer_type] += 1
            kept_count += 1

    doc_ids_path.write_text(json.dumps(sorted(kept_doc_ids), indent=2) + "\n", encoding="utf-8")
    doc_id_map_path.write_text(json.dumps(doc_id_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "qa_count": kept_count,
        "skipped_no_gold_page_count": skipped_no_gold_page_count,
        "missing_gold_page_count": len(missing_gold_pages),
        "missing_gold_pages_sample": missing_gold_pages[:20],
        "answer_type_counts_all": dict(sorted(answer_type_counts_all.items())),
        "answer_type_counts_kept": dict(sorted(answer_type_counts_kept.items())),
        "doc_ids": sorted(kept_doc_ids),
    }


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    source_rows = filter_rows(
        load_dude_rows(args, output_root),
        max_docs=args.max_docs,
        max_questions=args.max_questions,
    )
    if not source_rows:
        raise ValueError(
            f"No DUDE rows loaded from source split {args.source_split!r}. "
            "Try --source-split train or verify the loader/data_dir."
        )

    if args.include_no_gold_page:
        rows = source_rows
        pre_skipped_no_gold_page_count = 0
    else:
        rows = [row for row in source_rows if collect_answer_pages(row.get("answers_page_bounding_boxes"))]
        pre_skipped_no_gold_page_count = len(source_rows) - len(rows)
    if not rows:
        raise ValueError(
            "No DUDE rows with answer page boxes remain. Use --include-no-gold-page "
            "if you want to keep not-answerable/no-page-label rows."
        )

    docs = unique_doc_paths(rows)
    page_counts, missing_doc_ids = build_doc_pages(
        docs=docs,
        output_root=output_root,
        split=args.split,
        overwrite_rendered_pages=args.overwrite_rendered_pages,
        pdf_dpi=args.pdf_dpi,
        allow_missing_images=args.allow_missing_images,
    )
    answer_page_base, missing_by_base = choose_answer_page_base(rows, page_counts, args.answer_page_base)
    conversion_summary = write_converted_rows(
        rows=rows,
        output_root=output_root,
        split=args.split,
        source_split=args.source_split,
        hf_config=args.hf_config,
        page_counts=page_counts,
        answer_page_base=answer_page_base,
        include_no_gold_page=args.include_no_gold_page,
    )

    doc_ids = conversion_summary.pop("doc_ids")
    total_skipped_no_gold_page_count = (
        int(pre_skipped_no_gold_page_count)
        + int(conversion_summary.get("skipped_no_gold_page_count", 0))
    )
    page_count = sum(page_counts.get(doc_id, 0) for doc_id in doc_ids)
    summary = {
        "prepared_output_root": str(output_root),
        "hf_repo": args.hf_repo,
        "hf_config": args.hf_config,
        "data_dir": args.data_dir,
        "annotations_json": args.annotations_json,
        "binaries_url": args.binaries_url,
        "source_split": args.source_split,
        "split": args.split,
        "answer_page_base": answer_page_base,
        "answer_page_base_missing_counts": missing_by_base,
        "include_no_gold_page": bool(args.include_no_gold_page),
        "source_row_count": len(source_rows),
        "candidate_row_count": len(rows),
        "doc_count": len(doc_ids),
        "page_count": page_count,
        "missing_doc_count": len(missing_doc_ids),
        "missing_doc_ids_sample": missing_doc_ids[:20],
        **conversion_summary,
    }
    summary["skipped_no_gold_page_count"] = total_skipped_no_gold_page_count
    summary_path = output_root / f"prepare_{args.split}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"prepared_output_root={output_root}")
    print(f"hf_repo={args.hf_repo}")
    print(f"hf_config={args.hf_config}")
    print(f"source_split={args.source_split}")
    print(f"doc_count={summary['doc_count']}")
    print(f"page_count={summary['page_count']}")
    print(f"qa_count={summary['qa_count']}")
    print(f"skipped_no_gold_page_count={summary['skipped_no_gold_page_count']}")
    print(f"answer_page_base={answer_page_base}")
    print(f"answer_page_base_missing_counts={missing_by_base}")
    print(f"answer_type_counts_kept={summary['answer_type_counts_kept']}")
    print(f"saved_summary={summary_path}")


if __name__ == "__main__":
    main()
