#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


QID_KEYS = ["qid", "question_id", "questionId", "questionID", "id"]
QUESTION_KEYS = ["question", "query"]
ANSWER_KEYS = ["answers", "answer", "answer_text", "answerText", "labels"]
DOC_ID_KEYS = [
    "doc_id",
    "docid",
    "document_id",
    "documentId",
    "ucsf_document_id",
    "image_id",
    "doc_name",
    "document",
]
PAGE_LIST_KEYS = [
    "page_list",
    "page_paths",
    "image_paths",
    "image_name",
    "image_names",
    "images",
    "pages",
    "page_images",
    "document_pages",
    "doc_pages",
]
GOLD_PAGE_KEYS = [
    "answer_page_idx",
    "answer_page_idxs",
    "answer_page_indices",
    "answer_page",
    "answer_pages",
    "answer_page_id",
    "answer_page_ids",
    "ans_page_idx",
    "ans_page_list",
    "page_idx",
    "page_indices",
    "evidence_page_idx",
    "evidence_pages",
    "gold_page_ids",
    "gold_pages",
]
PAGE_TEXT_KEYS = [
    "page_text_list",
    "page_texts",
    "ocr_texts",
    "ocr_tokens",
    "texts",
    "document_text",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert MP-DocVQA into the M3DocRAG external page-retrieval format. "
            "The output matches the MMDocIR/MMLongBench manifest contract."
        )
    )
    parser.add_argument("--input-root", required=True, help="Root containing MP-DocVQA annotations/images.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--annotation-file",
        action="append",
        default=[],
        help=(
            "Annotation JSON/JSONL/NPY file. Repeatable. Relative paths are resolved "
            "under --input-root. If omitted, common split filenames are searched."
        ),
    )
    parser.add_argument("--image-root", default="", help="Root containing page images. Defaults to --input-root.")
    parser.add_argument("--source-split", default="val", help="Source split filename stem, commonly train/val/test.")
    parser.add_argument("--split", default="dev", help="Output split name, e.g. dev.")
    parser.add_argument(
        "--gold-page-base",
        default="auto",
        choices=["auto", "zero", "one"],
        help="Whether gold answer page indices are zero-based or one-based.",
    )
    parser.add_argument(
        "--scan-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scan --image-root and group images by document id when annotation rows do not list every page.",
    )
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--store-raw-row", action="store_true", help="Store compact raw source row in metadata.")
    return parser.parse_args()


def to_builtin(value: Any) -> Any:
    if hasattr(value, "item") and not isinstance(value, (dict, list, tuple, str, bytes)):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return to_builtin(value.tolist())
        except Exception:
            return str(value)
    return value


def load_annotation_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return [to_builtin(row) for row in rows if isinstance(to_builtin(row), dict)]
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload = to_builtin(payload)
        if isinstance(payload, dict):
            for key in ["data", "questions", "annotations", "examples", "items"]:
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            raise TypeError(f"JSON annotation must be a list or dict with data/questions: {path}")
        return [row for row in payload if isinstance(row, dict)]
    if suffix == ".npy":
        import numpy as np

        payload = np.load(path, allow_pickle=True)
        payload = to_builtin(payload)
        if isinstance(payload, dict):
            for key in ["data", "questions", "annotations", "examples", "items"]:
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            raise TypeError(f"NPY annotation must contain a list/dict of rows: {path}")
        return [row for row in payload if isinstance(row, dict)]
    raise ValueError(f"Unsupported annotation file type: {path}")


def existing_annotation_files(input_root: Path, source_split: str, explicit: list[str]) -> list[Path]:
    if explicit:
        out = []
        for raw in explicit:
            path = Path(raw)
            if not path.is_absolute():
                path = input_root / path
            if not path.exists():
                raise FileNotFoundError(path)
            out.append(path)
        return out
    names = [
        f"{source_split}.jsonl",
        f"{source_split}.json",
        f"mpdocvqa_{source_split}.jsonl",
        f"mpdocvqa_{source_split}.json",
        f"MPDocVQA_{source_split}.jsonl",
        f"MPDocVQA_{source_split}.json",
        f"imdb_{source_split}.npy",
        f"imdb_{source_split}.json",
        f"imdb_{source_split}.jsonl",
    ]
    matches = []
    for name in names:
        candidate = input_root / name
        if candidate.exists():
            matches.append(candidate)
    if matches:
        return matches
    recursive = []
    for pattern in [f"*{source_split}*.jsonl", f"*{source_split}*.json", f"*{source_split}*.npy"]:
        recursive.extend(sorted(input_root.rglob(pattern)))
    if recursive:
        return recursive[:1]
    raise FileNotFoundError(
        f"Could not find annotation file for split {source_split!r} under {input_root}. "
        "Pass --annotation-file explicitly."
    )


def first_value(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def normalize_list(value: Any) -> list[Any]:
    if value is None:
        return []
    value = to_builtin(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                return normalize_list(parsed)
            except json.JSONDecodeError:
                pass
        return [part.strip() for part in re.split(r"[,;]", text) if part.strip()]
    if isinstance(value, dict):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [item for item in value if item not in (None, "")]
    return [value]


def normalize_answer(value: Any) -> str:
    values = normalize_list(value)
    if not values:
        return ""
    first = values[0]
    if isinstance(first, dict):
        for key in ["answer", "text", "label", "value"]:
            if key in first:
                return normalize_answer(first[key])
        return json.dumps(first, ensure_ascii=False, sort_keys=True)
    return str(first).strip()


def safe_doc_id(value: Any) -> str:
    text = str(value or "unknown_doc").strip()
    return quote(text, safe="._-")


def safe_qid(value: Any, row_index: int) -> str:
    text = str(value or f"row{row_index:06d}").strip()
    return quote(f"mpdocvqa__{text}", safe="._-")


def image_file_sort_key(path: Path) -> tuple[str, list[int], str]:
    nums = [int(value) for value in re.findall(r"\d+", path.stem)]
    return (str(path.parent), nums, path.name)


def infer_doc_id_from_image(path: Path, image_root: Path) -> str:
    try:
        rel = path.relative_to(image_root)
    except ValueError:
        rel = path
    if len(rel.parts) > 1:
        return safe_doc_id(rel.parts[-2])
    stem = path.stem
    stem = re.sub(r"(?i)(?:^|[_-])page[_-]?\d+$", "", stem)
    stem = re.sub(r"[_-]\d+$", "", stem)
    return safe_doc_id(stem or path.stem)


def scan_image_root(image_root: Path) -> dict[str, list[Path]]:
    by_doc: dict[str, list[Path]] = defaultdict(list)
    if not image_root.exists():
        return {}
    for path in image_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            by_doc[infer_doc_id_from_image(path, image_root)].append(path)
    return {doc_id: sorted(paths, key=image_file_sort_key) for doc_id, paths in by_doc.items()}


def extract_path_from_page_item(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ["image_path", "path", "file_name", "filename", "image", "page_image", "name"]:
            if item.get(key):
                return str(item[key])
    return str(item)


def extract_text_from_page_item(item: Any) -> str:
    if isinstance(item, dict):
        for key in ["text", "ocr_text", "page_text", "words", "tokens"]:
            if key in item and item[key] not in (None, ""):
                value = item[key]
                if isinstance(value, list):
                    return " ".join(str(v) for v in value)
                return str(value)
    return ""


def resolve_image_path(raw_path: str, image_root: Path, input_root: Path) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    candidates = [
        image_root / path,
        input_root / path,
        image_root / path.name,
        input_root / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if not path.suffix:
        for ext in IMAGE_EXTS:
            for candidate in [
                image_root / f"{path}{ext}",
                input_root / f"{path}{ext}",
                image_root / f"{path.name}{ext}",
                input_root / f"{path.name}{ext}",
            ]:
                if candidate.exists():
                    return candidate
    return image_root / path


def row_page_items(row: dict[str, Any]) -> list[Any]:
    pages = first_value(row, PAGE_LIST_KEYS)
    return normalize_list(pages)


def row_page_texts(row: dict[str, Any]) -> list[str]:
    values = normalize_list(first_value(row, PAGE_TEXT_KEYS))
    out = []
    for value in values:
        if isinstance(value, list):
            out.append(" ".join(str(v) for v in value))
        else:
            out.append(str(value))
    return out


def parse_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        text = str(value)
        nums = re.findall(r"\d+", text)
        if nums:
            return int(nums[-1])
    return None


def normalize_gold_page_indices(raw_values: list[Any], page_count: int, base: str) -> list[int]:
    values = []
    for raw in raw_values:
        if isinstance(raw, dict):
            for key in ["page_idx", "page_id", "page", "answer_page_idx", "answer_page"]:
                if key in raw:
                    raw = raw[key]
                    break
        parsed = parse_int(raw)
        if parsed is not None:
            values.append(parsed)
    if not values:
        return []
    if base == "zero":
        converted = values
    elif base == "one":
        converted = [value - 1 for value in values]
    else:
        if any(value == 0 for value in values):
            converted = values
        elif page_count > 0 and all(1 <= value <= page_count for value in values):
            converted = [value - 1 for value in values]
        else:
            converted = values
    return sorted({idx for idx in converted if idx >= 0})


def extract_gold_pages_from_page_items(items: list[Any]) -> list[Any]:
    out = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        flag = item.get("is_answer_page", item.get("is_gold", item.get("answer_page", item.get("gold"))))
        if flag is True or str(flag).strip().lower() in {"1", "true", "yes"}:
            out.append(item.get("page_idx", item.get("page_id", idx)))
    return out


def rel_or_abs(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def compact_raw_row(row: dict[str, Any]) -> dict[str, Any]:
    keep = QID_KEYS + QUESTION_KEYS + ANSWER_KEYS + DOC_ID_KEYS + GOLD_PAGE_KEYS
    return {key: row[key] for key in keep if key in row}


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    image_root = Path(args.image_root) if args.image_root else input_root
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    annotation_files = existing_annotation_files(input_root, args.source_split, args.annotation_file)
    rows: list[dict[str, Any]] = []
    source_file_counts: Counter[str] = Counter()
    for path in annotation_files:
        loaded = load_annotation_rows(path)
        rows.extend(loaded)
        source_file_counts[str(path)] += len(loaded)
    if args.max_examples > 0:
        rows = rows[: args.max_examples]

    image_index = scan_image_root(image_root) if args.scan_images else {}

    split = args.split
    mmqa_path = output_root / f"MMQA_{split}.jsonl"
    qids_path = output_root / f"qids_{split}.jsonl"
    gold_pages_path = output_root / f"gold_pages_{split}.jsonl"
    doc_pages_path = output_root / f"doc_pages_{split}.jsonl"
    doc_ids_path = output_root / f"{split}_doc_ids.json"
    doc_id_map_path = output_root / "doc_id_map.json"
    summary_path = output_root / f"prepare_{split}_summary.json"

    doc_pages: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    doc_id_map: dict[str, Any] = {}
    missing_images: list[dict[str, Any]] = []
    missing_gold_pages: list[dict[str, Any]] = []
    question_count = 0
    labeled_count = 0
    doc_page_count_hist: Counter[int] = Counter()
    gold_count_hist: Counter[int] = Counter()
    source_schema_counts: Counter[str] = Counter()

    with mmqa_path.open("w", encoding="utf-8") as mmqa_out, qids_path.open(
        "w", encoding="utf-8"
    ) as qids_out, gold_pages_path.open("w", encoding="utf-8") as gold_out:
        used_qids: set[str] = set()
        for row_index, row in enumerate(rows):
            page_items = row_page_items(row)
            if not first_value(row, QUESTION_KEYS) and not page_items:
                source_schema_counts["skipped_non_question_row"] += 1
                continue
            raw_doc_id = first_value(row, DOC_ID_KEYS)
            if raw_doc_id is None and isinstance(row.get("extra_info"), dict):
                raw_doc_id = first_value(row["extra_info"], ["ucsf_doc_id", "single_doc_vqa_id"])
            if raw_doc_id is None and page_items:
                raw_doc_id = Path(extract_path_from_page_item(page_items[0])).parent.name
            doc_id = safe_doc_id(raw_doc_id or f"doc_{row_index:06d}")
            doc_id_map.setdefault(doc_id, {"source_doc_id": str(raw_doc_id or doc_id)})

            page_texts = row_page_texts(row)
            page_paths: list[Path] = []
            page_item_texts: list[str] = []
            if page_items:
                for page_order, item in enumerate(page_items):
                    raw_path = extract_path_from_page_item(item)
                    image_path = resolve_image_path(raw_path, image_root, input_root)
                    page_paths.append(image_path)
                    text = extract_text_from_page_item(item)
                    if not text and page_order < len(page_texts):
                        text = page_texts[page_order]
                    page_item_texts.append(text)
                source_schema_counts["row_page_list"] += 1
                if page_paths and all(not path.exists() for path in page_paths) and doc_id in image_index:
                    page_paths = list(image_index[doc_id])
                    page_item_texts = ["" for _ in page_paths]
                    source_schema_counts["row_page_list_replaced_by_scanned_image_root"] += 1
            elif doc_id in image_index:
                page_paths = list(image_index[doc_id])
                page_item_texts = ["" for _ in page_paths]
                source_schema_counts["scanned_image_root"] += 1
            else:
                source_schema_counts["missing_page_list"] += 1

            for page_idx, image_path in enumerate(page_paths):
                if not image_path.exists():
                    missing_images.append({"doc_id": doc_id, "page_idx": page_idx, "image_path": str(image_path)})
                doc_pages[doc_id][page_idx] = {
                    "doc_id": doc_id,
                    "doc_name": str(raw_doc_id or doc_id),
                    "page_idx": page_idx,
                    "page_number": page_idx + 1,
                    "page_uid": f"{doc_id}_page{page_idx}",
                    "image_path": rel_or_abs(image_path, output_root),
                    "text": page_item_texts[page_idx] if page_idx < len(page_item_texts) else "",
                    "ocr_text": page_item_texts[page_idx] if page_idx < len(page_item_texts) else "",
                    "source_image_path": str(image_path),
                }

            raw_gold = normalize_list(first_value(row, GOLD_PAGE_KEYS))
            if not raw_gold:
                raw_gold = extract_gold_pages_from_page_items(page_items)
            gold_page_ids = normalize_gold_page_indices(raw_gold, len(page_paths), args.gold_page_base)
            gold_page_ids = [idx for idx in gold_page_ids if idx < len(page_paths) or len(page_paths) == 0]
            gold_page_uids = [f"{doc_id}_page{idx}" for idx in gold_page_ids]
            for idx, uid in zip(gold_page_ids, gold_page_uids):
                if idx not in doc_pages.get(doc_id, {}):
                    missing_gold_pages.append({"qid": safe_qid(first_value(row, QID_KEYS), row_index), "doc_id": doc_id, "page_idx": idx, "page_uid": uid})

            qid = safe_qid(first_value(row, QID_KEYS), row_index)
            if qid in used_qids:
                suffix = 2
                base_qid = qid
                while qid in used_qids:
                    qid = f"{base_qid}__dup{suffix}"
                    suffix += 1
            used_qids.add(qid)
            question = str(first_value(row, QUESTION_KEYS) or "").strip()
            answer = normalize_answer(first_value(row, ANSWER_KEYS))
            supporting_context = [
                {
                    "doc_id": doc_id,
                    "doc_part": "answer_page",
                    "page_idx": idx,
                    "page_id": idx,
                    "source_page_number": idx + 1,
                }
                for idx in gold_page_ids
            ]
            if not supporting_context:
                supporting_context = [{"doc_id": doc_id, "doc_part": "document"}]

            metadata = {
                "type": "MP-DocVQA",
                "source": "MP-DocVQA",
                "source_split": args.source_split,
                "source_doc_id": str(raw_doc_id or doc_id),
                "doc_id": doc_id,
                "gold_page_ids": gold_page_ids,
                "gold_page_uids": gold_page_uids,
                "original_answer": first_value(row, ANSWER_KEYS),
                "annotation_files": [str(path) for path in annotation_files],
            }
            if args.store_raw_row:
                metadata["raw_row"] = compact_raw_row(row)

            mmqa_row = {
                "qid": qid,
                "question": question,
                "answers": [{"answer": answer, "modality": "document"}],
                "metadata": metadata,
                "supporting_context": supporting_context,
            }
            mmqa_out.write(json.dumps(mmqa_row, ensure_ascii=False) + "\n")
            qids_out.write(json.dumps({"qid": qid}, ensure_ascii=False) + "\n")
            gold_out.write(
                json.dumps(
                    {
                        "qid": qid,
                        "doc_id": doc_id,
                        "gold_page_ids": gold_page_ids,
                        "gold_page_uids": gold_page_uids,
                        "question": question,
                        "answer": answer,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            question_count += 1
            labeled_count += int(bool(gold_page_ids))
            doc_page_count_hist[len(page_paths)] += 1
            gold_count_hist[len(gold_page_ids)] += 1

    if missing_images and not args.allow_missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} page images; first missing: {missing_images[:5]}")

    doc_ids = sorted(doc_pages)
    with doc_pages_path.open("w", encoding="utf-8") as out:
        for doc_id in doc_ids:
            for page_idx in sorted(doc_pages[doc_id]):
                out.write(json.dumps(doc_pages[doc_id][page_idx], ensure_ascii=False) + "\n")
    doc_ids_path.write_text(json.dumps(doc_ids, indent=2) + "\n", encoding="utf-8")
    doc_id_map_path.write_text(json.dumps(doc_id_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    page_count = sum(len(pages) for pages in doc_pages.values())
    summary = {
        "prepared_output_root": str(output_root),
        "input_root": str(input_root),
        "image_root": str(image_root),
        "split": split,
        "source_split": args.source_split,
        "annotation_files": [str(path) for path in annotation_files],
        "source_file_counts": dict(source_file_counts),
        "doc_count": len(doc_ids),
        "page_count": page_count,
        "qa_count": question_count,
        "labeled_qids": labeled_count,
        "unlabeled_qids": question_count - labeled_count,
        "missing_image_count": len(missing_images),
        "missing_images_sample": missing_images[:20],
        "missing_gold_page_count": len(missing_gold_pages),
        "missing_gold_pages_sample": missing_gold_pages[:20],
        "doc_page_count_hist": dict(sorted(doc_page_count_hist.items())),
        "gold_page_count_hist": dict(sorted(gold_count_hist.items())),
        "source_schema_counts": dict(sorted(source_schema_counts.items())),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"prepared_output_root={output_root}")
    print(f"doc_count={len(doc_ids)}")
    print(f"page_count={page_count}")
    print(f"qa_count={question_count}")
    print(f"labeled_qids={labeled_count}")
    print(f"missing_image_count={len(missing_images)}")
    print(f"missing_gold_page_count={len(missing_gold_pages)}")
    print(f"source_schema_counts={dict(sorted(source_schema_counts.items()))}")
    print(f"saved_summary={summary_path}")


if __name__ == "__main__":
    main()
