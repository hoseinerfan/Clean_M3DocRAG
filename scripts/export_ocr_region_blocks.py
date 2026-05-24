#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export real OCR region blocks with bounding boxes for selected pages. "
            "Use this on hard subsets before rerank_layout_evidence_graph.py so the "
            "evidence graph uses actual OCR/layout region nodes instead of fallback text blocks."
        )
    )
    parser.add_argument("--doc-pages-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument(
        "--prediction-json",
        action="append",
        default=[],
        help="Prediction JSON whose top pages define pages to OCR. Repeatable.",
    )
    parser.add_argument(
        "--qid-filter-jsonl",
        action="append",
        default=[],
        help="Gold/subset JSONL with qid fields used to filter prediction rows.",
    )
    parser.add_argument("--prediction-top-pages", type=int, default=50)
    parser.add_argument(
        "--page-uid-jsonl",
        action="append",
        default=[],
        help="Optional JSONL with page_uid or doc_id/page_idx rows to OCR.",
    )
    parser.add_argument("--page-uid", action="append", default=[])
    parser.add_argument("--max-pages", type=int, default=0)
    parser.add_argument("--image-root", default="")
    parser.add_argument(
        "--image-path-prefix",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help=(
            "Rewrite stale image path prefixes before existence checks. "
            "Repeatable; useful when doc_pages has absolute paths from another filesystem."
        ),
    )
    parser.add_argument("--ocr-engine", choices=["tesseract", "easyocr"], default="easyocr")
    parser.add_argument("--ocr-bin", default="tesseract")
    parser.add_argument("--ocr-lang", default="eng")
    parser.add_argument("--ocr-psm", default="")
    parser.add_argument("--ocr-timeout", type=int, default=120)
    parser.add_argument("--easyocr-gpu", action="store_true")
    parser.add_argument("--easyocr-model-dir", default="")
    parser.add_argument("--no-easyocr-download", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def maybe_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def page_uid(doc_id: Any, page_idx: Any) -> str:
    return f"{doc_id}_page{int(page_idx)}"


def page_key_from_row(row: dict[str, Any]) -> tuple[str, int] | None:
    doc_id = str(row.get("doc_id", row.get("doc_name", ""))).strip()
    page_idx = maybe_int(row.get("page_idx", row.get("page_id", row.get("page_number"))))
    if not doc_id or page_idx is None:
        return None
    return doc_id, page_idx


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).replace("\x0c", " ").replace("\u0000", " ")).strip()


def load_prediction(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "predictions" in payload and isinstance(
        payload["predictions"],
        (dict, list),
    ):
        payload = payload["predictions"]
    rows_by_qid: dict[str, dict[str, Any]] = {}
    iterable: Any
    if isinstance(payload, list):
        iterable = enumerate(payload)
    elif isinstance(payload, dict):
        iterable = payload.items()
    else:
        raise TypeError(f"Prediction JSON must be list or object: {path}")
    for raw_key, row in iterable:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("qid", "")).strip() or str(raw_key).strip()
        if qid:
            rows_by_qid[qid] = row
    return rows_by_qid


def load_qid_filter(paths: list[str]) -> set[str]:
    qids: set[str] = set()
    for raw_path in paths:
        for row in read_jsonl(Path(raw_path)):
            qid = str(row.get("qid", "")).strip()
            if qid:
                qids.add(qid)
    return qids


def collect_prediction_pages(
    path: Path,
    *,
    qids: set[str],
    top_pages: int,
) -> set[str]:
    pages: set[str] = set()
    prediction = load_prediction(path)
    selected_qids = sorted(qids) if qids else sorted(prediction)
    for qid in selected_qids:
        row = prediction.get(qid)
        if row is None:
            continue
        seen_for_qid: set[str] = set()
        for item in row.get("page_retrieval_results", []):
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            try:
                uid = page_uid(str(item[0]), int(item[1]))
            except (TypeError, ValueError):
                continue
            if uid in seen_for_qid:
                continue
            seen_for_qid.add(uid)
            pages.add(uid)
            if top_pages > 0 and len(seen_for_qid) >= top_pages:
                break
    return pages


def collect_requested_pages(args: argparse.Namespace) -> set[str]:
    pages = {str(uid).strip() for uid in args.page_uid if str(uid).strip()}
    qids = load_qid_filter(args.qid_filter_jsonl)
    for raw_path in args.prediction_json:
        pages |= collect_prediction_pages(
            Path(raw_path),
            qids=qids,
            top_pages=max(0, int(args.prediction_top_pages)),
        )
    for raw_path in args.page_uid_jsonl:
        for row in read_jsonl(Path(raw_path)):
            uid = str(row.get("page_uid", "")).strip()
            if not uid:
                key = page_key_from_row(row)
                uid = page_uid(*key) if key else ""
            if uid:
                pages.add(uid)
    return pages


def load_doc_pages(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = page_key_from_row(row)
            if key is None:
                continue
            rows[page_uid(*key)] = row
    return rows


def parse_image_path_prefixes(values: list[str]) -> list[tuple[str, str]]:
    prefixes: list[tuple[str, str]] = []
    for value in values:
        if "=" not in str(value):
            raise ValueError(f"--image-path-prefix must be OLD=NEW, got: {value}")
        old, new = str(value).split("=", 1)
        old = old.rstrip("/")
        new = new.rstrip("/")
        if not old or not new:
            raise ValueError(f"--image-path-prefix must have non-empty OLD and NEW: {value}")
        prefixes.append((old, new))
    prefixes.sort(key=lambda pair: len(pair[0]), reverse=True)
    return prefixes


def rewrite_path(text: str, prefixes: list[tuple[str, str]]) -> list[str]:
    rewritten: list[str] = []
    for old, new in prefixes:
        if text == old:
            rewritten.append(new)
        elif text.startswith(old + "/"):
            rewritten.append(new + text[len(old) :])
    return rewritten


def candidate_image_paths(
    row: dict[str, Any],
    image_root: Path | None,
    image_path_prefixes: list[tuple[str, str]],
) -> list[Path]:
    candidates: list[Path] = []

    def add(value: Any) -> None:
        text = str(value or "").strip()
        if not text:
            return
        for candidate_text in [text, *rewrite_path(text, image_path_prefixes)]:
            path = Path(candidate_text)
            if path.is_absolute():
                candidates.append(path)
            elif image_root is not None:
                candidates.append(image_root / path)
            else:
                candidates.append(path)

    for key in ("image_path", "source_image_path", "page_image_path", "image"):
        add(row.get(key))
    seen: set[Path] = set()
    out: list[Path] = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def resolve_image_path(
    row: dict[str, Any],
    image_root: Path | None,
    image_path_prefixes: list[tuple[str, str]],
) -> Path | None:
    for path in candidate_image_paths(row, image_root, image_path_prefixes):
        if path.exists():
            return path
    return None


def bbox_from_points(points: Any) -> list[float]:
    xs: list[float] = []
    ys: list[float] = []
    for point in points or []:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            try:
                xs.append(float(point[0]))
                ys.append(float(point[1]))
            except (TypeError, ValueError):
                continue
    if not xs or not ys:
        return [0.0, 0.0, 0.0, 0.0]
    return [min(xs), min(ys), max(xs), max(ys)]


def easyocr_lang_list(raw_lang: str) -> list[str]:
    mapping = {
        "eng": "en",
        "fra": "fr",
        "fre": "fr",
        "deu": "de",
        "ger": "de",
        "spa": "es",
        "ita": "it",
        "por": "pt",
    }
    values = [part.strip() for part in re.split(r"[,+ ]+", raw_lang) if part.strip()]
    return [mapping.get(value.lower(), value.lower()) for value in values] or ["en"]


def build_easyocr_reader(args: argparse.Namespace) -> Any:
    try:
        import easyocr  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("EasyOCR is not installed; use --ocr-engine=tesseract.") from exc
    kwargs: dict[str, Any] = {
        "gpu": bool(args.easyocr_gpu),
        "download_enabled": not bool(args.no_easyocr_download),
    }
    if str(args.easyocr_model_dir).strip():
        kwargs["model_storage_directory"] = str(args.easyocr_model_dir)
    return easyocr.Reader(easyocr_lang_list(str(args.ocr_lang)), **kwargs)


def easyocr_blocks(reader: Any, image_path: Path, min_confidence: float) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for idx, item in enumerate(reader.readtext(str(image_path), detail=1, paragraph=False)):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        bbox = bbox_from_points(item[0])
        text = normalize_text(item[1])
        try:
            confidence = float(item[2]) if len(item) >= 3 else 1.0
        except (TypeError, ValueError):
            confidence = 0.0
        if not text or confidence < min_confidence:
            continue
        blocks.append(
            {
                "text": text,
                "bbox": bbox,
                "confidence": confidence,
                "type": "ocr_line",
                "order": idx,
            }
        )
    return blocks


def parse_tesseract_conf(value: str) -> float | None:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return None
    if conf < 0:
        return None
    return conf / 100.0 if conf > 1.0 else conf


def tesseract_blocks(args: argparse.Namespace, image_path: Path, min_confidence: float) -> list[dict[str, Any]]:
    cmd = [str(args.ocr_bin), str(image_path), "stdout", "tsv", "-l", str(args.ocr_lang)]
    if str(args.ocr_psm).strip():
        cmd.extend(["--psm", str(args.ocr_psm)])
    proc = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=max(1, int(args.ocr_timeout)),
    )
    reader = csv.DictReader(proc.stdout.splitlines(), delimiter="\t")
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in reader:
        if row.get("level") != "5":
            continue
        text = normalize_text(row.get("text"))
        conf = parse_tesseract_conf(str(row.get("conf", "")))
        if not text or conf is None or conf < min_confidence:
            continue
        key = (
            str(row.get("page_num", "")),
            str(row.get("block_num", "")),
            str(row.get("par_num", "")),
            str(row.get("line_num", "")),
        )
        grouped[key].append(row)

    blocks: list[dict[str, Any]] = []
    for order, (_key, words) in enumerate(sorted(grouped.items(), key=lambda item: item[0])):
        texts: list[str] = []
        confs: list[float] = []
        lefts: list[int] = []
        tops: list[int] = []
        rights: list[int] = []
        bottoms: list[int] = []
        for word in words:
            text = normalize_text(word.get("text"))
            conf = parse_tesseract_conf(str(word.get("conf", "")))
            if not text or conf is None:
                continue
            try:
                left = int(float(word.get("left", 0)))
                top = int(float(word.get("top", 0)))
                width = int(float(word.get("width", 0)))
                height = int(float(word.get("height", 0)))
            except (TypeError, ValueError):
                continue
            texts.append(text)
            confs.append(conf)
            lefts.append(left)
            tops.append(top)
            rights.append(left + width)
            bottoms.append(top + height)
        line_text = normalize_text(" ".join(texts))
        if not line_text:
            continue
        blocks.append(
            {
                "text": line_text,
                "bbox": [min(lefts), min(tops), max(rights), max(bottoms)],
                "confidence": sum(confs) / float(len(confs)) if confs else 0.0,
                "type": "ocr_line",
                "order": order,
            }
        )
    return blocks


def export_regions(args: argparse.Namespace) -> dict[str, Any]:
    requested_pages = collect_requested_pages(args)
    doc_pages = load_doc_pages(Path(args.doc_pages_jsonl))
    if not requested_pages:
        requested_pages = set(doc_pages)
    selected_pages = sorted(uid for uid in requested_pages if uid in doc_pages)
    if int(args.max_pages) > 0:
        selected_pages = selected_pages[: int(args.max_pages)]
    missing_requested_pages = len(requested_pages - set(doc_pages))

    image_root = Path(args.image_root) if str(args.image_root).strip() else None
    image_path_prefixes = parse_image_path_prefixes(args.image_path_prefix)
    reader = build_easyocr_reader(args) if str(args.ocr_engine) == "easyocr" else None

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    page_count = 0
    page_with_blocks_count = 0
    block_count = 0
    missing_image_count = 0
    error_count = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for idx, uid in enumerate(selected_pages, start=1):
            row = doc_pages[uid]
            key = page_key_from_row(row)
            if key is None:
                continue
            image_path = resolve_image_path(row, image_root, image_path_prefixes)
            if image_path is None:
                missing_image_count += 1
                if not args.continue_on_error:
                    raise FileNotFoundError(f"No image found for page {uid}")
                blocks: list[dict[str, Any]] = []
                image_path_str = ""
            else:
                image_path_str = str(image_path)
                try:
                    if str(args.ocr_engine) == "easyocr":
                        blocks = easyocr_blocks(reader, image_path, float(args.min_confidence))
                    else:
                        blocks = tesseract_blocks(args, image_path, float(args.min_confidence))
                except Exception:
                    error_count += 1
                    if not args.continue_on_error:
                        raise
                    blocks = []
            page_count += 1
            block_count += len(blocks)
            page_with_blocks_count += int(bool(blocks))
            out = {
                "doc_id": key[0],
                "page_idx": int(key[1]),
                "page_uid": uid,
                "image_path": image_path_str,
                "ocr_engine": str(args.ocr_engine),
                "ocr_blocks": blocks,
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            if int(args.progress_every) > 0 and idx % int(args.progress_every) == 0:
                print(
                    f"processed_pages {idx}/{len(selected_pages)} blocks {block_count}",
                    file=sys.stderr,
                    flush=True,
                )

    summary = {
        "doc_pages_jsonl": str(args.doc_pages_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "ocr_engine": str(args.ocr_engine),
        "requested_page_count": len(requested_pages),
        "selected_page_count": len(selected_pages),
        "missing_requested_page_count": missing_requested_pages,
        "page_count": page_count,
        "page_with_blocks_count": page_with_blocks_count,
        "missing_image_count": missing_image_count,
        "error_count": error_count,
        "block_count": block_count,
        "mean_blocks_per_page": block_count / float(page_count) if page_count else 0.0,
        "prediction_json": [str(path) for path in args.prediction_json],
        "qid_filter_jsonl": [str(path) for path in args.qid_filter_jsonl],
        "prediction_top_pages": int(args.prediction_top_pages),
        "min_confidence": float(args.min_confidence),
        "image_root": str(args.image_root),
        "image_path_prefix": list(args.image_path_prefix),
    }
    Path(args.output_summary_json).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = export_regions(args)
    print(f"saved_regions: {args.output_jsonl}")
    print(f"saved_summary: {args.output_summary_json}")
    for key, value in summary.items():
        if key in {"prediction_json", "qid_filter_jsonl"}:
            continue
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
