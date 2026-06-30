#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import fmean
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export per-page visual metadata for M3DocVQA PDFs. The output is a "
            "sidecar JSONL used only for image-title pseudo-label tie-breaking."
        )
    )
    parser.add_argument("--data-name", default="m3-docvqa")
    parser.add_argument("--split", default="dev")
    parser.add_argument(
        "--doc-id-json",
        help=(
            "Optional JSON / JSONL / plain-text file listing doc ids to export. "
            "If omitted, exports all supporting docs from the split."
        ),
    )
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--max-pages-per-doc", type=int, default=0)
    parser.add_argument(
        "--large-image-min-area-ratio",
        type=float,
        default=0.005,
        help=(
            "Minimum page-area ratio for an image object to count as a large image. "
            "Default 0.005 ignores tiny icons and bullets."
        ),
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-summary-json", required=True)
    return parser.parse_args()


def make_dataset_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        data_name=args.data_name,
        split=args.split,
        loop_unique_doc_ids=False,
        data_len=None,
    )


def load_doc_ids(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw[0] == "[":
        return dedupe_doc_ids(json.loads(raw))
    doc_ids: list[Any] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            doc_ids.append(json.loads(line))
        except json.JSONDecodeError:
            doc_ids.append(line)
    return dedupe_doc_ids(doc_ids)


def dedupe_doc_ids(values: list[Any]) -> list[str]:
    doc_ids: list[str] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, dict):
            raw_doc_id = value.get("doc_id")
        else:
            raw_doc_id = value
        doc_id = str(raw_doc_id or "").strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        doc_ids.append(doc_id)
    return doc_ids


def page_image_metadata(page: Any, *, large_image_min_area_ratio: float) -> dict[str, Any]:
    page_area = max(float(page.rect.width) * float(page.rect.height), 1.0)
    image_count = 0
    large_image_count = 0
    area_ratio_sum = 0.0
    largest_area_ratio = 0.0

    for image_info in page.get_images(full=True):
        xref = int(image_info[0])
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            rects = []
        for rect in rects:
            area_ratio = max(float(rect.width) * float(rect.height), 0.0) / page_area
            image_count += 1
            area_ratio_sum += area_ratio
            largest_area_ratio = max(largest_area_ratio, area_ratio)
            if area_ratio >= float(large_image_min_area_ratio):
                large_image_count += 1

    image_area_ratio = min(area_ratio_sum, 1.0)
    return {
        "image_count": int(image_count),
        "large_image_count": int(large_image_count),
        "image_area_ratio": round(float(image_area_ratio), 8),
        "largest_image_area_ratio": round(float(largest_area_ratio), 8),
        "has_image": bool(image_count > 0),
        "has_large_image": bool(large_image_count > 0),
    }


def main() -> None:
    args = parse_args()
    try:
        import fitz
    except ImportError as exc:
        raise SystemExit(
            "PyMuPDF is required for visual metadata export. Install/use an environment "
            "with the 'fitz' module available."
        ) from exc
    from m3docrag.datasets.m3_docvqa import M3DocVQADataset

    dataset = M3DocVQADataset(make_dataset_args(args))
    if args.doc_id_json:
        doc_ids = load_doc_ids(Path(args.doc_id_json))
    else:
        doc_ids = [str(doc_id) for doc_id in dataset.all_supporting_doc_ids]
    if int(args.max_docs) > 0:
        doc_ids = doc_ids[: int(args.max_docs)]

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json = Path(args.output_summary_json)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)

    page_count = 0
    pages_with_image = 0
    pages_with_large_image = 0
    image_counts: list[int] = []
    large_image_counts: list[int] = []
    area_ratios: list[float] = []
    missing_pdf_doc_ids: list[str] = []
    exported_doc_page_counts: dict[str, int] = {}

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for doc_idx, doc_id in enumerate(doc_ids, start=1):
            pdf_path = dataset.pdf_dir / f"{doc_id}.pdf"
            if not pdf_path.exists():
                missing_pdf_doc_ids.append(doc_id)
                continue
            with fitz.open(pdf_path) as pdf:
                page_limit = int(pdf.page_count)
                if int(args.max_pages_per_doc) > 0:
                    page_limit = min(page_limit, int(args.max_pages_per_doc))
                exported_doc_page_counts[doc_id] = page_limit
                for page_idx in range(page_limit):
                    page = pdf.load_page(page_idx)
                    visual = page_image_metadata(
                        page,
                        large_image_min_area_ratio=float(args.large_image_min_area_ratio),
                    )
                    page_count += 1
                    if visual["has_image"]:
                        pages_with_image += 1
                    if visual["has_large_image"]:
                        pages_with_large_image += 1
                    image_counts.append(int(visual["image_count"]))
                    large_image_counts.append(int(visual["large_image_count"]))
                    area_ratios.append(float(visual["image_area_ratio"]))
                    row = {
                        "page_uid": f"{doc_id}_page{page_idx}",
                        "doc_id": doc_id,
                        "page_idx": int(page_idx),
                        **visual,
                    }
                    handle.write(json.dumps(row) + "\n")
            print(
                f"exported_visual_doc: {doc_idx}/{len(doc_ids)} "
                f"doc_id={doc_id} pages={exported_doc_page_counts[doc_id]}"
            )

    summary = {
        "data_name": args.data_name,
        "split": args.split,
        "requested_doc_count": len(doc_ids),
        "exported_doc_count": len(exported_doc_page_counts),
        "page_count": int(page_count),
        "pages_with_image": int(pages_with_image),
        "pages_with_large_image": int(pages_with_large_image),
        "page_image_fraction": round(pages_with_image / page_count, 6) if page_count else None,
        "page_large_image_fraction": (
            round(pages_with_large_image / page_count, 6) if page_count else None
        ),
        "mean_image_count": round(float(fmean(image_counts)), 6) if image_counts else None,
        "mean_large_image_count": (
            round(float(fmean(large_image_counts)), 6) if large_image_counts else None
        ),
        "mean_image_area_ratio": round(float(fmean(area_ratios)), 6) if area_ratios else None,
        "large_image_min_area_ratio": float(args.large_image_min_area_ratio),
        "missing_pdf_doc_count": len(missing_pdf_doc_ids),
        "missing_pdf_doc_ids": missing_pdf_doc_ids,
        "output_jsonl": str(output_jsonl),
    }
    output_summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"saved_jsonl: {output_jsonl}")
    print(f"saved_summary: {output_summary_json}")
    print(f"page_count: {page_count}")
    print(f"pages_with_image: {pages_with_image}")
    print(f"pages_with_large_image: {pages_with_large_image}")


if __name__ == "__main__":
    main()
