#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import tarfile
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare SciEGQA-Bench for M3DocRAG-style page retrieval. "
            "The converter keeps original 1-based evidence pages/boxes in metadata "
            "and emits zero-based page_idx labels for retrieval evaluation."
        )
    )
    parser.add_argument("--hf-repo", default="Yuwh07/SciEGQA-Bench")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--snapshot-dir", default="")
    parser.add_argument("--annotations-jsonl", default="")
    parser.add_argument("--images-tar", default="")
    parser.add_argument("--image-root", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--overwrite-extract", action="store_true")
    parser.add_argument(
        "--overwrite-rendered-pages",
        action="store_true",
        help="Re-render page images from extracted PDFs even if pages_dev already exists.",
    )
    parser.add_argument("--pdf-dpi", type=int, default=144)
    return parser.parse_args()


def safe_doc_id(category: str, doc_name: str) -> str:
    return quote(f"{category}__{doc_name}", safe="._-")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_file(root: Path, names: list[str], suffix: str | None = None) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    for path in root.rglob("*"):
        if path.name in names:
            return path
        if suffix is not None and path.name.endswith(suffix):
            return path
    return None


def download_snapshot(args: argparse.Namespace, snapshot_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=args.hf_repo,
        repo_type="dataset",
        local_dir=str(snapshot_dir),
        allow_patterns=["*.jsonl", "*.tar", "*.md"],
    )


def resolve_inputs(args: argparse.Namespace, output_root: Path) -> tuple[Path, Path | None]:
    snapshot_dir = Path(args.snapshot_dir) if args.snapshot_dir else output_root / "hf_snapshot"
    if args.download:
        download_snapshot(args, snapshot_dir)

    annotations_path = Path(args.annotations_jsonl) if args.annotations_jsonl else None
    if annotations_path is None:
        annotations_path = find_file(
            snapshot_dir,
            ["SciEGQA_Bench.jsonl", "SciEGQA-Bench.jsonl", "SciEGQA-Bench.jsonl"],
            suffix=".jsonl",
        )
    if annotations_path is None or not annotations_path.exists():
        raise FileNotFoundError(
            "Could not find SciEGQA benchmark JSONL. Pass --annotations-jsonl or use --download."
        )

    images_tar = Path(args.images_tar) if args.images_tar else None
    if images_tar is None:
        images_tar = find_file(snapshot_dir, ["images.tar"], suffix=".tar")
    if images_tar is not None and not images_tar.exists():
        images_tar = None
    return annotations_path, images_tar


def extract_images(images_tar: Path, image_root: Path, overwrite: bool) -> None:
    marker = image_root / ".extracted"
    if marker.exists() and not overwrite:
        return
    image_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(images_tar, "r:*") as tar:
        tar.extractall(image_root)
    marker.write_text(str(images_tar) + "\n", encoding="utf-8")


def normalize_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def page_to_idx(page_number: int) -> int:
    return int(page_number) - 1


def find_doc_dir(image_root: Path, category: str, doc_name: str) -> Path | None:
    candidates = [
        image_root / category / doc_name,
        image_root / "images" / category / doc_name,
        image_root / "data" / category / doc_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(image_root.rglob(f"{doc_name}_*.png"))
    if matches:
        return matches[0].parent
    return None


def find_pdf_path(image_root: Path, category: str, doc_name: str) -> Path | None:
    candidates = [
        image_root / "PDF" / category / f"{doc_name}.pdf",
        image_root / category / f"{doc_name}.pdf",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted((image_root / "PDF" / category).glob(f"{doc_name}*.pdf")) if (image_root / "PDF" / category).exists() else []
    if not matches:
        matches = sorted(image_root.rglob(f"{doc_name}*.pdf"))
    return matches[0] if matches else None


def existing_rendered_pages(page_dir: Path) -> list[tuple[int, Path]]:
    records = []
    if not page_dir.exists():
        return records
    for image_path in sorted(page_dir.glob("*.jpg")):
        try:
            page_idx = int(image_path.stem)
        except ValueError:
            continue
        records.append((page_idx + 1, image_path))
    return records


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
        out_path = page_dir / f"{page_number - 1}.jpg"
        if out_path.exists() and not overwrite:
            rendered.append((page_number, out_path))
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
        rendered.append((page_number, out_path))
    return rendered


def parse_page_number(path: Path, doc_name: str) -> int | None:
    match = re.search(rf"{re.escape(doc_name)}_(\d+)\.png$", path.name)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)\.png$", path.name)
    if match:
        return int(match.group(1))
    return None


def rel_or_abs(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def build_doc_page_manifest(
    *,
    rows: list[dict],
    output_root: Path,
    image_root: Path | None,
    split: str,
    overwrite_rendered_pages: bool,
    pdf_dpi: int,
) -> dict[str, int]:
    docs: dict[str, tuple[str, str]] = {}
    for row in rows:
        category = str(row.get("category", "")).strip()
        doc_name = str(row.get("doc_name", "")).strip()
        docs[safe_doc_id(category, doc_name)] = (category, doc_name)

    page_counts: dict[str, int] = {}
    manifest_path = output_root / f"doc_pages_{split}.jsonl"
    with manifest_path.open("w", encoding="utf-8") as out:
        for doc_id, (category, doc_name) in sorted(docs.items()):
            page_records = []
            doc_dir = find_doc_dir(image_root, category, doc_name) if image_root else None
            if doc_dir is not None:
                for image_path in sorted(doc_dir.glob(f"{doc_name}_*.png")):
                    page_number = parse_page_number(image_path, doc_name)
                    if page_number is None:
                        continue
                    page_records.append((page_number, image_path))
            if not page_records and image_root is not None:
                rendered_dir = output_root / f"pages_{split}" / doc_id
                pdf_path = find_pdf_path(image_root, category, doc_name)
                if pdf_path is not None:
                    page_records = render_pdf_pages(
                        pdf_path=pdf_path,
                        page_dir=rendered_dir,
                        overwrite=overwrite_rendered_pages,
                        dpi=pdf_dpi,
                    )
            page_records.sort(key=lambda item: item[0])
            page_counts[doc_id] = len(page_records)
            for page_number, image_path in page_records:
                page_idx = page_to_idx(page_number)
                out.write(
                    json.dumps(
                        {
                            "doc_id": doc_id,
                            "doc_name": doc_name,
                            "category": category,
                            "page_idx": page_idx,
                            "page_number": page_number,
                            "page_uid": f"{doc_id}_page{page_idx}",
                            "image_path": rel_or_abs(image_path, output_root),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return page_counts


def convert_annotations(
    *,
    rows: list[dict],
    output_root: Path,
    split: str,
    max_questions: int,
) -> dict[str, str]:
    mmqa_path = output_root / f"MMQA_{split}.jsonl"
    qids_path = output_root / f"qids_{split}.jsonl"
    gold_pages_path = output_root / f"gold_pages_{split}.jsonl"
    doc_ids_path = output_root / f"{split}_doc_ids.json"
    doc_id_map_path = output_root / "doc_id_map.json"

    doc_ids: list[str] = []
    doc_id_map: dict[str, str] = {}
    per_doc_question_idx: dict[str, int] = defaultdict(int)

    with mmqa_path.open("w", encoding="utf-8") as mmqa_out, qids_path.open(
        "w", encoding="utf-8"
    ) as qids_out, gold_pages_path.open("w", encoding="utf-8") as gold_out:
        for global_idx, row in enumerate(rows):
            if max_questions > 0 and global_idx >= max_questions:
                break
            category = str(row.get("category", "")).strip()
            doc_name = str(row.get("doc_name", "")).strip()
            doc_id = safe_doc_id(category, doc_name)
            if doc_id not in doc_id_map:
                doc_ids.append(doc_id)
                doc_id_map[doc_id] = f"{category}/{doc_name}"
            local_idx = per_doc_question_idx[doc_id]
            per_doc_question_idx[doc_id] += 1
            qid = f"{doc_id}__q{local_idx:04d}"

            evidence_pages = [int(page) for page in normalize_list(row.get("evidence_page"))]
            gold_page_ids = [page_to_idx(page) for page in evidence_pages]
            gold_page_uids = [f"{doc_id}_page{page_idx}" for page_idx in gold_page_ids]
            bbox = normalize_list(row.get("bbox"))
            rel_bbox = normalize_list(row.get("rel_bbox"))
            subimg_type = normalize_list(row.get("subimg_type"))

            supporting_context = [
                {
                    "doc_id": doc_id,
                    "doc_part": str(types) if types else "evidence",
                    "page_idx": page_idx,
                    "page_id": page_idx,
                    "source_page_number": page_number,
                }
                for page_idx, page_number, types in zip(
                    gold_page_ids,
                    evidence_pages,
                    subimg_type or [None] * len(gold_page_ids),
                )
            ]
            if not supporting_context:
                supporting_context = [{"doc_id": doc_id, "doc_part": "evidence"}]

            mmqa_row = {
                "qid": qid,
                "question": str(row.get("query", "")).strip(),
                "answers": [
                    {
                        "answer": str(row.get("answer", "")).strip(),
                        "modality": "scientific_document",
                    }
                ],
                "metadata": {
                    "type": "SciEGQA",
                    "category": category,
                    "source": "SciEGQA-Bench",
                    "doc_name": doc_name,
                    "doc_id": doc_id,
                    "evidence_page_numbers": evidence_pages,
                    "gold_page_ids": gold_page_ids,
                    "gold_page_uids": gold_page_uids,
                    "bbox": bbox,
                    "rel_bbox": rel_bbox,
                    "subimg_type": subimg_type,
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
                        "doc_name": doc_name,
                        "category": category,
                        "evidence_page_numbers": evidence_pages,
                        "gold_page_ids": gold_page_ids,
                        "gold_page_uids": gold_page_uids,
                        "bbox": bbox,
                        "rel_bbox": rel_bbox,
                        "subimg_type": subimg_type,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    doc_ids_path.write_text(json.dumps(doc_ids, indent=2) + "\n", encoding="utf-8")
    doc_id_map_path.write_text(json.dumps(doc_id_map, indent=2) + "\n", encoding="utf-8")
    return doc_id_map


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    annotations_path, images_tar = resolve_inputs(args, output_root)
    rows = read_jsonl(annotations_path)
    if args.max_questions > 0:
        rows_for_docs = rows[: args.max_questions]
    else:
        rows_for_docs = rows

    image_root = Path(args.image_root) if args.image_root else output_root / "images_raw"
    if not args.skip_images:
        if images_tar is None and not image_root.exists():
            raise FileNotFoundError(
                "Could not find images.tar. Pass --images-tar, pass --image-root, or use --skip-images."
            )
        if images_tar is not None:
            extract_images(images_tar, image_root, args.overwrite_extract)
    else:
        image_root = None

    convert_annotations(
        rows=rows,
        output_root=output_root,
        split=args.split,
        max_questions=args.max_questions,
    )
    page_counts = build_doc_page_manifest(
        rows=rows_for_docs,
        output_root=output_root,
        image_root=image_root,
        split=args.split,
        overwrite_rendered_pages=args.overwrite_rendered_pages,
        pdf_dpi=args.pdf_dpi,
    )

    print(f"prepared_output_root={output_root}")
    print(f"annotations={annotations_path}")
    print(f"image_root={image_root}")
    print(f"doc_count={len(page_counts)}")
    print(f"page_count={sum(page_counts.values())}")


if __name__ == "__main__":
    main()
