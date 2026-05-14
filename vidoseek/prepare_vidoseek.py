#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare ViDoSeek for M3DocRAG-style page retrieval. The converter "
            "keeps ViDoSeek's 1-based reference_page labels in metadata and emits "
            "zero-based page_idx labels for retrieval evaluation."
        )
    )
    parser.add_argument("--hf-repo", default="Qiuchen-Wang/ViDoSeek")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--snapshot-dir", default="")
    parser.add_argument("--annotations-json", default="")
    parser.add_argument("--pdf-zip", default="")
    parser.add_argument("--pdf-root", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--skip-pdfs", action="store_true")
    parser.add_argument("--overwrite-extract", action="store_true")
    parser.add_argument("--overwrite-rendered-pages", action="store_true")
    parser.add_argument("--pdf-dpi", type=int, default=144)
    return parser.parse_args()


def normalize_file_name(file_name: str) -> str:
    return Path(str(file_name).strip()).name


def safe_doc_id(file_name: str) -> str:
    return quote(Path(normalize_file_name(file_name)).stem, safe="._-")


def page_to_idx(page_number: int) -> int:
    return int(page_number) - 1


def normalize_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_int_list(value) -> list[int]:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") or text.startswith("{"):
            value = json.loads(text)
        elif "," in text:
            value = [part.strip() for part in text.split(",")]
    out = []
    for item in normalize_list(value):
        if item is None or item == "":
            continue
        out.append(int(item))
    return out


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
        allow_patterns=["README.md", "vidoseek.json", "vidoseek_pdf_document.zip"],
    )


def resolve_inputs(args: argparse.Namespace, output_root: Path) -> tuple[Path, Path | None]:
    snapshot_dir = Path(args.snapshot_dir) if args.snapshot_dir else output_root / "hf_snapshot"
    if args.download:
        download_snapshot(args, snapshot_dir)

    annotations_path = Path(args.annotations_json) if args.annotations_json else None
    if annotations_path is None:
        annotations_path = find_file(snapshot_dir, ["vidoseek.json"], suffix=".json")
    if annotations_path is None or not annotations_path.exists():
        raise FileNotFoundError(
            "Could not find vidoseek.json. Pass --annotations-json or use --download."
        )

    pdf_zip = Path(args.pdf_zip) if args.pdf_zip else None
    if pdf_zip is None:
        pdf_zip = find_file(snapshot_dir, ["vidoseek_pdf_document.zip"], suffix=".zip")
    if pdf_zip is not None and not pdf_zip.exists():
        pdf_zip = None
    return annotations_path, pdf_zip


def read_annotation_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return [unwrap_row(row) for row in rows]

    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        for key in ("examples", "data", "annotations", "rows"):
            if isinstance(payload.get(key), list):
                rows = payload[key]
                break
        else:
            list_values = [value for value in payload.values() if isinstance(value, list)]
            if len(list_values) == 1:
                rows = list_values[0]
            else:
                raise ValueError(f"Could not infer ViDoSeek rows from JSON object keys: {sorted(payload)}")
    else:
        raise TypeError(f"Unsupported ViDoSeek annotation JSON type: {type(payload)}")
    return [unwrap_row(row) for row in rows]


def unwrap_row(row: dict) -> dict:
    if isinstance(row, dict) and isinstance(row.get("examples"), dict):
        return row["examples"]
    return row


def row_meta(row: dict) -> dict:
    meta = row.get("meta_info", {})
    return meta if isinstance(meta, dict) else {}


def row_file_name(row: dict) -> str:
    meta = row_meta(row)
    file_name = meta.get("file_name", row.get("file_name", ""))
    file_name = normalize_file_name(str(file_name))
    if not file_name:
        raise ValueError(f"Missing file_name in row: {row}")
    return file_name


def filter_rows(rows: list[dict], max_docs: int, max_questions: int) -> list[dict]:
    kept_rows = []
    kept_docs: list[str] = []
    kept_doc_set: set[str] = set()
    for row in rows:
        file_name = row_file_name(row)
        doc_id = safe_doc_id(file_name)
        if max_docs > 0 and doc_id not in kept_doc_set and len(kept_docs) >= max_docs:
            continue
        if doc_id not in kept_doc_set:
            kept_docs.append(doc_id)
            kept_doc_set.add(doc_id)
        kept_rows.append(row)
        if max_questions > 0 and len(kept_rows) >= max_questions:
            break
    return kept_rows


def safe_extract_zip(zip_path: Path, output_dir: Path, overwrite: bool) -> None:
    marker = output_dir / ".extracted"
    if marker.exists() and not overwrite:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        root = output_dir.resolve()
        for info in archive.infolist():
            target = (output_dir / info.filename).resolve()
            try:
                target.relative_to(root)
            except ValueError:
                raise ValueError(f"Refusing to extract unsafe zip member: {info.filename}")
        archive.extractall(output_dir)
    marker.write_text(str(zip_path) + "\n", encoding="utf-8")


def find_pdf_path(pdf_root: Path, file_name: str) -> Path | None:
    normalized = normalize_file_name(file_name)
    direct_candidates = [
        pdf_root / normalized,
        pdf_root / "vidoseek_pdf_document" / normalized,
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate
    matches = sorted(pdf_root.rglob(normalized))
    if matches:
        return matches[0]
    stem = Path(normalized).stem
    matches = sorted(pdf_root.rglob(f"{stem}*.pdf"))
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


def rel_or_abs(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def build_doc_page_manifest(
    *,
    rows: list[dict],
    output_root: Path,
    pdf_root: Path | None,
    split: str,
    overwrite_rendered_pages: bool,
    pdf_dpi: int,
) -> dict[str, int]:
    docs: dict[str, str] = {}
    for row in rows:
        file_name = row_file_name(row)
        docs[safe_doc_id(file_name)] = file_name

    page_counts: dict[str, int] = {}
    missing_pdfs: list[tuple[str, str]] = []
    manifest_path = output_root / f"doc_pages_{split}.jsonl"
    with manifest_path.open("w", encoding="utf-8") as out:
        for doc_id, file_name in sorted(docs.items()):
            page_records = []
            if pdf_root is not None:
                pdf_path = find_pdf_path(pdf_root, file_name)
                if pdf_path is not None:
                    rendered_dir = output_root / f"pages_{split}" / doc_id
                    page_records = render_pdf_pages(
                        pdf_path=pdf_path,
                        page_dir=rendered_dir,
                        overwrite=overwrite_rendered_pages,
                        dpi=pdf_dpi,
                    )
                else:
                    missing_pdfs.append((doc_id, file_name))
            page_records.sort(key=lambda item: item[0])
            page_counts[doc_id] = len(page_records)
            for page_number, image_path in page_records:
                page_idx = page_to_idx(page_number)
                out.write(
                    json.dumps(
                        {
                            "doc_id": doc_id,
                            "doc_name": Path(file_name).stem,
                            "file_name": file_name,
                            "page_idx": page_idx,
                            "page_number": page_number,
                            "page_uid": f"{doc_id}_page{page_idx}",
                            "image_path": rel_or_abs(image_path, output_root),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    if missing_pdfs:
        raise FileNotFoundError(f"Missing {len(missing_pdfs)} PDFs, first missing: {missing_pdfs[:10]}")
    return page_counts


def convert_annotations(
    *,
    rows: list[dict],
    output_root: Path,
    split: str,
) -> dict[str, str]:
    mmqa_path = output_root / f"MMQA_{split}.jsonl"
    qids_path = output_root / f"qids_{split}.jsonl"
    gold_pages_path = output_root / f"gold_pages_{split}.jsonl"
    doc_ids_path = output_root / f"{split}_doc_ids.json"
    doc_id_map_path = output_root / "doc_id_map.json"

    doc_ids: list[str] = []
    doc_id_map: dict[str, str] = {}
    per_doc_question_idx: dict[str, int] = defaultdict(int)
    used_qids: set[str] = set()

    with mmqa_path.open("w", encoding="utf-8") as mmqa_out, qids_path.open(
        "w", encoding="utf-8"
    ) as qids_out, gold_pages_path.open("w", encoding="utf-8") as gold_out:
        for row in rows:
            meta = row_meta(row)
            file_name = row_file_name(row)
            doc_id = safe_doc_id(file_name)
            if doc_id not in doc_id_map:
                doc_ids.append(doc_id)
                doc_id_map[doc_id] = file_name

            source_qid = str(row.get("uid", "")).strip()
            if source_qid:
                qid_base = quote(source_qid, safe="._-")
            else:
                local_idx = per_doc_question_idx[doc_id]
                qid_base = f"{doc_id}__q{local_idx:04d}"
            per_doc_question_idx[doc_id] += 1
            qid = qid_base
            suffix = 1
            while qid in used_qids:
                suffix += 1
                qid = f"{qid_base}__dup{suffix}"
            used_qids.add(qid)

            reference_pages = normalize_int_list(meta.get("reference_page", row.get("reference_page")))
            gold_page_ids = [page_to_idx(page) for page in reference_pages]
            gold_page_uids = [f"{doc_id}_page{page_idx}" for page_idx in gold_page_ids]
            query_type = str(meta.get("query_type", row.get("query_type", ""))).strip()
            source_type = str(meta.get("source_type", row.get("source_type", ""))).strip()

            supporting_context = [
                {
                    "doc_id": doc_id,
                    "doc_part": source_type or query_type or "evidence",
                    "page_idx": page_idx,
                    "page_id": page_idx,
                    "source_page_number": page_number,
                }
                for page_idx, page_number in zip(gold_page_ids, reference_pages)
            ]
            if not supporting_context:
                supporting_context = [{"doc_id": doc_id, "doc_part": source_type or query_type or "evidence"}]

            answer = str(row.get("reference_answer", row.get("answer", ""))).strip()
            question = str(row.get("query", row.get("question", ""))).strip()
            mmqa_row = {
                "qid": qid,
                "question": question,
                "answers": [{"answer": answer, "modality": source_type or "visual_document"}],
                "metadata": {
                    "type": "ViDoSeek",
                    "source": "ViDoSeek",
                    "doc_name": Path(file_name).stem,
                    "file_name": file_name,
                    "doc_id": doc_id,
                    "source_uid": source_qid,
                    "query_type": query_type,
                    "source_type": source_type,
                    "reference_page_numbers": reference_pages,
                    "gold_page_ids": gold_page_ids,
                    "gold_page_uids": gold_page_uids,
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
                        "doc_name": Path(file_name).stem,
                        "file_name": file_name,
                        "query_type": query_type,
                        "source_type": source_type,
                        "reference_page_numbers": reference_pages,
                        "gold_page_ids": gold_page_ids,
                        "gold_page_uids": gold_page_uids,
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

    annotations_path, pdf_zip = resolve_inputs(args, output_root)
    rows = filter_rows(
        read_annotation_rows(annotations_path),
        max_docs=args.max_docs,
        max_questions=args.max_questions,
    )

    pdf_root = Path(args.pdf_root) if args.pdf_root else output_root / "pdfs_raw"
    if not args.skip_pdfs:
        if pdf_zip is None and not pdf_root.exists():
            raise FileNotFoundError(
                "Could not find vidoseek_pdf_document.zip. Pass --pdf-zip, pass --pdf-root, or use --skip-pdfs."
            )
        if pdf_zip is not None:
            safe_extract_zip(pdf_zip, pdf_root, args.overwrite_extract)
    else:
        pdf_root = None

    convert_annotations(rows=rows, output_root=output_root, split=args.split)
    page_counts = build_doc_page_manifest(
        rows=rows,
        output_root=output_root,
        pdf_root=pdf_root,
        split=args.split,
        overwrite_rendered_pages=args.overwrite_rendered_pages,
        pdf_dpi=args.pdf_dpi,
    )

    query_type_counts = defaultdict(int)
    source_type_counts = defaultdict(int)
    for row in rows:
        meta = row_meta(row)
        query_type_counts[str(meta.get("query_type", row.get("query_type", ""))).strip() or "UNKNOWN"] += 1
        source_type_counts[str(meta.get("source_type", row.get("source_type", ""))).strip() or "UNKNOWN"] += 1

    print(f"prepared_output_root={output_root}")
    print(f"annotations={annotations_path}")
    print(f"pdf_root={pdf_root}")
    print(f"doc_count={len(page_counts)}")
    print(f"page_count={sum(page_counts.values())}")
    print(f"qa_count={len(rows)}")
    print(f"query_type_counts={dict(sorted(query_type_counts.items()))}")
    print(f"source_type_counts={dict(sorted(source_type_counts.items()))}")


if __name__ == "__main__":
    main()
