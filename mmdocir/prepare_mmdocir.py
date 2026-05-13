#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import json
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare MMDocIR Evaluation Dataset artifacts for M3DocRAG-style retrieval. "
            "The output keeps exact gold page/layout labels while also emitting an "
            "MMQA_dev.jsonl-compatible file."
        )
    )
    parser.add_argument("--hf-repo", default="MMDocIR/MMDocIR_Evaluation_Dataset")
    parser.add_argument("--download", action="store_true", help="Download the HF dataset snapshot first.")
    parser.add_argument("--snapshot-dir", default="", help="Local Hugging Face snapshot directory.")
    parser.add_argument("--annotations-jsonl", default="", help="Override path to MMDocIR_annotations.jsonl.")
    parser.add_argument("--pages-parquet", default="", help="Override path to MMDocIR_pages.parquet.")
    parser.add_argument("--output-root", required=True, help="Output root, e.g. $LOCAL_DATA_DIR/mm-docir.")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--max-docs", type=int, default=0, help="Optional smoke-test cap.")
    parser.add_argument("--max-questions", type=int, default=0, help="Optional smoke-test cap.")
    parser.add_argument("--skip-images", action="store_true", help="Only write annotations, not page JPEGs.")
    parser.add_argument("--overwrite-images", action="store_true")
    return parser.parse_args()


def normalize_doc_name(doc_name: str) -> str:
    text = str(doc_name).strip()
    if text.lower().endswith(".pdf"):
        text = text[:-4]
    return text


def safe_doc_id(doc_name: str) -> str:
    return quote(normalize_doc_name(doc_name), safe="._-")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def maybe_parse_json(value):
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") or text.startswith("{"):
            return json.loads(text)
    return value


def find_file(root: Path, candidates: list[str]) -> Path | None:
    for name in candidates:
        direct = root / name
        if direct.exists():
            return direct
    names = set(candidates)
    for path in root.rglob("*"):
        if path.name in names:
            return path
    return None


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path | None]:
    snapshot_dir = Path(args.snapshot_dir) if args.snapshot_dir else Path(args.output_root) / "hf_snapshot"
    if args.download:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=args.hf_repo,
            repo_type="dataset",
            local_dir=str(snapshot_dir),
            allow_patterns=["*.jsonl", "*.json", "*.parquet", "*.md"],
        )

    annotations_path = Path(args.annotations_jsonl) if args.annotations_jsonl else None
    if annotations_path is None:
        annotations_path = find_file(snapshot_dir, ["MMDocIR_annotations.jsonl"])
    if annotations_path is None or not annotations_path.exists():
        raise FileNotFoundError(
            "Could not find MMDocIR_annotations.jsonl. Pass --annotations-jsonl or use --download."
        )

    pages_path = Path(args.pages_parquet) if args.pages_parquet else None
    if pages_path is None:
        pages_path = find_file(snapshot_dir, ["MMDocIR_pages.parquet"])
    if pages_path is not None and not pages_path.exists():
        pages_path = None
    return annotations_path, pages_path


def normalize_page_ids(value) -> list[int]:
    value = maybe_parse_json(value)
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    page_ids = []
    for item in value:
        if isinstance(item, dict):
            item = item.get("page_id", item.get("page", item.get("page_idx")))
        if item is None:
            continue
        page_ids.append(int(item))
    return sorted(set(page_ids))


def make_qid(doc_id: str, question_idx: int) -> str:
    return f"{doc_id}__q{question_idx:04d}"


def convert_annotations(
    *,
    annotations_path: Path,
    output_root: Path,
    split: str,
    max_docs: int,
    max_questions: int,
) -> dict[str, str]:
    rows = read_jsonl(annotations_path)
    if max_docs > 0:
        rows = rows[:max_docs]

    mmqa_path = output_root / f"MMQA_{split}.jsonl"
    qids_path = output_root / f"qids_{split}.jsonl"
    gold_pages_path = output_root / f"gold_pages_{split}.jsonl"
    doc_ids_path = output_root / f"{split}_doc_ids.json"
    doc_id_map_path = output_root / "doc_id_map.json"

    doc_ids: list[str] = []
    doc_id_map: dict[str, str] = {}
    question_count = 0

    with mmqa_path.open("w", encoding="utf-8") as mmqa_out, qids_path.open(
        "w", encoding="utf-8"
    ) as qids_out, gold_pages_path.open("w", encoding="utf-8") as gold_out:
        for doc_row in rows:
            original_doc_name = str(doc_row["doc_name"]).strip()
            normalized_doc_name = normalize_doc_name(original_doc_name)
            doc_id = safe_doc_id(normalized_doc_name)
            if doc_id not in doc_id_map:
                doc_ids.append(doc_id)
                doc_id_map[doc_id] = original_doc_name

            domain = str(doc_row.get("domain", "")).strip()
            questions = maybe_parse_json(doc_row.get("questions", [])) or []
            for q_idx, qa in enumerate(questions):
                if max_questions > 0 and question_count >= max_questions:
                    break
                qa = maybe_parse_json(qa)
                if not isinstance(qa, dict):
                    continue
                qid = make_qid(doc_id, q_idx)
                question = str(qa.get("Q", qa.get("question", ""))).strip()
                answer = str(qa.get("A", qa.get("answer", ""))).strip()
                question_type = str(qa.get("type", "UNKNOWN")).strip() or "UNKNOWN"
                page_ids = normalize_page_ids(qa.get("page_id", []))
                layout_mapping = maybe_parse_json(qa.get("layout_mapping", [])) or []
                gold_page_uids = [f"{doc_id}_page{page_id}" for page_id in page_ids]

                supporting_context = [
                    {
                        "doc_id": doc_id,
                        "doc_part": question_type,
                        "page_idx": page_id,
                        "page_id": page_id,
                    }
                    for page_id in page_ids
                ]
                if not supporting_context:
                    supporting_context = [{"doc_id": doc_id, "doc_part": question_type}]

                mmqa_row = {
                    "qid": qid,
                    "question": question,
                    "answers": [{"answer": answer, "modality": question_type}],
                    "metadata": {
                        "type": question_type,
                        "domain": domain,
                        "source": "MMDocIR",
                        "doc_name": original_doc_name,
                        "normalized_doc_name": normalized_doc_name,
                        "doc_id": doc_id,
                        "gold_page_ids": page_ids,
                        "gold_page_uids": gold_page_uids,
                        "layout_mapping": layout_mapping,
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
                            "doc_name": original_doc_name,
                            "gold_page_ids": page_ids,
                            "gold_page_uids": gold_page_uids,
                            "layout_mapping": layout_mapping,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                question_count += 1
            if max_questions > 0 and question_count >= max_questions:
                break

    doc_ids_path.write_text(json.dumps(doc_ids, indent=2) + "\n", encoding="utf-8")
    doc_id_map_path.write_text(json.dumps(doc_id_map, indent=2) + "\n", encoding="utf-8")
    return doc_id_map


def extract_trailing_int(value) -> int | None:
    if value is None:
        return None
    text = str(value)
    match = re.search(r"(\d+)(?:\.[A-Za-z0-9]+)?$", text)
    if match:
        return int(match.group(1))
    return None


def decode_image_bytes(value) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("data:"):
            text = text.split(",", 1)[1]
        return base64.b64decode(text)
    raise TypeError(f"Unsupported image_binary type: {type(value)}")


def iter_parquet_rows(path: Path, batch_size: int = 64):
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        data = batch.to_pydict()
        keys = list(data)
        for idx in range(batch.num_rows):
            yield {key: data[key][idx] for key in keys}


def export_page_images(
    *,
    pages_path: Path,
    output_root: Path,
    doc_id_map: dict[str, str],
    split: str,
    overwrite_images: bool,
) -> None:
    pages_dir = output_root / f"pages_{split}"
    manifest_path = output_root / f"doc_pages_{split}.jsonl"
    original_to_doc_id = {}
    for doc_id, original in doc_id_map.items():
        original_to_doc_id[original] = doc_id
        original_to_doc_id[normalize_doc_name(original)] = doc_id
    fallback_page_counters: dict[str, int] = defaultdict(int)
    manifest_rows: list[dict] = []

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for row in iter_parquet_rows(pages_path):
            original_doc_name = str(row.get("doc_name", "")).strip()
            if not original_doc_name:
                continue
            normalized_doc_name = normalize_doc_name(original_doc_name)
            doc_id = original_to_doc_id.get(
                original_doc_name,
                original_to_doc_id.get(normalized_doc_name, safe_doc_id(normalized_doc_name)),
            )
            page_idx = row.get("page_id")
            if page_idx is None:
                page_idx = extract_trailing_int(row.get("passage_id"))
            if page_idx is None:
                page_idx = extract_trailing_int(row.get("image_path"))
            if page_idx is None:
                page_idx = fallback_page_counters[doc_id]
                fallback_page_counters[doc_id] += 1
            page_idx = int(page_idx)

            image_binary = row.get("image_binary")
            if image_binary is None:
                continue
            image_rel_path = Path(f"pages_{split}") / doc_id / f"{page_idx}.jpg"
            image_path = output_root / image_rel_path
            image_path.parent.mkdir(parents=True, exist_ok=True)
            if overwrite_images or not image_path.exists():
                image_path.write_bytes(decode_image_bytes(image_binary))

            manifest_row = {
                "doc_id": doc_id,
                "doc_name": original_doc_name,
                "domain": row.get("domain", ""),
                "page_idx": page_idx,
                "page_uid": f"{doc_id}_page{page_idx}",
                "image_path": str(image_rel_path),
                "source_image_path": row.get("image_path", ""),
                "ocr_text": row.get("ocr_text", ""),
                "vlm_text": row.get("vlm_text", ""),
            }
            manifest_rows.append(manifest_row)

        manifest_rows.sort(key=lambda item: (item["doc_id"], int(item["page_idx"])))
        for item in manifest_rows:
            manifest.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    annotations_path, pages_path = resolve_inputs(args)
    doc_id_map = convert_annotations(
        annotations_path=annotations_path,
        output_root=output_root,
        split=args.split,
        max_docs=args.max_docs,
        max_questions=args.max_questions,
    )
    if not args.skip_images:
        if pages_path is None:
            raise FileNotFoundError(
                "Could not find MMDocIR_pages.parquet. Pass --pages-parquet or use --skip-images."
            )
        export_page_images(
            pages_path=pages_path,
            output_root=output_root,
            doc_id_map=doc_id_map,
            split=args.split,
            overwrite_images=args.overwrite_images,
        )

    print(f"prepared_output_root={output_root}")
    print(f"annotations={annotations_path}")
    print(f"pages={pages_path}")


if __name__ == "__main__":
    main()
