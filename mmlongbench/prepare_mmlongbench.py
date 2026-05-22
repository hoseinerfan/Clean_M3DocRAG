#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import tarfile
from collections import Counter
from pathlib import Path
from urllib.parse import quote


DEFAULT_TASKS = ["longdocurl", "mmlongdoc", "slidevqa"]
DEFAULT_LENGTHS = ["K8", "K16", "K32", "K64", "K128"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare MMLongBench DocQA subsets for M3DocRAG-style page retrieval. "
            "The converter emits MMQA_<split>.jsonl, qids_<split>.jsonl, "
            "gold_pages_<split>.jsonl, and doc_pages_<split>.jsonl."
        )
    )
    parser.add_argument("--hf-repo", default="ZhaoweiWang/MMLongBench")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--snapshot-dir", default="")
    parser.add_argument("--text-tar", default="")
    parser.add_argument("--image-tar", default="")
    parser.add_argument("--mmlb-data-root", default="")
    parser.add_argument("--mmlb-image-root", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        choices=DEFAULT_TASKS,
        help="DocQA task to include. Repeatable. Defaults to all DocQA tasks.",
    )
    parser.add_argument(
        "--length",
        action="append",
        default=[],
        help="Context length to include, e.g. K8, K16, K32, K64, K128. Repeatable.",
    )
    parser.add_argument(
        "--test-file",
        action="append",
        default=[],
        help=(
            "Explicit MMLongBench JSONL test file, relative to mmlb_data or absolute. "
            "When provided, --task/--length are ignored."
        ),
    )
    parser.add_argument("--max-examples-per-file", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--overwrite-extract", action="store_true")
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Write manifests even if some page image paths do not exist.",
    )
    return parser.parse_args()


def normalize_length(raw: str) -> str:
    text = str(raw).strip().upper()
    if not text:
        return text
    return text if text.startswith("K") else f"K{text}"


def safe_doc_id(task: str, doc_name: str) -> str:
    return quote(f"{task}__{doc_name}", safe="._-")


def safe_qid(task: str, length: str, source_id: str, row_index: int) -> str:
    raw = source_id.strip() if source_id.strip() else f"row{row_index:06d}"
    return quote(f"mmlongbench__{task}__{length}__{raw}", safe="._-")


def read_jsonl(path: Path, max_examples: int = 0) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_examples > 0 and len(rows) >= max_examples:
                break
    return rows


def rel_or_abs(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def safe_extract_tar(tar_path: Path, output_root: Path, marker_name: str, overwrite: bool) -> None:
    marker = output_root / marker_name
    if marker.exists() and not overwrite:
        return
    output_root.mkdir(parents=True, exist_ok=True)
    root = output_root.resolve()
    with tarfile.open(tar_path, "r:*") as archive:
        for member in archive.getmembers():
            target = (output_root / member.name).resolve()
            try:
                target.relative_to(root)
            except ValueError:
                raise ValueError(f"Refusing to extract unsafe tar member: {member.name}")
        archive.extractall(output_root)
    marker.write_text(str(tar_path) + "\n", encoding="utf-8")


def download_snapshot(args: argparse.Namespace, snapshot_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=args.hf_repo,
        repo_type="dataset",
        local_dir=str(snapshot_dir),
        allow_patterns=["README.md", "0_mmlb_data.tar.gz", "5_docqa_image.tar.gz"],
    )


def resolve_tar(snapshot_dir: Path, explicit_path: str, name: str) -> Path | None:
    if explicit_path:
        path = Path(explicit_path)
        return path if path.exists() else None
    path = snapshot_dir / name
    if path.exists():
        return path
    matches = sorted(snapshot_dir.rglob(name))
    return matches[0] if matches else None


def resolve_roots(args: argparse.Namespace, output_root: Path) -> tuple[Path, Path]:
    snapshot_dir = Path(args.snapshot_dir) if args.snapshot_dir else output_root / "hf_snapshot"
    if args.download:
        download_snapshot(args, snapshot_dir)

    text_tar = resolve_tar(snapshot_dir, args.text_tar, "0_mmlb_data.tar.gz")
    image_tar = resolve_tar(snapshot_dir, args.image_tar, "5_docqa_image.tar.gz")
    if not args.skip_extract:
        if text_tar is not None:
            safe_extract_tar(text_tar, output_root, ".mmlongbench_text_extracted", args.overwrite_extract)
        if image_tar is not None:
            safe_extract_tar(image_tar, output_root, ".mmlongbench_docqa_images_extracted", args.overwrite_extract)

    data_root = Path(args.mmlb_data_root) if args.mmlb_data_root else output_root / "mmlb_data"
    image_root = Path(args.mmlb_image_root) if args.mmlb_image_root else output_root / "mmlb_image"
    if not data_root.exists():
        raise FileNotFoundError(
            f"Could not find MMLongBench text root: {data_root}. "
            "Pass --mmlb-data-root or use --download/--text-tar."
        )
    if not image_root.exists():
        raise FileNotFoundError(
            f"Could not find MMLongBench image root: {image_root}. "
            "Pass --mmlb-image-root or use --download/--image-tar."
        )
    return data_root, image_root


def infer_task_from_path(path: Path) -> str:
    name = path.name.lower()
    for task in DEFAULT_TASKS:
        if task in name:
            return task
    return path.stem.split("_")[0].lower()


def infer_length_from_path(path: Path) -> str:
    match = re.search(r"_K(\d+)", path.name, flags=re.IGNORECASE)
    return f"K{match.group(1)}" if match else "KUNKNOWN"


def selected_test_files(args: argparse.Namespace, data_root: Path) -> list[tuple[str, str, Path]]:
    if args.test_file:
        out = []
        for raw in args.test_file:
            path = Path(raw)
            if not path.is_absolute():
                path = data_root / raw
            out.append((infer_task_from_path(path), infer_length_from_path(path), path))
        return out

    tasks = args.task or DEFAULT_TASKS
    lengths = [normalize_length(length) for length in (args.length or DEFAULT_LENGTHS)]
    out = []
    for task in tasks:
        for length in lengths:
            out.append((task, length, data_root / "documentQA" / f"{task}_{length}.jsonl"))
    return out


def resolve_image_path(raw_path: str, image_root: Path, output_root: Path) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    candidates = [
        image_root / path,
        output_root / path,
        image_root / "documentQA" / path,
        output_root / "mmlb_image" / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return image_root / path


def parse_page_idx(raw_path: str, task: str, fallback_index: int) -> int:
    name = Path(str(raw_path)).name
    stem = Path(name).stem
    match = re.search(r"(?:^|[_-])page[_-]?(\d+)(?:$|[_-])", stem, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"page(\d+)", stem, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    numbers = [int(value) for value in re.findall(r"\d+", stem)]
    if task == "slidevqa" and numbers:
        if len(numbers) >= 2 and numbers[-1] in {512, 768, 1024, 1280, 1536, 2048}:
            return max(numbers[-2] - 1, 0)
        return max(numbers[-1] - 1, 0)
    return int(fallback_index)


def normalize_answer(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def normalize_int_list(value: object) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            value = json.loads(text)
        else:
            value = [part.strip() for part in text.split(",")]
    if not isinstance(value, list):
        value = [value]
    out = []
    for item in value:
        if item is None or item == "":
            continue
        out.append(int(item))
    return out


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    data_root, image_root = resolve_roots(args, output_root)
    test_files = selected_test_files(args, data_root)

    mmqa_path = output_root / f"MMQA_{args.split}.jsonl"
    qids_path = output_root / f"qids_{args.split}.jsonl"
    gold_pages_path = output_root / f"gold_pages_{args.split}.jsonl"
    doc_pages_path = output_root / f"doc_pages_{args.split}.jsonl"
    doc_ids_path = output_root / f"{args.split}_doc_ids.json"
    doc_id_map_path = output_root / "doc_id_map.json"
    summary_path = output_root / f"prepare_{args.split}_summary.json"

    doc_pages: dict[str, dict[int, dict]] = {}
    doc_id_map: dict[str, str] = {}
    used_qids: set[str] = set()
    missing_images: list[str] = []
    missing_gold_pages: list[dict] = []
    task_counts: Counter[str] = Counter()
    length_counts: Counter[str] = Counter()
    answer_format_counts: Counter[str] = Counter()

    total_examples = 0
    with mmqa_path.open("w", encoding="utf-8") as mmqa_out, qids_path.open(
        "w", encoding="utf-8"
    ) as qids_out, gold_pages_path.open("w", encoding="utf-8") as gold_out:
        for task, length, path in test_files:
            if not path.exists():
                raise FileNotFoundError(path)
            rows = read_jsonl(path, args.max_examples_per_file)
            for row_index, row in enumerate(rows):
                if args.max_examples > 0 and total_examples >= args.max_examples:
                    break
                page_list = list(row.get("page_list", []) or [])
                page_text_list = list(row.get("page_text_list", []) or [])
                doc_name = str(row.get("doc_name", "")).strip()
                if not doc_name:
                    first_page = str(page_list[0]) if page_list else "unknown"
                    doc_name = Path(first_page).parent.name or Path(first_page).stem or "unknown"
                doc_id = safe_doc_id(task, doc_name)
                doc_id_map.setdefault(doc_id, f"{task}/{doc_name}")
                page_idx_set = doc_pages.setdefault(doc_id, {})
                for page_order, raw_page_path in enumerate(page_list):
                    page_idx = parse_page_idx(str(raw_page_path), task, page_order)
                    image_path = resolve_image_path(str(raw_page_path), image_root, output_root)
                    if not image_path.exists() and not args.allow_missing_images:
                        missing_images.append(str(image_path))
                    page_text = (
                        str(page_text_list[page_order])
                        if page_order < len(page_text_list) and page_text_list[page_order] is not None
                        else ""
                    )
                    existing = page_idx_set.get(page_idx)
                    if existing is None or (page_text and not existing.get("text")):
                        page_idx_set[page_idx] = {
                            "doc_id": doc_id,
                            "doc_name": doc_name,
                            "source_task": task,
                            "page_idx": page_idx,
                            "page_number": page_idx + 1,
                            "page_uid": f"{doc_id}_page{page_idx}",
                            "image_path": rel_or_abs(image_path, output_root),
                            "text": page_text,
                            "ocr_text": page_text,
                            "source_image_path": str(raw_page_path),
                        }

                source_id = str(row.get("id", row.get("qid", ""))).strip()
                qid = safe_qid(task, length, source_id, total_examples)
                if qid in used_qids:
                    suffix = 1
                    base = qid
                    while qid in used_qids:
                        suffix += 1
                        qid = f"{base}__dup{suffix}"
                used_qids.add(qid)

                gold_page_ids = normalize_int_list(row.get("ans_page_list", row.get("answer_page_idx")))
                gold_page_uids = [f"{doc_id}_page{page_idx}" for page_idx in gold_page_ids]
                for page_idx, page_uid in zip(gold_page_ids, gold_page_uids):
                    if page_idx not in page_idx_set:
                        missing_gold_pages.append(
                            {
                                "qid": qid,
                                "doc_id": doc_id,
                                "page_idx": page_idx,
                                "page_uid": page_uid,
                            }
                        )
                supporting_context = [
                    {
                        "doc_id": doc_id,
                        "doc_part": "answer_page",
                        "page_idx": page_idx,
                        "page_id": page_idx,
                        "source_page_number": page_idx + 1,
                    }
                    for page_idx in gold_page_ids
                ]
                if not supporting_context:
                    supporting_context = [{"doc_id": doc_id, "doc_part": "document"}]

                answer_format = str(row.get("answer_format", "")).strip()
                answer_format_counts[answer_format or "UNKNOWN"] += 1
                task_counts[task] += 1
                length_counts[length] += 1
                mmqa_row = {
                    "qid": qid,
                    "question": str(row.get("question", "")).strip(),
                    "answers": [
                        {
                            "answer": normalize_answer(row.get("answer")),
                            "modality": "document",
                        }
                    ],
                    "metadata": {
                        "type": "MMLongBench-DocQA",
                        "source": "MMLongBench",
                        "source_task": task,
                        "context_length": length,
                        "source_file": rel_or_abs(path, data_root),
                        "source_id": source_id,
                        "doc_name": doc_name,
                        "doc_id": doc_id,
                        "answer_format": answer_format,
                        "gold_page_ids": gold_page_ids,
                        "gold_page_uids": gold_page_uids,
                        "original_answer": row.get("answer"),
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
                            "source_task": task,
                            "context_length": length,
                            "gold_page_ids": gold_page_ids,
                            "gold_page_uids": gold_page_uids,
                            "answer_format": answer_format,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                total_examples += 1
            if args.max_examples > 0 and total_examples >= args.max_examples:
                break

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
        "mmlb_data_root": str(data_root),
        "mmlb_image_root": str(image_root),
        "split": args.split,
        "test_files": [
            {"task": task, "length": length, "path": str(path)}
            for task, length, path in test_files
            if path.exists()
        ],
        "doc_count": len(doc_ids),
        "page_count": page_count,
        "qa_count": total_examples,
        "missing_gold_page_count": len(missing_gold_pages),
        "missing_gold_pages_sample": missing_gold_pages[:20],
        "task_counts": dict(sorted(task_counts.items())),
        "length_counts": dict(sorted(length_counts.items())),
        "answer_format_counts": dict(sorted(answer_format_counts.items())),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"prepared_output_root={output_root}")
    print(f"mmlb_data_root={data_root}")
    print(f"mmlb_image_root={image_root}")
    print(f"doc_count={len(doc_ids)}")
    print(f"page_count={page_count}")
    print(f"qa_count={total_examples}")
    print(f"missing_gold_page_count={len(missing_gold_pages)}")
    print(f"task_counts={dict(sorted(task_counts.items()))}")
    print(f"length_counts={dict(sorted(length_counts.items()))}")
    print(f"answer_format_counts={dict(sorted(answer_format_counts.items()))}")
    print(f"saved_summary={summary_path}")


if __name__ == "__main__":
    main()
