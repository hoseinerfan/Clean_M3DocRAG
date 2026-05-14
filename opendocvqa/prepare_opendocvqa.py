#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

from PIL import Image
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare OpenDocVQA + OpenDocVQA-Corpus for M3DocRAG-style page retrieval. "
            "The corpus stores individual images, so this converter groups images into "
            "artificial multi-page packs while preserving exact page-level gold labels."
        )
    )
    parser.add_argument("--qa-repo", default="NTT-hil-insight/OpenDocVQA")
    parser.add_argument("--corpus-repo", default="NTT-hil-insight/OpenDocVQA-Corpus")
    parser.add_argument("--qa-config", default="default")
    parser.add_argument("--qa-split", default="test")
    parser.add_argument("--corpus-split", default="train")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument(
        "--dataset-name",
        action="append",
        default=[],
        help="Optional dataset_name filter. Repeat for multiple names, e.g. --dataset-name infovqa.",
    )
    parser.add_argument(
        "--corpus-scope",
        default="all",
        choices=["all", "relevant_only"],
        help=(
            "Use the full filtered corpus, or only documents referenced by retained queries. "
            "relevant_only is useful for smoke tests but is not a real retrieval benchmark."
        ),
    )
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--max-corpus-docs", type=int, default=0)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--overwrite-images", action="store_true")
    parser.add_argument("--streaming-corpus", action="store_true")
    parser.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or "",
        help="Optional Hugging Face token. The gated corpus also works with an existing HF login.",
    )
    return parser.parse_args()


def configure_hf_cache(cache_dir: Path | None) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    hf_home = cache_dir / "hf_home"
    hub_cache = hf_home / "hub"
    datasets_cache = hf_home / "datasets"
    transformers_cache = hf_home / "transformers"
    xdg_cache = cache_dir / "xdg_cache"
    for path in (hf_home, hub_cache, datasets_cache, transformers_cache, xdg_cache):
        path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(datasets_cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_cache))
    os.environ.setdefault("HF_HUB_CACHE", str(hub_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(transformers_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))


def load_hf_dataset(
    *,
    repo_id: str,
    config: str | None,
    split: str,
    cache_dir: Path | None,
    token: str,
    streaming: bool = False,
):
    from datasets import load_dataset

    kwargs = {"split": split}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if token:
        kwargs["token"] = token
    if streaming:
        kwargs["streaming"] = True

    args = [repo_id]
    if config:
        args.append(config)
    try:
        return load_dataset(*args, **kwargs)
    except TypeError:
        if "token" in kwargs:
            kwargs["use_auth_token"] = kwargs.pop("token")
        return load_dataset(*args, **kwargs)


def jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        return jsonable(value.item())
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if isinstance(value, bytes):
        return "<bytes>"
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    return str(value)


def normalize_list(value) -> list:
    value = jsonable(value)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_doc_id(value) -> str:
    return str(jsonable(value)).strip()


def safe_file_stem(value: str) -> str:
    return quote(str(value), safe="._-")


def pack_doc_id(pack_idx: int) -> str:
    return f"opendocvqa_pack_{pack_idx:06d}"


def query_dataset_names(row: dict) -> list[str]:
    names = [str(item).strip() for item in normalize_list(row.get("dataset_names")) if str(item).strip()]
    if not names and row.get("dataset_name"):
        names = [str(row["dataset_name"]).strip()]
    return names


def corpus_dataset_name(row: dict) -> str:
    return str(row.get("dataset_name", "") or "").strip()


def keep_query(row: dict, dataset_filters: set[str]) -> bool:
    if not dataset_filters:
        return True
    names = {name.lower() for name in query_dataset_names(row)}
    return bool(names & dataset_filters)


def keep_corpus_row(row: dict, dataset_filters: set[str], wanted_doc_ids: set[str] | None) -> bool:
    doc_id = normalize_doc_id(row.get("doc_id"))
    if wanted_doc_ids is not None and doc_id not in wanted_doc_ids:
        return False
    if wanted_doc_ids is not None:
        return True
    if not dataset_filters:
        return True
    return corpus_dataset_name(row).lower() in dataset_filters


def doc_id_variants(value: str) -> set[str]:
    text = normalize_doc_id(value)
    if not text:
        return set()
    variants = {text}
    path = Path(text)
    variants.add(path.name)
    variants.add(path.stem)
    if "/" in text:
        variants.add(text.split("/", 1)[1])
        variants.add(text.rsplit("/", 1)[-1])
    for ext in (".png", ".jpg", ".jpeg", ".pdf"):
        if text.lower().endswith(ext):
            variants.add(text[: -len(ext)])
    return {item for item in variants if item}


def save_image(value, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.MAX_IMAGE_PIXELS = None
    if isinstance(value, Image.Image):
        image = value
    elif isinstance(value, dict):
        if value.get("bytes") is not None:
            image = Image.open(BytesIO(value["bytes"]))
        elif value.get("path"):
            image = Image.open(value["path"])
        else:
            raise TypeError(f"Unsupported image dict keys: {sorted(value)}")
    elif isinstance(value, (bytes, bytearray, memoryview)):
        image = Image.open(BytesIO(bytes(value)))
    elif isinstance(value, str):
        image = Image.open(value)
    else:
        raise TypeError(f"Unsupported image value type: {type(value)}")
    image.convert("RGB").save(path, quality=95)


def read_queries(
    *,
    qa_repo: str,
    qa_config: str,
    qa_split: str,
    cache_dir: Path | None,
    token: str,
    dataset_filters: set[str],
    max_queries: int,
) -> tuple[list[dict], set[str]]:
    dataset = load_hf_dataset(
        repo_id=qa_repo,
        config=qa_config or None,
        split=qa_split,
        cache_dir=cache_dir,
        token=token,
    )
    rows = []
    wanted_doc_ids: set[str] = set()
    for row in tqdm(dataset, desc="OpenDocVQA queries"):
        row = dict(row)
        if not keep_query(row, dataset_filters):
            continue
        raw_relevant_doc_ids = [normalize_doc_id(item) for item in normalize_list(row.get("relevant_doc_ids"))]
        relevant_doc_ids = [item for item in raw_relevant_doc_ids if item]
        if not relevant_doc_ids:
            continue
        row["_relevant_doc_ids"] = relevant_doc_ids
        row["_dataset_names"] = query_dataset_names(row)
        rows.append(row)
        for doc_id in relevant_doc_ids:
            wanted_doc_ids.update(doc_id_variants(doc_id))
        if max_queries > 0 and len(rows) >= max_queries:
            break
    return rows, wanted_doc_ids


def write_outputs(
    *,
    output_root: Path,
    split: str,
    query_rows: list[dict],
    source_to_page: dict[str, dict],
    doc_ids: list[str],
    doc_id_map: dict[str, dict],
) -> tuple[int, int]:
    mmqa_path = output_root / f"MMQA_{split}.jsonl"
    qids_path = output_root / f"qids_{split}.jsonl"
    gold_pages_path = output_root / f"gold_pages_{split}.jsonl"
    doc_ids_path = output_root / f"{split}_doc_ids.json"
    doc_id_map_path = output_root / "doc_id_map.json"
    source_map_path = output_root / "source_doc_id_map.json"

    missing_gold = 0
    qid_count = 0
    used_qids: set[str] = set()
    with mmqa_path.open("w", encoding="utf-8") as mmqa_out, qids_path.open(
        "w", encoding="utf-8"
    ) as qids_out, gold_pages_path.open("w", encoding="utf-8") as gold_out:
        for idx, row in enumerate(query_rows):
            source_qid = str(row.get("query_id", "") or idx).strip()
            qid_base = f"opendocvqa__q{safe_file_stem(source_qid)}"
            qid = qid_base
            duplicate_idx = 1
            while qid in used_qids:
                duplicate_idx += 1
                qid = f"{qid_base}__dup{duplicate_idx}"
            used_qids.add(qid)

            gold_items = []
            for source_doc_id in row["_relevant_doc_ids"]:
                page = lookup_source_page(source_to_page, source_doc_id)
                if page is None:
                    missing_gold += 1
                    continue
                gold_items.append(page)
            if not gold_items:
                continue

            gold_page_ids = [page["page_idx"] for page in gold_items]
            gold_page_uids = [page["page_uid"] for page in gold_items]
            supporting_context = [
                {
                    "doc_id": page["doc_id"],
                    "doc_part": page.get("dataset_name") or "evidence",
                    "page_idx": page["page_idx"],
                    "page_id": page["page_idx"],
                    "source_doc_id": page["source_doc_id"],
                    "dataset_name": page.get("dataset_name", ""),
                }
                for page in gold_items
            ]
            answers = [str(answer).strip() for answer in normalize_list(row.get("answers")) if str(answer).strip()]
            mmqa_row = {
                "qid": qid,
                "question": str(row.get("query", "") or "").strip(),
                "answers": [{"answer": answer, "modality": "document_image"} for answer in answers],
                "metadata": {
                    "type": "OpenDocVQA",
                    "source": "OpenDocVQA",
                    "source_query_id": source_qid,
                    "dataset_names": row.get("_dataset_names", []),
                    "source_relevant_doc_ids": row["_relevant_doc_ids"],
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
                        "source_query_id": source_qid,
                        "dataset_names": row.get("_dataset_names", []),
                        "source_relevant_doc_ids": row["_relevant_doc_ids"],
                        "gold_page_ids": gold_page_ids,
                        "gold_page_uids": gold_page_uids,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            qid_count += 1

    doc_ids_path.write_text(json.dumps(doc_ids, indent=2) + "\n", encoding="utf-8")
    doc_id_map_path.write_text(json.dumps(doc_id_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    source_map_path.write_text(
        json.dumps(dedupe_source_to_page(source_to_page), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return qid_count, missing_gold


def lookup_source_page(source_to_page: dict[str, dict], source_doc_id: str) -> dict | None:
    for variant in doc_id_variants(source_doc_id):
        page = source_to_page.get(variant)
        if page is not None:
            return page
    return None


def dedupe_source_to_page(source_to_page: dict[str, dict]) -> dict[str, dict]:
    deduped: dict[str, dict] = {}
    for page in source_to_page.values():
        deduped[page["source_doc_id"]] = page
    return deduped


def main() -> None:
    args = parse_args()
    if args.group_size <= 0:
        raise ValueError("--group-size must be positive.")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    configure_hf_cache(cache_dir)

    dataset_filters = {name.strip().lower() for name in args.dataset_name if name.strip()}
    query_rows, relevant_doc_ids = read_queries(
        qa_repo=args.qa_repo,
        qa_config=args.qa_config,
        qa_split=args.qa_split,
        cache_dir=cache_dir,
        token=args.hf_token,
        dataset_filters=dataset_filters,
        max_queries=args.max_queries,
    )
    wanted_doc_ids = relevant_doc_ids if args.corpus_scope == "relevant_only" else None

    corpus = load_hf_dataset(
        repo_id=args.corpus_repo,
        config=None,
        split=args.corpus_split,
        cache_dir=cache_dir,
        token=args.hf_token,
        streaming=args.streaming_corpus,
    )

    doc_ids: list[str] = []
    doc_id_map: dict[str, dict] = {}
    source_to_page: dict[str, dict] = {}
    page_count = 0
    corpus_count = 0
    manifest_path = output_root / f"doc_pages_{args.split}.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for row in tqdm(corpus, desc="OpenDocVQA corpus"):
            row = dict(row)
            source_doc_id = normalize_doc_id(row.get("doc_id"))
            if not source_doc_id:
                continue
            if not keep_corpus_row(row, dataset_filters, wanted_doc_ids):
                continue
            pack_idx = corpus_count // args.group_size
            page_idx = corpus_count % args.group_size
            doc_id = pack_doc_id(pack_idx)
            if page_idx == 0:
                doc_ids.append(doc_id)
                doc_id_map[doc_id] = {
                    "source": "OpenDocVQA-Corpus",
                    "pack_index": pack_idx,
                    "group_size": args.group_size,
                }

            dataset_name = corpus_dataset_name(row)
            image_rel = f"pages_{args.split}/{doc_id}/{page_idx}.jpg"
            if not args.skip_images:
                save_image(row["image"], output_root / image_rel, args.overwrite_images)
            page_uid = f"{doc_id}_page{page_idx}"
            page_row = {
                "doc_id": doc_id,
                "doc_name": doc_id,
                "page_idx": page_idx,
                "page_number": page_idx + 1,
                "page_uid": page_uid,
                "image_path": image_rel,
                "source_doc_id": source_doc_id,
                "dataset_name": dataset_name,
            }
            manifest.write(json.dumps(page_row, ensure_ascii=False) + "\n")
            for variant in doc_id_variants(source_doc_id):
                source_to_page.setdefault(variant, page_row)
            corpus_count += 1
            page_count += 1
            if args.max_corpus_docs > 0 and corpus_count >= args.max_corpus_docs:
                break
            if wanted_doc_ids is not None and all(doc_id in source_to_page for doc_id in wanted_doc_ids):
                break

    qid_count, missing_gold = write_outputs(
        output_root=output_root,
        split=args.split,
        query_rows=query_rows,
        source_to_page=source_to_page,
        doc_ids=doc_ids,
        doc_id_map=doc_id_map,
    )

    dataset_counts = defaultdict(int)
    unique_pages = {page["page_uid"]: page for page in source_to_page.values()}
    for page in unique_pages.values():
        dataset_counts[page.get("dataset_name", "") or "UNKNOWN"] += 1

    print(f"prepared_output_root={output_root}")
    print(f"qa_repo={args.qa_repo}")
    print(f"corpus_repo={args.corpus_repo}")
    print(f"dataset_filters={sorted(dataset_filters) if dataset_filters else 'ALL'}")
    print(f"corpus_scope={args.corpus_scope}")
    print(f"group_size={args.group_size}")
    print(f"doc_count={len(doc_ids)}")
    print(f"page_count={page_count}")
    print(f"qa_count={qid_count}")
    print(f"missing_gold_pages={missing_gold}")
    if wanted_doc_ids is not None:
        missing_wanted = sorted(doc_id for doc_id in wanted_doc_ids if doc_id not in source_to_page)
        print(f"missing_wanted_doc_id_variants={len(missing_wanted)}")
        print(f"first_missing_wanted_doc_id_variants={missing_wanted[:20]}")
    print(f"dataset_counts={dict(sorted(dataset_counts.items()))}")
    if args.streaming_corpus:
        # Some HPC Python/datasets/pyarrow combinations can abort during
        # interpreter finalization after streaming iteration. At this point all
        # output files are closed and the run has completed successfully.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
