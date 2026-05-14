#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

from PIL import Image
from tqdm.auto import tqdm


DEFAULT_DATASET_REPOS = [
    "vidore/vidore_v3_computer_science",
    "vidore/vidore_v3_energy",
    "vidore/vidore_v3_finance_en",
    "vidore/vidore_v3_finance_fr",
    "vidore/vidore_v3_hr",
    "vidore/vidore_v3_industrial",
    "vidore/vidore_v3_pharmaceuticals",
    "vidore/vidore_v3_physics",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare public ViDoRe V3 datasets for M3DocRAG-style page retrieval. "
            "The converter uses ViDoRe corpus page images directly and maps qrels "
            "to zero-based page_idx labels in a combined benchmark folder."
        )
    )
    parser.add_argument(
        "--dataset-repo",
        action="append",
        dest="dataset_repos",
        default=[],
        help=(
            "ViDoRe V3 dataset repo to include. Repeat for multiple repos. "
            "Defaults to the 8 public V3 repos."
        ),
    )
    parser.add_argument("--cache-dir", default="", help="Hugging Face datasets cache directory.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="dev", help="Output split name.")
    parser.add_argument("--hf-split", default="test", help="Input Hugging Face split name.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Accepted for consistency; Hugging Face datasets downloads/caches as needed.",
    )
    parser.add_argument("--max-repos", type=int, default=0)
    parser.add_argument("--max-docs-per-repo", type=int, default=0)
    parser.add_argument("--max-queries-per-repo", type=int, default=0)
    parser.add_argument(
        "--language",
        action="append",
        default=[],
        help="Optional query language filter. Repeat for multiple languages, e.g. --language english.",
    )
    parser.add_argument("--qrel-score-threshold", type=int, default=1)
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--overwrite-images", action="store_true")
    return parser.parse_args()


def repo_slug(repo_id: str) -> str:
    name = repo_id.rstrip("/").split("/")[-1]
    if name.startswith("vidore_v3_"):
        name = name[len("vidore_v3_") :]
    return quote(name, safe="._-")


def normalize_id(value) -> str:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip()


def maybe_int(value, default: int = 0) -> int:
    if value is None:
        return default
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


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
    if value is None:
        return []
    value = jsonable(value)
    if isinstance(value, list):
        return value
    return [value]


def safe_doc_id(slug: str, source_doc_id: str) -> str:
    return quote(f"{slug}__{str(source_doc_id).strip()}", safe="._-")


def safe_file_stem(value: str) -> str:
    return quote(str(value), safe="._-")


def load_subset(repo_id: str, subset: str, split: str, cache_dir: Path | None):
    from datasets import load_dataset

    kwargs = {}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    return load_dataset(repo_id, subset, split=split, **kwargs)


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


def sorted_page_rows(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            maybe_int(row.get("source_page_number"), 0),
            maybe_int(row.get("source_corpus_id"), 0),
            str(row.get("source_corpus_id", "")),
        ),
    )


def sort_query_key(item: tuple[tuple[str, str], dict]) -> tuple[str, int, str]:
    (slug, query_id), _row = item
    try:
        numeric_id = int(query_id)
    except ValueError:
        numeric_id = 10**18
    return slug, numeric_id, query_id


def collect_repo(
    *,
    repo_id: str,
    output_root: Path,
    cache_dir: Path | None,
    hf_split: str,
    max_docs_per_repo: int,
    max_queries_per_repo: int,
    languages: set[str],
    skip_images: bool,
    overwrite_images: bool,
):
    slug = repo_slug(repo_id)
    print(f"loading_repo={repo_id}")

    corpus = load_subset(repo_id, "corpus", hf_split, cache_dir)
    queries = load_subset(repo_id, "queries", hf_split, cache_dir)
    qrels = load_subset(repo_id, "qrels", hf_split, cache_dir)

    kept_doc_ids: set[str] = set()
    pages_by_doc: dict[str, list[dict]] = defaultdict(list)
    doc_id_map: dict[str, dict] = {}

    for row in tqdm(corpus, desc=f"{slug} corpus"):
        source_doc_id = str(row.get("doc_id", "")).strip()
        doc_id = safe_doc_id(slug, source_doc_id)
        if max_docs_per_repo > 0 and doc_id not in kept_doc_ids and len(kept_doc_ids) >= max_docs_per_repo:
            continue
        kept_doc_ids.add(doc_id)

        corpus_id = normalize_id(row.get("corpus_id"))
        image_rel = f"pages_{hf_split}/{doc_id}/corpus_{safe_file_stem(corpus_id)}.jpg"
        if not skip_images:
            save_image(row["image"], output_root / image_rel, overwrite_images)

        doc_id_map[doc_id] = {
            "repo_id": repo_id,
            "repo_slug": slug,
            "source_doc_id": source_doc_id,
        }
        pages_by_doc[doc_id].append(
            {
                "repo_id": repo_id,
                "repo_slug": slug,
                "doc_id": doc_id,
                "source_doc_id": source_doc_id,
                "source_corpus_id": corpus_id,
                "source_page_number": maybe_int(row.get("page_number_in_doc"), 0),
                "ocr_text": str(row.get("markdown", "") or ""),
                "image_path": image_rel,
            }
        )

    query_rows: dict[tuple[str, str], dict] = {}
    kept_query_count = 0
    for row in tqdm(queries, desc=f"{slug} queries"):
        language = str(row.get("language", "") or "").strip()
        if languages and language.lower() not in languages:
            continue
        if max_queries_per_repo > 0 and kept_query_count >= max_queries_per_repo:
            break
        kept_query_count += 1
        query_id = normalize_id(row.get("query_id"))
        query_rows[(slug, query_id)] = {
            "repo_id": repo_id,
            "repo_slug": slug,
            "query_id": query_id,
            "query": str(row.get("query", "") or "").strip(),
            "language": language,
            "query_types": normalize_list(row.get("query_types")),
            "query_format": str(row.get("query_format", "") or "").strip(),
            "content_type": str(row.get("content_type", "") or "").strip(),
            "raw_answers": normalize_list(row.get("raw_answers")),
            "answer": str(row.get("answer", "") or "").strip(),
            "query_generator": str(row.get("query_generator", "") or "").strip(),
            "query_generation_pipeline": str(row.get("query_generation_pipeline", "") or "").strip(),
            "source_type": str(row.get("source_type", "") or "").strip(),
            "query_type_for_generation": str(row.get("query_type_for_generation", "") or "").strip(),
        }

    allowed_query_keys = set(query_rows)
    qrels_by_query: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in tqdm(qrels, desc=f"{slug} qrels"):
        query_id = normalize_id(row.get("query_id"))
        query_key = (slug, query_id)
        if query_key not in allowed_query_keys:
            continue
        qrels_by_query[query_key].append(
            {
                "repo_id": repo_id,
                "repo_slug": slug,
                "query_id": query_id,
                "corpus_id": normalize_id(row.get("corpus_id")),
                "score": maybe_int(row.get("score"), 0),
                "content_type": str(row.get("content_type", "") or "").strip(),
                "bounding_boxes": jsonable(row.get("bounding_boxes")),
            }
        )

    return pages_by_doc, doc_id_map, query_rows, qrels_by_query


def write_doc_pages(
    *,
    output_root: Path,
    split: str,
    pages_by_doc: dict[str, list[dict]],
) -> tuple[list[str], dict[tuple[str, str], dict], int]:
    doc_ids = sorted(pages_by_doc)
    page_by_corpus: dict[tuple[str, str], dict] = {}
    page_count = 0
    manifest_path = output_root / f"doc_pages_{split}.jsonl"
    with manifest_path.open("w", encoding="utf-8") as out:
        for doc_id in doc_ids:
            for page_idx, page in enumerate(sorted_page_rows(pages_by_doc[doc_id])):
                page_uid = f"{doc_id}_page{page_idx}"
                row = {
                    "doc_id": doc_id,
                    "doc_name": page["source_doc_id"],
                    "repo_id": page["repo_id"],
                    "repo_slug": page["repo_slug"],
                    "source_doc_id": page["source_doc_id"],
                    "source_corpus_id": page["source_corpus_id"],
                    "source_page_number": page["source_page_number"],
                    "page_idx": page_idx,
                    "page_number": page_idx + 1,
                    "page_uid": page_uid,
                    "image_path": page["image_path"],
                    "ocr_text": page["ocr_text"],
                }
                page_by_corpus[(page["repo_slug"], page["source_corpus_id"])] = row
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                page_count += 1
    return doc_ids, page_by_corpus, page_count


def write_queries(
    *,
    output_root: Path,
    split: str,
    query_rows: dict[tuple[str, str], dict],
    qrels_by_query: dict[tuple[str, str], list[dict]],
    page_by_corpus: dict[tuple[str, str], dict],
    qrel_score_threshold: int,
) -> tuple[int, int]:
    mmqa_path = output_root / f"MMQA_{split}.jsonl"
    qids_path = output_root / f"qids_{split}.jsonl"
    gold_pages_path = output_root / f"gold_pages_{split}.jsonl"
    missing_qrels = 0
    qid_count = 0
    used_qids: set[str] = set()

    with mmqa_path.open("w", encoding="utf-8") as mmqa_out, qids_path.open(
        "w", encoding="utf-8"
    ) as qids_out, gold_pages_path.open("w", encoding="utf-8") as gold_out:
        for query_key, query_row in sorted(query_rows.items(), key=sort_query_key):
            gold_items = []
            seen_page_uids = set()
            for qrel in sorted(
                qrels_by_query.get(query_key, []),
                key=lambda item: (-int(item["score"]), item["corpus_id"]),
            ):
                if int(qrel["score"]) < qrel_score_threshold:
                    continue
                page = page_by_corpus.get((qrel["repo_slug"], qrel["corpus_id"]))
                if page is None:
                    missing_qrels += 1
                    continue
                if page["page_uid"] in seen_page_uids:
                    continue
                seen_page_uids.add(page["page_uid"])
                gold_items.append((page, qrel))
            if not gold_items:
                continue

            qid_base = f"{query_row['repo_slug']}__q{safe_file_stem(query_row['query_id'])}"
            qid = qid_base
            duplicate_idx = 1
            while qid in used_qids:
                duplicate_idx += 1
                qid = f"{qid_base}__dup{duplicate_idx}"
            used_qids.add(qid)

            gold_page_ids = [page["page_idx"] for page, _qrel in gold_items]
            gold_page_uids = [page["page_uid"] for page, _qrel in gold_items]
            supporting_context = [
                {
                    "doc_id": page["doc_id"],
                    "doc_part": qrel.get("content_type") or query_row.get("content_type") or "evidence",
                    "page_idx": page["page_idx"],
                    "page_id": page["page_idx"],
                    "source_corpus_id": page["source_corpus_id"],
                    "source_page_number": page["source_page_number"],
                    "qrel_score": qrel["score"],
                }
                for page, qrel in gold_items
            ]
            qrel_metadata = [
                {
                    "doc_id": page["doc_id"],
                    "page_uid": page["page_uid"],
                    "source_corpus_id": page["source_corpus_id"],
                    "source_page_number": page["source_page_number"],
                    "score": qrel["score"],
                    "content_type": qrel.get("content_type", ""),
                    "bounding_boxes": qrel.get("bounding_boxes"),
                }
                for page, qrel in gold_items
            ]

            mmqa_row = {
                "qid": qid,
                "question": query_row["query"],
                "answers": [
                    {
                        "answer": query_row.get("answer", ""),
                        "modality": query_row.get("content_type") or "visual_document",
                    }
                ],
                "metadata": {
                    "type": "ViDoReV3",
                    "source": "ViDoRe V3",
                    "repo_id": query_row["repo_id"],
                    "repo_slug": query_row["repo_slug"],
                    "source_query_id": query_row["query_id"],
                    "language": query_row.get("language", ""),
                    "query_types": query_row.get("query_types", []),
                    "query_format": query_row.get("query_format", ""),
                    "content_type": query_row.get("content_type", ""),
                    "raw_answers": query_row.get("raw_answers", []),
                    "query_generator": query_row.get("query_generator", ""),
                    "query_generation_pipeline": query_row.get("query_generation_pipeline", ""),
                    "source_type": query_row.get("source_type", ""),
                    "query_type_for_generation": query_row.get("query_type_for_generation", ""),
                    "gold_page_ids": gold_page_ids,
                    "gold_page_uids": gold_page_uids,
                    "qrels": qrel_metadata,
                },
                "supporting_context": supporting_context,
            }
            mmqa_out.write(json.dumps(mmqa_row, ensure_ascii=False) + "\n")
            qids_out.write(json.dumps({"qid": qid}, ensure_ascii=False) + "\n")
            gold_out.write(
                json.dumps(
                    {
                        "qid": qid,
                        "repo_id": query_row["repo_id"],
                        "repo_slug": query_row["repo_slug"],
                        "source_query_id": query_row["query_id"],
                        "language": query_row.get("language", ""),
                        "query_types": query_row.get("query_types", []),
                        "query_format": query_row.get("query_format", ""),
                        "content_type": query_row.get("content_type", ""),
                        "gold_page_ids": gold_page_ids,
                        "gold_page_uids": gold_page_uids,
                        "qrels": qrel_metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            qid_count += 1

    return qid_count, missing_qrels


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    configure_hf_cache(cache_dir)

    repos = args.dataset_repos or DEFAULT_DATASET_REPOS
    if args.max_repos > 0:
        repos = repos[: args.max_repos]
    languages = {language.strip().lower() for language in args.language if language.strip()}

    all_pages_by_doc: dict[str, list[dict]] = defaultdict(list)
    all_doc_id_map: dict[str, dict] = {}
    all_query_rows: dict[tuple[str, str], dict] = {}
    all_qrels_by_query: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for repo_id in repos:
        pages_by_doc, doc_id_map, query_rows, qrels_by_query = collect_repo(
            repo_id=repo_id,
            output_root=output_root,
            cache_dir=cache_dir,
            hf_split=args.hf_split,
            max_docs_per_repo=args.max_docs_per_repo,
            max_queries_per_repo=args.max_queries_per_repo,
            languages=languages,
            skip_images=args.skip_images,
            overwrite_images=args.overwrite_images,
        )
        for doc_id, pages in pages_by_doc.items():
            all_pages_by_doc[doc_id].extend(pages)
        all_doc_id_map.update(doc_id_map)
        all_query_rows.update(query_rows)
        for key, qrels in qrels_by_query.items():
            all_qrels_by_query[key].extend(qrels)

    doc_ids, page_by_corpus, page_count = write_doc_pages(
        output_root=output_root,
        split=args.split,
        pages_by_doc=all_pages_by_doc,
    )
    qid_count, missing_qrels = write_queries(
        output_root=output_root,
        split=args.split,
        query_rows=all_query_rows,
        qrels_by_query=all_qrels_by_query,
        page_by_corpus=page_by_corpus,
        qrel_score_threshold=args.qrel_score_threshold,
    )

    (output_root / f"{args.split}_doc_ids.json").write_text(
        json.dumps(doc_ids, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "doc_id_map.json").write_text(
        json.dumps(all_doc_id_map, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_root / "vidore_repos.json").write_text(
        json.dumps(repos, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"prepared_output_root={output_root}")
    print(f"repos={repos}")
    print(f"languages={sorted(languages) if languages else 'ALL'}")
    print(f"doc_count={len(doc_ids)}")
    print(f"page_count={page_count}")
    print(f"qa_count={qid_count}")
    print(f"missing_qrels={missing_qrels}")


if __name__ == "__main__":
    main()
