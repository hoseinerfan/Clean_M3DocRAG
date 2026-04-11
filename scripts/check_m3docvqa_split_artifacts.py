#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check M3DocVQA split artifacts: PDFs, embeddings, and FAISS index."
    )
    parser.add_argument("--repo-root", default=".", help="Repo root on the target machine")
    parser.add_argument("--data-name", default="m3-docvqa")
    parser.add_argument("--split", default="train", choices=["dev", "train"])
    parser.add_argument("--retrieval-adapter-model-name", default="colpali-v1.2")
    parser.add_argument("--faiss-index-type", default="ivfflat")
    return parser.parse_args()


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{num_bytes}B"


def count_files(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.glob(pattern))


def main() -> None:
    args = parse_args()

    repo_root = Path(args.repo_root).resolve()
    data_root = repo_root / "data" / args.data_name
    split_doc_ids_path = data_root / f"{args.split}_doc_ids.json"

    expected_docs = None
    if split_doc_ids_path.exists():
        expected_docs = len(json.loads(split_doc_ids_path.read_text()))

    pdf_dir = data_root / "splits" / f"pdfs_{args.split}"
    embedding_name = f"{args.retrieval_adapter_model_name}_{args.data_name}_{args.split}"
    embedding_dir = repo_root / "embeddings" / embedding_name
    index_dir = repo_root / "embeddings" / f"{embedding_name}_pageindex_{args.faiss_index_type}"
    index_path = index_dir / "index.bin"

    pdf_count = count_files(pdf_dir, "*.pdf")
    emb_count = count_files(embedding_dir, "*.safetensors")

    pdf_ok = expected_docs is not None and pdf_count == expected_docs
    emb_ok = expected_docs is not None and emb_count == expected_docs
    index_ok = index_path.exists() and index_path.is_file()

    print(f"repo_root={repo_root}")
    print(f"split={args.split}")
    print(f"data_root={data_root}")
    print(f"expected_docs={expected_docs}")
    print()

    print("[pdfs]")
    print(f"path={pdf_dir}")
    print(f"exists={pdf_dir.exists()}")
    print(f"count={pdf_count}")
    print(f"status={'READY' if pdf_ok else 'MISSING_OR_INCOMPLETE'}")
    print()

    print("[embeddings]")
    print(f"path={embedding_dir}")
    print(f"exists={embedding_dir.exists()}")
    print(f"count={emb_count}")
    if embedding_dir.exists():
        print(f"size={human_size(sum(p.stat().st_size for p in embedding_dir.glob('*') if p.is_file()))}")
    print(f"status={'READY' if emb_ok else 'MISSING_OR_INCOMPLETE'}")
    print()

    print("[index]")
    print(f"path={index_path}")
    print(f"exists={index_ok}")
    if index_ok:
        print(f"size={human_size(index_path.stat().st_size)}")
    print(f"status={'READY' if index_ok else 'MISSING'}")
    print()

    overall_ok = pdf_ok and emb_ok and index_ok
    print(f"overall_status={'READY' if overall_ok else 'INCOMPLETE'}")


if __name__ == "__main__":
    main()
