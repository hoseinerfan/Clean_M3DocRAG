#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import torch
import safetensors
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FAISS page-token index for MMDocIR embeddings.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--faiss-index-type", default="ivfflat", choices=["flatip", "ivfflat", "ivfpq"])
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--ncentroids", type=int, default=1024)
    return parser.parse_args()


def load_doc_ids(data_root: Path, split: str) -> list[str]:
    return json.loads((data_root / f"{split}_doc_ids.json").read_text(encoding="utf-8"))


def load_doc_embedding(path: Path) -> torch.Tensor:
    with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_tensor("embeddings")


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    embedding_dir = Path(args.embedding_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    doc_ids = load_doc_ids(data_root, args.split)
    d = int(args.embedding_dim)
    quantizer = faiss.IndexFlatIP(d)
    if args.faiss_index_type == "flatip":
        index = quantizer
    elif args.faiss_index_type == "ivfflat":
        index = faiss.IndexIVFFlat(quantizer, d, int(args.ncentroids))
    else:
        index = faiss.IndexIVFPQ(quantizer, d, 100, 8, 8)

    all_token_embeddings = []
    page_count = 0
    token_count = 0
    missing = []
    for doc_id in tqdm(doc_ids, desc="Loading embeddings"):
        emb_path = embedding_dir / f"{doc_id}.safetensors"
        if not emb_path.exists():
            missing.append(doc_id)
            continue
        doc_emb = load_doc_embedding(emb_path)
        for page_idx in range(len(doc_emb)):
            page_emb = doc_emb[page_idx].view(-1, d)
            all_token_embeddings.append(page_emb)
            page_count += 1
            token_count += int(page_emb.shape[0])
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} embedding files, first missing: {missing[:10]}")

    flat = torch.cat(all_token_embeddings, dim=0).float().numpy()
    if not index.is_trained:
        index.train(flat)
    index.add(flat)
    faiss.write_index(index, str(output_dir / "index.bin"))
    (output_dir / "index_meta.json").write_text(
        json.dumps(
            {
                "data_root": str(data_root),
                "embedding_dir": str(embedding_dir),
                "split": args.split,
                "faiss_index_type": args.faiss_index_type,
                "doc_count": len(doc_ids),
                "page_count": page_count,
                "token_count": token_count,
                "doc_ids": doc_ids,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved_index={output_dir / 'index.bin'}")


if __name__ == "__main__":
    main()

