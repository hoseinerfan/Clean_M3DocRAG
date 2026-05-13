#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import safetensors.torch
import torch
from PIL import Image
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from m3docrag.retrieval import ColPaliRetrievalModel
from m3docrag.utils.paths import LOCAL_MODEL_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed MMDocIR page JPEGs into per-document safetensors.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--retrieval-model-name-or-path", default="colpaligemma-3b-pt-448-base")
    parser.add_argument("--retrieval-adapter-model-name-or-path", default="colpali-v1.2")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--doc-id", action="append", default=[], help="Optional doc_id filter.")
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split docs across this many array tasks. Use with --shard-index.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="0-based shard index to process when --num-shards > 1.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve_model_path(name_or_path: str) -> Path:
    candidate = Path(name_or_path)
    if candidate.exists():
        return candidate
    local_candidate = Path(LOCAL_MODEL_DIR) / name_or_path
    if local_candidate.exists():
        return local_candidate
    raise FileNotFoundError(f"Could not resolve model path: {name_or_path}")


def load_doc_pages(data_root: Path, split: str) -> dict[str, list[dict]]:
    manifest_path = data_root / f"doc_pages_{split}.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    doc_pages: dict[str, list[dict]] = defaultdict(list)
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            doc_pages[str(row["doc_id"])].append(row)
    return {
        doc_id: sorted(rows, key=lambda item: int(item["page_idx"]))
        for doc_id, rows in doc_pages.items()
    }


def load_image(path: Path) -> Image.Image:
    image = Image.open(path)
    return image.convert("RGB")


def resolve_image_path(data_root: Path, image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    return data_root / path


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    doc_pages = load_doc_pages(data_root, args.split)
    doc_ids = sorted(doc_pages)
    if args.doc_id:
        keep = set(args.doc_id)
        doc_ids = [doc_id for doc_id in doc_ids if doc_id in keep]
    if args.max_docs > 0:
        doc_ids = doc_ids[: args.max_docs]
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards).")
    if args.num_shards > 1:
        doc_ids = [
            doc_id
            for idx, doc_id in enumerate(doc_ids)
            if idx % args.num_shards == args.shard_index
        ]
    print(
        "embedding_shard "
        f"index={args.shard_index} num_shards={args.num_shards} docs={len(doc_ids)}"
    )

    retrieval_model = ColPaliRetrievalModel(
        backbone_name_or_path=resolve_model_path(args.retrieval_model_name_or_path),
        adapter_name_or_path=resolve_model_path(args.retrieval_adapter_model_name_or_path),
    )
    retrieval_model.model.eval()

    Image.MAX_IMAGE_PIXELS = None
    for doc_id in tqdm(doc_ids, desc="Embedding MMDocIR docs"):
        output_path = output_dir / f"{doc_id}.safetensors"
        if args.resume and output_path.exists():
            continue
        pages = doc_pages[doc_id]
        images = [load_image(resolve_image_path(data_root, row["image_path"])) for row in pages]
        with torch.no_grad():
            doc_embs = retrieval_model.encode_images(
                images=images,
                batch_size=args.per_device_eval_batch_size,
                to_cpu=True,
                use_tqdm=False,
            )
        doc_tensor = torch.stack(doc_embs, dim=0).to(torch.bfloat16)
        safetensors.torch.save_file({"embeddings": doc_tensor}, output_path)

    print(f"saved_embeddings={output_dir}")


if __name__ == "__main__":
    main()
