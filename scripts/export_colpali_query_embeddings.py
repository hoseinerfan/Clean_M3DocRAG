#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from m3docrag.retrieval import ColPaliRetrievalModel
from m3docrag.retrieval.colpali import QUERY_TOKEN_FILTER_CHOICES
from m3docrag.utils.paths import LOCAL_MODEL_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one ColPali query embedding safetensors file per qid."
    )
    parser.add_argument("--gold-jsonl", required=True, help="MMQA-style JSONL with qid/question fields.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--retrieval-model-name-or-path", default="colpaligemma-3b-pt-448-base")
    parser.add_argument("--retrieval-adapter-model-name-or-path", default="colpali-v1.2")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--query-token-filter", default="full", choices=QUERY_TOKEN_FILTER_CHOICES)
    parser.add_argument("--embedding-key", default="embeddings")
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--metadata-jsonl", default="")
    parser.add_argument("--max-qids", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--device",
        default="auto",
        help="Use auto, cuda, cpu, or a torch device string. auto uses CUDA when available.",
    )
    return parser.parse_args()


def resolve_model_path(name_or_path: str) -> Path:
    candidate = Path(name_or_path)
    if candidate.exists():
        return candidate
    local_candidate = Path(LOCAL_MODEL_DIR) / name_or_path
    if local_candidate.exists():
        return local_candidate
    raise FileNotFoundError(f"Could not resolve model path: {name_or_path}")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def export_batch(
    *,
    retrieval_model: ColPaliRetrievalModel,
    rows: list[dict],
    output_dir: Path,
    embedding_key: str,
) -> list[dict]:
    queries = [str(row["question"]) for row in rows]
    qids = [str(row["qid"]) for row in rows]
    embeddings = retrieval_model.encode_queries(queries, batch_size=len(queries), to_cpu=True)
    metadata_rows = []
    for qid, query, embedding in zip(qids, queries, embeddings):
        output_path = output_dir / f"{qid}.safetensors"
        embedding = embedding.detach().cpu().contiguous()
        save_file({embedding_key: embedding}, str(output_path))
        metadata_rows.append(
            {
                "qid": qid,
                "question": query,
                "path": str(output_path),
                "shape": list(embedding.shape),
                "dtype": str(embedding.dtype).replace("torch.", ""),
            }
        )
    return metadata_rows


def export_one_by_one(
    *,
    retrieval_model: ColPaliRetrievalModel,
    rows: list[dict],
    output_dir: Path,
    embedding_key: str,
    query_token_filter: str,
) -> list[dict]:
    metadata_rows = []
    for row in tqdm(rows, desc="Exporting query embeddings"):
        qid = str(row["qid"])
        query = str(row["question"])
        output_path = output_dir / f"{qid}.safetensors"
        query_meta = retrieval_model.encode_query_with_metadata(
            query=query,
            to_cpu=True,
            query_token_filter=query_token_filter,
        )
        embedding = query_meta["embeddings"].detach().cpu().contiguous()
        save_file({embedding_key: embedding}, str(output_path))
        metadata_rows.append(
            {
                "qid": qid,
                "question": query,
                "path": str(output_path),
                "shape": list(embedding.shape),
                "dtype": str(embedding.dtype).replace("torch.", ""),
                "query_token_filter": query_token_filter,
                "raw_tokens": query_meta.get("raw_tokens", []),
                "kept_token_indices": query_meta.get("kept_token_indices", []),
            }
        )
    return metadata_rows


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards).")

    gold_path = Path(args.gold_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(gold_path)
    if args.max_qids > 0:
        rows = rows[: args.max_qids]
    total_rows_before_shard = len(rows)
    if args.num_shards > 1:
        rows = [row for idx, row in enumerate(rows) if idx % args.num_shards == args.shard_index]

    if args.resume:
        rows = [row for row in rows if not (output_dir / f"{row['qid']}.safetensors").exists()]

    print(
        "query_embedding_export "
        f"total_qids={total_rows_before_shard} shard_qids={len(rows)} "
        f"output_dir={output_dir}"
    )

    start = time.time()
    device = resolve_device(str(args.device))
    retrieval_model = ColPaliRetrievalModel(
        backbone_name_or_path=resolve_model_path(args.retrieval_model_name_or_path),
        adapter_name_or_path=resolve_model_path(args.retrieval_adapter_model_name_or_path),
    )
    retrieval_model.model.to(device)
    retrieval_model.model.eval()

    metadata_rows: list[dict] = []
    if args.query_token_filter == "full":
        for start_idx in tqdm(range(0, len(rows), args.batch_size), desc="Exporting query embeddings"):
            batch_rows = rows[start_idx : start_idx + args.batch_size]
            metadata_rows.extend(
                export_batch(
                    retrieval_model=retrieval_model,
                    rows=batch_rows,
                    output_dir=output_dir,
                    embedding_key=str(args.embedding_key),
                )
            )
    else:
        metadata_rows.extend(
            export_one_by_one(
                retrieval_model=retrieval_model,
                rows=rows,
                output_dir=output_dir,
                embedding_key=str(args.embedding_key),
                query_token_filter=str(args.query_token_filter),
            )
        )

    if args.metadata_jsonl:
        metadata_path = Path(args.metadata_jsonl)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as handle:
            for row in metadata_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "gold_jsonl": str(gold_path),
        "output_dir": str(output_dir),
        "total_rows_before_shard": total_rows_before_shard,
        "processed_rows": len(rows),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "query_token_filter": str(args.query_token_filter),
        "embedding_key": str(args.embedding_key),
        "device": str(device),
        "elapsed_sec": time.time() - start,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
