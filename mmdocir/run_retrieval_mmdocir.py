#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import safetensors
import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from m3docrag.retrieval import ColPaliRetrievalModel
from m3docrag.utils.paths import LOCAL_MODEL_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M3DocRAG-style MMDocIR page retrieval.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--retrieval-model-name-or-path", default="colpaligemma-3b-pt-448-base")
    parser.add_argument("--retrieval-adapter-model-name-or-path", default="colpali-v1.2")
    parser.add_argument("--n-retrieval-pages", type=int, default=1000)
    parser.add_argument("--faiss-nprobe", type=int, default=4)
    parser.add_argument("--query-token-filter", default="full", choices=["full", "drop_pad_like", "semantic_only"])
    parser.add_argument("--ignore-pad-scores-in-final-ranking", action="store_true")
    parser.add_argument("--max-qids", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Write partial predictions every N newly processed queries. 0 writes only at the end.",
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


def write_predictions(path: Path, predictions: dict[str, dict]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def load_doc_ids(data_root: Path, split: str) -> list[str]:
    return json.loads((data_root / f"{split}_doc_ids.json").read_text(encoding="utf-8"))


def load_doc_embedding(path: Path) -> torch.Tensor:
    with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_tensor("embeddings")


def load_embeddings_and_token_map(
    *,
    doc_ids: list[str],
    embedding_dir: Path,
    embedding_dim: int = 128,
) -> tuple[dict[str, torch.Tensor], np.ndarray, list[str]]:
    docid2embs = {}
    all_token_embeddings = []
    token2pageuid: list[str] = []
    for doc_id in tqdm(doc_ids, desc="Loading retrieval embeddings"):
        doc_emb = load_doc_embedding(embedding_dir / f"{doc_id}.safetensors")
        docid2embs[doc_id] = doc_emb.bfloat16()
        for page_idx in range(len(doc_emb)):
            page_emb = doc_emb[page_idx].view(-1, embedding_dim)
            all_token_embeddings.append(page_emb)
            token2pageuid.extend([f"{doc_id}_page{page_idx}"] * int(page_emb.shape[0]))
    flat = torch.cat(all_token_embeddings, dim=0).float().numpy()
    return docid2embs, flat, token2pageuid


def retrieve_one(
    *,
    query: str,
    retrieval_model: ColPaliRetrievalModel,
    index,
    all_token_embeddings: np.ndarray,
    token2pageuid: list[str],
    n_return_pages: int,
    query_token_filter: str,
    ignore_pad_scores_in_final_ranking: bool,
) -> list[list[object]]:
    query_meta = retrieval_model.encode_query_with_metadata(
        query=query,
        to_cpu=True,
        query_token_filter=query_token_filter,
    )
    query_emb = query_meta["embeddings"].float().numpy().astype(np.float32)
    raw_tokens = query_meta.get("raw_tokens", [])
    score_mask = [True] * len(query_emb)
    if ignore_pad_scores_in_final_ranking:
        score_mask = [token != "<pad>" for token in raw_tokens]
        if not any(score_mask):
            raise ValueError("PAD filtering removed every scoring query token.")

    nearest_k = min(int(n_return_pages), int(index.ntotal))
    _distances, indices = index.search(query_emb, nearest_k)

    final_page2scores: dict[str, float] = {}
    for q_idx, current_query_emb in enumerate(query_emb):
        if not score_mask[q_idx]:
            continue
        current_q_page2scores: dict[str, float] = {}
        for nn_idx in range(nearest_k):
            token_idx = int(indices[q_idx, nn_idx])
            if token_idx < 0:
                continue
            page_uid = token2pageuid[token_idx]
            doc_token_emb = all_token_embeddings[token_idx]
            score = float((current_query_emb * doc_token_emb).sum())
            current_q_page2scores[page_uid] = max(score, current_q_page2scores.get(page_uid, -1e30))
        for page_uid, score in current_q_page2scores.items():
            final_page2scores[page_uid] = final_page2scores.get(page_uid, 0.0) + score

    ranked_pages = sorted(final_page2scores.items(), key=lambda item: item[1], reverse=True)[:n_return_pages]
    rows = []
    for page_uid, score in ranked_pages:
        doc_id, page_suffix = page_uid.rsplit("_page", 1)
        rows.append([doc_id, int(page_suffix), float(score)])
    return rows


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards).")
    if args.save_every < 0:
        raise ValueError("--save-every must be non-negative.")

    data_root = Path(args.data_root)
    embedding_dir = Path(args.embedding_dir)
    index_path = Path(args.index_dir) / "index.bin"
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc_ids = load_doc_ids(data_root, args.split)
    _docid2embs, all_token_embeddings, token2pageuid = load_embeddings_and_token_map(
        doc_ids=doc_ids,
        embedding_dir=embedding_dir,
    )

    index = faiss.read_index(str(index_path))
    if hasattr(index, "nprobe"):
        index.nprobe = int(args.faiss_nprobe)

    retrieval_model = ColPaliRetrievalModel(
        backbone_name_or_path=resolve_model_path(args.retrieval_model_name_or_path),
        adapter_name_or_path=resolve_model_path(args.retrieval_adapter_model_name_or_path),
    )
    retrieval_model.model.eval()

    gold_rows = read_jsonl(data_root / f"MMQA_{args.split}.jsonl")
    if args.max_qids > 0:
        gold_rows = gold_rows[: args.max_qids]
    total_rows_before_shard = len(gold_rows)
    if args.num_shards > 1:
        gold_rows = [row for idx, row in enumerate(gold_rows) if idx % args.num_shards == args.shard_index]

    predictions: dict[str, dict] = {}
    if args.resume and output_path.exists():
        predictions = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(predictions, dict):
            raise TypeError(f"Existing prediction file is not a JSON object: {output_path}")
    done_qids = set(predictions)
    print(
        "retrieval_shard "
        f"index={args.shard_index} num_shards={args.num_shards} "
        f"total_qids={total_rows_before_shard} shard_qids={len(gold_rows)} resume_done={len(done_qids)}"
    )

    new_since_save = 0
    for gold_row in tqdm(gold_rows, desc="Retrieving MMDocIR queries"):
        qid = str(gold_row["qid"])
        if qid in done_qids:
            continue
        query = str(gold_row["question"])
        start = time.perf_counter()
        retrieval_rows = retrieve_one(
            query=query,
            retrieval_model=retrieval_model,
            index=index,
            all_token_embeddings=all_token_embeddings,
            token2pageuid=token2pageuid,
            n_return_pages=args.n_retrieval_pages,
            query_token_filter=args.query_token_filter,
            ignore_pad_scores_in_final_ranking=args.ignore_pad_scores_in_final_ranking,
        )
        predictions[qid] = {
            "pred_answer": "",
            "page_retrieval_results": retrieval_rows,
            "qid": qid,
            "question": query,
            "time_retrieval": time.perf_counter() - start,
            "time_qa": None,
        }
        new_since_save += 1
        if args.save_every and new_since_save >= args.save_every:
            write_predictions(output_path, predictions)
            new_since_save = 0

    write_predictions(output_path, predictions)
    print(f"saved_prediction={output_path}")


if __name__ == "__main__":
    main()
