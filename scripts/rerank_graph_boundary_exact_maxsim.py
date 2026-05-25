#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_RECALL_KS = [1, 2, 4, 5, 10, 20, 50, 100]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use exact ColPali MaxSim as a local verifier for the Graph-PPR top-4/rank-5 "
            "page boundary. The script scores only the current top-k pages plus the first "
            "boundary page, then swaps the boundary page into top-k when exact MaxSim ranks "
            "it above the weakest current top-k page."
        )
    )
    parser.add_argument("--gold", required=True, help="Gold MMQA JSONL.")
    parser.add_argument("--base-prediction", required=True, help="Graph-PPR/base prediction JSON.")
    parser.add_argument("--embedding-dir", required=True, help="Directory of per-doc .safetensors embeddings.")
    parser.add_argument("--output-prediction-json", required=True)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument("--output-case-json", default="")
    parser.add_argument("--hit-k", type=int, default=4)
    parser.add_argument(
        "--boundary-rank",
        type=int,
        default=5,
        help="One-indexed boundary page rank to compare against top-k. Default compares rank 5 to top 4.",
    )
    parser.add_argument("--recall-k", dest="recall_ks", type=int, nargs="+", default=DEFAULT_RECALL_KS)
    parser.add_argument("--max-qids", type=int, default=0, help="Optional smoke-test cap.")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--doc-cache-size", type=int, default=64)
    parser.add_argument("--rerank-batch-size", type=int, default=16)
    parser.add_argument(
        "--mode",
        choices=["swap_weakest", "rerank_top_boundary"],
        default="swap_weakest",
        help=(
            "swap_weakest only swaps the boundary page with the weakest top-k page. "
            "rerank_top_boundary exactly reranks the local top-k+boundary set."
        ),
    )
    parser.add_argument(
        "--same-doc-only",
        action="store_true",
        help="Only consider the boundary page if its document already appears in the current top-k docs.",
    )
    parser.add_argument(
        "--retrieval-model-name-or-path",
        default="colpaligemma-3b-pt-448-base",
        help="Backbone path/name. Resolved directly or under LOCAL_MODEL_DIR.",
    )
    parser.add_argument(
        "--retrieval-adapter-model-name-or-path",
        default="colpali-v1.2",
        help="Adapter path/name. Resolved directly or under LOCAL_MODEL_DIR.",
    )
    parser.add_argument(
        "--query-token-filter",
        default="full",
        choices=["full", "drop_pad_like", "semantic_only"],
        help="Query-token filtering passed to ColPali exact scoring.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype for exact MaxSim scoring.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_prediction(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "predictions" in payload and isinstance(
        payload["predictions"], (dict, list)
    ):
        payload = payload["predictions"]
    rows: dict[str, dict[str, Any]] = {}
    if isinstance(payload, dict):
        iterable = payload.items()
    elif isinstance(payload, list):
        iterable = enumerate(payload)
    else:
        raise TypeError(f"Prediction JSON must be an object or list: {path}")
    for raw_key, row in iterable:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("qid", "")).strip() or str(raw_key).strip()
        if qid:
            rows[qid] = row
    return rows


def write_prediction(path: Path, rows: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"predictions": rows}, ensure_ascii=False), encoding="utf-8")


def page_uid(doc_id: Any, page_idx: Any) -> str:
    return f"{doc_id}_page{int(page_idx)}"


def parse_page_uid(uid: str) -> tuple[str, int]:
    if "_page" not in uid:
        raise ValueError(f"Invalid page uid: {uid}")
    doc_id, raw_page = uid.rsplit("_page", 1)
    return doc_id, int(raw_page)


def page_doc(uid: str) -> str:
    return uid.rsplit("_page", 1)[0]


def item_uid(item: Any) -> str | None:
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        return None
    try:
        return page_uid(str(item[0]), int(item[1]))
    except (TypeError, ValueError):
        return None


def ranked_items(pred_row: dict[str, Any] | None, limit: int = 0) -> list[Any]:
    if pred_row is None:
        return []
    items: list[Any] = []
    seen: set[str] = set()
    for item in pred_row.get("page_retrieval_results", []):
        uid = item_uid(item)
        if not uid or uid in seen:
            continue
        seen.add(uid)
        items.append(item)
        if limit > 0 and len(items) >= limit:
            break
    return items


def ranked_pages(pred_row: dict[str, Any] | None, limit: int = 0) -> list[str]:
    pages: list[str] = []
    for item in ranked_items(pred_row, limit):
        uid = item_uid(item)
        if uid:
            pages.append(uid)
    return pages


def ranked_docs(pred_row: dict[str, Any] | None, limit: int = 0) -> list[str]:
    docs: list[str] = []
    seen: set[str] = set()
    for uid in ranked_pages(pred_row):
        doc_id = page_doc(uid)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        docs.append(doc_id)
        if limit > 0 and len(docs) >= limit:
            break
    return docs


def first_rank(ranked: list[str], gold: set[str]) -> int | None:
    for idx, value in enumerate(ranked, start=1):
        if value in gold:
            return idx
    return None


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / float(len(gold))


def gold_page_uids(row: dict[str, Any]) -> set[str]:
    metadata = row.get("metadata", {})
    uids = {
        str(value).strip()
        for value in metadata.get("gold_page_uids", [])
        if str(value).strip()
    }
    for ctx in row.get("supporting_context", []):
        doc_id = str(ctx.get("doc_id", "")).strip()
        page_idx = ctx.get("page_idx", ctx.get("page_id"))
        if doc_id and page_idx is not None:
            uids.add(page_uid(doc_id, int(page_idx)))
    return uids


def gold_doc_ids(row: dict[str, Any]) -> set[str]:
    metadata = row.get("metadata", {})
    docs = {
        str(value).strip()
        for value in metadata.get("gold_doc_ids", [])
        if str(value).strip()
    }
    for ctx in row.get("supporting_context", []):
        doc_id = str(ctx.get("doc_id", "")).strip()
        if doc_id:
            docs.add(doc_id)
    return docs


def metric_scores(
    pred_row: dict[str, Any] | None,
    gold_pages: set[str],
    gold_docs: set[str],
    recall_ks: list[int],
    hit_k: int,
) -> dict[str, Any]:
    pages = ranked_pages(pred_row)
    docs = ranked_docs(pred_row)
    page_rank = first_rank(pages, gold_pages)
    doc_rank = first_rank(docs, gold_docs)
    out: dict[str, Any] = {
        "page_first_rank": page_rank,
        "doc_first_rank": doc_rank,
        f"page_hit@{hit_k}": page_rank is not None and page_rank <= hit_k,
        f"doc_hit@{hit_k}": doc_rank is not None and doc_rank <= hit_k,
    }
    for k in recall_ks:
        out[f"page_recall@{k}"] = recall_at_k(pages, gold_pages, k)
        out[f"doc_recall@{k}"] = recall_at_k(docs, gold_docs, k)
    return out


def movement_for_hit(base_rank: int | None, candidate_rank: int | None, hit_k: int) -> str:
    base_hit = base_rank is not None and base_rank <= hit_k
    candidate_hit = candidate_rank is not None and candidate_rank <= hit_k
    if not base_hit and candidate_hit:
        return "recovered"
    if base_hit and not candidate_hit:
        return "lost"
    if base_rank is None and candidate_rank is None:
        return "missing_in_both"
    if base_rank is not None and candidate_rank is not None and candidate_rank < base_rank:
        return "improved_rank"
    if base_rank is not None and candidate_rank is not None and candidate_rank > base_rank:
        return "worsened_rank"
    return "unchanged"


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def resolve_model_path(name_or_path: str) -> Path:
    from m3docrag.utils.paths import LOCAL_MODEL_DIR

    candidate = Path(name_or_path)
    if candidate.exists():
        return candidate
    local_candidate = Path(LOCAL_MODEL_DIR) / name_or_path
    if local_candidate.exists():
        return local_candidate
    raise FileNotFoundError(f"Could not resolve model path: {name_or_path}")


def dtype_from_name(name: str) -> Any:
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_doc_embedding(path: Path) -> Any:
    import safetensors

    with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_tensor("embeddings")


class DocEmbeddingCache:
    def __init__(self, embedding_dir: Path, max_docs: int) -> None:
        if max_docs <= 0:
            raise ValueError("--doc-cache-size must be positive.")
        self.embedding_dir = embedding_dir
        self.max_docs = max_docs
        self.cache: OrderedDict[str, Any] = OrderedDict()

    def get_doc(self, doc_id: str) -> Any:
        if doc_id in self.cache:
            value = self.cache.pop(doc_id)
            self.cache[doc_id] = value
            return value
        path = self.embedding_dir / f"{doc_id}.safetensors"
        if not path.exists():
            raise FileNotFoundError(f"Missing embedding file for doc {doc_id}: {path}")
        value = load_doc_embedding(path)
        self.cache[doc_id] = value
        while len(self.cache) > self.max_docs:
            self.cache.popitem(last=False)
        return value

    def get_page(self, uid: str) -> Any:
        doc_id, page_idx = parse_page_uid(uid)
        doc_emb = self.get_doc(doc_id)
        if page_idx < 0 or page_idx >= len(doc_emb):
            raise IndexError(f"Page index {page_idx} out of range for {doc_id}: {tuple(doc_emb.shape)}")
        page_emb = doc_emb[page_idx]
        return page_emb.view(-1, page_emb.shape[-1])


def exact_maxsim_scores(
    *,
    retrieval_model: Any,
    query_embeds: Any,
    page_uids: list[str],
    cache: DocEmbeddingCache,
    batch_size: int,
) -> dict[str, float]:
    if batch_size <= 0:
        raise ValueError("--rerank-batch-size must be positive.")
    scores: dict[str, float] = {}
    for start in range(0, len(page_uids), batch_size):
        batch_page_uids = page_uids[start : start + batch_size]
        doc_embeds = [cache.get_page(uid) for uid in batch_page_uids]
        batch_scores = retrieval_model.retrieve(
            query=None,
            doc_embeds=doc_embeds,
            query_embeds=[query_embeds],
            to_cpu=True,
            return_top_1=False,
        )
        values = batch_scores.flatten().tolist()
        scores.update({uid: float(score) for uid, score in zip(batch_page_uids, values)})
    return scores


def reorder_boundary(
    *,
    base_pages: list[str],
    exact_scores: dict[str, float],
    hit_k: int,
    boundary_rank: int,
    mode: str,
) -> tuple[list[str], bool, str | None, str | None]:
    if len(base_pages) < boundary_rank or boundary_rank <= hit_k:
        return list(base_pages), False, None, None
    top_pages = base_pages[:hit_k]
    boundary_page = base_pages[boundary_rank - 1]
    weakest_top = min(top_pages, key=lambda uid: exact_scores.get(uid, -math.inf))
    boundary_score = exact_scores.get(boundary_page, -math.inf)
    weakest_score = exact_scores.get(weakest_top, -math.inf)
    if boundary_score <= weakest_score:
        return list(base_pages), False, boundary_page, weakest_top

    if mode == "rerank_top_boundary":
        local = base_pages[:hit_k] + [boundary_page]
        local_set = set(local)
        reranked_local = sorted(local, key=lambda uid: exact_scores.get(uid, -math.inf), reverse=True)
        remaining = [uid for uid in base_pages if uid not in local_set]
        return reranked_local + remaining, True, boundary_page, weakest_top

    reordered = list(base_pages)
    top_idx = reordered.index(weakest_top)
    boundary_idx = boundary_rank - 1
    reordered[top_idx], reordered[boundary_idx] = reordered[boundary_idx], reordered[top_idx]
    return reordered, True, boundary_page, weakest_top


def prediction_item_by_uid(pred_row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in ranked_items(pred_row):
        uid = item_uid(item)
        if uid and uid not in out:
            out[uid] = item
    return out


def get_question(gold_row: dict[str, Any]) -> str:
    for key in ("question", "query", "text"):
        value = gold_row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def rerank_one(
    *,
    qid: str,
    gold_row: dict[str, Any],
    base_row: dict[str, Any],
    retrieval_model: Any,
    cache: DocEmbeddingCache,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_pages = ranked_pages(base_row)
    output_row = dict(base_row)
    output_row["qid"] = str(base_row.get("qid", qid))

    if len(base_pages) < int(args.boundary_rank):
        case = {
            "qid": qid,
            "accepted": False,
            "reason": "not_enough_pages",
            "base_page_count": len(base_pages),
        }
        return output_row, case

    boundary_page = base_pages[int(args.boundary_rank) - 1]
    top_pages = base_pages[: int(args.hit_k)]
    if args.same_doc_only and page_doc(boundary_page) not in {page_doc(uid) for uid in top_pages}:
        case = {
            "qid": qid,
            "accepted": False,
            "reason": "boundary_doc_not_in_topk",
            "boundary_page": boundary_page,
            "top_pages": top_pages,
        }
        return output_row, case

    local_pages = list(dict.fromkeys(top_pages + [boundary_page]))
    question = get_question(gold_row)
    query_meta = retrieval_model.encode_query_with_metadata(
        query=question,
        to_cpu=True,
        query_token_filter=str(args.query_token_filter),
    )
    exact_scores = exact_maxsim_scores(
        retrieval_model=retrieval_model,
        query_embeds=query_meta["embeddings"],
        page_uids=local_pages,
        cache=cache,
        batch_size=int(args.rerank_batch_size),
    )
    reordered_pages, accepted, boundary_page, weakest_top = reorder_boundary(
        base_pages=base_pages,
        exact_scores=exact_scores,
        hit_k=int(args.hit_k),
        boundary_rank=int(args.boundary_rank),
        mode=str(args.mode),
    )

    item_map = prediction_item_by_uid(base_row)
    output_row["page_retrieval_results"] = [item_map[uid] for uid in reordered_pages if uid in item_map]
    gold_pages = gold_page_uids(gold_row)
    case = {
        "qid": qid,
        "question": question,
        "accepted": bool(accepted),
        "reason": "exact_maxsim_boundary_win" if accepted else "base_boundary_preserved",
        "mode": str(args.mode),
        "hit_k": int(args.hit_k),
        "boundary_rank": int(args.boundary_rank),
        "top_pages": top_pages,
        "boundary_page": boundary_page,
        "weakest_top_page": weakest_top,
        "exact_scores": {uid: exact_scores.get(uid) for uid in local_pages},
        "boundary_exact_score": exact_scores.get(boundary_page) if boundary_page else None,
        "weakest_top_exact_score": exact_scores.get(weakest_top) if weakest_top else None,
        "exact_margin": (
            exact_scores.get(boundary_page, 0.0) - exact_scores.get(weakest_top, 0.0)
            if boundary_page and weakest_top
            else None
        ),
        "query_token_count": int(query_meta["embeddings"].shape[0]),
        "base_first_gold_page_rank": first_rank(base_pages, gold_pages),
        "candidate_first_gold_page_rank": first_rank(ranked_pages(output_row), gold_pages),
        "boundary_is_gold": boundary_page in gold_pages if boundary_page else False,
        "weakest_top_is_gold": weakest_top in gold_pages if weakest_top else False,
    }
    return output_row, case


def evaluate(
    *,
    gold: dict[str, dict[str, Any]],
    base: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    cases: list[dict[str, Any]],
    recall_ks: list[int],
    hit_k: int,
) -> dict[str, Any]:
    page_recall: dict[int, list[float]] = defaultdict(list)
    doc_recall: dict[int, list[float]] = defaultdict(list)
    movement_counts: Counter[str] = Counter()
    page_hit_count = 0
    doc_hit_count = 0
    base_page_hit_count = 0
    base_doc_hit_count = 0

    for qid in sorted(set(gold) & set(base) & set(candidate)):
        gold_pages = gold_page_uids(gold[qid])
        gold_docs = gold_doc_ids(gold[qid])
        base_scores = metric_scores(base[qid], gold_pages, gold_docs, recall_ks, hit_k)
        cand_scores = metric_scores(candidate[qid], gold_pages, gold_docs, recall_ks, hit_k)
        base_page_rank = base_scores["page_first_rank"]
        cand_page_rank = cand_scores["page_first_rank"]

        base_page_hit_count += int(bool(base_scores[f"page_hit@{hit_k}"]))
        base_doc_hit_count += int(bool(base_scores[f"doc_hit@{hit_k}"]))
        page_hit_count += int(bool(cand_scores[f"page_hit@{hit_k}"]))
        doc_hit_count += int(bool(cand_scores[f"doc_hit@{hit_k}"]))
        movement_counts[movement_for_hit(base_page_rank, cand_page_rank, hit_k)] += 1

        for k in recall_ks:
            page_recall[int(k)].append(float(cand_scores.get(f"page_recall@{k}", 0.0)))
            doc_recall[int(k)].append(float(cand_scores.get(f"doc_recall@{k}", 0.0)))

    accepted_count = sum(1 for row in cases if row.get("accepted"))
    recovered = int(movement_counts.get("recovered", 0))
    lost = int(movement_counts.get("lost", 0))
    return {
        "n": len(set(gold) & set(base) & set(candidate)),
        "accepted_count": accepted_count,
        "accept_frac": accepted_count / float(len(cases)) if cases else 0.0,
        "base_page_hit_at_k_count": base_page_hit_count,
        "page_hit_at_k_count": page_hit_count,
        "base_doc_hit_at_k_count": base_doc_hit_count,
        "doc_hit_at_k_count": doc_hit_count,
        "movement_counts": dict(sorted(movement_counts.items())),
        "recovered": recovered,
        "lost": lost,
        "net_recovered": recovered - lost,
        "page_recall_at_k": {str(k): mean(values) for k, values in sorted(page_recall.items())},
        "doc_recall_at_k": {str(k): mean(values) for k, values in sorted(doc_recall.items())},
    }


def iter_qids(gold: dict[str, Any], base: dict[str, Any], args: argparse.Namespace) -> list[str]:
    qids = sorted(set(gold) & set(base))
    if int(args.num_shards) > 1:
        if int(args.shard_index) < 0 or int(args.shard_index) >= int(args.num_shards):
            raise ValueError("--shard-index must be in [0, num_shards).")
        qids = [qid for idx, qid in enumerate(qids) if idx % int(args.num_shards) == int(args.shard_index)]
    if int(args.max_qids) > 0:
        qids = qids[: int(args.max_qids)]
    return qids


def progress(iterable: list[str]) -> Any:
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, desc="Exact MaxSim boundary rerank")
    except Exception:
        return iterable


def main() -> None:
    args = parse_args()
    if int(args.boundary_rank) <= int(args.hit_k):
        raise ValueError("--boundary-rank must be greater than --hit-k.")
    if not Path(args.embedding_dir).exists():
        raise FileNotFoundError(f"Missing embedding dir: {args.embedding_dir}")

    from m3docrag.retrieval import ColPaliRetrievalModel

    gold_rows = {str(row["qid"]): row for row in read_jsonl(Path(args.gold))}
    base_prediction = load_prediction(Path(args.base_prediction))
    qids = iter_qids(gold_rows, base_prediction, args)

    retrieval_model = ColPaliRetrievalModel(
        backbone_name_or_path=resolve_model_path(str(args.retrieval_model_name_or_path)),
        adapter_name_or_path=resolve_model_path(str(args.retrieval_adapter_model_name_or_path)),
        dtype=dtype_from_name(str(args.dtype)),
    )
    cache = DocEmbeddingCache(Path(args.embedding_dir), int(args.doc_cache_size))

    output_rows: dict[str, dict[str, Any]] = {}
    cases: list[dict[str, Any]] = []
    for qid in progress(qids):
        try:
            output_row, case = rerank_one(
                qid=qid,
                gold_row=gold_rows[qid],
                base_row=base_prediction[qid],
                retrieval_model=retrieval_model,
                cache=cache,
                args=args,
            )
        except Exception as exc:
            output_row = dict(base_prediction[qid])
            output_row["qid"] = str(output_row.get("qid", qid))
            case = {
                "qid": qid,
                "accepted": False,
                "reason": "error_kept_base",
                "error": f"{type(exc).__name__}: {exc}",
            }
        output_rows[qid] = output_row
        cases.append(case)

    output_prediction = Path(args.output_prediction_json)
    write_prediction(output_prediction, output_rows)

    summary = evaluate(
        gold=gold_rows,
        base=base_prediction,
        candidate=output_rows,
        cases=cases,
        recall_ks=[int(k) for k in args.recall_ks],
        hit_k=int(args.hit_k),
    )
    summary.update(
        {
            "gold": str(args.gold),
            "base_prediction": str(args.base_prediction),
            "embedding_dir": str(args.embedding_dir),
            "retrieval_model_name_or_path": str(args.retrieval_model_name_or_path),
            "retrieval_adapter_model_name_or_path": str(args.retrieval_adapter_model_name_or_path),
            "query_token_filter": str(args.query_token_filter),
            "dtype": str(args.dtype),
            "hit_k": int(args.hit_k),
            "boundary_rank": int(args.boundary_rank),
            "mode": str(args.mode),
            "same_doc_only": bool(args.same_doc_only),
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
            "max_qids": int(args.max_qids),
        }
    )

    output_summary = Path(args.output_summary_json)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.output_case_json:
        output_case = Path(args.output_case_json)
        output_case.parent.mkdir(parents=True, exist_ok=True)
        output_case.write_text(json.dumps({"cases": cases}, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"saved_prediction: {output_prediction}")
    print(f"saved_summary: {output_summary}")
    if args.output_case_json:
        print(f"saved_cases: {args.output_case_json}")
    print(f"n {summary['n']}")
    print(f"accepted {summary['accepted_count']}")
    print(f"page_hit_at_{int(args.hit_k)}_count {summary['page_hit_at_k_count']}")
    print(f"base_page_hit_at_{int(args.hit_k)}_count {summary['base_page_hit_at_k_count']}")
    print(f"recovered {summary['recovered']}")
    print(f"lost {summary['lost']}")
    print(f"net_recovered {summary['net_recovered']}")
    print(f"page_recall_at_k {summary['page_recall_at_k']}")
    print(f"doc_recall_at_k {summary['doc_recall_at_k']}")


if __name__ == "__main__":
    main()
