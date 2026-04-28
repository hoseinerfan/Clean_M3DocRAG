#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

import faiss
import numpy as np
import torch

from m3docrag.datasets.m3_docvqa import M3DocVQADataset
from m3docrag.rag import MultimodalRAGModel
from m3docrag.retrieval import ColPaliRetrievalModel
from m3docrag.utils.paths import LOCAL_EMBEDDINGS_DIR
from scripts.rerank_target_docs_visual_aware import (
    QUERY_TOKEN_FILTER_CHOICES,
    WeightConfig,
    axis_class_counts,
    build_page_id_metadata,
    build_page_token_classes,
    build_prediction_payload,
    build_rankings,
    clean_token_label,
    compute_page_feature,
    load_doc_embeddings_for_doc_ids,
    load_gold_row_from_qid,
    load_patch_axis_classes_for_pages,
    load_splice_query_axis_classes,
    make_query_score_mask,
    parse_float_list,
    resolve_model_path,
    summarize_gold_doc_ranks,
    summarize_gold_page_ranks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone helper for compositional query decomposition experiments. "
            "It retrieves pages for the original query plus explicit subqueries, "
            "merges their candidate pools, and optionally applies the visual-aware reranker."
        )
    )
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--query", help="Free-form question text")
    query_group.add_argument("--qid", help="Benchmark qid to load from MMQA_<split>.jsonl")

    parser.add_argument("--data_name", default="m3-docvqa")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--gold", help="Optional MMQA_<split>.jsonl override for --qid mode")
    parser.add_argument("--embedding_name", default="colpali-v1.2_m3-docvqa_dev")
    parser.add_argument(
        "--faiss_index_type",
        default="ivfflat",
        choices=["flatip", "ivfflat", "ivfpq"],
    )
    parser.add_argument(
        "--query_token_filter",
        default="full",
        choices=QUERY_TOKEN_FILTER_CHOICES,
        help="Reranker-side query token filter for the original question.",
    )
    parser.add_argument(
        "--retrieval_query_token_filter",
        default="full",
        choices=QUERY_TOKEN_FILTER_CHOICES,
        help="Retrieval-side query token filter used for the original query and each subquery in FAISS search.",
    )
    parser.add_argument(
        "--ignore-pad-scores-in-final-ranking",
        action="store_true",
    )
    parser.add_argument(
        "--nonspatial-token-position",
        default="suffix",
        choices=["prefix", "suffix"],
    )
    parser.add_argument("--retrieval_model_name_or_path", default="colpaligemma-3b-pt-448-base")
    parser.add_argument("--retrieval_adapter_model_name_or_path", default="colpali-v1.2")

    parser.add_argument(
        "--subquery",
        action="append",
        default=[],
        help="Additional retrieval subquery. Pass multiple times.",
    )
    parser.add_argument(
        "--subqueries-file",
        default="",
        help="Optional text/JSON file with additional retrieval subqueries.",
    )
    parser.add_argument(
        "--include-original-query",
        action="store_true",
        help="Include the original question as one retrieval channel.",
    )
    parser.add_argument(
        "--top-pages-per-query",
        type=int,
        default=300,
        help="How many retrieved page rows to keep from each query channel before merging.",
    )
    parser.add_argument(
        "--merge-method",
        default="rrf",
        choices=["rrf", "min_rank"],
        help="How to merge per-query page ranks into one candidate pool.",
    )
    parser.add_argument(
        "--rrf-k",
        type=float,
        default=60.0,
        help="RRF denominator constant when --merge-method=rrf.",
    )

    parser.add_argument(
        "--splice-query-token-labels",
        required=True,
        help="Query-token visual/non-visual label file used for the reranker channels.",
    )
    parser.add_argument(
        "--splice-patch-labels-jsonl",
        required=True,
        help="Patch-level visual/non-visual label JSONL used for page-channel masking.",
    )

    parser.add_argument(
        "--gold-doc-id",
        action="append",
        default=[],
        help="Optional gold doc_id override(s). If omitted in --qid mode, uses supporting_context doc_ids.",
    )
    parser.add_argument(
        "--gold-page-uid",
        action="append",
        default=[],
        help="Optional manual gold page_uid(s), e.g. <doc_id>_page<idx>.",
    )

    parser.add_argument("--weight-base", type=float, default=1.0)
    parser.add_argument("--weight-visual", type=float, default=1.0)
    parser.add_argument("--weight-non-visual", type=float, default=0.0)
    parser.add_argument("--weight-balance", type=float, default=8.0)
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Search over weight combinations on the merged candidate pool.",
    )
    parser.add_argument("--grid-base-values", default="1.0")
    parser.add_argument("--grid-visual-values", default="0,0.5,1.0,2.0,4.0")
    parser.add_argument("--grid-non-visual-values", default="0,0.5,1.0,2.0")
    parser.add_argument("--grid-balance-values", default="0,1.0,2.0,4.0,6.0,8.0")
    parser.add_argument(
        "--report-topn",
        type=int,
        default=20,
        help="How many top merged/reranked docs/pages to keep in the JSON summary.",
    )
    parser.add_argument("--output-json", default="", help="Optional summary JSON output path.")
    parser.add_argument(
        "--output-prediction-json",
        default="",
        help="Optional run_rag-style prediction JSON path for reranked merged pages.",
    )
    return parser.parse_args()


def parse_string_list_file(path_str: str) -> list[str]:
    if not path_str:
        return []
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [str(item).strip() for item in payload if str(item).strip()]
        if isinstance(payload, dict):
            for key in ("subqueries", "queries", "items", "values"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [str(item).strip() for item in value if str(item).strip()]
        raise ValueError(f"Unsupported JSON structure in {path}")

    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_dataset_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        data_name=args.data_name,
        split=args.split,
        data_len=None,
        use_dummy_images=False,
        load_embedding=True,
        embedding_name=args.embedding_name,
        max_pages=20,
        do_page_padding=False,
        retrieval_model_type="colpali",
        use_retrieval=True,
        retrieval_only=True,
        page_retrieval_type="logits",
        loop_unique_doc_ids=False,
        n_retrieval_pages=args.top_pages_per_query,
        faiss_index_type=args.faiss_index_type,
        model_name_or_path="",
        retrieval_model_name_or_path=args.retrieval_model_name_or_path,
        retrieval_adapter_model_name_or_path=args.retrieval_adapter_model_name_or_path,
        bits=16,
        do_image_splitting=False,
    )


def load_query_text_and_gold_doc_ids(args: argparse.Namespace) -> tuple[str, list[str]]:
    gold_doc_ids = [str(value).strip() for value in args.gold_doc_id if str(value).strip()]
    if args.query:
        return args.query, gold_doc_ids

    gold_row = load_gold_row_from_qid(
        SimpleNamespace(
            qid=args.qid,
            gold=args.gold,
            data_name=args.data_name,
            split=args.split,
        )
    )
    if not gold_doc_ids:
        gold_doc_ids = sorted({str(item["doc_id"]).strip() for item in gold_row.get("supporting_context", [])})
    return gold_row["question"], gold_doc_ids


def resolve_index_dir(embedding_name: str, faiss_index_type: str) -> Path:
    candidate_dirs = [
        Path(LOCAL_EMBEDDINGS_DIR) / f"{embedding_name}_pageindex_{faiss_index_type}",
        REPO_ROOT / "embeddings" / f"{embedding_name}_pageindex_{faiss_index_type}",
    ]
    index_dir = next((path for path in candidate_dirs if path.exists()), None)
    if index_dir is None:
        searched = "\n".join(f"  {path}" for path in candidate_dirs)
        raise FileNotFoundError(
            "Could not resolve FAISS index directory in offline mode. Checked:\n"
            f"{searched}"
        )
    return index_dir


def build_flattened_index_inputs(docid2embs: dict[str, torch.Tensor]) -> tuple[list[str], torch.Tensor]:
    token2pageuid: list[str] = []
    all_token_embeddings = []
    for doc_id, doc_emb in docid2embs.items():
        for page_idx in range(len(doc_emb)):
            page_emb = doc_emb[page_idx].view(-1, doc_emb[page_idx].shape[-1])
            all_token_embeddings.append(page_emb)
            page_uid = f"{doc_id}_page{page_idx}"
            token2pageuid.extend([page_uid] * page_emb.shape[0])

    return token2pageuid, torch.cat(all_token_embeddings, dim=0)


def retrieve_pages_from_index_with_filter(
    *,
    retrieval_model: ColPaliRetrievalModel,
    query: str,
    index,
    token2pageuid: list[str],
    all_token_embeddings_np: np.ndarray,
    n_return_pages: int,
    query_token_filter: str,
    ignore_pad_scores_in_final_ranking: bool,
) -> list[tuple[str, int, float]]:
    query_meta = retrieval_model.encode_query_with_metadata(
        query=query,
        to_cpu=True,
        query_token_filter=query_token_filter,
    )
    query_emb = query_meta["embeddings"].float().numpy().astype(np.float32)
    raw_tokens = query_meta.get("raw_tokens", [])
    score_active_query_token_mask = None
    if ignore_pad_scores_in_final_ranking:
        score_active_query_token_mask = [token != "<pad>" for token in raw_tokens]
        if score_active_query_token_mask and not any(score_active_query_token_mask):
            raise ValueError(
                "Ignoring PAD scores in final ranking removed every scoring query token."
            )

    D, I = index.search(query_emb, n_return_pages)

    final_page2scores: dict[str, float] = {}
    for q_idx, query_token_emb in enumerate(query_emb):
        current_q_page2scores: dict[str, float] = {}
        for nn_idx in range(n_return_pages):
            found_nearest_doc_token_idx = int(I[q_idx, nn_idx])
            page_uid = token2pageuid[found_nearest_doc_token_idx]
            doc_token_emb = all_token_embeddings_np[found_nearest_doc_token_idx]
            score = float((query_token_emb * doc_token_emb).sum())
            if page_uid not in current_q_page2scores:
                current_q_page2scores[page_uid] = score
            else:
                current_q_page2scores[page_uid] = max(current_q_page2scores[page_uid], score)

        if score_active_query_token_mask is not None and not score_active_query_token_mask[q_idx]:
            continue

        for page_uid, score in current_q_page2scores.items():
            if page_uid in final_page2scores:
                final_page2scores[page_uid] += score
            else:
                final_page2scores[page_uid] = score

    sorted_pages = sorted(final_page2scores.items(), key=lambda x: x[1], reverse=True)
    top_k_pages = sorted_pages[:n_return_pages]

    return [
        (page_uid.split("_page")[0], int(page_uid.split("_page")[-1]), score)
        for page_uid, score in top_k_pages
    ]


def merge_page_rows(
    per_query_rows: dict[str, list[tuple[str, int, float]]],
    *,
    merge_method: str,
    rrf_k: float,
) -> list[dict]:
    merged: dict[str, dict] = {}
    for query_label, rows in per_query_rows.items():
        for rank, (doc_id, page_idx, score) in enumerate(rows, start=1):
            page_uid = f"{doc_id}_page{page_idx}"
            item = merged.setdefault(
                page_uid,
                {
                    "page_uid": page_uid,
                    "doc_id": doc_id,
                    "page_idx": page_idx,
                    "rrf_score": 0.0,
                    "best_rank": rank,
                    "best_raw_score": float(score),
                    "sources": [],
                },
            )
            item["best_rank"] = min(item["best_rank"], rank)
            item["best_raw_score"] = max(item["best_raw_score"], float(score))
            if merge_method == "rrf":
                item["rrf_score"] += 1.0 / (rrf_k + rank)
            item["sources"].append(
                {
                    "query_label": query_label,
                    "rank": rank,
                    "score": float(score),
                }
            )

    if merge_method == "rrf":
        key_fn = lambda row: (row["rrf_score"], -row["best_rank"], row["best_raw_score"])
    else:
        key_fn = lambda row: (-row["best_rank"], row["best_raw_score"])

    merged_rows = sorted(merged.values(), key=key_fn, reverse=True)
    for rank, row in enumerate(merged_rows, start=1):
        row["merged_rank"] = rank
    return merged_rows


def build_baseline_doc_rank_map_from_pages(merged_rows: list[dict]) -> dict[str, int]:
    doc_rank_map: dict[str, int] = {}
    for row in merged_rows:
        doc_id = row["doc_id"]
        if doc_id not in doc_rank_map:
            doc_rank_map[doc_id] = len(doc_rank_map) + 1
    return doc_rank_map


def summarize_merged_gold_doc_ranks(merged_rows: list[dict], gold_doc_ids: list[str]) -> dict:
    gold_doc_id_set = set(gold_doc_ids)
    doc_rank_map = build_baseline_doc_rank_map_from_pages(merged_rows)
    doc_hits = [
        {
            "doc_id": doc_id,
            "rank": rank,
        }
        for doc_id, rank in doc_rank_map.items()
        if doc_id in gold_doc_id_set
    ]
    doc_hits.sort(key=lambda row: row["rank"])
    first_rank = doc_hits[0]["rank"] if doc_hits else None
    return {
        "gold_doc_ids": gold_doc_ids,
        "first_gold_doc_rank": first_rank,
        "gold_doc_hits_at_4": sum(item["rank"] <= 4 for item in doc_hits),
        "gold_doc_ranks": doc_hits,
    }


def summarize_merged_gold_page_ranks(merged_rows: list[dict], gold_page_uids: list[str]) -> dict:
    gold_page_uid_set = set(gold_page_uids)
    hits = [row for row in merged_rows if row["page_uid"] in gold_page_uid_set]
    first_rank = hits[0]["merged_rank"] if hits else None
    return {
        "gold_page_uids": gold_page_uids,
        "first_gold_page_rank": first_rank,
        "gold_page_hits_at_4": sum(item["merged_rank"] <= 4 for item in hits),
        "gold_page_ranks": [
            {
                "page_uid": item["page_uid"],
                "rank": item["merged_rank"],
                "doc_id": item["doc_id"],
                "page_idx": item["page_idx"],
            }
            for item in hits
        ],
    }


def main() -> None:
    args = parse_args()

    original_query, gold_doc_ids = load_query_text_and_gold_doc_ids(args)
    gold_page_uids = [str(value).strip() for value in args.gold_page_uid if str(value).strip()]

    subqueries = [str(item).strip() for item in args.subquery if str(item).strip()]
    subqueries.extend(parse_string_list_file(args.subqueries_file))
    if args.include_original_query:
        retrieval_queries = [("original_query", original_query)]
    else:
        retrieval_queries = []
    retrieval_queries.extend((f"subquery_{idx+1}", text) for idx, text in enumerate(subqueries))
    if not retrieval_queries:
        raise ValueError("Provide at least one retrieval query via --include-original-query and/or --subquery.")

    dataset = M3DocVQADataset(make_dataset_args(args))
    candidate_doc_ids = list(dataset.all_supporting_doc_ids)
    docid2embs = load_doc_embeddings_for_doc_ids(candidate_doc_ids, args.embedding_name)

    index_dir = resolve_index_dir(args.embedding_name, args.faiss_index_type)
    index = faiss.read_index(str(index_dir / "index.bin"))
    token2pageuid, all_token_embeddings = build_flattened_index_inputs(docid2embs)
    all_token_embeddings_np = all_token_embeddings.float().numpy()

    retrieval_model = ColPaliRetrievalModel(
        backbone_name_or_path=resolve_model_path(args.retrieval_model_name_or_path),
        adapter_name_or_path=resolve_model_path(args.retrieval_adapter_model_name_or_path),
    )
    rag_model = MultimodalRAGModel(retrieval_model=retrieval_model, vqa_model=None)

    per_query_rows: dict[str, list[tuple[str, int, float]]] = {}
    for query_label, query_text in retrieval_queries:
        rows = retrieve_pages_from_index_with_filter(
            retrieval_model=retrieval_model,
            query=query_text,
            index=index,
            token2pageuid=token2pageuid,
            all_token_embeddings_np=all_token_embeddings_np,
            n_return_pages=args.top_pages_per_query,
            query_token_filter=args.retrieval_query_token_filter,
            ignore_pad_scores_in_final_ranking=False,
        )
        per_query_rows[query_label] = rows

    merged_rows = merge_page_rows(
        per_query_rows=per_query_rows,
        merge_method=args.merge_method,
        rrf_k=args.rrf_k,
    )
    explicit_page_uids = {row["page_uid"] for row in merged_rows}
    merged_candidate_doc_ids = []
    seen_docs = set()
    for row in merged_rows:
        if row["doc_id"] not in seen_docs:
            seen_docs.add(row["doc_id"])
            merged_candidate_doc_ids.append(row["doc_id"])
    baseline_doc_rank_map = build_baseline_doc_rank_map_from_pages(merged_rows)

    page_specs, page_meta = build_page_id_metadata(
        docid2embs=docid2embs,
        explicit_page_uids=explicit_page_uids,
        nonspatial_token_position=args.nonspatial_token_position,
    )
    patch_axis_classes_by_uid = load_patch_axis_classes_for_pages(
        labels_jsonl=args.splice_patch_labels_jsonl,
        page_meta=page_meta,
    )

    query_meta = retrieval_model.encode_query_with_metadata(
        query=original_query,
        to_cpu=True,
        query_token_filter=args.query_token_filter,
    )
    query_emb = query_meta["embeddings"].float()
    query_raw_tokens = query_meta.get("raw_tokens", [])
    query_token_labels = [clean_token_label(token) for token in query_raw_tokens]
    query_axis_classes = load_splice_query_axis_classes(
        query_labels_path=args.splice_query_token_labels,
        qid=args.qid,
        query_token_labels=query_token_labels,
        query_raw_tokens=query_raw_tokens,
    )
    query_score_mask = make_query_score_mask(
        query_raw_tokens=query_raw_tokens,
        ignore_pad_scores_in_final_ranking=args.ignore_pad_scores_in_final_ranking,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query_emb = query_emb.to(device=device, dtype=torch.float32)

    page_features = []
    with torch.no_grad():
        for doc_id, page_idx in page_specs:
            page_uid = f"{doc_id}_page{page_idx}"
            page_emb = docid2embs[doc_id][page_idx].view(-1, docid2embs[doc_id][page_idx].shape[-1]).to(
                device=device,
                dtype=torch.float32,
            )
            page_token_classes = build_page_token_classes(
                page_meta=page_meta[page_uid],
                patch_axis_classes=patch_axis_classes_by_uid[page_uid],
            )
            page_features.append(
                compute_page_feature(
                    page_emb=page_emb,
                    query_emb=query_emb,
                    query_axis_classes=query_axis_classes,
                    query_score_mask=query_score_mask,
                    page_token_classes=page_token_classes,
                    doc_id=doc_id,
                    page_idx=page_idx,
                )
            )

    if args.grid_search:
        from scripts.rerank_target_docs_visual_aware import grid_search_weights

        weights, best_grid_record, grid_leaderboard = grid_search_weights(
            page_features=page_features,
            baseline_doc_rank_map=baseline_doc_rank_map,
            gold_doc_ids=gold_doc_ids,
            gold_page_uids=gold_page_uids,
            base_values=parse_float_list(args.grid_base_values),
            visual_values=parse_float_list(args.grid_visual_values),
            non_visual_values=parse_float_list(args.grid_non_visual_values),
            balance_values=parse_float_list(args.grid_balance_values),
        )
    else:
        weights = WeightConfig(
            base=args.weight_base,
            visual=args.weight_visual,
            non_visual=args.weight_non_visual,
            balance=args.weight_balance,
        )
        best_grid_record = None
        grid_leaderboard = []

    reranked_docs, reranked_pages = build_rankings(
        page_features=page_features,
        weights=weights,
        baseline_doc_rank_map=baseline_doc_rank_map,
    )
    merged_gold_doc_summary = summarize_merged_gold_doc_ranks(merged_rows, gold_doc_ids)
    merged_gold_page_summary = summarize_merged_gold_page_ranks(merged_rows, gold_page_uids)
    reranked_gold_doc_summary = summarize_gold_doc_ranks(reranked_docs, gold_doc_ids)
    reranked_gold_page_summary = summarize_gold_page_ranks(reranked_pages, gold_page_uids)

    summary = {
        "qid": args.qid,
        "query": original_query,
        "subqueries": [text for _label, text in retrieval_queries],
        "retrieval_queries": [{"label": label, "query": text} for label, text in retrieval_queries],
        "merge_method": args.merge_method,
        "rrf_k": args.rrf_k,
        "top_pages_per_query": args.top_pages_per_query,
        "embedding_name": args.embedding_name,
        "faiss_index_type": args.faiss_index_type,
        "query_token_filter": args.query_token_filter,
        "retrieval_query_token_filter": args.retrieval_query_token_filter,
        "ignore_pad_scores_in_final_ranking": args.ignore_pad_scores_in_final_ranking,
        "candidate_doc_count": len(merged_candidate_doc_ids),
        "candidate_page_count": len(page_features),
        "query_token_labels": query_token_labels,
        "query_axis_classes": query_axis_classes,
        "query_axis_class_counts": axis_class_counts(query_axis_classes),
        "weights": asdict(weights),
        "grid_search": {
            "enabled": args.grid_search,
            "best": best_grid_record,
            "leaderboard": grid_leaderboard,
        },
        "per_query_top_rows": {
            label: [
                {
                    "rank": rank,
                    "doc_id": doc_id,
                    "page_idx": page_idx,
                    "page_uid": f"{doc_id}_page{page_idx}",
                    "score": float(score),
                }
                for rank, (doc_id, page_idx, score) in enumerate(rows[: args.report_topn], start=1)
            ]
            for label, rows in per_query_rows.items()
        },
        "merged_top_pages": merged_rows[: args.report_topn],
        "merged_gold_doc_summary": merged_gold_doc_summary,
        "merged_gold_page_summary": merged_gold_page_summary,
        "reranked_gold_doc_summary": reranked_gold_doc_summary,
        "reranked_gold_page_summary": reranked_gold_page_summary,
        "top_reranked_docs": reranked_docs[: args.report_topn],
        "top_reranked_pages": reranked_pages[: args.report_topn],
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.output_prediction_json:
        prediction_payload = build_prediction_payload(
            qid=args.qid,
            query=original_query,
            reranked_pages=reranked_pages,
            metadata={
                "weights": asdict(weights),
                "query_token_filter": args.query_token_filter,
                "merge_method": args.merge_method,
                "rrf_k": args.rrf_k,
                "retrieval_queries": [{"label": label, "query": text} for label, text in retrieval_queries],
                "top_pages_per_query": args.top_pages_per_query,
                "gold_doc_ids": gold_doc_ids,
                "gold_page_uids": gold_page_uids,
            },
        )
        prediction_path = Path(args.output_prediction_json)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.write_text(json.dumps(prediction_payload, indent=2) + "\n", encoding="utf-8")

    print(f"query_token_filter: {args.query_token_filter}")
    print(f"retrieval_query_token_filter: {args.retrieval_query_token_filter}")
    print(f"merge_method: {args.merge_method}")
    print(f"candidate_doc_count: {len(merged_candidate_doc_ids)}")
    print(f"candidate_page_count: {len(page_features)}")
    print(f"query_axis_class_counts: {axis_class_counts(query_axis_classes)}")
    print(f"retrieval_queries: {[text for _label, text in retrieval_queries]}")
    if gold_doc_ids:
        print(f"gold_doc_ids: {gold_doc_ids}")
        print(f"merged_first_gold_doc_rank: {merged_gold_doc_summary['first_gold_doc_rank']}")
        print(f"reranked_first_gold_doc_rank: {reranked_gold_doc_summary['first_gold_doc_rank']}")
    if gold_page_uids:
        print(f"gold_page_uids: {gold_page_uids}")
        print(f"merged_first_gold_page_rank: {merged_gold_page_summary['first_gold_page_rank']}")
        print(f"reranked_first_gold_page_rank: {reranked_gold_page_summary['first_gold_page_rank']}")
    print(f"weights: {asdict(weights)}")
    print("top_reranked_docs:")
    for item in reranked_docs[: min(args.report_topn, 10)]:
        print(
            f"  {item['rank']:>4} {item['doc_id']} "
            f"score={item['fused_doc_score']:.4f} page={item['best_page_uid']}"
        )
    if args.output_json:
        print(f"saved_summary: {args.output_json}")
    if args.output_prediction_json:
        print(f"saved_prediction: {args.output_prediction_json}")


if __name__ == "__main__":
    main()
