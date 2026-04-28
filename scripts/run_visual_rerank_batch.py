#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rerank_target_docs_visual_aware import (
    QUERY_TOKEN_FILTER_CHOICES,
    WeightConfig,
    axis_class_counts,
    build_page_id_metadata,
    build_page_token_classes,
    build_rankings,
    clean_token_label,
    compute_page_feature,
    grid_search_weights,
    load_patch_axis_classes_for_pages,
    load_splice_query_axis_classes,
    make_query_score_mask,
    parse_float_list,
    resolve_model_path,
    summarize_gold_doc_ranks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-run the standalone visual-aware reranker over a set of qids and "
            "write per-qid + aggregate summaries."
        )
    )
    parser.add_argument(
        "--qid-jsonl",
        required=True,
        help="JSONL containing qids to evaluate, e.g. ret4_imagelistq_failures_no_gold_doc_in_top4.jsonl",
    )
    parser.add_argument(
        "--qid-field",
        default="qid",
        help="Field name containing the qid inside --qid-jsonl rows.",
    )
    parser.add_argument("--gold", required=True, help="Path to MMQA_<split>.jsonl")
    parser.add_argument(
        "--embedding_name",
        default="colpali-v1.2_m3-docvqa_dev",
    )
    parser.add_argument(
        "--query_token_filter",
        default="full",
        choices=QUERY_TOKEN_FILTER_CHOICES,
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
        "--splice-query-token-labels",
        required=True,
    )
    parser.add_argument(
        "--splice-patch-labels-jsonl",
        required=True,
    )
    parser.add_argument(
        "--baseline-pred",
        required=True,
        help="Baseline retrieval JSON used to source the fixed page pool.",
    )
    parser.add_argument(
        "--from-baseline-top-pages",
        type=int,
        default=1000,
        help="Number of baseline page rows to keep per qid.",
    )
    parser.add_argument("--weight-base", type=float, default=1.0)
    parser.add_argument("--weight-visual", type=float, default=1.0)
    parser.add_argument("--weight-non-visual", type=float, default=0.0)
    parser.add_argument("--weight-balance", type=float, default=8.0)
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Run qid-specific grid search instead of using the fixed weights.",
    )
    parser.add_argument("--grid-base-values", default="1.0")
    parser.add_argument("--grid-visual-values", default="0,0.25,0.5,1.0,2.0")
    parser.add_argument("--grid-non-visual-values", default="0,0.25,0.5,1.0,2.0")
    parser.add_argument("--grid-balance-values", default="0,0.25,0.5,1.0,2.0")
    parser.add_argument(
        "--max-qids",
        type=int,
        default=0,
        help="Optional cap for smoke tests.",
    )
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Per-qid JSONL summary output.",
    )
    parser.add_argument(
        "--output-summary-json",
        required=True,
        help="Aggregate summary JSON output.",
    )
    return parser.parse_args()


def load_qids(path: Path, qid_field: str, max_qids: int) -> list[str]:
    qids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get(qid_field, "")).strip()
            if not qid:
                continue
            qids.append(qid)
            if max_qids > 0 and len(qids) >= max_qids:
                break
    if not qids:
        raise ValueError(f"No qids found in {path} using field {qid_field!r}")
    return qids


def load_gold_rows(path: Path, qids: set[str]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            qid = str(row.get("qid", "")).strip()
            if qid in qids:
                rows[qid] = row
    missing = sorted(qids - set(rows))
    if missing:
        raise KeyError(f"Missing {len(missing)} qids in gold file: {missing[:10]}")
    return rows


def load_baseline_payload(path: Path, qids: set[str]) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = sorted(qids - set(payload))
    if missing:
        raise KeyError(f"Missing {len(missing)} qids in baseline prediction: {missing[:10]}")
    return {qid: payload[qid] for qid in qids}


def build_baseline_pool(
    rows: list[list[object]],
    top_pages: int,
) -> tuple[list[str], list[str], dict[str, int], list[str]]:
    baseline_page_uids: list[str] = []
    candidate_doc_ids: list[str] = []
    baseline_doc_rank_map: dict[str, int] = {}
    seen_docs: set[str] = set()

    for row_rank, row in enumerate(rows, start=1):
        doc_id = str(row[0]).strip()
        page_idx = int(row[1])
        page_uid = f"{doc_id}_page{page_idx}"
        if row_rank <= top_pages:
            baseline_page_uids.append(page_uid)
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                candidate_doc_ids.append(doc_id)
        if doc_id and doc_id not in baseline_doc_rank_map:
            baseline_doc_rank_map[doc_id] = len(baseline_doc_rank_map) + 1

    return candidate_doc_ids, baseline_page_uids, baseline_doc_rank_map, [str(x.split("_page")[0]) for x in baseline_page_uids]


def baseline_gold_page_hits(
    rows: list[list[object]],
    gold_doc_ids: set[str],
) -> list[dict]:
    hits = []
    for row_rank, row in enumerate(rows, start=1):
        doc_id = str(row[0]).strip()
        if doc_id not in gold_doc_ids:
            continue
        page_idx = int(row[1])
        hits.append(
            {
                "rank": row_rank,
                "page_uid": f"{doc_id}_page{page_idx}",
                "doc_id": doc_id,
                "page_idx": page_idx,
                "score": float(row[2]),
            }
        )
    return hits


def reranked_gold_page_hits(
    reranked_pages: list[dict],
    gold_doc_ids: set[str],
) -> list[dict]:
    return [row for row in reranked_pages if row["doc_id"] in gold_doc_ids]


def median_or_none(values: list[int]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def main() -> None:
    args = parse_args()

    qids = load_qids(Path(args.qid_jsonl), args.qid_field, args.max_qids)
    gold_rows = load_gold_rows(Path(args.gold), set(qids))
    baseline_payload = load_baseline_payload(Path(args.baseline_pred), set(qids))

    import torch
    from m3docrag.retrieval import ColPaliRetrievalModel
    from scripts.rerank_target_docs_visual_aware import load_doc_embeddings_for_doc_ids

    retrieval_model = ColPaliRetrievalModel(
        backbone_name_or_path=resolve_model_path(args.retrieval_model_name_or_path),
        adapter_name_or_path=resolve_model_path(args.retrieval_adapter_model_name_or_path),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fixed_weights = WeightConfig(
        base=args.weight_base,
        visual=args.weight_visual,
        non_visual=args.weight_non_visual,
        balance=args.weight_balance,
    )
    grid_base_values = parse_float_list(args.grid_base_values)
    grid_visual_values = parse_float_list(args.grid_visual_values)
    grid_non_visual_values = parse_float_list(args.grid_non_visual_values)
    grid_balance_values = parse_float_list(args.grid_balance_values)

    all_rows: list[dict] = []
    improved_doc = 0
    worsened_doc = 0
    unchanged_doc = 0
    reranked_top4_doc = 0
    baseline_top4_doc = 0

    for idx, qid in enumerate(qids, start=1):
        gold_row = gold_rows[qid]
        gold_doc_ids = sorted({str(item["doc_id"]).strip() for item in gold_row.get("supporting_context", [])})
        gold_doc_id_set = set(gold_doc_ids)

        baseline_rows = baseline_payload[qid].get("page_retrieval_results", [])
        candidate_doc_ids, baseline_page_uids, baseline_doc_rank_map, _ = build_baseline_pool(
            baseline_rows,
            args.from_baseline_top_pages,
        )
        explicit_page_uids = set(baseline_page_uids)

        docid2embs = load_doc_embeddings_for_doc_ids(candidate_doc_ids, args.embedding_name)
        page_specs, page_meta = build_page_id_metadata(
            docid2embs=docid2embs,
            explicit_page_uids=explicit_page_uids,
            nonspatial_token_position=args.nonspatial_token_position,
        )
        patch_axis_classes_by_uid = load_patch_axis_classes_for_pages(
            labels_jsonl=args.splice_patch_labels_jsonl,
            page_meta=page_meta,
        )

        query_text = gold_row["question"]
        query_meta = retrieval_model.encode_query_with_metadata(
            query=query_text,
            to_cpu=True,
            query_token_filter=args.query_token_filter,
        )
        query_emb = query_meta["embeddings"].float().to(device=device, dtype=torch.float32)
        query_raw_tokens = query_meta.get("raw_tokens", [])
        query_token_labels = [clean_token_label(token) for token in query_raw_tokens]
        query_axis_classes = load_splice_query_axis_classes(
            query_labels_path=args.splice_query_token_labels,
            qid=qid,
            query_token_labels=query_token_labels,
            query_raw_tokens=query_raw_tokens,
        )
        query_score_mask = make_query_score_mask(
            query_raw_tokens=query_raw_tokens,
            ignore_pad_scores_in_final_ranking=args.ignore_pad_scores_in_final_ranking,
        )

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
            weights, best_grid_record, _grid_leaderboard = grid_search_weights(
                page_features=page_features,
                baseline_doc_rank_map=baseline_doc_rank_map,
                gold_doc_ids=gold_doc_ids,
                gold_page_uids=[],
                base_values=grid_base_values,
                visual_values=grid_visual_values,
                non_visual_values=grid_non_visual_values,
                balance_values=grid_balance_values,
            )
        else:
            weights = fixed_weights
            best_grid_record = None

        reranked_docs, reranked_pages = build_rankings(
            page_features=page_features,
            weights=weights,
            baseline_doc_rank_map=baseline_doc_rank_map,
        )
        gold_doc_summary = summarize_gold_doc_ranks(reranked_docs, gold_doc_ids)

        baseline_page_hits = baseline_gold_page_hits(baseline_rows, gold_doc_id_set)
        baseline_first_gold_doc_rank = min((baseline_doc_rank_map.get(doc_id) for doc_id in gold_doc_ids if doc_id in baseline_doc_rank_map), default=None)
        baseline_first_gold_page_rank = baseline_page_hits[0]["rank"] if baseline_page_hits else None

        reranked_page_hits = reranked_gold_page_hits(reranked_pages, gold_doc_id_set)
        reranked_first_gold_page_rank = reranked_page_hits[0]["rank"] if reranked_page_hits else None

        reranked_first_gold_doc_rank = gold_doc_summary["first_gold_doc_rank"]
        if baseline_first_gold_doc_rank is not None and reranked_first_gold_doc_rank is not None:
            if reranked_first_gold_doc_rank < baseline_first_gold_doc_rank:
                improved_doc += 1
            elif reranked_first_gold_doc_rank > baseline_first_gold_doc_rank:
                worsened_doc += 1
            else:
                unchanged_doc += 1

        if baseline_first_gold_doc_rank is not None and baseline_first_gold_doc_rank <= 4:
            baseline_top4_doc += 1
        if reranked_first_gold_doc_rank is not None and reranked_first_gold_doc_rank <= 4:
            reranked_top4_doc += 1

        row = {
            "qid": qid,
            "question": query_text,
            "answers": [str(item["answer"]) for item in gold_row.get("answers", [])],
            "gold_doc_ids": gold_doc_ids,
            "candidate_doc_count": len(candidate_doc_ids),
            "candidate_page_count": len(page_features),
            "query_axis_class_counts": axis_class_counts(query_axis_classes),
            "weights": asdict(weights),
            "grid_search_enabled": args.grid_search,
            "grid_search_best": best_grid_record,
            "baseline_first_gold_doc_rank": baseline_first_gold_doc_rank,
            "baseline_first_gold_page_rank": baseline_first_gold_page_rank,
            "baseline_gold_page_hits_top10": baseline_page_hits[:10],
            "reranked_first_gold_doc_rank": reranked_first_gold_doc_rank,
            "reranked_gold_doc_hits_at_4": gold_doc_summary["gold_doc_hits_at_4"],
            "reranked_first_gold_page_rank_any_gold_doc_page": reranked_first_gold_page_rank,
            "reranked_gold_page_hits_top10_any_gold_doc_page": reranked_page_hits[:10],
            "reranked_gold_doc_ranks": gold_doc_summary["gold_doc_ranks"],
        }
        all_rows.append(row)
        print(
            f"[{idx}/{len(qids)}] {qid} "
            f"baseline_doc={baseline_first_gold_doc_rank} "
            f"baseline_page={baseline_first_gold_page_rank} "
            f"reranked_doc={reranked_first_gold_doc_rank} "
            f"reranked_page={reranked_first_gold_page_rank}"
        )

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    baseline_doc_ranks = [row["baseline_first_gold_doc_rank"] for row in all_rows if row["baseline_first_gold_doc_rank"] is not None]
    reranked_doc_ranks = [row["reranked_first_gold_doc_rank"] for row in all_rows if row["reranked_first_gold_doc_rank"] is not None]
    baseline_page_ranks = [row["baseline_first_gold_page_rank"] for row in all_rows if row["baseline_first_gold_page_rank"] is not None]
    reranked_page_ranks = [
        row["reranked_first_gold_page_rank_any_gold_doc_page"]
        for row in all_rows
        if row["reranked_first_gold_page_rank_any_gold_doc_page"] is not None
    ]

    summary = {
        "input_qid_jsonl": args.qid_jsonl,
        "gold": args.gold,
        "baseline_pred": args.baseline_pred,
        "from_baseline_top_pages": args.from_baseline_top_pages,
        "embedding_name": args.embedding_name,
        "query_token_filter": args.query_token_filter,
        "splice_query_token_labels": args.splice_query_token_labels,
        "splice_patch_labels_jsonl": args.splice_patch_labels_jsonl,
        "grid_search_enabled": args.grid_search,
        "fixed_weights": asdict(fixed_weights),
        "grid_base_values": grid_base_values,
        "grid_visual_values": grid_visual_values,
        "grid_non_visual_values": grid_non_visual_values,
        "grid_balance_values": grid_balance_values,
        "num_qids": len(all_rows),
        "baseline_top4_doc_count": baseline_top4_doc,
        "reranked_top4_doc_count": reranked_top4_doc,
        "improved_doc_rank_count": improved_doc,
        "worsened_doc_rank_count": worsened_doc,
        "unchanged_doc_rank_count": unchanged_doc,
        "baseline_doc_rank_median": median_or_none(baseline_doc_ranks),
        "reranked_doc_rank_median": median_or_none(reranked_doc_ranks),
        "baseline_page_rank_median": median_or_none(baseline_page_ranks),
        "reranked_page_rank_median": median_or_none(reranked_page_ranks),
        "top4_doc_qids": [row["qid"] for row in all_rows if row["reranked_first_gold_doc_rank"] is not None and row["reranked_first_gold_doc_rank"] <= 4],
    }

    output_summary_json = Path(args.output_summary_json)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"saved_jsonl: {output_jsonl}")
    print(f"saved_summary: {output_summary_json}")
    print(f"num_qids: {len(all_rows)}")
    print(f"improved_doc_rank_count: {improved_doc}")
    print(f"reranked_top4_doc_count: {reranked_top4_doc}")


if __name__ == "__main__":
    main()
