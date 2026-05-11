#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

import torch

from m3docrag.retrieval import ColPaliRetrievalModel
from scripts.rerank_target_docs_visual_aware import (
    NON_VISUAL_PAGE_MODE_CHOICES,
    VISUAL_SCORE_QUERY_MODE_CHOICES,
    VISUAL_PREFILTER_SORT_KEY_CHOICES,
    apply_visual_rerank_to_top_pages,
    build_page_id_metadata,
    build_page_token_classes,
    clean_token_label,
    load_doc_embeddings_for_doc_ids,
    load_patch_axis_classes_for_pages,
    load_splice_query_axis_classes,
    make_base_only_page_feature,
    make_query_score_mask,
    resolve_model_path,
    visual_prefilter_primary_score,
    visual_prefilter_sort_key_with_threshold,
)
from scripts.run_visual_rerank_batch import (
    build_baseline_pool,
    load_baseline_payload,
    load_gold_rows,
    load_qids,
)

PREFILTER_SORT_KEY_CHOICES = list(VISUAL_PREFILTER_SORT_KEY_CHOICES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check gold-doc coverage after visual prefiltering inside a saved baseline page pool. "
            "This uses the ready --baseline-pred top-page pool, computes visual-grounded page scores, "
            "and reports gold-doc coverage among the top-N visually prefiltered pages."
        )
    )
    parser.add_argument("--gold", required=True, help="Gold MMQA_<split>.jsonl")
    parser.add_argument("--baseline-pred", required=True, help="Saved retrieval JSON containing page_retrieval_results")
    parser.add_argument(
        "--qid-jsonl",
        help="Optional JSONL file listing qids to evaluate. If omitted, use all qids from the baseline file.",
    )
    parser.add_argument("--qid-field", default="qid")
    parser.add_argument("--embedding_name", default="colpali-v1.2_m3-docvqa_dev")
    parser.add_argument("--query_token_filter", default="full", choices=["full", "drop_pad_like", "semantic_only"])
    parser.add_argument("--retrieval_model_name_or_path", required=True)
    parser.add_argument("--retrieval_adapter_model_name_or_path", required=True)
    parser.add_argument("--splice-query-token-labels", required=True)
    parser.add_argument("--splice-patch-labels-jsonl", required=True)
    parser.add_argument("--from-baseline-top-pages", type=int, default=1000)
    parser.add_argument(
        "--visual-prefilter-top-pages",
        type=int,
        default=50,
        help="Evaluate gold-doc coverage among the top-N visually prefiltered page rows.",
    )
    parser.add_argument(
        "--require-informative-visual-query",
        action="store_true",
        help="If the query has no informative visual tokens, skip visual scoring and keep baseline order.",
    )
    parser.add_argument(
        "--filter-to-informative-visual-query",
        action="store_true",
        help="Restrict visual scoring to informative visual query tokens only.",
    )
    parser.add_argument(
        "--balance-score-mode",
        default="visual_x_grounded_nonvisual_avg",
        choices=["min_avg", "visual_x_nonvisual_avg", "visual_x_grounded_nonvisual_avg"],
    )
    parser.add_argument(
        "--prefilter-sort-key",
        default="balance_then_visual",
        choices=PREFILTER_SORT_KEY_CHOICES,
        help="How to rank pages during the visual prefilter stage.",
    )
    parser.add_argument(
        "--confirmed-visual-gate-threshold",
        type=float,
        default=0.1,
        help=(
            "Threshold used by prefilter_sort_key=non_visual_with_confirmed_visual_gate. "
            "Pages below this confirmed_visual score are demoted behind all gated-in pages."
        ),
    )
    parser.add_argument("--grounded-context-radius", type=int, default=2)
    parser.add_argument("--visual-fallback-all-token-weight", type=float, default=0.0)
    parser.add_argument(
        "--visual-score-query-mode",
        default="visual_query_only",
        choices=VISUAL_SCORE_QUERY_MODE_CHOICES,
    )
    parser.add_argument(
        "--non-visual-page-mode",
        default="labeled_only",
        choices=NON_VISUAL_PAGE_MODE_CHOICES,
    )
    parser.add_argument("--nonspatial-token-position", default="suffix", choices=["prefix", "suffix"])
    parser.add_argument("--ignore-pad-scores-in-final-ranking", action="store_true")
    parser.add_argument("--max-qids", type=int, default=0)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-summary-json", required=True)
    return parser.parse_args()


def dedupe_doc_ids_from_pages(rows: list[dict]) -> list[str]:
    doc_ids: list[str] = []
    seen: set[str] = set()
    for row in rows:
        doc_id = str(row["doc_id"])
        if doc_id in seen:
            continue
        seen.add(doc_id)
        doc_ids.append(doc_id)
    return doc_ids


def summarize_top_pages(
    *,
    selected_pages: list[dict],
    gold_doc_ids: list[str],
) -> dict[str, object]:
    gold_doc_id_set = set(gold_doc_ids)
    page_hits = [
        {
            "rank": int(idx),
            "page_uid": row["page_uid"],
            "doc_id": row["doc_id"],
            "page_idx": int(row["page_idx"]),
        }
        for idx, row in enumerate(selected_pages, start=1)
        if row["doc_id"] in gold_doc_id_set
    ]
    dedup_doc_ids = dedupe_doc_ids_from_pages(selected_pages)
    gold_doc_ranks = [
        {
            "rank": int(idx),
            "doc_id": doc_id,
        }
        for idx, doc_id in enumerate(dedup_doc_ids, start=1)
        if doc_id in gold_doc_id_set
    ]
    hit_doc_ids = sorted({row["doc_id"] for row in page_hits})
    coverage_ratio = (
        len(hit_doc_ids) / float(len(gold_doc_ids))
        if gold_doc_ids
        else 0.0
    )
    return {
        "selected_page_count": len(selected_pages),
        "gold_doc_hit_count": len(hit_doc_ids),
        "gold_doc_coverage_ratio": float(coverage_ratio),
        "first_gold_page_rank": None if not page_hits else int(page_hits[0]["rank"]),
        "first_gold_doc_rank_deduped": None if not gold_doc_ranks else int(gold_doc_ranks[0]["rank"]),
        "gold_page_hits": page_hits,
        "gold_doc_hits_deduped": gold_doc_ranks,
        "hit_gold_doc_ids": hit_doc_ids,
    }


def aggregate_summary(per_qid: list[dict], key: str) -> dict[str, object]:
    summaries = [row[key] for row in per_qid]
    coverages = [float(item["gold_doc_coverage_ratio"]) for item in summaries]
    first_doc_ranks = [
        int(item["first_gold_doc_rank_deduped"])
        for item in summaries
        if item["first_gold_doc_rank_deduped"] is not None
    ]
    return {
        "mean_gold_doc_coverage_ratio": float(sum(coverages) / len(coverages)) if coverages else 0.0,
        "median_gold_doc_coverage_ratio": float(median(coverages)) if coverages else None,
        "full_coverage_qids": sum(float(item["gold_doc_coverage_ratio"]) >= 0.999999 for item in summaries),
        "any_coverage_qids": sum(float(item["gold_doc_coverage_ratio"]) > 0.0 for item in summaries),
        "zero_coverage_qids": sum(float(item["gold_doc_coverage_ratio"]) == 0.0 for item in summaries),
        "mean_first_gold_doc_rank_deduped": (
            float(sum(first_doc_ranks) / len(first_doc_ranks))
            if first_doc_ranks
            else None
        ),
        "median_first_gold_doc_rank_deduped": (
            float(median(first_doc_ranks))
            if first_doc_ranks
            else None
        ),
    }


def main() -> None:
    args = parse_args()

    if args.visual_prefilter_top_pages <= 0:
        raise ValueError("--visual-prefilter-top-pages must be > 0")
    if args.from_baseline_top_pages <= 0:
        raise ValueError("--from-baseline-top-pages must be > 0")
    if args.visual_prefilter_top_pages > args.from_baseline_top_pages:
        raise ValueError("--visual-prefilter-top-pages cannot exceed --from-baseline-top-pages")

    baseline_path = Path(args.baseline_pred)
    baseline_payload_all = json.loads(baseline_path.read_text(encoding="utf-8"))
    if args.qid_jsonl:
        qids = load_qids(Path(args.qid_jsonl), args.qid_field, args.max_qids)
    else:
        qids = sorted(baseline_payload_all.keys())
        if args.max_qids > 0:
            qids = qids[: args.max_qids]
    if not qids:
        raise ValueError("No qids selected.")

    gold_rows = load_gold_rows(Path(args.gold), set(qids))
    baseline_payload = load_baseline_payload(baseline_path, set(qids))

    retrieval_model = ColPaliRetrievalModel(
        backbone_name_or_path=resolve_model_path(args.retrieval_model_name_or_path),
        adapter_name_or_path=resolve_model_path(args.retrieval_adapter_model_name_or_path),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_rows: list[dict] = []

    for idx, qid in enumerate(qids, start=1):
        gold_row = gold_rows[qid]
        gold_doc_ids = sorted({str(item["doc_id"]).strip() for item in gold_row.get("supporting_context", [])})

        baseline_rows = baseline_payload[qid].get("page_retrieval_results", [])
        candidate_doc_ids, baseline_page_uids, _baseline_doc_rank_map, _ignored, baseline_page_score_map = build_baseline_pool(
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

        query_meta = retrieval_model.encode_query_with_metadata(
            query=gold_row["question"],
            to_cpu=True,
            query_token_filter=args.query_token_filter,
        )
        query_emb = query_meta["embeddings"].float().to(device=device, dtype=torch.float32)
        query_raw_tokens = query_meta.get("raw_tokens", [])
        query_token_labels = [clean_token_label(token) for token in query_raw_tokens]
        query_score_mask = make_query_score_mask(
            query_raw_tokens=query_raw_tokens,
            ignore_pad_scores_in_final_ranking=args.ignore_pad_scores_in_final_ranking,
        )

        query_axis_classes = load_splice_query_axis_classes(
            query_labels_path=args.splice_query_token_labels,
            qid=qid,
            query_token_labels=query_token_labels,
            query_raw_tokens=query_raw_tokens,
        )
        patch_axis_classes_by_uid = load_patch_axis_classes_for_pages(
            labels_jsonl=args.splice_patch_labels_jsonl,
            page_meta=page_meta,
        )
        page_token_classes_by_uid = {
            page_uid: build_page_token_classes(
                page_meta=page_meta[page_uid],
                patch_axis_classes=patch_axis_classes,
            )
            for page_uid, patch_axis_classes in patch_axis_classes_by_uid.items()
        }

        baseline_page_features = []
        for doc_id, page_idx in page_specs:
            page_uid = f"{doc_id}_page{page_idx}"
            baseline_page_features.append(
                make_base_only_page_feature(
                    doc_id=doc_id,
                    page_idx=page_idx,
                    base_page_score=baseline_page_score_map[page_uid],
                )
            )

        with torch.no_grad():
            visual_features = apply_visual_rerank_to_top_pages(
                page_features=baseline_page_features,
                docid2embs=docid2embs,
                query_emb=query_emb,
                query_axis_classes=query_axis_classes,
                query_token_labels=query_token_labels,
                query_score_mask=query_score_mask,
                page_token_classes_by_uid=page_token_classes_by_uid,
                page_meta_by_uid=page_meta,
                top_pages=args.from_baseline_top_pages,
                require_informative_visual_query=args.require_informative_visual_query,
                filter_to_informative_visual_query=args.filter_to_informative_visual_query,
                preserve_stage1_base_score=True,
                balance_score_mode=args.balance_score_mode,
                grounded_context_radius=args.grounded_context_radius,
                visual_fallback_all_token_weight=args.visual_fallback_all_token_weight,
                visual_score_query_mode=args.visual_score_query_mode,
                non_visual_page_mode=args.non_visual_page_mode,
            )

        baseline_top_pages = [
            {
                "rank": int(rank),
                "page_uid": page_uid,
                "doc_id": page_uid.rsplit("_page", 1)[0],
                "page_idx": int(page_uid.rsplit("_page", 1)[1]),
            }
            for rank, page_uid in enumerate(baseline_page_uids[: args.visual_prefilter_top_pages], start=1)
        ]

        selected_visual_features = sorted(
            visual_features,
            key=lambda item: visual_prefilter_sort_key_with_threshold(
                item,
                mode=args.prefilter_sort_key,
                confirmed_visual_gate_threshold=args.confirmed_visual_gate_threshold,
            ),
            reverse=True,
        )[: args.visual_prefilter_top_pages]
        visual_top_pages = [
            {
                "rank": int(rank),
                "page_uid": feature.page_uid,
                "doc_id": feature.doc_id,
                "page_idx": int(feature.page_idx),
                "prefilter_primary_score": visual_prefilter_primary_score(
                    feature,
                    args.prefilter_sort_key,
                    confirmed_visual_gate_threshold=args.confirmed_visual_gate_threshold,
                ),
                "base_page_score": float(feature.base_page_score),
                "visual_page_score": float(feature.visual_page_score),
                "confirmed_visual_page_score": float(feature.confirmed_visual_page_score),
                "grounded_non_visual_page_score": float(feature.grounded_non_visual_page_score),
                "grounded_context_page_score": float(feature.grounded_context_page_score),
                "non_visual_page_score": float(feature.non_visual_page_score),
                "balance_score": float(feature.balance_score),
            }
            for rank, feature in enumerate(selected_visual_features, start=1)
        ]

        baseline_top_summary = summarize_top_pages(
            selected_pages=baseline_top_pages,
            gold_doc_ids=gold_doc_ids,
        )
        visual_top_summary = summarize_top_pages(
            selected_pages=visual_top_pages,
            gold_doc_ids=gold_doc_ids,
        )

        row = {
            "qid": qid,
            "question": gold_row["question"],
            "gold_doc_ids": gold_doc_ids,
            "from_baseline_top_pages": args.from_baseline_top_pages,
            "visual_prefilter_top_pages": args.visual_prefilter_top_pages,
            "baseline_top_pages_summary": baseline_top_summary,
            "visual_prefilter_top_pages_summary": visual_top_summary,
            "visual_prefilter_top_pages_trace": visual_top_pages,
        }
        all_rows.append(row)
        print(
            f"[{idx}/{len(qids)}] {qid} "
            f"mode={args.prefilter_sort_key} "
            f"baseline_cov@{args.visual_prefilter_top_pages}={baseline_top_summary['gold_doc_coverage_ratio']:.3f} "
            f"visual_cov@{args.visual_prefilter_top_pages}={visual_top_summary['gold_doc_coverage_ratio']:.3f}"
        )

    baseline_summary = aggregate_summary(all_rows, "baseline_top_pages_summary")
    visual_summary = aggregate_summary(all_rows, "visual_prefilter_top_pages_summary")
    improved = 0
    worsened = 0
    unchanged = 0
    rescued_from_zero = 0
    for row in all_rows:
        base_cov = float(row["baseline_top_pages_summary"]["gold_doc_coverage_ratio"])
        vis_cov = float(row["visual_prefilter_top_pages_summary"]["gold_doc_coverage_ratio"])
        if vis_cov > base_cov:
            improved += 1
        elif vis_cov < base_cov:
            worsened += 1
        else:
            unchanged += 1
        if base_cov == 0.0 and vis_cov > 0.0:
            rescued_from_zero += 1

    summary = {
        "n_qids": len(all_rows),
        "from_baseline_top_pages": args.from_baseline_top_pages,
        "visual_prefilter_top_pages": args.visual_prefilter_top_pages,
        "prefilter_sort_key": args.prefilter_sort_key,
        "confirmed_visual_gate_threshold": args.confirmed_visual_gate_threshold,
        "balance_score_mode": args.balance_score_mode,
        "non_visual_page_mode": args.non_visual_page_mode,
        "visual_score_query_mode": args.visual_score_query_mode,
        "grounded_context_radius": args.grounded_context_radius,
        "require_informative_visual_query": args.require_informative_visual_query,
        "filter_to_informative_visual_query": args.filter_to_informative_visual_query,
        "baseline_top_pages_summary": baseline_summary,
        "visual_prefilter_top_pages_summary": visual_summary,
        "coverage_comparison": {
            "improved_qids": improved,
            "worsened_qids": worsened,
            "unchanged_qids": unchanged,
            "rescued_from_zero_coverage_qids": rescued_from_zero,
        },
    }

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    output_summary_json = Path(args.output_summary_json)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"saved_jsonl: {output_jsonl}")
    print(f"saved_summary: {output_summary_json}")
    print(f"num_qids: {len(all_rows)}")
    print(f"baseline_mean_cov@{args.visual_prefilter_top_pages}: {baseline_summary['mean_gold_doc_coverage_ratio']:.6f}")
    print(f"visual_mean_cov@{args.visual_prefilter_top_pages}: {visual_summary['mean_gold_doc_coverage_ratio']:.6f}")


if __name__ == "__main__":
    main()
