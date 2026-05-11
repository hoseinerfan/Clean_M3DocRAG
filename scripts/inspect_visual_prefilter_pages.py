#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

import torch

from m3docrag.retrieval import ColPaliRetrievalModel
from scripts.rerank_target_docs_visual_aware import (
    VISUAL_PREFILTER_SORT_KEY_CHOICES,
    VISUAL_SCORE_QUERY_MODE_CHOICES,
    apply_visual_rerank_to_top_pages,
    build_page_id_metadata,
    build_page_token_classes,
    clean_token_label,
    extract_query_route_features,
    load_doc_embeddings_for_doc_ids,
    load_patch_axis_classes_for_pages,
    load_splice_query_axis_classes,
    make_base_only_page_feature,
    make_query_score_mask,
    resolve_model_path,
    visual_prefilter_primary_score,
    visual_prefilter_sort_key,
)
from scripts.run_visual_rerank_batch import (
    build_baseline_pool,
    load_baseline_payload,
    load_gold_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect explicit gold/wrong pages for one qid under visual-prefilter scoring. "
            "This scores the saved baseline top-page pool, then reports how the requested pages "
            "rank under each visual prefilter mode."
        )
    )
    parser.add_argument("--qid", required=True)
    parser.add_argument("--gold", required=True, help="Gold MMQA_<split>.jsonl")
    parser.add_argument("--baseline-pred", required=True, help="Saved retrieval JSON with page_retrieval_results")
    parser.add_argument("--embedding_name", default="colpali-v1.2_m3-docvqa_dev")
    parser.add_argument("--query_token_filter", default="full", choices=["full", "drop_pad_like", "semantic_only"])
    parser.add_argument("--retrieval_model_name_or_path", required=True)
    parser.add_argument("--retrieval_adapter_model_name_or_path", required=True)
    parser.add_argument("--splice-query-token-labels", required=True)
    parser.add_argument("--splice-patch-labels-jsonl", required=True)
    parser.add_argument("--from-baseline-top-pages", type=int, default=1000)
    parser.add_argument("--gold-page-uid", action="append", default=[], help="Explicit gold page uid; pass multiple times if needed.")
    parser.add_argument("--wrong-page-uid", action="append", default=[], help="Explicit wrong page uid; pass multiple times if needed.")
    parser.add_argument("--extra-page-uid", action="append", default=[], help="Optional extra page uid to inspect.")
    parser.add_argument(
        "--mode",
        action="append",
        default=[],
        choices=list(VISUAL_PREFILTER_SORT_KEY_CHOICES),
        help="Optional specific prefilter mode(s) to report. Defaults to all modes.",
    )
    parser.add_argument("--require-informative-visual-query", action="store_true")
    parser.add_argument("--filter-to-informative-visual-query", action="store_true")
    parser.add_argument(
        "--balance-score-mode",
        default="visual_x_grounded_nonvisual_avg",
        choices=["min_avg", "visual_x_nonvisual_avg", "visual_x_grounded_nonvisual_avg"],
    )
    parser.add_argument("--grounded-context-radius", type=int, default=2)
    parser.add_argument("--visual-fallback-all-token-weight", type=float, default=0.0)
    parser.add_argument(
        "--visual-score-query-mode",
        default="visual_query_only",
        choices=VISUAL_SCORE_QUERY_MODE_CHOICES,
    )
    parser.add_argument("--nonspatial-token-position", default="suffix", choices=["prefix", "suffix"])
    parser.add_argument("--ignore-pad-scores-in-final-ranking", action="store_true")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def split_page_uid(page_uid: str) -> tuple[str, int]:
    if "_page" not in page_uid:
        raise ValueError(f"Invalid page uid: {page_uid!r}")
    doc_id, page_idx_text = page_uid.rsplit("_page", 1)
    return doc_id, int(page_idx_text)


def ordered_unique(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def main() -> None:
    args = parse_args()
    inspected_page_uids = ordered_unique(args.gold_page_uid + args.wrong_page_uid + args.extra_page_uid)
    if not inspected_page_uids:
        raise ValueError("Provide at least one --gold-page-uid, --wrong-page-uid, or --extra-page-uid.")

    qids = {args.qid}
    gold_rows = load_gold_rows(Path(args.gold), qids)
    baseline_payload = load_baseline_payload(Path(args.baseline_pred), qids)
    gold_row = gold_rows[args.qid]
    gold_doc_ids = sorted({str(item["doc_id"]).strip() for item in gold_row.get("supporting_context", [])})
    gold_doc_id_set = set(gold_doc_ids)

    baseline_rows = baseline_payload[args.qid].get("page_retrieval_results", [])
    candidate_doc_ids, baseline_page_uids, _baseline_doc_rank_map, _ignored, baseline_page_score_map = build_baseline_pool(
        baseline_rows,
        args.from_baseline_top_pages,
    )
    baseline_rank_map = {
        page_uid: rank
        for rank, page_uid in enumerate(baseline_page_uids, start=1)
    }

    for page_uid in inspected_page_uids:
        doc_id, _page_idx = split_page_uid(page_uid)
        if doc_id not in candidate_doc_ids:
            candidate_doc_ids.append(doc_id)

    retrieval_model = ColPaliRetrievalModel(
        backbone_name_or_path=resolve_model_path(args.retrieval_model_name_or_path),
        adapter_name_or_path=resolve_model_path(args.retrieval_adapter_model_name_or_path),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    docid2embs = load_doc_embeddings_for_doc_ids(candidate_doc_ids, args.embedding_name)
    explicit_page_uids = set(baseline_page_uids) | set(inspected_page_uids)
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
        qid=args.qid,
        query_token_labels=query_token_labels,
        query_raw_tokens=query_raw_tokens,
    )
    route_features = extract_query_route_features(
        question_type=str(gold_row.get("metadata", {}).get("type", "UNKNOWN")),
        query_axis_classes=query_axis_classes,
        query_token_labels=query_token_labels,
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
                base_page_score=float(baseline_page_score_map.get(page_uid, 0.0)),
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
            top_pages=len(page_specs),
            require_informative_visual_query=args.require_informative_visual_query,
            filter_to_informative_visual_query=args.filter_to_informative_visual_query,
            preserve_stage1_base_score=True,
            balance_score_mode=args.balance_score_mode,
            grounded_context_radius=args.grounded_context_radius,
            visual_fallback_all_token_weight=args.visual_fallback_all_token_weight,
            visual_score_query_mode=args.visual_score_query_mode,
        )

    feature_by_uid = {feature.page_uid: feature for feature in visual_features}
    for page_uid in inspected_page_uids:
        if page_uid not in feature_by_uid:
            raise KeyError(f"Requested page uid was not scored: {page_uid}")

    modes = args.mode or list(VISUAL_PREFILTER_SORT_KEY_CHOICES)
    rank_maps: dict[str, dict[str, int]] = {}
    for mode in modes:
        sorted_features = sorted(
            visual_features,
            key=lambda item: visual_prefilter_sort_key(item, mode),
            reverse=True,
        )
        rank_maps[mode] = {
            feature.page_uid: rank
            for rank, feature in enumerate(sorted_features, start=1)
        }

    role_by_uid: dict[str, str] = {}
    for page_uid in args.gold_page_uid:
        role_by_uid[page_uid] = "gold"
    for page_uid in args.wrong_page_uid:
        role_by_uid[page_uid] = "wrong"
    for page_uid in args.extra_page_uid:
        role_by_uid.setdefault(page_uid, "extra")

    inspected_pages: list[dict[str, object]] = []
    for page_uid in inspected_page_uids:
        feature = feature_by_uid[page_uid]
        inspected_pages.append(
            {
                "page_uid": page_uid,
                "role": role_by_uid.get(page_uid, "extra"),
                "doc_id": feature.doc_id,
                "page_idx": int(feature.page_idx),
                "is_gold_doc": feature.doc_id in gold_doc_id_set,
                "in_baseline_top_pool": page_uid in baseline_rank_map,
                "baseline_rank": baseline_rank_map.get(page_uid),
                "base_page_score": float(feature.base_page_score),
                "visual_page_score": float(feature.visual_page_score),
                "confirmed_visual_page_score": float(feature.confirmed_visual_page_score),
                "grounded_non_visual_page_score": float(feature.grounded_non_visual_page_score),
                "grounded_context_page_score": float(feature.grounded_context_page_score),
                "non_visual_page_score": float(feature.non_visual_page_score),
                "balance_score": float(feature.balance_score),
                "visual_alignment_ratio": float(feature.visual_alignment_ratio),
                "non_visual_alignment_ratio": float(feature.non_visual_alignment_ratio),
                "visual_anchor_patch_count": int(feature.visual_anchor_patch_count),
                "grounded_non_visual_patch_count": int(feature.grounded_non_visual_patch_count),
                "grounded_context_patch_count": int(feature.grounded_context_patch_count),
                "mode_primary_scores": {
                    mode: float(visual_prefilter_primary_score(feature, mode))
                    for mode in modes
                },
                "mode_full_ranks": {
                    mode: int(rank_maps[mode][page_uid])
                    for mode in modes
                },
                "mode_top50_flags": {
                    mode: bool(rank_maps[mode][page_uid] <= 50)
                    for mode in modes
                },
            }
        )

    gold_page_uids = [page_uid for page_uid in inspected_page_uids if role_by_uid.get(page_uid) == "gold"]
    wrong_page_uids = [page_uid for page_uid in inspected_page_uids if role_by_uid.get(page_uid) == "wrong"]
    mode_comparisons: dict[str, dict[str, object]] = {}
    for mode in modes:
        best_gold_rank = min((rank_maps[mode][page_uid] for page_uid in gold_page_uids), default=None)
        best_wrong_rank = min((rank_maps[mode][page_uid] for page_uid in wrong_page_uids), default=None)
        inspected_order = sorted(
            inspected_page_uids,
            key=lambda page_uid: rank_maps[mode][page_uid],
        )
        mode_comparisons[mode] = {
            "best_gold_rank": best_gold_rank,
            "best_wrong_rank": best_wrong_rank,
            "gold_beats_wrong": (
                None if best_gold_rank is None or best_wrong_rank is None else bool(best_gold_rank < best_wrong_rank)
            ),
            "inspected_order": inspected_order,
        }

    output = {
        "qid": args.qid,
        "question": gold_row["question"],
        "gold_doc_ids": gold_doc_ids,
        "route_features": route_features,
        "visual_score_query_mode": args.visual_score_query_mode,
        "balance_score_mode": args.balance_score_mode,
        "grounded_context_radius": args.grounded_context_radius,
        "modes": modes,
        "inspected_pages": inspected_pages,
        "mode_comparisons": mode_comparisons,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    for mode in modes:
        comparison = mode_comparisons[mode]
        print(
            f"mode={mode} "
            f"best_gold_rank={comparison['best_gold_rank']} "
            f"best_wrong_rank={comparison['best_wrong_rank']} "
            f"gold_beats_wrong={comparison['gold_beats_wrong']}"
        )
    print(f"saved_json: {output_path}")


if __name__ == "__main__":
    main()
