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
)
from scripts.run_visual_rerank_batch import (
    build_baseline_pool,
    load_baseline_payload,
    load_gold_rows,
    load_qids,
)

PREFILTER_SORT_KEY_CHOICES = [
    "balance_then_visual",
    "balance_only",
    "visual_only",
    "non_visual_only",
    "grounded_non_visual_only",
]

GENERIC_VISUAL_QUERY_TOKENS = {
    "poster",
    "logo",
    "title",
    "team",
    "album",
    "movie",
    "film",
    "club",
    "player",
    "scoring",
    "list",
    "member",
    "members",
    "season",
    "results",
    "filmography",
    "discography",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether visual query tokens and visual page patches look informative/reliable "
            "for visual prefiltering on selected qids from a saved baseline page pool."
        )
    )
    parser.add_argument("--gold", required=True, help="Gold MMQA_<split>.jsonl")
    parser.add_argument("--baseline-pred", required=True, help="Saved retrieval JSON with page_retrieval_results")
    parser.add_argument("--qid-jsonl", help="Optional JSONL file listing qids")
    parser.add_argument("--qid-field", default="qid")
    parser.add_argument("--qid", action="append", default=[], help="Optional individual qid; pass multiple times")
    parser.add_argument("--embedding_name", default="colpali-v1.2_m3-docvqa_dev")
    parser.add_argument("--query_token_filter", default="full", choices=["full", "drop_pad_like", "semantic_only"])
    parser.add_argument("--retrieval_model_name_or_path", required=True)
    parser.add_argument("--retrieval_adapter_model_name_or_path", required=True)
    parser.add_argument("--splice-query-token-labels", required=True)
    parser.add_argument("--splice-patch-labels-jsonl", required=True)
    parser.add_argument("--from-baseline-top-pages", type=int, default=1000)
    parser.add_argument("--visual-prefilter-top-pages", type=int, default=50)
    parser.add_argument("--report-topn", type=int, default=20)
    parser.add_argument("--require-informative-visual-query", action="store_true")
    parser.add_argument("--filter-to-informative-visual-query", action="store_true")
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
    parser.add_argument("--grounded-context-radius", type=int, default=2)
    parser.add_argument("--visual-fallback-all-token-weight", type=float, default=0.0)
    parser.add_argument(
        "--visual-score-query-mode",
        default="visual_query_only",
        choices=VISUAL_SCORE_QUERY_MODE_CHOICES,
    )
    parser.add_argument("--nonspatial-token-position", default="suffix", choices=["prefix", "suffix"])
    parser.add_argument("--ignore-pad-scores-in-final-ranking", action="store_true")
    parser.add_argument("--max-qids", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def summarize_group(features: list[dict]) -> dict[str, object]:
    return {
        "page_count": len(features),
        "mean_prefilter_primary_score": mean_or_none([float(x["prefilter_primary_score"]) for x in features]),
        "mean_balance_score": mean_or_none([float(x["balance_score"]) for x in features]),
        "mean_visual_page_score": mean_or_none([float(x["visual_page_score"]) for x in features]),
        "mean_grounded_non_visual_page_score": mean_or_none([float(x["grounded_non_visual_page_score"]) for x in features]),
        "mean_visual_alignment_ratio": mean_or_none([float(x["visual_alignment_ratio"]) for x in features]),
        "mean_non_visual_alignment_ratio": mean_or_none([float(x["non_visual_alignment_ratio"]) for x in features]),
        "mean_visual_anchor_patch_count": mean_or_none([float(x["visual_anchor_patch_count"]) for x in features]),
        "mean_grounded_non_visual_patch_count": mean_or_none([float(x["grounded_non_visual_patch_count"]) for x in features]),
    }


def prefilter_primary_score(feature, mode: str) -> float:
    if mode == "balance_then_visual":
        return (
            float(feature.balance_score)
            if float(feature.balance_score) > 0.0
            else float(feature.visual_page_score)
        )
    if mode == "balance_only":
        return float(feature.balance_score)
    if mode == "visual_only":
        return float(feature.visual_page_score)
    if mode == "non_visual_only":
        return float(feature.non_visual_page_score)
    if mode == "grounded_non_visual_only":
        return float(feature.grounded_non_visual_page_score)
    raise ValueError(f"Unsupported prefilter sort mode: {mode!r}")


def prefilter_sort_key(feature, mode: str) -> tuple[float, float, float, float, str]:
    primary = prefilter_primary_score(feature, mode)
    return (
        primary,
        float(feature.balance_score),
        float(feature.visual_page_score),
        float(feature.grounded_non_visual_page_score),
        feature.page_uid,
    )


def build_reliability_checklist(
    *,
    route_features: dict,
    best_gold_visual: dict | None,
    baseline_first_gold_rank: int | None,
    visual_prefilter_top_pages: int,
    visual_topn_gold: list[dict],
    visual_topn_non_gold: list[dict],
) -> dict[str, object]:
    informative_visual_tokens = list(route_features.get("informative_visual_query_tokens", []))
    generic_tokens = [
        token for token in informative_visual_tokens
        if str(token).strip().lower() in GENERIC_VISUAL_QUERY_TOKENS
    ]
    specific_tokens = [
        token for token in informative_visual_tokens
        if str(token).strip().lower() not in GENERIC_VISUAL_QUERY_TOKENS
    ]

    gold_visual_rank = None if best_gold_visual is None else int(best_gold_visual["visual_rank"])
    gold_rank_gain = (
        None
        if best_gold_visual is None or baseline_first_gold_rank is None
        else int(baseline_first_gold_rank) - int(best_gold_visual["visual_rank"])
    )
    gold_reaches_topn = bool(best_gold_visual is not None and int(best_gold_visual["visual_rank"]) <= visual_prefilter_top_pages)
    gold_reaches_top2n = bool(best_gold_visual is not None and int(best_gold_visual["visual_rank"]) <= 2 * visual_prefilter_top_pages)
    grounded_support_active_for_gold = bool(
        best_gold_visual is not None
        and float(best_gold_visual["grounded_non_visual_page_score"]) > 0.0
        and float(best_gold_visual["balance_score"]) > 0.0
    )

    topn_non_gold_mean_primary = mean_or_none([float(row["prefilter_primary_score"]) for row in visual_topn_non_gold])
    topn_non_gold_max_primary = (
        None
        if not visual_topn_non_gold
        else float(max(float(row["prefilter_primary_score"]) for row in visual_topn_non_gold))
    )
    topn_cutoff_primary = (
        None
        if not visual_topn_non_gold and not visual_topn_gold
        else float(
            min(
                float(row["prefilter_primary_score"])
                for row in (visual_topn_gold + visual_topn_non_gold)
            )
        )
    )
    gold_primary_score = None if best_gold_visual is None else float(best_gold_visual["prefilter_primary_score"])
    gold_beats_topn_cutoff = bool(
        gold_primary_score is not None
        and topn_cutoff_primary is not None
        and gold_primary_score >= topn_cutoff_primary
    )

    if gold_reaches_topn:
        verdict = "strong"
    elif gold_reaches_top2n and grounded_support_active_for_gold:
        verdict = "borderline"
    elif gold_rank_gain is not None and gold_rank_gain >= 100:
        verdict = "useful_but_not_reliable"
    else:
        verdict = "weak"

    return {
        "informative_visual_token_count": len(informative_visual_tokens),
        "informative_visual_tokens": informative_visual_tokens,
        "generic_informative_visual_tokens": generic_tokens,
        "specific_informative_visual_tokens": specific_tokens,
        "generic_token_fraction": (
            float(len(generic_tokens) / len(informative_visual_tokens))
            if informative_visual_tokens
            else None
        ),
        "baseline_first_gold_rank": baseline_first_gold_rank,
        "gold_visual_rank": gold_visual_rank,
        "gold_rank_gain": gold_rank_gain,
        "gold_reaches_topn": gold_reaches_topn,
        "gold_reaches_top2n": gold_reaches_top2n,
        "topn_gold_page_count": len(visual_topn_gold),
        "grounded_support_active_for_gold": grounded_support_active_for_gold,
        "gold_primary_score": gold_primary_score,
        "topn_non_gold_mean_primary_score": topn_non_gold_mean_primary,
        "topn_non_gold_max_primary_score": topn_non_gold_max_primary,
        "topn_cutoff_primary_score": topn_cutoff_primary,
        "gold_beats_topn_cutoff": gold_beats_topn_cutoff,
        "verdict": verdict,
    }


def main() -> None:
    args = parse_args()

    qids = list(args.qid)
    if args.qid_jsonl:
        qids.extend(load_qids(Path(args.qid_jsonl), args.qid_field, args.max_qids))
    qids = list(dict.fromkeys(qids))
    if args.max_qids > 0:
        qids = qids[: args.max_qids]
    if not qids:
        raise ValueError("Provide at least one qid via --qid or --qid-jsonl.")

    gold_rows = load_gold_rows(Path(args.gold), set(qids))
    baseline_payload = load_baseline_payload(Path(args.baseline_pred), set(qids))

    retrieval_model = ColPaliRetrievalModel(
        backbone_name_or_path=resolve_model_path(args.retrieval_model_name_or_path),
        adapter_name_or_path=resolve_model_path(args.retrieval_adapter_model_name_or_path),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audits: list[dict] = []

    for idx, qid in enumerate(qids, start=1):
        gold_row = gold_rows[qid]
        gold_doc_ids = sorted({str(item["doc_id"]).strip() for item in gold_row.get("supporting_context", [])})
        gold_doc_id_set = set(gold_doc_ids)

        baseline_rows = baseline_payload[qid].get("page_retrieval_results", [])
        candidate_doc_ids, baseline_page_uids, baseline_doc_rank_map, _ignored, baseline_page_score_map = build_baseline_pool(
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
            )

        baseline_rank_map = {
            page_uid: rank
            for rank, page_uid in enumerate(baseline_page_uids, start=1)
        }
        sorted_visual_features = sorted(
            visual_features,
            key=lambda item: prefilter_sort_key(item, args.prefilter_sort_key),
            reverse=True,
        )

        visual_trace = []
        gold_pages_all = []
        non_gold_pages_all = []
        for rank, feature in enumerate(sorted_visual_features, start=1):
            primary = prefilter_primary_score(feature, args.prefilter_sort_key)
            record = {
                "visual_rank": int(rank),
                "baseline_rank": int(baseline_rank_map.get(feature.page_uid, -1)),
                "page_uid": feature.page_uid,
                "doc_id": feature.doc_id,
                "page_idx": int(feature.page_idx),
                "is_gold_doc": feature.doc_id in gold_doc_id_set,
                "prefilter_primary_score": primary,
                "base_page_score": float(feature.base_page_score),
                "visual_page_score": float(feature.visual_page_score),
                "grounded_non_visual_page_score": float(feature.grounded_non_visual_page_score),
                "non_visual_page_score": float(feature.non_visual_page_score),
                "balance_score": float(feature.balance_score),
                "visual_alignment_ratio": float(feature.visual_alignment_ratio),
                "non_visual_alignment_ratio": float(feature.non_visual_alignment_ratio),
                "visual_query_token_count": int(feature.visual_query_token_count),
                "non_visual_query_token_count": int(feature.non_visual_query_token_count),
                "visual_patch_count": int(feature.visual_patch_count),
                "grounded_non_visual_patch_count": int(feature.grounded_non_visual_patch_count),
                "visual_anchor_patch_count": int(feature.visual_anchor_patch_count),
            }
            visual_trace.append(record)
            if record["is_gold_doc"]:
                gold_pages_all.append(record)
            else:
                non_gold_pages_all.append(record)

        visual_topn = visual_trace[: args.visual_prefilter_top_pages]
        visual_topn_gold = [row for row in visual_topn if row["is_gold_doc"]]
        visual_topn_non_gold = [row for row in visual_topn if not row["is_gold_doc"]]

        best_gold_visual = next((row for row in visual_trace if row["is_gold_doc"]), None)
        best_gold_baseline = next((page_uid for page_uid in baseline_page_uids if page_uid.rsplit("_page", 1)[0] in gold_doc_id_set), None)
        baseline_first_gold_rank = (
            None if best_gold_baseline is None else int(baseline_rank_map[best_gold_baseline])
        )
        reliability_checklist = build_reliability_checklist(
            route_features=route_features,
            best_gold_visual=best_gold_visual,
            baseline_first_gold_rank=baseline_first_gold_rank,
            visual_prefilter_top_pages=args.visual_prefilter_top_pages,
            visual_topn_gold=visual_topn_gold,
            visual_topn_non_gold=visual_topn_non_gold,
        )

        audit = {
            "qid": qid,
            "question": gold_row["question"],
            "gold_doc_ids": gold_doc_ids,
            "route_features": route_features,
            "reliability_checklist": reliability_checklist,
            "prefilter_sort_key": args.prefilter_sort_key,
            "visual_prefilter_top_pages": args.visual_prefilter_top_pages,
            "baseline_first_gold_page": None if best_gold_baseline is None else {
                "page_uid": best_gold_baseline,
                "baseline_rank": baseline_first_gold_rank,
            },
            "visual_first_gold_page": None if best_gold_visual is None else {
                "visual_rank": int(best_gold_visual["visual_rank"]),
                "baseline_rank": int(best_gold_visual["baseline_rank"]),
                "page_uid": best_gold_visual["page_uid"],
                "doc_id": best_gold_visual["doc_id"],
                "prefilter_primary_score": float(best_gold_visual["prefilter_primary_score"]),
                "visual_page_score": float(best_gold_visual["visual_page_score"]),
                "grounded_non_visual_page_score": float(best_gold_visual["grounded_non_visual_page_score"]),
                "balance_score": float(best_gold_visual["balance_score"]),
            },
            "top_visual_prefilter_pages": visual_topn[: args.report_topn],
            "top_visual_prefilter_gold_pages": visual_topn_gold[: args.report_topn],
            "top_visual_prefilter_non_gold_pages": visual_topn_non_gold[: args.report_topn],
            "all_gold_pages_summary": summarize_group(gold_pages_all),
            "all_non_gold_pages_summary": summarize_group(non_gold_pages_all),
            "top_visual_prefilter_gold_pages_summary": summarize_group(visual_topn_gold),
            "top_visual_prefilter_non_gold_pages_summary": summarize_group(visual_topn_non_gold),
        }
        audits.append(audit)

        print(
            f"[{idx}/{len(qids)}] {qid} "
            f"mode={args.prefilter_sort_key} "
            f"informative_visual={route_features['informative_visual_query_tokens']} "
            f"best_gold_visual_rank={None if best_gold_visual is None else best_gold_visual['visual_rank']} "
            f"verdict={reliability_checklist['verdict']}"
        )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"audits": audits}, indent=2) + "\n", encoding="utf-8")
    print(f"saved_json: {output_path}")


if __name__ == "__main__":
    main()
