#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
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

from scripts.rerank_target_docs_visual_aware import (
    APPROX_BASE_PAGE_TOKEN_SCORER_CHOICES,
    APPROX_BASE_PAGE_TOKEN_SELECTOR_CHOICES,
    BALANCE_SCORE_MODE_CHOICES,
    BASE_SCORE_SOURCE_CHOICES,
    COARSE_SCORE_DTYPE_CHOICES,
    QUERY_TOKEN_FILTER_CHOICES,
    WeightConfig,
    apply_two_stage_exact_rerank_to_doc_features,
    apply_visual_rerank_to_top_pages,
    apply_visual_rerank_to_top_docs,
    apply_two_stage_exact_rerank_to_page_features,
    axis_class_counts,
    build_page_id_metadata,
    build_page_token_classes,
    build_rankings,
    build_stage1_base_doc_rank_map,
    clean_token_label,
    compute_base_only_page_features,
    compute_base_only_page_feature,
    build_doc_feature_records,
    build_learned_doc_rankings,
    compute_page_feature,
    decide_query_route,
    enrich_page_features_with_channels,
    extract_query_route_features,
    grid_search_weights,
    is_base_only_weights,
    load_learned_doc_reranker,
    load_patch_axis_classes_for_pages,
    load_query_route_config,
    load_splice_query_axis_classes,
    make_base_only_page_feature,
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
    parser.add_argument("--data-name", default="m3-docvqa")
    parser.add_argument("--split", default="dev")
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
        "--base-score-source",
        default="approx_page_maxsim_topk",
        choices=BASE_SCORE_SOURCE_CHOICES,
        help=(
            "Source used for the base term in the fusion. "
            "'exact_page_maxsim' recomputes page-local MaxSim on the fixed pool; "
            "'baseline_pred' reuses the page score stored in --baseline-pred. "
            "'approx_page_maxsim_topk' uses query-guided top-K page-token pruning before "
            "MaxSim in base-only mode. "
            "'two_stage_page_maxsim' uses approximate top-K pruning on all pages, then "
            "recomputes exact MaxSim only on the top-N stage-1 pages. "
            "'two_stage_doc_maxsim' uses approximate top-K pruning on all pages, then "
            "recomputes exact MaxSim on all pages inside the top-N stage-1 docs."
        ),
    )
    parser.add_argument(
        "--approx-base-page-token-topk",
        type=int,
        default=256,
        help=(
            "Top-K page tokens kept for query-guided pruning when "
            "--base-score-source=approx_page_maxsim_topk in base-only mode."
        ),
    )
    parser.add_argument(
        "--approx-base-page-token-scorer",
        default="query_mean",
        choices=APPROX_BASE_PAGE_TOKEN_SCORER_CHOICES,
        help=(
            "Coarse scorer used to select page tokens before top-K pruning in "
            "approx_page_maxsim_topk mode. 'query_mean' is the original fast mean-query scorer; "
            "'query_token_max' uses per-page-token max similarity over query tokens and is usually "
            "stronger but less efficient; 'query_prototype_max' clusters query tokens into a small "
            "set of learned prototypes and scores each page token by its best prototype match."
        ),
    )
    parser.add_argument(
        "--approx-base-page-token-query-prototypes",
        type=int,
        default=4,
        help=(
            "When --approx-base-page-token-scorer=query_prototype_max, build this many query "
            "prototypes before coarse top-K page-token pruning."
        ),
    )
    parser.add_argument(
        "--approx-base-page-token-selector",
        default="global_topk",
        choices=APPROX_BASE_PAGE_TOKEN_SELECTOR_CHOICES,
        help=(
            "Token-selection strategy used before top-K approximate MaxSim pruning. "
            "'global_topk' is the current global-only selection. "
            "'spatial_quadrant_mix' reserves part of the token budget across page quadrants. "
            "'query_coverage_mix' reserves part of the token budget for tokens that cover more "
            "distinct informative query tokens before filling the rest globally. "
            "'soft_label_prior' keeps global top-K selection but adds soft bonuses from "
            "informative visual query tokens and visual-labeled page patches. "
            "'visual_patch_query_prior' only boosts visual-labeled page patches using "
            "informative visual-query-token similarity before global top-K selection."
        ),
    )
    parser.add_argument(
        "--approx-base-page-token-spatial-reserve",
        type=int,
        default=64,
        help=(
            "When --approx-base-page-token-selector=spatial_quadrant_mix, reserve this many "
            "token slots across spatial quadrants before filling the rest globally."
        ),
    )
    parser.add_argument(
        "--approx-base-page-token-label-reserve",
        type=int,
        default=64,
        help=(
            "When --approx-base-page-token-selector=query_label_mix, reserve this many "
            "token slots for pruning against the query's visual-token subset."
        ),
    )
    parser.add_argument(
        "--approx-base-page-token-coverage-reserve",
        type=int,
        default=64,
        help=(
            "When --approx-base-page-token-selector=query_coverage_mix, reserve this many "
            "token slots for per-query-token coverage before filling the rest globally."
        ),
    )
    parser.add_argument(
        "--approx-base-page-token-soft-visual-query-weight",
        type=float,
        default=0.5,
        help=(
            "When --approx-base-page-token-selector=soft_label_prior or "
            "visual_patch_query_prior, add this weight times the informative "
            "visual-query-token alignment score to the pruning score."
        ),
    )
    parser.add_argument(
        "--approx-base-page-token-soft-patch-visual-bonus",
        type=float,
        default=0.2,
        help=(
            "When --approx-base-page-token-selector=soft_label_prior or "
            "visual_patch_query_prior, add this bonus to page tokens whose patch "
            "label is visual before top-K selection."
        ),
    )
    parser.add_argument(
        "--base-only-page-batch-size",
        type=int,
        default=0,
        help=(
            "Optional batch size for base-only page scoring. Currently accelerates exact and "
            "global-topK approximate scoring paths when page token counts match."
        ),
    )
    parser.add_argument(
        "--approx-base-page-token-coarse-dtype",
        default="fp32",
        choices=COARSE_SCORE_DTYPE_CHOICES,
        help=(
            "Numeric precision used only for the coarse pruning-score computation in approximate "
            "base-only modes. Final exact MaxSim remains fp32."
        ),
    )
    parser.add_argument(
        "--two-stage-exact-top-pages",
        type=int,
        default=0,
        help=(
            "For --base-score-source=two_stage_page_maxsim, recompute exact MaxSim only on the "
            "top-N pages from the approximate stage-1 ranking."
        ),
    )
    parser.add_argument(
        "--two-stage-exact-top-docs",
        type=int,
        default=0,
        help=(
            "For --base-score-source=two_stage_doc_maxsim, recompute exact MaxSim on all pages "
            "inside the top-N docs from the approximate stage-1 ranking."
        ),
    )
    parser.add_argument(
        "--visual-rerank-top-pages",
        type=int,
        default=0,
        help=(
            "Experimental staged mode. First compute a base-only ranking over all candidate pages, "
            "then recompute full visual-aware features only for the top-N stage-1 pages."
        ),
    )
    parser.add_argument(
        "--visual-rerank-top-docs",
        type=int,
        default=0,
        help=(
            "Experimental staged mode. First compute a base-only ranking over all candidate pages, "
            "then recompute full visual-aware features for all pages inside the top-N stage-1 docs."
        ),
    )
    parser.add_argument(
        "--visual-rerank-require-informative-visual-query",
        action="store_true",
        help=(
            "Only apply staged visual reranking when the query has at least one informative "
            "visual cue token after filtering weak words like articles and prepositions."
        ),
    )
    parser.add_argument(
        "--visual-rerank-filter-to-informative-visual-query",
        action="store_true",
        help=(
            "When staged visual reranking is active, only let informative visual query tokens "
            "contribute to the visual channel; weak visual-labeled tokens are ignored."
        ),
    )
    parser.add_argument(
        "--visual-rerank-preserve-stage1-base-score",
        action="store_true",
        help=(
            "When staged visual reranking is active, keep the stage-1 base score for shortlisted "
            "pages and recompute only the visual/non-visual/balance channels. This makes the "
            "second stage a late tie-breaker instead of a partial exact-base replacement."
        ),
    )
    parser.add_argument(
        "--query-route-config-json",
        help=(
            "Optional JSON config for binary query routing. When provided, each qid chooses "
            "between plain stage-1 base-only output and the staged visual rerank arm."
        ),
    )
    parser.add_argument(
        "--learned-doc-reranker-model",
        help=(
            "Optional learned linear doc reranker JSON. When provided, aggregate doc features "
            "from the stage-1 page features and rerank docs/pages using the learned model instead "
            "of the hand-weighted fusion."
        ),
    )
    parser.add_argument(
        "--learned-doc-reranker-top-docs",
        type=int,
        default=0,
        help=(
            "Optional shortlist size for the learned doc reranker. Use 0 to rerank all candidate "
            "docs; otherwise only rerank the top-N stage-1 base docs and leave the rest in stage-1 order."
        ),
    )
    parser.add_argument(
        "--output-doc-feature-jsonl",
        help=(
            "Optional JSONL path to export one learned-reranker doc feature record per candidate doc."
        ),
    )
    parser.add_argument(
        "--vlm-rerank-top-docs",
        type=int,
        default=0,
        help=(
            "Optional VLM late rerank shortlist size. Uses the stage-1 best page from each top-N "
            "base doc, asks a VLM whether that page contains evidence for the query, and reranks "
            "only that shortlist."
        ),
    )
    parser.add_argument(
        "--vlm-rerank-bonus",
        type=float,
        default=1.0,
        help=(
            "Add this bonus times the parsed VLM evidence score to each selected doc's stage-1 "
            "base score during VLM late reranking."
        ),
    )
    parser.add_argument(
        "--vlm-rerank-pages-per-doc",
        type=int,
        default=1,
        help=(
            "How many top stage-1 pages to evaluate per shortlisted doc during VLM late reranking. "
            "The best VLM-scored page becomes the doc's evidence page."
        ),
    )
    parser.add_argument(
        "--vlm-model-name-or-path",
        default="Qwen2-VL-7B-Instruct",
        help="Local path or model name for the VLM late reranker.",
    )
    parser.add_argument(
        "--vlm-model-type",
        default="",
        help="Optional explicit VLM type. If omitted, infer from --vlm-model-name-or-path.",
    )
    parser.add_argument(
        "--vlm-bits",
        type=int,
        default=16,
        help="Precision/quantization bits for the VLM late reranker.",
    )
    parser.add_argument(
        "--gated-visual-top-docs",
        type=int,
        default=0,
        help=(
            "Only apply non-base rerank channels (visual, non-visual, balance) to docs whose "
            "stage-1 base-only doc rank is <= this value. Use 0 to disable gating."
        ),
    )
    parser.add_argument(
        "--scale-auxiliary-by-base-score",
        action="store_true",
        help=(
            "Scale non-base rerank channels by normalized base page score "
            "(base_page_score / max_base_page_score) so auxiliary signals help more on pages "
            "that are already strong under the base retriever."
        ),
    )
    parser.add_argument(
        "--balance-score-mode",
        default="min_avg",
        choices=BALANCE_SCORE_MODE_CHOICES,
        help=(
            "How to compute the balance/conjunction channel. 'min_avg' preserves the original "
            "min(visual_avg_score, non_visual_avg_score). 'visual_x_nonvisual_avg' makes the "
            "conjunction a stronger semantic gate by multiplying the effective visual page score "
            "by the average non-visual support. 'visual_x_grounded_nonvisual_avg' only counts "
            "non-visual support from patches near the best visual anchors."
        ),
    )
    parser.add_argument(
        "--grounded-context-radius",
        type=int,
        default=0,
        help=(
            "Neighborhood radius in patch space for context-aware visual grounding. When using "
            "--balance-score-mode=visual_x_grounded_nonvisual_avg, only non_visual patches within "
            "this radius of the best visual anchor patches contribute to the grounded context score."
        ),
    )
    parser.add_argument(
        "--visual-patch-dilation-radius",
        type=int,
        default=0,
        help=(
            "Optional visual-patch smoothing radius. When > 0, visual patch labels are dilated "
            "over neighboring patches before building page token classes."
        ),
    )
    parser.add_argument(
        "--visual-patch-dilation-include-non-visual",
        action="store_true",
        help=(
            "By default visual-patch dilation only expands into unknown patches. Enable this to "
            "let the dilation overwrite non_visual patch labels too."
        ),
    )
    parser.add_argument(
        "--visual-fallback-all-token-weight",
        type=float,
        default=0.0,
        help=(
            "Optional fallback for the visual channel. When > 0, also score visual query tokens "
            "against all page tokens and use max(visual_labeled_score, weight * visual_all_token_score) "
            "as the effective visual page score."
        ),
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
    parser.add_argument("--weight-visual", type=float, default=0.0)
    parser.add_argument("--weight-non-visual", type=float, default=0.0)
    parser.add_argument("--weight-balance", type=float, default=0.0)
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
) -> tuple[list[str], list[str], dict[str, int], list[str], dict[str, float]]:
    baseline_page_uids: list[str] = []
    candidate_doc_ids: list[str] = []
    baseline_doc_rank_map: dict[str, int] = {}
    baseline_page_score_map: dict[str, float] = {}
    seen_docs: set[str] = set()

    for row_rank, row in enumerate(rows, start=1):
        doc_id = str(row[0]).strip()
        page_idx = int(row[1])
        page_uid = f"{doc_id}_page{page_idx}"
        baseline_page_score_map[page_uid] = float(row[2])
        if row_rank <= top_pages:
            baseline_page_uids.append(page_uid)
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                candidate_doc_ids.append(doc_id)
        if doc_id and doc_id not in baseline_doc_rank_map:
            baseline_doc_rank_map[doc_id] = len(baseline_doc_rank_map) + 1

    return (
        candidate_doc_ids,
        baseline_page_uids,
        baseline_doc_rank_map,
        [str(x.split("_page")[0]) for x in baseline_page_uids],
        baseline_page_score_map,
    )


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


def infer_vqa_model_type(model_name_or_path: str) -> str:
    lowered = model_name_or_path.lower()
    if "florence" in lowered:
        return "florence2"
    if "idefics2" in lowered:
        return "idefics2"
    if "idefics3" in lowered:
        return "idefics3"
    if "internvl2" in lowered:
        return "internvl2"
    if "qwen2" in lowered:
        return "qwen2"
    raise KeyError(f"Unknown VLM model type for {model_name_or_path}")


def make_vlm_dataset_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        data_name=args.data_name,
        split=args.split,
        data_len=None,
        use_dummy_images=False,
        load_embedding=False,
        embedding_name="",
        max_pages=20,
        do_page_padding=False,
        retrieval_model_type="colpali",
        use_retrieval=False,
        retrieval_only=False,
        page_retrieval_type="logits",
        loop_unique_doc_ids=False,
        n_retrieval_pages=0,
        faiss_index_type="ivfflat",
        model_name_or_path=args.vlm_model_name_or_path,
        retrieval_model_name_or_path="",
        retrieval_adapter_model_name_or_path="",
        bits=args.vlm_bits,
        do_image_splitting=False,
    )


def make_vlm_evidence_prompt(query: str) -> str:
    return (
        "You are checking whether a single document page contains evidence needed to answer a question.\n"
        f"Question: {query}\n"
        "Does this page contain evidence that would help answer the question correctly?\n"
        "Reply with exactly one word: yes or no."
    )


def parse_vlm_evidence_score(response_text: str) -> tuple[float, str]:
    normalized = " ".join(str(response_text or "").strip().lower().split())
    if not normalized:
        return 0.0, normalized
    if normalized.startswith("yes"):
        return 1.0, normalized
    if normalized.startswith("no"):
        return 0.0, normalized
    if any(token in normalized for token in ("maybe", "unclear", "partially", "possibly", "unsure")):
        return 0.5, normalized
    if "yes" in normalized and "no" not in normalized:
        return 0.75, normalized
    if "no" in normalized and "yes" not in normalized:
        return 0.25, normalized
    return 0.5, normalized


def build_vlm_late_reranked_results(
    *,
    page_features: list,
    baseline_doc_rank_map: dict[str, int],
    dataset,
    vqa_model,
    query_text: str,
    top_docs: int,
    bonus: float,
    pages_per_doc: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    stage1_ranked_docs, _stage1_ranked_pages = build_rankings(
        page_features=page_features,
        weights=WeightConfig(base=1.0, visual=0.0, non_visual=0.0, balance=0.0),
        baseline_doc_rank_map=baseline_doc_rank_map,
    )
    stage1_doc_rank_map = {item["doc_id"]: item["rank"] for item in stage1_ranked_docs}
    stage1_doc_payload = {item["doc_id"]: item for item in stage1_ranked_docs}
    stage1_pages_by_doc: dict[str, list] = {}
    for feature in sorted(page_features, key=lambda item: item.base_page_score, reverse=True):
        stage1_pages_by_doc.setdefault(feature.doc_id, []).append(feature)

    selected_doc_ids = [item["doc_id"] for item in stage1_ranked_docs[:top_docs]]
    selected_doc_id_set = set(selected_doc_ids)
    prompt = make_vlm_evidence_prompt(query_text)
    page_image_cache: dict[str, list] = {}
    vlm_records: list[dict] = []

    for doc_id in selected_doc_ids:
        stage1_item = stage1_doc_payload[doc_id]
        if doc_id not in page_image_cache:
            page_image_cache[doc_id] = dataset.get_images_from_doc_id(doc_id)
        page_images = page_image_cache[doc_id]
        candidate_page_features = stage1_pages_by_doc.get(doc_id, [])[:pages_per_doc]
        if not candidate_page_features:
            continue
        page_candidates: list[dict] = []
        for feature in candidate_page_features:
            page_idx = int(feature.page_idx)
            if page_idx < 0 or page_idx >= len(page_images):
                raise IndexError(
                    f"Page index out of range for VLM rerank: {doc_id}:{page_idx} not in [0, {len(page_images) - 1}]"
                )
            response_text = vqa_model.generate(images=[page_images[page_idx]], question=prompt)
            evidence_score, normalized_response = parse_vlm_evidence_score(response_text)
            page_candidates.append(
                {
                    "page_uid": feature.page_uid,
                    "page_idx": page_idx,
                    "base_page_score": float(feature.base_page_score),
                    "vlm_response": str(response_text).strip(),
                    "vlm_response_normalized": normalized_response,
                    "vlm_evidence_score": float(evidence_score),
                }
            )
        best_page = max(
            page_candidates,
            key=lambda item: (item["vlm_evidence_score"], item["base_page_score"]),
        )
        final_score = float(stage1_item["fused_doc_score"]) + float(bonus) * float(best_page["vlm_evidence_score"])
        vlm_records.append(
            {
                "doc_id": doc_id,
                "stage1_base_doc_rank": int(stage1_item["rank"]),
                "best_page_uid": best_page["page_uid"],
                "best_page_idx": best_page["page_idx"],
                "stage1_base_doc_score": float(stage1_item["fused_doc_score"]),
                "vlm_response": best_page["vlm_response"],
                "vlm_response_normalized": best_page["vlm_response_normalized"],
                "vlm_evidence_score": float(best_page["vlm_evidence_score"]),
                "vlm_final_doc_score": float(final_score),
                "vlm_pages_per_doc": int(pages_per_doc),
                "vlm_page_candidates": page_candidates,
            }
        )

    vlm_record_map = {item["doc_id"]: item for item in vlm_records}
    reranked_selected = sorted(
        vlm_records,
        key=lambda item: (item["vlm_final_doc_score"], item["stage1_base_doc_score"]),
        reverse=True,
    )
    remaining_stage1_docs = [item for item in stage1_ranked_docs if item["doc_id"] not in selected_doc_id_set]

    doc_rank_map: dict[str, int] = {}
    reranked_docs: list[dict] = []
    for rank, item in enumerate(reranked_selected, start=1):
        doc_rank_map[item["doc_id"]] = rank
        reranked_docs.append(
            {
                "doc_id": item["doc_id"],
                "rank": rank,
                "fused_doc_score": item["vlm_final_doc_score"],
                "best_page_uid": item["best_page_uid"],
                "best_page_idx": item["best_page_idx"],
                "best_page_base_score": item["stage1_base_doc_score"],
                "best_page_visual_score": None,
                "best_page_non_visual_score": None,
                "best_page_grounded_non_visual_score": None,
                "best_page_balance_score": None,
                "stage1_base_doc_rank": item["stage1_base_doc_rank"],
                "baseline_doc_rank": baseline_doc_rank_map.get(item["doc_id"]),
                "vlm_response": item["vlm_response"],
                "vlm_response_normalized": item["vlm_response_normalized"],
                "vlm_evidence_score": item["vlm_evidence_score"],
            }
        )

    start_rank = len(reranked_docs) + 1
    for offset, item in enumerate(remaining_stage1_docs, start=0):
        rank = start_rank + offset
        doc_rank_map[item["doc_id"]] = rank
        reranked_docs.append(
            {
                **item,
                "rank": rank,
                "vlm_response": None,
                "vlm_response_normalized": None,
                "vlm_evidence_score": None,
            }
        )

    reranked_pages = sorted(
        [
            {
                **asdict(feature),
                "doc_rank": doc_rank_map[feature.doc_id],
                "vlm_evidence_score": (
                    vlm_record_map[feature.doc_id]["vlm_evidence_score"]
                    if feature.doc_id in vlm_record_map
                    else None
                ),
                "vlm_selected_page": (
                    feature.doc_id in vlm_record_map
                    and feature.page_uid == vlm_record_map[feature.doc_id]["best_page_uid"]
                ),
                "fused_page_score": (
                    float(vlm_record_map[feature.doc_id]["vlm_final_doc_score"])
                    + (1e-3 if feature.doc_id in vlm_record_map and feature.page_uid == vlm_record_map[feature.doc_id]["best_page_uid"] else 0.0)
                    + 1e-6 * float(feature.base_page_score)
                    if feature.doc_id in vlm_record_map
                    else -float(doc_rank_map[feature.doc_id]) + 1e-6 * float(feature.base_page_score)
                ),
            }
            for feature in page_features
        ],
        key=lambda item: (
            -item["doc_rank"],
            item["vlm_selected_page"],
            item["base_page_score"],
            item["visual_page_score"],
            item["non_visual_page_score"],
        ),
        reverse=True,
    )
    for rank, item in enumerate(reranked_pages, start=1):
        item["rank"] = rank

    return reranked_docs, reranked_pages, vlm_records


def main() -> None:
    args = parse_args()

    if args.base_score_source == "approx_page_maxsim_topk" and args.approx_base_page_token_topk <= 0:
        raise ValueError(
            "--base-score-source=approx_page_maxsim_topk requires "
            "--approx-base-page-token-topk > 0."
        )
    if args.base_score_source == "two_stage_page_maxsim" and args.approx_base_page_token_topk <= 0:
        raise ValueError(
            "--base-score-source=two_stage_page_maxsim requires "
            "--approx-base-page-token-topk > 0."
        )
    if args.base_score_source == "two_stage_doc_maxsim" and args.approx_base_page_token_topk <= 0:
        raise ValueError(
            "--base-score-source=two_stage_doc_maxsim requires "
            "--approx-base-page-token-topk > 0."
        )
    if args.base_score_source not in {"approx_page_maxsim_topk", "two_stage_page_maxsim", "two_stage_doc_maxsim"} and args.approx_base_page_token_topk > 0:
        raise ValueError(
            "--approx-base-page-token-topk is only valid with "
            "--base-score-source=approx_page_maxsim_topk, two_stage_page_maxsim, or two_stage_doc_maxsim."
        )
    if (
        args.base_score_source not in {"approx_page_maxsim_topk", "two_stage_page_maxsim", "two_stage_doc_maxsim"}
        and args.approx_base_page_token_scorer != "query_mean"
    ):
        raise ValueError(
            "--approx-base-page-token-scorer is only valid with "
            "--base-score-source=approx_page_maxsim_topk, two_stage_page_maxsim, or two_stage_doc_maxsim."
        )
    if args.approx_base_page_token_scorer == "query_prototype_max" and args.approx_base_page_token_query_prototypes <= 0:
        raise ValueError("--approx-base-page-token-query-prototypes must be > 0.")
    if (
        args.base_score_source not in {"approx_page_maxsim_topk", "two_stage_page_maxsim", "two_stage_doc_maxsim"}
        and args.approx_base_page_token_selector != "global_topk"
    ):
        raise ValueError(
            "--approx-base-page-token-selector is only valid with "
            "--base-score-source=approx_page_maxsim_topk, two_stage_page_maxsim, or two_stage_doc_maxsim."
        )
    if (
        args.approx_base_page_token_selector != "spatial_quadrant_mix"
        and args.approx_base_page_token_spatial_reserve != 64
    ):
        raise ValueError(
            "--approx-base-page-token-spatial-reserve is only valid with "
            "--approx-base-page-token-selector=spatial_quadrant_mix."
        )
    if (
        args.approx_base_page_token_selector != "query_label_mix"
        and args.approx_base_page_token_label_reserve != 64
    ):
        raise ValueError(
            "--approx-base-page-token-label-reserve is only valid with "
            "--approx-base-page-token-selector=query_label_mix."
        )
    if (
        args.approx_base_page_token_selector != "query_coverage_mix"
        and args.approx_base_page_token_coverage_reserve != 64
    ):
        raise ValueError(
            "--approx-base-page-token-coverage-reserve is only valid with "
            "--approx-base-page-token-selector=query_coverage_mix."
        )
    if (
        args.approx_base_page_token_selector not in {"soft_label_prior", "visual_patch_query_prior"}
        and args.approx_base_page_token_soft_visual_query_weight != 0.5
    ):
        raise ValueError(
            "--approx-base-page-token-soft-visual-query-weight is only valid with "
            "--approx-base-page-token-selector=soft_label_prior or visual_patch_query_prior."
        )
    if (
        args.approx_base_page_token_selector not in {"soft_label_prior", "visual_patch_query_prior"}
        and args.approx_base_page_token_soft_patch_visual_bonus != 0.2
    ):
        raise ValueError(
            "--approx-base-page-token-soft-patch-visual-bonus is only valid with "
            "--approx-base-page-token-selector=soft_label_prior or visual_patch_query_prior."
        )
    if args.base_only_page_batch_size < 0:
        raise ValueError("--base-only-page-batch-size must be >= 0.")
    if (
        args.base_score_source not in {"approx_page_maxsim_topk", "two_stage_page_maxsim", "two_stage_doc_maxsim"}
        and args.approx_base_page_token_coarse_dtype != "fp32"
    ):
        raise ValueError(
            "--approx-base-page-token-coarse-dtype is only valid with "
            "--base-score-source=approx_page_maxsim_topk, two_stage_page_maxsim, or two_stage_doc_maxsim."
        )
    if args.base_score_source == "two_stage_page_maxsim" and args.two_stage_exact_top_pages <= 0:
        raise ValueError(
            "--base-score-source=two_stage_page_maxsim requires "
            "--two-stage-exact-top-pages > 0."
        )
    if args.base_score_source != "two_stage_page_maxsim" and args.two_stage_exact_top_pages > 0:
        raise ValueError(
            "--two-stage-exact-top-pages is only valid with "
            "--base-score-source=two_stage_page_maxsim."
        )
    if args.base_score_source == "two_stage_doc_maxsim" and args.two_stage_exact_top_docs <= 0:
        raise ValueError(
            "--base-score-source=two_stage_doc_maxsim requires "
            "--two-stage-exact-top-docs > 0."
        )
    if args.base_score_source != "two_stage_doc_maxsim" and args.two_stage_exact_top_docs > 0:
        raise ValueError(
            "--two-stage-exact-top-docs is only valid with "
            "--base-score-source=two_stage_doc_maxsim."
        )
    if args.visual_rerank_top_pages < 0:
        raise ValueError("--visual-rerank-top-pages must be >= 0.")
    if args.visual_rerank_top_docs < 0:
        raise ValueError("--visual-rerank-top-docs must be >= 0.")
    if args.visual_rerank_top_pages > 0 and args.visual_rerank_top_docs > 0:
        raise ValueError(
            "--visual-rerank-top-pages and --visual-rerank-top-docs are mutually exclusive."
        )
    if (
        args.visual_rerank_top_pages == 0
        and args.visual_rerank_top_docs == 0
        and (
            args.visual_rerank_require_informative_visual_query
            or args.visual_rerank_filter_to_informative_visual_query
            or args.visual_rerank_preserve_stage1_base_score
        )
    ):
        raise ValueError(
            "--visual-rerank-require-informative-visual-query, "
            "--visual-rerank-filter-to-informative-visual-query, and "
            "--visual-rerank-preserve-stage1-base-score require "
            "--visual-rerank-top-pages > 0 or --visual-rerank-top-docs > 0."
        )
    if args.gated_visual_top_docs < 0:
        raise ValueError("--gated-visual-top-docs must be >= 0.")
    if args.visual_patch_dilation_radius < 0:
        raise ValueError("--visual-patch-dilation-radius must be >= 0.")
    if args.grounded_context_radius < 0:
        raise ValueError("--grounded-context-radius must be >= 0.")
    if (
        args.balance_score_mode == "visual_x_grounded_nonvisual_avg"
        and args.grounded_context_radius <= 0
    ):
        raise ValueError(
            "--grounded-context-radius must be > 0 when "
            "--balance-score-mode=visual_x_grounded_nonvisual_avg."
        )
    if args.visual_fallback_all_token_weight < 0.0:
        raise ValueError("--visual-fallback-all-token-weight must be >= 0.")
    if args.learned_doc_reranker_top_docs < 0:
        raise ValueError("--learned-doc-reranker-top-docs must be >= 0.")
    if args.vlm_rerank_top_docs < 0:
        raise ValueError("--vlm-rerank-top-docs must be >= 0.")
    if args.vlm_rerank_pages_per_doc <= 0:
        raise ValueError("--vlm-rerank-pages-per-doc must be > 0.")

    fixed_weights = WeightConfig(
        base=args.weight_base,
        visual=args.weight_visual,
        non_visual=args.weight_non_visual,
        balance=args.weight_balance,
    )
    fixed_base_only = (not args.grid_search) and is_base_only_weights(fixed_weights)
    staged_visual_rerank = (
        (not args.grid_search)
        and (args.visual_rerank_top_pages > 0 or args.visual_rerank_top_docs > 0)
    )
    learned_doc_reranker_active = bool(args.learned_doc_reranker_model)
    vlm_rerank_active = args.vlm_rerank_top_docs > 0
    if args.query_route_config_json and not staged_visual_rerank:
        raise ValueError(
            "--query-route-config-json requires --visual-rerank-top-pages > 0 "
            "or --visual-rerank-top-docs > 0."
        )
    if learned_doc_reranker_active and args.grid_search:
        raise ValueError("--learned-doc-reranker-model is not supported with --grid-search.")
    if vlm_rerank_active and args.grid_search:
        raise ValueError("--vlm-rerank-top-docs is not supported with --grid-search.")
    if vlm_rerank_active and not fixed_base_only:
        raise ValueError("--vlm-rerank-top-docs currently requires base-only weights.")
    if vlm_rerank_active and staged_visual_rerank:
        raise ValueError("--vlm-rerank-top-docs cannot be combined with staged visual reranking.")
    if vlm_rerank_active and learned_doc_reranker_active:
        raise ValueError("--vlm-rerank-top-docs cannot be combined with --learned-doc-reranker-model.")
    if vlm_rerank_active and args.base_score_source == "baseline_pred":
        raise ValueError("--vlm-rerank-top-docs is not supported with --base-score-source=baseline_pred.")
    if (
        args.base_score_source in {"approx_page_maxsim_topk", "two_stage_page_maxsim", "two_stage_doc_maxsim"}
        and not (fixed_base_only or staged_visual_rerank or learned_doc_reranker_active or vlm_rerank_active)
    ):
        raise ValueError(
            f"--base-score-source={args.base_score_source} is currently only supported in base-only mode."
        )
    if args.base_only_page_batch_size > 0 and not (
        fixed_base_only or staged_visual_rerank or learned_doc_reranker_active or vlm_rerank_active
    ):
        raise ValueError("--base-only-page-batch-size is currently only supported in base-only mode.")
    if (args.visual_rerank_top_pages > 0 or args.visual_rerank_top_docs > 0) and fixed_base_only:
        raise ValueError(
            "--visual-rerank-top-pages / --visual-rerank-top-docs require non-base-only fusion weights."
        )
    if (args.visual_rerank_top_pages > 0 or args.visual_rerank_top_docs > 0) and args.grid_search:
        raise ValueError(
            "--visual-rerank-top-pages / --visual-rerank-top-docs are currently only supported "
            "with fixed weights."
        )

    qids = load_qids(Path(args.qid_jsonl), args.qid_field, args.max_qids)
    gold_rows = load_gold_rows(Path(args.gold), set(qids))
    baseline_payload = load_baseline_payload(Path(args.baseline_pred), set(qids))
    route_config = load_query_route_config(args.query_route_config_json)
    learned_doc_reranker_model = load_learned_doc_reranker(args.learned_doc_reranker_model)
    default_route_decision = "visual" if staged_visual_rerank else "base"

    torch = None
    retrieval_model = None
    load_doc_embeddings_for_doc_ids = None
    device = None
    if not (
        fixed_base_only
        and args.base_score_source == "baseline_pred"
        and not learned_doc_reranker_active
        and not args.output_doc_feature_jsonl
    ):
        import torch as _torch
        from m3docrag.retrieval import ColPaliRetrievalModel
        from scripts.rerank_target_docs_visual_aware import load_doc_embeddings_for_doc_ids as _load_doc_embeddings_for_doc_ids

        torch = _torch
        load_doc_embeddings_for_doc_ids = _load_doc_embeddings_for_doc_ids
        retrieval_model = ColPaliRetrievalModel(
            backbone_name_or_path=resolve_model_path(args.retrieval_model_name_or_path),
            adapter_name_or_path=resolve_model_path(args.retrieval_adapter_model_name_or_path),
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vlm_dataset = None
    vlm_model = None
    if vlm_rerank_active:
        from accelerate import Accelerator

        from m3docrag.datasets.m3_docvqa import M3DocVQADataset
        from m3docrag.utils.distributed import supports_flash_attention
        from m3docrag.vqa import VQAModel

        vlm_model_path = resolve_model_path(args.vlm_model_name_or_path)
        use_flash_attn = torch.cuda.is_available() and supports_flash_attention()
        vlm_model = VQAModel(
            model_name_or_path=vlm_model_path,
            model_type=args.vlm_model_type or infer_vqa_model_type(args.vlm_model_name_or_path),
            bits=args.vlm_bits,
            use_flash_attn=use_flash_attn,
            attn_implementation="flash_attention_2" if use_flash_attn else "eager",
        )
        accelerator = Accelerator()
        if hasattr(vlm_model.model, "parameters"):
            vlm_model.model = accelerator.prepare(vlm_model.model)
        vlm_dataset = M3DocVQADataset(make_vlm_dataset_args(args))

    grid_base_values = parse_float_list(args.grid_base_values)
    grid_visual_values = parse_float_list(args.grid_visual_values)
    grid_non_visual_values = parse_float_list(args.grid_non_visual_values)
    grid_balance_values = parse_float_list(args.grid_balance_values)

    all_rows: list[dict] = []
    all_doc_feature_rows: list[dict] = []
    improved_doc = 0
    worsened_doc = 0
    unchanged_doc = 0
    reranked_top4_doc = 0
    baseline_top4_doc = 0
    routed_to_visual_qids = 0
    routed_to_base_qids = 0

    for idx, qid in enumerate(qids, start=1):
        gold_row = gold_rows[qid]
        gold_doc_ids = sorted({str(item["doc_id"]).strip() for item in gold_row.get("supporting_context", [])})
        gold_doc_id_set = set(gold_doc_ids)
        question_type = str(gold_row.get("metadata", {}).get("type", "UNKNOWN")).strip() or "UNKNOWN"

        baseline_rows = baseline_payload[qid].get("page_retrieval_results", [])
        candidate_doc_ids, baseline_page_uids, baseline_doc_rank_map, _, baseline_page_score_map = build_baseline_pool(
            baseline_rows,
            args.from_baseline_top_pages,
        )
        explicit_page_uids = set(baseline_page_uids)

        query_text = gold_row["question"]
        stage1_base_doc_rank_map: dict[str, int] | None = None
        route_features: dict = {
            "question_type": question_type,
            "visual_query_token_count": 0,
            "non_visual_query_token_count": 0,
            "informative_visual_query_count": 0,
            "informative_visual_query_tokens": [],
        }
        route_info = (
            decide_query_route(route_config=route_config, route_features=route_features)
            if route_config is not None
            else {"route_decision": default_route_decision, "matched_rule_index": None, "matched_rule": None}
        )
        if (
            fixed_base_only
            and args.base_score_source == "baseline_pred"
            and not learned_doc_reranker_active
            and not args.output_doc_feature_jsonl
        ):
            query_axis_classes = []
            page_features = []
            for page_uid in baseline_page_uids:
                doc_id, page_suffix = page_uid.rsplit("_page", 1)
                page_idx = int(page_suffix)
                page_features.append(
                    make_base_only_page_feature(
                        doc_id=doc_id,
                        page_idx=page_idx,
                        base_page_score=baseline_page_score_map[page_uid],
                    )
                )
            route_features = extract_query_route_features(
                question_type=question_type,
                query_axis_classes=query_axis_classes,
                query_token_labels=[],
            )
            route_info = (
                decide_query_route(route_config=route_config, route_features=route_features)
                if route_config is not None
                else {"route_decision": default_route_decision, "matched_rule_index": None, "matched_rule": None}
            )
        else:
            docid2embs = load_doc_embeddings_for_doc_ids(candidate_doc_ids, args.embedding_name)
            page_specs, page_meta = build_page_id_metadata(
                docid2embs=docid2embs,
                explicit_page_uids=explicit_page_uids,
                nonspatial_token_position=args.nonspatial_token_position,
            )

            query_meta = retrieval_model.encode_query_with_metadata(
                query=query_text,
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

            if fixed_base_only or staged_visual_rerank:
                needs_query_axis_classes = (
                    staged_visual_rerank
                    or learned_doc_reranker_active
                    or bool(args.output_doc_feature_jsonl)
                    or args.approx_base_page_token_selector in {
                    "query_label_mix",
                    "soft_label_prior",
                    "visual_patch_query_prior",
                    }
                )
                needs_patch_axis_classes = (
                    staged_visual_rerank
                    or learned_doc_reranker_active
                    or bool(args.output_doc_feature_jsonl)
                    or args.approx_base_page_token_selector
                    in {"soft_label_prior", "visual_patch_query_prior"}
                )
                patch_axis_classes_by_uid = (
                    load_patch_axis_classes_for_pages(
                        labels_jsonl=args.splice_patch_labels_jsonl,
                        page_meta=page_meta,
                    )
                    if needs_patch_axis_classes
                    else None
                )
                if needs_query_axis_classes:
                    query_axis_classes = load_splice_query_axis_classes(
                        query_labels_path=args.splice_query_token_labels,
                        qid=qid,
                        query_token_labels=query_token_labels,
                        query_raw_tokens=query_raw_tokens,
                    )
                else:
                    query_axis_classes = []
                route_features = extract_query_route_features(
                    question_type=question_type,
                    query_axis_classes=query_axis_classes,
                    query_token_labels=query_token_labels,
                )
                route_info = (
                    decide_query_route(route_config=route_config, route_features=route_features)
                    if route_config is not None
                    else {"route_decision": default_route_decision, "matched_rule_index": None, "matched_rule": None}
                )
            else:
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
                route_features = extract_query_route_features(
                    question_type=question_type,
                    query_axis_classes=query_axis_classes,
                    query_token_labels=query_token_labels,
                )
                route_info = (
                    decide_query_route(route_config=route_config, route_features=route_features)
                    if route_config is not None
                    else {"route_decision": default_route_decision, "matched_rule_index": None, "matched_rule": None}
                )

            page_features = []
            with torch.no_grad():
                page_token_classes_by_uid = (
                    None
                    if patch_axis_classes_by_uid is None
                    else {
                        page_uid: build_page_token_classes(
                            page_meta=page_meta[page_uid],
                            patch_axis_classes=patch_axis_classes,
                            visual_patch_dilation_radius=args.visual_patch_dilation_radius,
                            visual_patch_dilation_include_non_visual=args.visual_patch_dilation_include_non_visual,
                        )
                        for page_uid, patch_axis_classes in patch_axis_classes_by_uid.items()
                    }
                )
                if fixed_base_only:
                    page_features = compute_base_only_page_features(
                        page_specs=page_specs,
                        docid2embs=docid2embs,
                        query_emb=query_emb,
                        query_score_mask=query_score_mask,
                        base_score_source=args.base_score_source,
                        baseline_page_score_map=baseline_page_score_map,
                        approx_page_token_topk=(
                            args.approx_base_page_token_topk
                            if args.base_score_source in {"approx_page_maxsim_topk", "two_stage_page_maxsim", "two_stage_doc_maxsim"}
                            else 0
                        ),
                        approx_page_token_scorer=args.approx_base_page_token_scorer,
                        approx_page_token_query_prototypes=args.approx_base_page_token_query_prototypes,
                        approx_page_token_selector=args.approx_base_page_token_selector,
                        approx_page_token_spatial_reserve=args.approx_base_page_token_spatial_reserve,
                        query_axis_classes=query_axis_classes,
                        query_token_labels=query_token_labels,
                        page_token_classes_by_uid=page_token_classes_by_uid,
                        approx_page_token_coverage_reserve=args.approx_base_page_token_coverage_reserve,
                        approx_page_token_label_reserve=args.approx_base_page_token_label_reserve,
                        approx_page_token_soft_visual_query_weight=args.approx_base_page_token_soft_visual_query_weight,
                        approx_page_token_soft_patch_visual_bonus=args.approx_base_page_token_soft_patch_visual_bonus,
                        coarse_score_dtype=args.approx_base_page_token_coarse_dtype,
                        page_batch_size=args.base_only_page_batch_size,
                    )
                elif staged_visual_rerank:
                    page_features = compute_base_only_page_features(
                        page_specs=page_specs,
                        docid2embs=docid2embs,
                        query_emb=query_emb,
                        query_score_mask=query_score_mask,
                        base_score_source=args.base_score_source,
                        baseline_page_score_map=baseline_page_score_map,
                        approx_page_token_topk=(
                            args.approx_base_page_token_topk
                            if args.base_score_source in {"approx_page_maxsim_topk", "two_stage_page_maxsim", "two_stage_doc_maxsim"}
                            else 0
                        ),
                        approx_page_token_scorer=args.approx_base_page_token_scorer,
                        approx_page_token_query_prototypes=args.approx_base_page_token_query_prototypes,
                        approx_page_token_selector=args.approx_base_page_token_selector,
                        approx_page_token_spatial_reserve=args.approx_base_page_token_spatial_reserve,
                        query_axis_classes=query_axis_classes,
                        query_token_labels=query_token_labels,
                        page_token_classes_by_uid=page_token_classes_by_uid,
                        approx_page_token_coverage_reserve=args.approx_base_page_token_coverage_reserve,
                        approx_page_token_label_reserve=args.approx_base_page_token_label_reserve,
                        approx_page_token_soft_visual_query_weight=args.approx_base_page_token_soft_visual_query_weight,
                        approx_page_token_soft_patch_visual_bonus=args.approx_base_page_token_soft_patch_visual_bonus,
                        coarse_score_dtype=args.approx_base_page_token_coarse_dtype,
                        page_batch_size=args.base_only_page_batch_size,
                    )
                else:
                    for doc_id, page_idx in page_specs:
                        page_uid = f"{doc_id}_page{page_idx}"
                        page_emb = docid2embs[doc_id][page_idx].view(-1, docid2embs[doc_id][page_idx].shape[-1]).to(
                            device=device,
                            dtype=torch.float32,
                        )
                        assert page_token_classes_by_uid is not None
                        page_token_classes = page_token_classes_by_uid[page_uid]
                        page_features.append(
                            compute_page_feature(
                                page_emb=page_emb,
                                query_emb=query_emb,
                                query_axis_classes=query_axis_classes,
                                query_score_mask=query_score_mask,
                                page_token_classes=page_token_classes,
                                page_meta=page_meta[page_uid],
                                doc_id=doc_id,
                                page_idx=page_idx,
                                balance_score_mode=args.balance_score_mode,
                                grounded_context_radius=args.grounded_context_radius,
                                visual_fallback_all_token_weight=args.visual_fallback_all_token_weight,
                                base_score_override=(
                                    baseline_page_score_map.get(page_uid)
                                    if args.base_score_source == "baseline_pred"
                                    else None
                                ),
                            )
                        )

            if args.base_score_source == "two_stage_page_maxsim":
                page_features = apply_two_stage_exact_rerank_to_page_features(
                    page_features=page_features,
                    docid2embs=docid2embs,
                    query_emb=query_emb,
                    query_score_mask=query_score_mask,
                    top_pages=args.two_stage_exact_top_pages,
                )
            if args.base_score_source == "two_stage_doc_maxsim":
                page_features = apply_two_stage_exact_rerank_to_doc_features(
                    page_features=page_features,
                    docid2embs=docid2embs,
                    query_emb=query_emb,
                    query_score_mask=query_score_mask,
                    top_docs=args.two_stage_exact_top_docs,
                )
            if learned_doc_reranker_active or args.output_doc_feature_jsonl:
                assert page_token_classes_by_uid is not None
                page_features = enrich_page_features_with_channels(
                    page_features=page_features,
                    docid2embs=docid2embs,
                    query_emb=query_emb,
                    query_axis_classes=query_axis_classes,
                    query_score_mask=query_score_mask,
                    page_token_classes_by_uid=page_token_classes_by_uid,
                    page_meta_by_uid=page_meta,
                    balance_score_mode=args.balance_score_mode,
                    grounded_context_radius=args.grounded_context_radius,
                    visual_fallback_all_token_weight=args.visual_fallback_all_token_weight,
                )
            apply_staged_visual_rerank = staged_visual_rerank and route_info["route_decision"] == "visual"
            if route_config is not None:
                if route_info["route_decision"] == "visual":
                    routed_to_visual_qids += 1
                elif route_info["route_decision"] == "base":
                    routed_to_base_qids += 1

            if args.gated_visual_top_docs > 0 and apply_staged_visual_rerank:
                stage1_base_doc_rank_map = build_stage1_base_doc_rank_map(page_features)
            if apply_staged_visual_rerank:
                assert page_token_classes_by_uid is not None
                if args.visual_rerank_top_pages > 0:
                    page_features = apply_visual_rerank_to_top_pages(
                        page_features=page_features,
                        docid2embs=docid2embs,
                        query_emb=query_emb,
                        query_axis_classes=query_axis_classes,
                        query_token_labels=query_token_labels,
                        query_score_mask=query_score_mask,
                        page_token_classes_by_uid=page_token_classes_by_uid,
                        page_meta_by_uid=page_meta,
                        top_pages=args.visual_rerank_top_pages,
                        require_informative_visual_query=args.visual_rerank_require_informative_visual_query,
                        filter_to_informative_visual_query=args.visual_rerank_filter_to_informative_visual_query,
                        preserve_stage1_base_score=args.visual_rerank_preserve_stage1_base_score,
                        balance_score_mode=args.balance_score_mode,
                        grounded_context_radius=args.grounded_context_radius,
                        visual_fallback_all_token_weight=args.visual_fallback_all_token_weight,
                    )
                else:
                    page_features = apply_visual_rerank_to_top_docs(
                        page_features=page_features,
                        docid2embs=docid2embs,
                        query_emb=query_emb,
                        query_axis_classes=query_axis_classes,
                        query_token_labels=query_token_labels,
                        query_score_mask=query_score_mask,
                        page_token_classes_by_uid=page_token_classes_by_uid,
                        page_meta_by_uid=page_meta,
                        top_docs=args.visual_rerank_top_docs,
                        require_informative_visual_query=args.visual_rerank_require_informative_visual_query,
                        filter_to_informative_visual_query=args.visual_rerank_filter_to_informative_visual_query,
                        preserve_stage1_base_score=args.visual_rerank_preserve_stage1_base_score,
                        balance_score_mode=args.balance_score_mode,
                        grounded_context_radius=args.grounded_context_radius,
                        visual_fallback_all_token_weight=args.visual_fallback_all_token_weight,
                    )
            if args.gated_visual_top_docs > 0 and apply_staged_visual_rerank and stage1_base_doc_rank_map is None:
                stage1_base_doc_rank_map = build_stage1_base_doc_rank_map(page_features)
        if args.grid_search:
            weights, best_grid_record, _grid_leaderboard = grid_search_weights(
                page_features=page_features,
                baseline_doc_rank_map=baseline_doc_rank_map,
                stage1_base_doc_rank_map=stage1_base_doc_rank_map,
                gated_visual_top_docs=args.gated_visual_top_docs,
                scale_auxiliary_by_base_score=args.scale_auxiliary_by_base_score,
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

        doc_feature_records: list[dict] = []
        if learned_doc_reranker_active or args.output_doc_feature_jsonl:
            doc_feature_records = build_doc_feature_records(
                page_features=page_features,
                baseline_doc_rank_map=baseline_doc_rank_map,
            )
            for record in doc_feature_records:
                export_row = dict(record)
                export_row["qid"] = qid
                export_row["question_type"] = question_type
                export_row["gold_doc_ids"] = gold_doc_ids
                export_row["label_is_gold"] = record["doc_id"] in gold_doc_id_set
                all_doc_feature_rows.append(export_row)

        vlm_records: list[dict] = []
        if vlm_rerank_active:
            assert vlm_dataset is not None
            assert vlm_model is not None
            reranked_docs, reranked_pages, vlm_records = build_vlm_late_reranked_results(
                page_features=page_features,
                baseline_doc_rank_map=baseline_doc_rank_map,
                dataset=vlm_dataset,
                vqa_model=vlm_model,
                query_text=query_text,
                top_docs=args.vlm_rerank_top_docs,
                bonus=args.vlm_rerank_bonus,
                pages_per_doc=args.vlm_rerank_pages_per_doc,
            )
        elif learned_doc_reranker_active:
            reranked_docs, reranked_pages, _doc_records = build_learned_doc_rankings(
                page_features=page_features,
                baseline_doc_rank_map=baseline_doc_rank_map,
                model_payload=learned_doc_reranker_model,
                learned_top_docs=args.learned_doc_reranker_top_docs,
            )
        else:
            reranked_docs, reranked_pages = build_rankings(
                page_features=page_features,
                weights=weights,
                baseline_doc_rank_map=baseline_doc_rank_map,
                stage1_base_doc_rank_map=stage1_base_doc_rank_map,
                gated_visual_top_docs=args.gated_visual_top_docs,
                scale_auxiliary_by_base_score=args.scale_auxiliary_by_base_score,
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
            "question_type": question_type,
            "answers": [str(item["answer"]) for item in gold_row.get("answers", [])],
            "gold_doc_ids": gold_doc_ids,
            "candidate_doc_count": len(candidate_doc_ids),
            "candidate_page_count": len(page_features),
            "query_axis_class_counts": axis_class_counts(query_axis_classes),
            "base_score_source": args.base_score_source,
            "approx_base_page_token_topk": args.approx_base_page_token_topk,
            "approx_base_page_token_scorer": args.approx_base_page_token_scorer,
            "approx_base_page_token_query_prototypes": args.approx_base_page_token_query_prototypes,
            "approx_base_page_token_selector": args.approx_base_page_token_selector,
            "approx_base_page_token_spatial_reserve": args.approx_base_page_token_spatial_reserve,
            "approx_base_page_token_coverage_reserve": args.approx_base_page_token_coverage_reserve,
            "approx_base_page_token_label_reserve": args.approx_base_page_token_label_reserve,
            "approx_base_page_token_soft_visual_query_weight": args.approx_base_page_token_soft_visual_query_weight,
            "approx_base_page_token_soft_patch_visual_bonus": args.approx_base_page_token_soft_patch_visual_bonus,
            "base_only_page_batch_size": args.base_only_page_batch_size,
            "approx_base_page_token_coarse_dtype": args.approx_base_page_token_coarse_dtype,
            "two_stage_exact_top_pages": args.two_stage_exact_top_pages,
            "two_stage_exact_top_docs": args.two_stage_exact_top_docs,
            "visual_rerank_top_pages": args.visual_rerank_top_pages,
            "visual_rerank_top_docs": args.visual_rerank_top_docs,
            "query_route_config_json": args.query_route_config_json,
            "learned_doc_reranker_model": args.learned_doc_reranker_model,
            "learned_doc_reranker_top_docs": args.learned_doc_reranker_top_docs,
            "vlm_rerank_top_docs": args.vlm_rerank_top_docs,
            "vlm_rerank_bonus": args.vlm_rerank_bonus,
            "vlm_rerank_pages_per_doc": args.vlm_rerank_pages_per_doc,
            "vlm_model_name_or_path": args.vlm_model_name_or_path,
            "vlm_model_type": args.vlm_model_type or None,
            "vlm_bits": args.vlm_bits,
            "gated_visual_top_docs": args.gated_visual_top_docs,
            "scale_auxiliary_by_base_score": args.scale_auxiliary_by_base_score,
            "balance_score_mode": args.balance_score_mode,
            "grounded_context_radius": args.grounded_context_radius,
            "visual_patch_dilation_radius": args.visual_patch_dilation_radius,
            "visual_patch_dilation_include_non_visual": args.visual_patch_dilation_include_non_visual,
            "visual_fallback_all_token_weight": args.visual_fallback_all_token_weight,
            "visual_rerank_require_informative_visual_query": args.visual_rerank_require_informative_visual_query,
            "visual_rerank_filter_to_informative_visual_query": args.visual_rerank_filter_to_informative_visual_query,
            "visual_rerank_preserve_stage1_base_score": args.visual_rerank_preserve_stage1_base_score,
            "route_features": route_features,
            "route_decision": route_info["route_decision"],
            "route_matched_rule_index": route_info["matched_rule_index"],
            "route_matched_rule": route_info["matched_rule"],
            "learned_doc_reranker_applied": learned_doc_reranker_active,
            "vlm_reranker_applied": vlm_rerank_active,
            "staged_visual_rerank_applied": apply_staged_visual_rerank if not (fixed_base_only and args.base_score_source == "baseline_pred") else False,
            "weights": asdict(weights),
            "grid_search_enabled": args.grid_search,
            "grid_search_best": best_grid_record,
            "doc_feature_record_count": len(doc_feature_records),
            "vlm_record_count": len(vlm_records),
            "vlm_records": vlm_records,
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
    if args.output_doc_feature_jsonl:
        output_doc_feature_jsonl = Path(args.output_doc_feature_jsonl)
        output_doc_feature_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with output_doc_feature_jsonl.open("w", encoding="utf-8") as handle:
            for row in all_doc_feature_rows:
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
        "base_score_source": args.base_score_source,
        "approx_base_page_token_topk": args.approx_base_page_token_topk,
        "approx_base_page_token_scorer": args.approx_base_page_token_scorer,
        "approx_base_page_token_query_prototypes": args.approx_base_page_token_query_prototypes,
        "approx_base_page_token_selector": args.approx_base_page_token_selector,
        "approx_base_page_token_spatial_reserve": args.approx_base_page_token_spatial_reserve,
        "approx_base_page_token_coverage_reserve": args.approx_base_page_token_coverage_reserve,
        "approx_base_page_token_label_reserve": args.approx_base_page_token_label_reserve,
        "approx_base_page_token_soft_visual_query_weight": args.approx_base_page_token_soft_visual_query_weight,
        "approx_base_page_token_soft_patch_visual_bonus": args.approx_base_page_token_soft_patch_visual_bonus,
        "base_only_page_batch_size": args.base_only_page_batch_size,
        "approx_base_page_token_coarse_dtype": args.approx_base_page_token_coarse_dtype,
        "two_stage_exact_top_pages": args.two_stage_exact_top_pages,
        "two_stage_exact_top_docs": args.two_stage_exact_top_docs,
        "visual_rerank_top_pages": args.visual_rerank_top_pages,
        "visual_rerank_top_docs": args.visual_rerank_top_docs,
        "query_route_config_json": args.query_route_config_json,
        "learned_doc_reranker_model": args.learned_doc_reranker_model,
        "learned_doc_reranker_top_docs": args.learned_doc_reranker_top_docs,
        "output_doc_feature_jsonl": args.output_doc_feature_jsonl,
        "vlm_rerank_top_docs": args.vlm_rerank_top_docs,
        "vlm_rerank_bonus": args.vlm_rerank_bonus,
        "vlm_rerank_pages_per_doc": args.vlm_rerank_pages_per_doc,
        "vlm_model_name_or_path": args.vlm_model_name_or_path,
        "vlm_model_type": args.vlm_model_type or None,
        "vlm_bits": args.vlm_bits,
        "gated_visual_top_docs": args.gated_visual_top_docs,
        "scale_auxiliary_by_base_score": args.scale_auxiliary_by_base_score,
        "balance_score_mode": args.balance_score_mode,
        "grounded_context_radius": args.grounded_context_radius,
        "visual_patch_dilation_radius": args.visual_patch_dilation_radius,
        "visual_patch_dilation_include_non_visual": args.visual_patch_dilation_include_non_visual,
        "visual_fallback_all_token_weight": args.visual_fallback_all_token_weight,
        "visual_rerank_require_informative_visual_query": args.visual_rerank_require_informative_visual_query,
        "visual_rerank_filter_to_informative_visual_query": args.visual_rerank_filter_to_informative_visual_query,
        "visual_rerank_preserve_stage1_base_score": args.visual_rerank_preserve_stage1_base_score,
        "splice_query_token_labels": args.splice_query_token_labels,
        "splice_patch_labels_jsonl": args.splice_patch_labels_jsonl,
        "grid_search_enabled": args.grid_search,
        "fixed_weights": asdict(fixed_weights),
        "routed_to_visual_qid_count": routed_to_visual_qids,
        "routed_to_base_qid_count": routed_to_base_qids,
        "learned_doc_reranker_applied": learned_doc_reranker_active,
        "vlm_reranker_applied": vlm_rerank_active,
        "exported_doc_feature_row_count": len(all_doc_feature_rows),
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
