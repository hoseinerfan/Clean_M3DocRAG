#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

from m3docrag.datasets.m3_docvqa import M3DocVQADataset
from m3docrag.retrieval import ColPaliRetrievalModel
from m3docrag.retrieval.colpali import QUERY_TOKEN_FILTER_CHOICES
from scripts.audit_gold_page_crops import (
    build_crops,
    build_visual_patch_components,
    filter_crops_to_region_centers,
    infer_patch_grid,
    load_patch_axis_classes_for_single_page,
    normalize_token_label,
    parse_page_uid,
    patch_bbox_to_pixel_bbox,
    patch_center_to_pixel_xy,
    resolve_model_path,
    restrict_query_to_cue_tokens,
    score_single_embedding,
)
from scripts.run_visual_rerank_batch import load_gold_rows, load_qids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a small batched cue-token crop-verifier experiment over all pages inside "
            "the top-N docs from an existing page-ranking prediction JSON."
        )
    )
    parser.add_argument("--qid-jsonl", required=True, help="JSONL listing qids to evaluate.")
    parser.add_argument("--qid-field", default="qid")
    parser.add_argument("--gold", required=True, help="Path to MMQA_<split>.jsonl")
    parser.add_argument(
        "--prediction-json",
        required=True,
        help=(
            "Prediction JSON from run_visual_rerank_batch.py containing page_retrieval_results "
            "for the same qids."
        ),
    )
    parser.add_argument(
        "--manual-override-jsonl",
        required=True,
        help=(
            "JSONL with one row per qid containing at least "
            "{qid, positive_page_uid, cue_token_substrings}."
        ),
    )
    parser.add_argument("--data-name", default="m3-docvqa")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--embedding-name", default="colpali-v1.2_m3-docvqa_dev")
    parser.add_argument(
        "--query-token-filter",
        default="semantic_only",
        choices=QUERY_TOKEN_FILTER_CHOICES,
    )
    parser.add_argument("--retrieval-model-name-or-path", required=True)
    parser.add_argument("--retrieval-adapter-model-name-or-path", required=True)
    parser.add_argument("--splice-patch-labels-jsonl", required=True)
    parser.add_argument(
        "--top-docs",
        type=int,
        default=20,
        help="How many deduped docs to expand from the source prediction. Default: 20.",
    )
    parser.add_argument(
        "--max-pages-per-doc",
        type=int,
        default=0,
        help="Optional cap on pages expanded per doc. Use 0 for all pages. Default: 0.",
    )
    parser.add_argument(
        "--window-frac",
        type=float,
        action="append",
        default=None,
        help="Fraction of page width/height used for one crop scale. Default: 0.33",
    )
    parser.add_argument(
        "--stride-frac",
        type=float,
        default=1.0,
        help="Stride as a fraction of crop width/height. Default: 1.0",
    )
    parser.add_argument(
        "--crop-region-source",
        default="visual_patch_centers",
        choices=["full_page", "visual_patch_centers"],
    )
    parser.add_argument(
        "--visual-region-fallback",
        default="error",
        choices=["full_page", "error"],
    )
    parser.add_argument(
        "--top-crop-count",
        type=int,
        default=4,
        help="How many top crops to keep in each page record. Default: 4.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-qids", type=int, default=0)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-summary-json", required=True)
    return parser.parse_args()


def make_dataset_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        data_name=args.data_name,
        split=args.split,
        data_len=None,
        use_dummy_images=False,
        load_embedding=False,
        embedding_name=args.embedding_name,
        max_pages=100,
        do_page_padding=False,
        retrieval_model_type="colpali",
        use_retrieval=True,
        retrieval_only=True,
        page_retrieval_type="logits",
        loop_unique_doc_ids=False,
        n_retrieval_pages=4,
        faiss_index_type="ivfflat",
        model_name_or_path="",
        retrieval_model_name_or_path=resolve_model_path(args.retrieval_model_name_or_path),
        retrieval_adapter_model_name_or_path=resolve_model_path(args.retrieval_adapter_model_name_or_path),
        bits=16,
        do_image_splitting=False,
    )


def load_prediction_payload(path: Path, qids: set[str]) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = sorted(qids - set(payload))
    if missing:
        raise KeyError(f"Missing {len(missing)} qids in prediction JSON: {missing[:10]}")
    return {qid: payload[qid] for qid in qids}


def load_manual_overrides(path: Path, qids: set[str]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("qid", "")).strip()
            if not qid or qid not in qids:
                continue
            positive_page_uid = (
                row.get("positive_page_uid")
                or row.get("gold_page_uid")
                or row.get("page_uid")
            )
            if not positive_page_uid:
                raise ValueError(f"Missing positive_page_uid-like field for qid={qid}")
            cue_token_substrings = row.get("cue_token_substrings") or row.get("cue_tokens") or []
            if not isinstance(cue_token_substrings, list) or not cue_token_substrings:
                raise ValueError(f"Missing non-empty cue_token_substrings for qid={qid}")
            exclude_doc_ids = row.get("exclude_doc_ids") or []
            if not isinstance(exclude_doc_ids, list):
                raise ValueError(f"exclude_doc_ids must be a list for qid={qid}")
            rows[qid] = {
                "positive_page_uid": str(positive_page_uid),
                "cue_token_substrings": [str(value) for value in cue_token_substrings],
                "exclude_doc_ids": [str(value) for value in exclude_doc_ids],
                "notes": row.get("notes"),
            }
    missing = sorted(qids - set(rows))
    if missing:
        raise KeyError(f"Missing {len(missing)} qids in manual override JSONL: {missing[:10]}")
    return rows


def dedupe_top_doc_ids(page_rows: list[list[object]], top_docs: int) -> list[str]:
    doc_ids: list[str] = []
    seen: set[str] = set()
    for row in page_rows:
        if not isinstance(row, list) or len(row) < 2:
            continue
        doc_id = str(row[0]).strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        doc_ids.append(doc_id)
        if len(doc_ids) >= top_docs:
            break
    return doc_ids


def prepare_query_package(
    *,
    retrieval_model: ColPaliRetrievalModel,
    question: str,
    query_token_filter: str,
    cue_token_substrings: list[str],
) -> dict:
    query_meta = retrieval_model.encode_query_with_metadata(
        query=question,
        to_cpu=True,
        query_token_filter=query_token_filter,
    )
    query_emb_full = query_meta["embeddings"]
    query_tokens_full = query_meta["raw_tokens"]
    (
        query_emb,
        query_tokens,
        query_token_labels,
        selected_query_token_indices,
        normalized_cue_substrings,
    ) = restrict_query_to_cue_tokens(
        query_emb=query_emb_full,
        query_tokens=query_tokens_full,
        cue_token_substrings=cue_token_substrings,
    )
    return {
        "query_emb": query_emb,
        "query_tokens": query_tokens,
        "query_token_labels": query_token_labels,
        "query_tokens_full": query_tokens_full,
        "query_token_labels_full": [normalize_token_label(token) for token in query_tokens_full],
        "selected_query_token_indices": selected_query_token_indices,
        "cue_token_substrings_applied": normalized_cue_substrings,
    }


def score_page_with_cue_verifier(
    *,
    args: argparse.Namespace,
    retrieval_model: ColPaliRetrievalModel,
    query_package: dict,
    page_uid: str,
    page_image,
) -> dict:
    query_emb = query_package["query_emb"]
    query_tokens = query_package["query_tokens"]
    query_token_labels = query_package["query_token_labels"]
    query_tokens_full = query_package["query_tokens_full"]
    selected_query_token_indices = query_package["selected_query_token_indices"]
    normalized_cue_substrings = query_package["cue_token_substrings_applied"]

    full_page_emb = retrieval_model.encode_images(
        images=[page_image],
        batch_size=1,
        to_cpu=True,
        use_tqdm=False,
    )[0]

    patch_region_records: list[dict] = []
    patch_region_source_applied = "full_page"
    patch_region_fallback_reason: str | None = None
    doc_id, page_idx = parse_page_uid(page_uid)

    if args.crop_region_source == "visual_patch_centers":
        page_token_count = int(full_page_emb.view(-1, full_page_emb.shape[-1]).shape[0])
        extra_tokens, grid_side = infer_patch_grid(page_token_count)
        page_meta = {
            "page_uid": page_uid,
            "page_id": f"{doc_id}:{page_idx}",
            "page_token_count": page_token_count,
            "extra_tokens": extra_tokens,
            "grid_side": grid_side,
            "n_spatial_patches": grid_side * grid_side,
        }
        patch_axis_classes = load_patch_axis_classes_for_single_page(
            labels_jsonl=args.splice_patch_labels_jsonl,
            page_id=page_meta["page_id"],
            n_spatial_patches=page_meta["n_spatial_patches"],
        )
        patch_components = build_visual_patch_components(
            patch_axis_classes=patch_axis_classes,
            grid_side=grid_side,
        )
        for component in patch_components:
            patch_region_records.append(
                {
                    **component,
                    "pixel_bbox_xyxy": patch_bbox_to_pixel_bbox(
                        patch_bbox_rc=component["patch_bbox_rc"],
                        image_width=page_image.size[0],
                        image_height=page_image.size[1],
                        grid_side=grid_side,
                    ),
                    "pixel_center_xy": patch_center_to_pixel_xy(
                        patch_center_rc=component["patch_center_rc"],
                        image_width=page_image.size[0],
                        image_height=page_image.size[1],
                        grid_side=grid_side,
                    ),
                }
            )
        if patch_region_records:
            patch_region_source_applied = "visual_patch_centers"
        elif args.visual_region_fallback == "error":
            raise ValueError(
                f"No visual patch components found for page {page_uid} in "
                f"{args.splice_patch_labels_jsonl}."
            )
        else:
            patch_region_fallback_reason = "no_visual_patch_components"

    window_fracs = args.window_frac if args.window_frac else [0.33]
    crop_records_full = build_crops(
        page_image=page_image,
        window_fracs=window_fracs,
        stride_frac=float(args.stride_frac),
    )
    if patch_region_source_applied == "visual_patch_centers":
        crop_records = filter_crops_to_region_centers(
            crop_records=crop_records_full,
            region_records=patch_region_records,
        )
        if not crop_records:
            if args.visual_region_fallback == "error":
                raise ValueError(
                    f"No crops matched visual patch centers for page {page_uid}. "
                    "Try a larger --window-frac or use --visual-region-fallback=full_page."
                )
            patch_region_fallback_reason = "no_crops_matched_visual_patch_centers"
            patch_region_source_applied = "full_page"
            crop_records = crop_records_full
    else:
        crop_records = crop_records_full

    crop_images = [record["image"] for record in crop_records]
    crop_embs = retrieval_model.encode_images(
        images=crop_images,
        batch_size=int(args.batch_size),
        to_cpu=True,
        use_tqdm=False,
    ) if crop_images else []

    full_page_result = score_single_embedding(
        query_emb=query_emb,
        page_emb=full_page_emb,
        query_tokens=query_tokens,
    )

    scored_crops: list[dict] = []
    for record, crop_emb in zip(crop_records, crop_embs):
        score_info = score_single_embedding(
            query_emb=query_emb,
            page_emb=crop_emb,
            query_tokens=query_tokens,
        )
        scored_crops.append(
            {
                "crop_id": record["crop_id"],
                "scale_frac": record["scale_frac"],
                "bbox_xyxy": record["bbox_xyxy"],
                "matched_component_ids": record.get("matched_component_ids", []),
                "score": score_info["score"],
                "top_query_tokens": score_info["top_query_tokens"],
            }
        )

    scored_crops.sort(key=lambda item: (-item["score"], item["crop_id"]))
    top_crops = scored_crops[: max(1, int(args.top_crop_count))]
    best_crop_score = None if not top_crops else float(top_crops[0]["score"])

    return {
        "status": "ok",
        "page_uid": page_uid,
        "doc_id": doc_id,
        "page_idx": page_idx,
        "query_token_labels": query_token_labels,
        "query_token_labels_full": query_package["query_token_labels_full"],
        "cue_token_substrings_applied": normalized_cue_substrings,
        "selected_query_token_indices": selected_query_token_indices,
        "full_page_score": float(full_page_result["score"]),
        "full_page_top_query_tokens": full_page_result["top_query_tokens"],
        "best_crop_score": best_crop_score,
        "top_crops": top_crops,
        "visual_patch_region_count": len(patch_region_records),
        "crop_count_before_region_filter": len(crop_records_full),
        "crop_count": len(scored_crops),
        "crop_region_source_applied": patch_region_source_applied,
        "visual_region_fallback_reason": patch_region_fallback_reason,
    }


def sort_page_records_for_crop_rank(page_records: list[dict]) -> list[dict]:
    def sort_key(row: dict) -> tuple[float, float, str]:
        if row.get("status") != "ok":
            return (-math.inf, -math.inf, str(row.get("page_uid", "")))
        best_crop_score = float(row.get("best_crop_score") or -math.inf)
        full_page_score = float(row.get("full_page_score") or -math.inf)
        return (best_crop_score, full_page_score, str(row.get("page_uid", "")))

    success_rows = [row for row in page_records if row.get("status") == "ok"]
    failed_rows = [row for row in page_records if row.get("status") != "ok"]
    success_rows.sort(key=sort_key, reverse=True)
    failed_rows.sort(key=lambda row: str(row.get("page_uid", "")))
    return success_rows + failed_rows


def sort_doc_records_for_crop_rank(doc_records: list[dict]) -> list[dict]:
    def sort_key(row: dict) -> tuple[float, float, str]:
        if row.get("status") != "ok":
            return (-math.inf, -math.inf, str(row.get("doc_id", "")))
        return (
            float(row.get("best_crop_score") or -math.inf),
            float(row.get("best_full_page_score") or -math.inf),
            str(row.get("doc_id", "")),
        )

    success_rows = [row for row in doc_records if row.get("status") == "ok"]
    failed_rows = [row for row in doc_records if row.get("status") != "ok"]
    success_rows.sort(key=sort_key, reverse=True)
    failed_rows.sort(key=lambda row: str(row.get("doc_id", "")))
    return success_rows + failed_rows


def build_doc_records(page_records: list[dict], doc_ids_in_order: list[str]) -> list[dict]:
    page_rows_by_doc: dict[str, list[dict]] = {doc_id: [] for doc_id in doc_ids_in_order}
    for row in page_records:
        page_rows_by_doc.setdefault(str(row["doc_id"]), []).append(row)

    doc_records: list[dict] = []
    for doc_id in doc_ids_in_order:
        rows = page_rows_by_doc.get(doc_id, [])
        success_rows = [row for row in rows if row.get("status") == "ok" and row.get("best_crop_score") is not None]
        if success_rows:
            best_page = max(
                success_rows,
                key=lambda row: (float(row["best_crop_score"]), float(row["full_page_score"])),
            )
            doc_records.append(
                {
                    "status": "ok",
                    "doc_id": doc_id,
                    "page_count": len(rows),
                    "successful_page_count": len(success_rows),
                    "failed_page_count": len(rows) - len(success_rows),
                    "best_page_uid": best_page["page_uid"],
                    "best_crop_score": float(best_page["best_crop_score"]),
                    "best_full_page_score": float(best_page["full_page_score"]),
                }
            )
        else:
            doc_records.append(
                {
                    "status": "failed",
                    "doc_id": doc_id,
                    "page_count": len(rows),
                    "successful_page_count": 0,
                    "failed_page_count": len(rows),
                    "best_page_uid": None,
                    "best_crop_score": None,
                    "best_full_page_score": None,
                }
            )
    return doc_records


def rank_of_page(page_records: list[dict], page_uid: str) -> int | None:
    for rank, row in enumerate(page_records, start=1):
        if str(row["page_uid"]) == page_uid:
            return rank
    return None


def rank_of_doc(doc_records: list[dict], doc_id: str) -> int | None:
    for rank, row in enumerate(doc_records, start=1):
        if str(row["doc_id"]) == doc_id:
            return rank
    return None


def summarize_runs(rows: list[dict]) -> dict[str, object]:
    qid_count = len(rows)
    positive_doc_in_top_docs = sum(bool(row["positive_doc_in_top_docs"]) for row in rows)
    positive_page_rank1 = sum(int(row["positive_page_rank_by_best_crop"] or 10**9) == 1 for row in rows)
    positive_doc_rank1 = sum(int(row["positive_doc_rank_by_best_crop"] or 10**9) == 1 for row in rows)
    positive_page_ranks = [
        int(row["positive_page_rank_by_best_crop"])
        for row in rows
        if row["positive_page_rank_by_best_crop"] is not None
    ]
    positive_doc_ranks = [
        int(row["positive_doc_rank_by_best_crop"])
        for row in rows
        if row["positive_doc_rank_by_best_crop"] is not None
    ]
    return {
        "qid_count": qid_count,
        "positive_doc_in_top_docs_count": positive_doc_in_top_docs,
        "positive_page_rank1_count": positive_page_rank1,
        "positive_doc_rank1_count": positive_doc_rank1,
        "mean_positive_page_rank_by_best_crop": (
            float(mean(positive_page_ranks)) if positive_page_ranks else None
        ),
        "mean_positive_doc_rank_by_best_crop": (
            float(mean(positive_doc_ranks)) if positive_doc_ranks else None
        ),
    }


def main() -> None:
    args = parse_args()

    qids = load_qids(Path(args.qid_jsonl), args.qid_field, args.max_qids)
    qid_set = set(qids)
    gold_rows = load_gold_rows(Path(args.gold), qid_set)
    prediction_payload = load_prediction_payload(Path(args.prediction_json), qid_set)
    manual_overrides = load_manual_overrides(Path(args.manual_override_jsonl), qid_set)

    dataset = M3DocVQADataset(make_dataset_args(args))
    retrieval_model = ColPaliRetrievalModel(
        backbone_name_or_path=resolve_model_path(args.retrieval_model_name_or_path),
        adapter_name_or_path=resolve_model_path(args.retrieval_adapter_model_name_or_path),
    )

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json = Path(args.output_summary_json)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)

    doc_image_cache: dict[str, list] = {}
    per_qid_rows: list[dict] = []

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for qid in qids:
            gold_row = gold_rows[qid]
            override = manual_overrides[qid]
            positive_page_uid = str(override["positive_page_uid"])
            positive_doc_id, _positive_page_idx = parse_page_uid(positive_page_uid)
            gold_doc_ids = sorted({str(item["doc_id"]).strip() for item in gold_row.get("supporting_context", [])})
            excluded_doc_ids = (
                [str(value) for value in override["exclude_doc_ids"]]
                if override["exclude_doc_ids"]
                else gold_doc_ids
            )

            prediction_rows = prediction_payload[qid].get("page_retrieval_results", [])
            top_doc_ids = dedupe_top_doc_ids(prediction_rows, int(args.top_docs))
            positive_doc_in_top_docs = positive_doc_id in set(top_doc_ids)
            query_package = prepare_query_package(
                retrieval_model=retrieval_model,
                question=gold_row["question"],
                query_token_filter=args.query_token_filter,
                cue_token_substrings=override["cue_token_substrings"],
            )

            candidate_page_uids: list[str] = []
            for doc_id in top_doc_ids:
                if doc_id not in doc_image_cache:
                    doc_image_cache[doc_id] = dataset.get_images_from_doc_id(doc_id)
                page_images = doc_image_cache[doc_id]
                page_limit = len(page_images) if args.max_pages_per_doc <= 0 else min(len(page_images), int(args.max_pages_per_doc))
                for page_idx in range(page_limit):
                    candidate_page_uids.append(f"{doc_id}_page{page_idx}")

            page_records: list[dict] = []
            for page_uid in candidate_page_uids:
                doc_id, page_idx = parse_page_uid(page_uid)
                page_images = doc_image_cache[doc_id]
                page_image = page_images[page_idx].convert("RGB")
                try:
                    page_record = score_page_with_cue_verifier(
                        args=args,
                        retrieval_model=retrieval_model,
                        query_package=query_package,
                        page_uid=page_uid,
                        page_image=page_image,
                    )
                except Exception as exc:  # noqa: BLE001
                    page_record = {
                        "status": "failed",
                        "page_uid": page_uid,
                        "doc_id": doc_id,
                        "page_idx": page_idx,
                        "error": str(exc),
                        "best_crop_score": None,
                        "full_page_score": None,
                    }
                page_records.append(page_record)

            page_records = sort_page_records_for_crop_rank(page_records)
            doc_records = sort_doc_records_for_crop_rank(build_doc_records(page_records, top_doc_ids))

            result_row = {
                "qid": qid,
                "question": gold_row["question"],
                "gold_doc_ids": gold_doc_ids,
                "top_doc_ids": top_doc_ids,
                "positive_page_uid": positive_page_uid,
                "positive_doc_id": positive_doc_id,
                "positive_doc_in_top_docs": positive_doc_in_top_docs,
                "cue_token_substrings": override["cue_token_substrings"],
                "excluded_doc_ids": excluded_doc_ids,
                "candidate_page_count": len(candidate_page_uids),
                "successful_page_count": sum(row.get("status") == "ok" for row in page_records),
                "failed_page_count": sum(row.get("status") != "ok" for row in page_records),
                "positive_page_rank_by_best_crop": rank_of_page(page_records, positive_page_uid),
                "positive_doc_rank_by_best_crop": rank_of_doc(doc_records, positive_doc_id),
                "top_page_uid_by_best_crop": None if not page_records else page_records[0]["page_uid"],
                "top_doc_id_by_best_crop": None if not doc_records else doc_records[0]["doc_id"],
                "page_results": page_records,
                "doc_results": doc_records,
            }
            handle.write(json.dumps(result_row) + "\n")
            per_qid_rows.append(result_row)

    summary = {
        "top_docs": int(args.top_docs),
        "max_pages_per_doc": int(args.max_pages_per_doc),
        "query_token_filter": args.query_token_filter,
        "crop_region_source": args.crop_region_source,
        "visual_region_fallback": args.visual_region_fallback,
        "window_fracs": args.window_frac if args.window_frac else [0.33],
        "stride_frac": float(args.stride_frac),
        "per_qid_summary": summarize_runs(per_qid_rows),
    }
    output_summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"saved_jsonl: {output_jsonl}")
    print(f"saved_summary: {output_summary_json}")
    print(f"num_qids: {len(per_qid_rows)}")
    print(f"positive_doc_rank1_count: {summary['per_qid_summary']['positive_doc_rank1_count']}")
    print(f"positive_page_rank1_count: {summary['per_qid_summary']['positive_page_rank1_count']}")


if __name__ == "__main__":
    main()
