#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image, ImageDraw

from m3docrag.datasets.m3_docvqa import M3DocVQADataset
from m3docrag.retrieval import ColPaliRetrievalModel
from m3docrag.retrieval.colpali import QUERY_TOKEN_FILTER_CHOICES
from m3docrag.utils.paths import LOCAL_DATA_DIR, LOCAL_MODEL_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit one gold page by scoring multi-scale page crops with the current ColPali "
            "query-page MaxSim objective. Useful for checking whether a hard fine-grained cue "
            "becomes visible when scored locally instead of only at full-page scale."
        )
    )
    parser.add_argument("--qid", required=True, help="MMQA qid to audit.")
    parser.add_argument("--data-name", default="m3-docvqa")
    parser.add_argument("--split", default="dev")
    parser.add_argument(
        "--gold",
        help="Optional MMQA jsonl override. Defaults to LOCAL_DATA_DIR/<data-name>/multimodalqa/MMQA_<split>.jsonl",
    )
    parser.add_argument(
        "--run-jsonl",
        required=True,
        help=(
            "run_visual_rerank_batch JSONL used to resolve the target gold page. "
            "The script uses reranked_gold_doc_ranks[gold_doc_pick_index].best_page_uid."
        ),
    )
    parser.add_argument(
        "--gold-doc-pick-index",
        type=int,
        default=0,
        help="Which ranked gold doc entry from reranked_gold_doc_ranks to use. Default: 0.",
    )
    parser.add_argument("--page-uid", help="Optional explicit override like <doc_id>_page<idx>.")
    parser.add_argument("--embedding-name", default="colpali-v1.2_m3-docvqa_dev")
    parser.add_argument("--retrieval-model-name-or-path", default="colpaligemma-3b-pt-448-base")
    parser.add_argument("--retrieval-adapter-model-name-or-path", default="colpali-v1.2")
    parser.add_argument(
        "--splice-patch-labels-jsonl",
        help=(
            "Optional page-patch label JSONL. Required when "
            "--crop-region-source=visual_patch_centers."
        ),
    )
    parser.add_argument(
        "--query-token-filter",
        default="full",
        choices=QUERY_TOKEN_FILTER_CHOICES,
        help="Query-token filter used before crop scoring.",
    )
    parser.add_argument(
        "--cue-token-substring",
        action="append",
        default=[],
        help=(
            "Optional cue-token substring used to restrict query scoring to only matching query "
            "tokens after light normalization. Pass multiple times, e.g. "
            "--cue-token-substring dolphin or --cue-token-substring soccer "
            "--cue-token-substring ball."
        ),
    )
    parser.add_argument(
        "--window-frac",
        type=float,
        action="append",
        default=None,
        help=(
            "Fraction of page width/height used for one crop scale. "
            "Pass multiple times. Defaults: 0.50, 0.33, 0.25"
        ),
    )
    parser.add_argument(
        "--stride-frac",
        type=float,
        default=0.5,
        help="Stride as a fraction of crop width/height. Default: 0.5",
    )
    parser.add_argument(
        "--crop-region-source",
        default="full_page",
        choices=["full_page", "visual_patch_centers"],
        help=(
            "How to choose crop search locations. 'full_page' scans the whole page, while "
            "'visual_patch_centers' keeps only crops whose window contains the center of at "
            "least one visual-labeled patch component from --splice-patch-labels-jsonl."
        ),
    )
    parser.add_argument(
        "--visual-region-fallback",
        default="full_page",
        choices=["full_page", "error"],
        help=(
            "What to do if --crop-region-source=visual_patch_centers finds no visual patch "
            "components on the page."
        ),
    )
    parser.add_argument(
        "--top-crop-count",
        type=int,
        default=12,
        help="How many top-scoring crops to save and report. Default: 12.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Image batch size for crop embedding. Default: 8.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for JSON and crop image outputs.")
    return parser.parse_args()


def resolve_model_path(raw_value: str) -> str:
    path = Path(raw_value)
    if path.exists():
        return str(path)
    local_path = Path(LOCAL_MODEL_DIR) / raw_value
    if local_path.exists():
        return str(local_path)
    return raw_value


def load_gold_row(qid: str, data_name: str, split: str, gold_override: str | None) -> dict:
    gold_path = (
        Path(gold_override)
        if gold_override
        else Path(LOCAL_DATA_DIR) / data_name / "multimodalqa" / f"MMQA_{split}.jsonl"
    )
    with gold_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if str(row.get("qid")) == qid:
                return row
    raise KeyError(f"QID not found in gold file: {qid}")


def load_run_row(path: str, qid: str) -> dict:
    run_path = Path(path)
    with run_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if str(row.get("qid")) == qid:
                return row
    raise KeyError(f"QID {qid!r} not found in run JSONL: {path}")


def resolve_page_uid(args: argparse.Namespace, run_row: dict) -> str:
    if args.page_uid:
        return args.page_uid
    gold_doc_ranks = run_row.get("reranked_gold_doc_ranks", [])
    pick_index = int(args.gold_doc_pick_index)
    if pick_index < 0 or pick_index >= len(gold_doc_ranks):
        raise IndexError(
            f"gold_doc_pick_index={pick_index} out of range for reranked_gold_doc_ranks length={len(gold_doc_ranks)}"
        )
    best_page_uid = gold_doc_ranks[pick_index].get("best_page_uid")
    if not best_page_uid:
        raise KeyError("Selected reranked_gold_doc_ranks item does not contain best_page_uid")
    return str(best_page_uid)


def parse_page_uid(page_uid: str) -> tuple[str, int]:
    if "_page" not in page_uid:
        raise ValueError(f"Invalid page_uid: {page_uid!r}")
    doc_id, page_suffix = page_uid.rsplit("_page", 1)
    return doc_id, int(page_suffix)


def make_dataset_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        data_name=args.data_name,
        split=args.split,
        data_len=None,
        use_dummy_images=False,
        load_embedding=False,
        embedding_name=args.embedding_name,
        max_pages=20,
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


def unique_positions(length: int, window: int, stride: int) -> list[int]:
    if window >= length:
        return [0]
    positions = list(range(0, max(length - window, 0) + 1, stride))
    if positions[-1] != length - window:
        positions.append(length - window)
    return sorted(set(positions))


def infer_patch_grid(page_token_count: int) -> tuple[int, int]:
    for side in range(int(page_token_count**0.5), 0, -1):
        patch_count = side * side
        if patch_count <= page_token_count:
            prefix_tokens = page_token_count - patch_count
            if prefix_tokens >= 0:
                return prefix_tokens, side
    raise ValueError(f"Unable to infer patch grid from page_token_count={page_token_count}")


def clean_token_label(token: str) -> str:
    token = token.replace("▁", " ")
    token = token.replace("<pad>", "[PAD]")
    token = token.replace("<bos>", "[BOS]")
    token = token.replace("<eos>", "[EOS]")
    token = token.strip()
    return token if token else "[WS]"


def normalize_token_label(token: str) -> str:
    return " ".join(clean_token_label(token).strip().lower().split())


def classify_patch_from_splice_row(row: dict) -> str:
    patch_class = str(row.get("patch_class", "")).strip().lower()
    if patch_class in {"visual", "non_visual", "unknown"}:
        return patch_class
    if patch_class == "neutral":
        return "unknown"
    concepts = row.get("top_concepts", []) or []
    concept_names = {str(item.get("concept", "")).strip() for item in concepts}
    if "visual_region" in concept_names:
        return "visual"
    if {"ocr_text", "table_text", "table_structure"} & concept_names:
        return "non_visual"
    return "unknown"


def load_patch_axis_classes_for_single_page(
    *,
    labels_jsonl: str,
    page_id: str,
    n_spatial_patches: int,
) -> list[str]:
    path = Path(labels_jsonl)
    if not path.exists():
        raise FileNotFoundError(path)

    patch_axis_classes = ["unknown"] * int(n_spatial_patches)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if str(row.get("page_id", "")).strip() != page_id:
                continue
            patch_index = int(row.get("patch_index", -1))
            if 0 <= patch_index < len(patch_axis_classes):
                patch_axis_classes[patch_index] = classify_patch_from_splice_row(row)
    return patch_axis_classes


def build_visual_patch_components(
    *,
    patch_axis_classes: list[str],
    grid_side: int,
) -> list[dict]:
    visual_patch_indices = {
        idx for idx, patch_class in enumerate(patch_axis_classes) if patch_class == "visual"
    }
    visited: set[int] = set()
    components: list[dict] = []
    neighbor_offsets = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    for start_idx in sorted(visual_patch_indices):
        if start_idx in visited:
            continue
        queue: deque[int] = deque([start_idx])
        visited.add(start_idx)
        component_indices: list[int] = []
        while queue:
            patch_idx = queue.popleft()
            component_indices.append(patch_idx)
            row = patch_idx // grid_side
            col = patch_idx % grid_side
            for row_offset, col_offset in neighbor_offsets:
                next_row = row + row_offset
                next_col = col + col_offset
                if not (0 <= next_row < grid_side and 0 <= next_col < grid_side):
                    continue
                next_idx = next_row * grid_side + next_col
                if next_idx in visited or next_idx not in visual_patch_indices:
                    continue
                visited.add(next_idx)
                queue.append(next_idx)

        rows = [patch_idx // grid_side for patch_idx in component_indices]
        cols = [patch_idx % grid_side for patch_idx in component_indices]
        row0 = min(rows)
        row1 = max(rows)
        col0 = min(cols)
        col1 = max(cols)
        components.append(
            {
                "component_id": len(components),
                "patch_indices": sorted(component_indices),
                "patch_bbox_rc": [row0, col0, row1, col1],
                "patch_center_rc": [
                    (row0 + row1 + 1) / 2.0,
                    (col0 + col1 + 1) / 2.0,
                ],
            }
        )
    return components


def patch_bbox_to_pixel_bbox(
    *,
    patch_bbox_rc: list[int],
    image_width: int,
    image_height: int,
    grid_side: int,
) -> list[int]:
    row0, col0, row1, col1 = patch_bbox_rc
    patch_w = float(image_width) / float(grid_side)
    patch_h = float(image_height) / float(grid_side)
    x0 = max(0, min(image_width, int(math.floor(col0 * patch_w))))
    y0 = max(0, min(image_height, int(math.floor(row0 * patch_h))))
    x1 = max(0, min(image_width, int(math.ceil((col1 + 1) * patch_w))))
    y1 = max(0, min(image_height, int(math.ceil((row1 + 1) * patch_h))))
    return [x0, y0, x1, y1]


def patch_center_to_pixel_xy(
    *,
    patch_center_rc: list[float],
    image_width: int,
    image_height: int,
    grid_side: int,
) -> list[float]:
    row_center, col_center = patch_center_rc
    patch_w = float(image_width) / float(grid_side)
    patch_h = float(image_height) / float(grid_side)
    center_x = float(col_center) * patch_w
    center_y = float(row_center) * patch_h
    return [center_x, center_y]


def build_crops(page_image: Image.Image, window_fracs: list[float], stride_frac: float) -> list[dict]:
    width, height = page_image.size
    records: list[dict] = []
    crop_id = 0
    for frac in window_fracs:
        if frac <= 0 or frac > 1:
            raise ValueError(f"window-frac must be in (0, 1], got {frac}")
        crop_w = max(1, min(width, int(round(width * frac))))
        crop_h = max(1, min(height, int(round(height * frac))))
        stride_w = max(1, int(round(crop_w * stride_frac)))
        stride_h = max(1, int(round(crop_h * stride_frac)))
        xs = unique_positions(width, crop_w, stride_w)
        ys = unique_positions(height, crop_h, stride_h)
        for y0 in ys:
            for x0 in xs:
                x1 = min(width, x0 + crop_w)
                y1 = min(height, y0 + crop_h)
                crop = page_image.crop((x0, y0, x1, y1))
                records.append(
                    {
                        "crop_id": crop_id,
                        "scale_frac": frac,
                        "bbox_xyxy": [x0, y0, x1, y1],
                        "image": crop,
                    }
                )
                crop_id += 1
    return records


def filter_crops_to_region_centers(
    *,
    crop_records: list[dict],
    region_records: list[dict],
) -> list[dict]:
    filtered: list[dict] = []
    for record in crop_records:
        x0, y0, x1, y1 = record["bbox_xyxy"]
        matched_component_ids: list[int] = []
        for region in region_records:
            center_x, center_y = region["pixel_center_xy"]
            if x0 <= center_x <= x1 and y0 <= center_y <= y1:
                matched_component_ids.append(int(region["component_id"]))
        if matched_component_ids:
            filtered.append(
                {
                    **record,
                    "matched_component_ids": matched_component_ids,
                }
            )
    return filtered


def score_single_embedding(
    query_emb: torch.Tensor,
    page_emb: torch.Tensor,
    query_tokens: list[str],
) -> dict:
    q = query_emb.float().cpu()
    d = page_emb.view(-1, page_emb.shape[-1]).float().cpu()
    sim = q @ d.T
    best_scores, best_indices = sim.max(dim=1)
    total_score = float(best_scores.sum().item())
    token_contributions = [
        {
            "query_token_idx": idx,
            "query_token": token,
            "best_score": float(best_scores[idx].item()),
            "best_page_token_idx": int(best_indices[idx].item()),
        }
        for idx, token in enumerate(query_tokens)
    ]
    token_contributions.sort(key=lambda item: (-item["best_score"], item["query_token_idx"]))
    return {
        "score": total_score,
        "top_query_tokens": token_contributions[:10],
    }


def restrict_query_to_cue_tokens(
    *,
    query_emb: torch.Tensor,
    query_tokens: list[str],
    cue_token_substrings: list[str],
) -> tuple[torch.Tensor, list[str], list[str], list[int], list[str]]:
    normalized_query_tokens = [normalize_token_label(token) for token in query_tokens]
    normalized_cue_substrings = [
        " ".join(str(value).strip().lower().split())
        for value in cue_token_substrings
        if str(value).strip()
    ]
    if not normalized_cue_substrings:
        return query_emb, query_tokens, normalized_query_tokens, list(range(len(query_tokens))), []

    selected_indices = [
        idx
        for idx, token_label in enumerate(normalized_query_tokens)
        if any(substring in token_label for substring in normalized_cue_substrings)
    ]
    if not selected_indices:
        raise ValueError(
            "No query tokens matched cue-token-substring filters. "
            f"filters={normalized_cue_substrings} "
            f"query_tokens={normalized_query_tokens}"
        )
    index_tensor = torch.tensor(selected_indices, dtype=torch.long, device=query_emb.device)
    filtered_query_emb = query_emb.index_select(0, index_tensor)
    filtered_query_tokens = [query_tokens[idx] for idx in selected_indices]
    filtered_query_token_labels = [normalized_query_tokens[idx] for idx in selected_indices]
    return (
        filtered_query_emb,
        filtered_query_tokens,
        filtered_query_token_labels,
        selected_indices,
        normalized_cue_substrings,
    )


def annotate_page(
    page_image: Image.Image,
    top_crops: list[dict],
    *,
    region_records: list[dict] | None = None,
) -> Image.Image:
    canvas = page_image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    if region_records:
        for region in region_records:
            x0, y0, x1, y1 = region["pixel_bbox_xyxy"]
            draw.rectangle([x0, y0, x1, y1], outline=(0, 200, 0), width=3)
            label = f"V{region['component_id']}"
            text_y = min(canvas.size[1] - 16, max(0, y0))
            draw.rectangle(
                [x0, text_y, min(canvas.size[0], x0 + 46), min(canvas.size[1], text_y + 16)],
                fill=(255, 255, 255),
            )
            draw.text((x0 + 2, text_y), label, fill=(0, 140, 0))
    colors = [
        (255, 0, 0),
        (0, 128, 255),
        (0, 180, 0),
        (255, 140, 0),
        (170, 0, 255),
        (255, 0, 140),
    ]
    for rank, record in enumerate(top_crops, start=1):
        x0, y0, x1, y1 = record["bbox_xyxy"]
        color = colors[(rank - 1) % len(colors)]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
        label = f"{rank}:{record['score']:.2f}"
        text_y = max(0, y0 - 16)
        draw.rectangle([x0, text_y, min(canvas.size[0], x0 + 140), y0], fill=(255, 255, 255))
        draw.text((x0 + 2, text_y), label, fill=color)
    return canvas


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gold_row = load_gold_row(
        qid=args.qid,
        data_name=args.data_name,
        split=args.split,
        gold_override=args.gold,
    )
    run_row = load_run_row(args.run_jsonl, args.qid)
    page_uid = resolve_page_uid(args, run_row)
    doc_id, page_idx = parse_page_uid(page_uid)

    dataset = M3DocVQADataset(make_dataset_args(args))
    page_images = dataset.get_images_from_doc_id(doc_id)
    if page_idx < 0 or page_idx >= len(page_images):
        raise IndexError(f"Page index out of range: {page_uid} for doc {doc_id} with {len(page_images)} pages")
    page_image = page_images[page_idx].convert("RGB")

    retrieval_model = ColPaliRetrievalModel(
        backbone_name_or_path=resolve_model_path(args.retrieval_model_name_or_path),
        adapter_name_or_path=resolve_model_path(args.retrieval_adapter_model_name_or_path),
    )

    query_meta = retrieval_model.encode_query_with_metadata(
        query=gold_row["question"],
        to_cpu=True,
        query_token_filter=args.query_token_filter,
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
        cue_token_substrings=args.cue_token_substring,
    )

    full_page_emb = retrieval_model.encode_images(
        images=[page_image],
        batch_size=1,
        to_cpu=True,
        use_tqdm=False,
    )[0]

    patch_region_records: list[dict] = []
    patch_region_source_applied = "full_page"
    patch_region_fallback_reason: str | None = None
    if args.crop_region_source == "visual_patch_centers":
        if not args.splice_patch_labels_jsonl:
            raise ValueError(
                "--crop-region-source=visual_patch_centers requires "
                "--splice-patch-labels-jsonl."
            )
        page_token_count = int(full_page_emb.view(-1, full_page_emb.shape[-1]).shape[0])
        extra_tokens, grid_side = infer_patch_grid(page_token_count)
        page_meta = {
            "page_uid": page_uid,
            "page_id": f"{doc_id}:{page_idx}",
            "page_token_count": page_token_count,
            "extra_tokens": extra_tokens,
            "grid_side": grid_side,
            "n_spatial_patches": grid_side * grid_side,
            "nonspatial_token_position": "suffix",
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

    window_fracs = args.window_frac if args.window_frac else [0.50, 0.33, 0.25]
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
        use_tqdm=True,
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
    top_count = max(1, int(args.top_crop_count))
    top_crops = scored_crops[:top_count]

    page_path = output_dir / f"{args.qid}_{page_uid}_page.png"
    page_image.save(page_path)

    annotated_path = output_dir / f"{args.qid}_{page_uid}_topcrops_overlay.png"
    annotate_page(page_image, top_crops, region_records=patch_region_records).save(annotated_path)

    crop_dir = output_dir / f"{args.qid}_{page_uid}_top_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    for rank, item in enumerate(top_crops, start=1):
        x0, y0, x1, y1 = item["bbox_xyxy"]
        crop = page_image.crop((x0, y0, x1, y1))
        crop_path = crop_dir / (
            f"crop_{rank:02d}_score_{item['score']:.4f}_"
            f"x{x0}_y{y0}_x{x1}_y{y1}_scale_{item['scale_frac']:.2f}.png"
        )
        crop.save(crop_path)
        item["image_path"] = str(crop_path)

    summary = {
        "qid": args.qid,
        "question": gold_row["question"],
        "gold_doc_ids": sorted({str(item["doc_id"]).strip() for item in gold_row.get("supporting_context", [])}),
        "source_run_jsonl": args.run_jsonl,
        "page_uid": page_uid,
        "doc_id": doc_id,
        "page_idx": page_idx,
        "page_image_path": str(page_path),
        "overlay_path": str(annotated_path),
        "query_token_filter": args.query_token_filter,
        "query_tokens_full": query_tokens_full,
        "query_token_labels_full": [normalize_token_label(token) for token in query_tokens_full],
        "query_tokens": query_tokens,
        "query_token_labels": query_token_labels,
        "cue_token_substrings_requested": args.cue_token_substring,
        "cue_token_substrings_applied": normalized_cue_substrings,
        "selected_query_token_indices": selected_query_token_indices,
        "window_fracs": window_fracs,
        "stride_frac": float(args.stride_frac),
        "crop_region_source_requested": args.crop_region_source,
        "crop_region_source_applied": patch_region_source_applied,
        "visual_region_fallback": args.visual_region_fallback,
        "visual_region_fallback_reason": patch_region_fallback_reason,
        "splice_patch_labels_jsonl": args.splice_patch_labels_jsonl,
        "visual_patch_region_count": len(patch_region_records),
        "visual_patch_regions": patch_region_records,
        "full_page_score": full_page_result["score"],
        "full_page_top_query_tokens": full_page_result["top_query_tokens"],
        "crop_count_before_region_filter": len(crop_records_full),
        "crop_count": len(scored_crops),
        "top_crops": top_crops,
    }

    summary_path = output_dir / f"{args.qid}_{page_uid}_crop_audit.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"qid: {args.qid}")
    print(f"question: {gold_row['question']}")
    print(f"page_uid: {page_uid}")
    print(f"page_image_path: {page_path}")
    print(f"overlay_path: {annotated_path}")
    print(f"summary_json: {summary_path}")
    if normalized_cue_substrings:
        print(f"cue_token_substrings_applied: {normalized_cue_substrings}")
        print(f"selected_query_token_labels: {query_token_labels}")
    print(f"full_page_score: {full_page_result['score']:.6f}")
    if top_crops:
        best_crop = top_crops[0]
        print(
            f"best_crop_score: {best_crop['score']:.6f} "
            f"bbox={best_crop['bbox_xyxy']} scale={best_crop['scale_frac']:.2f}"
        )
    print(f"crop_region_source_applied: {patch_region_source_applied}")
    print(f"visual_patch_region_count: {len(patch_region_records)}")
    print(f"crop_count: {len(scored_crops)}")


if __name__ == "__main__":
    main()
