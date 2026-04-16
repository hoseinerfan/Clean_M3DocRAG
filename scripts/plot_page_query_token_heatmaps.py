#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path
from types import SimpleNamespace

import faiss
import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image, ImageDraw, ImageFont

from m3docrag.datasets.m3_docvqa import M3DocVQADataset
from m3docrag.retrieval import ColPaliRetrievalModel, QUERY_TOKEN_FILTER_CHOICES
from m3docrag.utils.paths import LOCAL_DATA_DIR, LOCAL_EMBEDDINGS_DIR, LOCAL_MODEL_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one per-page heatmap for a query: x-axis=query tokens, "
            "y-axis=page token indices."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", help="Free-form query text")
    group.add_argument("--qid", help="Benchmark qid to load from MMQA_<split>.jsonl")

    parser.add_argument("--data_name", default="m3-docvqa")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--gold", help="Optional MMQA_<split>.jsonl override for --qid mode")
    parser.add_argument("--embedding_name", default="colpali-v1.2_m3-docvqa_dev")
    parser.add_argument(
        "--faiss_index_type",
        default="ivfflat",
        choices=["flatip", "ivfflat", "ivfpq"],
    )
    parser.add_argument("--retrieval_model_type", default="colpali", choices=["colpali"])
    parser.add_argument("--retrieval_model_name_or_path", default="colpaligemma-3b-pt-448-base")
    parser.add_argument("--retrieval_adapter_model_name_or_path", default="colpali-v1.2")
    parser.add_argument("--n_retrieval_pages", type=int, default=4)
    parser.add_argument(
        "--query_token_filter",
        default="full",
        choices=QUERY_TOKEN_FILTER_CHOICES,
        help="Match the real retrieval ablation mode when recomputing token-level scores.",
    )
    parser.add_argument(
        "--plot-rank-start",
        type=int,
        default=1,
        help="1-based starting rank of retrieved pages to render",
    )
    parser.add_argument(
        "--plot-rank-count",
        type=int,
        default=4,
        help="How many retrieved pages to render",
    )
    parser.add_argument(
        "--page-uid",
        action="append",
        default=[],
        help="Explicit page_uid to render, e.g. <doc_id>_page<idx>. Can be passed multiple times.",
    )
    parser.add_argument(
        "--explicit-page-mode",
        default="direct_page_maxsim",
        choices=["direct_page_maxsim", "retrieved_contrib"],
        help=(
            "When --page-uid is used, either compute page-local MaxSim directly on the selected page "
            "or preserve the retrieved-contribution semantics from the global FAISS run."
        ),
    )
    parser.add_argument(
        "--cell-width",
        type=int,
        default=28,
        help="Pixel width for each query-token column",
    )
    parser.add_argument(
        "--cell-height",
        type=int,
        default=4,
        help="Pixel height for each page-token row",
    )
    parser.add_argument(
        "--page-token-tick-step",
        type=int,
        default=64,
        help="Show a y-axis tick label every N page tokens",
    )
    parser.add_argument(
        "--contrib-only",
        action="store_true",
        help="Plot only the actual contributing query-token/page-token pairs",
    )
    parser.add_argument(
        "--swap-axes",
        action="store_true",
        help="Swap axes so x=page tokens and y=query tokens",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where per-page PNGs and JSON will be written",
    )
    parser.add_argument(
        "--overlay-on-page",
        action="store_true",
        help="Also render contributing page patches directly on the page image with a score legend",
    )
    parser.add_argument(
        "--overlay-mode",
        default="original_exact",
        choices=["aspectfit", "processor_exact", "original_exact"],
        help=(
            "Overlay geometry: aspectfit keeps the page unwarped inside a square canvas; "
            "processor_exact draws on the square processor canvas; "
            "original_exact projects exact square-grid patches back onto the original page."
        ),
    )
    parser.add_argument(
        "--overlay-image-size",
        type=int,
        default=896,
        help="Square output size for the overlaid page image",
    )
    parser.add_argument(
        "--overlay-clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hide numeric labels inside red boxes. Metadata is shown in a top banner.",
    )
    parser.add_argument(
        "--nonspatial-token-position",
        default="suffix",
        choices=["prefix", "suffix"],
        help=(
            "How to interpret the non-spatial extra tokens in a page embedding. "
            "Use 'suffix' to treat the first grid_side^2 tokens as rasterized page patches."
        ),
    )
    parser.add_argument(
        "--save-patch-crops",
        action="store_true",
        help="Also save the winning original-page patch crops for contributing page tokens.",
    )
    return parser.parse_args()


def make_dataset_args(cli_args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        data_name=cli_args.data_name,
        split=cli_args.split,
        data_len=None,
        use_dummy_images=False,
        load_embedding=True,
        embedding_name=cli_args.embedding_name,
        max_pages=20,
        do_page_padding=False,
        retrieval_model_type=cli_args.retrieval_model_type,
        use_retrieval=True,
        retrieval_only=True,
        page_retrieval_type="logits",
        loop_unique_doc_ids=False,
        n_retrieval_pages=cli_args.n_retrieval_pages,
        query_token_filter=cli_args.query_token_filter,
        faiss_index_type=cli_args.faiss_index_type,
        model_name_or_path="Qwen2-VL-7B-Instruct",
        retrieval_model_name_or_path=cli_args.retrieval_model_name_or_path,
        retrieval_adapter_model_name_or_path=cli_args.retrieval_adapter_model_name_or_path,
        bits=16,
        do_image_splitting=False,
    )


def load_query_from_qid(args: argparse.Namespace) -> str:
    gold_path = Path(args.gold) if args.gold else (
        Path(LOCAL_DATA_DIR) / args.data_name / "multimodalqa" / f"MMQA_{args.split}.jsonl"
    )
    with open(gold_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj["qid"] == args.qid:
                return obj["question"]
    raise KeyError(f"QID not found in gold file: {args.qid}")


def build_flattened_index_inputs(
    docid2embs: dict[str, torch.Tensor],
) -> tuple[list[str], list[int], torch.Tensor]:
    token2pageuid: list[str] = []
    token2localidx: list[int] = []
    all_token_embeddings = []
    for doc_id, doc_emb in docid2embs.items():
        for page_id in range(len(doc_emb)):
            page_emb = doc_emb[page_id].view(-1, 128)
            all_token_embeddings.append(page_emb)
            page_uid = f"{doc_id}_page{page_id}"
            token2pageuid.extend([page_uid] * page_emb.shape[0])
            token2localidx.extend(list(range(page_emb.shape[0])))
    all_token_embeddings = torch.cat(all_token_embeddings, dim=0)
    return token2pageuid, token2localidx, all_token_embeddings


def clean_token_label(token: str) -> str:
    token = token.replace("▁", " ")
    token = token.replace("<pad>", "[PAD]")
    token = token.replace("<bos>", "[BOS]")
    token = token.replace("<eos>", "[EOS]")
    token = token.strip()
    return token if token else "[WS]"


def load_font(size: int) -> ImageFont.ImageFont:
    for font_name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def compute_page_contributions(
    query_emb: np.ndarray,
    index,
    token2pageuid: list[str],
    token2localidx: list[int],
    all_token_embeddings_np: np.ndarray,
    n_retrieval_pages: int,
) -> dict:
    distances, indices = index.search(query_emb, n_retrieval_pages)

    final_page2scores: dict[str, float] = {}
    query_token_page_details: list[dict[str, dict]] = []

    for q_idx, query_token_emb in enumerate(query_emb):
        current_q_page2details: dict[str, dict] = {}
        for nn_idx in range(n_retrieval_pages):
            found_idx = int(indices[q_idx, nn_idx])
            page_uid = token2pageuid[found_idx]
            local_page_token_idx = token2localidx[found_idx]
            doc_token_emb = all_token_embeddings_np[found_idx]
            score = float((query_token_emb * doc_token_emb).sum())

            existing = current_q_page2details.get(page_uid)
            if existing is None or score > existing["score"]:
                current_q_page2details[page_uid] = {
                    "score": score,
                    "found_nearest_doc_token_idx": found_idx,
                    "page_token_idx": local_page_token_idx,
                    "faiss_distance": float(distances[q_idx, nn_idx]),
                    "nn_rank_for_query_token": nn_idx + 1,
                }

        for page_uid, details in current_q_page2details.items():
            final_page2scores[page_uid] = final_page2scores.get(page_uid, 0.0) + details["score"]

        query_token_page_details.append(current_q_page2details)

    sorted_pages = sorted(final_page2scores.items(), key=lambda x: x[1], reverse=True)
    return {
        "query_emb": query_emb,
        "sorted_pages": sorted_pages,
        "top_pages": sorted_pages[:n_retrieval_pages],
        "final_page2scores": final_page2scores,
        "query_token_page_details": query_token_page_details,
    }


def compute_direct_page_maxsim(page_emb: np.ndarray, query_emb: np.ndarray) -> tuple[np.ndarray, dict[int, dict], float]:
    score_matrix = page_emb @ query_emb.T
    contributing_cells: dict[int, dict] = {}
    final_page_score = 0.0

    for query_token_idx in range(score_matrix.shape[1]):
        best_page_token_idx = int(np.argmax(score_matrix[:, query_token_idx]))
        best_score = float(score_matrix[best_page_token_idx, query_token_idx])
        contributing_cells[query_token_idx] = {
            "score": best_score,
            "found_nearest_doc_token_idx": None,
            "page_token_idx": best_page_token_idx,
            "faiss_distance": None,
            "nn_rank_for_query_token": None,
        }
        final_page_score += best_score

    return score_matrix, contributing_cells, final_page_score


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def diverging_color(value: float, bound: float) -> tuple[int, int, int]:
    if bound <= 0:
        return (255, 255, 255)
    x = max(-bound, min(bound, value)) / bound
    if x >= 0:
        t = x
        base = np.array([255, 255, 255], dtype=float)
        high = np.array([214, 39, 40], dtype=float)
    else:
        t = -x
        base = np.array([255, 255, 255], dtype=float)
        high = np.array([31, 119, 180], dtype=float)
    rgb = (1 - t) * base + t * high
    return tuple(int(round(v)) for v in rgb.tolist())


def build_page_heatmap_image(
    query_token_labels: list[str],
    score_matrix: np.ndarray,
    page_title: str,
    page_score: float,
    contributing_cells: dict[int, dict],
    cell_width: int,
    cell_height: int,
    page_token_tick_step: int,
) -> Image.Image:
    n_page_tokens, n_query_tokens = score_matrix.shape

    left_margin = 76
    top_margin = 40
    right_margin = 20
    bottom_margin = 170
    width = left_margin + n_query_tokens * cell_width + right_margin
    height = top_margin + n_page_tokens * cell_height + bottom_margin

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    max_abs = float(np.max(np.abs(score_matrix))) if score_matrix.size else 0.0

    title = f"{page_title} | final_page_score={page_score:.4f}"
    draw.text((10, 12), title, fill="black", font=font)
    subtitle = "Cell = exact dot product(query token, page token). Black box = actual contributing token used in page score."
    draw.text((10, 26), subtitle, fill="black", font=font)

    for page_token_idx in range(n_page_tokens):
        y0 = top_margin + page_token_idx * cell_height
        y1 = y0 + cell_height
        if page_token_idx % page_token_tick_step == 0:
            draw.text((8, y0 - 3), str(page_token_idx), fill="black", font=font)
            draw.line([(left_margin - 4, y0), (left_margin, y0)], fill=(80, 80, 80), width=1)
        for query_token_idx in range(n_query_tokens):
            x0 = left_margin + query_token_idx * cell_width
            x1 = x0 + cell_width
            fill = diverging_color(float(score_matrix[page_token_idx, query_token_idx]), max_abs)
            draw.rectangle([x0, y0, x1, y1], fill=fill, outline=None)

    for query_token_idx, details in contributing_cells.items():
        page_token_idx = details["page_token_idx"]
        x0 = left_margin + query_token_idx * cell_width
        y0 = top_margin + page_token_idx * cell_height
        x1 = x0 + cell_width
        y1 = y0 + cell_height
        draw.rectangle([x0, y0, x1, y1], outline="black", width=2)

    for query_token_idx, token_label in enumerate(query_token_labels):
        token_img = Image.new("RGBA", (140, 24), (255, 255, 255, 0))
        token_draw = ImageDraw.Draw(token_img)
        token_draw.text((0, 0), token_label[:18], fill="black", font=font)
        rotated = token_img.rotate(90, expand=True)
        x = left_margin + query_token_idx * cell_width + max(0, (cell_width - rotated.width) // 2)
        y = top_margin + n_page_tokens * cell_height + 8
        img.paste(rotated, (x, y), rotated)

        grid_x = left_margin + query_token_idx * cell_width
        draw.line(
            [(grid_x, top_margin), (grid_x, top_margin + n_page_tokens * cell_height)],
            fill=(230, 230, 230),
            width=1,
        )

    draw.line(
        [
            (left_margin, top_margin + n_page_tokens * cell_height),
            (left_margin + n_query_tokens * cell_width, top_margin + n_page_tokens * cell_height),
        ],
        fill=(140, 140, 140),
        width=1,
    )

    legend_y = height - 28
    legend_x = left_margin
    for i in range(180):
        t = (i / 179.0) * 2.0 - 1.0
        color = diverging_color(t * max_abs, max_abs)
        draw.rectangle([legend_x + i * 3, legend_y, legend_x + i * 3 + 3, legend_y + 14], fill=color, outline=None)
    draw.text((legend_x, legend_y - 14), f"-{max_abs:.2f}", fill="black", font=font)
    draw.text((legend_x + 240, legend_y - 14), "0", fill="black", font=font)
    draw.text((legend_x + 480, legend_y - 14), f"+{max_abs:.2f}", fill="black", font=font)

    return img


def build_sparse_contrib_image(
    query_token_labels: list[str],
    page_uid: str,
    page_score: float,
    contributing_cells: dict[int, dict],
    cell_width: int,
    cell_height: int,
    swap_axes: bool,
) -> Image.Image:
    font = ImageFont.load_default()

    contrib_items = [
        {
            "query_token_idx": q_idx,
            "query_token": query_token_labels[q_idx],
            **details,
        }
        for q_idx, details in sorted(contributing_cells.items())
    ]
    unique_page_token_indices = sorted({item["page_token_idx"] for item in contrib_items})
    page_token_to_col = {page_token_idx: idx for idx, page_token_idx in enumerate(unique_page_token_indices)}

    if swap_axes:
        row_labels = [f"{item['query_token_idx']}: {item['query_token'][:28]}" for item in contrib_items]
        col_labels = [str(idx) for idx in unique_page_token_indices]
        n_rows = len(row_labels)
        n_cols = len(col_labels)
        left_margin = 200
        top_margin = 40
        right_margin = 20
        bottom_margin = 120
        width = left_margin + n_cols * cell_width + right_margin
        height = top_margin + n_rows * max(cell_height, 24) + bottom_margin

        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        title = f"{page_uid} | final_page_score={page_score:.4f}"
        subtitle = "Only contributing pairs. x=page token idx, y=query token. Cell text = exact dot product."
        draw.text((10, 10), title, fill="black", font=font)
        draw.text((10, 24), subtitle, fill="black", font=font)

        for row_idx, label in enumerate(row_labels):
            y0 = top_margin + row_idx * max(cell_height, 24)
            y1 = y0 + max(cell_height, 24)
            draw.text((8, y0 + 6), label, fill="black", font=font)
            for col_idx in range(n_cols):
                x0 = left_margin + col_idx * cell_width
                x1 = x0 + cell_width
                draw.rectangle([x0, y0, x1, y1], outline=(180, 180, 180), fill="white")

            item = contrib_items[row_idx]
            col_idx = page_token_to_col[item["page_token_idx"]]
            x0 = left_margin + col_idx * cell_width
            y0 = top_margin + row_idx * max(cell_height, 24)
            x1 = x0 + cell_width
            y1 = y0 + max(cell_height, 24)
            draw.rectangle([x0, y0, x1, y1], outline="black", width=2, fill="white")
            draw.text((x0 + 4, y0 + 6), f"{item['score']:.2f}", fill="black", font=font)

        for col_idx, label in enumerate(col_labels):
            token_img = Image.new("RGBA", (140, 24), (255, 255, 255, 0))
            token_draw = ImageDraw.Draw(token_img)
            token_draw.text((0, 0), label, fill="black", font=font)
            rotated = token_img.rotate(90, expand=True)
            x = left_margin + col_idx * cell_width + max(0, (cell_width - rotated.width) // 2)
            y = top_margin + n_rows * max(cell_height, 24) + 6
            img.paste(rotated, (x, y), rotated)

        draw.text((left_margin, height - 18), "page token indices", fill="black", font=font)
        return img

    row_labels = [str(idx) for idx in unique_page_token_indices]
    col_labels = [f"{item['query_token_idx']}: {item['query_token'][:18]}" for item in contrib_items]
    n_rows = len(row_labels)
    n_cols = len(col_labels)
    left_margin = 76
    top_margin = 40
    right_margin = 20
    bottom_margin = 170
    width = left_margin + n_cols * cell_width + right_margin
    height = top_margin + n_rows * max(cell_height, 18) + bottom_margin

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title = f"{page_uid} | final_page_score={page_score:.4f}"
    subtitle = "Only contributing pairs. x=query token, y=page token idx. Cell text = exact dot product."
    draw.text((10, 10), title, fill="black", font=font)
    draw.text((10, 24), subtitle, fill="black", font=font)

    for row_idx, label in enumerate(row_labels):
        y0 = top_margin + row_idx * max(cell_height, 18)
        y1 = y0 + max(cell_height, 18)
        draw.text((8, y0 + 3), label, fill="black", font=font)
        for col_idx in range(n_cols):
            x0 = left_margin + col_idx * cell_width
            x1 = x0 + cell_width
            draw.rectangle([x0, y0, x1, y1], outline=(180, 180, 180), fill="white")

    for col_idx, item in enumerate(contrib_items):
        row_idx = page_token_to_col[item["page_token_idx"]]
        x0 = left_margin + col_idx * cell_width
        y0 = top_margin + row_idx * max(cell_height, 18)
        x1 = x0 + cell_width
        y1 = y0 + max(cell_height, 18)
        draw.rectangle([x0, y0, x1, y1], outline="black", width=2, fill="white")
        draw.text((x0 + 4, y0 + 3), f"{item['score']:.2f}", fill="black", font=font)

    for col_idx, label in enumerate(col_labels):
        token_img = Image.new("RGBA", (200, 24), (255, 255, 255, 0))
        token_draw = ImageDraw.Draw(token_img)
        token_draw.text((0, 0), label, fill="black", font=font)
        rotated = token_img.rotate(90, expand=True)
        x = left_margin + col_idx * cell_width + max(0, (cell_width - rotated.width) // 2)
        y = top_margin + n_rows * max(cell_height, 18) + 6
        img.paste(rotated, (x, y), rotated)

    return img


def infer_patch_grid(page_token_count: int) -> tuple[int, int]:
    for side in range(int(page_token_count**0.5), 0, -1):
        patch_count = side * side
        if patch_count <= page_token_count:
            prefix_tokens = page_token_count - patch_count
            if prefix_tokens >= 0:
                return prefix_tokens, side
    raise ValueError(f"Unable to infer patch grid from page_token_count={page_token_count}")


def collect_spatial_patch_records(
    page_token_count: int,
    contributing_cells: dict[int, dict],
    query_token_labels: list[str],
    nonspatial_token_position: str,
) -> tuple[int, int, list[dict], list[dict]]:
    extra_tokens, grid_side = infer_patch_grid(page_token_count)
    if nonspatial_token_position not in {"prefix", "suffix"}:
        raise ValueError(
            f"Unsupported nonspatial_token_position={nonspatial_token_position!r}"
        )

    patch_to_items: dict[int, list[dict]] = {}
    non_spatial_items: list[dict] = []
    for query_token_idx, details in sorted(contributing_cells.items()):
        item = {
            "query_token_idx": query_token_idx,
            "query_token": query_token_labels[query_token_idx],
            **details,
        }
        page_token_idx = details["page_token_idx"]
        if nonspatial_token_position == "prefix":
            patch_idx = page_token_idx - extra_tokens
        else:
            patch_idx = page_token_idx
        if 0 <= patch_idx < grid_side * grid_side:
            patch_to_items.setdefault(patch_idx, []).append(item)
        else:
            non_spatial_items.append(item)

    sorted_patch_records = []
    for patch_idx, items in sorted(
        patch_to_items.items(),
        key=lambda kv: max(item["score"] for item in kv[1]),
        reverse=True,
    ):
        row = patch_idx // grid_side
        col = patch_idx % grid_side
        best_item = max(items, key=lambda x: x["score"])
        sorted_patch_records.append(
            {
                "patch_idx": patch_idx,
                "grid_row": row,
                "grid_col": col,
                "page_token_idx": best_item["page_token_idx"],
                "items": sorted(items, key=lambda x: x["score"], reverse=True),
            }
        )

    return extra_tokens, grid_side, sorted_patch_records, non_spatial_items


def patch_bbox_xyxy(width: int, height: int, grid_side: int, patch_idx: int) -> tuple[int, int, int, int]:
    row = patch_idx // grid_side
    col = patch_idx % grid_side
    x0 = int(round(col * width / grid_side))
    y0 = int(round(row * height / grid_side))
    x1 = int(round((col + 1) * width / grid_side))
    y1 = int(round((row + 1) * height / grid_side))
    return x0, y0, x1, y1


def build_overlay_image(
    page_image: Image.Image,
    page_uid: str,
    page_score: float,
    page_token_count: int,
    contributing_cells: dict[int, dict],
    query_token_labels: list[str],
    query_token_filter: str,
    output_size: int,
    overlay_mode: str,
    overlay_clean: bool,
    nonspatial_token_position: str,
) -> Image.Image:
    extra_tokens, grid_side, sorted_patch_records, non_spatial_items = collect_spatial_patch_records(
        page_token_count=page_token_count,
        contributing_cells=contributing_cells,
        query_token_labels=query_token_labels,
        nonspatial_token_position=nonspatial_token_position,
    )
    page_img = page_image.convert("RGB")
    original_width, original_height = page_img.size

    if overlay_mode == "aspectfit":
        scale = min(output_size / original_width, output_size / original_height)
        resized_width = max(1, int(round(original_width * scale)))
        resized_height = max(1, int(round(original_height * scale)))
        resized_page = page_img.resize((resized_width, resized_height))
        display_image = Image.new("RGB", (output_size, output_size), "white")
        x_offset = (output_size - resized_width) // 2
        y_offset = (output_size - resized_height) // 2
        display_image.paste(resized_page, (x_offset, y_offset))
    elif overlay_mode == "processor_exact":
        display_image = page_img.resize((output_size, output_size))
        x_offset = 0
        y_offset = 0
        resized_width = output_size
        resized_height = output_size
    elif overlay_mode == "original_exact":
        scale = min(output_size / original_width, output_size / original_height)
        resized_width = max(1, int(round(original_width * scale)))
        resized_height = max(1, int(round(original_height * scale)))
        display_image = page_img.resize((resized_width, resized_height))
        x_offset = 0
        y_offset = 0
    else:
        raise ValueError(f"Unsupported overlay_mode={overlay_mode!r}")

    title_font = load_font(24)
    meta_font = load_font(18)

    header_lines = [
        f"{page_uid} | final_page_score={page_score:.4f}",
        (
            f"grid={grid_side}x{grid_side} | extra_tokens={extra_tokens} | "
            f"query_token_filter={query_token_filter} | token_layout={nonspatial_token_position} | "
            f"overlay_mode={overlay_mode}"
        ),
    ]
    for overlay_id, patch_record in enumerate(sorted_patch_records, start=1):
        contributors = "; ".join(
            [
                f'q{item["query_token_idx"]} "{item["query_token"][:18]}" {item["score"]:.4f} tok={item["page_token_idx"]}'
                for item in patch_record["items"]
            ]
        )
        header_lines.extend(
            textwrap.wrap(
                f"#{overlay_id} patch={patch_record['patch_idx']} grid=({patch_record['grid_row']},{patch_record['grid_col']}) | {contributors}",
                width=120,
            )
        )
    if non_spatial_items:
        non_spatial_summary = "; ".join(
            [
                f'q{item["query_token_idx"]} "{item["query_token"][:18]}" {item["score"]:.4f} tok={item["page_token_idx"]}'
                for item in sorted(non_spatial_items, key=lambda x: x["score"], reverse=True)
            ]
        )
        header_lines.extend(textwrap.wrap(f"Non-spatial | {non_spatial_summary}", width=120))

    measure_image = Image.new("RGB", (1, 1), "white")
    measure_draw = ImageDraw.Draw(measure_image)
    header_width = 0
    header_height = 12
    for idx, line in enumerate(header_lines):
        font = title_font if idx == 0 else meta_font
        bbox = measure_draw.textbbox((0, 0), line, font=font)
        header_width = max(header_width, bbox[2] - bbox[0])
        header_height += (bbox[3] - bbox[1]) + (8 if idx == 0 else 6)

    side_padding = 10
    canvas_width = max(display_image.width + 2 * side_padding, header_width + 2 * side_padding)
    canvas_height = header_height + display_image.height + side_padding
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    page_x = max(side_padding, (canvas_width - display_image.width) // 2)
    top_margin = header_height
    canvas.paste(display_image, (page_x, top_margin))
    draw = ImageDraw.Draw(canvas)

    text_y = 8
    for idx, line in enumerate(header_lines):
        font = title_font if idx == 0 else meta_font
        draw.text((side_padding, text_y), line, fill="black", font=font)
        bbox = draw.textbbox((side_padding, text_y), line, font=font)
        text_y = bbox[3] + (8 if idx == 0 else 6)

    for overlay_id, patch_record in enumerate(sorted_patch_records, start=1):
        patch_idx = patch_record["patch_idx"]
        if overlay_mode == "aspectfit":
            x0, y0, x1, y1 = patch_bbox_xyxy(output_size, output_size, grid_side, patch_idx)
            x0 += page_x
            x1 += page_x
            y0 += top_margin
            y1 += top_margin
        elif overlay_mode == "processor_exact":
            x0, y0, x1, y1 = patch_bbox_xyxy(output_size, output_size, grid_side, patch_idx)
            x0 += page_x
            x1 += page_x
            y0 += top_margin
            y1 += top_margin
        else:
            x0, y0, x1, y1 = patch_bbox_xyxy(display_image.width, display_image.height, grid_side, patch_idx)
            x0 += page_x
            x1 += page_x
            y0 += top_margin
            y1 += top_margin
        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
        if not overlay_clean:
            draw.rectangle([x0, y0, x0 + 20, y0 + 14], fill="red")
            draw.text((x0 + 3, y0 + 2), str(overlay_id), fill="white", font=meta_font)

    return canvas


def page_uid_to_doc_page(page_uid: str) -> tuple[str, int]:
    doc_id, page_idx_text = page_uid.split("_page")
    return doc_id, int(page_idx_text)


def build_selected_pages(
    args: argparse.Namespace,
    contribution_output: dict,
    docid2embs: dict[str, torch.Tensor],
) -> list[dict]:
    sorted_pages = contribution_output["sorted_pages"]
    rank_map = {page_uid: rank for rank, (page_uid, _score) in enumerate(sorted_pages, start=1)}
    score_map = contribution_output["final_page2scores"]

    if args.page_uid:
        selected_pages = []
        for page_uid in args.page_uid:
            doc_id, page_idx = page_uid_to_doc_page(page_uid)
            if doc_id not in docid2embs:
                raise KeyError(f"Explicit page doc_id not found in embeddings: {doc_id}")
            if not (0 <= page_idx < len(docid2embs[doc_id])):
                raise IndexError(
                    f"Explicit page index out of range for {doc_id}: {page_idx} not in [0, {len(docid2embs[doc_id]) - 1}]"
                )
            selected_pages.append(
                {
                    "rank": rank_map.get(page_uid),
                    "page_uid": page_uid,
                    "final_page_score": float(score_map.get(page_uid, 0.0)),
                    "selected_via": (
                        "explicit_page_uid_retrieved_contrib"
                        if args.explicit_page_mode == "retrieved_contrib"
                        else "explicit_page_uid_direct_page_maxsim"
                    ),
                }
            )
        return selected_pages

    plot_rank_start = max(1, args.plot_rank_start)
    if args.plot_rank_count <= 0:
        raise ValueError("--plot-rank-count must be positive")
    plot_start_idx = plot_rank_start - 1
    plot_end_idx = plot_start_idx + args.plot_rank_count
    selected_pages = [
        {
            "rank": rank,
            "page_uid": page_uid,
            "final_page_score": page_score,
            "selected_via": "retrieved_rank_window",
        }
        for rank, (page_uid, page_score) in enumerate(
            sorted_pages[plot_start_idx:plot_end_idx],
            start=plot_rank_start,
        )
    ]
    if not selected_pages:
        raise ValueError(
            f"No retrieved pages available for requested plot window start={plot_rank_start}, "
            f"count={args.plot_rank_count}, total={len(sorted_pages)}"
        )
    return selected_pages


def build_patch_crop_records(
    page_image: Image.Image,
    page_token_count: int,
    contributing_cells: dict[int, dict],
    query_token_labels: list[str],
    output_dir: Path,
    stem: str,
    nonspatial_token_position: str,
) -> list[dict]:
    _extra_tokens, grid_side, sorted_patch_records, _non_spatial_items = collect_spatial_patch_records(
        page_token_count=page_token_count,
        contributing_cells=contributing_cells,
        query_token_labels=query_token_labels,
        nonspatial_token_position=nonspatial_token_position,
    )
    width, height = page_image.size

    if not sorted_patch_records:
        return []

    patch_dir = output_dir / f"{stem}_patch_crops"
    patch_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for crop_rank, patch_record in enumerate(sorted_patch_records, start=1):
        patch_idx = patch_record["patch_idx"]
        x0, y0, x1, y1 = patch_bbox_xyxy(width, height, grid_side, patch_idx)

        crop = page_image.crop((x0, y0, x1, y1))
        best_item = patch_record["items"][0]
        crop_name = (
            f"patch_{crop_rank:03d}_patch{patch_idx:04d}_"
            f"tok{best_item['page_token_idx']:04d}.png"
        )
        crop_path = patch_dir / crop_name
        crop.save(crop_path)

        records.append(
            {
                "crop_rank": crop_rank,
                "patch_idx": patch_idx,
                "page_token_idx": best_item["page_token_idx"],
                "grid_row": patch_record["grid_row"],
                "grid_col": patch_record["grid_col"],
                "bbox_xyxy": [x0, y0, x1, y1],
                "path": str(crop_path),
                "contributing_query_tokens": [
                    {
                        "query_token_idx": item["query_token_idx"],
                        "query_token": item["query_token"],
                        "score": item["score"],
                    }
                    for item in patch_record["items"]
                ],
            }
        )

    return records


def make_page_payload(
    query: str,
    qid: str | None,
    contribution_mode: str,
    query_token_filter: str,
    nonspatial_token_position: str,
    query_token_labels: list[str],
    rank: int,
    page_uid: str,
    final_page_score: float,
    score_matrix: np.ndarray,
    contributing_cells: dict[int, dict],
    patch_crops: list[dict] | None = None,
) -> dict:
    doc_id, page_idx = page_uid_to_doc_page(page_uid)
    contributions = []
    for query_token_idx, details in sorted(contributing_cells.items()):
        contributions.append(
            {
                "query_token_idx": query_token_idx,
                "query_token": query_token_labels[query_token_idx],
                "page_token_idx": details["page_token_idx"],
                "score": details["score"],
                "found_nearest_doc_token_idx": details["found_nearest_doc_token_idx"],
                "faiss_distance": details["faiss_distance"],
                "nn_rank_for_query_token": details["nn_rank_for_query_token"],
            }
        )

    return {
        "qid": qid,
        "query": query,
        "contribution_mode": contribution_mode,
        "query_token_filter": query_token_filter,
        "nonspatial_token_position": nonspatial_token_position,
        "rank": rank,
        "page_uid": page_uid,
        "doc_id": doc_id,
        "page_idx": page_idx,
        "final_page_score": final_page_score,
        "query_token_labels": query_token_labels,
        "page_token_count": int(score_matrix.shape[0]),
        "query_token_count": int(score_matrix.shape[1]),
        "contributing_cells": contributions,
        "patch_crops": patch_crops or [],
    }


def main() -> None:
    args = parse_args()
    query = args.query if args.query is not None else load_query_from_qid(args)

    local_model_dir = Path(LOCAL_MODEL_DIR)
    local_embedding_dir = Path(LOCAL_EMBEDDINGS_DIR)

    retrieval_backbone_dir = local_model_dir / args.retrieval_model_name_or_path
    retrieval_adapter_dir = local_model_dir / args.retrieval_adapter_model_name_or_path
    index_dir = local_embedding_dir / f"{args.embedding_name}_pageindex_{args.faiss_index_type}"

    for path in [retrieval_backbone_dir, retrieval_adapter_dir, index_dir]:
        if not path.exists():
            raise FileNotFoundError(path)

    retrieval_model = ColPaliRetrievalModel(
        backbone_name_or_path=retrieval_backbone_dir,
        adapter_name_or_path=retrieval_adapter_dir,
    )
    accelerator = Accelerator()
    retrieval_model.model = accelerator.prepare(retrieval_model.model)

    dataset = M3DocVQADataset(make_dataset_args(args))
    docid2embs = dataset.load_all_embeddings()

    query_meta = retrieval_model.encode_query_with_metadata(
        query=query,
        to_cpu=True,
        query_token_filter=args.query_token_filter,
    )
    query_emb = query_meta["embeddings"].float().numpy().astype(np.float32)
    query_token_labels = [clean_token_label(token) for token in query_meta["raw_tokens"]]

    contribution_output = None
    if not (args.page_uid and args.explicit_page_mode == "direct_page_maxsim"):
        index = faiss.read_index(str(index_dir / "index.bin"))
        token2pageuid, token2localidx, all_token_embeddings = build_flattened_index_inputs(docid2embs)
        all_token_embeddings_np = all_token_embeddings.float().numpy()

        contribution_output = compute_page_contributions(
            query_emb=query_emb,
            index=index,
            token2pageuid=token2pageuid,
            token2localidx=token2localidx,
            all_token_embeddings_np=all_token_embeddings_np,
            n_retrieval_pages=args.n_retrieval_pages,
        )

    plot_rank_start = max(1, args.plot_rank_start)
    if args.page_uid and args.explicit_page_mode == "direct_page_maxsim":
        selected_pages = []
        for page_uid in args.page_uid:
            doc_id, page_idx = page_uid_to_doc_page(page_uid)
            if doc_id not in docid2embs:
                raise KeyError(f"Explicit page doc_id not found in embeddings: {doc_id}")
            if not (0 <= page_idx < len(docid2embs[doc_id])):
                raise IndexError(
                    f"Explicit page index out of range for {doc_id}: {page_idx} not in [0, {len(docid2embs[doc_id]) - 1}]"
                )
            selected_pages.append(
                {
                    "rank": None,
                    "page_uid": page_uid,
                    "final_page_score": None,
                    "selected_via": "explicit_page_uid_direct_page_maxsim",
                }
            )
    else:
        selected_pages = build_selected_pages(
            args=args,
            contribution_output=contribution_output,
            docid2embs=docid2embs,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "qid": args.qid,
        "query": query,
        "n_retrieval_pages": args.n_retrieval_pages,
        "explicit_page_mode": args.explicit_page_mode if args.page_uid else None,
        "query_token_filter": args.query_token_filter,
        "nonspatial_token_position": args.nonspatial_token_position,
        "plot_rank_start": plot_rank_start if not args.page_uid else None,
        "plot_rank_count": len(selected_pages),
        "explicit_page_uids": args.page_uid,
        "retrieved_pages": selected_pages,
        "files": [],
    }

    for selected_idx, item in enumerate(selected_pages, start=1):
        page_uid = item["page_uid"]
        page_score = item["final_page_score"]
        rank = item["rank"]
        doc_id, page_idx = page_uid_to_doc_page(page_uid)

        page_emb = docid2embs[doc_id][page_idx].view(-1, 128).float().numpy()
        if args.page_uid and args.explicit_page_mode == "direct_page_maxsim":
            score_matrix, contributing_cells, page_score = compute_direct_page_maxsim(
                page_emb=page_emb,
                query_emb=query_emb,
            )
            item["final_page_score"] = page_score
        else:
            score_matrix = page_emb @ contribution_output["query_emb"].T
            contributing_cells = {}
            for query_token_idx, details in enumerate(contribution_output["query_token_page_details"]):
                page_details = details.get(page_uid)
                if page_details is not None:
                    contributing_cells[query_token_idx] = page_details

        display_rank = f"rank={rank}" if rank is not None else f"explicit={selected_idx}"
        page_title = f"{display_rank} {page_uid}"
        if args.contrib_only:
            image = build_sparse_contrib_image(
                query_token_labels=query_token_labels,
                page_uid=page_title,
                page_score=page_score,
                contributing_cells=contributing_cells,
                cell_width=max(args.cell_width, 70),
                cell_height=max(args.cell_height, 24),
                swap_axes=args.swap_axes,
            )
        else:
            image = build_page_heatmap_image(
                query_token_labels=query_token_labels,
                score_matrix=score_matrix,
                page_title=page_title,
                page_score=page_score,
                contributing_cells=contributing_cells,
                cell_width=args.cell_width,
                cell_height=args.cell_height,
                page_token_tick_step=args.page_token_tick_step,
            )

        stem_prefix = f"rank_{rank:04d}" if rank is not None else f"explicit_{selected_idx:04d}"
        stem = f"{stem_prefix}_{sanitize_filename(page_uid)}"
        png_path = output_dir / f"{stem}.png"
        json_path = output_dir / f"{stem}.json"
        image.save(png_path)

        patch_crops = []
        page_image = None
        if args.save_patch_crops or args.overlay_on_page:
            page_image = dataset.get_images_from_doc_id(doc_id)[page_idx]
        if args.save_patch_crops:
            patch_crops = build_patch_crop_records(
                page_image=page_image,
                page_token_count=page_emb.shape[0],
                contributing_cells=contributing_cells,
                query_token_labels=query_token_labels,
                output_dir=output_dir,
                stem=stem,
                nonspatial_token_position=args.nonspatial_token_position,
            )

        json_path.write_text(
            json.dumps(
                make_page_payload(
                    query=query,
                    qid=args.qid,
                    contribution_mode=(
                        "direct_page_maxsim"
                        if args.page_uid and args.explicit_page_mode == "direct_page_maxsim"
                        else "retrieved_contrib"
                    ),
                    query_token_filter=args.query_token_filter,
                    nonspatial_token_position=args.nonspatial_token_position,
                    query_token_labels=query_token_labels,
                    rank=rank,
                    page_uid=page_uid,
                    final_page_score=page_score,
                    score_matrix=score_matrix,
                    contributing_cells=contributing_cells,
                    patch_crops=patch_crops,
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        summary["files"].append(
            {
                "rank": rank,
                "selected_index": selected_idx,
                "page_uid": page_uid,
                "png": str(png_path),
                "json": str(json_path),
                "patch_crops": patch_crops,
            }
        )

        if args.overlay_on_page:
            overlay = build_overlay_image(
                page_image=page_image,
                page_uid=page_uid,
                page_score=page_score,
                page_token_count=page_emb.shape[0],
                contributing_cells=contributing_cells,
                query_token_labels=query_token_labels,
                query_token_filter=args.query_token_filter,
                output_size=args.overlay_image_size,
                overlay_mode=args.overlay_mode,
                overlay_clean=args.overlay_clean,
                nonspatial_token_position=args.nonspatial_token_position,
            )
            overlay_path = output_dir / f"{stem}_overlay.png"
            overlay.save(overlay_path)
            summary["files"][-1]["overlay_png"] = str(overlay_path)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved summary to {summary_path}")
    for file_info in summary["files"]:
        label = (
            f"rank {file_info['rank']}"
            if file_info["rank"] is not None
            else f"explicit page {file_info['selected_index']}"
        )
        print(f"Saved {label} heatmap to {file_info['png']}")


if __name__ == "__main__":
    main()
