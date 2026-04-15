#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace

import faiss
import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image, ImageDraw, ImageFont

from m3docrag.datasets.m3_docvqa import M3DocVQADataset
from m3docrag.retrieval import ColPaliRetrievalModel
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


def extract_query_token_labels(retrieval_model: ColPaliRetrievalModel, query: str) -> list[str]:
    processor = retrieval_model.processor
    batch_query = processor.process_queries([query])
    token_ids = batch_query.get("input_ids")
    attention_mask = batch_query.get("attention_mask")

    if token_ids is None:
        return []

    token_ids = token_ids[0].detach().cpu()
    if attention_mask is not None:
        attention_mask = attention_mask[0].detach().cpu()
        if attention_mask.ndim > 1:
            attention_mask = attention_mask.squeeze(-1)
        token_ids = token_ids[attention_mask.bool()]

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return [f"tok_{i}" for i in range(len(token_ids))]
    return [clean_token_label(tok) for tok in tokenizer.convert_ids_to_tokens(token_ids.tolist())]


def compute_page_contributions(
    retrieval_model: ColPaliRetrievalModel,
    query: str,
    index,
    token2pageuid: list[str],
    token2localidx: list[int],
    all_token_embeddings_np: np.ndarray,
    n_retrieval_pages: int,
) -> dict:
    query_emb = retrieval_model.encode_queries([query])[0]
    query_emb = query_emb.cpu().float().numpy().astype(np.float32)

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
        "top_pages": sorted_pages[:n_retrieval_pages],
        "query_token_page_details": query_token_page_details,
    }


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


def page_uid_to_doc_page(page_uid: str) -> tuple[str, int]:
    doc_id, page_idx_text = page_uid.split("_page")
    return doc_id, int(page_idx_text)


def make_page_payload(
    query: str,
    qid: str | None,
    query_token_labels: list[str],
    rank: int,
    page_uid: str,
    final_page_score: float,
    score_matrix: np.ndarray,
    contributing_cells: dict[int, dict],
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
        "rank": rank,
        "page_uid": page_uid,
        "doc_id": doc_id,
        "page_idx": page_idx,
        "final_page_score": final_page_score,
        "query_token_labels": query_token_labels,
        "page_token_count": int(score_matrix.shape[0]),
        "query_token_count": int(score_matrix.shape[1]),
        "contributing_cells": contributions,
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
    index = faiss.read_index(str(index_dir / "index.bin"))
    docid2embs = dataset.load_all_embeddings()
    token2pageuid, token2localidx, all_token_embeddings = build_flattened_index_inputs(docid2embs)
    all_token_embeddings_np = all_token_embeddings.float().numpy()

    contribution_output = compute_page_contributions(
        retrieval_model=retrieval_model,
        query=query,
        index=index,
        token2pageuid=token2pageuid,
        token2localidx=token2localidx,
        all_token_embeddings_np=all_token_embeddings_np,
        n_retrieval_pages=args.n_retrieval_pages,
    )

    query_token_labels = extract_query_token_labels(retrieval_model, query)
    if len(query_token_labels) != len(contribution_output["query_emb"]):
        query_token_labels = [f"tok_{i}" for i in range(len(contribution_output["query_emb"]))]

    plot_rank_start = max(1, args.plot_rank_start)
    if args.plot_rank_count <= 0:
        raise ValueError("--plot-rank-count must be positive")
    full_top_pages = contribution_output["top_pages"]
    plot_start_idx = plot_rank_start - 1
    plot_end_idx = plot_start_idx + args.plot_rank_count
    selected_pages = [
        {"rank": rank, "page_uid": page_uid, "final_page_score": page_score}
        for rank, (page_uid, page_score) in enumerate(full_top_pages[plot_start_idx:plot_end_idx], start=plot_rank_start)
    ]
    if not selected_pages:
        raise ValueError(
            f"No retrieved pages available for requested plot window start={plot_rank_start}, "
            f"count={args.plot_rank_count}, total={len(full_top_pages)}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "qid": args.qid,
        "query": query,
        "n_retrieval_pages": args.n_retrieval_pages,
        "plot_rank_start": plot_rank_start,
        "plot_rank_count": len(selected_pages),
        "retrieved_pages": selected_pages,
        "files": [],
    }

    for item in selected_pages:
        page_uid = item["page_uid"]
        page_score = item["final_page_score"]
        rank = item["rank"]
        doc_id, page_idx = page_uid_to_doc_page(page_uid)

        page_emb = docid2embs[doc_id][page_idx].view(-1, 128).float().numpy()
        score_matrix = page_emb @ contribution_output["query_emb"].T

        contributing_cells = {}
        for query_token_idx, details in enumerate(contribution_output["query_token_page_details"]):
            page_details = details.get(page_uid)
            if page_details is not None:
                contributing_cells[query_token_idx] = page_details

        page_title = f"rank={rank} {page_uid}"
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

        stem = f"rank_{rank:04d}_{sanitize_filename(page_uid)}"
        png_path = output_dir / f"{stem}.png"
        json_path = output_dir / f"{stem}.json"
        image.save(png_path)
        json_path.write_text(
            json.dumps(
                make_page_payload(
                    query=query,
                    qid=args.qid,
                    query_token_labels=query_token_labels,
                    rank=rank,
                    page_uid=page_uid,
                    final_page_score=page_score,
                    score_matrix=score_matrix,
                    contributing_cells=contributing_cells,
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        summary["files"].append(
            {
                "rank": rank,
                "page_uid": page_uid,
                "png": str(png_path),
                "json": str(json_path),
            }
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved summary to {summary_path}")
    for file_info in summary["files"]:
        print(f"Saved rank {file_info['rank']} heatmap to {file_info['png']}")


if __name__ == "__main__":
    main()
