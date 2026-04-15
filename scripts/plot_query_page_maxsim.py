#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import faiss
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from m3docrag.datasets.m3_docvqa import M3DocVQADataset
from m3docrag.retrieval import ColPaliRetrievalModel
from m3docrag.utils.paths import LOCAL_DATA_DIR, LOCAL_EMBEDDINGS_DIR, LOCAL_MODEL_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-query-token MaxSim contributions for the retrieved pages of a "
            "single M3DocVQA query."
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
    parser.add_argument("--n_retrieval_pages", type=int, default=10)
    parser.add_argument(
        "--plot-rank-start",
        type=int,
        default=1,
        help="1-based starting rank of retrieved pages to visualize",
    )
    parser.add_argument(
        "--plot-rank-count",
        type=int,
        default=30,
        help="How many retrieved-page rows to visualize in the heatmap",
    )
    parser.add_argument(
        "--cell-width",
        type=int,
        default=56,
        help="Pixel width for each query-token column in the heatmap",
    )
    parser.add_argument(
        "--cell-height",
        type=int,
        default=28,
        help="Pixel height for each retrieved-page row in the heatmap",
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Output prefix without extension; writes <prefix>.png and <prefix>.json",
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


def build_flattened_index_inputs(
    docid2embs: dict[str, torch.Tensor],
) -> tuple[list[str], torch.Tensor]:
    token2pageuid: list[str] = []
    all_token_embeddings = []
    for doc_id, doc_emb in docid2embs.items():
        for page_id in range(len(doc_emb)):
            page_emb = doc_emb[page_id].view(-1, 128)
            all_token_embeddings.append(page_emb)
            page_uid = f"{doc_id}_page{page_id}"
            token2pageuid.extend([page_uid] * page_emb.shape[0])

    all_token_embeddings = torch.cat(all_token_embeddings, dim=0)
    return token2pageuid, all_token_embeddings


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


def extract_query_token_labels(retrieval_model: ColPaliRetrievalModel, query: str) -> list[str]:
    processor = retrieval_model.processor
    token_ids = None
    attention_mask = None

    batch_query = processor.process_queries([query])
    if "input_ids" in batch_query:
        token_ids = batch_query["input_ids"][0].detach().cpu()
    if "attention_mask" in batch_query:
        attention_mask = batch_query["attention_mask"][0].detach().cpu()

    if token_ids is None:
        return []

    if attention_mask is not None:
        if attention_mask.ndim > 1:
            attention_mask = attention_mask.squeeze(-1)
        valid_positions = attention_mask.bool()
        token_ids = token_ids[valid_positions]

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return [f"tok_{i}" for i in range(len(token_ids))]

    labels = tokenizer.convert_ids_to_tokens(token_ids.tolist())
    return [clean_token_label(token) for token in labels]


def clean_token_label(token: str) -> str:
    token = token.replace("▁", " ")
    token = token.replace("<pad>", "[PAD]")
    token = token.replace("<bos>", "[BOS]")
    token = token.replace("<eos>", "[EOS]")
    token = token.strip()
    return token if token else "[WS]"


def compute_page_contributions(
    retrieval_model: ColPaliRetrievalModel,
    query: str,
    index,
    token2pageuid: list[str],
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
            doc_token_emb = all_token_embeddings_np[found_idx]
            score = float((query_token_emb * doc_token_emb).sum())

            existing = current_q_page2details.get(page_uid)
            if existing is None or score > existing["score"]:
                current_q_page2details[page_uid] = {
                    "score": score,
                    "found_nearest_doc_token_idx": found_idx,
                    "faiss_distance": float(distances[q_idx, nn_idx]),
                    "nn_rank_for_query_token": nn_idx + 1,
                }

        for page_uid, details in current_q_page2details.items():
            final_page2scores[page_uid] = final_page2scores.get(page_uid, 0.0) + details["score"]

        query_token_page_details.append(current_q_page2details)

    sorted_pages = sorted(final_page2scores.items(), key=lambda x: x[1], reverse=True)
    top_pages = sorted_pages[:n_retrieval_pages]

    return {
        "query_emb": query_emb,
        "top_pages": top_pages,
        "query_token_page_details": query_token_page_details,
    }


def rgb_blend(low: tuple[int, int, int], high: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return tuple(int(round((1 - t) * a + t * b)) for a, b in zip(low, high))


def build_heatmap_image(
    query_token_labels: list[str],
    plotted_pages: list[dict],
    query_token_page_details: list[dict[str, dict]],
    cell_width: int,
    cell_height: int,
) -> Image.Image:
    n_cols = len(query_token_labels)
    n_rows = len(plotted_pages)

    if n_cols == 0:
        raise ValueError("No query tokens available to plot.")

    row_label_width = 360
    margin_left = row_label_width
    margin_top = 36
    margin_bottom = 180
    margin_right = 20

    width = margin_left + n_cols * cell_width + margin_right
    height = margin_top + n_rows * cell_height + margin_bottom

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    low = (245, 247, 250)
    high = (31, 119, 180)

    matrix = []
    max_value = 0.0
    for item in plotted_pages:
        page_uid = item["page_uid"]
        row = []
        for details in query_token_page_details:
            value = details.get(page_uid, {}).get("score", 0.0)
            row.append(value)
            max_value = max(max_value, value)
        matrix.append(row)

    title = "Per-query-token MaxSim contribution to retrieved page score"
    draw.text((12, 10), title, fill="black", font=font)

    for row_idx, item in enumerate(plotted_pages):
        page_uid = item["page_uid"]
        page_score = item["final_page_score"]
        rank = item["rank"]
        y0 = margin_top + row_idx * cell_height
        y1 = y0 + cell_height

        label = f"{rank:>4}. {page_uid} | sum={page_score:.3f}"
        draw.text((10, y0 + 8), label, fill="black", font=font)

        for col_idx, value in enumerate(matrix[row_idx]):
            x0 = margin_left + col_idx * cell_width
            x1 = x0 + cell_width
            norm = value / max_value if max_value > 0 else 0.0
            fill = rgb_blend(low, high, norm)
            draw.rectangle([x0, y0, x1, y1], fill=fill, outline=(210, 210, 210))
            if value > 0:
                text = f"{value:.2f}"
                text_color = "white" if norm > 0.55 else "black"
                draw.text((x0 + 4, y0 + 8), text, fill=text_color, font=font)

    for col_idx, token_label in enumerate(query_token_labels):
        token = token_label[:18]
        token_img = Image.new("RGBA", (140, 24), (255, 255, 255, 0))
        token_draw = ImageDraw.Draw(token_img)
        token_draw.text((0, 0), token, fill="black", font=font)
        rotated = token_img.rotate(90, expand=True)

        x = margin_left + col_idx * cell_width + max(0, (cell_width - rotated.width) // 2)
        y = margin_top + n_rows * cell_height + 8
        img.paste(rotated, (x, y), rotated)

        x0 = margin_left + col_idx * cell_width
        draw.line(
            [(x0, margin_top), (x0, margin_top + n_rows * cell_height)],
            fill=(225, 225, 225),
            width=1,
        )

    draw.line(
        [
            (margin_left, margin_top + n_rows * cell_height),
            (margin_left + n_cols * cell_width, margin_top + n_rows * cell_height),
        ],
        fill=(180, 180, 180),
        width=1,
    )

    legend_y = height - 28
    for i in range(100):
        t = i / 99 if 99 else 0.0
        x0 = margin_left + i * 3
        draw.rectangle([x0, legend_y, x0 + 3, legend_y + 14], fill=rgb_blend(low, high, t), outline=None)
    draw.text((margin_left, legend_y - 14), "low", fill="black", font=font)
    draw.text((margin_left + 300, legend_y - 14), "high", fill="black", font=font)

    return img


def build_json_payload(
    query: str,
    qid: str | None,
    query_token_labels: list[str],
    plotted_pages: list[dict],
    query_token_page_details: list[dict[str, dict]],
    full_retrieved_page_count: int,
    plot_rank_start: int,
    plot_rank_count: int,
) -> dict:
    pages = []
    for item in plotted_pages:
        page_uid = item["page_uid"]
        page_score = item["final_page_score"]
        token_contributions = []
        for q_idx, details in enumerate(query_token_page_details):
            page_details = details.get(page_uid)
            if page_details is None:
                continue
            token_contributions.append(
                {
                    "query_token_idx": q_idx,
                    "query_token": query_token_labels[q_idx] if q_idx < len(query_token_labels) else f"tok_{q_idx}",
                    "score": page_details["score"],
                    "found_nearest_doc_token_idx": page_details["found_nearest_doc_token_idx"],
                    "faiss_distance": page_details["faiss_distance"],
                    "nn_rank_for_query_token": page_details["nn_rank_for_query_token"],
                }
            )

        doc_id, page_idx_text = page_uid.split("_page")
        pages.append(
            {
                "rank": item["rank"],
                "page_uid": page_uid,
                "doc_id": doc_id,
                "page_idx": int(page_idx_text),
                "final_page_score": page_score,
                "token_contributions": token_contributions,
            }
        )

    return {
        "qid": qid,
        "query": query,
        "full_retrieved_page_count": full_retrieved_page_count,
        "plotted_rank_start": plot_rank_start,
        "plotted_rank_count": plot_rank_count,
        "query_token_labels": query_token_labels,
        "plotted_pages": pages,
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

    dataset_args = make_dataset_args(args)
    dataset = M3DocVQADataset(dataset_args)

    index = faiss.read_index(str(index_dir / "index.bin"))
    docid2embs = dataset.load_all_embeddings()
    token2pageuid, all_token_embeddings = build_flattened_index_inputs(docid2embs)
    all_token_embeddings_np = all_token_embeddings.float().numpy()

    contribution_output = compute_page_contributions(
        retrieval_model=retrieval_model,
        query=query,
        index=index,
        token2pageuid=token2pageuid,
        all_token_embeddings_np=all_token_embeddings_np,
        n_retrieval_pages=args.n_retrieval_pages,
    )

    query_token_labels = extract_query_token_labels(retrieval_model, query)
    if len(query_token_labels) != len(contribution_output["query_emb"]):
        query_token_labels = [f"tok_{i}" for i in range(len(contribution_output["query_emb"]))]

    plot_rank_start = max(1, args.plot_rank_start)
    if args.plot_rank_count <= 0:
        raise ValueError("--plot-rank-count must be positive")
    plot_start_idx = plot_rank_start - 1
    plot_end_idx = plot_start_idx + args.plot_rank_count
    full_top_pages = contribution_output["top_pages"]
    plotted_pages = [
        {"rank": rank, "page_uid": page_uid, "final_page_score": page_score}
        for rank, (page_uid, page_score) in enumerate(full_top_pages[plot_start_idx:plot_end_idx], start=plot_rank_start)
    ]
    if not plotted_pages:
        raise ValueError(
            f"No retrieved pages available for requested plot window "
            f"start={plot_rank_start}, count={args.plot_rank_count}, total={len(full_top_pages)}"
        )

    image = build_heatmap_image(
        query_token_labels=query_token_labels,
        plotted_pages=plotted_pages,
        query_token_page_details=contribution_output["query_token_page_details"],
        cell_width=args.cell_width,
        cell_height=args.cell_height,
    )

    payload = build_json_payload(
        query=query,
        qid=args.qid,
        query_token_labels=query_token_labels,
        plotted_pages=plotted_pages,
        query_token_page_details=contribution_output["query_token_page_details"],
        full_retrieved_page_count=len(full_top_pages),
        plot_rank_start=plot_rank_start,
        plot_rank_count=len(plotted_pages),
    )

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    json_path = output_prefix.with_suffix(".json")

    image.save(png_path)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"Saved heatmap to {png_path}")
    print(f"Saved contribution JSON to {json_path}")


if __name__ == "__main__":
    main()
