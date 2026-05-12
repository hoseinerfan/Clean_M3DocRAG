#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
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
        "--query-token-filter",
        default="full",
        choices=QUERY_TOKEN_FILTER_CHOICES,
        help="Query-token filter used before crop scoring.",
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


def annotate_page(page_image: Image.Image, top_crops: list[dict]) -> Image.Image:
    canvas = page_image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
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
    query_emb = query_meta["embeddings"]
    query_tokens = query_meta["raw_tokens"]

    window_fracs = args.window_frac if args.window_frac else [0.50, 0.33, 0.25]
    crop_records = build_crops(
        page_image=page_image,
        window_fracs=window_fracs,
        stride_frac=float(args.stride_frac),
    )

    all_images = [page_image] + [record["image"] for record in crop_records]
    image_embs = retrieval_model.encode_images(
        images=all_images,
        batch_size=int(args.batch_size),
        to_cpu=True,
        use_tqdm=True,
    )

    full_page_result = score_single_embedding(
        query_emb=query_emb,
        page_emb=image_embs[0],
        query_tokens=query_tokens,
    )

    scored_crops: list[dict] = []
    for record, crop_emb in zip(crop_records, image_embs[1:]):
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
    annotate_page(page_image, top_crops).save(annotated_path)

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
        "query_tokens": query_tokens,
        "window_fracs": window_fracs,
        "stride_frac": float(args.stride_frac),
        "full_page_score": full_page_result["score"],
        "full_page_top_query_tokens": full_page_result["top_query_tokens"],
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
    print(f"full_page_score: {full_page_result['score']:.6f}")
    if top_crops:
        best_crop = top_crops[0]
        print(
            f"best_crop_score: {best_crop['score']:.6f} "
            f"bbox={best_crop['bbox_xyxy']} scale={best_crop['scale_frac']:.2f}"
        )
    print(f"crop_count: {len(scored_crops)}")


if __name__ == "__main__":
    main()
