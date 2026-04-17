#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plot_page_query_token_heatmaps import (
    build_flattened_index_inputs,
    build_selected_pages,
    clean_token_label,
    collect_spatial_patch_records,
    compute_direct_page_maxsim,
    compute_page_contributions,
    load_font,
    load_gold_row_from_qid,
    make_dataset_args,
    page_uid_to_doc_page,
    wrap_text_lines,
)

QUERY_TOKEN_FILTER_CHOICES = ("full", "drop_pad_like", "semantic_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a dense query-token vs all-page-patches dot grid. "
            "Rows=query tokens, columns=all spatial page patches, red dots mark "
            "the contributing page patch for a query token."
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
        help="Match the retrieval ablation mode when recomputing contributions.",
    )
    parser.add_argument(
        "--ignore-pad-scores-in-final-ranking",
        action="store_true",
        help="Keep PAD tokens in ANN search but exclude their scores from the final page-score sum used for reranking.",
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
        "--gold-doc-pages",
        action="store_true",
        help="When used with --qid, expand all pages from the qid's gold supporting docs into explicit page_uids.",
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
        "--nonspatial-token-position",
        default="suffix",
        choices=["prefix", "suffix"],
        help=(
            "How to interpret the non-spatial extra tokens in a page embedding. "
            "Use 'suffix' to treat the first grid_side^2 tokens as rasterized page patches."
        ),
    )
    parser.add_argument(
        "--cell-width",
        type=int,
        default=12,
        help="Pixel width for each page-patch column.",
    )
    parser.add_argument(
        "--cell-height",
        type=int,
        default=38,
        help="Pixel height for each query-token row.",
    )
    parser.add_argument(
        "--patch-tick-step",
        type=int,
        default=32,
        help="Show an x-axis patch index label every N page patches.",
    )
    parser.add_argument(
        "--dot-radius",
        type=int,
        default=4,
        help="Radius in pixels for the red contribution dots.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where per-page SVGs and JSON will be written.",
    )
    parser.add_argument(
        "--splice-patch-labels-jsonl",
        default="",
        help=(
            "Optional SpLiCE patch-label JSONL from build_layout_patch_labels.py. "
            "Used to classify x-axis patches as visual / non-visual / unknown."
        ),
    )
    parser.add_argument(
        "--splice-query-token-labels",
        default="",
        help=(
            "Optional JSON or JSONL carrying query-token visual/non-visual labels keyed by qid/query_id. "
            "Supported fields: query_token_classes, visual_token_indices, non_visual_token_indices, "
            "visual_tokens, non_visual_tokens."
        ),
    )
    return parser.parse_args()


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


def load_splice_patch_axis_classes(
    labels_jsonl: str,
    page_uid: str,
    n_spatial_patches: int,
) -> list[str]:
    classes = ["unknown"] * n_spatial_patches
    if not labels_jsonl:
        return classes

    labels_path = Path(labels_jsonl)
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)

    doc_id, page_idx = page_uid_to_doc_page(page_uid)
    splice_page_id = f"{doc_id}:{page_idx}"

    with labels_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if str(row.get("page_id", "")).strip() != splice_page_id:
                continue
            patch_index = int(row.get("patch_index", -1))
            if 0 <= patch_index < n_spatial_patches:
                classes[patch_index] = classify_patch_from_splice_row(row)
    return classes


def _query_label_record_from_path(path: Path, qid: str | None) -> dict | None:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row_qid = str(row.get("qid", row.get("query_id", ""))).strip()
                if qid is not None and row_qid == qid:
                    return row
        return None

    payload = json.load(path.open("r", encoding="utf-8"))
    if isinstance(payload, dict):
        if qid is not None and qid in payload and isinstance(payload[qid], dict):
            return payload[qid]
        row_qid = str(payload.get("qid", payload.get("query_id", ""))).strip()
        if qid is not None and row_qid == qid:
            return payload
    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue
            row_qid = str(row.get("qid", row.get("query_id", ""))).strip()
            if qid is not None and row_qid == qid:
                return row
    return None


def load_splice_query_axis_classes(
    query_labels_path: str,
    qid: str | None,
    query_token_labels: list[str],
) -> list[str]:
    classes = ["unknown"] * len(query_token_labels)
    if not query_labels_path:
        return classes

    path = Path(query_labels_path)
    if not path.exists():
        raise FileNotFoundError(path)

    record = _query_label_record_from_path(path, qid)
    if record is None:
        return classes

    explicit_classes = record.get("query_token_classes")
    if isinstance(explicit_classes, list) and explicit_classes:
        if all(isinstance(value, dict) for value in explicit_classes):
            for value in explicit_classes:
                idx = int(value.get("index", -1))
                if not (0 <= idx < len(classes)):
                    continue
                text = str(value.get("class", "")).strip().lower()
                if text == "neutral":
                    text = "unknown"
                if text in {"visual", "non_visual", "unknown"}:
                    classes[idx] = text
            if any(x != "unknown" for x in classes):
                return classes
        elif len(explicit_classes) == len(query_token_labels):
            normalized = []
            for value in explicit_classes:
                text = str(value).strip().lower()
                if text == "neutral":
                    text = "unknown"
                if text in {"visual", "non_visual", "unknown"}:
                    normalized.append(text)
                else:
                    normalized.append("unknown")
            return normalized

    visual_token_indices = {int(x) for x in record.get("visual_token_indices", [])}
    non_visual_token_indices = {int(x) for x in record.get("non_visual_token_indices", [])}

    for idx in visual_token_indices:
        if 0 <= idx < len(classes):
            classes[idx] = "visual"
    for idx in non_visual_token_indices:
        if 0 <= idx < len(classes) and classes[idx] == "unknown":
            classes[idx] = "non_visual"

    visual_tokens = {str(x).strip().lower() for x in record.get("visual_tokens", [])}
    non_visual_tokens = {str(x).strip().lower() for x in record.get("non_visual_tokens", [])}
    if visual_tokens or non_visual_tokens:
        for idx, token in enumerate(query_token_labels):
            token_key = token.strip().lower()
            if token_key in visual_tokens:
                classes[idx] = "visual"
            elif token_key in non_visual_tokens and classes[idx] == "unknown":
                classes[idx] = "non_visual"

    return classes


def axis_class_counts(axis_classes: list[str]) -> dict[str, int]:
    counts = {"visual": 0, "non_visual": 0, "unknown": 0}
    for value in axis_classes:
        counts[value if value in counts else "unknown"] += 1
    return counts


def axis_class_fill(axis_class: str) -> str:
    if axis_class == "visual":
        return "rgb(255,170,170)"
    if axis_class == "non_visual":
        return "rgb(180,220,255)"
    return "rgb(255,255,255)"


def axis_class_text_fill(axis_class: str) -> str:
    if axis_class == "visual":
        return "rgb(160,0,0)"
    if axis_class == "non_visual":
        return "rgb(35,75,120)"
    return "black"


def build_dot_matrix_svg(
    *,
    query_token_labels: list[str],
    page_uid: str,
    page_score: float,
    page_token_count: int,
    contributing_cells: dict[int, dict],
    query_token_filter: str,
    nonspatial_token_position: str,
    cell_width: int,
    cell_height: int,
    patch_tick_step: int,
    dot_radius: int,
    query_axis_classes: list[str] | None = None,
    patch_axis_classes: list[str] | None = None,
) -> tuple[str, list[dict], list[dict], int, int]:
    extra_tokens, grid_side, _spatial_patch_records, non_spatial_items = collect_spatial_patch_records(
        page_token_count=page_token_count,
        contributing_cells=contributing_cells,
        query_token_labels=query_token_labels,
        nonspatial_token_position=nonspatial_token_position,
    )

    n_cols = grid_side * grid_side
    n_rows = len(query_token_labels)
    if n_rows == 0:
        raise ValueError("No query tokens available to plot.")

    title_font = load_font(18)
    label_font = load_font(16)
    tick_font = load_font(14)
    score_font = load_font(11)

    probe = Image.new("RGB", (1, 1), "white")
    probe_draw = ImageDraw.Draw(probe)

    row_labels = [f'{idx}: {token[:28]}' for idx, token in enumerate(query_token_labels)]
    max_row_label_width = 0
    for label in row_labels:
        bbox = probe_draw.textbbox((0, 0), label, font=label_font)
        max_row_label_width = max(max_row_label_width, bbox[2] - bbox[0])

    left_margin = max(120, max_row_label_width + 20)
    right_margin = 20
    width = left_margin + n_cols * cell_width + right_margin

    title = f"{page_uid} | final_page_score={page_score:.4f}"
    subtitle = (
        f"Rows=query tokens | cols=all spatial page patches ({grid_side}x{grid_side} row-major) | "
        f"extra_tokens={extra_tokens} | query_token_filter={query_token_filter} | token_layout={nonspatial_token_position}"
    )
    title_lines = wrap_text_lines(probe_draw, title, font=title_font, max_width=width - 20)
    subtitle_lines = wrap_text_lines(probe_draw, subtitle, font=label_font, max_width=width - 20)

    def total_line_height(lines: list[str], font) -> int:
        return sum(
            (probe_draw.textbbox((0, 0), line, font=font)[3] - probe_draw.textbbox((0, 0), line, font=font)[1]) + 4
            for line in lines
        )

    legend_lines = [
        "axis classes: visual=rose | non_visual=blue | unknown=gray"
    ] if (query_axis_classes or patch_axis_classes) else []
    top_margin = (
        18
        + total_line_height(title_lines, title_font)
        + total_line_height(subtitle_lines, label_font)
        + total_line_height(legend_lines, tick_font)
        + 8
    )
    bottom_margin = 110
    height = top_margin + n_rows * cell_height + bottom_margin

    def line_height(font) -> int:
        bbox = probe_draw.textbbox((0, 0), "Ag", font=font)
        return (bbox[3] - bbox[1]) + 4

    title_line_height = line_height(title_font)
    label_line_height = line_height(label_font)
    tick_line_height = line_height(tick_font)
    score_line_height = line_height(score_font)

    def text_block_lines(start_x: int, start_y: int, lines: list[str], font_size: int, line_height_px: int) -> list[str]:
        blocks = []
        y = start_y
        for line in lines:
            blocks.append(
                f'<text x="{start_x}" y="{y + font_size}" font-size="{font_size}" '
                f'font-family="DejaVu Sans, Arial, sans-serif" fill="black">{escape(line)}</text>'
            )
            y += line_height_px
        return blocks

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    text_y = 10
    svg_parts.extend(text_block_lines(10, text_y, title_lines, 18, title_line_height))
    text_y += len(title_lines) * title_line_height
    svg_parts.extend(text_block_lines(10, text_y, subtitle_lines, 16, label_line_height))
    text_y += len(subtitle_lines) * label_line_height
    if legend_lines:
        svg_parts.extend(text_block_lines(10, text_y, legend_lines, 14, tick_line_height))

    grid_x0 = left_margin
    grid_y0 = top_margin
    grid_width = n_cols * cell_width
    grid_height = n_rows * cell_height

    if patch_axis_classes:
        for patch_idx, patch_class in enumerate(patch_axis_classes):
            x0 = grid_x0 + patch_idx * cell_width
            svg_parts.append(
                f'<rect x="{x0}" y="{grid_y0}" width="{cell_width}" height="{grid_height}" '
                f'fill="{axis_class_fill(patch_class)}" fill-opacity="0.60" stroke="none"/>'
            )

    for row_idx, row_label in enumerate(row_labels):
        y0 = grid_y0 + row_idx * cell_height
        y_center = y0 + cell_height / 2 + 6
        row_class = query_axis_classes[row_idx] if query_axis_classes else "unknown"
        if query_axis_classes:
            svg_parts.append(
                f'<rect x="{grid_x0}" y="{y0}" width="{grid_width}" height="{cell_height}" '
                f'fill="{axis_class_fill(row_class)}" fill-opacity="0.45" stroke="none"/>'
            )
        svg_parts.append(
            f'<text x="8" y="{y_center:.1f}" font-size="16" '
            f'font-family="DejaVu Sans, Arial, sans-serif" fill="{axis_class_text_fill(row_class)}">{escape(row_label)}</text>'
        )

    for col_boundary in range(n_cols + 1):
        x = grid_x0 + col_boundary * cell_width
        stroke = "rgb(220,220,220)"
        stroke_width = 1
        if col_boundary < n_cols and col_boundary % grid_side == 0:
            stroke = "rgb(180,180,180)"
            stroke_width = 2
        svg_parts.append(
            f'<line x1="{x}" y1="{grid_y0}" x2="{x}" y2="{grid_y0 + grid_height}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )
    for row_boundary in range(n_rows + 1):
        y = grid_y0 + row_boundary * cell_height
        svg_parts.append(
            f'<line x1="{grid_x0}" y1="{y}" x2="{grid_x0 + grid_width}" y2="{y}" '
            'stroke="rgb(220,220,220)" stroke-width="1"/>'
        )

    dots = []
    for query_token_idx, details in sorted(contributing_cells.items()):
        page_token_idx = details["page_token_idx"]
        patch_idx = page_token_idx - extra_tokens if nonspatial_token_position == "prefix" else page_token_idx
        if not (0 <= patch_idx < n_cols):
            continue
        x0 = grid_x0 + patch_idx * cell_width
        y0 = grid_y0 + query_token_idx * cell_height
        cx = x0 + cell_width / 2
        cy = y0 + cell_height * 0.62
        score_text = f'{float(details["score"]):.2f}'
        score_bbox = probe_draw.textbbox((0, 0), score_text, font=score_font)
        score_width = score_bbox[2] - score_bbox[0]
        score_x = cx - score_width / 2
        score_y = max(y0 + score_line_height - 2, cy - dot_radius - 4)
        svg_parts.append(
            f'<text x="{score_x:.2f}" y="{score_y:.2f}" font-size="11" '
            f'font-family="DejaVu Sans, Arial, sans-serif" fill="black">{escape(score_text)}</text>'
        )
        svg_parts.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{dot_radius}" fill="rgb(220,20,20)"/>'
        )
        dots.append(
            {
                "query_token_idx": query_token_idx,
                "query_token": query_token_labels[query_token_idx],
                "page_token_idx": page_token_idx,
                "patch_idx": patch_idx,
                "grid_row": patch_idx // grid_side,
                "grid_col": patch_idx % grid_side,
                "score": float(details["score"]),
            }
        )

    tick_y = top_margin + n_rows * cell_height + 8
    tick_positions = list(range(0, n_cols, patch_tick_step))
    if (n_cols - 1) not in tick_positions:
        tick_positions.append(n_cols - 1)
    for patch_idx in tick_positions:
        label = str(patch_idx)
        x0 = grid_x0 + patch_idx * cell_width
        bbox = probe_draw.textbbox((0, 0), label, font=tick_font)
        text_width = bbox[2] - bbox[0]
        label_x = x0 + (cell_width - text_width) / 2
        patch_class = patch_axis_classes[patch_idx] if patch_axis_classes and patch_idx < len(patch_axis_classes) else "unknown"
        svg_parts.append(
            f'<text x="{label_x}" y="{tick_y + 14}" font-size="14" '
            f'font-family="DejaVu Sans, Arial, sans-serif" fill="{axis_class_text_fill(patch_class)}">{escape(label)}</text>'
        )

    axis_label = "x-axis: all spatial page patches (row-major patch index)"
    axis_bbox = probe_draw.textbbox((0, 0), axis_label, font=tick_font)
    axis_width = axis_bbox[2] - axis_bbox[0]
    axis_x = grid_x0 + max(0, (n_cols * cell_width - axis_width) // 2)
    svg_parts.append(
        f'<text x="{axis_x}" y="{tick_y + 32}" font-size="14" '
        f'font-family="DejaVu Sans, Arial, sans-serif" fill="black">{escape(axis_label)}</text>'
    )

    if non_spatial_items:
        non_spatial_text = "Non-spatial contributors not plotted: " + "; ".join(
            f'q{item["query_token_idx"]} "{item["query_token"][:16]}" tok={item["page_token_idx"]} score={item["score"]:.4f}'
            for item in sorted(non_spatial_items, key=lambda x: x["score"], reverse=True)
        )
        non_spatial_lines = wrap_text_lines(probe_draw, non_spatial_text, font=tick_font, max_width=width - 20)
        svg_parts.extend(text_block_lines(10, tick_y + 38, non_spatial_lines, 14, tick_line_height))

    svg_parts.append("</svg>")
    return "\n".join(svg_parts), dots, non_spatial_items, extra_tokens, grid_side


def make_page_payload(
    *,
    qid: str | None,
    query: str,
    page_uid: str,
    rank: int | None,
    final_page_score: float,
    contribution_mode: str,
    query_token_filter: str,
    ignore_pad_scores_in_final_ranking: bool,
    nonspatial_token_position: str,
    query_token_labels: list[str],
    page_token_count: int,
    grid_side: int,
    extra_tokens: int,
    dots: list[dict],
    non_spatial_items: list[dict],
    query_axis_classes: list[str],
    patch_axis_classes: list[str],
) -> dict:
    doc_id, page_idx = page_uid_to_doc_page(page_uid)
    return {
        "qid": qid,
        "query": query,
        "contribution_mode": contribution_mode,
        "query_token_filter": query_token_filter,
        "ignore_pad_scores_in_final_ranking": ignore_pad_scores_in_final_ranking,
        "nonspatial_token_position": nonspatial_token_position,
        "rank": rank,
        "page_uid": page_uid,
        "doc_id": doc_id,
        "page_idx": page_idx,
        "final_page_score": final_page_score,
        "query_token_labels": query_token_labels,
        "query_token_count": len(query_token_labels),
        "page_token_count": int(page_token_count),
        "grid_side": grid_side,
        "extra_tokens": extra_tokens,
        "n_spatial_patches": grid_side * grid_side,
        "n_dots": len(dots),
        "n_non_spatial_contributors": len(non_spatial_items),
        "query_axis_classes": query_axis_classes,
        "query_axis_class_counts": axis_class_counts(query_axis_classes),
        "patch_axis_classes": patch_axis_classes,
        "patch_axis_class_counts": axis_class_counts(patch_axis_classes),
        "dots": dots,
        "non_spatial_contributors": [
            {
                "query_token_idx": item["query_token_idx"],
                "query_token": item["query_token"],
                "page_token_idx": item["page_token_idx"],
                "score": float(item["score"]),
                "found_nearest_doc_token_idx": item["found_nearest_doc_token_idx"],
                "faiss_distance": item["faiss_distance"],
                "nn_rank_for_query_token": item["nn_rank_for_query_token"],
            }
            for item in non_spatial_items
        ],
    }


def main() -> None:
    import faiss
    from accelerate import Accelerator
    from m3docrag.datasets.m3_docvqa import M3DocVQADataset
    from m3docrag.retrieval import ColPaliRetrievalModel

    args = parse_args()
    gold_row = None
    if args.qid is not None:
        gold_row = load_gold_row_from_qid(args)
    query = args.query if args.query is not None else gold_row["question"]

    local_model_dir = Path(os.getenv("LOCAL_MODEL_DIR", "/job/model"))
    local_embedding_dir = Path(os.getenv("LOCAL_EMBEDDINGS_DIR", "/job/embeddings"))

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

    if args.gold_doc_pages:
        if args.qid is None:
            raise ValueError("--gold-doc-pages requires --qid")
        seen_doc_ids: set[str] = set()
        expanded_page_uids = []
        for obj in gold_row.get("supporting_context", []):
            doc_id = obj["doc_id"]
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            page_images = dataset.get_images_from_doc_id(doc_id)
            for page_idx in range(len(page_images)):
                expanded_page_uids.append(f"{doc_id}_page{page_idx}")
        args.page_uid = expanded_page_uids

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
            query_raw_tokens=query_meta["raw_tokens"],
            index=index,
            token2pageuid=token2pageuid,
            token2localidx=token2localidx,
            all_token_embeddings_np=all_token_embeddings_np,
            n_retrieval_pages=args.n_retrieval_pages,
            ignore_pad_scores_in_final_ranking=args.ignore_pad_scores_in_final_ranking,
        )

    plot_rank_start = max(1, args.plot_rank_start)
    skipped_explicit_page_uids: list[str] = []
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
        selected_pages, skipped_explicit_page_uids = build_selected_pages(
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
        "ignore_pad_scores_in_final_ranking": args.ignore_pad_scores_in_final_ranking,
        "nonspatial_token_position": args.nonspatial_token_position,
        "splice_patch_labels_jsonl": args.splice_patch_labels_jsonl,
        "splice_query_token_labels": args.splice_query_token_labels,
        "plot_rank_start": plot_rank_start if not args.page_uid else None,
        "plot_rank_count": len(selected_pages),
        "explicit_page_uids": args.page_uid,
        "skipped_explicit_page_uids": skipped_explicit_page_uids,
        "retrieved_pages": selected_pages,
        "files": [],
    }

    if not selected_pages:
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Saved summary to {summary_path}")
        print(
            "No pages received retrieval contributions for the requested selection. "
            "See skipped_explicit_page_uids in summary.json or use --explicit-page-mode direct_page_maxsim."
        )
        return

    for selected_idx, item in enumerate(selected_pages, start=1):
        page_uid = item["page_uid"]
        page_score = item["final_page_score"]
        rank = item["rank"]
        doc_id, page_idx = page_uid_to_doc_page(page_uid)

        page_emb = docid2embs[doc_id][page_idx].view(-1, 128).float().numpy()
        if args.page_uid and args.explicit_page_mode == "direct_page_maxsim":
            _score_matrix, contributing_cells, page_score = compute_direct_page_maxsim(
                page_emb=page_emb,
                query_emb=query_emb,
            )
            item["final_page_score"] = page_score
        else:
            contributing_cells = {}
            for query_token_idx, details in enumerate(contribution_output["query_token_page_details"]):
                page_details = details.get(page_uid)
                if page_details is not None:
                    contributing_cells[query_token_idx] = page_details

        query_axis_classes = load_splice_query_axis_classes(
            query_labels_path=args.splice_query_token_labels,
            qid=args.qid,
            query_token_labels=query_token_labels,
        )
        _tmp_extra_tokens, tmp_grid_side, _tmp_patch_records, _tmp_non_spatial = collect_spatial_patch_records(
            page_token_count=page_emb.shape[0],
            contributing_cells={},
            query_token_labels=query_token_labels,
            nonspatial_token_position=args.nonspatial_token_position,
        )
        patch_axis_classes = load_splice_patch_axis_classes(
            labels_jsonl=args.splice_patch_labels_jsonl,
            page_uid=page_uid,
            n_spatial_patches=tmp_grid_side * tmp_grid_side,
        )
        display_query_axis_classes = query_axis_classes if any(x != "unknown" for x in query_axis_classes) else None
        display_patch_axis_classes = patch_axis_classes if any(x != "unknown" for x in patch_axis_classes) else None

        display_rank = f"rank={rank}" if rank is not None else f"explicit={selected_idx}"
        page_title = f"{display_rank} {page_uid}"

        svg_text, dots, non_spatial_items, extra_tokens, grid_side = build_dot_matrix_svg(
            query_token_labels=query_token_labels,
            page_uid=page_title,
            page_score=page_score,
            page_token_count=page_emb.shape[0],
            contributing_cells=contributing_cells,
            query_token_filter=args.query_token_filter,
            nonspatial_token_position=args.nonspatial_token_position,
            cell_width=args.cell_width,
            cell_height=args.cell_height,
            patch_tick_step=args.patch_tick_step,
            dot_radius=args.dot_radius,
            query_axis_classes=display_query_axis_classes,
            patch_axis_classes=display_patch_axis_classes,
        )

        stem_prefix = f"rank_{rank:04d}" if rank is not None else f"explicit_{selected_idx:04d}"
        stem = f"{stem_prefix}_{page_uid}"
        svg_path = output_dir / f"{stem}_patch_dots.svg"
        json_path = output_dir / f"{stem}_patch_dots.json"
        svg_path.write_text(svg_text, encoding="utf-8")

        payload = make_page_payload(
            qid=args.qid,
            query=query,
            page_uid=page_uid,
            rank=rank,
            final_page_score=page_score,
            contribution_mode=(
                "direct_page_maxsim"
                if args.page_uid and args.explicit_page_mode == "direct_page_maxsim"
                else "retrieved_contrib"
            ),
            query_token_filter=args.query_token_filter,
            ignore_pad_scores_in_final_ranking=args.ignore_pad_scores_in_final_ranking,
            nonspatial_token_position=args.nonspatial_token_position,
            query_token_labels=query_token_labels,
            page_token_count=page_emb.shape[0],
            grid_side=grid_side,
            extra_tokens=extra_tokens,
            dots=dots,
            non_spatial_items=non_spatial_items,
            query_axis_classes=query_axis_classes,
            patch_axis_classes=patch_axis_classes,
        )
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        summary["files"].append(
            {
                "rank": rank,
                "selected_index": selected_idx,
                "page_uid": page_uid,
                "svg": str(svg_path),
                "json": str(json_path),
                "n_dots": len(dots),
                "n_non_spatial_contributors": len(non_spatial_items),
            }
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved summary to {summary_path}")
    for file_info in summary["files"]:
        label = (
            f"rank {file_info['rank']}"
            if file_info["rank"] is not None
            else f"explicit page {file_info['selected_index']}"
        )
        print(f"Saved {label} dot-grid plot to {file_info['svg']}")


if __name__ == "__main__":
    main()
