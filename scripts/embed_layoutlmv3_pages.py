#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


DEFAULT_TEXT_FIELDS = ["ocr_text", "markdown", "text", "page_text", "content"]
WORD_FIELDS = ["words", "ocr_words", "tokens", "ocr_tokens"]
TEXT_KEYS = ["text", "word", "token", "value"]
BOX_KEYS = ["bbox", "box", "bounding_box", "bounds"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Embed document pages with LayoutLMv3 token/layout representations. "
            "The output JSONL can be converted into page graph edges with "
            "scripts/build_page_embedding_knn_graph.py."
        )
    )
    parser.add_argument("--doc-pages-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument("--text-field", nargs="*", default=DEFAULT_TEXT_FIELDS)
    parser.add_argument("--model-name", default="microsoft/layoutlmv3-base")
    parser.add_argument("--max-pages", type=int, default=0)
    parser.add_argument("--max-words", type=int, default=384)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value.replace("\x0c", " ").replace("\u0000", " ")).strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return " ".join(part for part in (normalize_text(item) for item in value) if part)
    if isinstance(value, dict):
        return " ".join(part for part in (normalize_text(item) for item in value.values()) if part)
    return str(value).strip()


def page_uid(doc_id: str, page_idx: int) -> str:
    return f"{doc_id}_page{int(page_idx)}"


def get_page_dims(row: dict[str, Any]) -> tuple[float | None, float | None]:
    width = None
    height = None
    for key in ("width", "page_width", "image_width"):
        if row.get(key) is not None:
            try:
                width = float(row[key])
                break
            except (TypeError, ValueError):
                pass
    for key in ("height", "page_height", "image_height"):
        if row.get(key) is not None:
            try:
                height = float(row[key])
                break
            except (TypeError, ValueError):
                pass
    return width, height


def normalize_box(raw_box: Any, width: float | None, height: float | None) -> list[int]:
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) < 4:
        return [0, 0, 1000, 1000]
    try:
        x0, y0, x1, y1 = [float(value) for value in raw_box[:4]]
    except (TypeError, ValueError):
        return [0, 0, 1000, 1000]
    max_value = max(abs(x0), abs(y0), abs(x1), abs(y1))
    if max_value <= 1.5:
        x0, x1 = x0 * 1000.0, x1 * 1000.0
        y0, y1 = y0 * 1000.0, y1 * 1000.0
    elif max_value > 1000.0 and width and height and width > 0 and height > 0:
        x0, x1 = x0 * 1000.0 / width, x1 * 1000.0 / width
        y0, y1 = y0 * 1000.0 / height, y1 * 1000.0 / height
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    return [
        int(max(0, min(1000, round(x0)))),
        int(max(0, min(1000, round(y0)))),
        int(max(0, min(1000, round(x1)))),
        int(max(0, min(1000, round(y1)))),
    ]


def extract_word_entries(row: dict[str, Any], max_words: int) -> tuple[list[str], list[list[int]], str]:
    width, height = get_page_dims(row)
    for field in WORD_FIELDS:
        values = row.get(field)
        if not isinstance(values, list):
            continue
        words: list[str] = []
        boxes: list[list[int]] = []
        for item in values:
            if isinstance(item, dict):
                text = ""
                for key in TEXT_KEYS:
                    if item.get(key) is not None:
                        text = normalize_text(item.get(key))
                        break
                raw_box = None
                for key in BOX_KEYS:
                    if item.get(key) is not None:
                        raw_box = item.get(key)
                        break
            else:
                text = normalize_text(item)
                raw_box = None
            if not text:
                continue
            words.append(text)
            boxes.append(normalize_box(raw_box, width, height))
            if len(words) >= max_words:
                break
        if words:
            return words, boxes, f"word_field:{field}"
    return [], [], ""


def fallback_words(row: dict[str, Any], text_fields: list[str], max_words: int) -> tuple[list[str], list[list[int]], str]:
    parts = []
    for field in text_fields:
        text = normalize_text(row.get(field))
        if text:
            parts.append(text)
    text = " ".join(parts)
    words = re.findall(r"\S+", text)[:max_words]
    boxes = [[0, 0, 1000, 1000] for _ in words]
    return words, boxes, "fallback_text"


def iter_page_inputs(path: Path, text_fields: list[str], max_words: int, max_pages: int):
    count = 0
    for row in read_jsonl(path):
        doc_id = str(row.get("doc_id", "")).strip()
        raw_page_idx = row.get("page_idx", row.get("page_id", row.get("page_number")))
        if not doc_id or raw_page_idx is None:
            continue
        try:
            page_idx = int(raw_page_idx)
        except (TypeError, ValueError):
            continue
        words, boxes, source = extract_word_entries(row, max_words)
        if not words:
            words, boxes, source = fallback_words(row, text_fields, max_words)
        if not words:
            continue
        yield {
            "page_uid": page_uid(doc_id, page_idx),
            "doc_id": doc_id,
            "page_idx": page_idx,
            "words": words,
            "boxes": boxes,
            "source": source,
        }
        count += 1
        if max_pages > 0 and count >= max_pages:
            break


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return vector
    return [value / norm for value in vector]


def main() -> None:
    args = parse_args()
    try:
        import torch
        from transformers import LayoutLMv3Model, LayoutLMv3TokenizerFast
    except ImportError as exc:
        raise RuntimeError(
            "LayoutLMv3 embedding requires torch and transformers. Install or load an env "
            "with both packages available."
        ) from exc

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(
        args.model_name,
        local_files_only=bool(args.local_files_only),
    )
    model = LayoutLMv3Model.from_pretrained(
        args.model_name,
        local_files_only=bool(args.local_files_only),
    )
    model.eval()
    model.to(device)

    embedded_count = 0
    source_counts: dict[str, int] = {}
    with Path(args.output_jsonl).open("w", encoding="utf-8") as out:
        for item in iter_page_inputs(
            Path(args.doc_pages_jsonl),
            list(args.text_field),
            max_words=max(1, int(args.max_words)),
            max_pages=max(0, int(args.max_pages)),
        ):
            encoded = tokenizer(
                item["words"],
                boxes=item["boxes"],
                truncation=True,
                padding="max_length",
                max_length=max(8, int(args.max_length)),
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                output = model(**encoded)
            hidden = output.last_hidden_state[0]
            mask = encoded["attention_mask"][0].to(hidden.dtype).unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=0) / mask.sum(dim=0).clamp_min(1.0)
            embedding = l2_normalize([float(value) for value in pooled.detach().cpu().tolist()])
            out.write(
                json.dumps(
                    {
                        "page_uid": item["page_uid"],
                        "doc_id": item["doc_id"],
                        "page_idx": item["page_idx"],
                        "embedding": embedding,
                        "encoder": "layoutlmv3",
                        "model_name": args.model_name,
                        "word_count": len(item["words"]),
                        "source": item["source"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            embedded_count += 1
            source_counts[item["source"]] = source_counts.get(item["source"], 0) + 1
            if embedded_count % 100 == 0:
                print("embedded_pages", embedded_count, flush=True)

    summary = {
        "embedded_page_count": embedded_count,
        "model_name": args.model_name,
        "device": device,
        "max_words": int(args.max_words),
        "max_length": int(args.max_length),
        "source_counts": source_counts,
    }
    Path(args.output_summary_json).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("saved_embeddings", args.output_jsonl)
    print("saved_summary", args.output_summary_json)
    for key, value in summary.items():
        print(key, value)


if __name__ == "__main__":
    main()
