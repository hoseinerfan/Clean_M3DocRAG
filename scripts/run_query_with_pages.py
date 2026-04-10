#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from accelerate import Accelerator

from m3docrag.datasets.m3_docvqa import M3DocVQADataset
from m3docrag.utils.distributed import supports_flash_attention
from m3docrag.utils.prompts import short_answer_template
from m3docrag.vqa import VQAModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Qwen2-VL (or another supported VQA model) on explicitly selected doc_id:page_idx pages."
        )
    )
    parser.add_argument("--query", required=True, help="Free-form question text")
    parser.add_argument("--data_name", default="m3-docvqa")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--model_name_or_path", default="Qwen2-VL-7B-Instruct")
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument(
        "--page",
        dest="pages",
        action="append",
        required=True,
        help="Explicit page in the form doc_id:page_idx; pass multiple times",
    )
    parser.add_argument("--output", help="Optional JSON output path")
    return parser.parse_args()


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
    raise KeyError(f"Unknown model type for {model_name_or_path}")


def make_dataset_args(cli_args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        data_name=cli_args.data_name,
        split=cli_args.split,
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
        model_name_or_path=cli_args.model_name_or_path,
        retrieval_model_name_or_path="",
        retrieval_adapter_model_name_or_path="",
        bits=cli_args.bits,
        do_image_splitting=False,
    )


def parse_page_specs(page_specs: list[str]) -> list[tuple[str, int]]:
    parsed = []
    for spec in page_specs:
        if ":" not in spec:
            raise ValueError(f"Invalid --page value: {spec}. Expected doc_id:page_idx")
        doc_id, page_idx = spec.rsplit(":", 1)
        parsed.append((doc_id, int(page_idx)))
    return parsed


def main() -> None:
    args = parse_args()

    dataset = M3DocVQADataset(make_dataset_args(args))

    use_flash_attn = torch.cuda.is_available() and supports_flash_attention()
    vqa_model = VQAModel(
        model_name_or_path=args.model_name_or_path,
        model_type=infer_vqa_model_type(args.model_name_or_path),
        bits=args.bits,
        use_flash_attn=use_flash_attn,
        attn_implementation="flash_attention_2" if use_flash_attn else "eager",
    )

    accelerator = Accelerator()
    vqa_model.model = accelerator.prepare(vqa_model.model)

    explicit_pages = parse_page_specs(args.pages)

    images = []
    resolved_pages = []
    for doc_id, page_idx in explicit_pages:
        page_images = dataset.get_images_from_doc_id(doc_id)
        if page_idx < 0 or page_idx >= len(page_images):
            raise IndexError(
                f"Page index out of range for {doc_id}: {page_idx} not in [0, {len(page_images) - 1}]"
            )
        images.append(page_images[page_idx])
        resolved_pages.append(
            {
                "doc_id": doc_id,
                "page_idx": page_idx,
                "num_pages_in_doc": len(page_images),
            }
        )

    question = short_answer_template.substitute({"question": args.query})
    pred_answer = vqa_model.generate(images=images, question=question)

    output = {
        "query": args.query,
        "data_name": args.data_name,
        "split": args.split,
        "model_name_or_path": args.model_name_or_path,
        "bits": args.bits,
        "explicit_pages": resolved_pages,
        "pred_answer": pred_answer,
    }

    text = json.dumps(output, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
        print(f"Saved output to {output_path}")
    print(text)


if __name__ == "__main__":
    main()
