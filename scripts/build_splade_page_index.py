#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a SPLADE-style sparse page index from exported M3DocVQA page text."
        )
    )
    parser.add_argument("--page-text-jsonl", required=True)
    parser.add_argument(
        "--model-name-or-path",
        default="naver/splade-cocondenser-ensembledistil",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--topk-terms",
        type=int,
        default=128,
        help="Keep at most this many sparse terms per page. Default: 128.",
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.0,
        help="Drop sparse terms below this weight after pooling. Default: 0.0.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to run the model on. Default: auto",
    )
    parser.add_argument(
        "--require-nonempty-text",
        action="store_true",
        help="Fail before model loading if every page has empty text.",
    )
    parser.add_argument("--max-pages", type=int, default=0)
    parser.add_argument("--output-index-pt", required=True)
    parser.add_argument("--output-summary-json", required=True)
    return parser.parse_args()


def load_page_rows(path: Path, max_pages: int) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_pages > 0 and len(rows) >= max_pages:
                break
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def resolve_device(raw: str) -> torch.device:
    if raw != "auto":
        return torch.device(raw)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def splade_pool(
    *,
    logits: torch.Tensor,
    attention_mask: torch.Tensor,
    special_token_ids: list[int],
) -> torch.Tensor:
    values = torch.log1p(torch.relu(logits))
    values = values * attention_mask.unsqueeze(-1)
    pooled = values.max(dim=1).values
    if special_token_ids:
        pooled[:, special_token_ids] = 0.0
    return pooled


def prune_sparse_vector(
    *,
    weights: torch.Tensor,
    topk_terms: int,
    min_weight: float,
) -> tuple[list[int], list[float]]:
    nonzero_mask = weights > float(min_weight)
    if not bool(nonzero_mask.any()):
        return [], []
    active_ids = torch.nonzero(nonzero_mask, as_tuple=False).squeeze(-1)
    active_weights = weights[active_ids]
    if topk_terms > 0 and active_ids.numel() > topk_terms:
        top_values, top_indices = torch.topk(active_weights, k=topk_terms)
        active_ids = active_ids[top_indices]
        active_weights = top_values
    order = torch.argsort(active_weights, descending=True)
    active_ids = active_ids[order]
    active_weights = active_weights[order]
    return active_ids.tolist(), [float(value) for value in active_weights.tolist()]


def main() -> None:
    args = parse_args()

    page_rows = load_page_rows(Path(args.page_text_jsonl), int(args.max_pages))
    texts = [str(row.get("text", "") or "") for row in page_rows]
    nonempty_text_page_count = sum(1 for text in texts if text.strip())
    if args.require_nonempty_text and nonempty_text_page_count == 0:
        raise ValueError(
            "No non-empty page text rows found in "
            f"{args.page_text_jsonl}. Re-run text export with OCR/PDF text or "
            "disable --require-nonempty-text only for an explicit empty-text ablation."
        )
    device = resolve_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)
    model.eval()
    model.to(device)

    page_uids: list[str] = []
    doc_ids: list[str] = []
    page_indices: list[int] = []
    offsets: list[int] = [0]
    flat_term_ids: list[int] = []
    flat_term_weights: list[float] = []
    nnz_counts: list[int] = []

    special_token_ids = list(getattr(tokenizer, "all_special_ids", []) or [])
    with torch.inference_mode():
        for start in tqdm(range(0, len(page_rows), int(args.batch_size)), desc="encode_pages"):
            batch_rows = page_rows[start : start + int(args.batch_size)]
            batch_texts = texts[start : start + int(args.batch_size)]
            batch = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=int(args.max_length),
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            pooled = splade_pool(
                logits=outputs.logits,
                attention_mask=batch["attention_mask"],
                special_token_ids=special_token_ids,
            ).cpu()

            for row, page_weights in zip(batch_rows, pooled):
                term_ids, term_weights = prune_sparse_vector(
                    weights=page_weights,
                    topk_terms=int(args.topk_terms),
                    min_weight=float(args.min_weight),
                )
                page_uids.append(str(row["page_uid"]))
                doc_ids.append(str(row["doc_id"]))
                page_indices.append(int(row["page_idx"]))
                flat_term_ids.extend(term_ids)
                flat_term_weights.extend(term_weights)
                offsets.append(len(flat_term_ids))
                nnz_counts.append(len(term_ids))

    index_payload = {
        "format": "splade_page_index_v1",
        "model_name_or_path": args.model_name_or_path,
        "page_text_jsonl": args.page_text_jsonl,
        "max_length": int(args.max_length),
        "topk_terms": int(args.topk_terms),
        "min_weight": float(args.min_weight),
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0) or 0),
        "page_uids": page_uids,
        "doc_ids": doc_ids,
        "page_indices": torch.tensor(page_indices, dtype=torch.int32),
        "offsets": torch.tensor(offsets, dtype=torch.int64),
        "term_ids": torch.tensor(flat_term_ids, dtype=torch.int32),
        "term_weights": torch.tensor(flat_term_weights, dtype=torch.float32),
    }

    output_index_pt = Path(args.output_index_pt)
    output_index_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(index_payload, output_index_pt)

    summary = {
        "format": index_payload["format"],
        "model_name_or_path": args.model_name_or_path,
        "page_text_jsonl": args.page_text_jsonl,
        "page_count": len(page_uids),
        "doc_count": len(set(doc_ids)),
        "sample_doc_ids": sorted(set(doc_ids))[:10],
        "nonempty_text_page_count": nonempty_text_page_count,
        "empty_text_page_count": len(page_rows) - nonempty_text_page_count,
        "max_pages": int(args.max_pages),
        "batch_size": int(args.batch_size),
        "max_length": int(args.max_length),
        "topk_terms": int(args.topk_terms),
        "min_weight": float(args.min_weight),
        "device": str(device),
        "vocab_size": int(index_payload["vocab_size"]),
        "stored_posting_count": len(flat_term_ids),
        "mean_terms_per_page": (sum(nnz_counts) / len(nnz_counts)) if nnz_counts else None,
        "max_terms_per_page": max(nnz_counts) if nnz_counts else 0,
        "zero_term_page_count": sum(count == 0 for count in nnz_counts),
    }
    output_summary_json = Path(args.output_summary_json)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"saved_index: {output_index_pt}")
    print(f"saved_summary: {output_summary_json}")
    print(f"page_count: {summary['page_count']}")
    print(f"stored_posting_count: {summary['stored_posting_count']}")
    print(f"mean_terms_per_page: {summary['mean_terms_per_page']}")
    print(f"zero_term_page_count: {summary['zero_term_page_count']}")


if __name__ == "__main__":
    main()
