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
        description="Run page-level SPLADE sparse retrieval over an exported M3DocVQA page index."
    )
    parser.add_argument("--qid-jsonl", required=True)
    parser.add_argument("--qid-field", default="qid")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--index-pt", required=True)
    parser.add_argument(
        "--model-name-or-path",
        default="naver/splade-cocondenser-ensembledistil",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--query-topk-terms", type=int, default=32)
    parser.add_argument("--query-min-weight", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-pages", type=int, default=1000)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-prediction-json", required=True)
    parser.add_argument("--output-summary-json", required=True)
    return parser.parse_args()


def load_qids(path: Path, qid_field: str) -> list[str]:
    qids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get(qid_field, "")).strip()
            if qid:
                qids.append(qid)
    if not qids:
        raise ValueError(f"No qids found in {path}")
    return qids


def load_gold_rows(path: Path, qids: set[str]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("qid", "")).strip()
            if qid and qid in qids:
                rows[qid] = row
    missing = sorted(qids - set(rows))
    if missing:
        raise KeyError(f"Missing qids in gold file: {missing[:10]}")
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


def prune_query_vector(
    *,
    weights: torch.Tensor,
    topk_terms: int,
    min_weight: float,
) -> tuple[list[int], list[float]]:
    mask = weights > float(min_weight)
    if not bool(mask.any()):
        return [], []
    ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    vals = weights[ids]
    if topk_terms > 0 and ids.numel() > topk_terms:
        top_vals, top_idx = torch.topk(vals, k=topk_terms)
        ids = ids[top_idx]
        vals = top_vals
    order = torch.argsort(vals, descending=True)
    ids = ids[order]
    vals = vals[order]
    return ids.tolist(), [float(value) for value in vals.tolist()]


def first_unique_doc_ranks(retrieval_rows: list[list[object]]) -> dict[str, int]:
    doc2rank: dict[str, int] = {}
    seen: set[str] = set()
    rank = 0
    for row in retrieval_rows:
        doc_id = str(row[0])
        if doc_id in seen:
            continue
        seen.add(doc_id)
        rank += 1
        doc2rank[doc_id] = rank
    return doc2rank


def median_or_none(values: list[int | None]) -> float | None:
    filtered = sorted(int(value) for value in values if value is not None)
    if not filtered:
        return None
    mid = len(filtered) // 2
    if len(filtered) % 2 == 1:
        return float(filtered[mid])
    return float((filtered[mid - 1] + filtered[mid]) / 2.0)


def main() -> None:
    args = parse_args()

    qids = load_qids(Path(args.qid_jsonl), args.qid_field)
    gold_rows = load_gold_rows(Path(args.gold), set(qids))
    device = resolve_device(args.device)

    index_payload = torch.load(Path(args.index_pt), map_location="cpu")
    page_uids: list[str] = list(index_payload["page_uids"])
    doc_ids: list[str] = list(index_payload["doc_ids"])
    page_indices = index_payload["page_indices"].to(torch.int64)
    offsets = index_payload["offsets"].to(torch.int64)
    term_ids = index_payload["term_ids"].to(torch.int64)
    term_weights = index_payload["term_weights"].to(torch.float32)
    page_count = len(page_uids)

    postings: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    posting_pages: dict[int, list[int]] = {}
    posting_weights: dict[int, list[float]] = {}
    for page_idx in range(page_count):
        start = int(offsets[page_idx].item())
        end = int(offsets[page_idx + 1].item())
        for term_id, weight in zip(term_ids[start:end].tolist(), term_weights[start:end].tolist()):
            posting_pages.setdefault(int(term_id), []).append(page_idx)
            posting_weights.setdefault(int(term_id), []).append(float(weight))
    for term_id, page_list in posting_pages.items():
        postings[int(term_id)] = (
            torch.tensor(page_list, dtype=torch.int64),
            torch.tensor(posting_weights[term_id], dtype=torch.float32),
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)
    model.eval()
    model.to(device)
    special_token_ids = list(getattr(tokenizer, "all_special_ids", []) or [])

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_prediction_json = Path(args.output_prediction_json)
    output_prediction_json.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json = Path(args.output_summary_json)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)

    prediction_payload: dict[str, dict] = {}
    rows: list[dict] = []

    with torch.inference_mode():
        for start in tqdm(range(0, len(qids), int(args.batch_size)), desc="encode_queries"):
            batch_qids = qids[start : start + int(args.batch_size)]
            batch_questions = [str(gold_rows[qid]["question"]) for qid in batch_qids]
            batch = tokenizer(
                batch_questions,
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

            for qid, query_weights in zip(batch_qids, pooled):
                query_term_ids, query_term_weights = prune_query_vector(
                    weights=query_weights,
                    topk_terms=int(args.query_topk_terms),
                    min_weight=float(args.query_min_weight),
                )
                scores = torch.zeros(page_count, dtype=torch.float32)
                for term_id, query_weight in zip(query_term_ids, query_term_weights):
                    posting = postings.get(int(term_id))
                    if posting is None:
                        continue
                    posting_page_ids, posting_doc_weights = posting
                    scores.index_add_(
                        0,
                        posting_page_ids,
                        posting_doc_weights * float(query_weight),
                    )

                positive_page_ids = torch.nonzero(scores > 0, as_tuple=False).squeeze(-1)
                if positive_page_ids.numel() > 0:
                    top_count = min(int(args.top_pages), int(positive_page_ids.numel()))
                    top_scores, top_pos = torch.topk(scores[positive_page_ids], k=top_count)
                    ranked_page_ids = positive_page_ids[top_pos]
                    retrieval_rows = [
                        [
                            str(doc_ids[int(page_id)]),
                            int(page_indices[int(page_id)].item()),
                            float(score),
                        ]
                        for page_id, score in zip(ranked_page_ids.tolist(), top_scores.tolist())
                    ]
                else:
                    retrieval_rows = []

                gold_doc_ids = sorted(
                    {str(item["doc_id"]).strip() for item in gold_rows[qid].get("supporting_context", [])}
                )
                gold_doc_set = set(gold_doc_ids)
                doc_rank_map = first_unique_doc_ranks(retrieval_rows)
                first_gold_doc_rank = min(
                    (doc_rank_map.get(doc_id) for doc_id in gold_doc_ids if doc_rank_map.get(doc_id) is not None),
                    default=None,
                )
                first_gold_page_rank = None
                for rank, row in enumerate(retrieval_rows, start=1):
                    if str(row[0]) in gold_doc_set:
                        first_gold_page_rank = rank
                        break

                row = {
                    "qid": qid,
                    "question": gold_rows[qid]["question"],
                    "question_type": gold_rows[qid].get("metadata", {}).get("type")
                    or gold_rows[qid].get("question_type")
                    or "UNKNOWN",
                    "retrieval_method": "splade_page",
                    "query_topk_terms": int(args.query_topk_terms),
                    "query_term_ids": query_term_ids,
                    "query_term_weights": query_term_weights,
                    "gold_doc_ids": gold_doc_ids,
                    "reranked_first_gold_doc_rank": first_gold_doc_rank,
                    "reranked_first_gold_page_rank_any_gold_doc_page": first_gold_page_rank,
                    "top_retrieved_docs": [row[0] for row in retrieval_rows[:10]],
                    "page_retrieval_results": retrieval_rows,
                }
                rows.append(row)
                prediction_payload[qid] = {
                    "pred_answer": "",
                    "page_retrieval_results": retrieval_rows,
                    "qid": qid,
                    "question": gold_rows[qid]["question"],
                    "top_retrieved_docs": row["top_retrieved_docs"],
                    "reranker_metadata": {
                        "retrieval_method": "splade_page",
                        "index_pt": args.index_pt,
                        "model_name_or_path": args.model_name_or_path,
                    },
                }

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    output_prediction_json.write_text(json.dumps(prediction_payload, indent=2) + "\n", encoding="utf-8")

    top4_doc_count = sum(
        1
        for row in rows
        if row["reranked_first_gold_doc_rank"] is not None
        and int(row["reranked_first_gold_doc_rank"]) <= 4
    )
    top20_doc_count = sum(
        1
        for row in rows
        if row["reranked_first_gold_doc_rank"] is not None
        and int(row["reranked_first_gold_doc_rank"]) <= 20
    )
    summary = {
        "retrieval_method": "splade_page",
        "qid_count": len(rows),
        "index_pt": args.index_pt,
        "model_name_or_path": args.model_name_or_path,
        "top_pages": int(args.top_pages),
        "query_topk_terms": int(args.query_topk_terms),
        "query_min_weight": float(args.query_min_weight),
        "reranked_top4_doc_count": top4_doc_count,
        "reranked_top20_doc_count": top20_doc_count,
        "reranked_doc_rank_median": median_or_none(
            [row["reranked_first_gold_doc_rank"] for row in rows]
        ),
    }
    output_summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"saved_jsonl: {output_jsonl}")
    print(f"saved_prediction: {output_prediction_json}")
    print(f"saved_summary: {output_summary_json}")
    print(f"qid_count: {len(rows)}")
    print(f"reranked_top4_doc_count: {top4_doc_count}")
    print(f"reranked_top20_doc_count: {top20_doc_count}")


if __name__ == "__main__":
    main()
