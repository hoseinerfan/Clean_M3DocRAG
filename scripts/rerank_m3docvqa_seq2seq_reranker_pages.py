#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import train_content_aware_pseudo_page_reranker as ca
from rerank_m3docvqa_cross_encoder_pages import (
    load_page_texts,
    normalize,
    raw_with_score,
    resolve_device,
    truncate_page_text,
    write_prediction,
)


DEFAULT_RECALL_KS = [1, 2, 4, 5, 10, 20, 50, 100]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rerank M3DocVQA page candidates with a pretrained seq2seq reranker "
            "such as monoT5/RankT5. The script scores the top-N pages from an "
            "existing page prediction file and appends the remaining base "
            "candidates unchanged."
        )
    )
    parser.add_argument("--gold", required=True, help="MMQA/M3DocVQA gold JSONL; used for questions and evaluation.")
    parser.add_argument("--base-pred", required=True, help="Base page retrieval prediction JSON.")
    parser.add_argument("--page-text-jsonl", required=True, help="Exported page text JSONL.")
    parser.add_argument("--model-name-or-path", default="castorini/monot5-base-msmarco-10k")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-page-chars", type=int, default=4000)
    parser.add_argument("--candidate-top-k", type=int, default=1000)
    parser.add_argument("--rerank-top-k", type=int, default=100)
    parser.add_argument(
        "--blend-alpha",
        type=float,
        default=1.0,
        help="Final score = alpha * normalized seq2seq score + (1-alpha) * normalized base score.",
    )
    parser.add_argument(
        "--prompt-template",
        default="Query: {question} Document: {document} Relevant:",
        help="Prompt with {question} and {document} placeholders.",
    )
    parser.add_argument("--positive-token", default="true")
    parser.add_argument("--negative-token", default="false")
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--qid-limit", type=int, default=0)
    parser.add_argument("--recall-k", type=int, nargs="+", default=DEFAULT_RECALL_KS)
    parser.add_argument("--output-prediction-json", required=True)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument("--output-table-md", default="")
    parser.add_argument("--output-score-jsonl", default="")
    return parser.parse_args()


def first_token_id(tokenizer: Any, text: str) -> int:
    token_ids = tokenizer.encode(str(text), add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Could not tokenize scoring label: {text!r}")
    return int(token_ids[0])


def decoder_start_id(model: Any, tokenizer: Any) -> int:
    for value in (
        getattr(model.config, "decoder_start_token_id", None),
        getattr(model.config, "bos_token_id", None),
        getattr(model.config, "pad_token_id", None),
        getattr(tokenizer, "pad_token_id", None),
    ):
        if value is not None:
            return int(value)
    raise ValueError("Could not determine decoder start token id for seq2seq scoring.")


def format_prompt(template: str, question: str, document: str) -> str:
    return template.format(question=str(question or "").strip(), document=str(document or "").strip())


def score_prompts(
    *,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompts: list[str],
    max_length: int,
    positive_token_id: int,
    negative_token_id: int,
    decoder_start_token_id: int,
) -> list[float]:
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=int(max_length),
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    decoder_input_ids = torch.full(
        (len(prompts), 1),
        int(decoder_start_token_id),
        dtype=torch.long,
        device=device,
    )
    with torch.inference_mode():
        logits = model(**encoded, decoder_input_ids=decoder_input_ids).logits[:, 0, :].detach().float()
        pair_logits = logits[:, [int(positive_token_id), int(negative_token_id)]]
        log_probs = torch.log_softmax(pair_logits, dim=-1)
        scores = log_probs[:, 0] - log_probs[:, 1]
    return [float(value) for value in scores.cpu().tolist()]


def main() -> None:
    args = parse_args()
    if not 0.0 <= float(args.blend_alpha) <= 1.0:
        raise ValueError("--blend-alpha must be in [0, 1].")
    if int(args.rerank_top_k) <= 0:
        raise ValueError("--rerank-top-k must be positive.")

    gold = ca.load_gold(Path(args.gold))
    base_pred = ca.load_prediction(Path(args.base_pred))
    page_texts = load_page_texts(Path(args.page_text_jsonl))
    output_pred_path = Path(args.output_prediction_json)

    output_pred: dict[str, dict[str, Any]] = {}
    if bool(args.resume) and output_pred_path.exists():
        output_pred = ca.load_prediction(output_pred_path)
        print(f"resume_existing_qids={len(output_pred)}")

    device = resolve_device(str(args.device))
    dtype = torch.float16 if bool(args.use_fp16) and device.type == "cuda" else None
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=bool(args.local_files_only),
    )
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": bool(args.trust_remote_code),
        "local_files_only": bool(args.local_files_only),
    }
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    model.to(device)
    model.eval()

    positive_token_id = first_token_id(tokenizer, str(args.positive_token))
    negative_token_id = first_token_id(tokenizer, str(args.negative_token))
    start_token_id = decoder_start_id(model, tokenizer)
    print(f"positive_token={args.positive_token!r} positive_token_id={positive_token_id}")
    print(f"negative_token={args.negative_token!r} negative_token_id={negative_token_id}")
    print(f"decoder_start_token_id={start_token_id}")

    qids = list(base_pred.keys())
    if int(args.qid_limit) > 0:
        qids = qids[: int(args.qid_limit)]

    score_rows: list[dict[str, Any]] = []
    missing_text_count = 0
    processed = 0
    start_time = time.time()

    for qid in tqdm(qids, desc="rerank_qids"):
        if qid in output_pred:
            continue
        base_row = base_pred[qid]
        records = ca.ranked_page_records(base_row, int(args.candidate_top_k))
        if not records:
            output_pred[qid] = dict(base_row)
            continue

        gold_row = gold.get(qid, {})
        question = str(gold_row.get("question") or base_row.get("question") or "")
        rerank_count = min(int(args.rerank_top_k), len(records))
        candidates = records[:rerank_count]

        seq2seq_scores: list[float] = []
        for start in range(0, len(candidates), int(args.batch_size)):
            batch = candidates[start : start + int(args.batch_size)]
            prompts = []
            for record in batch:
                text = truncate_page_text(page_texts.get(str(record["uid"]), ""), int(args.max_page_chars))
                if not text:
                    missing_text_count += 1
                prompts.append(format_prompt(str(args.prompt_template), question, text))
            seq2seq_scores.extend(
                score_prompts(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    prompts=prompts,
                    max_length=int(args.max_length),
                    positive_token_id=positive_token_id,
                    negative_token_id=negative_token_id,
                    decoder_start_token_id=start_token_id,
                )
            )

        seq2seq_norm = normalize(seq2seq_scores)
        base_norm_map = ca.normalize_scores(candidates)
        for record, raw_score, norm_score in zip(candidates, seq2seq_scores, seq2seq_norm):
            base_norm = float(base_norm_map.get(str(record["uid"]), 0.0))
            final_score = float(args.blend_alpha) * float(norm_score) + (1.0 - float(args.blend_alpha)) * base_norm
            record["seq2seq_score"] = float(raw_score)
            record["rerank_score"] = float(final_score)

        reranked_top = sorted(
            candidates,
            key=lambda row: (-float(row.get("rerank_score", 0.0)), int(row["base_rank"]), str(row["uid"])),
        )
        reranked_uids = {str(row["uid"]) for row in reranked_top}
        output_rows = [raw_with_score(row, float(row.get("rerank_score", 0.0))) for row in reranked_top]
        output_rows.extend(record["raw"] for record in records if str(record["uid"]) not in reranked_uids)

        out_row = dict(base_row)
        out_row["page_retrieval_results"] = output_rows
        out_row["reranker_metadata"] = {
            **(out_row.get("reranker_metadata", {}) if isinstance(out_row.get("reranker_metadata"), dict) else {}),
            "standard_seq2seq_page_reranker": {
                "model_name_or_path": str(args.model_name_or_path),
                "candidate_top_k": int(args.candidate_top_k),
                "rerank_top_k": int(args.rerank_top_k),
                "blend_alpha": float(args.blend_alpha),
                "max_length": int(args.max_length),
                "max_page_chars": int(args.max_page_chars),
                "prompt_template": str(args.prompt_template),
                "positive_token": str(args.positive_token),
                "negative_token": str(args.negative_token),
            },
        }
        output_pred[qid] = out_row

        if args.output_score_jsonl:
            for rank, record in enumerate(reranked_top, start=1):
                score_rows.append(
                    {
                        "qid": qid,
                        "page_uid": record["uid"],
                        "doc_id": record["doc_id"],
                        "page_idx": int(record["page_idx"]),
                        "base_rank": int(record["base_rank"]),
                        "rerank_rank": int(rank),
                        "seq2seq_score": float(record.get("seq2seq_score", 0.0)),
                        "rerank_score": float(record.get("rerank_score", 0.0)),
                    }
                )

        processed += 1
        if int(args.save_every) > 0 and processed % int(args.save_every) == 0:
            write_prediction(output_pred_path, output_pred)
            print(f"saved_partial_prediction={output_pred_path}")
            print(f"completed_qids={len(output_pred)}")

    write_prediction(output_pred_path, output_pred)

    if args.output_score_jsonl:
        score_path = Path(args.output_score_jsonl)
        score_path.parent.mkdir(parents=True, exist_ok=True)
        with score_path.open("w", encoding="utf-8") as handle:
            for row in score_rows:
                handle.write(json.dumps(row) + "\n")

    metrics = [
        ca.evaluate_run(label="base", pred=base_pred, gold=gold, recall_ks=list(args.recall_k)),
        ca.evaluate_run(label="seq2seq_reranker", pred=output_pred, gold=gold, recall_ks=list(args.recall_k)),
    ]
    summary = {
        "gold": args.gold,
        "base_pred": args.base_pred,
        "page_text_jsonl": args.page_text_jsonl,
        "model_name_or_path": args.model_name_or_path,
        "candidate_top_k": int(args.candidate_top_k),
        "rerank_top_k": int(args.rerank_top_k),
        "blend_alpha": float(args.blend_alpha),
        "max_length": int(args.max_length),
        "max_page_chars": int(args.max_page_chars),
        "prompt_template": str(args.prompt_template),
        "positive_token": str(args.positive_token),
        "negative_token": str(args.negative_token),
        "positive_token_id": int(positive_token_id),
        "negative_token_id": int(negative_token_id),
        "device": str(device),
        "use_fp16": bool(args.use_fp16 and device.type == "cuda"),
        "processed_qid_count": int(len(output_pred)),
        "missing_text_count": int(missing_text_count),
        "elapsed_seconds": float(time.time() - start_time),
        "metrics": metrics,
        "movement_vs_base": ca.movement_vs_base(base_pred=base_pred, candidate_pred=output_pred, gold=gold),
    }
    summary_path = Path(args.output_summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.output_table_md:
        ca.write_table(Path(args.output_table_md), metrics, list(args.recall_k))

    print(f"saved_prediction={output_pred_path}")
    print(f"saved_summary={summary_path}")
    if args.output_table_md:
        print(f"saved_table={args.output_table_md}")
    if args.output_score_jsonl:
        print(f"saved_scores={args.output_score_jsonl}")
    print(f"processed_qids={len(output_pred)}")
    print(f"missing_text_count={missing_text_count}")
    for row in metrics:
        print(row)


if __name__ == "__main__":
    main()
