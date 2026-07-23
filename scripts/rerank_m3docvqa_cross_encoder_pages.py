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
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import train_content_aware_pseudo_page_reranker as ca


DEFAULT_RECALL_KS = [1, 2, 4, 5, 10, 20, 50, 100]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rerank M3DocVQA page candidates with a pretrained text cross-encoder "
            "such as BGE-Reranker or monoBERT. The script reranks the top-N pages "
            "from an existing page prediction file and appends the remaining base "
            "candidates unchanged."
        )
    )
    parser.add_argument("--gold", required=True, help="MMQA/M3DocVQA gold JSONL; used for questions and evaluation.")
    parser.add_argument("--base-pred", required=True, help="Base page retrieval prediction JSON.")
    parser.add_argument("--page-text-jsonl", required=True, help="Exported page text JSONL.")
    parser.add_argument("--model-name-or-path", default="BAAI/bge-reranker-base")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-page-chars", type=int, default=6000)
    parser.add_argument("--candidate-top-k", type=int, default=1000)
    parser.add_argument("--rerank-top-k", type=int, default=100)
    parser.add_argument(
        "--blend-alpha",
        type=float,
        default=1.0,
        help="Final score = alpha * normalized cross-encoder score + (1-alpha) * normalized base score.",
    )
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


def resolve_device(raw: str) -> torch.device:
    if raw != "auto":
        return torch.device(raw)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_page_texts(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in ca.read_jsonl(path):
        doc_id = str(row.get("doc_id", "")).strip()
        if not doc_id:
            uid = str(row.get("page_uid", ""))
            if "_page" in uid:
                doc_id = uid.rsplit("_page", 1)[0]
        if not doc_id:
            continue
        page_idx = ca.parse_page_idx(row)
        uid = str(row.get("page_uid") or ca.page_uid(doc_id, page_idx))
        out[uid] = ca.page_text(row)
    return out


def truncate_page_text(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0]


def normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        return [1.0 for _ in values]
    return [(float(value) - lo) / (hi - lo) for value in values]


def raw_with_score(record: dict[str, Any], score: float) -> Any:
    raw = record.get("raw")
    if isinstance(raw, (list, tuple)):
        values = list(raw)
        while len(values) < 3:
            values.append(0.0)
        values[2] = float(score)
        return values
    if isinstance(raw, dict):
        values = dict(raw)
        values["score"] = float(score)
        values["cross_encoder_score"] = float(record.get("cross_encoder_score", score))
        values["rerank_score"] = float(score)
        return values
    return [record["doc_id"], int(record["page_idx"]), float(score)]


def score_pairs(
    *,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    questions: list[str],
    texts: list[str],
    max_length: int,
) -> list[float]:
    encoded = tokenizer(
        questions,
        texts,
        padding=True,
        truncation=True,
        max_length=int(max_length),
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        logits = model(**encoded).logits.detach().float().cpu()
    if logits.ndim == 1:
        return [float(value) for value in logits.tolist()]
    if logits.shape[-1] == 1:
        return [float(value) for value in logits[:, 0].tolist()]
    return [float(value) for value in logits[:, -1].tolist()]


def write_prediction(path: Path, payload: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


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
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, **model_kwargs)
    model.to(device)
    model.eval()

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

        ce_scores: list[float] = []
        for start in range(0, len(candidates), int(args.batch_size)):
            batch = candidates[start : start + int(args.batch_size)]
            batch_questions = [question for _ in batch]
            batch_texts = []
            for record in batch:
                text = truncate_page_text(page_texts.get(str(record["uid"]), ""), int(args.max_page_chars))
                if not text:
                    missing_text_count += 1
                batch_texts.append(text)
            ce_scores.extend(
                score_pairs(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    questions=batch_questions,
                    texts=batch_texts,
                    max_length=int(args.max_length),
                )
            )

        ce_norm = normalize(ce_scores)
        base_norm_map = ca.normalize_scores(candidates)
        for record, raw_score, norm_score in zip(candidates, ce_scores, ce_norm):
            base_norm = float(base_norm_map.get(str(record["uid"]), 0.0))
            final_score = float(args.blend_alpha) * float(norm_score) + (1.0 - float(args.blend_alpha)) * base_norm
            record["cross_encoder_score"] = float(raw_score)
            record["rerank_score"] = float(final_score)

        reranked_top = sorted(
            candidates,
            key=lambda row: (-float(row.get("rerank_score", 0.0)), int(row["base_rank"]), str(row["uid"])),
        )
        reranked_uids = {str(row["uid"]) for row in reranked_top}
        output_rows = [raw_with_score(row, float(row.get("rerank_score", 0.0))) for row in reranked_top]
        output_rows.extend(
            record["raw"]
            for record in records
            if str(record["uid"]) not in reranked_uids
        )

        out_row = dict(base_row)
        out_row["page_retrieval_results"] = output_rows
        out_row["reranker_metadata"] = {
            **(out_row.get("reranker_metadata", {}) if isinstance(out_row.get("reranker_metadata"), dict) else {}),
            "standard_cross_encoder_page_reranker": {
                "model_name_or_path": str(args.model_name_or_path),
                "candidate_top_k": int(args.candidate_top_k),
                "rerank_top_k": int(args.rerank_top_k),
                "blend_alpha": float(args.blend_alpha),
                "max_length": int(args.max_length),
                "max_page_chars": int(args.max_page_chars),
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
                        "cross_encoder_score": float(record.get("cross_encoder_score", 0.0)),
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
        ca.evaluate_run(label="cross_encoder_reranker", pred=output_pred, gold=gold, recall_ks=list(args.recall_k)),
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
