#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import torch
from accelerate import Accelerator

from m3docrag.datasets.m3_docvqa import M3DocVQADataset, evaluate_prediction_file
from m3docrag.utils.distributed import supports_flash_attention
from m3docrag.utils.paths import LOCAL_MODEL_DIR
from m3docrag.utils.prompts import short_answer_template
from m3docrag.vqa import VQAModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run baseline M3DocRAG VQA over externally supplied page_retrieval_results. "
            "This keeps the QA prompt/model behavior aligned with examples/run_rag_m3docvqa.py."
        )
    )
    parser.add_argument("--prediction-json", required=True, help="External retrieval prediction JSON.")
    parser.add_argument("--gold", required=True, help="MMQA_<split>.jsonl used for qid order and evaluation.")
    parser.add_argument("--data-name", default="m3-docvqa")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--model-name-or-path", default="Qwen2-VL-7B-Instruct")
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument(
        "--qa-top-pages",
        type=int,
        default=4,
        help="How many retrieved page rows to send into QA. Default matches common baseline runs.",
    )
    parser.add_argument(
        "--question-type-filter",
        default="",
        help="Optional MMQA metadata.type filter, e.g. ImageListQ.",
    )
    parser.add_argument(
        "--qid",
        dest="qids",
        action="append",
        default=[],
        help="Restrict to one or more qids; pass multiple times.",
    )
    parser.add_argument("--limit", type=int, help="Optional max number of qids to process after filtering.")
    parser.add_argument(
        "--eval-num-shards",
        "--eval_num_shards",
        dest="eval_num_shards",
        type=int,
        default=1,
        help="Number of modulo shards to split the selected qids across.",
    )
    parser.add_argument(
        "--eval-shard-id",
        "--eval_shard_id",
        dest="eval_shard_id",
        type=int,
        default=0,
        help="Modulo shard id to process when --eval-num-shards > 1.",
    )
    parser.add_argument(
        "--doc-image-cache-size",
        type=int,
        default=16,
        help="Number of document image lists to memoize. Helps avoid repeated PDF rendering.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=25,
        help="Flush partial predictions every N newly completed qids. Default: 25.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output prediction JSON and skip completed qids.",
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="Run baseline EM/F1 + retrieval evaluation after writing predictions.",
    )
    parser.add_argument("--output-prediction-json", required=True)
    parser.add_argument(
        "--output-eval-json",
        help="Optional evaluation JSON path. Defaults to <output-prediction-json>.eval.json when --run-eval is set.",
    )
    return parser.parse_args()


def load_prediction(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "predictions" in payload and isinstance(
        payload["predictions"], (dict, list)
    ):
        payload = payload["predictions"]

    rows_by_qid: dict[str, dict] = {}
    if isinstance(payload, list):
        iterable = enumerate(payload)
    elif isinstance(payload, dict):
        iterable = payload.items()
    else:
        raise TypeError(f"Prediction JSON must be a list or object of prediction rows: {path}")

    for raw_key, row in iterable:
        if not isinstance(row, dict):
            raise TypeError(f"Prediction row must be an object: {path} key={raw_key!r}")
        qid = str(row.get("qid", "")).strip() or str(raw_key).strip()
        if not qid:
            raise ValueError(f"Prediction row is missing qid and key is empty: {path} key={raw_key!r}")
        if qid in rows_by_qid:
            raise ValueError(f"Duplicate qid after normalization: {qid} ({path})")
        rows_by_qid[qid] = row
    return rows_by_qid


def load_gold_rows(path: Path, question_type_filter: str = "") -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if question_type_filter:
                row_type = str(row.get("metadata", {}).get("type", "")).strip()
                if row_type != question_type_filter:
                    continue
            rows.append(row)
    return rows


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


def resolve_model_path(model_name_or_path: str) -> Path:
    candidate = Path(model_name_or_path)
    if candidate.exists():
        return candidate
    local_candidate = Path(LOCAL_MODEL_DIR) / model_name_or_path
    if local_candidate.exists():
        return local_candidate
    return candidate


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


def ranked_docs_from_rows(rows: list[list[object]], top_docs: int = 10) -> list[str]:
    ranked: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, list) or not row:
            continue
        doc_id = str(row[0]).strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        ranked.append(doc_id)
        if len(ranked) >= top_docs:
            break
    return ranked


def normalize_retrieval_rows(rows: list[object], qa_top_pages: int) -> tuple[list[list[object]], list[list[object]]]:
    source_rows: list[list[object]] = []
    for item in rows:
        if not isinstance(item, list) or len(item) < 3:
            continue
        source_rows.append([str(item[0]), int(item[1]), float(item[2])])
    selected_rows = source_rows[: max(int(qa_top_pages), 0)]
    return source_rows, selected_rows


def save_prediction(path: Path, payload: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    accelerator = Accelerator()
    if accelerator.num_processes != 1:
        raise NotImplementedError(
            "run_m3docvqa_external_retrieval_qa.py currently supports single-process execution only."
        )

    prediction_rows = load_prediction(Path(args.prediction_json))
    gold_rows = load_gold_rows(Path(args.gold), args.question_type_filter)
    gold_by_qid = {str(row["qid"]).strip(): row for row in gold_rows}

    qids = [qid for qid in gold_by_qid if qid in prediction_rows]
    if args.qids:
        requested = {str(qid).strip() for qid in args.qids if str(qid).strip()}
        qids = [qid for qid in qids if qid in requested]
    if args.limit is not None:
        qids = qids[: int(args.limit)]
    if args.eval_num_shards < 1:
        raise ValueError(f"eval_num_shards must be >= 1, got {args.eval_num_shards}")
    if args.eval_shard_id < 0 or args.eval_shard_id >= args.eval_num_shards:
        raise ValueError(
            f"eval_shard_id must be in [0, {args.eval_num_shards}), got {args.eval_shard_id}"
        )
    selected_qid_count_before_shard = len(qids)
    if args.eval_num_shards > 1:
        qids = [
            qid
            for idx, qid in enumerate(qids)
            if idx % int(args.eval_num_shards) == int(args.eval_shard_id)
        ]
    if not qids:
        raise ValueError("No qids remain after intersecting prediction rows with gold/filter.")

    output_prediction_path = Path(args.output_prediction_json)
    completed: dict[str, dict] = {}
    if args.resume and output_prediction_path.exists():
        completed = load_prediction(output_prediction_path)
        qids = [qid for qid in qids if qid not in completed]

    model_path = resolve_model_path(args.model_name_or_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not resolve model path for {args.model_name_or_path}. Tried {model_path}"
        )

    dataset = M3DocVQADataset(make_dataset_args(args))

    use_flash_attn = torch.cuda.is_available() and supports_flash_attention()
    vqa_model = VQAModel(
        model_name_or_path=model_path,
        model_type=infer_vqa_model_type(args.model_name_or_path),
        bits=args.bits,
        use_flash_attn=use_flash_attn,
        attn_implementation="flash_attention_2" if use_flash_attn else "eager",
    )
    vqa_model.model = accelerator.prepare(vqa_model.model)

    @lru_cache(maxsize=max(int(args.doc_image_cache_size), 1))
    def cached_doc_images(doc_id: str):
        return tuple(dataset.get_images_from_doc_id(doc_id))

    print(f"qid_count_total={len(gold_by_qid)}")
    print(f"qid_count_input_pred={len(prediction_rows)}")
    print(f"qid_count_selected_before_shard={selected_qid_count_before_shard}")
    print(f"eval_num_shards={args.eval_num_shards}")
    print(f"eval_shard_id={args.eval_shard_id}")
    print(f"qid_count_selected={len(qids) + len(completed) if args.resume else len(qids)}")
    print(f"qid_count_pending={len(qids)}")
    print(f"qa_top_pages={args.qa_top_pages}")
    print(f"model_name_or_path={model_path}")
    print(f"question_type_filter={args.question_type_filter or 'ALL'}")

    pending_since_flush = 0
    for offset, qid in enumerate(qids, start=1):
        gold_row = gold_by_qid[qid]
        source_row = prediction_rows[qid]
        question = str(gold_row.get("question", "")).strip() or str(source_row.get("question", "")).strip()
        source_rows, selected_rows = normalize_retrieval_rows(
            source_row.get("page_retrieval_results", []),
            args.qa_top_pages,
        )

        start = time.perf_counter()
        if selected_rows:
            images = []
            for doc_id, page_idx, _score in selected_rows:
                page_images = cached_doc_images(doc_id)
                if page_idx < 0 or page_idx >= len(page_images):
                    raise IndexError(
                        f"Page index out of range for {doc_id}: {page_idx} not in [0, {len(page_images) - 1}]"
                    )
                images.append(page_images[page_idx])

            text_input = question if "florence" in args.model_name_or_path.lower() else short_answer_template.substitute({"question": question})
            with torch.no_grad():
                pred_answer = vqa_model.generate(images=images, question=text_input)
        else:
            pred_answer = ""
        time_qa = time.perf_counter() - start

        completed[qid] = {
            "qid": qid,
            "question": question,
            "pred_answer": pred_answer,
            "page_retrieval_results": selected_rows,
            "selected_page_retrieval_results": selected_rows,
            "source_page_retrieval_results": source_rows,
            "top_retrieved_docs": ranked_docs_from_rows(selected_rows),
            "source_top_retrieved_docs": source_row.get("top_retrieved_docs", ranked_docs_from_rows(source_rows)),
            "time_retrieval": source_row.get("time_retrieval"),
            "time_qa": time_qa,
        }

        pending_since_flush += 1
        if pending_since_flush >= max(int(args.save_every), 1):
            save_prediction(output_prediction_path, completed)
            print(f"saved_partial_prediction={output_prediction_path}")
            print(f"completed_qids={len(completed)}")
            pending_since_flush = 0

        if offset % 10 == 0 or offset == len(qids):
            print(f"processed_qids={offset}/{len(qids)}")

    save_prediction(output_prediction_path, completed)
    print(f"saved_prediction={output_prediction_path}")

    if args.run_eval:
        eval_scores = evaluate_prediction_file(completed, gold_path=args.gold)
        eval_path = (
            Path(args.output_eval_json)
            if args.output_eval_json
            else output_prediction_path.with_suffix(output_prediction_path.suffix + ".eval.json")
        )
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_path.write_text(json.dumps(eval_scores, indent=2) + "\n", encoding="utf-8")
        print(f"saved_eval={eval_path}")


if __name__ == "__main__":
    main()
