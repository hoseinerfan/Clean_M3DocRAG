#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import faiss
import torch
from accelerate import Accelerator

from m3docrag.datasets.m3_docvqa import M3DocVQADataset
from m3docrag.rag import MultimodalRAGModel
from m3docrag.retrieval import ColPaliRetrievalModel
from m3docrag.utils.distributed import supports_flash_attention
from m3docrag.utils.paths import LOCAL_DATA_DIR, LOCAL_EMBEDDINGS_DIR, LOCAL_MODEL_DIR
from m3docrag.utils.prompts import short_answer_template
from m3docrag.vqa import VQAModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run M3DocRAG on a free-form question against an existing M3DocVQA-style PDF corpus."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", help="Free-form question text")
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
    parser.add_argument("--model_name_or_path", default="Qwen2-VL-7B-Instruct")
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument("--n_retrieval_pages", type=int, default=4)
    parser.add_argument(
        "--ignore-pad-scores-in-final-ranking",
        action="store_true",
        help="Keep PAD tokens in ANN search but exclude their scores from the final page-score sum used for reranking.",
    )
    parser.add_argument("--retrieval_only", action="store_true")
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


def load_query_from_qid(cli_args: argparse.Namespace) -> str:
    gold_path = Path(cli_args.gold) if cli_args.gold else (
        Path(LOCAL_DATA_DIR) / cli_args.data_name / "multimodalqa" / f"MMQA_{cli_args.split}.jsonl"
    )
    with open(gold_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj["qid"] == cli_args.qid:
                return obj["question"]
    raise KeyError(f"QID not found in gold file: {cli_args.qid}")


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
        retrieval_only=cli_args.retrieval_only,
        page_retrieval_type="logits",
        loop_unique_doc_ids=False,
        n_retrieval_pages=cli_args.n_retrieval_pages,
        faiss_index_type=cli_args.faiss_index_type,
        model_name_or_path=cli_args.model_name_or_path,
        retrieval_model_name_or_path=cli_args.retrieval_model_name_or_path,
        retrieval_adapter_model_name_or_path=cli_args.retrieval_adapter_model_name_or_path,
        bits=cli_args.bits,
        do_image_splitting=False,
    )


def build_flattened_index_inputs(docid2embs: dict[str, torch.Tensor]) -> tuple[list[str], torch.Tensor]:
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


def main() -> None:
    args = parse_args()
    if args.qid and not args.query:
        args.query = load_query_from_qid(args)

    local_model_dir = Path(LOCAL_MODEL_DIR)
    local_embedding_dir = Path(LOCAL_EMBEDDINGS_DIR)

    retrieval_backbone_dir = local_model_dir / args.retrieval_model_name_or_path
    retrieval_adapter_dir = local_model_dir / args.retrieval_adapter_model_name_or_path
    vqa_model_dir = local_model_dir / args.model_name_or_path
    index_dir = local_embedding_dir / f"{args.embedding_name}_pageindex_{args.faiss_index_type}"

    for path in [retrieval_backbone_dir, retrieval_adapter_dir, index_dir]:
        if not path.exists():
            raise FileNotFoundError(path)
    if not args.retrieval_only and not vqa_model_dir.exists():
        raise FileNotFoundError(vqa_model_dir)

    use_flash_attn = torch.cuda.is_available() and supports_flash_attention()

    retrieval_model = ColPaliRetrievalModel(
        backbone_name_or_path=retrieval_backbone_dir,
        adapter_name_or_path=retrieval_adapter_dir,
    )

    vqa_model = None
    if not args.retrieval_only:
        vqa_model = VQAModel(
            model_name_or_path=vqa_model_dir,
            model_type=infer_vqa_model_type(args.model_name_or_path),
            bits=args.bits,
            use_flash_attn=use_flash_attn,
            attn_implementation="flash_attention_2" if use_flash_attn else "eager",
        )

    accelerator = Accelerator()
    if args.retrieval_only:
        retrieval_model.model = accelerator.prepare(retrieval_model.model)
    else:
        retrieval_model.model, vqa_model.model = accelerator.prepare(
            retrieval_model.model, vqa_model.model
        )

    rag_model = MultimodalRAGModel(retrieval_model=retrieval_model, vqa_model=vqa_model)

    dataset_args = make_dataset_args(args)
    dataset = M3DocVQADataset(dataset_args)

    index = faiss.read_index(str(index_dir / "index.bin"))
    docid2embs = dataset.load_all_embeddings()
    token2pageuid, all_token_embeddings = build_flattened_index_inputs(docid2embs)
    all_token_embeddings_np = all_token_embeddings.float().numpy()

    retrieval_results = rag_model.retrieve_pages_from_docs(
        query=args.query,
        docid2embs=docid2embs,
        index=index,
        token2pageuid=token2pageuid,
        all_token_embeddings=all_token_embeddings_np,
        n_return_pages=args.n_retrieval_pages,
        ignore_pad_scores_in_final_ranking=args.ignore_pad_scores_in_final_ranking,
        show_progress=True,
    )

    output = {
        "qid": args.qid,
        "query": args.query,
        "data_name": args.data_name,
        "split": args.split,
        "embedding_name": args.embedding_name,
        "faiss_index_type": args.faiss_index_type,
        "n_retrieval_pages": args.n_retrieval_pages,
        "ignore_pad_scores_in_final_ranking": args.ignore_pad_scores_in_final_ranking,
        "retrieval_only": args.retrieval_only,
        "page_retrieval_results": retrieval_results,
    }

    if not args.retrieval_only:
        images = []
        for doc_id, page_idx, _score in retrieval_results:
            page_images = dataset.get_images_from_doc_id(doc_id)
            images.append(page_images[page_idx])

        question = short_answer_template.substitute({"question": args.query})
        output["pred_answer"] = rag_model.run_vqa(images=images, question=question)

    text = json.dumps(output, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
        print(f"Saved output to {output_path}")
    print(text)


if __name__ == "__main__":
    main()
