#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_RECALL_KS = [1, 2, 4, 5, 10, 20, 50, 100]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use a VLM as a local rank-boundary verifier. For each query, compare the current "
            "rank-hit_k page (rank 4 by default) against the boundary page (rank 5 by default), "
            "then swap the boundary page into top-k only when the VLM explicitly prefers it. "
            "Gold labels are used only for reporting."
        )
    )
    parser.add_argument("--gold", required=True, help="Gold JSONL containing qid/question/support labels.")
    parser.add_argument("--base-prediction", required=True)
    parser.add_argument("--doc-pages-jsonl", required=True, help="Converted doc_pages JSONL with image_path fields.")
    parser.add_argument("--output-prediction-json", required=True)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument("--output-case-json", default="")
    parser.add_argument("--hit-k", type=int, default=4)
    parser.add_argument("--boundary-rank", type=int, default=5)
    parser.add_argument("--recall-k", dest="recall_ks", type=int, nargs="+", default=DEFAULT_RECALL_KS)
    parser.add_argument(
        "--qid-jsonl",
        default="",
        help=(
            "Optional qid filter. Accepts JSONL rows with a qid field, a JSON list/object, "
            "or plain one-qid-per-line text."
        ),
    )
    parser.add_argument("--qid-field", default="qid")
    parser.add_argument("--max-qids", type=int, default=0)
    parser.add_argument("--sample-qids", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=13)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--boundary-doc-policy",
        choices=["any", "topk_doc", "top1_doc", "rank_hit_doc"],
        default="topk_doc",
        help=(
            "Observable doc gate for the rank-5 page. rank_hit_doc requires the boundary page "
            "to be in the same document as the rank-hit_k page."
        ),
    )
    parser.add_argument(
        "--min-boundary-doc-topk-count",
        type=int,
        default=0,
        help="Require this many current top-k pages from the boundary page document.",
    )
    parser.add_argument(
        "--max-base-margin-4-5",
        type=float,
        default=None,
        help="Optional base score margin cap: score(rank hit_k) - score(boundary rank).",
    )
    parser.add_argument(
        "--max-base-margin-ratio-4-5",
        type=float,
        default=None,
        help="Optional normalized margin cap relative to abs(score@hit_k).",
    )
    parser.add_argument(
        "--decision-mode",
        choices=["pairwise", "independent_yes_no"],
        default="pairwise",
        help=(
            "pairwise sends both images together and expects A/B/tie. "
            "independent_yes_no scores each page with a separate evidence yes/no prompt."
        ),
    )
    parser.add_argument(
        "--min-vlm-margin",
        type=float,
        default=0.5,
        help="For independent_yes_no, require boundary_score - rank_hit_score >= this margin.",
    )
    parser.add_argument(
        "--min-boundary-score",
        type=float,
        default=1.0,
        help="For independent_yes_no, minimum boundary evidence score required for acceptance.",
    )
    parser.add_argument(
        "--max-rank-hit-score",
        type=float,
        default=0.5,
        help="For independent_yes_no, maximum rank-hit page evidence score allowed for acceptance.",
    )
    parser.add_argument(
        "--accept-pairwise-tie",
        action="store_true",
        help="Treat pairwise tie/unclear as acceptance. Default rejects ties.",
    )
    parser.add_argument(
        "--vlm-model-name-or-path",
        default="Qwen2-VL-7B-Instruct",
        help="VLM path/name. Resolved directly or under LOCAL_MODEL_DIR.",
    )
    parser.add_argument(
        "--vlm-model-type",
        default="",
        help="Optional explicit VLM type. If omitted, inferred from model name.",
    )
    parser.add_argument("--vlm-bits", type=int, default=16)
    parser.add_argument(
        "--no-accelerate",
        action="store_true",
        help="Do not wrap the loaded VLM with accelerate.Accelerator().prepare.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build cases and keep base without loading/running a VLM. Useful for smoke tests.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_qid_filter(path: Path, qid_field: str) -> set[str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return set()
    qids: set[str] = set()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                value = item.get(qid_field, item.get("qid"))
            else:
                value = item
            if value is not None and str(value).strip():
                qids.add(str(value).strip())
        return qids
    if isinstance(payload, dict):
        if qid_field in payload or "qid" in payload:
            value = payload.get(qid_field, payload.get("qid"))
            if value is not None and str(value).strip():
                qids.add(str(value).strip())
            return qids
        return {str(key).strip() for key in payload if str(key).strip()}

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            item = line
        if isinstance(item, dict):
            value = item.get(qid_field, item.get("qid"))
        else:
            value = item
        if value is not None and str(value).strip():
            qids.add(str(value).strip())
    return qids


def load_prediction(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "predictions" in payload and isinstance(payload["predictions"], (dict, list)):
        payload = payload["predictions"]
    rows: dict[str, dict[str, Any]] = {}
    iterable: Any
    if isinstance(payload, dict):
        iterable = payload.items()
    elif isinstance(payload, list):
        iterable = enumerate(payload)
    else:
        raise TypeError(f"Prediction JSON must be an object or list: {path}")
    for raw_key, row in iterable:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("qid", "")).strip() or str(raw_key).strip()
        if qid:
            rows[qid] = row
    return rows


def page_uid(doc_id: Any, page_idx: Any) -> str:
    return f"{doc_id}_page{int(page_idx)}"


def parse_page_uid(uid: str) -> tuple[str, int]:
    if "_page" not in uid:
        raise ValueError(f"Invalid page uid: {uid}")
    doc_id, raw_page = uid.rsplit("_page", 1)
    return doc_id, int(raw_page)


def page_doc(uid: str) -> str:
    return parse_page_uid(uid)[0]


def item_uid(item: Any) -> str | None:
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        return None
    try:
        return page_uid(str(item[0]), int(item[1]))
    except (TypeError, ValueError):
        return None


def item_score(item: Any) -> float | None:
    if not isinstance(item, (list, tuple)) or len(item) < 3:
        return None
    try:
        return float(item[2])
    except (TypeError, ValueError):
        return None


def ranked_items(pred_row: dict[str, Any] | None, limit: int = 0) -> list[Any]:
    if pred_row is None:
        return []
    items: list[Any] = []
    seen: set[str] = set()
    for item in pred_row.get("page_retrieval_results", []):
        uid = item_uid(item)
        if not uid or uid in seen:
            continue
        seen.add(uid)
        items.append(item)
        if limit > 0 and len(items) >= limit:
            break
    return items


def ranked_pages(pred_row: dict[str, Any] | None, limit: int = 0) -> list[str]:
    pages: list[str] = []
    for item in ranked_items(pred_row, limit):
        uid = item_uid(item)
        if uid:
            pages.append(uid)
    return pages


def ranked_docs(pred_row: dict[str, Any] | None, limit: int = 0) -> list[str]:
    docs: list[str] = []
    seen: set[str] = set()
    for uid in ranked_pages(pred_row):
        doc_id = page_doc(uid)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        docs.append(doc_id)
        if limit > 0 and len(docs) >= limit:
            break
    return docs


def prediction_item_by_uid(pred_row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in ranked_items(pred_row):
        uid = item_uid(item)
        if uid and uid not in out:
            out[uid] = item
    return out


def first_rank(ranked: list[str], gold: set[str]) -> int | None:
    for idx, value in enumerate(ranked, start=1):
        if value in gold:
            return idx
    return None


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / float(len(gold))


def gold_page_uids(row: dict[str, Any]) -> set[str]:
    metadata = row.get("metadata", {})
    uids = {
        str(value).strip()
        for value in metadata.get("gold_page_uids", [])
        if str(value).strip()
    }
    for ctx in row.get("supporting_context", []):
        doc_id = str(ctx.get("doc_id", "")).strip()
        page_idx = ctx.get("page_idx", ctx.get("page_id"))
        if doc_id and page_idx is not None:
            uids.add(page_uid(doc_id, int(page_idx)))
    return uids


def gold_doc_ids(row: dict[str, Any]) -> set[str]:
    metadata = row.get("metadata", {})
    docs = {
        str(value).strip()
        for value in metadata.get("gold_doc_ids", [])
        if str(value).strip()
    }
    for ctx in row.get("supporting_context", []):
        doc_id = str(ctx.get("doc_id", "")).strip()
        if doc_id:
            docs.add(doc_id)
    return docs


def get_question(gold_row: dict[str, Any]) -> str:
    for key in ("question", "query", "text"):
        value = gold_row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def movement_for_hit(base_rank: int | None, candidate_rank: int | None, hit_k: int) -> str:
    base_hit = base_rank is not None and base_rank <= hit_k
    candidate_hit = candidate_rank is not None and candidate_rank <= hit_k
    if not base_hit and candidate_hit:
        return "recovered"
    if base_hit and not candidate_hit:
        return "lost"
    if base_rank is None and candidate_rank is None:
        return "missing_in_both"
    if base_rank is not None and candidate_rank is not None and candidate_rank < base_rank:
        return "improved_rank"
    if base_rank is not None and candidate_rank is not None and candidate_rank > base_rank:
        return "worsened_rank"
    return "unchanged"


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def resolve_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_page_image_paths(doc_pages_jsonl: Path) -> dict[str, Path]:
    base_dir = doc_pages_jsonl.parent
    mapping: dict[str, Path] = {}
    for row in read_jsonl(doc_pages_jsonl):
        uid = str(row.get("page_uid", "")).strip()
        if not uid:
            doc_id = str(row.get("doc_id", "")).strip()
            page_idx = row.get("page_idx")
            if not doc_id or page_idx is None:
                continue
            uid = page_uid(doc_id, page_idx)
        raw_path = row.get("image_path") or row.get("source_image_path")
        if raw_path:
            mapping[uid] = resolve_path(str(raw_path), base_dir)
    return mapping


def open_image(path: Path) -> Image.Image:
    image = Image.open(path)
    return image.convert("RGB")


def infer_vqa_model_type(model_name_or_path: str) -> str:
    lowered = str(model_name_or_path).lower()
    if "florence" in lowered:
        return "florence2"
    if "idefics2" in lowered:
        return "idefics2"
    if "idefics3" in lowered:
        return "idefics3"
    if "internvl2" in lowered:
        return "internvl2"
    if "qwen2" in lowered or "qwen-vl" in lowered:
        return "qwen2"
    raise KeyError(f"Unknown VLM model type for {model_name_or_path}; pass --vlm-model-type.")


def resolve_model_path(name_or_path: str) -> Path:
    from m3docrag.utils.paths import LOCAL_MODEL_DIR

    candidate = Path(name_or_path)
    if candidate.exists():
        return candidate
    local_candidate = Path(LOCAL_MODEL_DIR) / name_or_path
    if local_candidate.exists():
        return local_candidate
    return candidate


def load_vqa_model(args: argparse.Namespace) -> Any:
    from m3docrag.vqa import VQAModel

    model_type = str(args.vlm_model_type).strip() or infer_vqa_model_type(str(args.vlm_model_name_or_path))
    vqa_model = VQAModel(
        model_name_or_path=resolve_model_path(str(args.vlm_model_name_or_path)),
        model_type=model_type,
        bits=int(args.vlm_bits),
    )
    if not bool(args.no_accelerate):
        try:
            from accelerate import Accelerator

            accelerator = Accelerator()
            if hasattr(vqa_model.model, "parameters"):
                vqa_model.model = accelerator.prepare(vqa_model.model)
        except Exception as exc:
            print(f"warning: accelerate prepare skipped: {type(exc).__name__}: {exc}", file=sys.stderr)
    return vqa_model


def pairwise_prompt(question: str) -> str:
    return (
        "You are comparing two document pages for retrieval.\n"
        "The first image is Page A: the current rank-4 page.\n"
        "The second image is Page B: the candidate rank-5 page.\n"
        f"Question: {question}\n"
        "Which page contains stronger evidence needed to answer the question correctly?\n"
        "Reply with exactly one token: A, B, or tie."
    )


def single_page_prompt(question: str) -> str:
    return (
        "You are checking whether one document page contains evidence needed to answer a question.\n"
        f"Question: {question}\n"
        "Does this page contain evidence that would help answer the question correctly?\n"
        "Reply with exactly one token: yes, no, or unclear."
    )


def parse_pairwise_response(response: str) -> tuple[str, str]:
    normalized = " ".join(str(response or "").strip().lower().split())
    stripped = normalized.strip(" .,:;()[]{}")
    first = stripped.split()[0] if stripped else ""
    if first in {"a", "page-a", "page_a"} or stripped.startswith("page a"):
        return "rank_hit", normalized
    if first in {"b", "page-b", "page_b"} or stripped.startswith("page b"):
        return "boundary", normalized
    if first in {"tie", "same", "unclear", "neither", "both"}:
        return "tie", normalized
    if "page b" in stripped and "page a" not in stripped:
        return "boundary", normalized
    if "page a" in stripped and "page b" not in stripped:
        return "rank_hit", normalized
    if " b" in f" {stripped} " and " a" not in f" {stripped} ":
        return "boundary", normalized
    if " a" in f" {stripped} " and " b" not in f" {stripped} ":
        return "rank_hit", normalized
    return "tie", normalized


def parse_yes_no_response(response: str) -> tuple[float, str]:
    normalized = " ".join(str(response or "").strip().lower().split())
    if not normalized:
        return 0.5, normalized
    first = normalized.strip(" .,:;()[]{}").split()[0]
    if first == "yes":
        return 1.0, normalized
    if first == "no":
        return 0.0, normalized
    if first in {"unclear", "maybe", "partially", "possibly", "unsure"}:
        return 0.5, normalized
    if "yes" in normalized and "no" not in normalized:
        return 0.75, normalized
    if "no" in normalized and "yes" not in normalized:
        return 0.25, normalized
    return 0.5, normalized


def base_boundary_margins(
    *,
    base_row: dict[str, Any],
    rank_hit_page: str,
    boundary_page: str,
) -> tuple[float | None, float | None]:
    item_map = prediction_item_by_uid(base_row)
    rank_hit_score = item_score(item_map.get(rank_hit_page))
    boundary_score = item_score(item_map.get(boundary_page))
    margin = rank_hit_score - boundary_score if rank_hit_score is not None and boundary_score is not None else None
    ratio = margin / max(abs(rank_hit_score), 1e-12) if margin is not None and rank_hit_score is not None else None
    return margin, ratio


def build_reordered_row(
    *,
    base_row: dict[str, Any],
    rank_hit_page: str,
    boundary_page: str,
    accepted: bool,
) -> dict[str, Any]:
    output = copy.deepcopy(base_row)
    if not accepted:
        return output
    pages = ranked_pages(base_row)
    item_map = prediction_item_by_uid(base_row)
    reordered = list(pages)
    try:
        rank_hit_idx = reordered.index(rank_hit_page)
        boundary_idx = reordered.index(boundary_page)
    except ValueError:
        return output
    reordered[rank_hit_idx], reordered[boundary_idx] = reordered[boundary_idx], reordered[rank_hit_idx]
    output["page_retrieval_results"] = [copy.deepcopy(item_map[uid]) for uid in reordered if uid in item_map]
    output["top_retrieved_docs"] = ranked_docs(output, 10)
    return output


def preflight_case(
    *,
    qid: str,
    question: str,
    base_row: dict[str, Any],
    image_paths: dict[str, Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    hit_k = int(args.hit_k)
    boundary_rank = int(args.boundary_rank)
    pages = ranked_pages(base_row)
    top_pages = pages[:hit_k]
    if len(pages) < boundary_rank or len(pages) < hit_k:
        return {
            "qid": qid,
            "question": question,
            "accepted": False,
            "reason": "not_enough_pages",
            "top_pages": top_pages,
            "boundary_page": None,
        }
    rank_hit_page = pages[hit_k - 1]
    boundary_page = pages[boundary_rank - 1]
    rank_hit_doc = page_doc(rank_hit_page)
    boundary_doc = page_doc(boundary_page)
    top_docs = [page_doc(uid) for uid in top_pages]
    boundary_doc_topk_count = sum(1 for doc_id in top_docs if doc_id == boundary_doc)
    margin, ratio = base_boundary_margins(
        base_row=base_row,
        rank_hit_page=rank_hit_page,
        boundary_page=boundary_page,
    )
    case = {
        "qid": qid,
        "question": question,
        "accepted": False,
        "reason": "",
        "hit_k": hit_k,
        "boundary_rank": boundary_rank,
        "top_pages": top_pages,
        "rank_hit_page": rank_hit_page,
        "boundary_page": boundary_page,
        "rank_hit_doc": rank_hit_doc,
        "boundary_doc": boundary_doc,
        "boundary_doc_policy": str(args.boundary_doc_policy),
        "boundary_doc_topk_count": boundary_doc_topk_count,
        "base_boundary_margin": margin,
        "base_boundary_margin_ratio": ratio,
    }
    if str(args.boundary_doc_policy) == "topk_doc" and boundary_doc not in set(top_docs):
        case["reason"] = "boundary_doc_not_in_topk"
        return case
    if str(args.boundary_doc_policy) == "top1_doc" and (not top_pages or boundary_doc != page_doc(top_pages[0])):
        case["reason"] = "boundary_doc_not_top1_doc"
        return case
    if str(args.boundary_doc_policy) == "rank_hit_doc" and boundary_doc != rank_hit_doc:
        case["reason"] = "boundary_doc_not_rank_hit_doc"
        return case
    if int(args.min_boundary_doc_topk_count) > 0 and boundary_doc_topk_count < int(args.min_boundary_doc_topk_count):
        case["reason"] = "boundary_doc_topk_count_too_low"
        return case
    if args.max_base_margin_4_5 is not None and (margin is None or margin > float(args.max_base_margin_4_5)):
        case["reason"] = "base_boundary_margin_too_large"
        return case
    if args.max_base_margin_ratio_4_5 is not None and (ratio is None or ratio > float(args.max_base_margin_ratio_4_5)):
        case["reason"] = "base_boundary_margin_ratio_too_large"
        return case
    missing_images = [
        uid
        for uid in (rank_hit_page, boundary_page)
        if uid not in image_paths or not image_paths[uid].is_file()
    ]
    if missing_images:
        case["reason"] = "missing_page_image"
        case["missing_page_images"] = missing_images
        return case
    case["reason"] = "eligible"
    case["rank_hit_image_path"] = str(image_paths[rank_hit_page])
    case["boundary_image_path"] = str(image_paths[boundary_page])
    return case


def run_vlm_decision(case: dict[str, Any], vqa_model: Any, args: argparse.Namespace) -> dict[str, Any]:
    if bool(args.dry_run):
        case["reason"] = "dry_run_kept_base"
        return case
    if case.get("reason") != "eligible":
        return case
    rank_hit_image = open_image(Path(str(case["rank_hit_image_path"])))
    boundary_image = open_image(Path(str(case["boundary_image_path"])))
    question = str(case.get("question", ""))

    if str(args.decision_mode) == "pairwise":
        response = vqa_model.generate(images=[rank_hit_image, boundary_image], question=pairwise_prompt(question))
        winner, normalized = parse_pairwise_response(response)
        case["vlm_response"] = str(response).strip()
        case["vlm_response_normalized"] = normalized
        case["vlm_winner"] = winner
        if winner == "boundary" or (winner == "tie" and bool(args.accept_pairwise_tie)):
            case["accepted"] = True
            case["reason"] = "vlm_prefers_boundary"
        else:
            case["reason"] = f"vlm_prefers_{winner}"
        return case

    prompt = single_page_prompt(question)
    rank_hit_response = vqa_model.generate(images=[rank_hit_image], question=prompt)
    boundary_response = vqa_model.generate(images=[boundary_image], question=prompt)
    rank_hit_score, rank_hit_norm = parse_yes_no_response(rank_hit_response)
    boundary_score, boundary_norm = parse_yes_no_response(boundary_response)
    margin = boundary_score - rank_hit_score
    case.update(
        {
            "rank_hit_vlm_response": str(rank_hit_response).strip(),
            "rank_hit_vlm_response_normalized": rank_hit_norm,
            "rank_hit_vlm_score": rank_hit_score,
            "boundary_vlm_response": str(boundary_response).strip(),
            "boundary_vlm_response_normalized": boundary_norm,
            "boundary_vlm_score": boundary_score,
            "vlm_score_margin": margin,
        }
    )
    if (
        boundary_score >= float(args.min_boundary_score)
        and rank_hit_score <= float(args.max_rank_hit_score)
        and margin >= float(args.min_vlm_margin)
    ):
        case["accepted"] = True
        case["reason"] = "vlm_independent_boundary_win"
    else:
        case["reason"] = "vlm_independent_reject"
    return case


def metric_scores(
    pred_row: dict[str, Any] | None,
    gold_pages: set[str],
    gold_docs: set[str],
    recall_ks: list[int],
    hit_k: int,
) -> dict[str, Any]:
    pages = ranked_pages(pred_row)
    docs = ranked_docs(pred_row)
    page_rank = first_rank(pages, gold_pages)
    doc_rank = first_rank(docs, gold_docs)
    out: dict[str, Any] = {
        "page_first_rank": page_rank,
        "doc_first_rank": doc_rank,
        f"page_hit@{hit_k}": page_rank is not None and page_rank <= hit_k,
        f"doc_hit@{hit_k}": doc_rank is not None and doc_rank <= hit_k,
    }
    for k in recall_ks:
        out[f"page_recall@{k}"] = recall_at_k(pages, gold_pages, k)
        out[f"doc_recall@{k}"] = recall_at_k(docs, gold_docs, k)
    return out


def evaluate(
    *,
    gold: dict[str, dict[str, Any]],
    base: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    cases: list[dict[str, Any]],
    recall_ks: list[int],
    hit_k: int,
) -> dict[str, Any]:
    page_recall: dict[int, list[float]] = defaultdict(list)
    doc_recall: dict[int, list[float]] = defaultdict(list)
    movement_counts: Counter[str] = Counter()
    reason_counts = Counter(str(case.get("reason", "")) for case in cases)
    page_hit_count = 0
    doc_hit_count = 0
    base_page_hit_count = 0
    base_doc_hit_count = 0

    for qid in sorted(set(gold) & set(base) & set(candidate)):
        gold_pages = gold_page_uids(gold[qid])
        gold_docs = gold_doc_ids(gold[qid])
        base_scores = metric_scores(base[qid], gold_pages, gold_docs, recall_ks, hit_k)
        cand_scores = metric_scores(candidate[qid], gold_pages, gold_docs, recall_ks, hit_k)
        base_page_rank = base_scores["page_first_rank"]
        cand_page_rank = cand_scores["page_first_rank"]
        base_page_hit_count += int(bool(base_scores[f"page_hit@{hit_k}"]))
        base_doc_hit_count += int(bool(base_scores[f"doc_hit@{hit_k}"]))
        page_hit_count += int(bool(cand_scores[f"page_hit@{hit_k}"]))
        doc_hit_count += int(bool(cand_scores[f"doc_hit@{hit_k}"]))
        movement_counts[movement_for_hit(base_page_rank, cand_page_rank, hit_k)] += 1
        for k in recall_ks:
            page_recall[int(k)].append(float(cand_scores.get(f"page_recall@{k}", 0.0)))
            doc_recall[int(k)].append(float(cand_scores.get(f"doc_recall@{k}", 0.0)))

    accepted_count = sum(1 for row in cases if row.get("accepted"))
    recovered = int(movement_counts.get("recovered", 0))
    lost = int(movement_counts.get("lost", 0))
    return {
        "n": len(set(gold) & set(base) & set(candidate)),
        "case_count": len(cases),
        "accepted_count": accepted_count,
        "accept_frac": accepted_count / float(len(cases)) if cases else 0.0,
        "reason_counts": dict(sorted(reason_counts.items())),
        "base_page_hit_at_k_count": base_page_hit_count,
        "page_hit_at_k_count": page_hit_count,
        "base_doc_hit_at_k_count": base_doc_hit_count,
        "doc_hit_at_k_count": doc_hit_count,
        "movement_counts": dict(sorted(movement_counts.items())),
        "recovered": recovered,
        "lost": lost,
        "net_recovered": recovered - lost,
        "page_recall_at_k": {str(k): mean(values) for k, values in sorted(page_recall.items())},
        "doc_recall_at_k": {str(k): mean(values) for k, values in sorted(doc_recall.items())},
    }


def iter_qids(gold: dict[str, Any], base: dict[str, Any], args: argparse.Namespace) -> list[str]:
    qids = sorted(set(gold) & set(base))
    if str(args.qid_jsonl).strip():
        qid_filter = load_qid_filter(Path(args.qid_jsonl), str(args.qid_field))
        qids = [qid for qid in qids if qid in qid_filter]
    if int(args.num_shards) > 1:
        if int(args.shard_index) < 0 or int(args.shard_index) >= int(args.num_shards):
            raise ValueError("--shard-index must be in [0, num_shards).")
        qids = [qid for idx, qid in enumerate(qids) if idx % int(args.num_shards) == int(args.shard_index)]
    if int(args.sample_qids) > 0 and len(qids) > int(args.sample_qids):
        rng = random.Random(int(args.sample_seed))
        qids = sorted(rng.sample(qids, int(args.sample_qids)))
    if int(args.max_qids) > 0:
        qids = qids[: int(args.max_qids)]
    return qids


def progress(iterable: list[str]) -> Any:
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, desc="Boundary VLM verifier")
    except Exception:
        return iterable


def main() -> None:
    args = parse_args()
    if int(args.boundary_rank) <= int(args.hit_k):
        raise ValueError("--boundary-rank must be greater than --hit-k.")
    gold = {str(row["qid"]): row for row in read_jsonl(Path(args.gold))}
    base = load_prediction(Path(args.base_prediction))
    image_paths = load_page_image_paths(Path(args.doc_pages_jsonl))
    qids = iter_qids(gold, base, args)
    vqa_model = None if bool(args.dry_run) else load_vqa_model(args)

    output_rows: dict[str, dict[str, Any]] = {}
    cases: list[dict[str, Any]] = []
    for qid in progress(qids):
        base_row = base[qid]
        question = get_question(gold[qid])
        try:
            case = preflight_case(
                qid=qid,
                question=question,
                base_row=base_row,
                image_paths=image_paths,
                args=args,
            )
            if vqa_model is not None or bool(args.dry_run):
                case = run_vlm_decision(case, vqa_model, args)
            output_row = build_reordered_row(
                base_row=base_row,
                rank_hit_page=str(case.get("rank_hit_page", "")),
                boundary_page=str(case.get("boundary_page", "")),
                accepted=bool(case.get("accepted")),
            )
        except Exception as exc:
            output_row = copy.deepcopy(base_row)
            case = {
                "qid": qid,
                "question": question,
                "accepted": False,
                "reason": "error_kept_base",
                "error": f"{type(exc).__name__}: {exc}",
            }
        output_row["qid"] = str(output_row.get("qid", qid))
        output_rows[qid] = output_row
        gold_pages = gold_page_uids(gold[qid])
        case["base_first_gold_page_rank"] = first_rank(ranked_pages(base_row), gold_pages)
        case["candidate_first_gold_page_rank"] = first_rank(ranked_pages(output_row), gold_pages)
        case["rank_hit_is_gold"] = case.get("rank_hit_page") in gold_pages
        case["boundary_is_gold"] = case.get("boundary_page") in gold_pages
        cases.append(case)

    output_prediction = Path(args.output_prediction_json)
    output_prediction.parent.mkdir(parents=True, exist_ok=True)
    output_prediction.write_text(json.dumps({"predictions": output_rows}, ensure_ascii=False), encoding="utf-8")

    summary = evaluate(
        gold=gold,
        base=base,
        candidate=output_rows,
        cases=cases,
        recall_ks=[int(k) for k in args.recall_ks],
        hit_k=int(args.hit_k),
    )
    summary.update(
        {
            "gold": str(args.gold),
            "base_prediction": str(args.base_prediction),
            "doc_pages_jsonl": str(args.doc_pages_jsonl),
            "hit_k": int(args.hit_k),
            "boundary_rank": int(args.boundary_rank),
            "boundary_doc_policy": str(args.boundary_doc_policy),
            "min_boundary_doc_topk_count": int(args.min_boundary_doc_topk_count),
            "max_base_margin_4_5": args.max_base_margin_4_5,
            "max_base_margin_ratio_4_5": args.max_base_margin_ratio_4_5,
            "decision_mode": str(args.decision_mode),
            "min_vlm_margin": float(args.min_vlm_margin),
            "min_boundary_score": float(args.min_boundary_score),
            "max_rank_hit_score": float(args.max_rank_hit_score),
            "accept_pairwise_tie": bool(args.accept_pairwise_tie),
            "vlm_model_name_or_path": str(args.vlm_model_name_or_path),
            "vlm_model_type": str(args.vlm_model_type) or None,
            "vlm_bits": int(args.vlm_bits),
            "dry_run": bool(args.dry_run),
            "qid_jsonl": str(args.qid_jsonl) if str(args.qid_jsonl).strip() else "",
            "qid_field": str(args.qid_field),
            "filtered_qid_count": len(qids),
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
            "max_qids": int(args.max_qids),
            "sample_qids": int(args.sample_qids),
            "sample_seed": int(args.sample_seed),
        }
    )
    output_summary = Path(args.output_summary_json)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.output_case_json:
        output_case = Path(args.output_case_json)
        output_case.parent.mkdir(parents=True, exist_ok=True)
        output_case.write_text(json.dumps({"cases": cases}, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"saved_prediction: {output_prediction}")
    print(f"saved_summary: {output_summary}")
    if args.output_case_json:
        print(f"saved_cases: {args.output_case_json}")
    print(f"n {summary['n']}")
    print(f"accepted {summary['accepted_count']}")
    print(f"page_hit_at_{int(args.hit_k)}_count {summary['page_hit_at_k_count']}")
    print(f"base_page_hit_at_{int(args.hit_k)}_count {summary['base_page_hit_at_k_count']}")
    print(f"recovered {summary['recovered']}")
    print(f"lost {summary['lost']}")
    print(f"net_recovered {summary['net_recovered']}")
    print(f"page_recall_at_k {summary['page_recall_at_k']}")
    print(f"doc_recall_at_k {summary['doc_recall_at_k']}")


if __name__ == "__main__":
    main()
