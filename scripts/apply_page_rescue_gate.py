#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Conservatively apply a candidate page ranking only as a local rescue. "
            "The base ranking is preserved unless the candidate promotes an already-retrieved "
            "page from a configurable base-rank window and the candidate top-k mostly agrees "
            "with the base top-k. Gold labels are optional and are used only for reporting."
        )
    )
    parser.add_argument("--base-prediction", required=True)
    parser.add_argument("--candidate-prediction", required=True)
    parser.add_argument(
        "--support-prediction",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help=(
            "Optional support prediction JSON. Repeat to add weak graph views. "
            "Support views never provide output rows; they only vote for candidate promotions."
        ),
    )
    parser.add_argument("--gold", default="", help="Optional gold JSONL for reporting metrics.")
    parser.add_argument("--hit-k", type=int, default=4)
    parser.add_argument(
        "--recall-k",
        dest="recall_ks",
        type=int,
        nargs="+",
        default=[1, 2, 4, 5, 10, 20, 50, 100],
    )
    parser.add_argument(
        "--candidate-rank-max",
        type=int,
        default=2,
        help="Only candidate pages at or above this rank can trigger a rescue.",
    )
    parser.add_argument(
        "--rescue-rank-min",
        type=int,
        default=5,
        help="Minimum base page rank for a promoted page to be considered a rescue.",
    )
    parser.add_argument(
        "--rescue-rank-max",
        type=int,
        default=20,
        help="Maximum base page rank for a promoted page. Use 0 to disable the upper bound.",
    )
    parser.add_argument(
        "--min-page-overlap",
        type=int,
        default=3,
        help="Minimum shared pages between base top-k and candidate top-k.",
    )
    parser.add_argument(
        "--min-doc-overlap",
        type=int,
        default=0,
        help="Minimum shared docs between docs represented in base top-k and candidate top-k.",
    )
    parser.add_argument(
        "--promoted-doc-max-base-rank",
        type=int,
        default=5,
        help=(
            "Require the promoted page's document to appear within this base doc rank. "
            "Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--max-base-score-margin",
        type=float,
        default=None,
        help=(
            "Optional cap on base score(rank hit-k) - score(rank hit-k+1). "
            "This rejects stable top-k boundaries when row scores are available."
        ),
    )
    parser.add_argument(
        "--min-candidate-score-margin",
        type=float,
        default=None,
        help=(
            "Optional floor on candidate score(promoted rank) - score(next rank). "
            "If set, candidates without comparable scores are rejected."
        ),
    )
    parser.add_argument(
        "--support-page-rank-max",
        type=int,
        default=0,
        help=(
            "Count a support page vote when the promoted page appears within this support "
            "page rank. Use 0 to disable page support votes."
        ),
    )
    parser.add_argument(
        "--support-doc-rank-max",
        type=int,
        default=0,
        help=(
            "Count a support doc vote when the promoted page's doc appears within this support "
            "doc rank. Use 0 to disable doc support votes."
        ),
    )
    parser.add_argument(
        "--min-support-page-votes",
        type=int,
        default=0,
        help="Minimum support views that must contain the promoted page within --support-page-rank-max.",
    )
    parser.add_argument(
        "--min-support-doc-votes",
        type=int,
        default=0,
        help="Minimum support views that must contain the promoted doc within --support-doc-rank-max.",
    )
    parser.add_argument(
        "--mode",
        choices=["swap_promoted", "use_candidate"],
        default="swap_promoted",
        help=(
            "swap_promoted inserts accepted promoted pages into the base ranking; "
            "use_candidate replaces the whole ranking after acceptance."
        ),
    )
    parser.add_argument(
        "--insert-position",
        type=int,
        default=4,
        help="1-based insertion rank for swap_promoted mode.",
    )
    parser.add_argument("--max-promotions", type=int, default=1)
    parser.add_argument(
        "--output-top-pages",
        type=int,
        default=0,
        help="Maximum rows to write per qid. Use 0 to keep all deduped rows.",
    )
    parser.add_argument(
        "--score-mode",
        choices=["preserve", "rank"],
        default="preserve",
        help="Whether to preserve original row scores or rewrite them as 1/rank.",
    )
    parser.add_argument("--top-examples", type=int, default=30)
    parser.add_argument("--output-prediction-json", required=True)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument(
        "--output-cases-json",
        default="",
        help="Optional path to write per-qid diagnostics outside the summary JSON.",
    )
    return parser.parse_args()


def parse_labeled_path(value: str) -> tuple[str, Path]:
    raw = str(value).strip()
    if not raw:
        raise ValueError("Empty labeled path")
    if "=" in raw:
        label, path = raw.split("=", 1)
        label = label.strip()
        parsed = Path(path.strip())
    else:
        parsed = Path(raw)
        label = parsed.stem
    if not label:
        label = parsed.stem
    return label, parsed


def load_prediction(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "predictions" in payload and isinstance(
        payload["predictions"],
        (dict, list),
    ):
        payload = payload["predictions"]

    rows_by_qid: dict[str, dict[str, Any]] = {}
    iterable: Any
    if isinstance(payload, list):
        iterable = enumerate(payload)
    elif isinstance(payload, dict):
        iterable = payload.items()
    else:
        raise TypeError(f"Prediction JSON must be a list or object: {path}")

    for raw_key, row in iterable:
        if not isinstance(row, dict):
            raise TypeError(f"Prediction row must be an object: {path} key={raw_key!r}")
        qid = str(row.get("qid", "")).strip() or str(raw_key).strip()
        if not qid:
            raise ValueError(f"Prediction row is missing qid: {path} key={raw_key!r}")
        if qid in rows_by_qid:
            raise ValueError(f"Duplicate qid after normalization: {qid} ({path})")
        rows_by_qid[qid] = row
    return rows_by_qid


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def page_uid(doc_id: object, page_idx: object) -> str:
    return f"{doc_id}_page{int(page_idx)}"


def parse_page_uid(uid: str) -> tuple[str, int]:
    marker = "_page"
    if marker not in uid:
        raise ValueError(f"Invalid page uid: {uid}")
    doc_id, page_idx = uid.rsplit(marker, 1)
    return doc_id, int(page_idx)


def prediction_rows(row: dict[str, Any]) -> list[Any]:
    rows = row.get("page_retrieval_results", [])
    return rows if isinstance(rows, list) else []


def row_page_uid(row: Any) -> str | None:
    if not isinstance(row, list) or len(row) < 2:
        return None
    try:
        return page_uid(row[0], row[1])
    except (TypeError, ValueError):
        return None


def row_score(row: Any) -> float | None:
    if not isinstance(row, list) or len(row) < 3:
        return None
    try:
        return float(row[2])
    except (TypeError, ValueError):
        return None


def ranked_page_uids(rows: list[Any], limit: int = 0) -> list[str]:
    pages: list[str] = []
    seen: set[str] = set()
    for row in rows:
        uid = row_page_uid(row)
        if uid is None or uid in seen:
            continue
        seen.add(uid)
        pages.append(uid)
        if limit > 0 and len(pages) >= limit:
            break
    return pages


def ranked_doc_ids(rows: list[Any], limit: int = 0) -> list[str]:
    doc_ids: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, list) or not row:
            continue
        doc_id = str(row[0]).strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        doc_ids.append(doc_id)
        if limit > 0 and len(doc_ids) >= limit:
            break
    return doc_ids


def docs_from_page_uids(page_uids: list[str]) -> list[str]:
    docs: list[str] = []
    seen: set[str] = set()
    for uid in page_uids:
        doc_id, _page_idx = parse_page_uid(uid)
        if doc_id not in seen:
            seen.add(doc_id)
            docs.append(doc_id)
    return docs


def rows_by_uid(rows: list[Any]) -> dict[str, Any]:
    by_uid: dict[str, Any] = {}
    for row in rows:
        uid = row_page_uid(row)
        if uid is not None and uid not in by_uid:
            by_uid[uid] = row
    return by_uid


def rank_map(values: list[str]) -> dict[str, int]:
    return {value: rank for rank, value in enumerate(values, start=1)}


def score_at_rank(rows: list[Any], rank: int) -> float | None:
    if rank <= 0 or rank > len(rows):
        return None
    return row_score(rows[rank - 1])


def score_margin(rows: list[Any], rank: int) -> float | None:
    left = score_at_rank(rows, rank)
    right = score_at_rank(rows, rank + 1)
    if left is None or right is None:
        return None
    return float(left) - float(right)


def support_votes_for_uid(
    uid: str,
    support_view_rows: list[tuple[str, dict[str, Any] | None]],
    *,
    page_rank_max: int,
    doc_rank_max: int,
) -> dict[str, Any]:
    doc_id, _page_idx = parse_page_uid(uid)
    page_vote_sources: list[str] = []
    doc_vote_sources: list[str] = []
    view_ranks: dict[str, dict[str, int | None]] = {}

    for label, row in support_view_rows:
        if row is None:
            view_ranks[label] = {"page_rank": None, "doc_rank": None}
            continue
        rows = prediction_rows(row)
        pages = ranked_page_uids(rows)
        docs = ranked_doc_ids(rows)
        page_rank = rank_map(pages).get(uid)
        doc_rank = rank_map(docs).get(doc_id)
        view_ranks[label] = {"page_rank": page_rank, "doc_rank": doc_rank}
        if page_rank_max > 0 and page_rank is not None and page_rank <= page_rank_max:
            page_vote_sources.append(label)
        if doc_rank_max > 0 and doc_rank is not None and doc_rank <= doc_rank_max:
            doc_vote_sources.append(label)

    return {
        "support_page_vote_count": len(page_vote_sources),
        "support_doc_vote_count": len(doc_vote_sources),
        "support_page_vote_sources": page_vote_sources,
        "support_doc_vote_sources": doc_vote_sources,
        "support_view_ranks": view_ranks,
    }


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
            uids.add(page_uid(doc_id, page_idx))
    return uids


def gold_doc_ids(row: dict[str, Any]) -> set[str]:
    metadata = row.get("metadata", {})
    doc_ids = {
        str(value).strip()
        for value in metadata.get("gold_doc_ids", [])
        if str(value).strip()
    }
    for ctx in row.get("supporting_context", []):
        doc_id = str(ctx.get("doc_id", "")).strip()
        if doc_id:
            doc_ids.add(doc_id)
    return doc_ids


def recall_at_k(ranked: list[str], gold: set[str], topk: int) -> float | None:
    if not gold:
        return None
    return len(set(ranked[:topk]) & gold) / float(len(gold))


def first_rank(ranked: list[str], gold: set[str]) -> int | None:
    for idx, value in enumerate(ranked, start=1):
        if value in gold:
            return idx
    return None


def hit_at(rank: int | None, topk: int) -> bool:
    return rank is not None and int(rank) <= int(topk)


def movement_for_hit(
    baseline_rank: int | None,
    candidate_rank: int | None,
    topk: int,
) -> str:
    baseline_hit = hit_at(baseline_rank, topk)
    candidate_hit = hit_at(candidate_rank, topk)
    if not baseline_hit and candidate_hit:
        return "recovered"
    if baseline_hit and not candidate_hit:
        return "lost"
    if baseline_rank is None and candidate_rank is None:
        return "missing_in_both"
    if baseline_rank is not None and candidate_rank is not None and candidate_rank < baseline_rank:
        return "improved_rank"
    if baseline_rank is not None and candidate_rank is not None and candidate_rank > baseline_rank:
        return "worsened_rank"
    return "unchanged"


def mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def maybe_rewrite_scores(rows: list[Any], score_mode: str) -> list[Any]:
    if score_mode != "rank":
        return rows
    rewritten: list[Any] = []
    for rank, row in enumerate(rows, start=1):
        if not isinstance(row, list):
            rewritten.append(row)
            continue
        new_row = copy.deepcopy(row)
        score = 1.0 / float(rank)
        if len(new_row) >= 3:
            new_row[2] = score
        else:
            new_row.append(score)
        rewritten.append(new_row)
    return rewritten


def append_unique_rows(
    output: list[Any],
    rows: list[Any],
    seen: set[str],
    *,
    skip_uids: set[str] | None = None,
) -> None:
    skip_uids = skip_uids or set()
    for row in rows:
        uid = row_page_uid(row)
        if uid is None or uid in seen or uid in skip_uids:
            continue
        seen.add(uid)
        output.append(copy.deepcopy(row))


def build_swapped_rows(
    *,
    base_rows: list[Any],
    candidate_rows: list[Any],
    promoted_uids: list[str],
    insert_position: int,
    output_top_pages: int,
    score_mode: str,
) -> list[Any]:
    candidate_by_uid = rows_by_uid(candidate_rows)
    insert_idx = max(0, int(insert_position) - 1)
    promoted_set = set(promoted_uids)
    output: list[Any] = []
    seen: set[str] = set()

    append_unique_rows(output, base_rows[:insert_idx], seen, skip_uids=promoted_set)
    for uid in promoted_uids:
        row = candidate_by_uid.get(uid)
        if row is None or uid in seen:
            continue
        seen.add(uid)
        output.append(copy.deepcopy(row))
    append_unique_rows(output, base_rows[insert_idx:], seen, skip_uids=promoted_set)
    append_unique_rows(output, candidate_rows, seen)

    if output_top_pages > 0:
        output = output[: int(output_top_pages)]
    return maybe_rewrite_scores(output, score_mode)


def build_candidate_rows(
    *,
    candidate_rows: list[Any],
    output_top_pages: int,
    score_mode: str,
) -> list[Any]:
    output: list[Any] = []
    seen: set[str] = set()
    append_unique_rows(output, candidate_rows, seen)
    if output_top_pages > 0:
        output = output[: int(output_top_pages)]
    return maybe_rewrite_scores(output, score_mode)


def reject_promotion_reason(
    *,
    uid: str,
    candidate_rank: int,
    base_page_rank_by_uid: dict[str, int],
    base_doc_rank_by_id: dict[str, int],
    candidate_rows: list[Any],
    support_view_rows: list[tuple[str, dict[str, Any] | None]],
    args: argparse.Namespace,
) -> tuple[str | None, dict[str, Any]]:
    doc_id, _page_idx = parse_page_uid(uid)
    base_rank = base_page_rank_by_uid.get(uid)
    doc_rank = base_doc_rank_by_id.get(doc_id)
    margin = score_margin(candidate_rows, candidate_rank)
    detail = {
        "page_uid": uid,
        "doc_id": doc_id,
        "candidate_rank": candidate_rank,
        "base_rank": base_rank,
        "base_doc_rank": doc_rank,
        "candidate_score_margin": margin,
    }
    support_detail = support_votes_for_uid(
        uid,
        support_view_rows,
        page_rank_max=int(args.support_page_rank_max),
        doc_rank_max=int(args.support_doc_rank_max),
    )
    detail.update(support_detail)

    if base_rank is None:
        return "promoted_page_missing_from_base_pool", detail
    if base_rank < int(args.rescue_rank_min):
        return "promoted_page_before_rescue_window", detail
    if int(args.rescue_rank_max) > 0 and base_rank > int(args.rescue_rank_max):
        return "promoted_page_after_rescue_window", detail
    if int(args.promoted_doc_max_base_rank) > 0:
        if doc_rank is None:
            return "promoted_doc_missing_from_base_pool", detail
        if doc_rank > int(args.promoted_doc_max_base_rank):
            return "promoted_doc_after_allowed_rank", detail
    if args.min_candidate_score_margin is not None:
        if margin is None:
            return "candidate_score_margin_missing", detail
        if margin < float(args.min_candidate_score_margin):
            return "candidate_score_margin_below_min", detail
    if int(args.min_support_page_votes) > 0:
        if int(detail["support_page_vote_count"]) < int(args.min_support_page_votes):
            return "support_page_votes_below_min", detail
    if int(args.min_support_doc_votes) > 0:
        if int(detail["support_doc_vote_count"]) < int(args.min_support_doc_votes):
            return "support_doc_votes_below_min", detail
    return None, detail


def select_promotions(
    *,
    base_row: dict[str, Any],
    candidate_row: dict[str, Any] | None,
    support_view_rows: list[tuple[str, dict[str, Any] | None]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    base_rows = prediction_rows(base_row)
    candidate_rows = prediction_rows(candidate_row) if candidate_row is not None else []
    base_pages = ranked_page_uids(base_rows)
    candidate_pages = ranked_page_uids(candidate_rows)
    base_docs = ranked_doc_ids(base_rows)
    candidate_docs = ranked_doc_ids(candidate_rows)
    base_top_pages = base_pages[: int(args.hit_k)]
    candidate_top_pages = candidate_pages[: int(args.hit_k)]
    base_top_docs = docs_from_page_uids(base_top_pages)
    candidate_top_docs = docs_from_page_uids(candidate_top_pages)
    base_page_rank_by_uid = rank_map(base_pages)
    base_doc_rank_by_id = rank_map(base_docs)
    candidate_page_rank_by_uid = rank_map(candidate_pages)
    page_overlap = len(set(base_top_pages) & set(candidate_top_pages))
    doc_overlap = len(set(base_top_docs) & set(candidate_top_docs))
    base_boundary_margin = score_margin(base_rows, int(args.hit_k))

    decision: dict[str, Any] = {
        "selected_source": "base",
        "selection_reason": "",
        "accepted": False,
        "promoted_pages": [],
        "candidate_promoted_pages": [],
        "rejected_promoted_pages": [],
        "base_top_pages": base_top_pages,
        "candidate_top_pages": candidate_top_pages,
        "base_top_docs": base_top_docs,
        "candidate_top_docs": candidate_top_docs,
        "topk_page_overlap": page_overlap,
        "topk_doc_overlap": doc_overlap,
        "base_boundary_score_margin": base_boundary_margin,
        "support_view_count": len(support_view_rows),
        "support_labels": [label for label, _row in support_view_rows],
    }

    if candidate_row is None:
        decision["selection_reason"] = "missing_candidate_qid"
        return decision
    if not candidate_pages:
        decision["selection_reason"] = "empty_candidate_ranking"
        return decision
    if page_overlap < int(args.min_page_overlap):
        decision["selection_reason"] = "page_overlap_below_min"
        return decision
    if doc_overlap < int(args.min_doc_overlap):
        decision["selection_reason"] = "doc_overlap_below_min"
        return decision
    if args.max_base_score_margin is not None:
        if base_boundary_margin is None:
            decision["selection_reason"] = "base_score_margin_missing"
            return decision
        if base_boundary_margin > float(args.max_base_score_margin):
            decision["selection_reason"] = "base_score_margin_above_max"
            return decision

    candidate_promotions = [
        uid
        for uid in candidate_pages[: int(args.candidate_rank_max)]
        if uid not in set(base_top_pages)
    ]
    decision["candidate_promoted_pages"] = candidate_promotions
    if not candidate_promotions:
        decision["selection_reason"] = "no_candidate_promotions"
        return decision

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for uid in candidate_promotions:
        candidate_rank = candidate_page_rank_by_uid[uid]
        reason, detail = reject_promotion_reason(
            uid=uid,
            candidate_rank=candidate_rank,
            base_page_rank_by_uid=base_page_rank_by_uid,
            base_doc_rank_by_id=base_doc_rank_by_id,
            candidate_rows=candidate_rows,
            support_view_rows=support_view_rows,
            args=args,
        )
        if reason is None:
            accepted.append(detail)
        else:
            detail["reject_reason"] = reason
            rejected.append(detail)
    decision["rejected_promoted_pages"] = rejected
    if not accepted:
        if rejected:
            decision["selection_reason"] = str(rejected[0]["reject_reason"])
        else:
            decision["selection_reason"] = "no_accepted_promotions"
        return decision

    accepted.sort(
        key=lambda item: (
            int(item["candidate_rank"]),
            int(item["base_rank"]) if item.get("base_rank") is not None else 10**9,
            str(item["page_uid"]),
        )
    )
    accepted = accepted[: max(1, int(args.max_promotions))]
    promoted_pages = [str(item["page_uid"]) for item in accepted]
    decision["selected_source"] = "candidate" if args.mode == "use_candidate" else "base_plus_promotions"
    decision["selection_reason"] = "accepted_rank_window_promotion"
    decision["accepted"] = True
    decision["promoted_pages"] = promoted_pages
    decision["accepted_promoted_pages"] = accepted
    return decision


def build_output_row(
    *,
    qid: str,
    base_row: dict[str, Any],
    candidate_row: dict[str, Any] | None,
    decision: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if decision.get("accepted") and candidate_row is not None and args.mode == "use_candidate":
        output = copy.deepcopy(candidate_row)
        rows = build_candidate_rows(
            candidate_rows=prediction_rows(candidate_row),
            output_top_pages=int(args.output_top_pages),
            score_mode=str(args.score_mode),
        )
    elif decision.get("accepted") and candidate_row is not None:
        output = copy.deepcopy(base_row)
        rows = build_swapped_rows(
            base_rows=prediction_rows(base_row),
            candidate_rows=prediction_rows(candidate_row),
            promoted_uids=[str(uid) for uid in decision.get("promoted_pages", [])],
            insert_position=int(args.insert_position),
            output_top_pages=int(args.output_top_pages),
            score_mode=str(args.score_mode),
        )
    else:
        output = copy.deepcopy(base_row)
        rows = build_swapped_rows(
            base_rows=prediction_rows(base_row),
            candidate_rows=[],
            promoted_uids=[],
            insert_position=int(args.insert_position),
            output_top_pages=int(args.output_top_pages),
            score_mode=str(args.score_mode),
        )

    output["qid"] = qid
    output["page_retrieval_results"] = rows
    output["top_retrieved_docs"] = ranked_doc_ids(rows, 10)
    metadata = output.get("reranker_metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    metadata = copy.deepcopy(metadata)
    metadata["page_rescue_gate"] = {
        key: value
        for key, value in decision.items()
        if key
        not in {
            "base_top_pages",
            "candidate_top_pages",
            "base_top_docs",
            "candidate_top_docs",
        }
    }
    metadata["page_rescue_gate"]["config"] = {
        "hit_k": int(args.hit_k),
        "candidate_rank_max": int(args.candidate_rank_max),
        "rescue_rank_min": int(args.rescue_rank_min),
        "rescue_rank_max": int(args.rescue_rank_max),
        "min_page_overlap": int(args.min_page_overlap),
        "min_doc_overlap": int(args.min_doc_overlap),
        "promoted_doc_max_base_rank": int(args.promoted_doc_max_base_rank),
        "max_base_score_margin": args.max_base_score_margin,
        "min_candidate_score_margin": args.min_candidate_score_margin,
        "support_page_rank_max": int(args.support_page_rank_max),
        "support_doc_rank_max": int(args.support_doc_rank_max),
        "min_support_page_votes": int(args.min_support_page_votes),
        "min_support_doc_votes": int(args.min_support_doc_votes),
        "mode": str(args.mode),
        "insert_position": int(args.insert_position),
        "max_promotions": int(args.max_promotions),
        "score_mode": str(args.score_mode),
    }
    output["reranker_metadata"] = metadata
    return output


def add_gold_metrics(
    *,
    case: dict[str, Any],
    gold_row: dict[str, Any],
    base_row: dict[str, Any],
    candidate_row: dict[str, Any] | None,
    output_row: dict[str, Any],
    recall_ks: list[int],
    hit_k: int,
) -> None:
    page_gold = gold_page_uids(gold_row)
    doc_gold = gold_doc_ids(gold_row)
    base_pages = ranked_page_uids(prediction_rows(base_row))
    base_docs = ranked_doc_ids(prediction_rows(base_row))
    candidate_pages = ranked_page_uids(prediction_rows(candidate_row)) if candidate_row else []
    candidate_docs = ranked_doc_ids(prediction_rows(candidate_row)) if candidate_row else []
    output_pages = ranked_page_uids(prediction_rows(output_row))
    output_docs = ranked_doc_ids(prediction_rows(output_row))

    base_page_rank = first_rank(base_pages, page_gold)
    candidate_page_rank = first_rank(candidate_pages, page_gold) if candidate_row else None
    output_page_rank = first_rank(output_pages, page_gold)
    base_doc_rank = first_rank(base_docs, doc_gold)
    candidate_doc_rank = first_rank(candidate_docs, doc_gold) if candidate_row else None
    output_doc_rank = first_rank(output_docs, doc_gold)

    case["gold_page_uids"] = sorted(page_gold)
    case["gold_doc_ids"] = sorted(doc_gold)
    case["base_first_gold_page_rank"] = base_page_rank
    case["candidate_first_gold_page_rank"] = candidate_page_rank
    case["output_first_gold_page_rank"] = output_page_rank
    case["base_first_gold_doc_rank"] = base_doc_rank
    case["candidate_first_gold_doc_rank"] = candidate_doc_rank
    case["output_first_gold_doc_rank"] = output_doc_rank
    case["movement_vs_base"] = movement_for_hit(base_page_rank, output_page_rank, hit_k)
    case["candidate_movement_vs_base"] = movement_for_hit(
        base_page_rank,
        candidate_page_rank,
        hit_k,
    )
    case["page_recall_at_k"] = {}
    case["doc_recall_at_k"] = {}
    for cutoff in recall_ks:
        page_value = recall_at_k(output_pages, page_gold, int(cutoff))
        doc_value = recall_at_k(output_docs, doc_gold, int(cutoff))
        case["page_recall_at_k"][str(cutoff)] = page_value
        case["doc_recall_at_k"][str(cutoff)] = doc_value


def summarize_cases(
    *,
    cases: list[dict[str, Any]],
    gold_cases: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    selection_counts = Counter(str(case["selected_source"]) for case in cases)
    reason_counts = Counter(str(case["selection_reason"]) for case in cases)
    accepted_promotion_details = [
        item
        for case in cases
        for item in case.get("accepted_promoted_pages", [])
        if isinstance(item, dict)
    ]
    rejected_promotion_counts: Counter[str] = Counter()
    for case in cases:
        for item in case.get("rejected_promoted_pages", []):
            rejected_promotion_counts[str(item.get("reject_reason", ""))] += 1

    summary: dict[str, Any] = {
        "qid_count": len(cases),
        "gold_qid_count": len(gold_cases),
        "accepted_count": int(selection_counts.get("base_plus_promotions", 0))
        + int(selection_counts.get("candidate", 0)),
        "accept_frac": (
            (
                int(selection_counts.get("base_plus_promotions", 0))
                + int(selection_counts.get("candidate", 0))
            )
            / float(len(cases))
            if cases
            else 0.0
        ),
        "selection_counts": dict(sorted(selection_counts.items())),
        "selection_reason_counts": dict(sorted(reason_counts.items())),
        "rejected_promotion_reason_counts": dict(sorted(rejected_promotion_counts.items())),
        "mean_topk_page_overlap": mean_or_none(
            [float(case["topk_page_overlap"]) for case in cases]
        ),
        "mean_topk_doc_overlap": mean_or_none(
            [float(case["topk_doc_overlap"]) for case in cases]
        ),
        "mean_accepted_support_page_vote_count": mean_or_none(
            [
                float(item.get("support_page_vote_count", 0))
                for item in accepted_promotion_details
            ]
        ),
        "mean_accepted_support_doc_vote_count": mean_or_none(
            [
                float(item.get("support_doc_vote_count", 0))
                for item in accepted_promotion_details
            ]
        ),
        "config": {
            "hit_k": int(args.hit_k),
            "recall_k": [int(k) for k in args.recall_ks],
            "candidate_rank_max": int(args.candidate_rank_max),
            "rescue_rank_min": int(args.rescue_rank_min),
            "rescue_rank_max": int(args.rescue_rank_max),
            "min_page_overlap": int(args.min_page_overlap),
            "min_doc_overlap": int(args.min_doc_overlap),
            "promoted_doc_max_base_rank": int(args.promoted_doc_max_base_rank),
            "max_base_score_margin": args.max_base_score_margin,
            "min_candidate_score_margin": args.min_candidate_score_margin,
            "support_page_rank_max": int(args.support_page_rank_max),
            "support_doc_rank_max": int(args.support_doc_rank_max),
            "min_support_page_votes": int(args.min_support_page_votes),
            "min_support_doc_votes": int(args.min_support_doc_votes),
            "mode": str(args.mode),
            "insert_position": int(args.insert_position),
            "max_promotions": int(args.max_promotions),
            "output_top_pages": int(args.output_top_pages),
            "score_mode": str(args.score_mode),
        },
        "base_prediction": str(args.base_prediction),
        "candidate_prediction": str(args.candidate_prediction),
        "support_predictions": list(args.support_prediction),
        "gold": str(args.gold) if args.gold else "",
    }

    if gold_cases:
        movement_counts = Counter(str(case["movement_vs_base"]) for case in gold_cases)
        candidate_movement_counts = Counter(
            str(case["candidate_movement_vs_base"]) for case in gold_cases
        )
        summary["movement_vs_base_counts"] = dict(sorted(movement_counts.items()))
        summary["candidate_movement_vs_base_counts"] = dict(
            sorted(candidate_movement_counts.items())
        )
        summary["recovered"] = int(movement_counts.get("recovered", 0))
        summary["lost"] = int(movement_counts.get("lost", 0))
        summary["net_recovered"] = int(movement_counts.get("recovered", 0)) - int(
            movement_counts.get("lost", 0)
        )
        summary["candidate_recovered"] = int(candidate_movement_counts.get("recovered", 0))
        summary["candidate_lost"] = int(candidate_movement_counts.get("lost", 0))
        summary["candidate_net_recovered"] = int(
            candidate_movement_counts.get("recovered", 0)
        ) - int(candidate_movement_counts.get("lost", 0))

        hit_k = int(args.hit_k)
        summary["base_page_hit_at_k_count"] = sum(
            1 for case in gold_cases if hit_at(case["base_first_gold_page_rank"], hit_k)
        )
        summary["candidate_page_hit_at_k_count"] = sum(
            1
            for case in gold_cases
            if hit_at(case["candidate_first_gold_page_rank"], hit_k)
        )
        summary["page_hit_at_k_count"] = sum(
            1 for case in gold_cases if hit_at(case["output_first_gold_page_rank"], hit_k)
        )
        summary["base_doc_hit_at_k_count"] = sum(
            1 for case in gold_cases if hit_at(case["base_first_gold_doc_rank"], hit_k)
        )
        summary["candidate_doc_hit_at_k_count"] = sum(
            1
            for case in gold_cases
            if hit_at(case["candidate_first_gold_doc_rank"], hit_k)
        )
        summary["doc_hit_at_k_count"] = sum(
            1 for case in gold_cases if hit_at(case["output_first_gold_doc_rank"], hit_k)
        )
        summary[f"base_page_hit_at_{hit_k}_count"] = summary["base_page_hit_at_k_count"]
        summary[f"candidate_page_hit_at_{hit_k}_count"] = summary[
            "candidate_page_hit_at_k_count"
        ]
        summary[f"page_hit_at_{hit_k}_count"] = summary["page_hit_at_k_count"]
        summary[f"base_doc_hit_at_{hit_k}_count"] = summary["base_doc_hit_at_k_count"]
        summary[f"candidate_doc_hit_at_{hit_k}_count"] = summary[
            "candidate_doc_hit_at_k_count"
        ]
        summary[f"doc_hit_at_{hit_k}_count"] = summary["doc_hit_at_k_count"]

        page_values: dict[str, list[float]] = {str(k): [] for k in args.recall_ks}
        doc_values: dict[str, list[float]] = {str(k): [] for k in args.recall_ks}
        for case in gold_cases:
            for cutoff in args.recall_ks:
                key = str(cutoff)
                page_value = case["page_recall_at_k"].get(key)
                doc_value = case["doc_recall_at_k"].get(key)
                if page_value is not None:
                    page_values[key].append(float(page_value))
                if doc_value is not None:
                    doc_values[key].append(float(doc_value))
        summary["page_recall_at_k"] = {
            key: mean_or_none(values) for key, values in page_values.items()
        }
        summary["doc_recall_at_k"] = {
            key: mean_or_none(values) for key, values in doc_values.items()
        }
        summary["top_recovered"] = top_cases(gold_cases, "recovered", int(args.top_examples))
        summary["top_lost"] = top_cases(gold_cases, "lost", int(args.top_examples))

    accepted_cases = [case for case in cases if bool(case.get("accepted"))]
    summary["top_accepted"] = accepted_cases[: int(args.top_examples)]
    return summary


def top_cases(cases: list[dict[str, Any]], movement: str, limit: int) -> list[dict[str, Any]]:
    filtered = [case for case in cases if case.get("movement_vs_base") == movement]
    filtered.sort(
        key=lambda case: (
            case.get("output_first_gold_page_rank")
            if case.get("output_first_gold_page_rank") is not None
            else 10**9,
            case.get("base_first_gold_page_rank")
            if case.get("base_first_gold_page_rank") is not None
            else 10**9,
            str(case.get("qid", "")),
        )
    )
    keys = [
        "qid",
        "question",
        "base_first_gold_page_rank",
        "candidate_first_gold_page_rank",
        "output_first_gold_page_rank",
        "gold_page_uids",
        "promoted_pages",
        "accepted_promoted_pages",
        "topk_page_overlap",
        "topk_doc_overlap",
    ]
    return [{key: case.get(key) for key in keys if key in case} for case in filtered[:limit]]


def main() -> None:
    args = parse_args()
    base = load_prediction(Path(args.base_prediction))
    candidate = load_prediction(Path(args.candidate_prediction))
    support_inputs = [parse_labeled_path(value) for value in args.support_prediction]
    support_predictions = [
        (label, load_prediction(path)) for label, path in support_inputs
    ]
    gold_by_qid = (
        {str(row.get("qid", "")).strip(): row for row in read_jsonl(Path(args.gold))}
        if args.gold
        else {}
    )

    output: dict[str, dict[str, Any]] = {}
    cases: list[dict[str, Any]] = []
    gold_cases: list[dict[str, Any]] = []
    for qid in sorted(base):
        base_row = base[qid]
        candidate_row = candidate.get(qid)
        support_view_rows = [
            (label, prediction.get(qid)) for label, prediction in support_predictions
        ]
        decision = select_promotions(
            base_row=base_row,
            candidate_row=candidate_row,
            support_view_rows=support_view_rows,
            args=args,
        )
        output_row = build_output_row(
            qid=qid,
            base_row=base_row,
            candidate_row=candidate_row,
            decision=decision,
            args=args,
        )
        output[qid] = output_row

        case = {
            "qid": qid,
            "question": base_row.get("question", ""),
            **decision,
        }
        if qid in gold_by_qid:
            add_gold_metrics(
                case=case,
                gold_row=gold_by_qid[qid],
                base_row=base_row,
                candidate_row=candidate_row,
                output_row=output_row,
                recall_ks=[int(k) for k in args.recall_ks],
                hit_k=int(args.hit_k),
            )
            gold_cases.append(case)
        cases.append(case)

    output_prediction_json = Path(args.output_prediction_json)
    output_prediction_json.parent.mkdir(parents=True, exist_ok=True)
    output_prediction_json.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = summarize_cases(cases=cases, gold_cases=gold_cases, args=args)
    if not args.output_cases_json:
        summary["cases"] = cases

    output_summary_json = Path(args.output_summary_json)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if args.output_cases_json:
        output_cases_json = Path(args.output_cases_json)
        output_cases_json.parent.mkdir(parents=True, exist_ok=True)
        output_cases_json.write_text(
            json.dumps(cases, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"saved_cases: {output_cases_json}")

    print(f"saved_prediction: {output_prediction_json}")
    print(f"saved_summary: {output_summary_json}")
    print(f"qid_count: {summary['qid_count']}")
    print(f"accepted: {summary['accepted_count']}")
    if gold_cases:
        print(f"base_page_hit_at_{args.hit_k}_count: {summary['base_page_hit_at_k_count']}")
        print(
            f"candidate_page_hit_at_{args.hit_k}_count: "
            f"{summary['candidate_page_hit_at_k_count']}"
        )
        print(f"page_hit_at_{args.hit_k}_count: {summary['page_hit_at_k_count']}")
        print(f"recovered: {summary['recovered']}")
        print(f"lost: {summary['lost']}")
        print(f"net_recovered: {summary['net_recovered']}")
        print(f"page_recall_at_k: {summary['page_recall_at_k']}")
        print(f"doc_recall_at_k: {summary['doc_recall_at_k']}")


if __name__ == "__main__":
    main()
