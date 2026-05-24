#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect the pages that block gold pages from entering top-k. The audit is "
            "designed for rankable wrong-page subsets where the gold page is often at "
            "rank k+1. It reports same-doc structure, page-distance, dense/SPLADE ranks, "
            "and retriever-induced graph edges between top-k blocker pages and gold pages."
        )
    )
    parser.add_argument("--gold", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", default="")
    parser.add_argument("--dense-prediction-json", default="")
    parser.add_argument("--sparse-prediction-json", default="")
    parser.add_argument("--edges-jsonl", default="")
    parser.add_argument("--doc-pages-jsonl", default="")
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--rank-lookup-depth", type=int, default=1000)
    parser.add_argument("--adjacent-window", type=int, default=1)
    parser.add_argument("--topn", type=int, default=20)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-jsonl", default="")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_prediction(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "predictions" in payload and isinstance(
        payload["predictions"],
        (dict, list),
    ):
        payload = payload["predictions"]

    rows_by_qid: dict[str, dict[str, Any]] = {}
    if isinstance(payload, list):
        iterable: Any = enumerate(payload)
    elif isinstance(payload, dict):
        iterable = payload.items()
    else:
        raise TypeError(f"Prediction JSON must be a list or object: {path}")

    for raw_key, row in iterable:
        if not isinstance(row, dict):
            raise TypeError(f"Prediction row must be an object: {path} key={raw_key!r}")
        qid = str(row.get("qid", "")).strip() or str(raw_key).strip()
        if qid:
            rows_by_qid[qid] = row
    return rows_by_qid


def page_uid(doc_id: object, page_idx: object) -> str:
    return f"{doc_id}_page{int(page_idx)}"


def parse_page_uid(uid: str) -> tuple[str, int]:
    if "_page" not in uid:
        raise ValueError(f"Invalid page uid: {uid}")
    doc_id, page_idx = uid.rsplit("_page", 1)
    return doc_id, int(page_idx)


def page_doc_id(uid: str) -> str:
    return parse_page_uid(uid)[0]


def page_index(uid: str) -> int:
    return parse_page_uid(uid)[1]


def ranked_pages(row: dict[str, Any], limit: int = 0) -> list[str]:
    pages: list[str] = []
    seen: set[str] = set()
    for item in row.get("page_retrieval_results", []):
        if not isinstance(item, list) or len(item) < 2:
            continue
        try:
            uid = page_uid(item[0], item[1])
        except (TypeError, ValueError):
            continue
        if uid in seen:
            continue
        seen.add(uid)
        pages.append(uid)
        if limit > 0 and len(pages) >= limit:
            break
    return pages


def rank_map(row: dict[str, Any], limit: int) -> dict[str, int]:
    return {uid: rank for rank, uid in enumerate(ranked_pages(row, limit), start=1)}


def first_rank(ranked: list[str], gold: set[str]) -> int | None:
    for idx, uid in enumerate(ranked, start=1):
        if uid in gold:
            return idx
    return None


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


def movement_for_hit(
    baseline_rank: int | None,
    candidate_rank: int | None,
    topk: int,
) -> str:
    baseline_hit = baseline_rank is not None and baseline_rank <= topk
    candidate_hit = candidate_rank is not None and candidate_rank <= topk
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


def load_edges(path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not path:
        return outgoing, incoming
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            source = str(row.get("source_page_uid", "")).strip()
            target = str(row.get("target_page_uid", "")).strip()
            if not source or not target:
                continue
            score = float(row.get("score", row.get("weight", 0.0)) or 0.0)
            edge = {
                "source_page_uid": source,
                "target_page_uid": target,
                "score": score,
                "same_doc": page_doc_id(source) == page_doc_id(target),
            }
            outgoing[source].append(edge)
            incoming[target].append(edge)
    return outgoing, incoming


def load_page_map(path: Path) -> dict[str, dict[str, Any]]:
    if not path or not path.exists():
        return {}
    page_map: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        doc_id = str(row.get("doc_id", "")).strip()
        page_idx = row.get("page_idx", row.get("page_id"))
        uid = str(row.get("page_uid", "") or "").strip()
        if not uid and doc_id and page_idx is not None:
            uid = page_uid(doc_id, page_idx)
        if uid:
            page_map[uid] = row
    return page_map


def token_count(text: object) -> int:
    if text is None:
        return 0
    if isinstance(text, (list, tuple)):
        text = " ".join(str(item) for item in text if item is not None)
    elif isinstance(text, dict):
        text = json.dumps(text, ensure_ascii=False, sort_keys=True)
    return len(re.findall(r"[A-Za-z0-9]+", str(text)))


def page_text_token_count(page_map: dict[str, dict[str, Any]], uid: str) -> int | None:
    row = page_map.get(uid)
    if not row:
        return None
    pieces = [
        row.get("ocr_text"),
        row.get("vlm_text"),
        row.get("markdown"),
        row.get("text"),
    ]
    total = sum(token_count(piece) for piece in pieces)
    return total


def min_distance_to_gold(uid: str, gold_pages: set[str]) -> int | None:
    doc_id, idx = parse_page_uid(uid)
    distances = [
        abs(idx - page_index(gold_uid))
        for gold_uid in gold_pages
        if page_doc_id(gold_uid) == doc_id
    ]
    if not distances:
        return None
    return min(distances)


def relative_position_to_gold(uid: str, gold_pages: set[str]) -> str:
    doc_id, idx = parse_page_uid(uid)
    same_doc_gold = [
        page_index(gold_uid)
        for gold_uid in gold_pages
        if page_doc_id(gold_uid) == doc_id
    ]
    if not same_doc_gold:
        return "cross_doc"
    if idx in same_doc_gold:
        return "gold"
    if idx < min(same_doc_gold):
        return "before_gold"
    if idx > max(same_doc_gold):
        return "after_gold"
    return "between_gold_pages"


def edge_score(source: str, targets: set[str], outgoing: dict[str, list[dict[str, Any]]]) -> float | None:
    scores = [
        float(edge["score"])
        for edge in outgoing.get(source, [])
        if edge["target_page_uid"] in targets
    ]
    if not scores:
        return None
    return max(scores)


def edge_count(source: str, targets: set[str], outgoing: dict[str, list[dict[str, Any]]]) -> int:
    return sum(1 for edge in outgoing.get(source, []) if edge["target_page_uid"] in targets)


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    groups = sorted({case["movement"] for case in cases})
    numeric_features = [
        "baseline_first_gold_page_rank",
        "candidate_first_gold_page_rank",
        "topk_same_doc_gold_count",
        "topk_same_doc_gold_frac",
        "topk_cross_doc_count",
        "topk_adjacent_gold_count",
        "topk_mean_distance_to_gold",
        "topk_min_distance_to_gold",
        "topk_before_gold_count",
        "topk_after_gold_count",
        "topk_blocker_to_gold_edge_count",
        "topk_gold_to_blocker_edge_count",
        "topk_blocker_to_gold_max_score",
        "topk_gold_to_blocker_max_score",
        "topk_dense_rank_advantage_count",
        "topk_sparse_rank_advantage_count",
        "topk_mean_dense_rank",
        "topk_mean_sparse_rank",
        "gold_mean_text_token_count",
        "topk_mean_text_token_count",
    ]
    boolean_features = [
        "topk_all_same_doc_gold",
        "topk_any_adjacent_gold",
        "topk_any_blocker_to_gold_edge",
        "topk_any_gold_to_blocker_edge",
    ]
    categorical_features = [
        ("repo_slug", lambda case: case.get("metadata", {}).get("repo_slug")),
        ("query_types", lambda case: case.get("metadata", {}).get("query_types")),
        ("content_type", lambda case: case.get("metadata", {}).get("content_type")),
    ]
    summary: dict[str, Any] = {
        "movement_counts": dict(Counter(case["movement"] for case in cases)),
        "numeric_feature_means": {},
        "boolean_feature_rates": {},
        "categorical_top_values": {},
    }
    for feature in numeric_features:
        summary["numeric_feature_means"][feature] = {
            group: mean(
                [
                    float(case[feature])
                    for case in cases
                    if case["movement"] == group and case.get(feature) is not None
                ]
            )
            for group in groups
        }
    for feature in boolean_features:
        summary["boolean_feature_rates"][feature] = {
            group: mean(
                [
                    1.0 if bool(case.get(feature)) else 0.0
                    for case in cases
                    if case["movement"] == group
                ]
            )
            for group in groups
        }
    for name, getter in categorical_features:
        summary["categorical_top_values"][name] = {
            group: top_values([getter(case) for case in cases if case["movement"] == group])
            for group in groups
        }
    return summary


def top_values(values: list[Any], limit: int = 8) -> str:
    counts = Counter(str(value) for value in values if value not in (None, "", []))
    if not counts:
        return "-"
    return ", ".join(f"{key}:{count}" for key, count in counts.most_common(limit))


def print_summary(summary: dict[str, Any]) -> None:
    print("movement_counts", summary["movement_counts"])
    print("\nnumeric_feature_means")
    for feature, values in summary["numeric_feature_means"].items():
        print(feature, values)
    print("\nboolean_feature_rates")
    for feature, values in summary["boolean_feature_rates"].items():
        print(feature, values)
    print("\ncategorical_top_values")
    for feature, values in summary["categorical_top_values"].items():
        print(feature, values)


def build_case(
    *,
    qid: str,
    gold_row: dict[str, Any],
    baseline_row: dict[str, Any],
    candidate_row: dict[str, Any] | None,
    dense_row: dict[str, Any] | None,
    sparse_row: dict[str, Any] | None,
    outgoing: dict[str, list[dict[str, Any]]],
    page_map: dict[str, dict[str, Any]],
    topk: int,
    lookup_depth: int,
    adjacent_window: int,
) -> dict[str, Any]:
    gold_pages = gold_page_uids(gold_row)
    baseline_pages = ranked_pages(baseline_row, lookup_depth)
    candidate_pages = ranked_pages(candidate_row or {}, lookup_depth)
    dense_ranks = rank_map(dense_row or {}, lookup_depth)
    sparse_ranks = rank_map(sparse_row or {}, lookup_depth)
    baseline_rank = first_rank(baseline_pages, gold_pages)
    candidate_rank = first_rank(candidate_pages, gold_pages) if candidate_row else None
    movement = (
        movement_for_hit(baseline_rank, candidate_rank, topk)
        if candidate_row
        else "baseline_only"
    )
    gold_dense_rank = first_rank(list(dense_ranks), gold_pages) if dense_ranks else None
    gold_sparse_rank = first_rank(list(sparse_ranks), gold_pages) if sparse_ranks else None

    top_pages = baseline_pages[:topk]
    blocker_rows: list[dict[str, Any]] = []
    same_doc_count = 0
    cross_doc_count = 0
    adjacent_count = 0
    distances: list[int] = []
    before_count = 0
    after_count = 0
    blocker_to_gold_total = 0
    gold_to_blocker_total = 0
    blocker_to_gold_scores: list[float] = []
    gold_to_blocker_scores: list[float] = []
    dense_advantage_count = 0
    sparse_advantage_count = 0
    dense_rank_values: list[float] = []
    sparse_rank_values: list[float] = []
    topk_token_counts: list[float] = []
    gold_token_counts = [
        float(value)
        for value in (page_text_token_count(page_map, uid) for uid in gold_pages)
        if value is not None
    ]

    for rank, uid in enumerate(top_pages, start=1):
        distance = min_distance_to_gold(uid, gold_pages)
        same_doc = distance is not None
        position = relative_position_to_gold(uid, gold_pages)
        if same_doc:
            same_doc_count += 1
            distances.append(int(distance))
            if distance <= adjacent_window:
                adjacent_count += 1
        else:
            cross_doc_count += 1
        if position == "before_gold":
            before_count += 1
        elif position == "after_gold":
            after_count += 1

        blocker_to_gold_count = edge_count(uid, gold_pages, outgoing)
        blocker_to_gold_score = edge_score(uid, gold_pages, outgoing)
        gold_to_blocker_count = sum(edge_count(gold_uid, {uid}, outgoing) for gold_uid in gold_pages)
        gold_to_blocker_score_values = [
            score
            for gold_uid in gold_pages
            for score in [edge_score(gold_uid, {uid}, outgoing)]
            if score is not None
        ]
        gold_to_blocker_score = max(gold_to_blocker_score_values) if gold_to_blocker_score_values else None
        blocker_to_gold_total += blocker_to_gold_count
        gold_to_blocker_total += gold_to_blocker_count
        if blocker_to_gold_score is not None:
            blocker_to_gold_scores.append(float(blocker_to_gold_score))
        if gold_to_blocker_score is not None:
            gold_to_blocker_scores.append(float(gold_to_blocker_score))

        dense_rank = dense_ranks.get(uid)
        sparse_rank = sparse_ranks.get(uid)
        if dense_rank is not None:
            dense_rank_values.append(float(dense_rank))
            if gold_dense_rank is not None and dense_rank < gold_dense_rank:
                dense_advantage_count += 1
        if sparse_rank is not None:
            sparse_rank_values.append(float(sparse_rank))
            if gold_sparse_rank is not None and sparse_rank < gold_sparse_rank:
                sparse_advantage_count += 1
        token_value = page_text_token_count(page_map, uid)
        if token_value is not None:
            topk_token_counts.append(float(token_value))

        blocker_rows.append(
            {
                "rank": rank,
                "page_uid": uid,
                "same_doc_as_gold": same_doc,
                "min_page_distance_to_gold": distance,
                "adjacent_to_gold": distance is not None and distance <= adjacent_window,
                "relative_position_to_gold": position,
                "dense_rank": dense_rank,
                "sparse_rank": sparse_rank,
                "dense_ranked_before_gold": (
                    dense_rank is not None
                    and gold_dense_rank is not None
                    and dense_rank < gold_dense_rank
                ),
                "sparse_ranked_before_gold": (
                    sparse_rank is not None
                    and gold_sparse_rank is not None
                    and sparse_rank < gold_sparse_rank
                ),
                "blocker_to_gold_edge_count": blocker_to_gold_count,
                "blocker_to_gold_edge_max_score": blocker_to_gold_score,
                "gold_to_blocker_edge_count": gold_to_blocker_count,
                "gold_to_blocker_edge_max_score": gold_to_blocker_score,
                "text_token_count": token_value,
            }
        )

    return {
        "qid": qid,
        "question": gold_row.get("question", ""),
        "metadata": gold_row.get("metadata", {}),
        "gold_page_uids": sorted(gold_pages),
        "movement": movement,
        "baseline_first_gold_page_rank": baseline_rank,
        "candidate_first_gold_page_rank": candidate_rank,
        "gold_dense_rank": gold_dense_rank,
        "gold_sparse_rank": gold_sparse_rank,
        "topk_same_doc_gold_count": same_doc_count,
        "topk_same_doc_gold_frac": same_doc_count / float(max(1, len(top_pages))),
        "topk_cross_doc_count": cross_doc_count,
        "topk_all_same_doc_gold": same_doc_count == len(top_pages) and bool(top_pages),
        "topk_adjacent_gold_count": adjacent_count,
        "topk_any_adjacent_gold": adjacent_count > 0,
        "topk_mean_distance_to_gold": mean([float(value) for value in distances]),
        "topk_min_distance_to_gold": min(distances) if distances else None,
        "topk_before_gold_count": before_count,
        "topk_after_gold_count": after_count,
        "topk_blocker_to_gold_edge_count": blocker_to_gold_total,
        "topk_gold_to_blocker_edge_count": gold_to_blocker_total,
        "topk_any_blocker_to_gold_edge": blocker_to_gold_total > 0,
        "topk_any_gold_to_blocker_edge": gold_to_blocker_total > 0,
        "topk_blocker_to_gold_max_score": max(blocker_to_gold_scores) if blocker_to_gold_scores else None,
        "topk_gold_to_blocker_max_score": max(gold_to_blocker_scores) if gold_to_blocker_scores else None,
        "topk_dense_rank_advantage_count": dense_advantage_count,
        "topk_sparse_rank_advantage_count": sparse_advantage_count,
        "topk_mean_dense_rank": mean(dense_rank_values),
        "topk_mean_sparse_rank": mean(sparse_rank_values),
        "gold_mean_text_token_count": mean(gold_token_counts),
        "topk_mean_text_token_count": mean(topk_token_counts),
        "topk_blockers": blocker_rows,
    }


def main() -> None:
    args = parse_args()
    gold_rows = read_jsonl(Path(args.gold))
    gold_by_qid = {str(row["qid"]): row for row in gold_rows}
    baseline = load_prediction(Path(args.baseline))
    candidate = load_prediction(Path(args.candidate)) if args.candidate else {}
    dense = load_prediction(Path(args.dense_prediction_json)) if args.dense_prediction_json else {}
    sparse = load_prediction(Path(args.sparse_prediction_json)) if args.sparse_prediction_json else {}
    outgoing, _incoming = (
        load_edges(Path(args.edges_jsonl))
        if args.edges_jsonl
        else (defaultdict(list), defaultdict(list))
    )
    page_map = load_page_map(Path(args.doc_pages_jsonl)) if args.doc_pages_jsonl else {}

    qids = sorted(set(gold_by_qid) & set(baseline))
    if candidate:
        qids = sorted(set(qids) & set(candidate))
    cases = [
        build_case(
            qid=qid,
            gold_row=gold_by_qid[qid],
            baseline_row=baseline[qid],
            candidate_row=candidate.get(qid) if candidate else None,
            dense_row=dense.get(qid),
            sparse_row=sparse.get(qid),
            outgoing=outgoing,
            page_map=page_map,
            topk=int(args.topk),
            lookup_depth=int(args.rank_lookup_depth),
            adjacent_window=int(args.adjacent_window),
        )
        for qid in qids
    ]
    summary = summarize_cases(cases)
    payload = {"summary": summary, "cases": cases}

    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.output_jsonl:
        path = Path(args.output_jsonl)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for case in cases:
                handle.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"n_qids {len(cases)}")
    print_summary(summary)
    print("\ntop_cases_by_blocker_to_gold_edges")
    ranked = sorted(
        cases,
        key=lambda case: (
            -(case.get("topk_blocker_to_gold_edge_count") or 0),
            -(case.get("topk_blocker_to_gold_max_score") or 0.0),
            case["qid"],
        ),
    )
    for case in ranked[: int(args.topn)]:
        print(
            json.dumps(
                {
                    "qid": case["qid"],
                    "movement": case["movement"],
                    "baseline_rank": case["baseline_first_gold_page_rank"],
                    "candidate_rank": case["candidate_first_gold_page_rank"],
                    "topk_same_doc_gold_count": case["topk_same_doc_gold_count"],
                    "topk_adjacent_gold_count": case["topk_adjacent_gold_count"],
                    "topk_blocker_to_gold_edge_count": case["topk_blocker_to_gold_edge_count"],
                    "topk_blocker_to_gold_max_score": case["topk_blocker_to_gold_max_score"],
                    "question": case["question"],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
