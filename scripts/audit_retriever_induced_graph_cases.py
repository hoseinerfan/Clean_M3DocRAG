#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit why a retriever-induced graph view recovers some page-localization "
            "failures. The script compares baseline/candidate movement and summarizes "
            "whether gold pages are connected to high-ranked seed pages by SPLADE-kNN edges."
        )
    )
    parser.add_argument("--gold", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--edges-jsonl", required=True)
    parser.add_argument("--dense-prediction-json", default="")
    parser.add_argument("--sparse-prediction-json", default="")
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument(
        "--edge-source-topk",
        type=int,
        default=20,
        help="Use this many top baseline pages as query-local edge sources. Default: 20.",
    )
    parser.add_argument(
        "--rank-cutoffs",
        type=int,
        nargs="+",
        default=[4, 5, 10, 20, 50, 100],
        help="Rank cutoffs for source-rank features.",
    )
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


def page_doc_id(uid: str) -> str:
    return uid.rsplit("_page", 1)[0]


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


def first_rank(ranked: list[str], gold: set[str]) -> int | None:
    for idx, uid in enumerate(ranked, start=1):
        if uid in gold:
            return idx
    return None


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


def rank_at_cutoffs(rank: int | None, cutoffs: list[int], prefix: str) -> dict[str, bool]:
    return {f"{prefix}_at_{cutoff}": rank is not None and rank <= cutoff for cutoff in cutoffs}


def load_edges(path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)
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


def gold_edge_features(
    *,
    gold_pages: set[str],
    source_pages: list[str],
    outgoing: dict[str, list[dict[str, Any]]],
    incoming: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    source_set = set(source_pages)
    gold_incoming_from_sources: list[dict[str, Any]] = []
    source_neighbor_gold_count = 0
    source_neighbor_gold_max_score = 0.0

    for source in source_pages:
        for edge in outgoing.get(source, []):
            if edge["target_page_uid"] in gold_pages:
                source_neighbor_gold_count += 1
                source_neighbor_gold_max_score = max(source_neighbor_gold_max_score, float(edge["score"]))
                gold_incoming_from_sources.append(edge)

    gold_outgoing_count = sum(len(outgoing.get(uid, [])) for uid in gold_pages)
    gold_incoming_count = sum(len(incoming.get(uid, [])) for uid in gold_pages)
    gold_incoming_from_sources_same_doc = sum(1 for edge in gold_incoming_from_sources if edge["same_doc"])
    gold_incoming_from_sources_cross_doc = sum(1 for edge in gold_incoming_from_sources if not edge["same_doc"])
    gold_incoming_any_source_count = sum(
        1
        for uid in gold_pages
        for edge in incoming.get(uid, [])
        if edge["source_page_uid"] in source_set
    )

    return {
        "gold_incoming_edge_count": gold_incoming_count,
        "gold_outgoing_edge_count": gold_outgoing_count,
        "gold_neighbor_of_top_source_count": source_neighbor_gold_count,
        "gold_neighbor_of_top_source_any": source_neighbor_gold_count > 0,
        "gold_neighbor_of_top_source_max_score": source_neighbor_gold_max_score if source_neighbor_gold_count else None,
        "gold_neighbor_of_top_source_same_doc_count": gold_incoming_from_sources_same_doc,
        "gold_neighbor_of_top_source_cross_doc_count": gold_incoming_from_sources_cross_doc,
        "gold_incoming_from_top_source_count": gold_incoming_any_source_count,
    }


def top_values(values: list[Any], limit: int = 8) -> str:
    counts = Counter(str(value) for value in values if value not in (None, "", []))
    if not counts:
        return "-"
    return ", ".join(f"{key}:{count}" for key, count in counts.most_common(limit))


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def group_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    groups = sorted({case["movement"] for case in cases})
    numeric_features = [
        "baseline_first_gold_page_rank",
        "candidate_first_gold_page_rank",
        "dense_first_gold_page_rank",
        "sparse_first_gold_page_rank",
        "gold_incoming_edge_count",
        "gold_outgoing_edge_count",
        "gold_neighbor_of_top_source_count",
        "gold_neighbor_of_top_source_max_score",
        "gold_neighbor_of_top_source_same_doc_count",
        "gold_neighbor_of_top_source_cross_doc_count",
    ]
    boolean_features = [
        "gold_neighbor_of_top_source_any",
        "dense_gold_at_20",
        "sparse_gold_at_20",
        "sparse_gold_at_100",
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
            group: top_values(
                [getter(case) for case in cases if case["movement"] == group]
            )
            for group in groups
        }
    return summary


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


def main() -> None:
    args = parse_args()
    gold_rows = read_jsonl(Path(args.gold))
    gold_by_qid = {str(row["qid"]): row for row in gold_rows}
    baseline = load_prediction(Path(args.baseline))
    candidate = load_prediction(Path(args.candidate))
    dense = load_prediction(Path(args.dense_prediction_json)) if args.dense_prediction_json else {}
    sparse = load_prediction(Path(args.sparse_prediction_json)) if args.sparse_prediction_json else {}
    outgoing, incoming = load_edges(Path(args.edges_jsonl))

    qids = sorted(set(gold_by_qid) & set(baseline) & set(candidate))
    cases: list[dict[str, Any]] = []
    edge_source_topk = int(args.edge_source_topk)
    rank_cutoffs = [int(value) for value in args.rank_cutoffs]
    max_rank_needed = max(rank_cutoffs + [edge_source_topk, int(args.topk)])

    for qid in qids:
        gold_row = gold_by_qid[qid]
        gold_pages = gold_page_uids(gold_row)
        baseline_pages = ranked_pages(baseline[qid], max_rank_needed)
        candidate_pages = ranked_pages(candidate[qid], max_rank_needed)
        dense_pages = ranked_pages(dense[qid], max_rank_needed) if qid in dense else []
        sparse_pages = ranked_pages(sparse[qid], max_rank_needed) if qid in sparse else []

        baseline_rank = first_rank(baseline_pages, gold_pages)
        candidate_rank = first_rank(candidate_pages, gold_pages)
        dense_rank = first_rank(dense_pages, gold_pages) if dense_pages else None
        sparse_rank = first_rank(sparse_pages, gold_pages) if sparse_pages else None
        movement = movement_for_hit(baseline_rank, candidate_rank, int(args.topk))
        source_pages = baseline_pages[:edge_source_topk]
        edge_features = gold_edge_features(
            gold_pages=gold_pages,
            source_pages=source_pages,
            outgoing=outgoing,
            incoming=incoming,
        )

        case = {
            "qid": qid,
            "question": gold_row.get("question", ""),
            "metadata": gold_row.get("metadata", {}),
            "gold_page_uids": sorted(gold_pages),
            "movement": movement,
            "baseline_first_gold_page_rank": baseline_rank,
            "candidate_first_gold_page_rank": candidate_rank,
            "dense_first_gold_page_rank": dense_rank,
            "sparse_first_gold_page_rank": sparse_rank,
            **rank_at_cutoffs(dense_rank, rank_cutoffs, "dense_gold"),
            **rank_at_cutoffs(sparse_rank, rank_cutoffs, "sparse_gold"),
            **edge_features,
            "baseline_top_pages": baseline_pages[:10],
            "candidate_top_pages": candidate_pages[:10],
        }
        cases.append(case)

    summary = group_summary(cases)
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
    recovered = [
        case for case in cases if case["movement"] == "recovered"
    ]
    recovered.sort(
        key=lambda case: (
            case["candidate_first_gold_page_rank"] or 10**9,
            -(case.get("gold_neighbor_of_top_source_max_score") or 0.0),
            case["qid"],
        )
    )
    print("\ntop_recovered_examples")
    for case in recovered[: int(args.topn)]:
        print(
            json.dumps(
                {
                    "qid": case["qid"],
                    "baseline_rank": case["baseline_first_gold_page_rank"],
                    "candidate_rank": case["candidate_first_gold_page_rank"],
                    "sparse_rank": case["sparse_first_gold_page_rank"],
                    "gold_neighbor_of_top_source_count": case["gold_neighbor_of_top_source_count"],
                    "gold_neighbor_of_top_source_max_score": case["gold_neighbor_of_top_source_max_score"],
                    "question": case["question"],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
