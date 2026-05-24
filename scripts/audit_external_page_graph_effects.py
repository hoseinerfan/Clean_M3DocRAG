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
            "Audit page-level effects of an external page graph, such as LayoutLMv3 or "
            "DocGraphLM kNN edges. The audit compares a baseline and candidate run, then "
            "checks whether gold pages are directly reachable from baseline retrieved pages."
        )
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--external-page-graph-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--source-top-pages", type=int, default=1000)
    parser.add_argument("--hit-k", type=int, default=4)
    parser.add_argument("--topn", type=int, default=30)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_prediction(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "predictions" in payload:
        payload = payload["predictions"]
    if isinstance(payload, list):
        iterable = enumerate(payload)
    elif isinstance(payload, dict):
        iterable = payload.items()
    else:
        raise TypeError(f"Prediction payload must be a list or dict: {path}")
    out: dict[str, dict[str, Any]] = {}
    for key, row in iterable:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("qid", "")).strip() or str(key).strip()
        if qid:
            out[qid] = row
    return out


def parse_page_uid(value: str) -> tuple[str, int | None]:
    if "_page" not in value:
        return value, None
    doc_id, raw_page_idx = value.rsplit("_page", 1)
    try:
        return doc_id, int(raw_page_idx)
    except ValueError:
        return doc_id, None


def page_uid(doc_id: str, page_idx: int) -> str:
    return f"{doc_id}_page{int(page_idx)}"


def parse_page_row(row: Any) -> tuple[str, int, float] | None:
    if not isinstance(row, list) or len(row) < 2:
        return None
    doc_id = str(row[0]).strip()
    if not doc_id:
        return None
    try:
        page_idx = int(row[1])
    except (TypeError, ValueError):
        return None
    score = 0.0
    if len(row) >= 3:
        try:
            score = float(row[2])
        except (TypeError, ValueError):
            score = 0.0
    return doc_id, page_idx, score


def ranked_unique_pages(row: dict[str, Any], top_pages: int) -> list[tuple[str, int, float, int]]:
    out = []
    seen: set[str] = set()
    for raw in row.get("page_retrieval_results", []):
        parsed = parse_page_row(raw)
        if parsed is None:
            continue
        doc_id, page_idx, score = parsed
        uid = page_uid(doc_id, page_idx)
        if uid in seen:
            continue
        seen.add(uid)
        out.append((doc_id, page_idx, score, len(out) + 1))
        if len(out) >= top_pages:
            break
    return out


def first_page_rank(row: dict[str, Any], gold_pages: set[str]) -> int | None:
    for _doc_id, _page_idx, _score, rank in ranked_unique_pages(row, 10**9):
        if page_uid(_doc_id, _page_idx) in gold_pages:
            return rank
    return None


def hit_at_k(row: dict[str, Any], gold_pages: set[str], k: int) -> bool:
    rank = first_page_rank(row, gold_pages)
    return rank is not None and rank <= k


def movement_for_hit(base_rank: int | None, cand_rank: int | None, hit_k: int) -> str:
    base_hit = base_rank is not None and base_rank <= hit_k
    cand_hit = cand_rank is not None and cand_rank <= hit_k
    if not base_hit and cand_hit:
        return "recovered"
    if base_hit and not cand_hit:
        return "lost"
    if base_rank is None and cand_rank is None:
        return "missing_in_both"
    if base_rank is not None and cand_rank is not None and cand_rank < base_rank:
        return "improved_rank"
    if base_rank is not None and cand_rank is not None and cand_rank > base_rank:
        return "worsened_rank"
    return "unchanged"


def gold_page_uids(row: dict[str, Any]) -> list[str]:
    metadata = row.get("metadata", {})
    uids = {
        str(value).strip()
        for value in metadata.get("gold_page_uids", [])
        if str(value).strip()
    }
    for ctx in row.get("supporting_context", []):
        if not isinstance(ctx, dict):
            continue
        doc_id = str(ctx.get("doc_id", "")).strip()
        page_idx = ctx.get("page_idx", ctx.get("page_id"))
        if doc_id and page_idx is not None:
            try:
                uids.add(page_uid(doc_id, int(page_idx)))
            except (TypeError, ValueError):
                pass
    return sorted(uids)


def metadata_value(row: dict[str, Any], dotted_key: str) -> str:
    value: Any = row
    for part in dotted_key.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            value = None
            break
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def load_external_edges(path: Path) -> dict[str, Any]:
    by_source_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    incoming_by_target_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scores: list[float] = []
    same_doc_count = 0
    cross_doc_count = 0
    rows = read_jsonl(path)
    for row in rows:
        source_uid = str(row.get("source_page_uid", "")).strip()
        target_uid = str(row.get("target_page_uid", "")).strip()
        if not source_uid or not target_uid:
            continue
        source_doc = str(row.get("source_doc_id", "")).strip()
        target_doc = str(row.get("target_doc_id", "")).strip()
        if not source_doc:
            source_doc, _ = parse_page_uid(source_uid)
        if not target_doc:
            target_doc, _ = parse_page_uid(target_uid)
        try:
            score = float(row.get("score", row.get("weight", 1.0)))
        except (TypeError, ValueError):
            score = 1.0
        edge = {
            "source_page_uid": source_uid,
            "target_page_uid": target_uid,
            "source_doc_id": source_doc,
            "target_doc_id": target_doc,
            "score": score,
            "weight": float(row.get("weight", score) or score),
            "edge_type": str(row.get("edge_type", row.get("type", ""))).strip(),
            "same_doc": source_doc == target_doc,
        }
        scores.append(score)
        if edge["same_doc"]:
            same_doc_count += 1
        else:
            cross_doc_count += 1
        by_source_page[source_uid].append(edge)
        incoming_by_target_page[target_uid].append(edge)

    for source_uid in list(by_source_page):
        by_source_page[source_uid] = sorted(
            by_source_page[source_uid],
            key=lambda edge: (-float(edge["score"]), edge["target_page_uid"]),
        )

    summary = {
        "edge_count": sum(len(v) for v in by_source_page.values()),
        "source_page_count": len(by_source_page),
        "target_page_count": len(incoming_by_target_page),
        "same_doc_edge_count": same_doc_count,
        "cross_doc_edge_count": cross_doc_count,
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "score_mean": statistics.fmean(scores) if scores else None,
        "score_median": statistics.median(scores) if scores else None,
    }
    return {
        "by_source_page": dict(by_source_page),
        "incoming_by_target_page": dict(incoming_by_target_page),
        "summary": summary,
    }


def source_edge_stats(
    source_pages: list[tuple[str, int, float, int]],
    by_source_page: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    pages_with_edges = 0
    target_pages = set()
    target_docs = set()
    same_doc_edges = 0
    cross_doc_edges = 0
    scores = []
    for doc_id, page_idx, _score, _rank in source_pages:
        edges = by_source_page.get(page_uid(doc_id, page_idx), [])
        if not edges:
            continue
        pages_with_edges += 1
        for edge in edges:
            target_pages.add(edge["target_page_uid"])
            target_docs.add(edge["target_doc_id"])
            scores.append(float(edge["score"]))
            if edge["same_doc"]:
                same_doc_edges += 1
            else:
                cross_doc_edges += 1
    return {
        "source_pages_with_edges": pages_with_edges,
        "linked_target_page_count": len(target_pages),
        "linked_target_doc_count": len(target_docs),
        "same_doc_edge_count": same_doc_edges,
        "cross_doc_edge_count": cross_doc_edges,
        "edge_score_mean": statistics.fmean(scores) if scores else None,
        "edge_score_max": max(scores) if scores else None,
    }


def first_edge_to_gold(
    source_pages: list[tuple[str, int, float, int]],
    by_source_page: dict[str, list[dict[str, Any]]],
    gold_pages: set[str],
) -> dict[str, Any] | None:
    for doc_id, page_idx, _score, rank in source_pages:
        uid = page_uid(doc_id, page_idx)
        matches = [edge for edge in by_source_page.get(uid, []) if edge["target_page_uid"] in gold_pages]
        if not matches:
            continue
        best = sorted(matches, key=lambda edge: (-float(edge["score"]), edge["target_page_uid"]))[0]
        return {
            "source_page_rank": rank,
            "source_page_uid": uid,
            "source_doc_id": doc_id,
            "target_page_uid": best["target_page_uid"],
            "target_doc_id": best["target_doc_id"],
            "score": best["score"],
            "same_doc": best["same_doc"],
        }
    return None


def support_bucket(case: dict[str, Any]) -> str:
    if case["gold_reachable_from_baseline_sources"]:
        if case["first_edge_to_gold"] and case["first_edge_to_gold"].get("same_doc"):
            return "gold_reachable_same_doc"
        return "gold_reachable_cross_doc"
    if case["gold_has_any_incoming_edge"]:
        return "gold_has_global_inlinks_only"
    return "gold_has_no_inlinks"


def summarize_group(cases: list[dict[str, Any]]) -> dict[str, Any]:
    movement = Counter(row["movement"] for row in cases)
    return {
        "n": len(cases),
        "recovered": movement.get("recovered", 0),
        "lost": movement.get("lost", 0),
        "improved_rank": movement.get("improved_rank", 0),
        "worsened_rank": movement.get("worsened_rank", 0),
        "unchanged": movement.get("unchanged", 0),
        "missing_in_both": movement.get("missing_in_both", 0),
        "baseline_hit_at_k": sum(1 for row in cases if row["baseline_hit_at_k"]),
        "candidate_hit_at_k": sum(1 for row in cases if row["candidate_hit_at_k"]),
        "gold_reachable_from_sources": sum(
            1 for row in cases if row["gold_reachable_from_baseline_sources"]
        ),
    }


def markdown_table(rows: list[list[Any]]) -> list[str]:
    if not rows:
        return []
    lines = [
        "| " + " | ".join(str(value) for value in rows[0]) + " |",
        "| " + " | ".join("---" for _ in rows[0]) + " |",
    ]
    for row in rows[1:]:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def write_markdown(path: Path, payload: dict[str, Any], topn: int) -> None:
    lines = [
        "# External Page Graph Effect Audit",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in [
        "n_qids",
        "hit_k",
        "source_top_pages",
        "baseline_hit_at_k",
        "candidate_hit_at_k",
        "gold_has_any_incoming_edge_count",
        "gold_reachable_from_baseline_sources_count",
    ]:
        lines.append(f"| {key} | {payload.get(key)} |")

    lines.extend(["", "## Graph Summary", ""])
    graph_rows = [["metric", "value"]]
    for key, value in payload["graph_summary"].items():
        graph_rows.append([key, value])
    lines.extend(markdown_table(graph_rows))

    lines.extend(["", "## Movement", ""])
    movement_rows = [["movement", "count"]]
    for key, value in sorted(payload["movement_counts"].items(), key=lambda item: (-item[1], item[0])):
        movement_rows.append([key, value])
    lines.extend(markdown_table(movement_rows))

    lines.extend(["", "## By Graph Support", ""])
    support_rows = [
        [
            "support_bucket",
            "n",
            "recovered",
            "lost",
            "improved",
            "worsened",
            "candidate_hit@k",
            "gold_reachable",
        ]
    ]
    for key, value in sorted(payload["by_support_bucket"].items()):
        support_rows.append(
            [
                key,
                value["n"],
                value["recovered"],
                value["lost"],
                value["improved_rank"],
                value["worsened_rank"],
                value["candidate_hit_at_k"],
                value["gold_reachable_from_sources"],
            ]
        )
    lines.extend(markdown_table(support_rows))

    lines.extend(["", "## By Metadata Type", ""])
    type_rows = [["type", "n", "recovered", "lost", "improved", "worsened", "gold_reachable"]]
    for key, value in sorted(payload["by_metadata_type"].items(), key=lambda item: (-item[1]["n"], item[0])):
        type_rows.append(
            [
                key or "UNKNOWN",
                value["n"],
                value["recovered"],
                value["lost"],
                value["improved_rank"],
                value["worsened_rank"],
                value["gold_reachable_from_sources"],
            ]
        )
    lines.extend(markdown_table(type_rows))

    for title, key_name in [
        ("Recovered", "top_recovered"),
        ("Lost", "top_lost"),
        ("Improved Rank", "top_improved_rank"),
        ("Worsened Rank", "top_worsened_rank"),
    ]:
        lines.extend(["", f"## Top {title}", ""])
        for row in payload[key_name][:topn]:
            edge = row.get("first_edge_to_gold") or {}
            lines.append(
                f"- `{row['qid']}` base={row['baseline_rank']} cand={row['candidate_rank']} "
                f"delta={row['rank_delta']} type=`{row['metadata_type'] or 'UNKNOWN'}` "
                f"support=`{row['support_bucket']}`"
            )
            lines.append(f"  - question: {row['question']}")
            lines.append(f"  - gold_pages: {row['gold_page_uids']}")
            if edge:
                lines.append(
                    "  - first_edge: "
                    f"`{edge['source_page_uid']}` rank={edge['source_page_rank']} -> "
                    f"`{edge['target_page_uid']}` score={edge['score']:.4f} "
                    f"same_doc={edge['same_doc']}"
                )
            lines.append(
                "  - source_edge_stats: "
                f"pages_with_edges={row['source_pages_with_edges']}, "
                f"targets={row['linked_target_page_count']}, "
                f"same_doc_edges={row['same_doc_edge_count']}, "
                f"cross_doc_edges={row['cross_doc_edge_count']}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    baseline = load_prediction(Path(args.baseline))
    candidate = load_prediction(Path(args.candidate))
    gold_by_qid = {str(row["qid"]): row for row in read_jsonl(Path(args.gold))}
    graph = load_external_edges(Path(args.external_page_graph_jsonl))
    by_source_page = graph["by_source_page"]
    incoming_by_target_page = graph["incoming_by_target_page"]

    cases = []
    qids = sorted(set(baseline) & set(candidate) & set(gold_by_qid))
    for qid in qids:
        gold = gold_by_qid[qid]
        gold_pages = set(gold_page_uids(gold))
        if not gold_pages:
            continue
        base_rank = first_page_rank(baseline[qid], gold_pages)
        cand_rank = first_page_rank(candidate[qid], gold_pages)
        movement = movement_for_hit(base_rank, cand_rank, int(args.hit_k))
        source_pages = ranked_unique_pages(baseline[qid], max(1, int(args.source_top_pages)))
        edge_stats = source_edge_stats(source_pages, by_source_page)
        first_edge = first_edge_to_gold(source_pages, by_source_page, gold_pages)
        incoming_edges = [
            edge
            for gold_uid in gold_pages
            for edge in incoming_by_target_page.get(gold_uid, [])
        ]
        case = {
            "qid": qid,
            "question": str(gold.get("question", baseline[qid].get("question", ""))).strip(),
            "metadata_type": metadata_value(gold, "metadata.type"),
            "metadata_domain": metadata_value(gold, "metadata.domain"),
            "gold_page_uids": sorted(gold_pages),
            "baseline_rank": base_rank,
            "candidate_rank": cand_rank,
            "rank_delta": None if base_rank is None or cand_rank is None else int(base_rank) - int(cand_rank),
            "movement": movement,
            "baseline_hit_at_k": hit_at_k(baseline[qid], gold_pages, int(args.hit_k)),
            "candidate_hit_at_k": hit_at_k(candidate[qid], gold_pages, int(args.hit_k)),
            "gold_has_any_incoming_edge": bool(incoming_edges),
            "gold_incoming_edge_count": len(incoming_edges),
            "gold_reachable_from_baseline_sources": first_edge is not None,
            "first_edge_to_gold": first_edge,
            **edge_stats,
        }
        case["support_bucket"] = support_bucket(case)
        cases.append(case)

    by_support: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_support[case["support_bucket"]].append(case)
        by_type[case["metadata_type"] or "UNKNOWN"].append(case)
        by_domain[case["metadata_domain"] or "UNKNOWN"].append(case)

    recovered = sorted(
        [case for case in cases if case["movement"] == "recovered"],
        key=lambda row: (row["candidate_rank"] or 10**9, row["qid"]),
    )
    lost = sorted(
        [case for case in cases if case["movement"] == "lost"],
        key=lambda row: (row["baseline_rank"] or 10**9, row["candidate_rank"] or 10**9, row["qid"]),
    )
    improved_rank = sorted(
        [case for case in cases if case["movement"] == "improved_rank"],
        key=lambda row: (-(row["rank_delta"] or 0), row["candidate_rank"] or 10**9, row["qid"]),
    )
    worsened_rank = sorted(
        [case for case in cases if case["movement"] == "worsened_rank"],
        key=lambda row: ((row["rank_delta"] or 0), row["candidate_rank"] or 10**9, row["qid"]),
    )

    payload = {
        "n_qids": len(cases),
        "hit_k": int(args.hit_k),
        "source_top_pages": int(args.source_top_pages),
        "graph_summary": graph["summary"],
        "movement_counts": dict(Counter(case["movement"] for case in cases)),
        "baseline_hit_at_k": sum(1 for case in cases if case["baseline_hit_at_k"]),
        "candidate_hit_at_k": sum(1 for case in cases if case["candidate_hit_at_k"]),
        "gold_has_any_incoming_edge_count": sum(
            1 for case in cases if case["gold_has_any_incoming_edge"]
        ),
        "gold_reachable_from_baseline_sources_count": sum(
            1 for case in cases if case["gold_reachable_from_baseline_sources"]
        ),
        "by_support_bucket": {
            key: summarize_group(value) for key, value in sorted(by_support.items())
        },
        "by_metadata_type": {
            key: summarize_group(value) for key, value in sorted(by_type.items())
        },
        "by_metadata_domain": {
            key: summarize_group(value) for key, value in sorted(by_domain.items())
        },
        "top_recovered": recovered[: int(args.topn)],
        "top_lost": lost[: int(args.topn)],
        "top_improved_rank": improved_rank[: int(args.topn)],
        "top_worsened_rank": worsened_rank[: int(args.topn)],
        "all_cases": cases,
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(Path(args.output_md), payload, int(args.topn))
    print("n_qids", payload["n_qids"])
    print("movement_counts", payload["movement_counts"])
    print("graph_summary", payload["graph_summary"])
    print("gold_has_any_incoming_edge_count", payload["gold_has_any_incoming_edge_count"])
    print("gold_reachable_from_baseline_sources_count", payload["gold_reachable_from_baseline_sources_count"])
    print("saved_json", args.output_json)
    print("saved_md", args.output_md)


if __name__ == "__main__":
    main()
