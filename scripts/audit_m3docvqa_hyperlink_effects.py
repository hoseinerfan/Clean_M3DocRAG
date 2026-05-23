#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit how PDF hyperlink graph edges affect M3DocVQA doc retrieval. "
            "The audit compares a no-link baseline against a hyperlink candidate and "
            "checks whether gold docs are reachable from retrieved source pages."
        )
    )
    parser.add_argument("--baseline", required=True, help="No-hyperlink prediction JSON.")
    parser.add_argument("--candidate", required=True, help="Hyperlink prediction JSON.")
    parser.add_argument("--gold", required=True, help="MMQA gold JSONL.")
    parser.add_argument("--hyperlink-edges-jsonl", required=True, help="PDF hyperlink edge JSONL.")
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
    rows: dict[str, dict[str, Any]] = {}
    for key, row in iterable:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("qid", "")).strip() or str(key).strip()
        if qid:
            rows[qid] = row
    return rows


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


def page_uid(doc_id: str, page_idx: int) -> str:
    return f"{doc_id}_page{int(page_idx)}"


def ranked_unique_pages(row: dict[str, Any], top_pages: int) -> list[tuple[str, int, float, int]]:
    rows = row.get("page_retrieval_results", [])
    out = []
    seen: set[str] = set()
    for raw in rows:
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


def first_unique_doc_rank(row: dict[str, Any], gold_doc_ids: set[str]) -> int | None:
    seen_docs: set[str] = set()
    rank = 0
    for raw in row.get("page_retrieval_results", []):
        parsed = parse_page_row(raw)
        if parsed is None:
            continue
        doc_id, _page_idx, _score = parsed
        if doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)
        rank += 1
        if doc_id in gold_doc_ids:
            return rank
    return None


def doc_hit_at_k(row: dict[str, Any], gold_doc_ids: set[str], k: int) -> bool:
    seen_docs: set[str] = set()
    for raw in row.get("page_retrieval_results", []):
        parsed = parse_page_row(raw)
        if parsed is None:
            continue
        doc_id, _page_idx, _score = parsed
        if doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)
        if len(seen_docs) > k:
            return False
        if doc_id in gold_doc_ids:
            return True
    return False


def movement_label(base_rank: int | None, cand_rank: int | None) -> str:
    if base_rank is None and cand_rank is None:
        return "missing_in_both"
    if base_rank is None:
        return "newly_found"
    if cand_rank is None:
        return "newly_lost"
    if cand_rank < base_rank:
        return "improved"
    if cand_rank > base_rank:
        return "worsened"
    return "unchanged"


def rank_delta(base_rank: int | None, cand_rank: int | None) -> int | None:
    if base_rank is None or cand_rank is None:
        return None
    return base_rank - cand_rank


def metadata_value(row: dict[str, Any], key: str) -> str:
    metadata = row.get("metadata", {})
    if isinstance(metadata, dict) and metadata.get(key) is not None:
        return str(metadata.get(key)).strip()
    if row.get(key) is not None:
        return str(row.get(key)).strip()
    return ""


def gold_doc_ids(row: dict[str, Any]) -> set[str]:
    values = set()
    for ctx in row.get("supporting_context", []):
        doc_id = str(ctx.get("doc_id", "")).strip() if isinstance(ctx, dict) else ""
        if doc_id:
            values.add(doc_id)
    return values


def gold_answer_strings(row: dict[str, Any]) -> list[str]:
    answers = []
    for answer in row.get("answers", []):
        if isinstance(answer, dict):
            value = answer.get("answer")
        else:
            value = answer
        text = str(value).strip()
        if text:
            answers.append(text)
    return answers


def load_hyperlink_edges(path: Path) -> dict[str, Any]:
    by_source_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    incoming_by_target_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows = read_jsonl(path)
    for row in rows:
        source_page_uid = str(row.get("source_page_uid", "")).strip()
        source_doc_id = str(row.get("source_doc_id", "")).strip()
        target_doc_id = str(row.get("target_doc_id", "")).strip()
        if not source_doc_id and source_page_uid and "_page" in source_page_uid:
            source_doc_id = source_page_uid.rsplit("_page", 1)[0]
        if not source_page_uid or not target_doc_id:
            continue
        edge = {
            "source_page_uid": source_page_uid,
            "source_doc_id": source_doc_id,
            "target_doc_id": target_doc_id,
            "target_wiki_title": str(row.get("target_wiki_title", "")).strip(),
            "raw_link_count": int(row.get("raw_link_count", 1) or 1),
        }
        by_source_page[source_page_uid].append(edge)
        if source_doc_id:
            by_source_doc[source_doc_id].append(edge)
        incoming_by_target_doc[target_doc_id].append(edge)
    return {
        "edge_count": sum(len(v) for v in by_source_page.values()),
        "by_source_page": dict(by_source_page),
        "by_source_doc": dict(by_source_doc),
        "incoming_by_target_doc": dict(incoming_by_target_doc),
    }


def first_source_link_to_gold(
    source_pages: list[tuple[str, int, float, int]],
    by_source_page: dict[str, list[dict[str, Any]]],
    gold_docs: set[str],
) -> dict[str, Any] | None:
    for doc_id, page_idx, _score, rank in source_pages:
        uid = page_uid(doc_id, page_idx)
        hits = [edge for edge in by_source_page.get(uid, []) if edge["target_doc_id"] in gold_docs]
        if hits:
            best = sorted(hits, key=lambda edge: (-int(edge["raw_link_count"]), edge["target_doc_id"]))[0]
            return {
                "source_page_rank": rank,
                "source_page_uid": uid,
                "source_doc_id": doc_id,
                "target_doc_id": best["target_doc_id"],
                "target_wiki_title": best["target_wiki_title"],
                "raw_link_count": best["raw_link_count"],
            }
    return None


def linked_target_doc_stats(
    source_pages: list[tuple[str, int, float, int]],
    by_source_page: dict[str, list[dict[str, Any]]],
) -> tuple[int, int, int]:
    source_pages_with_edges = 0
    linked_docs = set()
    raw_count = 0
    for doc_id, page_idx, _score, _rank in source_pages:
        edges = by_source_page.get(page_uid(doc_id, page_idx), [])
        if not edges:
            continue
        source_pages_with_edges += 1
        for edge in edges:
            linked_docs.add(edge["target_doc_id"])
            raw_count += int(edge["raw_link_count"])
    return source_pages_with_edges, len(linked_docs), raw_count


def support_bucket(case: dict[str, Any]) -> str:
    if case["gold_linked_from_baseline_sources"]:
        return "gold_linked_from_retrieved_sources"
    if case["gold_has_any_incoming_link"]:
        return "gold_has_global_inlinks_only"
    return "gold_has_no_inlinks"


def summarize_group(cases: list[dict[str, Any]]) -> dict[str, Any]:
    movement = Counter(row["movement"] for row in cases)
    return {
        "n": len(cases),
        "improved": movement.get("improved", 0),
        "worsened": movement.get("worsened", 0),
        "unchanged": movement.get("unchanged", 0),
        "newly_found": movement.get("newly_found", 0),
        "newly_lost": movement.get("newly_lost", 0),
        "missing_in_both": movement.get("missing_in_both", 0),
        "baseline_hit_at_k": sum(1 for row in cases if row["baseline_hit_at_k"]),
        "candidate_hit_at_k": sum(1 for row in cases if row["candidate_hit_at_k"]),
        "gold_linked_from_sources": sum(1 for row in cases if row["gold_linked_from_baseline_sources"]),
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
        "# M3DocVQA Hyperlink Effect Audit",
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
        "edge_count",
        "baseline_hit_at_k",
        "candidate_hit_at_k",
        "gold_has_any_incoming_link_count",
        "gold_linked_from_baseline_sources_count",
    ]:
        lines.append(f"| {key} | {payload.get(key)} |")

    lines.extend(["", "## Movement", ""])
    movement_rows = [["movement", "count"]]
    for key, value in sorted(payload["movement_counts"].items(), key=lambda item: (-item[1], item[0])):
        movement_rows.append([key, value])
    lines.extend(markdown_table(movement_rows))

    lines.extend(["", "## By Hyperlink Support", ""])
    support_rows = [
        [
            "support_bucket",
            "n",
            "improved",
            "worsened",
            "candidate_hit@k",
            "gold_linked_from_sources",
        ]
    ]
    for key, value in sorted(payload["by_support_bucket"].items()):
        support_rows.append(
            [
                key,
                value["n"],
                value["improved"],
                value["worsened"],
                value["candidate_hit_at_k"],
                value["gold_linked_from_sources"],
            ]
        )
    lines.extend(markdown_table(support_rows))

    lines.extend(["", "## By Question Type", ""])
    type_rows = [["type", "n", "improved", "worsened", "candidate_hit@k", "gold_linked_from_sources"]]
    for key, value in sorted(payload["by_question_type"].items(), key=lambda item: (-item[1]["n"], item[0])):
        type_rows.append(
            [
                key or "UNKNOWN",
                value["n"],
                value["improved"],
                value["worsened"],
                value["candidate_hit_at_k"],
                value["gold_linked_from_sources"],
            ]
        )
    lines.extend(markdown_table(type_rows))

    for heading, key_name in [
        ("Top Improved", "top_improved"),
        ("Top Worsened", "top_worsened"),
    ]:
        lines.extend(["", f"## {heading}", ""])
        for row in payload[key_name][:topn]:
            link = row.get("first_source_link_to_gold") or {}
            lines.append(
                f"- `{row['qid']}` {row['baseline_rank']} -> {row['candidate_rank']} "
                f"delta={row['rank_delta']} type=`{row['question_type'] or 'UNKNOWN'}` "
                f"support=`{row['support_bucket']}`"
            )
            lines.append(f"  - question: {row['question']}")
            lines.append(f"  - gold_docs: {row['gold_doc_ids']}")
            if link:
                lines.append(
                    "  - first_link: "
                    f"`{link['source_page_uid']}` rank={link['source_page_rank']} -> "
                    f"`{link['target_doc_id']}` ({link['target_wiki_title']}, "
                    f"raw_count={link['raw_link_count']})"
                )
            lines.append(
                "  - source_link_stats: "
                f"pages_with_edges={row['baseline_source_pages_with_edges']}, "
                f"linked_target_docs={row['baseline_linked_target_doc_count']}, "
                f"raw_links={row['baseline_raw_link_count_from_sources']}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    baseline = load_prediction(Path(args.baseline))
    candidate = load_prediction(Path(args.candidate))
    gold_rows = {row["qid"]: row for row in read_jsonl(Path(args.gold))}
    graph = load_hyperlink_edges(Path(args.hyperlink_edges_jsonl))
    by_source_page = graph["by_source_page"]
    incoming_by_target_doc = graph["incoming_by_target_doc"]

    qids = sorted(set(baseline) & set(candidate) & set(gold_rows))
    cases = []
    for qid in qids:
        gold = gold_rows[qid]
        gold_docs = gold_doc_ids(gold)
        if not gold_docs:
            continue
        base_rank = first_unique_doc_rank(baseline[qid], gold_docs)
        cand_rank = first_unique_doc_rank(candidate[qid], gold_docs)
        movement = movement_label(base_rank, cand_rank)
        source_pages = ranked_unique_pages(baseline[qid], max(1, int(args.source_top_pages)))
        source_pages_with_edges, linked_target_docs, raw_links = linked_target_doc_stats(
            source_pages, by_source_page
        )
        first_link = first_source_link_to_gold(source_pages, by_source_page, gold_docs)
        gold_incoming_edges = [
            edge
            for doc_id in gold_docs
            for edge in incoming_by_target_doc.get(doc_id, [])
        ]
        case = {
            "qid": qid,
            "question": str(gold.get("question", baseline[qid].get("question", ""))).strip(),
            "question_type": metadata_value(gold, "type"),
            "answer_type": metadata_value(gold, "answer_type"),
            "gold_answers": gold_answer_strings(gold),
            "gold_doc_ids": sorted(gold_docs),
            "baseline_rank": base_rank,
            "candidate_rank": cand_rank,
            "rank_delta": rank_delta(base_rank, cand_rank),
            "movement": movement,
            "baseline_hit_at_k": doc_hit_at_k(baseline[qid], gold_docs, int(args.hit_k)),
            "candidate_hit_at_k": doc_hit_at_k(candidate[qid], gold_docs, int(args.hit_k)),
            "gold_has_any_incoming_link": bool(gold_incoming_edges),
            "gold_incoming_edge_count": len(gold_incoming_edges),
            "gold_linked_from_baseline_sources": first_link is not None,
            "first_source_link_to_gold": first_link,
            "baseline_source_page_count": len(source_pages),
            "baseline_source_pages_with_edges": source_pages_with_edges,
            "baseline_linked_target_doc_count": linked_target_docs,
            "baseline_raw_link_count_from_sources": raw_links,
        }
        case["support_bucket"] = support_bucket(case)
        cases.append(case)

    by_support: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_support[case["support_bucket"]].append(case)
        by_type[case["question_type"] or "UNKNOWN"].append(case)

    improved = sorted(
        [case for case in cases if case["movement"] in {"improved", "newly_found"}],
        key=lambda row: (
            -(row["rank_delta"] if row["rank_delta"] is not None else 10**9),
            row["qid"],
        ),
    )
    worsened = sorted(
        [case for case in cases if case["movement"] in {"worsened", "newly_lost"}],
        key=lambda row: (
            row["rank_delta"] if row["rank_delta"] is not None else -10**9,
            row["qid"],
        ),
    )
    payload = {
        "n_qids": len(cases),
        "hit_k": int(args.hit_k),
        "source_top_pages": int(args.source_top_pages),
        "edge_count": graph["edge_count"],
        "movement_counts": dict(Counter(case["movement"] for case in cases)),
        "baseline_hit_at_k": sum(1 for case in cases if case["baseline_hit_at_k"]),
        "candidate_hit_at_k": sum(1 for case in cases if case["candidate_hit_at_k"]),
        "gold_has_any_incoming_link_count": sum(
            1 for case in cases if case["gold_has_any_incoming_link"]
        ),
        "gold_linked_from_baseline_sources_count": sum(
            1 for case in cases if case["gold_linked_from_baseline_sources"]
        ),
        "by_support_bucket": {
            key: summarize_group(value) for key, value in sorted(by_support.items())
        },
        "by_question_type": {
            key: summarize_group(value) for key, value in sorted(by_type.items())
        },
        "top_improved": improved[: int(args.topn)],
        "top_worsened": worsened[: int(args.topn)],
        "all_cases": cases,
    }

    Path(args.output_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(Path(args.output_md), payload, int(args.topn))
    print("n_qids", payload["n_qids"])
    print("movement_counts", payload["movement_counts"])
    print("gold_has_any_incoming_link_count", payload["gold_has_any_incoming_link_count"])
    print("gold_linked_from_baseline_sources_count", payload["gold_linked_from_baseline_sources_count"])
    print("saved_json", args.output_json)
    print("saved_md", args.output_md)


if __name__ == "__main__":
    main()
