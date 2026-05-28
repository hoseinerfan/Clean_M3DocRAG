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
            "Inspect a PDF hyperlink edge JSONL and optionally estimate whether "
            "its doc-doc edges can activate for dense/SPLADE retrieval candidates."
        )
    )
    parser.add_argument("--hyperlink-edges-jsonl", required=True)
    parser.add_argument("--gold", default="", help="Optional MMQA-style gold JSONL.")
    parser.add_argument("--dense-prediction-json", default="")
    parser.add_argument("--sparse-prediction-json", default="")
    parser.add_argument("--dense-top-pages", type=int, default=1000)
    parser.add_argument("--sparse-top-pages", type=int, default=1000)
    parser.add_argument("--doc-doc-top-docs", type=int, default=20)
    parser.add_argument("--sample", type=int, default=5)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
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
        raise TypeError(f"Prediction JSON must be a list or object: {path}")

    rows: dict[str, dict[str, Any]] = {}
    for raw_key, row in iterable:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("qid", "")).strip() or str(raw_key).strip()
        if qid:
            rows[qid] = row
    return rows


def source_doc_id_from_uid(page_uid: str) -> str:
    if "_page" in page_uid:
        return page_uid.rsplit("_page", 1)[0]
    return ""


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


def ranked_unique_pages(row: dict[str, Any], limit: int) -> list[tuple[str, int, float, int]]:
    out = []
    seen_pages: set[str] = set()
    for raw in row.get("page_retrieval_results", []):
        parsed = parse_page_row(raw)
        if parsed is None:
            continue
        doc_id, page_idx, score = parsed
        uid = page_uid(doc_id, page_idx)
        if uid in seen_pages:
            continue
        seen_pages.add(uid)
        out.append((doc_id, page_idx, score, len(out) + 1))
        if len(out) >= limit:
            break
    return out


def first_doc_ranks(pages: list[tuple[str, int, float, int]]) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for doc_id, _page_idx, _score, _rank in pages:
        if doc_id not in ranks:
            ranks[doc_id] = len(ranks) + 1
    return ranks


def selected_top_docs(
    dense_pages: list[tuple[str, int, float, int]],
    sparse_pages: list[tuple[str, int, float, int]],
    top_docs: int,
) -> set[str]:
    best: dict[str, int] = {}
    for pages in (dense_pages, sparse_pages):
        doc_ranks = first_doc_ranks(pages)
        for doc_id, doc_rank in doc_ranks.items():
            best[doc_id] = min(best.get(doc_id, 10**9), int(doc_rank))
        for doc_id, _page_idx, _score, page_rank in pages:
            best[doc_id] = min(best.get(doc_id, 10**9), int(page_rank))
    ordered = sorted(best, key=lambda doc_id: (best[doc_id], doc_id))
    if top_docs > 0:
        ordered = ordered[:top_docs]
    return set(ordered)


def gold_doc_ids(row: dict[str, Any]) -> set[str]:
    docs: set[str] = set()
    for ctx in row.get("supporting_context", []):
        if isinstance(ctx, dict):
            doc_id = str(ctx.get("doc_id", "")).strip()
            if doc_id:
                docs.add(doc_id)
    return docs


def inspect_edges(rows: list[dict[str, Any]], sample_count: int) -> dict[str, Any]:
    by_source_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_target_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    edge_type_counts: Counter[str] = Counter()
    raw_link_total = 0
    invalid_missing_source_or_target = 0
    target_in_valid_doc_ids_counts: Counter[str] = Counter()
    source_docs: set[str] = set()
    target_docs: set[str] = set()
    samples: list[dict[str, Any]] = []

    for row in rows:
        source_page_uid = str(row.get("source_page_uid", "")).strip()
        target_doc_id = str(row.get("target_doc_id", "")).strip()
        if not source_page_uid or not target_doc_id:
            invalid_missing_source_or_target += 1
            continue
        source_doc_id = str(row.get("source_doc_id", "")).strip() or source_doc_id_from_uid(
            source_page_uid
        )
        try:
            raw_link_count = int(row.get("raw_link_count", 1) or 1)
        except (TypeError, ValueError):
            raw_link_count = 1
        edge = {
            "edge_type": str(row.get("edge_type", "")).strip(),
            "source_doc_id": source_doc_id,
            "source_page_uid": source_page_uid,
            "target_doc_id": target_doc_id,
            "target_wiki_title": str(row.get("target_wiki_title", "")).strip(),
            "raw_link_count": max(1, raw_link_count),
            "target_in_valid_doc_ids": row.get("target_in_valid_doc_ids"),
        }
        by_source_page[source_page_uid].append(edge)
        by_target_doc[target_doc_id].append(edge)
        source_docs.add(source_doc_id)
        target_docs.add(target_doc_id)
        edge_type_counts[edge["edge_type"] or "UNKNOWN"] += 1
        raw_link_total += edge["raw_link_count"]
        if edge["target_in_valid_doc_ids"] is not None:
            target_in_valid_doc_ids_counts[str(bool(edge["target_in_valid_doc_ids"]))] += 1
        if len(samples) < sample_count:
            samples.append(edge)

    return {
        "line_count": len(rows),
        "valid_edge_count": sum(len(v) for v in by_source_page.values()),
        "invalid_missing_source_or_target": invalid_missing_source_or_target,
        "source_page_count": len(by_source_page),
        "source_doc_count": len(source_docs),
        "target_doc_count": len(target_docs),
        "raw_link_count_total": raw_link_total,
        "edge_type_counts": dict(edge_type_counts),
        "target_in_valid_doc_ids_counts": dict(target_in_valid_doc_ids_counts),
        "samples": samples,
        "by_source_page": dict(by_source_page),
        "by_target_doc": dict(by_target_doc),
    }


def inspect_gold(gold_path: str, edge_info: dict[str, Any]) -> dict[str, Any]:
    if not gold_path:
        return {}
    gold_rows = read_jsonl(Path(gold_path))
    target_docs = set(edge_info["by_target_doc"])
    source_docs = {
        edge["source_doc_id"]
        for edges in edge_info["by_source_page"].values()
        for edge in edges
        if edge.get("source_doc_id")
    }
    gold_docs: set[str] = set()
    qids_with_target_inlinks = 0
    qids_with_source_doc_links = 0
    for row in gold_rows:
        docs = gold_doc_ids(row)
        gold_docs |= docs
        if docs & target_docs:
            qids_with_target_inlinks += 1
        if docs & source_docs:
            qids_with_source_doc_links += 1
    return {
        "gold_qid_count": len(gold_rows),
        "gold_doc_count": len(gold_docs),
        "gold_doc_with_inlink_count": len(gold_docs & target_docs),
        "gold_doc_as_source_count": len(gold_docs & source_docs),
        "gold_qids_with_any_gold_target_inlink": qids_with_target_inlinks,
        "gold_qids_with_any_gold_source_doc": qids_with_source_doc_links,
    }


def inspect_prediction_support(
    *,
    dense_pred_path: str,
    sparse_pred_path: str,
    gold_path: str,
    edge_info: dict[str, Any],
    dense_top_pages: int,
    sparse_top_pages: int,
    doc_doc_top_docs: int,
    sample_count: int,
) -> dict[str, Any]:
    if not dense_pred_path and not sparse_pred_path:
        return {}
    dense = load_prediction(Path(dense_pred_path)) if dense_pred_path else {}
    sparse = load_prediction(Path(sparse_pred_path)) if sparse_pred_path else {}
    gold = {row["qid"]: row for row in read_jsonl(Path(gold_path))} if gold_path else {}
    if dense and sparse:
        qids = sorted(set(dense) & set(sparse))
    else:
        qids = sorted(set(dense) | set(sparse))
    if gold:
        qids = [qid for qid in qids if qid in gold]

    by_source_page: dict[str, list[dict[str, Any]]] = edge_info["by_source_page"]
    edge_pair_qids = 0
    qids_with_source_page_edges = 0
    qids_with_link_to_gold_doc = 0
    pair_counts: list[int] = []
    source_edge_counts: list[int] = []
    selected_doc_counts: list[int] = []
    samples: list[dict[str, Any]] = []

    for qid in qids:
        dense_pages = ranked_unique_pages(dense.get(qid, {}), dense_top_pages) if dense else []
        sparse_pages = ranked_unique_pages(sparse.get(qid, {}), sparse_top_pages) if sparse else []
        selected_docs = selected_top_docs(dense_pages, sparse_pages, doc_doc_top_docs)
        selected_doc_counts.append(len(selected_docs))
        source_pages = {page_uid(doc_id, page_idx) for doc_id, page_idx, _score, _rank in dense_pages + sparse_pages}

        source_page_edge_count = sum(1 for uid in source_pages if uid in by_source_page)
        source_edge_counts.append(source_page_edge_count)
        if source_page_edge_count > 0:
            qids_with_source_page_edges += 1

        pair_count = 0
        sample_pairs = []
        gold_docs = gold_doc_ids(gold[qid]) if qid in gold else set()
        links_to_gold = 0
        for uid in source_pages:
            for edge in by_source_page.get(uid, []):
                source_doc = edge["source_doc_id"]
                target_doc = edge["target_doc_id"]
                if target_doc in gold_docs:
                    links_to_gold += 1
                if (
                    source_doc in selected_docs
                    and target_doc in selected_docs
                    and source_doc != target_doc
                ):
                    pair_count += 1
                    if len(sample_pairs) < 3:
                        sample_pairs.append(
                            {
                                "source_page_uid": uid,
                                "source_doc_id": source_doc,
                                "target_doc_id": target_doc,
                                "target_wiki_title": edge.get("target_wiki_title", ""),
                            }
                        )
        pair_counts.append(pair_count)
        if pair_count > 0:
            edge_pair_qids += 1
        if links_to_gold > 0:
            qids_with_link_to_gold_doc += 1
        if len(samples) < sample_count and (pair_count > 0 or links_to_gold > 0):
            samples.append(
                {
                    "qid": qid,
                    "selected_doc_count": len(selected_docs),
                    "source_page_edge_count": source_page_edge_count,
                    "selected_doc_pair_edge_count": pair_count,
                    "link_to_gold_doc_count": links_to_gold,
                    "sample_pairs": sample_pairs,
                }
            )

    return {
        "prediction_qid_count": len(qids),
        "doc_doc_top_docs": doc_doc_top_docs,
        "qids_with_any_retrieved_source_page_edge": qids_with_source_page_edges,
        "qids_with_selected_doc_hyperlink_pairs": edge_pair_qids,
        "qids_with_retrieved_link_to_gold_doc": qids_with_link_to_gold_doc,
        "mean_selected_doc_count": mean(selected_doc_counts),
        "mean_retrieved_source_page_edge_count": mean(source_edge_counts),
        "mean_selected_doc_hyperlink_pair_count": mean(pair_counts),
        "samples": samples,
    }


def mean(values: list[int | float]) -> float:
    return sum(float(value) for value in values) / len(values) if values else 0.0


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


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# PDF Hyperlink Graph Sanity Check", ""]
    rows = [["metric", "value"]]
    for key in [
        "path",
        "line_count",
        "valid_edge_count",
        "invalid_missing_source_or_target",
        "source_page_count",
        "source_doc_count",
        "target_doc_count",
        "raw_link_count_total",
    ]:
        rows.append([key, payload.get(key, "")])
    lines.extend(markdown_table(rows))

    lines.extend(["", "## Gold Overlap", ""])
    gold = payload.get("gold_overlap", {})
    if gold:
        lines.extend(markdown_table([["metric", "value"], *[[k, v] for k, v in gold.items()]]))
    else:
        lines.append("No gold file provided.")

    lines.extend(["", "## Retrieval Activation", ""])
    pred = payload.get("prediction_support", {})
    if pred:
        lines.extend(
            markdown_table(
                [["metric", "value"]]
                + [[k, v] for k, v in pred.items() if k != "samples"]
            )
        )
    else:
        lines.append("No prediction file provided.")

    lines.extend(["", "## Sample Edges", ""])
    for edge in payload.get("samples", []):
        lines.append(
            f"- `{edge['source_page_uid']}` `{edge['source_doc_id']}` -> "
            f"`{edge['target_doc_id']}` title={edge.get('target_wiki_title', '')!r} "
            f"raw_link_count={edge['raw_link_count']}"
        )

    if pred.get("samples"):
        lines.extend(["", "## Sample Activated QIDs", ""])
        for sample in pred["samples"]:
            lines.append(
                f"- `{sample['qid']}` source_page_edges={sample['source_page_edge_count']} "
                f"selected_doc_pairs={sample['selected_doc_pair_edge_count']} "
                f"links_to_gold={sample['link_to_gold_doc_count']}"
            )
            for pair in sample.get("sample_pairs", []):
                lines.append(
                    f"  - `{pair['source_page_uid']}` `{pair['source_doc_id']}` -> "
                    f"`{pair['target_doc_id']}` title={pair.get('target_wiki_title', '')!r}"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    path = Path(args.hyperlink_edges_jsonl)
    rows = read_jsonl(path)
    edge_info = inspect_edges(rows, int(args.sample))
    payload: dict[str, Any] = {
        "path": str(path),
        **{k: v for k, v in edge_info.items() if not k.startswith("by_")},
        "gold_overlap": inspect_gold(args.gold, edge_info),
        "prediction_support": inspect_prediction_support(
            dense_pred_path=args.dense_prediction_json,
            sparse_pred_path=args.sparse_prediction_json,
            gold_path=args.gold,
            edge_info=edge_info,
            dense_top_pages=max(0, int(args.dense_top_pages)),
            sparse_top_pages=max(0, int(args.sparse_top_pages)),
            doc_doc_top_docs=max(0, int(args.doc_doc_top_docs)),
            sample_count=max(0, int(args.sample)),
        ),
    }
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if args.output_md:
        write_markdown(Path(args.output_md), payload)

    print(f"path: {payload['path']}")
    print(f"line_count: {payload['line_count']}")
    print(f"valid_edge_count: {payload['valid_edge_count']}")
    print(f"source_page_count: {payload['source_page_count']}")
    print(f"source_doc_count: {payload['source_doc_count']}")
    print(f"target_doc_count: {payload['target_doc_count']}")
    print(f"edge_type_counts: {payload['edge_type_counts']}")
    if payload["gold_overlap"]:
        print(f"gold_overlap: {payload['gold_overlap']}")
    if payload["prediction_support"]:
        print(f"prediction_support: {payload['prediction_support']}")
    if args.output_json:
        print(f"saved_json: {args.output_json}")
    if args.output_md:
        print(f"saved_md: {args.output_md}")


if __name__ == "__main__":
    main()
