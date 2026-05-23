#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build page-page graph edges from page embeddings. This is the graph "
            "adapter for learned layout/semantic encoders such as LayoutLMv3 or DocGraphLM."
        )
    )
    parser.add_argument("--page-embeddings-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument(
        "--same-doc-only",
        action="store_true",
        help="Only emit edges between pages from the same document.",
    )
    parser.add_argument(
        "--cross-doc-only",
        action="store_true",
        help="Only emit edges between pages from different documents.",
    )
    parser.add_argument(
        "--bidirectional-dedup",
        action="store_true",
        help="Emit only one edge for each unordered page pair. PPR can add reverse edges later.",
    )
    return parser.parse_args()


def page_uid(doc_id: str, page_idx: int) -> str:
    return f"{doc_id}_page{int(page_idx)}"


def parse_page_uid(value: str) -> tuple[str, int | None]:
    if "_page" not in value:
        return value, None
    doc_id, raw_page = value.rsplit("_page", 1)
    try:
        return doc_id, int(raw_page)
    except ValueError:
        return doc_id, None


def read_embedding_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            embedding = row.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                continue
            page_uid_value = str(row.get("page_uid", "")).strip()
            doc_id = str(row.get("doc_id", "")).strip()
            page_idx = row.get("page_idx")
            if not page_uid_value and doc_id and page_idx is not None:
                page_uid_value = page_uid(doc_id, int(page_idx))
            if not page_uid_value:
                continue
            if not doc_id:
                doc_id, parsed_page_idx = parse_page_uid(page_uid_value)
                page_idx = parsed_page_idx if page_idx is None else page_idx
            try:
                vector = [float(value) for value in embedding]
            except (TypeError, ValueError):
                continue
            norm = math.sqrt(sum(value * value for value in vector))
            if norm <= 0:
                continue
            rows.append(
                {
                    "page_uid": page_uid_value,
                    "doc_id": doc_id,
                    "page_idx": None if page_idx is None else int(page_idx),
                    "embedding": [value / norm for value in vector],
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    if args.same_doc_only and args.cross_doc_only:
        raise ValueError("--same-doc-only and --cross-doc-only are mutually exclusive")

    rows = read_embedding_rows(Path(args.page_embeddings_jsonl))
    if not rows:
        raise ValueError("No valid page embeddings were loaded.")

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("This script requires numpy for chunked kNN search.") from exc

    matrix = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
    page_uids = [str(row["page_uid"]) for row in rows]
    doc_ids = [str(row["doc_id"]) for row in rows]
    top_k = max(0, int(args.top_k))
    if top_k <= 0:
        raise ValueError("--top-k must be positive")
    min_score = float(args.min_score)
    chunk_size = max(1, int(args.chunk_size))
    seen_pairs: set[tuple[str, str]] = set()
    edge_count = 0
    source_pages: set[str] = set()
    target_pages: set[str] = set()

    with Path(args.output_jsonl).open("w", encoding="utf-8") as out:
        for start in range(0, len(rows), chunk_size):
            end = min(len(rows), start + chunk_size)
            scores = matrix[start:end] @ matrix.T
            for local_idx, score_row in enumerate(scores):
                source_idx = start + local_idx
                source_uid = page_uids[source_idx]
                source_doc = doc_ids[source_idx]
                candidate_indices = np.argpartition(
                    -score_row,
                    kth=min(len(score_row) - 1, top_k + 32),
                )[: min(len(score_row), max(top_k + 32, top_k + 1))]
                ordered = sorted(
                    (int(idx) for idx in candidate_indices if int(idx) != source_idx),
                    key=lambda idx: (-float(score_row[idx]), page_uids[idx]),
                )
                emitted_for_source = 0
                for target_idx in ordered:
                    target_uid = page_uids[target_idx]
                    target_doc = doc_ids[target_idx]
                    score = float(score_row[target_idx])
                    if score < min_score:
                        continue
                    if args.same_doc_only and source_doc != target_doc:
                        continue
                    if args.cross_doc_only and source_doc == target_doc:
                        continue
                    if args.bidirectional_dedup:
                        pair = tuple(sorted((source_uid, target_uid)))
                        if pair in seen_pairs:
                            continue
                        seen_pairs.add(pair)
                    out.write(
                        json.dumps(
                            {
                                "edge_type": "page_embedding_knn",
                                "source_page_uid": source_uid,
                                "target_page_uid": target_uid,
                                "source_doc_id": source_doc,
                                "target_doc_id": target_doc,
                                "score": score,
                                "weight": score,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    edge_count += 1
                    emitted_for_source += 1
                    source_pages.add(source_uid)
                    target_pages.add(target_uid)
                    if emitted_for_source >= top_k:
                        break

    summary = {
        "page_embedding_count": len(rows),
        "edge_count": edge_count,
        "source_page_count": len(source_pages),
        "target_page_count": len(target_pages),
        "top_k": top_k,
        "min_score": min_score,
        "same_doc_only": bool(args.same_doc_only),
        "cross_doc_only": bool(args.cross_doc_only),
        "bidirectional_dedup": bool(args.bidirectional_dedup),
    }
    Path(args.output_summary_json).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("saved_edges", args.output_jsonl)
    print("saved_summary", args.output_summary_json)
    for key, value in summary.items():
        print(key, value)


if __name__ == "__main__":
    main()
