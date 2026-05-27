#!/usr/bin/env python3

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a readable SVG/HTML subgraph for one successful Graph-PPR / "
            "Graph Page Preserve query. The plot shows selected page nodes, doc nodes, "
            "page-doc edges, same-document adjacent-page edges, gold labels, and "
            "dense/SPLADE/graph ranks."
        )
    )
    parser.add_argument("--graph-prediction-json", required=True)
    parser.add_argument("--dense-prediction-json", default="")
    parser.add_argument("--sparse-prediction-json", default="")
    parser.add_argument(
        "--gold",
        default="",
        help="Optional MMQA-style JSONL. Required when --qid is omitted.",
    )
    parser.add_argument(
        "--qid",
        default="",
        help="QID to visualize. If omitted, the script picks a successful graph qid.",
    )
    parser.add_argument(
        "--success-target",
        choices=["auto", "page", "doc"],
        default="auto",
        help="Success criterion when auto-picking a qid. auto uses page labels when present.",
    )
    parser.add_argument("--hit-k", type=int, default=4)
    parser.add_argument(
        "--graph-top-pages",
        type=int,
        default=20,
        help="Graph-ranked pages to include before doc filtering.",
    )
    parser.add_argument(
        "--source-top-pages",
        type=int,
        default=8,
        help="Dense/SPLADE source pages to include before doc filtering.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=8,
        help="Maximum document columns to draw. Use 0 to disable the doc cap.",
    )
    parser.add_argument("--same-doc-window", type=int, default=1)
    parser.add_argument("--output-svg", required=True)
    parser.add_argument("--output-html", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-dot", default="")
    return parser.parse_args()


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

    rows_by_qid: dict[str, dict[str, Any]] = {}
    for raw_key, row in iterable:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("qid", "")).strip() or str(raw_key).strip()
        if qid:
            rows_by_qid[qid] = row
    return rows_by_qid


def load_gold(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("qid", "")).strip()
            if qid:
                rows[qid] = row
    return rows


def page_uid(doc_id: str, page_idx: int) -> str:
    return f"{doc_id}_page{int(page_idx)}"


def parse_page_uid(uid: str) -> tuple[str, int | None]:
    if "_page" not in uid:
        return uid, None
    doc_id, raw_page_idx = uid.rsplit("_page", 1)
    try:
        return doc_id, int(raw_page_idx)
    except ValueError:
        return doc_id, None


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


def ranked_unique_pages(row: dict[str, Any] | None, limit: int = 10**9) -> list[dict[str, Any]]:
    if row is None:
        return []
    out: list[dict[str, Any]] = []
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
        out.append(
            {
                "uid": uid,
                "doc_id": doc_id,
                "page_idx": int(page_idx),
                "score": float(score),
                "rank": len(out) + 1,
            }
        )
        if len(out) >= limit:
            break
    return out


def first_doc_rank(pages: list[dict[str, Any]], gold_docs: set[str]) -> int | None:
    seen_docs: set[str] = set()
    rank = 0
    for page in pages:
        doc_id = str(page["doc_id"])
        if doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)
        rank += 1
        if doc_id in gold_docs:
            return rank
    return None


def first_page_rank(pages: list[dict[str, Any]], gold_pages: set[str]) -> int | None:
    for page in pages:
        if str(page["uid"]) in gold_pages:
            return int(page["rank"])
    return None


def rank_maps(pages: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int], dict[str, float]]:
    page_ranks = {str(page["uid"]): int(page["rank"]) for page in pages}
    scores = {str(page["uid"]): float(page["score"]) for page in pages}
    doc_ranks: dict[str, int] = {}
    for page in pages:
        doc_id = str(page["doc_id"])
        if doc_id not in doc_ranks:
            doc_ranks[doc_id] = len(doc_ranks) + 1
    return page_ranks, doc_ranks, scores


def count_gold_pages(pages: list[dict[str, Any]], gold_pages: set[str]) -> int:
    return sum(1 for page in pages if str(page["uid"]) in gold_pages)


def count_gold_doc_pages(pages: list[dict[str, Any]], gold_docs: set[str]) -> int:
    return sum(1 for page in pages if str(page["doc_id"]) in gold_docs)


def gold_doc_ids(row: dict[str, Any] | None) -> set[str]:
    if row is None:
        return set()
    out = {
        str(item.get("doc_id", "")).strip()
        for item in row.get("supporting_context", [])
        if isinstance(item, dict) and str(item.get("doc_id", "")).strip()
    }
    metadata = row.get("metadata", {})
    for key in ("gold_doc_ids", "supporting_doc_ids"):
        for value in metadata.get(key, []) if isinstance(metadata.get(key), list) else []:
            value = str(value).strip()
            if value:
                out.add(value)
    return out


def gold_page_uids(row: dict[str, Any] | None) -> set[str]:
    if row is None:
        return set()
    out = {
        str(value).strip()
        for value in row.get("metadata", {}).get("gold_page_uids", [])
        if str(value).strip()
    }
    for item in row.get("supporting_context", []):
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("doc_id", "")).strip()
        page_idx = item.get("page_idx", item.get("page_id"))
        if doc_id and page_idx is not None:
            try:
                out.add(page_uid(doc_id, int(page_idx)))
            except (TypeError, ValueError):
                pass
    return out


def choose_target(success_target: str, gold_pages: set[str]) -> str:
    if success_target == "auto":
        return "page" if gold_pages else "doc"
    return success_target


def rank_for_target(
    pages: list[dict[str, Any]],
    *,
    target: str,
    gold_docs: set[str],
    gold_pages: set[str],
) -> int | None:
    if target == "page":
        return first_page_rank(pages, gold_pages)
    return first_doc_rank(pages, gold_docs)


def select_successful_qid(
    *,
    graph_pred: dict[str, dict[str, Any]],
    dense_pred: dict[str, dict[str, Any]],
    sparse_pred: dict[str, dict[str, Any]],
    gold_rows: dict[str, dict[str, Any]],
    success_target: str,
    hit_k: int,
) -> str:
    if not gold_rows:
        raise ValueError("--gold is required when --qid is omitted.")
    common = set(graph_pred) & set(gold_rows)
    if dense_pred:
        common &= set(dense_pred)
    if sparse_pred:
        common &= set(sparse_pred)
    if not common:
        raise ValueError("No qids are shared by graph predictions, gold, and source predictions.")

    candidates: list[tuple[tuple[Any, ...], str]] = []
    for qid in sorted(common):
        gold_row = gold_rows[qid]
        gold_docs = gold_doc_ids(gold_row)
        gold_pages = gold_page_uids(gold_row)
        target = choose_target(success_target, gold_pages)
        if target == "page" and not gold_pages:
            continue
        if target == "doc" and not gold_docs:
            continue

        graph_pages = ranked_unique_pages(graph_pred[qid])
        dense_pages = ranked_unique_pages(dense_pred.get(qid)) if dense_pred else []
        sparse_pages = ranked_unique_pages(sparse_pred.get(qid)) if sparse_pred else []
        graph_rank = rank_for_target(
            graph_pages,
            target=target,
            gold_docs=gold_docs,
            gold_pages=gold_pages,
        )
        if graph_rank is None or graph_rank > hit_k:
            continue
        dense_rank = rank_for_target(
            dense_pages,
            target=target,
            gold_docs=gold_docs,
            gold_pages=gold_pages,
        ) if dense_pages else None
        sparse_rank = rank_for_target(
            sparse_pages,
            target=target,
            gold_docs=gold_docs,
            gold_pages=gold_pages,
        ) if sparse_pages else None
        dense_recovered = dense_rank is None or dense_rank > hit_k
        sparse_recovered = sparse_rank is None or sparse_rank > hit_k
        dense_delta = 10**6 if dense_rank is None else int(dense_rank) - int(graph_rank)
        sparse_delta = 10**6 if sparse_rank is None else int(sparse_rank) - int(graph_rank)
        candidates.append(
            (
                (
                    0 if dense_recovered else 1,
                    0 if sparse_recovered else 1,
                    int(graph_rank),
                    -dense_delta,
                    -sparse_delta,
                    qid,
                ),
                qid,
            )
        )

    if not candidates:
        raise ValueError(
            f"No graph-success qid found at {success_target} hit@{hit_k}. "
            "Pass --qid explicitly or use a larger --hit-k."
        )
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def short_id(value: str, max_len: int = 18) -> str:
    if len(value) <= max_len:
        return value
    keep = max(4, (max_len - 3) // 2)
    return f"{value[:keep]}...{value[-keep:]}"


def fmt_rank(value: int | None) -> str:
    return "-" if value is None else f"#{value}"


def fmt_score(value: float | None) -> str:
    if value is None:
        return "-"
    if not math.isfinite(value):
        return "-"
    return f"{value:.3g}"


def wrap_text(text: str, width: int) -> list[str]:
    text = " ".join(str(text).split())
    if not text:
        return []
    words = text.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word[:width]
    if current:
        lines.append(current)
    return lines


def build_visual_payload(
    *,
    qid: str,
    graph_row: dict[str, Any],
    dense_row: dict[str, Any] | None,
    sparse_row: dict[str, Any] | None,
    gold_row: dict[str, Any] | None,
    success_target: str,
    hit_k: int,
    graph_top_pages: int,
    source_top_pages: int,
    max_docs: int,
) -> dict[str, Any]:
    graph_pages_all = ranked_unique_pages(graph_row)
    dense_pages_all = ranked_unique_pages(dense_row) if dense_row else []
    sparse_pages_all = ranked_unique_pages(sparse_row) if sparse_row else []

    graph_page_ranks, graph_doc_ranks, graph_scores = rank_maps(graph_pages_all)
    dense_page_ranks, dense_doc_ranks, dense_scores = rank_maps(dense_pages_all)
    sparse_page_ranks, sparse_doc_ranks, sparse_scores = rank_maps(sparse_pages_all)

    gold_docs = gold_doc_ids(gold_row)
    gold_pages = gold_page_uids(gold_row)
    target = choose_target(success_target, gold_pages)
    graph_target_rank = rank_for_target(
        graph_pages_all,
        target=target,
        gold_docs=gold_docs,
        gold_pages=gold_pages,
    )
    dense_target_rank = rank_for_target(
        dense_pages_all,
        target=target,
        gold_docs=gold_docs,
        gold_pages=gold_pages,
    ) if dense_pages_all else None
    sparse_target_rank = rank_for_target(
        sparse_pages_all,
        target=target,
        gold_docs=gold_docs,
        gold_pages=gold_pages,
    ) if sparse_pages_all else None

    selected_uids: set[str] = set()
    for page in graph_pages_all[: max(0, graph_top_pages)]:
        selected_uids.add(str(page["uid"]))
    for page in dense_pages_all[: max(0, source_top_pages)]:
        selected_uids.add(str(page["uid"]))
    for page in sparse_pages_all[: max(0, source_top_pages)]:
        selected_uids.add(str(page["uid"]))
    selected_uids |= gold_pages

    doc_priority: dict[str, tuple[Any, ...]] = {}
    for uid in selected_uids:
        doc_id, _page_idx = parse_page_uid(uid)
        best_graph = graph_doc_ranks.get(doc_id, 10**9)
        best_dense = dense_doc_ranks.get(doc_id, 10**9)
        best_sparse = sparse_doc_ranks.get(doc_id, 10**9)
        is_gold_doc = doc_id in gold_docs
        doc_priority[doc_id] = (
            0 if is_gold_doc else 1,
            best_graph,
            min(best_dense, best_sparse),
            doc_id,
        )
    ordered_selected_docs = [
        doc_id for doc_id, _priority in sorted(doc_priority.items(), key=lambda item: item[1])
    ]
    if int(max_docs) > 0:
        selected_docs = ordered_selected_docs[: int(max_docs)]
    else:
        selected_docs = ordered_selected_docs
    selected_doc_set = set(selected_docs)

    pages: dict[str, dict[str, Any]] = {}
    all_known_uids = set(graph_page_ranks) | set(dense_page_ranks) | set(sparse_page_ranks) | gold_pages
    for uid in all_known_uids:
        doc_id, page_idx = parse_page_uid(uid)
        if page_idx is None or doc_id not in selected_doc_set:
            continue
        if uid not in selected_uids and uid not in gold_pages:
            continue
        pages[uid] = {
            "uid": uid,
            "doc_id": doc_id,
            "page_idx": int(page_idx),
            "graph_rank": graph_page_ranks.get(uid),
            "dense_rank": dense_page_ranks.get(uid),
            "sparse_rank": sparse_page_ranks.get(uid),
            "graph_score": graph_scores.get(uid),
            "dense_score": dense_scores.get(uid),
            "sparse_score": sparse_scores.get(uid),
            "is_gold_page": uid in gold_pages,
            "is_gold_doc": doc_id in gold_docs,
        }

    docs = []
    for doc_id in selected_docs:
        doc_pages = [page for page in pages.values() if page["doc_id"] == doc_id]
        if not doc_pages:
            continue
        docs.append(
            {
                "doc_id": doc_id,
                "is_gold_doc": doc_id in gold_docs,
                "graph_doc_rank": graph_doc_ranks.get(doc_id),
                "dense_doc_rank": dense_doc_ranks.get(doc_id),
                "sparse_doc_rank": sparse_doc_ranks.get(doc_id),
                "pages": sorted(
                    doc_pages,
                    key=lambda page: (
                        page["page_idx"],
                        page.get("graph_rank") or 10**9,
                        page["uid"],
                    ),
                ),
            }
        )

    visualized_pages = [page for doc in docs for page in doc["pages"]]
    visualized_graph_pages = [page for page in visualized_pages if page.get("graph_rank") is not None]
    visualized_dense_pages = [page for page in visualized_pages if page.get("dense_rank") is not None]
    visualized_sparse_pages = [page for page in visualized_pages if page.get("sparse_rank") is not None]
    stats = {
        "graph_retrieved_page_count": len(graph_pages_all),
        "graph_retrieved_doc_count": len(graph_doc_ranks),
        "graph_retrieved_gold_page_count": count_gold_pages(graph_pages_all, gold_pages),
        "graph_retrieved_gold_doc_page_count": count_gold_doc_pages(graph_pages_all, gold_docs),
        "dense_retrieved_page_count": len(dense_pages_all),
        "dense_retrieved_doc_count": len(dense_doc_ranks),
        "dense_retrieved_gold_page_count": count_gold_pages(dense_pages_all, gold_pages),
        "dense_retrieved_gold_doc_page_count": count_gold_doc_pages(dense_pages_all, gold_docs),
        "sparse_retrieved_page_count": len(sparse_pages_all),
        "sparse_retrieved_doc_count": len(sparse_doc_ranks),
        "sparse_retrieved_gold_page_count": count_gold_pages(sparse_pages_all, gold_pages),
        "sparse_retrieved_gold_doc_page_count": count_gold_doc_pages(sparse_pages_all, gold_docs),
        "gold_doc_count": len(gold_docs),
        "gold_page_count": len(gold_pages),
        "visualized_page_count": len(visualized_pages),
        "visualized_doc_count": len(docs),
        "visualized_graph_page_count": len(visualized_graph_pages),
        "visualized_dense_page_count": len(visualized_dense_pages),
        "visualized_sparse_page_count": len(visualized_sparse_pages),
        "visualized_gold_page_count": sum(1 for page in visualized_pages if page["is_gold_page"]),
        "visualized_gold_doc_page_count": sum(1 for page in visualized_pages if page["is_gold_doc"]),
        "hidden_graph_retrieved_page_count": max(
            0,
            len(graph_pages_all) - len(visualized_graph_pages),
        ),
        "graph_top_pages_requested": int(graph_top_pages),
        "source_top_pages_requested": int(source_top_pages),
        "max_docs_requested": int(max_docs),
    }

    question = str(graph_row.get("question") or (gold_row or {}).get("question", ""))
    return {
        "qid": qid,
        "question": question,
        "target": target,
        "hit_k": int(hit_k),
        "graph_target_rank": graph_target_rank,
        "dense_target_rank": dense_target_rank,
        "sparse_target_rank": sparse_target_rank,
        "gold_doc_ids": sorted(gold_docs),
        "gold_page_uids": sorted(gold_pages),
        "stats": stats,
        "docs": docs,
    }


def render_svg(payload: dict[str, Any], same_doc_window: int) -> str:
    docs = payload["docs"]
    doc_w = 230
    doc_gap = 26
    page_h = 62
    page_gap = 16
    margin = 28
    top_h = 164
    doc_h = 54
    max_pages = max((len(doc["pages"]) for doc in docs), default=1)
    width = margin * 2 + len(docs) * doc_w + max(0, len(docs) - 1) * doc_gap
    height = top_h + doc_h + 28 + max_pages * (page_h + page_gap) + 80

    page_positions: dict[str, tuple[float, float, float, float]] = {}
    doc_positions: dict[str, tuple[float, float, float, float]] = {}
    pieces: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#111827}",
        ".small{font-size:11px}.body{font-size:12px}.title{font-size:18px;font-weight:700}",
        ".muted{fill:#64748b}.doc{fill:#f8fafc;stroke:#334155;stroke-width:1.4}",
        ".page{stroke:#334155;stroke-width:1.2}.edge{stroke:#94a3b8;stroke-width:1.2}",
        ".adj{stroke:#38bdf8;stroke-width:1.4;stroke-dasharray:4 4}",
        "</style>",
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
    ]

    title = f"Graph Page Preserve query graph: {payload['qid']}"
    pieces.append(f'<text x="{margin}" y="32" class="title">{html.escape(title)}</text>')
    question_lines = wrap_text(payload.get("question", ""), 118)[:3]
    y = 56
    for line in question_lines:
        pieces.append(f'<text x="{margin}" y="{y}" class="body">{html.escape(line)}</text>')
        y += 18
    rank_line = (
        f"target={payload['target']}@{payload['hit_k']}   "
        f"graph={fmt_rank(payload.get('graph_target_rank'))}   "
        f"dense={fmt_rank(payload.get('dense_target_rank'))}   "
        f"splade={fmt_rank(payload.get('sparse_target_rank'))}"
    )
    pieces.append(f'<text x="{margin}" y="{y + 8}" class="body muted">{html.escape(rank_line)}</text>')
    stats = payload.get("stats", {})
    count_line = (
        f"graph pages={stats.get('graph_retrieved_page_count', 0)} "
        f"(gold pages={stats.get('graph_retrieved_gold_page_count', 0)}, "
        f"gold-doc pages={stats.get('graph_retrieved_gold_doc_page_count', 0)}); "
        f"visualized={stats.get('visualized_page_count', 0)} pages / "
        f"{stats.get('visualized_doc_count', 0)} docs"
    )
    pieces.append(f'<text x="{margin}" y="{y + 26}" class="body muted">{html.escape(count_line)}</text>')
    legend_y = y + 54
    legend = [
        ("#ffe7a3", "gold page/doc"),
        ("#d7f4df", "graph top page"),
        ("#eef2ff", "source-only page"),
        ("#ffffff", "selected page"),
    ]
    x = margin
    for fill, label in legend:
        pieces.append(
            f'<rect x="{x}" y="{legend_y - 12}" width="18" height="12" rx="3" '
            f'fill="{fill}" stroke="#334155" stroke-width="1"/>'
        )
        pieces.append(f'<text x="{x + 24}" y="{legend_y - 2}" class="small muted">{html.escape(label)}</text>')
        x += 130

    doc_y = top_h
    for doc_idx, doc in enumerate(docs):
        doc_x = margin + doc_idx * (doc_w + doc_gap)
        doc_positions[doc["doc_id"]] = (doc_x, doc_y, doc_w, doc_h)
        doc_fill = "#ffe7a3" if doc["is_gold_doc"] else "#f8fafc"
        pieces.append(
            f'<rect x="{doc_x}" y="{doc_y}" width="{doc_w}" height="{doc_h}" rx="7" '
            f'fill="{doc_fill}" stroke="#334155" stroke-width="1.5"/>'
        )
        pieces.append(
            f'<text x="{doc_x + 10}" y="{doc_y + 20}" class="body">'
            f'{html.escape(short_id(doc["doc_id"], 24))}</text>'
        )
        doc_rank_text = (
            f"Gdoc {fmt_rank(doc.get('graph_doc_rank'))} | "
            f"D {fmt_rank(doc.get('dense_doc_rank'))} | S {fmt_rank(doc.get('sparse_doc_rank'))}"
        )
        pieces.append(
            f'<text x="{doc_x + 10}" y="{doc_y + 40}" class="small muted">'
            f'{html.escape(doc_rank_text)}</text>'
        )
        for page_idx, page in enumerate(doc["pages"]):
            page_x = doc_x
            page_y = doc_y + doc_h + 28 + page_idx * (page_h + page_gap)
            page_positions[page["uid"]] = (page_x, page_y, doc_w, page_h)

    for uid, (px, py, pw, _ph) in page_positions.items():
        doc_id, _page_idx = parse_page_uid(uid)
        dx, dy, dw, dh = doc_positions[doc_id]
        pieces.append(
            f'<line x1="{px + pw / 2:.1f}" y1="{py:.1f}" '
            f'x2="{dx + dw / 2:.1f}" y2="{dy + dh:.1f}" class="edge"/>'
        )

    for doc in docs:
        pages = doc["pages"]
        for left_idx, left in enumerate(pages):
            for right in pages[left_idx + 1 :]:
                if right["page_idx"] - left["page_idx"] > same_doc_window:
                    break
                if left["uid"] not in page_positions or right["uid"] not in page_positions:
                    continue
                lx, ly, lw, lh = page_positions[left["uid"]]
                rx, ry, _rw, rh = page_positions[right["uid"]]
                edge_x = lx + lw - 10
                pieces.append(
                    f'<line x1="{edge_x:.1f}" y1="{ly + lh / 2:.1f}" '
                    f'x2="{edge_x:.1f}" y2="{ry + rh / 2:.1f}" class="adj"/>'
                )

    for doc in docs:
        for page in doc["pages"]:
            px, py, pw, ph = page_positions[page["uid"]]
            graph_rank = page.get("graph_rank")
            if page["is_gold_page"]:
                fill = "#ffe7a3"
                stroke = "#f59e0b"
                stroke_w = 2.4
            elif graph_rank is not None:
                fill = "#d7f4df"
                stroke = "#15803d"
                stroke_w = 1.8
            elif page.get("dense_rank") is not None or page.get("sparse_rank") is not None:
                fill = "#eef2ff"
                stroke = "#334155"
                stroke_w = 1.2
            else:
                fill = "#ffffff"
                stroke = "#334155"
                stroke_w = 1.2
            pieces.append(
                f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="7" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_w}"/>'
            )
            if page.get("dense_rank") is not None:
                pieces.append(
                    f'<rect x="{px}" y="{py}" width="5" height="{ph}" rx="2" fill="#2563eb"/>'
                )
            if page.get("sparse_rank") is not None:
                pieces.append(
                    f'<rect x="{px + 7}" y="{py}" width="5" height="{ph}" rx="2" fill="#7c3aed"/>'
                )
            label = f"p{page['page_idx']}"
            if page["is_gold_page"]:
                label += "  GOLD"
            pieces.append(f'<text x="{px + 16}" y="{py + 20}" class="body">{html.escape(label)}</text>')
            rank_text = (
                f"G {fmt_rank(page.get('graph_rank'))} "
                f"D {fmt_rank(page.get('dense_rank'))} "
                f"S {fmt_rank(page.get('sparse_rank'))}"
            )
            pieces.append(
                f'<text x="{px + 16}" y="{py + 39}" class="small muted">'
                f'{html.escape(rank_text)}</text>'
            )
            score_text = f"graph score {fmt_score(page.get('graph_score'))}"
            pieces.append(
                f'<text x="{px + 16}" y="{py + 55}" class="small muted">'
                f'{html.escape(score_text)}</text>'
            )

    pieces.append("</svg>")
    return "\n".join(pieces)


def render_html(payload: dict[str, Any], svg_text: str) -> str:
    stats = payload.get("stats", {})
    stat_rows = []
    for key in [
        "graph_retrieved_page_count",
        "graph_retrieved_doc_count",
        "graph_retrieved_gold_page_count",
        "graph_retrieved_gold_doc_page_count",
        "dense_retrieved_page_count",
        "dense_retrieved_doc_count",
        "dense_retrieved_gold_page_count",
        "dense_retrieved_gold_doc_page_count",
        "sparse_retrieved_page_count",
        "sparse_retrieved_doc_count",
        "sparse_retrieved_gold_page_count",
        "sparse_retrieved_gold_doc_page_count",
        "gold_doc_count",
        "gold_page_count",
        "visualized_page_count",
        "visualized_doc_count",
        "hidden_graph_retrieved_page_count",
    ]:
        stat_rows.append(
            f"<tr><td>{html.escape(key)}</td><td>{html.escape(str(stats.get(key, '')))}</td></tr>"
        )
    rows = []
    for doc in payload["docs"]:
        for page in doc["pages"]:
            rows.append(
                "<tr>"
                f"<td>{html.escape(short_id(page['uid'], 42))}</td>"
                f"<td>{html.escape('yes' if page['is_gold_page'] else '')}</td>"
                f"<td>{html.escape(fmt_rank(page.get('graph_rank')))}</td>"
                f"<td>{html.escape(fmt_rank(page.get('dense_rank')))}</td>"
                f"<td>{html.escape(fmt_rank(page.get('sparse_rank')))}</td>"
                f"<td>{html.escape(fmt_score(page.get('graph_score')))}</td>"
                "</tr>"
            )
    table = "\n".join(rows)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Graph PPR qid {html.escape(payload['qid'])}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; margin-top: 20px; font-size: 13px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 6px 8px; text-align: left; }}
    th {{ background: #f8fafc; }}
  </style>
</head>
<body>
{svg_text}
<h2>Counts</h2>
<table>
  <tbody>
    {''.join(stat_rows)}
  </tbody>
</table>
<h2>Visualized Pages</h2>
<table>
  <thead>
    <tr><th>page_uid</th><th>gold</th><th>graph</th><th>dense</th><th>SPLADE</th><th>graph score</th></tr>
  </thead>
  <tbody>
    {table}
  </tbody>
</table>
</body>
</html>
"""


def render_dot(payload: dict[str, Any], same_doc_window: int) -> str:
    lines = [
        "digraph graph_ppr_qid {",
        "  graph [rankdir=LR, bgcolor=white];",
        "  node [shape=box, style=\"rounded,filled\", fontname=\"Helvetica\"];",
        "  edge [color=\"#94a3b8\"];",
    ]
    for doc in payload["docs"]:
        doc_id = doc["doc_id"]
        fill = "#ffe7a3" if doc["is_gold_doc"] else "#f8fafc"
        lines.append(
            f'  "{doc_id}" [label="{short_id(doc_id, 24)}\\n'
            f'Gdoc {fmt_rank(doc.get("graph_doc_rank"))}", fillcolor="{fill}"];'
        )
        for page in doc["pages"]:
            uid = page["uid"]
            fill = "#ffe7a3" if page["is_gold_page"] else "#d7f4df" if page.get("graph_rank") else "#eef2ff"
            label = (
                f'p{page["page_idx"]}'
                f'\\nG {fmt_rank(page.get("graph_rank"))}'
                f' D {fmt_rank(page.get("dense_rank"))}'
                f' S {fmt_rank(page.get("sparse_rank"))}'
            )
            if page["is_gold_page"]:
                label += "\\nGOLD"
            lines.append(f'  "{uid}" [label="{label}", fillcolor="{fill}"];')
            lines.append(f'  "{uid}" -> "{doc_id}" [label="page-doc"];')
            lines.append(f'  "{doc_id}" -> "{uid}" [label="doc-page"];')
        pages = doc["pages"]
        for left_idx, left in enumerate(pages):
            for right in pages[left_idx + 1 :]:
                if right["page_idx"] - left["page_idx"] > same_doc_window:
                    break
                lines.append(
                    f'  "{left["uid"]}" -> "{right["uid"]}" '
                    '[dir=both, style=dashed, label="adjacent"];'
                )
    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    graph_pred = load_prediction(Path(args.graph_prediction_json))
    dense_pred = load_prediction(Path(args.dense_prediction_json)) if args.dense_prediction_json else {}
    sparse_pred = load_prediction(Path(args.sparse_prediction_json)) if args.sparse_prediction_json else {}
    gold_rows = load_gold(Path(args.gold)) if args.gold else {}

    qid = str(args.qid).strip()
    if not qid:
        qid = select_successful_qid(
            graph_pred=graph_pred,
            dense_pred=dense_pred,
            sparse_pred=sparse_pred,
            gold_rows=gold_rows,
            success_target=args.success_target,
            hit_k=int(args.hit_k),
        )

    if qid not in graph_pred:
        raise KeyError(f"QID not found in graph prediction JSON: {qid}")
    if dense_pred and qid not in dense_pred:
        raise KeyError(f"QID not found in dense prediction JSON: {qid}")
    if sparse_pred and qid not in sparse_pred:
        raise KeyError(f"QID not found in sparse prediction JSON: {qid}")

    payload = build_visual_payload(
        qid=qid,
        graph_row=graph_pred[qid],
        dense_row=dense_pred.get(qid),
        sparse_row=sparse_pred.get(qid),
        gold_row=gold_rows.get(qid),
        success_target=args.success_target,
        hit_k=int(args.hit_k),
        graph_top_pages=int(args.graph_top_pages),
        source_top_pages=int(args.source_top_pages),
        max_docs=int(args.max_docs),
    )
    svg_text = render_svg(payload, same_doc_window=int(args.same_doc_window))

    svg_path = Path(args.output_svg)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(svg_text, encoding="utf-8")

    if args.output_html:
        html_path = Path(args.output_html)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(render_html(payload, svg_text), encoding="utf-8")
    if args.output_json:
        json_path = Path(args.output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.output_dot:
        dot_path = Path(args.output_dot)
        dot_path.parent.mkdir(parents=True, exist_ok=True)
        dot_path.write_text(render_dot(payload, same_doc_window=int(args.same_doc_window)), encoding="utf-8")

    print(f"selected_qid: {qid}")
    print(f"question: {payload.get('question', '')}")
    print(
        "target_ranks: "
        f"graph={fmt_rank(payload.get('graph_target_rank'))} "
        f"dense={fmt_rank(payload.get('dense_target_rank'))} "
        f"splade={fmt_rank(payload.get('sparse_target_rank'))}"
    )
    stats = payload.get("stats", {})
    print(
        "retrieved_counts: "
        f"graph_pages={stats.get('graph_retrieved_page_count')} "
        f"graph_docs={stats.get('graph_retrieved_doc_count')} "
        f"graph_gold_pages={stats.get('graph_retrieved_gold_page_count')} "
        f"graph_gold_doc_pages={stats.get('graph_retrieved_gold_doc_page_count')}"
    )
    print(
        "visualized_counts: "
        f"pages={stats.get('visualized_page_count')} "
        f"docs={stats.get('visualized_doc_count')} "
        f"hidden_graph_pages={stats.get('hidden_graph_retrieved_page_count')}"
    )
    print(f"saved_svg: {svg_path}")
    if args.output_html:
        print(f"saved_html: {args.output_html}")
    if args.output_json:
        print(f"saved_json: {args.output_json}")
    if args.output_dot:
        print(f"saved_dot: {args.output_dot}")


if __name__ == "__main__":
    main()
