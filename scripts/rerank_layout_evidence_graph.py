#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_TEXT_FIELDS = [
    "ocr_text",
    "vlm_text",
    "markdown",
    "text",
    "page_text",
    "content",
]

DEFAULT_REGION_FIELDS = [
    "layout_regions",
    "regions",
    "ocr_blocks",
    "text_blocks",
    "blocks",
    "paragraphs",
    "lines",
    "tables",
    "figures",
    "captions",
    "layout.regions",
    "layout.blocks",
    "ocr.blocks",
]

REGION_TEXT_KEYS = [
    "text",
    "ocr_text",
    "content",
    "markdown",
    "caption",
    "value",
    "html",
]

REGION_LABEL_KEYS = [
    "type",
    "label",
    "category",
    "block_type",
    "role",
    "class",
    "name",
]

TOKEN_RE = re.compile(r"(?u)\b\w+\b")


@dataclass(frozen=True)
class PageKey:
    doc_id: str
    page_idx: int

    @property
    def uid(self) -> str:
        return page_uid(self.doc_id, self.page_idx)


@dataclass
class PageRecord:
    doc_id: str
    page_idx: int
    text: str
    row: dict[str, Any]

    @property
    def uid(self) -> str:
        return page_uid(self.doc_id, self.page_idx)


@dataclass
class EvidenceRegion:
    page_uid: str
    region_idx: int
    text: str
    label: str
    source: str
    order: int

    @property
    def node_id(self) -> str:
        return f"region::{self.page_uid}::{self.region_idx}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rerank page retrieval with a query-local layout/evidence graph. "
            "The graph has query, evidence-region, and page nodes. Region nodes are "
            "read from OCR/layout fields when present, otherwise from converted page "
            "text or markdown blocks. Query-region edges use BM25, region-page edges "
            "use containment, and final ranking uses standard RRF with the input rank."
        )
    )
    parser.add_argument("--prediction", required=True, help="Input page prediction JSON.")
    parser.add_argument("--doc-pages-jsonl", required=True, help="Converted doc_pages JSONL.")
    parser.add_argument("--gold", default="", help="Optional gold MMQA JSONL for metrics.")
    parser.add_argument(
        "--text-field",
        action="append",
        default=[],
        help=(
            "Page text field to use. Repeat to concatenate fields. "
            f"Defaults to {DEFAULT_TEXT_FIELDS}."
        ),
    )
    parser.add_argument(
        "--region-field",
        action="append",
        default=[],
        help=(
            "Structured OCR/layout region field. Repeatable and supports dotted paths. "
            f"Defaults to {DEFAULT_REGION_FIELDS}."
        ),
    )
    parser.add_argument(
        "--candidate-scope",
        choices=["prediction_top_pages", "top_docs_prediction_pages", "all_top_doc_pages"],
        default="top_docs_prediction_pages",
        help=(
            "Pages whose regions are scored. prediction_top_pages uses the input top pages; "
            "top_docs_prediction_pages restricts those pages to top retrieved docs; "
            "all_top_doc_pages scores every catalog page in top retrieved docs."
        ),
    )
    parser.add_argument("--top-docs", type=int, default=4)
    parser.add_argument("--input-top-pages", type=int, default=1000)
    parser.add_argument("--candidate-top-pages", type=int, default=1000)
    parser.add_argument("--output-top-pages", type=int, default=1000)
    parser.add_argument("--max-pages-per-doc", type=int, default=250)
    parser.add_argument("--max-regions-per-page", type=int, default=32)
    parser.add_argument("--max-region-tokens", type=int, default=96)
    parser.add_argument("--bm25-k1", type=float, default=1.2)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--ppr-restart-prob", type=float, default=0.30)
    parser.add_argument("--ppr-iters", type=int, default=20)
    parser.add_argument("--query-region-edge-weight", type=float, default=1.0)
    parser.add_argument("--region-page-edge-weight", type=float, default=1.0)
    parser.add_argument("--region-adjacent-edge-weight", type=float, default=0.15)
    parser.add_argument(
        "--page-restart-weight",
        type=float,
        default=0.0,
        help=(
            "Optional restart mass on input pages. Default 0 keeps the evidence graph "
            "separate; final RRF fuses it with the base rank."
        ),
    )
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--hit-k", type=int, default=4)
    parser.add_argument(
        "--recall-k",
        dest="recall_ks",
        type=int,
        nargs="+",
        default=[1, 2, 4, 5, 10, 20, 50, 100],
    )
    parser.add_argument("--output-prediction-json", required=True)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument("--output-case-json", default="")
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
    iterable: Any
    if isinstance(payload, list):
        iterable = enumerate(payload)
    elif isinstance(payload, dict):
        iterable = payload.items()
    else:
        raise TypeError(f"Prediction JSON must be a list or object: {path}")

    for raw_key, row in iterable:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("qid", "")).strip() or str(raw_key).strip()
        if qid:
            rows_by_qid[qid] = row
    return rows_by_qid


def maybe_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def page_uid(doc_id: Any, page_idx: Any) -> str:
    return f"{doc_id}_page{int(page_idx)}"


def parse_page_uid(uid: str) -> PageKey | None:
    marker = "_page"
    if marker not in uid:
        return None
    doc_id, raw_idx = uid.rsplit(marker, 1)
    try:
        return PageKey(doc_id=doc_id, page_idx=int(raw_idx))
    except ValueError:
        return None


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = " ".join(str(item) for item in value if item is not None)
    elif isinstance(value, dict):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = str(value).replace("\x0c", " ").replace("\u0000", " ")
    return re.sub(r"[ \t\r\f\v]+", " ", text).strip()


def compact_text(value: Any) -> str:
    return re.sub(r"\s+", " ", normalize_text(value)).strip()


def nested_get(row: dict[str, Any], dotted: str) -> Any:
    current: Any = row
    for part in dotted.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def first_nonempty_text(row: dict[str, Any], keys: list[str]) -> str:
    values: list[str] = []
    for key in keys:
        value = nested_get(row, key) if "." in key else row.get(key)
        text = normalize_text(value)
        if text:
            values.append(text)
    return "\n\n".join(values)


def page_key_from_row(row: dict[str, Any]) -> PageKey | None:
    doc_id = str(row.get("doc_id", row.get("doc_name", ""))).strip()
    page_idx = maybe_int(row.get("page_idx", row.get("page_id", row.get("page_number"))))
    if not doc_id or page_idx is None:
        return None
    return PageKey(doc_id=doc_id, page_idx=page_idx)


def load_page_catalog(
    path: Path,
    *,
    text_fields: list[str],
) -> tuple[dict[str, PageRecord], dict[str, list[str]]]:
    by_uid: dict[str, PageRecord] = {}
    by_doc: dict[str, list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = page_key_from_row(row)
            if key is None:
                continue
            text = first_nonempty_text(row, text_fields)
            record = PageRecord(
                doc_id=key.doc_id,
                page_idx=key.page_idx,
                text=text,
                row=row,
            )
            by_uid[key.uid] = record
            by_doc[key.doc_id].append(key.uid)
    for doc_id in list(by_doc):
        by_doc[doc_id] = sorted(
            set(by_doc[doc_id]),
            key=lambda uid: parse_page_uid(uid).page_idx if parse_page_uid(uid) else 10**9,
        )
    return by_uid, dict(by_doc)


def prediction_rows(row: dict[str, Any]) -> list[Any]:
    rows = row.get("page_retrieval_results", [])
    return rows if isinstance(rows, list) else []


def parse_prediction_page(row: Any) -> tuple[str, int, float] | None:
    if not isinstance(row, (list, tuple)) or len(row) < 2:
        return None
    doc_id = str(row[0]).strip()
    page_idx = maybe_int(row[1])
    if not doc_id or page_idx is None:
        return None
    score = 0.0
    if len(row) >= 3:
        try:
            score = float(row[2])
        except (TypeError, ValueError):
            score = 0.0
    return doc_id, page_idx, score


def ranked_page_uids(row: dict[str, Any], limit: int) -> list[str]:
    pages: list[str] = []
    seen: set[str] = set()
    for item in prediction_rows(row):
        parsed = parse_prediction_page(item)
        if parsed is None:
            continue
        uid = page_uid(parsed[0], parsed[1])
        if uid in seen:
            continue
        seen.add(uid)
        pages.append(uid)
        if limit > 0 and len(pages) >= limit:
            break
    return pages


def ranked_doc_ids(page_uids: list[str], limit: int) -> list[str]:
    docs: list[str] = []
    seen: set[str] = set()
    for uid in page_uids:
        parsed = parse_page_uid(uid)
        if parsed is None:
            continue
        if parsed.doc_id in seen:
            continue
        seen.add(parsed.doc_id)
        docs.append(parsed.doc_id)
        if limit > 0 and len(docs) >= limit:
            break
    return docs


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "") if len(token) > 1]


def trim_region_text(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return compact_text(text)
    tokens = TOKEN_RE.findall(text)
    if len(tokens) <= max_tokens:
        return compact_text(text)
    return " ".join(tokens[:max_tokens])


def region_label(raw: Any, fallback: str) -> str:
    if isinstance(raw, dict):
        for key in REGION_LABEL_KEYS:
            value = raw.get(key)
            text = compact_text(value)
            if text:
                return text[:80]
    return fallback


def region_text(raw: Any) -> str:
    if isinstance(raw, str):
        return compact_text(raw)
    if isinstance(raw, dict):
        for key in REGION_TEXT_KEYS:
            text = compact_text(raw.get(key))
            if text:
                return text
        texts = []
        for value in raw.values():
            if isinstance(value, str):
                text = compact_text(value)
                if text:
                    texts.append(text)
        return " ".join(texts)
    if isinstance(raw, (list, tuple)):
        return compact_text(" ".join(str(value) for value in raw if value is not None))
    return ""


def flatten_region_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        out: list[Any] = []
        for key in ("regions", "blocks", "lines", "paragraphs", "items", "cells"):
            nested = value.get(key)
            if isinstance(nested, list):
                out.extend(nested)
        if out:
            return out
        return list(value.values())
    return [value]


def markdown_or_text_blocks(text: str, max_region_tokens: int) -> list[tuple[str, str, str]]:
    raw = normalize_text(text)
    if not raw:
        return []

    if "\n" in raw:
        blocks: list[tuple[str, str, str]] = []
        current: list[str] = []
        current_label = "text_block"

        def flush() -> None:
            nonlocal current, current_label
            block_text = "\n".join(current).strip()
            if block_text:
                blocks.append(
                    (
                        trim_region_text(block_text, max_region_tokens),
                        current_label,
                        "text_block",
                    )
                )
            current = []
            current_label = "text_block"

        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped:
                flush()
                continue
            label = "text_block"
            if stripped.startswith("#"):
                label = "heading"
            elif "|" in stripped or "\t" in stripped:
                label = "table_like"
            elif stripped[:2] in {"- ", "* "} or re.match(r"^\d+[.)]\s+", stripped):
                label = "list_item"
            if current and label != current_label:
                flush()
            current.append(stripped)
            current_label = label
        flush()
        if blocks:
            return blocks

    tokens = TOKEN_RE.findall(raw)
    if not tokens:
        return []
    chunk_size = max(24, max_region_tokens)
    blocks = []
    for start in range(0, len(tokens), chunk_size):
        chunk = " ".join(tokens[start : start + chunk_size])
        if chunk:
            blocks.append((chunk, "text_window", "text_window"))
    return blocks


def extract_regions(
    record: PageRecord,
    *,
    region_fields: list[str],
    max_regions_per_page: int,
    max_region_tokens: int,
) -> tuple[list[EvidenceRegion], bool]:
    regions: list[EvidenceRegion] = []
    used_explicit = False

    title_fields = ["title", "section_title", "page_title", "doc_title"]
    title = first_nonempty_text(record.row, title_fields)
    if title:
        regions.append(
            EvidenceRegion(
                page_uid=record.uid,
                region_idx=len(regions),
                text=trim_region_text(title, max_region_tokens),
                label="title",
                source="metadata_title",
                order=len(regions),
            )
        )

    for field in region_fields:
        raw_value = nested_get(record.row, field) if "." in field else record.row.get(field)
        values = flatten_region_values(raw_value)
        if not values:
            continue
        field_region_count = 0
        for raw_region in values:
            text = region_text(raw_region)
            if not text:
                continue
            regions.append(
                EvidenceRegion(
                    page_uid=record.uid,
                    region_idx=len(regions),
                    text=trim_region_text(text, max_region_tokens),
                    label=region_label(raw_region, field),
                    source=field,
                    order=len(regions),
                )
            )
            field_region_count += 1
            if max_regions_per_page > 0 and len(regions) >= max_regions_per_page:
                break
        if field_region_count > 0:
            used_explicit = True
        if max_regions_per_page > 0 and len(regions) >= max_regions_per_page:
            break

    if not used_explicit:
        for text, label, source in markdown_or_text_blocks(record.text, max_region_tokens):
            if not text:
                continue
            regions.append(
                EvidenceRegion(
                    page_uid=record.uid,
                    region_idx=len(regions),
                    text=text,
                    label=label,
                    source=source,
                    order=len(regions),
                )
            )
            if max_regions_per_page > 0 and len(regions) >= max_regions_per_page:
                break

    return regions, used_explicit


def bm25_region_scores(
    query: str,
    regions: list[EvidenceRegion],
    *,
    k1: float,
    b: float,
) -> dict[str, float]:
    query_terms = tokenize(query)
    if not query_terms or not regions:
        return {}
    query_counts = Counter(query_terms)
    region_tokens = [tokenize(region.text) for region in regions]
    lengths = [len(tokens) for tokens in region_tokens]
    avgdl = statistics.fmean(lengths) if lengths else 0.0
    if avgdl <= 0:
        return {}
    df: Counter[str] = Counter()
    for tokens in region_tokens:
        df.update(set(tokens))
    n_regions = len(regions)
    scores: dict[str, float] = {}
    for region, tokens, length in zip(regions, region_tokens, lengths):
        if not tokens:
            continue
        tf = Counter(tokens)
        score = 0.0
        for term, qtf in query_counts.items():
            freq = tf.get(term, 0)
            if freq <= 0:
                continue
            idf = math.log(1.0 + (n_regions - df[term] + 0.5) / (df[term] + 0.5))
            denom = freq + k1 * (1.0 - b + b * (float(length) / avgdl))
            score += float(qtf) * idf * ((freq * (k1 + 1.0)) / denom)
        if score > 0:
            scores[region.node_id] = score
    return scores


def add_edge(graph: dict[str, dict[str, float]], source: str, target: str, weight: float) -> None:
    if weight <= 0 or source == target:
        return
    graph.setdefault(source, {})
    graph.setdefault(target, {})
    graph[source][target] = graph[source].get(target, 0.0) + float(weight)


def normalize(values: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, value) for value in values.values())
    if total <= 0:
        return {}
    return {key: max(0.0, value) / total for key, value in values.items()}


def run_ppr(
    graph: dict[str, dict[str, float]],
    restart: dict[str, float],
    *,
    restart_prob: float,
    iters: int,
) -> dict[str, float]:
    restart = normalize(restart)
    if not restart:
        return {}
    nodes = set(graph) | set(restart)
    ranks = dict(restart)
    alpha = min(1.0, max(0.0, float(restart_prob)))
    for _ in range(max(1, int(iters))):
        next_ranks = {node: alpha * restart.get(node, 0.0) for node in nodes}
        dangling = 0.0
        for source in nodes:
            rank = ranks.get(source, 0.0)
            if rank <= 0:
                continue
            neighbors = graph.get(source, {})
            norm = sum(max(0.0, weight) for weight in neighbors.values())
            if norm <= 0:
                dangling += rank
                continue
            scale = (1.0 - alpha) * rank / norm
            for target, weight in neighbors.items():
                if weight > 0:
                    next_ranks[target] = next_ranks.get(target, 0.0) + scale * weight
        if dangling > 0:
            for node, value in restart.items():
                next_ranks[node] = next_ranks.get(node, 0.0) + (1.0 - alpha) * dangling * value
        ranks = next_ranks
    return ranks


def candidate_pages_for_qid(
    *,
    ranked_pages: list[str],
    by_doc: dict[str, list[str]],
    top_docs: int,
    candidate_scope: str,
    candidate_top_pages: int,
    max_pages_per_doc: int,
) -> list[str]:
    limited_ranked = ranked_pages[:candidate_top_pages] if candidate_top_pages > 0 else ranked_pages
    docs = set(ranked_doc_ids(ranked_pages, top_docs))
    if candidate_scope == "prediction_top_pages":
        return limited_ranked
    if candidate_scope == "top_docs_prediction_pages":
        return [uid for uid in limited_ranked if (parse_page_uid(uid) or PageKey("", -1)).doc_id in docs]

    pages: list[str] = []
    seen: set[str] = set()
    for doc_id in ranked_doc_ids(ranked_pages, top_docs):
        doc_pages = by_doc.get(doc_id, [])
        if max_pages_per_doc > 0:
            doc_pages = doc_pages[:max_pages_per_doc]
        for uid in doc_pages:
            if uid not in seen:
                seen.add(uid)
                pages.append(uid)
    return pages


def base_rank_scores(ranked_pages: list[str], rrf_k: float) -> dict[str, float]:
    scores: dict[str, float] = {}
    for rank, uid in enumerate(ranked_pages, start=1):
        scores[uid] = 1.0 / (float(rrf_k) + float(rank))
    return scores


def evidence_scores_for_qid(
    *,
    question: str,
    ranked_pages: list[str],
    candidate_pages: list[str],
    page_catalog: dict[str, PageRecord],
    args: argparse.Namespace,
    region_fields: list[str],
) -> tuple[dict[str, float], dict[str, Any]]:
    regions: list[EvidenceRegion] = []
    explicit_region_pages = 0
    fallback_region_pages = 0
    pages_with_regions = 0
    for uid in candidate_pages:
        record = page_catalog.get(uid)
        if record is None:
            continue
        page_regions, used_explicit = extract_regions(
            record,
            region_fields=region_fields,
            max_regions_per_page=int(args.max_regions_per_page),
            max_region_tokens=int(args.max_region_tokens),
        )
        if not page_regions:
            continue
        pages_with_regions += 1
        explicit_region_pages += int(used_explicit)
        fallback_region_pages += int(not used_explicit)
        regions.extend(page_regions)

    if not regions:
        return {}, {
            "candidate_page_count": len(candidate_pages),
            "region_count": 0,
            "pages_with_regions": 0,
            "explicit_region_pages": 0,
            "fallback_region_pages": 0,
            "positive_query_region_count": 0,
        }

    raw_region_scores = bm25_region_scores(
        question,
        regions,
        k1=float(args.bm25_k1),
        b=float(args.bm25_b),
    )
    max_region_score = max(raw_region_scores.values(), default=0.0)
    graph: dict[str, dict[str, float]] = {}
    query_node = "query"
    restart: dict[str, float] = {query_node: 1.0}

    if float(args.page_restart_weight) > 0:
        for uid, score in base_rank_scores(ranked_pages, float(args.rrf_k)).items():
            restart[f"page::{uid}"] = restart.get(f"page::{uid}", 0.0) + (
                float(args.page_restart_weight) * score
            )

    regions_by_page: dict[str, list[EvidenceRegion]] = defaultdict(list)
    for region in regions:
        page_node = f"page::{region.page_uid}"
        region_node = region.node_id
        regions_by_page[region.page_uid].append(region)
        add_edge(graph, region_node, page_node, float(args.region_page_edge_weight))
        add_edge(graph, page_node, region_node, float(args.region_page_edge_weight))
        if max_region_score > 0:
            query_weight = raw_region_scores.get(region_node, 0.0) / max_region_score
            add_edge(
                graph,
                query_node,
                region_node,
                float(args.query_region_edge_weight) * query_weight,
            )

    adjacent_weight = float(args.region_adjacent_edge_weight)
    if adjacent_weight > 0:
        for page_regions in regions_by_page.values():
            ordered = sorted(page_regions, key=lambda region: region.order)
            for left, right in zip(ordered, ordered[1:]):
                add_edge(graph, left.node_id, right.node_id, adjacent_weight)
                add_edge(graph, right.node_id, left.node_id, adjacent_weight)

    ranks = run_ppr(
        graph,
        restart,
        restart_prob=float(args.ppr_restart_prob),
        iters=int(args.ppr_iters),
    )
    page_scores: dict[str, float] = {}
    for uid in candidate_pages:
        node = f"page::{uid}"
        score = ranks.get(node, 0.0)
        if score > 0:
            page_scores[uid] = score
    max_page_score = max(page_scores.values(), default=0.0)
    if max_page_score > 0:
        page_scores = {uid: score / max_page_score for uid, score in page_scores.items()}

    return page_scores, {
        "candidate_page_count": len(candidate_pages),
        "region_count": len(regions),
        "pages_with_regions": pages_with_regions,
        "explicit_region_pages": explicit_region_pages,
        "fallback_region_pages": fallback_region_pages,
        "positive_query_region_count": len(raw_region_scores),
        "positive_evidence_page_count": sum(score > 0 for score in page_scores.values()),
    }


def rrf_fuse_pages(
    ranked_pages: list[str],
    evidence_scores: dict[str, float],
    *,
    rrf_k: float,
    output_top_pages: int,
) -> list[tuple[str, float]]:
    positive_evidence_pages = [
        uid for uid, score in evidence_scores.items() if float(score) > 0
    ]
    candidates = list(dict.fromkeys(ranked_pages + positive_evidence_pages))
    base_rank = {uid: rank for rank, uid in enumerate(ranked_pages, start=1)}
    evidence_ranked = sorted(
        positive_evidence_pages,
        key=lambda uid: (
            -evidence_scores.get(uid, 0.0),
            base_rank.get(uid, 10**9),
            uid,
        ),
    )
    evidence_rank = {uid: rank for rank, uid in enumerate(evidence_ranked, start=1)}
    scored: list[tuple[str, float]] = []
    for uid in candidates:
        score = 0.0
        if uid in base_rank:
            score += 1.0 / (float(rrf_k) + float(base_rank[uid]))
        if uid in evidence_rank:
            score += 1.0 / (float(rrf_k) + float(evidence_rank[uid]))
        scored.append((uid, score))
    scored.sort(
        key=lambda item: (
            -item[1],
            base_rank.get(item[0], 10**9),
            item[0],
        )
    )
    if output_top_pages > 0:
        scored = scored[:output_top_pages]
    return scored


def prediction_rows_from_scored(scored: list[tuple[str, float]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for uid, score in scored:
        parsed = parse_page_uid(uid)
        if parsed is None:
            continue
        rows.append([parsed.doc_id, int(parsed.page_idx), float(score)])
    return rows


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
            try:
                uids.add(page_uid(doc_id, int(page_idx)))
            except (TypeError, ValueError):
                pass
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


def first_rank(ranked: list[str], gold: set[str]) -> int | None:
    for idx, uid in enumerate(ranked, start=1):
        if uid in gold:
            return idx
    return None


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float | None:
    if not gold:
        return None
    return len(set(ranked[:k]) & gold) / float(len(gold))


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


def doc_ranking_from_pages(page_uids: list[str]) -> list[str]:
    docs: list[str] = []
    seen: set[str] = set()
    for uid in page_uids:
        parsed = parse_page_uid(uid)
        if parsed is None or parsed.doc_id in seen:
            continue
        seen.add(parsed.doc_id)
        docs.append(parsed.doc_id)
    return docs


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def rerank_prediction(
    *,
    prediction: dict[str, dict[str, Any]],
    page_catalog: dict[str, PageRecord],
    by_doc: dict[str, list[str]],
    gold_by_qid: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    text_fields: list[str],
    region_fields: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    cases: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    numeric_stats: dict[str, list[float]] = defaultdict(list)
    page_recall_values: dict[int, list[float]] = defaultdict(list)
    doc_recall_values: dict[int, list[float]] = defaultdict(list)
    page_hit_at_k_count = 0
    doc_hit_at_k_count = 0
    movement_counts: Counter[str] = Counter()

    for qid in sorted(prediction):
        row = prediction[qid]
        ranked_pages = ranked_page_uids(row, int(args.input_top_pages))
        if not ranked_pages:
            continue
        top_docs = ranked_doc_ids(ranked_pages, int(args.top_docs))
        candidate_pages = candidate_pages_for_qid(
            ranked_pages=ranked_pages,
            by_doc=by_doc,
            top_docs=int(args.top_docs),
            candidate_scope=str(args.candidate_scope),
            candidate_top_pages=int(args.candidate_top_pages),
            max_pages_per_doc=int(args.max_pages_per_doc),
        )
        question = str(row.get("question", "") or "")
        if not question and qid in gold_by_qid:
            question = str(gold_by_qid[qid].get("question", "") or "")
        evidence_scores, q_stats = evidence_scores_for_qid(
            question=question,
            ranked_pages=ranked_pages,
            candidate_pages=candidate_pages,
            page_catalog=page_catalog,
            args=args,
            region_fields=region_fields,
        )
        for key, value in q_stats.items():
            numeric_stats[key].append(float(value))
        scored = rrf_fuse_pages(
            ranked_pages,
            evidence_scores,
            rrf_k=float(args.rrf_k),
            output_top_pages=int(args.output_top_pages),
        )
        new_row = dict(row)
        new_row["qid"] = qid
        new_row["page_retrieval_results"] = prediction_rows_from_scored(scored)
        output[qid] = new_row

        candidate_ranked_pages = [uid for uid, _score in scored]
        if qid in gold_by_qid:
            gold_row = gold_by_qid[qid]
            gold_pages = gold_page_uids(gold_row)
            gold_docs = gold_doc_ids(gold_row)
            baseline_rank = first_rank(ranked_pages, gold_pages)
            candidate_rank = first_rank(candidate_ranked_pages, gold_pages)
            movement = movement_for_hit(baseline_rank, candidate_rank, int(args.hit_k))
            movement_counts[movement] += 1
            for k in args.recall_ks:
                page_value = recall_at_k(candidate_ranked_pages, gold_pages, int(k))
                if page_value is not None:
                    page_recall_values[int(k)].append(page_value)
                doc_value = recall_at_k(doc_ranking_from_pages(candidate_ranked_pages), gold_docs, int(k))
                if doc_value is not None:
                    doc_recall_values[int(k)].append(doc_value)
            if candidate_rank is not None and candidate_rank <= int(args.hit_k):
                page_hit_at_k_count += 1
            doc_rank = first_rank(doc_ranking_from_pages(candidate_ranked_pages), gold_docs)
            if doc_rank is not None and doc_rank <= int(args.hit_k):
                doc_hit_at_k_count += 1
            cases.append(
                {
                    "qid": qid,
                    "question": gold_row.get("question", question),
                    "gold_page_uids": sorted(gold_pages),
                    "baseline_first_gold_page_rank": baseline_rank,
                    "candidate_first_gold_page_rank": candidate_rank,
                    "movement": movement,
                    "candidate_page_count": q_stats.get("candidate_page_count", 0),
                    "region_count": q_stats.get("region_count", 0),
                    "positive_query_region_count": q_stats.get("positive_query_region_count", 0),
                    "positive_evidence_page_count": q_stats.get("positive_evidence_page_count", 0),
                }
            )

    summary: dict[str, Any] = {
        "qid_count": len(output),
        "input_prediction": str(args.prediction),
        "doc_pages_jsonl": str(args.doc_pages_jsonl),
        "candidate_scope": str(args.candidate_scope),
        "top_docs": int(args.top_docs),
        "input_top_pages": int(args.input_top_pages),
        "candidate_top_pages": int(args.candidate_top_pages),
        "output_top_pages": int(args.output_top_pages),
        "max_pages_per_doc": int(args.max_pages_per_doc),
        "max_regions_per_page": int(args.max_regions_per_page),
        "max_region_tokens": int(args.max_region_tokens),
        "text_fields": text_fields,
        "region_fields": region_fields,
        "bm25_k1": float(args.bm25_k1),
        "bm25_b": float(args.bm25_b),
        "ppr_restart_prob": float(args.ppr_restart_prob),
        "ppr_iters": int(args.ppr_iters),
        "query_region_edge_weight": float(args.query_region_edge_weight),
        "region_page_edge_weight": float(args.region_page_edge_weight),
        "region_adjacent_edge_weight": float(args.region_adjacent_edge_weight),
        "page_restart_weight": float(args.page_restart_weight),
        "rrf_k": float(args.rrf_k),
        "mean_candidate_page_count": mean(numeric_stats["candidate_page_count"]),
        "mean_region_count": mean(numeric_stats["region_count"]),
        "mean_pages_with_regions": mean(numeric_stats["pages_with_regions"]),
        "mean_explicit_region_pages": mean(numeric_stats["explicit_region_pages"]),
        "mean_fallback_region_pages": mean(numeric_stats["fallback_region_pages"]),
        "mean_positive_query_region_count": mean(numeric_stats["positive_query_region_count"]),
        "mean_positive_evidence_page_count": mean(numeric_stats["positive_evidence_page_count"]),
    }
    if gold_by_qid:
        summary.update(
            {
                "page_recall_at_k": {
                    str(k): mean(values) for k, values in sorted(page_recall_values.items())
                },
                "doc_recall_at_k": {
                    str(k): mean(values) for k, values in sorted(doc_recall_values.items())
                },
                f"page_hit_at_{int(args.hit_k)}_count": page_hit_at_k_count,
                f"doc_hit_at_{int(args.hit_k)}_count": doc_hit_at_k_count,
                "movement_vs_input_counts": dict(sorted(movement_counts.items())),
            }
        )
    if args.output_case_json:
        recovered = sorted(
            [case for case in cases if case["movement"] == "recovered"],
            key=lambda case: (
                case["candidate_first_gold_page_rank"] or 10**9,
                case["baseline_first_gold_page_rank"] or 10**9,
                case["qid"],
            ),
        )
        lost = sorted(
            [case for case in cases if case["movement"] == "lost"],
            key=lambda case: (
                case["baseline_first_gold_page_rank"] or 10**9,
                case["candidate_first_gold_page_rank"] or 10**9,
                case["qid"],
            ),
        )
        case_payload = {
            "counts": dict(sorted(movement_counts.items())),
            "top_recovered": recovered[:30],
            "top_lost": lost[:30],
            "cases": cases,
        }
        Path(args.output_case_json).write_text(
            json.dumps(case_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return output, summary


def main() -> None:
    args = parse_args()
    text_fields = args.text_field if args.text_field else DEFAULT_TEXT_FIELDS
    region_fields = args.region_field if args.region_field else DEFAULT_REGION_FIELDS
    prediction = load_prediction(Path(args.prediction))
    page_catalog, by_doc = load_page_catalog(Path(args.doc_pages_jsonl), text_fields=text_fields)
    gold_by_qid = (
        {str(row["qid"]): row for row in read_jsonl(Path(args.gold))}
        if str(args.gold).strip()
        else {}
    )
    output, summary = rerank_prediction(
        prediction=prediction,
        page_catalog=page_catalog,
        by_doc=by_doc,
        gold_by_qid=gold_by_qid,
        args=args,
        text_fields=text_fields,
        region_fields=region_fields,
    )

    Path(args.output_prediction_json).write_text(
        json.dumps({"predictions": output}, ensure_ascii=False),
        encoding="utf-8",
    )
    Path(args.output_summary_json).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"saved_prediction: {args.output_prediction_json}")
    print(f"saved_summary: {args.output_summary_json}")
    for key, value in summary.items():
        if key in {"input_prediction", "doc_pages_jsonl", "text_fields", "region_fields"}:
            continue
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
