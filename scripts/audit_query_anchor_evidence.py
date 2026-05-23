#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_TEXT_FIELDS = ["ocr_text", "vlm_text", "markdown", "text", "page_text", "content"]


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
        raise TypeError(f"Unsupported prediction payload: {path}")

    rows: dict[str, dict[str, Any]] = {}
    for raw_key, row in iterable:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("qid", "")).strip() or str(raw_key).strip()
        if qid:
            rows[qid] = row
    return rows


def load_summary_per_qid(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    per_qid = payload.get("per_qid", []) if isinstance(payload, dict) else []
    rows = {}
    for row in per_qid:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("qid", "")).strip()
        if qid:
            rows[qid] = row
    return rows


def page_uid(doc_id: str, page_idx: int) -> str:
    return f"{doc_id}_page{int(page_idx)}"


def parse_page_row(row: list[Any]) -> tuple[str, int] | None:
    if not isinstance(row, list) or len(row) < 2:
        return None
    doc_id = str(row[0]).strip()
    if not doc_id:
        return None
    try:
        page_idx = int(row[1])
    except (TypeError, ValueError):
        return None
    return doc_id, page_idx


def ranked_pages(pred_row: dict[str, Any]) -> list[str]:
    pages = []
    for row in pred_row.get("page_retrieval_results", []):
        parsed = parse_page_row(row)
        if parsed is not None:
            pages.append(page_uid(parsed[0], parsed[1]))
    return pages


def first_rank(items: list[str], gold: set[str]) -> int | None:
    for idx, item in enumerate(items, start=1):
        if item in gold:
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
            uids.add(page_uid(doc_id, int(page_idx)))
    return uids


def metadata_value(row: dict[str, Any], dotted_key: str) -> str:
    current: Any = row
    for part in dotted_key.split("."):
        if not isinstance(current, dict):
            return "UNKNOWN"
        current = current.get(part)
    return str(current).strip() or "UNKNOWN"


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


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value.replace("\x0c", " ").replace("\u0000", " ")).strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return " ".join(part for part in (normalize_text(item) for item in value) if part)
    if isinstance(value, dict):
        return " ".join(part for part in (normalize_text(item) for item in value.values()) if part)
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_anchor(anchor: str) -> str:
    normalized = re.sub(r"\s+", " ", str(anchor or "").strip(" \t\n\r,.;:!?()[]{}\"'`"))
    normalized = normalized.replace("’", "'")
    if normalized.lower().endswith("'s"):
        normalized = normalized[:-2]
    return normalized.strip().lower()


def anchor_matches_text(anchor: str, text: str) -> bool:
    normalized_anchor = normalize_anchor(anchor)
    if not normalized_anchor or not text:
        return False
    if re.match(r"^[a-z0-9_.+-]+$", normalized_anchor):
        return bool(
            re.search(rf"(?<![a-z0-9]){re.escape(normalized_anchor)}(?![a-z0-9])", text)
        )
    return normalized_anchor in text


def load_page_texts(path: Path, text_fields: list[str]) -> dict[str, str]:
    page_texts = {}
    for row in read_jsonl(path):
        doc_id = str(row.get("doc_id", "")).strip()
        raw_page_idx = row.get("page_idx", row.get("page_id", row.get("page_number")))
        if not doc_id or raw_page_idx is None:
            continue
        try:
            page_idx = int(raw_page_idx)
        except (TypeError, ValueError):
            continue
        parts = []
        seen = set()
        for field in text_fields:
            text = normalize_text(row.get(field))
            if text and text not in seen:
                parts.append(text)
                seen.add(text)
        if parts:
            page_texts[page_uid(doc_id, page_idx)] = " ".join(parts).lower()
    return page_texts


def bucket_page_matches(value: int) -> str:
    if value <= 0:
        return "0"
    if value <= 20:
        return "1-20"
    if value <= 100:
        return "21-100"
    if value <= 500:
        return "101-500"
    return ">500"


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    movement = Counter(row["movement"] for row in rows)
    return {
        "n": len(rows),
        "recovered": movement.get("recovered", 0),
        "lost": movement.get("lost", 0),
        "improved_rank": movement.get("improved_rank", 0),
        "worsened_rank": movement.get("worsened_rank", 0),
        "active": sum(1 for row in rows if row["matched_anchor_count"] > 0),
        "mean_matched_anchors": (
            statistics.fmean(row["matched_anchor_count"] for row in rows) if rows else 0.0
        ),
        "mean_page_matches": (
            statistics.fmean(row["page_match_count"] for row in rows) if rows else 0.0
        ),
        "gold_anchor_match": sum(1 for row in rows if row.get("gold_anchor_match") is True),
    }


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit query-anchor evidence behavior against a graph baseline."
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--candidate-summary", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--doc-pages-jsonl", default="")
    parser.add_argument("--text-field", nargs="*", default=DEFAULT_TEXT_FIELDS)
    parser.add_argument("--hit-k", type=int, default=4)
    parser.add_argument("--topn", type=int, default=20)
    parser.add_argument(
        "--filter-field",
        default="",
        help="Optional dotted gold-row field to audit a subset, e.g. metadata.domain.",
    )
    parser.add_argument(
        "--filter-value",
        action="append",
        default=[],
        help="Allowed exact value for --filter-field. Repeat to allow multiple values.",
    )
    parser.add_argument("--output-md", default="")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    baseline = load_prediction(Path(args.baseline))
    candidate = load_prediction(Path(args.candidate))
    summary_by_qid = load_summary_per_qid(Path(args.candidate_summary))
    gold_by_qid = {str(row["qid"]): row for row in read_jsonl(Path(args.gold))}
    page_texts = (
        load_page_texts(Path(args.doc_pages_jsonl), list(args.text_field))
        if args.doc_pages_jsonl
        else {}
    )

    rows = []
    for qid in sorted(set(baseline) & set(candidate) & set(gold_by_qid)):
        gold_row = gold_by_qid[qid]
        if args.filter_field:
            field_value = metadata_value(gold_row, args.filter_field)
            if args.filter_value and field_value not in set(args.filter_value):
                continue
        gold_pages = gold_page_uids(gold_row)
        base_rank = first_rank(ranked_pages(baseline[qid]), gold_pages)
        cand_rank = first_rank(ranked_pages(candidate[qid]), gold_pages)
        movement = movement_for_hit(base_rank, cand_rank, int(args.hit_k))
        graph = summary_by_qid.get(qid, {}).get("graph", {})
        anchor_labels = [str(value) for value in graph.get("query_anchor_labels", [])]
        matched_anchor_count = int(graph.get("query_anchor_matched_anchor_count", 0) or 0)
        page_match_count = int(graph.get("query_anchor_page_match_count", 0) or 0)
        gold_anchor_match = None
        matched_gold_anchors: list[str] = []
        if page_texts and anchor_labels and gold_pages:
            gold_anchor_match = False
            for gold_page in gold_pages:
                text = page_texts.get(gold_page, "")
                for anchor in anchor_labels:
                    if anchor_matches_text(anchor, text):
                        gold_anchor_match = True
                        matched_gold_anchors.append(anchor)
            matched_gold_anchors = sorted(set(matched_gold_anchors))
        rows.append(
            {
                "qid": qid,
                "question": gold_row.get("question", ""),
                "metadata.type": metadata_value(gold_row, "metadata.type"),
                "metadata.domain": metadata_value(gold_row, "metadata.domain"),
                "gold_page_uids": sorted(gold_pages),
                "baseline_rank": base_rank,
                "candidate_rank": cand_rank,
                "movement": movement,
                "active_anchor_count": int(graph.get("query_anchor_active_anchor_count", 0) or 0),
                "matched_anchor_count": matched_anchor_count,
                "page_match_count": page_match_count,
                "page_match_bucket": bucket_page_matches(page_match_count),
                "edge_count": int(graph.get("query_anchor_edge_count_directed", 0) or 0),
                "restart_seed_node_count": int(
                    graph.get("query_anchor_restart_seed_node_count", 0) or 0
                ),
                "anchor_labels": anchor_labels,
                "gold_anchor_match": gold_anchor_match,
                "matched_gold_anchors": matched_gold_anchors,
                "financial_reasoning_active": bool(
                    graph.get("query_anchor_financial_reasoning_active", False)
                ),
                "financial_reasoning_reason": str(
                    graph.get("query_anchor_financial_reasoning_reason", "")
                ),
                "financial_metric_anchor_count": int(
                    graph.get("query_anchor_financial_metric_anchor_count", 0) or 0
                ),
                "financial_year_anchor_count": int(
                    graph.get("query_anchor_financial_year_anchor_count", 0) or 0
                ),
                "financial_entity_anchor_count": int(
                    graph.get("query_anchor_financial_entity_anchor_count", 0) or 0
                ),
                "financial_bundle_page_match_count": int(
                    graph.get("query_anchor_financial_bundle_page_match_count", 0) or 0
                ),
                "financial_bundle_label": str(
                    graph.get("query_anchor_financial_bundle_label", "")
                ),
            }
        )

    movement_counts = Counter(row["movement"] for row in rows)
    active_rows = [row for row in rows if row["matched_anchor_count"] > 0]
    inactive_rows = [row for row in rows if row["matched_anchor_count"] <= 0]
    bucket_summaries = {
        bucket: summarize_group([row for row in rows if row["page_match_bucket"] == bucket])
        for bucket in ["0", "1-20", "21-100", "101-500", ">500"]
    }
    domain_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    type_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        domain_groups[row["metadata.domain"]].append(row)
        type_groups[row["metadata.type"]].append(row)
    by_domain = {key: summarize_group(group) for key, group in domain_groups.items()}
    by_type = {key: summarize_group(group) for key, group in type_groups.items()}
    financial_reason_counts = Counter(row["financial_reasoning_reason"] for row in rows)

    def top_cases(name: str) -> list[dict[str, Any]]:
        selected = [row for row in rows if row["movement"] == name]
        return sorted(
            selected,
            key=lambda row: (
                row["candidate_rank"] if row["candidate_rank"] is not None else 10**9,
                row["baseline_rank"] if row["baseline_rank"] is not None else 10**9,
                row["qid"],
            ),
        )[: int(args.topn)]

    payload = {
        "n_qids": len(rows),
        "hit_k": int(args.hit_k),
        "movement_counts": dict(movement_counts),
        "active_summary": summarize_group(active_rows),
        "inactive_summary": summarize_group(inactive_rows),
        "page_match_bucket_summaries": bucket_summaries,
        "by_domain": by_domain,
        "by_type": by_type,
        "top_recovered": top_cases("recovered"),
        "top_lost": top_cases("lost"),
        "top_improved_rank": top_cases("improved_rank"),
        "top_worsened_rank": top_cases("worsened_rank"),
        "page_text_available": bool(page_texts),
        "page_text_count": len(page_texts),
        "financial_reasoning_active_count": sum(
            1 for row in rows if row["financial_reasoning_active"]
        ),
        "financial_reasoning_reason_counts": dict(financial_reason_counts),
        "mean_financial_bundle_page_match_count": (
            statistics.fmean(row["financial_bundle_page_match_count"] for row in rows)
            if rows
            else 0.0
        ),
        "filter": {
            "field": args.filter_field,
            "values": list(args.filter_value),
        },
    }

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Query Anchor Evidence Audit",
        "",
        f"- qids: `{len(rows)}`",
        f"- hit_k: `{int(args.hit_k)}`",
        f"- page_text_available: `{bool(page_texts)}`",
        f"- page_text_count: `{len(page_texts)}`",
        f"- filter_field: `{args.filter_field or ''}`",
        f"- filter_value: `{', '.join(args.filter_value) if args.filter_value else ''}`",
        "",
        "## Movement",
        "",
        markdown_table(
            ["movement", "count"],
            [[key, movement_counts.get(key, 0)] for key in sorted(movement_counts)],
        ),
        "",
        "## Financial Reasoning",
        "",
        markdown_table(
            ["metric", "value"],
            [
                ["active_count", payload["financial_reasoning_active_count"]],
                [
                    "mean_bundle_page_matches",
                    f"{payload['mean_financial_bundle_page_match_count']:.1f}",
                ],
            ],
        ),
        "",
        markdown_table(
            ["reason", "count"],
            [
                [key or "missing", value]
                for key, value in sorted(payload["financial_reasoning_reason_counts"].items())
            ],
        ),
        "",
        "## Active vs Inactive",
        "",
        markdown_table(
            ["group", "n", "recovered", "lost", "improved", "worsened", "gold_anchor_match"],
            [
                [
                    "active",
                    payload["active_summary"]["n"],
                    payload["active_summary"]["recovered"],
                    payload["active_summary"]["lost"],
                    payload["active_summary"]["improved_rank"],
                    payload["active_summary"]["worsened_rank"],
                    payload["active_summary"]["gold_anchor_match"],
                ],
                [
                    "inactive",
                    payload["inactive_summary"]["n"],
                    payload["inactive_summary"]["recovered"],
                    payload["inactive_summary"]["lost"],
                    payload["inactive_summary"]["improved_rank"],
                    payload["inactive_summary"]["worsened_rank"],
                    payload["inactive_summary"]["gold_anchor_match"],
                ],
            ],
        ),
        "",
        "## Page Match Buckets",
        "",
        markdown_table(
            ["bucket", "n", "recovered", "lost", "improved", "worsened", "mean_matches"],
            [
                [
                    bucket,
                    summary["n"],
                    summary["recovered"],
                    summary["lost"],
                    summary["improved_rank"],
                    summary["worsened_rank"],
                    f"{summary['mean_page_matches']:.1f}",
                ]
                for bucket, summary in bucket_summaries.items()
            ],
        ),
        "",
        "## Domain Summary",
        "",
        markdown_table(
            ["domain", "n", "active", "recovered", "lost", "improved", "worsened"],
            [
                [
                    domain,
                    summary["n"],
                    summary["active"],
                    summary["recovered"],
                    summary["lost"],
                    summary["improved_rank"],
                    summary["worsened_rank"],
                ]
                for domain, summary in sorted(
                    by_domain.items(),
                    key=lambda item: (-(item[1]["recovered"] - item[1]["lost"]), item[0]),
                )
            ],
        ),
        "",
        "## Type Summary",
        "",
        markdown_table(
            ["type", "n", "active", "recovered", "lost", "improved", "worsened"],
            [
                [
                    group,
                    summary["n"],
                    summary["active"],
                    summary["recovered"],
                    summary["lost"],
                    summary["improved_rank"],
                    summary["worsened_rank"],
                ]
                for group, summary in sorted(
                    by_type.items(),
                    key=lambda item: (-(item[1]["recovered"] - item[1]["lost"]), item[0]),
                )
            ],
        ),
        "",
        "## Top Recovered",
        "",
    ]
    for row in payload["top_recovered"]:
        lines.extend(
            [
                f"- `{row['qid']}` base={row['baseline_rank']} cand={row['candidate_rank']} "
                f"type=`{row['metadata.type']}` domain=`{row['metadata.domain']}`",
                f"  - question: {row['question']}",
                f"  - anchors: {row['anchor_labels'][:12]}",
                f"  - gold_anchor_match: {row['gold_anchor_match']} {row['matched_gold_anchors'][:12]}",
                f"  - financial_reasoning: active={row['financial_reasoning_active']} "
                f"reason={row['financial_reasoning_reason']} "
                f"matches={row['financial_bundle_page_match_count']} "
                f"bundle={row['financial_bundle_label']}",
            ]
        )
    lines.extend(["", "## Top Lost", ""])
    for row in payload["top_lost"]:
        lines.extend(
            [
                f"- `{row['qid']}` base={row['baseline_rank']} cand={row['candidate_rank']} "
                f"type=`{row['metadata.type']}` domain=`{row['metadata.domain']}`",
                f"  - question: {row['question']}",
                f"  - anchors: {row['anchor_labels'][:12]}",
                f"  - gold_anchor_match: {row['gold_anchor_match']} {row['matched_gold_anchors'][:12]}",
                f"  - financial_reasoning: active={row['financial_reasoning_active']} "
                f"reason={row['financial_reasoning_reason']} "
                f"matches={row['financial_bundle_page_match_count']} "
                f"bundle={row['financial_bundle_label']}",
            ]
        )
    lines.extend(["", "## Top Improved Rank", ""])
    for row in payload["top_improved_rank"]:
        lines.extend(
            [
                f"- `{row['qid']}` base={row['baseline_rank']} cand={row['candidate_rank']} "
                f"type=`{row['metadata.type']}` domain=`{row['metadata.domain']}`",
                f"  - question: {row['question']}",
                f"  - anchors: {row['anchor_labels'][:12]}",
                f"  - gold_anchor_match: {row['gold_anchor_match']} {row['matched_gold_anchors'][:12]}",
                f"  - financial_reasoning: active={row['financial_reasoning_active']} "
                f"reason={row['financial_reasoning_reason']} "
                f"matches={row['financial_bundle_page_match_count']} "
                f"bundle={row['financial_bundle_label']}",
            ]
        )
    lines.extend(["", "## Top Worsened Rank", ""])
    for row in payload["top_worsened_rank"]:
        lines.extend(
            [
                f"- `{row['qid']}` base={row['baseline_rank']} cand={row['candidate_rank']} "
                f"type=`{row['metadata.type']}` domain=`{row['metadata.domain']}`",
                f"  - question: {row['question']}",
                f"  - anchors: {row['anchor_labels'][:12]}",
                f"  - gold_anchor_match: {row['gold_anchor_match']} {row['matched_gold_anchors'][:12]}",
                f"  - financial_reasoning: active={row['financial_reasoning_active']} "
                f"reason={row['financial_reasoning_reason']} "
                f"matches={row['financial_bundle_page_match_count']} "
                f"bundle={row['financial_bundle_label']}",
            ]
        )

    markdown = "\n".join(lines) + "\n"
    if args.output_md:
        Path(args.output_md).write_text(markdown, encoding="utf-8")
    else:
        print(markdown)

    print("n_qids", payload["n_qids"])
    print("movement_counts", payload["movement_counts"])
    print("active_summary", payload["active_summary"])
    print("saved_md", args.output_md)
    print("saved_json", args.output_json)


if __name__ == "__main__":
    main()
