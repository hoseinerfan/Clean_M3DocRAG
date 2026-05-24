#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a focused gold JSONL subset for page-localization experiments from a "
            "baseline prediction. The main target is right-document/wrong-page cases."
        )
    )
    parser.add_argument("--gold", required=True, help="Full MMQA_dev.jsonl-style gold file.")
    parser.add_argument("--prediction", required=True, help="Baseline prediction JSON.")
    parser.add_argument(
        "--mode",
        choices=[
            "all",
            "page_miss",
            "right_doc_wrong_page",
            "rankable_page_miss",
            "missing_page",
        ],
        default="right_doc_wrong_page",
    )
    parser.add_argument("--hit-k", type=int, default=4)
    parser.add_argument(
        "--max-page-rank",
        type=int,
        default=1000,
        help="For rankable modes, keep misses whose first gold page is at or above this rank.",
    )
    parser.add_argument(
        "--where",
        action="append",
        default=[],
        help=(
            "Optional metadata filter. FIELD=VALUE requires exact match; FIELD~=VALUE "
            "requires case-insensitive substring match. Repeat for AND filters."
        ),
    )
    parser.add_argument("--limit", type=int, default=0, help="Maximum rows to write. 0 means all.")
    parser.add_argument(
        "--sort-by",
        choices=["page_rank", "doc_rank", "qid"],
        default="page_rank",
        help="Sort subset before applying --limit.",
    )
    parser.add_argument("--output-gold", required=True)
    parser.add_argument("--output-summary-json", required=True)
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
    if isinstance(payload, dict) and isinstance(payload.get("predictions"), (dict, list)):
        payload = payload["predictions"]
    if isinstance(payload, dict):
        iterable = payload.items()
    elif isinstance(payload, list):
        iterable = enumerate(payload)
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


def page_uid(doc_id: str, page_idx: int) -> str:
    return f"{doc_id}_page{int(page_idx)}"


def gold_page_uids(row: dict[str, Any]) -> set[str]:
    metadata = row.get("metadata", {})
    pages = {
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
            pages.add(page_uid(doc_id, int(page_idx)))
    return pages


def gold_doc_ids(row: dict[str, Any]) -> set[str]:
    docs = {
        str(ctx.get("doc_id", "")).strip()
        for ctx in row.get("supporting_context", [])
        if isinstance(ctx, dict) and str(ctx.get("doc_id", "")).strip()
    }
    for uid in gold_page_uids(row):
        if "_page" in uid:
            docs.add(uid.rsplit("_page", 1)[0])
    return docs


def ranked_pages(pred_row: dict[str, Any] | None) -> list[str]:
    if not pred_row:
        return []
    out = []
    seen = set()
    for raw in pred_row.get("page_retrieval_results", []):
        if not isinstance(raw, list) or len(raw) < 2:
            continue
        try:
            uid = page_uid(str(raw[0]), int(raw[1]))
        except (TypeError, ValueError):
            continue
        if uid not in seen:
            seen.add(uid)
            out.append(uid)
    return out


def ranked_docs(pred_row: dict[str, Any] | None) -> list[str]:
    if not pred_row:
        return []
    out = []
    seen = set()
    for raw in pred_row.get("page_retrieval_results", []):
        if not isinstance(raw, list) or len(raw) < 1:
            continue
        doc_id = str(raw[0]).strip()
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)
    return out


def first_rank(items: list[str], gold: set[str]) -> int | None:
    for idx, item in enumerate(items, start=1):
        if item in gold:
            return idx
    return None


def get_path(row: dict[str, Any], path: str) -> Any:
    current: Any = row
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def normalize_values(value: Any) -> list[str]:
    if value is None:
        return ["UNKNOWN"]
    if isinstance(value, list):
        values = [str(item).strip() for item in value if str(item).strip()]
        return values or ["UNKNOWN"]
    text = str(value).strip()
    return [text or "UNKNOWN"]


def parse_filter(raw: str) -> tuple[str, str, str]:
    if "~=" in raw:
        field, value = raw.split("~=", 1)
        op = "~="
    elif "=" in raw:
        field, value = raw.split("=", 1)
        op = "="
    else:
        raise ValueError(f"Invalid --where filter: {raw!r}")
    field = field.strip()
    value = value.strip()
    if not field or not value:
        raise ValueError(f"Invalid --where filter: {raw!r}")
    return field, op, value


def row_matches_filters(row: dict[str, Any], filters: list[tuple[str, str, str]]) -> bool:
    for field, op, expected in filters:
        values = normalize_values(get_path(row, field))
        if op == "=" and expected not in values:
            return False
        if op == "~=":
            expected_lower = expected.lower()
            if not any(expected_lower in value.lower() for value in values):
                return False
    return True


def keep_row(mode: str, page_rank: int | None, doc_rank: int | None, hit_k: int, max_page_rank: int) -> bool:
    page_hit = page_rank is not None and page_rank <= hit_k
    doc_hit = doc_rank is not None and doc_rank <= hit_k
    if mode == "all":
        return True
    if mode == "page_miss":
        return not page_hit
    if mode == "right_doc_wrong_page":
        return doc_hit and not page_hit
    if mode == "rankable_page_miss":
        return (
            doc_hit
            and not page_hit
            and page_rank is not None
            and page_rank <= max_page_rank
        )
    if mode == "missing_page":
        return page_rank is None
    raise ValueError(mode)


def sort_key(row: dict[str, Any], sort_by: str) -> tuple[int, int, str]:
    page_rank = row.get("_subset_page_rank")
    doc_rank = row.get("_subset_doc_rank")
    page_sort = 10**9 if page_rank is None else int(page_rank)
    doc_sort = 10**9 if doc_rank is None else int(doc_rank)
    if sort_by == "page_rank":
        return (page_sort, doc_sort, str(row["qid"]))
    if sort_by == "doc_rank":
        return (doc_sort, page_sort, str(row["qid"]))
    return (0, 0, str(row["qid"]))


def main() -> None:
    args = parse_args()
    filters = [parse_filter(raw) for raw in args.where]
    gold_rows = read_jsonl(Path(args.gold))
    prediction = load_prediction(Path(args.prediction))

    selected = []
    scanned = 0
    filter_matched = 0
    for row in gold_rows:
        qid = str(row.get("qid", "")).strip()
        if not qid:
            continue
        scanned += 1
        if not row_matches_filters(row, filters):
            continue
        filter_matched += 1
        pred_row = prediction.get(qid)
        page_rank = first_rank(ranked_pages(pred_row), gold_page_uids(row))
        doc_rank = first_rank(ranked_docs(pred_row), gold_doc_ids(row))
        if not keep_row(args.mode, page_rank, doc_rank, args.hit_k, args.max_page_rank):
            continue
        item = dict(row)
        item["_subset_page_rank"] = page_rank
        item["_subset_doc_rank"] = doc_rank
        selected.append(item)

    selected.sort(key=lambda row: sort_key(row, args.sort_by))
    if int(args.limit) > 0:
        selected = selected[: int(args.limit)]

    output_rows = []
    for row in selected:
        clean = dict(row)
        clean.pop("_subset_page_rank", None)
        clean.pop("_subset_doc_rank", None)
        output_rows.append(clean)

    output_gold = Path(args.output_gold)
    output_gold.parent.mkdir(parents=True, exist_ok=True)
    with output_gold.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "gold": args.gold,
        "prediction": args.prediction,
        "mode": args.mode,
        "hit_k": int(args.hit_k),
        "max_page_rank": int(args.max_page_rank),
        "where": args.where,
        "scanned_qids": scanned,
        "filter_matched_qids": filter_matched,
        "selected_qids": len(output_rows),
        "output_gold": str(output_gold),
        "examples": [
            {
                "qid": row.get("qid"),
                "question": row.get("question", ""),
                "page_rank": row.get("_subset_page_rank"),
                "doc_rank": row.get("_subset_doc_rank"),
                "metadata": row.get("metadata", {}),
            }
            for row in selected[:10]
        ],
    }
    output_summary = Path(args.output_summary_json)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"scanned_qids {scanned}")
    print(f"filter_matched_qids {filter_matched}")
    print(f"selected_qids {len(output_rows)}")
    print(f"output_gold {output_gold}")
    print(f"output_summary_json {output_summary}")


if __name__ == "__main__":
    main()
