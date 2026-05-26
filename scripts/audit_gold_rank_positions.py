#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_RANK_BANDS = [
    ("top1", 1, 1),
    ("top2_4", 2, 4),
    ("rank5", 5, 5),
    ("rank6_10", 6, 10),
    ("rank11_20", 11, 20),
    ("rank21_100", 21, 100),
    ("rank101_1000", 101, 1000),
    ("rank1001_plus", 1001, None),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit where gold pages appear in a retrieval ranking. Reports slot-level "
            "gold counts, first-gold-rank counts, top-k gold patterns, and rank-(k+1) "
            "boundary opportunities. Also splits first gold page rank bands by first "
            "gold document rank bands so deeper rescue opportunities can be separated "
            "from document-retrieval failures."
        )
    )
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--boundary-rank", type=int, default=5)
    parser.add_argument(
        "--rank-band",
        action="append",
        default=[],
        metavar="LABEL=START-END",
        help=(
            "Optional rank band. Repeatable. Examples: top4=1-4, window=5-20, "
            "deep=21+, missing=missing. Defaults to top1/top2_4/rank5/rank6_10/"
            "rank11_20/rank21_100/rank101_1000/rank1001_plus/missing."
        ),
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--examples", type=int, default=20)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_prediction(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    if isinstance(payload, dict):
        if isinstance(payload.get("predictions"), dict):
            return {str(k): v for k, v in payload["predictions"].items()}
        if isinstance(payload.get("results"), dict):
            return {str(k): v for k, v in payload["results"].items()}
        if all(isinstance(v, dict) for v in payload.values()):
            return {str(k): v for k, v in payload.items()}
    if isinstance(payload, list):
        rows = {}
        for row in payload:
            if isinstance(row, dict) and row.get("qid") is not None:
                rows[str(row["qid"])] = row
        return rows
    raise ValueError(f"Unsupported prediction format: {path}")


def page_uid(doc_id: object, page_idx: object) -> str:
    return f"{doc_id}_page{int(page_idx)}"


def parse_page_uid(uid: str) -> tuple[str, int] | None:
    marker = "_page"
    if marker not in str(uid):
        return None
    doc_id, page_idx = str(uid).rsplit(marker, 1)
    try:
        return doc_id, int(page_idx)
    except ValueError:
        return None


def row_page_uid(row: Any) -> str | None:
    if isinstance(row, (list, tuple)) and len(row) >= 2:
        return page_uid(row[0], row[1])
    if isinstance(row, dict):
        if row.get("page_uid"):
            return str(row["page_uid"])
        doc_id = row.get("doc_id", row.get("doc_name"))
        page_idx = row.get("page_idx", row.get("page_id", row.get("page")))
        if doc_id is not None and page_idx is not None:
            return page_uid(doc_id, page_idx)
    return None


def row_doc_id(row: Any) -> str | None:
    if isinstance(row, (list, tuple)) and len(row) >= 1:
        value = str(row[0]).strip()
        return value or None
    if isinstance(row, dict):
        value = row.get("doc_id", row.get("doc_name"))
        if value is not None:
            value = str(value).strip()
            return value or None
        uid = row_page_uid(row)
        parsed = parse_page_uid(uid) if uid else None
        return parsed[0] if parsed else None
    return None


def prediction_rows(row: dict[str, Any] | None) -> list[Any]:
    if not row:
        return []
    for key in (
        "page_retrieval_results",
        "retrieved_pages",
        "ranked_pages",
        "results",
        "pages",
    ):
        value = row.get(key)
        if isinstance(value, list):
            return value
    return []


def ranked_page_uids(row: dict[str, Any] | None) -> list[str]:
    ranked: list[str] = []
    seen: set[str] = set()
    for item in prediction_rows(row):
        uid = row_page_uid(item)
        if uid and uid not in seen:
            seen.add(uid)
            ranked.append(uid)
    return ranked


def ranked_doc_ids(row: dict[str, Any] | None) -> list[str]:
    ranked: list[str] = []
    seen: set[str] = set()
    for item in prediction_rows(row):
        doc_id = row_doc_id(item)
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            ranked.append(doc_id)
    return ranked


def gold_page_uids(row: dict[str, Any]) -> set[str]:
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    values = metadata.get("gold_page_uids")
    if isinstance(values, list) and values:
        uids = {str(value) for value in values}
    else:
        uids = set()
    doc_ids = metadata.get("gold_doc_ids") or row.get("gold_doc_ids") or []
    page_ids = metadata.get("gold_page_ids") or row.get("gold_page_ids") or []
    if not isinstance(doc_ids, list):
        doc_ids = [doc_ids]
    if not isinstance(page_ids, list):
        page_ids = [page_ids]
    if len(doc_ids) == 1 and len(page_ids) > 1:
        doc_ids = doc_ids * len(page_ids)
    uids.update(
        page_uid(doc_id, page_idx)
        for doc_id, page_idx in zip(doc_ids, page_ids)
        if doc_id is not None and page_idx is not None
    )
    for ctx in row.get("supporting_context", []):
        if not isinstance(ctx, dict):
            continue
        doc_id = ctx.get("doc_id", ctx.get("doc_name"))
        page_idx = ctx.get("page_idx", ctx.get("page_id", ctx.get("page")))
        if doc_id is not None and page_idx is not None:
            uids.add(page_uid(doc_id, page_idx))
    return uids


def gold_doc_ids(row: dict[str, Any], page_gold: set[str]) -> set[str]:
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    values = metadata.get("gold_doc_ids") or row.get("gold_doc_ids") or []
    if not isinstance(values, list):
        values = [values]
    doc_ids = {str(value).strip() for value in values if str(value).strip()}
    for uid in page_gold:
        parsed = parse_page_uid(uid)
        if parsed:
            doc_ids.add(parsed[0])
    for ctx in row.get("supporting_context", []):
        if isinstance(ctx, dict):
            doc_id = str(ctx.get("doc_id", ctx.get("doc_name", ""))).strip()
            if doc_id:
                doc_ids.add(doc_id)
    return doc_ids


def first_rank(ranked: list[str], gold: set[str]) -> int | None:
    for idx, uid in enumerate(ranked, start=1):
        if uid in gold:
            return idx
    return None


def pct(count: int, denom: int) -> str:
    if denom <= 0:
        return "0.00%"
    return f"{100.0 * count / denom:.2f}%"


def parse_rank_band(value: str) -> tuple[str, int | None, int | None]:
    if "=" not in value:
        raise ValueError(f"Rank band must be LABEL=RANGE, got: {value}")
    label, raw_range = value.split("=", 1)
    label = label.strip()
    raw_range = raw_range.strip().lower()
    if not label:
        raise ValueError(f"Rank band label is empty: {value}")
    if raw_range == "missing":
        return label, None, None
    if raw_range.endswith("+"):
        return label, int(raw_range[:-1]), None
    if "-" in raw_range:
        start, end = raw_range.split("-", 1)
        return label, int(start), int(end)
    rank = int(raw_range)
    return label, rank, rank


def rank_band(rank: int | None, bands: list[tuple[str, int | None, int | None]]) -> str:
    if rank is None:
        return "missing"
    for label, start, end in bands:
        if start is None:
            continue
        if rank < start:
            continue
        if end is None or rank <= end:
            return label
    return f"rank{rank}"


def sorted_rank_counts(counter: Counter[str]) -> dict[str, int]:
    def key_fn(item: tuple[str, int]) -> tuple[int, int | str]:
        key, _count = item
        if key == "missing":
            return (1, 10**9)
        if key.isdigit():
            return (0, int(key))
        return (0, key)

    return dict(sorted(counter.items(), key=key_fn))


def main() -> None:
    args = parse_args()
    prediction = load_prediction(Path(args.prediction))
    gold_rows = {str(row.get("qid", "")).strip(): row for row in read_jsonl(Path(args.gold))}
    top_k = int(args.top_k)
    boundary_rank = int(args.boundary_rank)
    rank_bands = (
        [parse_rank_band(value) for value in args.rank_band]
        if args.rank_band
        else list(DEFAULT_RANK_BANDS)
    )

    slot_gold_counts: Counter[int] = Counter()
    first_rank_counts: Counter[str] = Counter()
    first_doc_rank_counts: Counter[str] = Counter()
    first_page_rank_band_counts: Counter[str] = Counter()
    first_doc_rank_band_counts: Counter[str] = Counter()
    page_doc_rank_band_counts: Counter[str] = Counter()
    gold_count_in_topk: Counter[int] = Counter()
    pattern_counts: Counter[str] = Counter()
    boundary_gold_pattern_counts: Counter[str] = Counter()
    page_window_doc_band_examples: dict[str, list[dict[str, Any]]] = {}
    boundary_examples: list[dict[str, Any]] = []
    inner_replacement_opportunities: list[dict[str, Any]] = []
    inner_replacement_opportunity_count = 0

    evaluated = 0
    missing_prediction = 0
    missing_gold = 0

    for qid, gold_row in sorted(gold_rows.items()):
        gold = gold_page_uids(gold_row)
        if not gold:
            missing_gold += 1
            continue
        pred_row = prediction.get(qid)
        if pred_row is None:
            missing_prediction += 1
            continue
        ranked = ranked_page_uids(pred_row)
        ranked_docs = ranked_doc_ids(pred_row)
        doc_gold = gold_doc_ids(gold_row, gold)
        evaluated += 1

        top_pages = ranked[:top_k]
        top_flags = [uid in gold for uid in top_pages]
        for idx, is_gold in enumerate(top_flags, start=1):
            if is_gold:
                slot_gold_counts[idx] += 1
        top_gold_count = sum(1 for flag in top_flags if flag)
        gold_count_in_topk[top_gold_count] += 1
        pattern = "".join("G" if flag else "_" for flag in top_flags)
        pattern_counts[pattern] += 1

        rank = first_rank(ranked, gold)
        first_rank_counts[str(rank) if rank is not None else "missing"] += 1
        doc_rank = first_rank(ranked_docs, doc_gold)
        first_doc_rank_counts[str(doc_rank) if doc_rank is not None else "missing"] += 1
        page_band = rank_band(rank, rank_bands)
        doc_band = rank_band(doc_rank, rank_bands)
        first_page_rank_band_counts[page_band] += 1
        first_doc_rank_band_counts[doc_band] += 1
        band_key = f"{page_band}__doc_{doc_band}"
        page_doc_rank_band_counts[band_key] += 1
        if len(page_window_doc_band_examples.setdefault(band_key, [])) < int(args.examples):
            page_window_doc_band_examples[band_key].append(
                {
                    "qid": qid,
                    "question": gold_row.get("question", ""),
                    "gold_page_uids": sorted(gold),
                    "gold_doc_ids": sorted(doc_gold),
                    "first_gold_page_rank": rank,
                    "first_gold_doc_rank": doc_rank,
                    "top_pages": top_pages,
                }
            )

        boundary_uid = ranked[boundary_rank - 1] if len(ranked) >= boundary_rank else None
        boundary_is_gold = boundary_uid in gold if boundary_uid else False
        boundary_key = (
            f"rank{boundary_rank}_gold__top{top_k}_gold_count_{top_gold_count}"
            if boundary_is_gold
            else f"rank{boundary_rank}_non_gold__top{top_k}_gold_count_{top_gold_count}"
        )
        boundary_gold_pattern_counts[boundary_key] += 1

        if boundary_is_gold:
            item = {
                "qid": qid,
                "question": gold_row.get("question", ""),
                "gold_page_uids": sorted(gold),
                "top_pages": top_pages,
                "top_pattern": pattern,
                f"rank{boundary_rank}_page": boundary_uid,
                f"rank{boundary_rank}_is_gold": True,
                "top_non_gold_ranks": [
                    idx for idx, flag in enumerate(top_flags, start=1) if not flag
                ],
            }
            if len(boundary_examples) < int(args.examples):
                boundary_examples.append(item)
            if any(not flag for flag in top_flags):
                inner_replacement_opportunity_count += 1
                if len(inner_replacement_opportunities) < int(args.examples):
                    inner_replacement_opportunities.append(item)

    summary = {
        "prediction": str(args.prediction),
        "gold": str(args.gold),
        "top_k": top_k,
        "boundary_rank": boundary_rank,
        "evaluated_qids": evaluated,
        "missing_prediction_qids": missing_prediction,
        "missing_gold_qids": missing_gold,
        "slot_gold_counts": {str(k): slot_gold_counts.get(k, 0) for k in range(1, top_k + 1)},
        "slot_gold_rates": {
            str(k): (slot_gold_counts.get(k, 0) / evaluated if evaluated else 0.0)
            for k in range(1, top_k + 1)
        },
        "rank_bands": [
            {"label": label, "start": start, "end": end}
            for label, start, end in rank_bands
        ]
        + [{"label": "missing", "start": None, "end": None}],
        "first_gold_rank_counts": sorted_rank_counts(first_rank_counts),
        "first_gold_doc_rank_counts": sorted_rank_counts(first_doc_rank_counts),
        "first_gold_page_rank_band_counts": {
            label: first_page_rank_band_counts.get(label, 0)
            for label, _start, _end in rank_bands
        }
        | {"missing": first_page_rank_band_counts.get("missing", 0)},
        "first_gold_doc_rank_band_counts": {
            label: first_doc_rank_band_counts.get(label, 0)
            for label, _start, _end in rank_bands
        }
        | {"missing": first_doc_rank_band_counts.get("missing", 0)},
        "page_doc_rank_band_counts": dict(page_doc_rank_band_counts.most_common()),
        "page_doc_rank_band_examples": page_window_doc_band_examples,
        "gold_count_in_topk_counts": {
            str(k): gold_count_in_topk.get(k, 0) for k in range(0, top_k + 1)
        },
        "topk_pattern_counts": dict(pattern_counts.most_common()),
        "boundary_gold_pattern_counts": dict(boundary_gold_pattern_counts.most_common()),
        "rank_boundary_gold_count": sum(
            count
            for key, count in boundary_gold_pattern_counts.items()
            if key.startswith(f"rank{boundary_rank}_gold")
        ),
        "rank_boundary_gold_with_topk_non_gold_count": inner_replacement_opportunity_count,
        "rank_boundary_gold_examples": boundary_examples,
        "inner_replacement_opportunity_examples": inner_replacement_opportunities,
    }

    print(f"evaluated_qids: {evaluated}")
    print(f"missing_prediction_qids: {missing_prediction}")
    print(f"missing_gold_qids: {missing_gold}")
    print("\nslot_gold_counts")
    for rank in range(1, top_k + 1):
        count = slot_gold_counts.get(rank, 0)
        print(f"rank {rank}: {count} ({pct(count, evaluated)})")
    print("\nfirst_gold_rank_counts")
    for key, count in summary["first_gold_rank_counts"].items():
        print(f"{key}: {count} ({pct(int(count), evaluated)})")
    print("\nfirst_gold_doc_rank_counts")
    for key, count in summary["first_gold_doc_rank_counts"].items():
        print(f"{key}: {count} ({pct(int(count), evaluated)})")
    print("\nfirst_gold_page_rank_band_counts")
    for key, count in summary["first_gold_page_rank_band_counts"].items():
        print(f"{key}: {count} ({pct(int(count), evaluated)})")
    print("\nfirst_gold_doc_rank_band_counts")
    for key, count in summary["first_gold_doc_rank_band_counts"].items():
        print(f"{key}: {count} ({pct(int(count), evaluated)})")
    print("\npage_doc_rank_band_counts")
    for key, count in page_doc_rank_band_counts.most_common(30):
        print(f"{key}: {count} ({pct(count, evaluated)})")
    print("\ngold_count_in_topk_counts")
    for key, count in summary["gold_count_in_topk_counts"].items():
        print(f"{key}: {count} ({pct(int(count), evaluated)})")
    print("\ntopk_pattern_counts")
    for key, count in list(pattern_counts.most_common(20)):
        print(f"{key}: {count} ({pct(count, evaluated)})")
    print("\nboundary_gold_pattern_counts")
    for key, count in boundary_gold_pattern_counts.most_common(20):
        print(f"{key}: {count} ({pct(count, evaluated)})")

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"saved_json: {args.output_json}")

    if args.output_md:
        lines = [
            "# Gold Rank Position Audit",
            "",
            f"prediction: `{args.prediction}`",
            f"gold: `{args.gold}`",
            f"evaluated_qids: {evaluated}",
            f"top_k: {top_k}",
            f"boundary_rank: {boundary_rank}",
            "",
            "## Slot Gold Counts",
            "",
            "| rank | gold count | rate |",
            "|---:|---:|---:|",
        ]
        for rank in range(1, top_k + 1):
            count = slot_gold_counts.get(rank, 0)
            lines.append(f"| {rank} | {count} | {pct(count, evaluated)} |")
        lines.extend(
            [
                "",
                "## First Gold Rank Counts",
                "",
                "| first gold rank | qids | rate |",
                "|---:|---:|---:|",
            ]
        )
        for key, count in summary["first_gold_rank_counts"].items():
            lines.append(f"| {key} | {count} | {pct(int(count), evaluated)} |")
        lines.extend(
            [
                "",
                "## First Gold Doc Rank Counts",
                "",
                "| first gold doc rank | qids | rate |",
                "|---:|---:|---:|",
            ]
        )
        for key, count in summary["first_gold_doc_rank_counts"].items():
            lines.append(f"| {key} | {count} | {pct(int(count), evaluated)} |")
        lines.extend(
            [
                "",
                "## First Gold Page Rank Bands",
                "",
                "| page rank band | qids | rate |",
                "|---|---:|---:|",
            ]
        )
        for key, count in summary["first_gold_page_rank_band_counts"].items():
            lines.append(f"| `{key}` | {count} | {pct(int(count), evaluated)} |")
        lines.extend(
            [
                "",
                "## First Gold Doc Rank Bands",
                "",
                "| doc rank band | qids | rate |",
                "|---|---:|---:|",
            ]
        )
        for key, count in summary["first_gold_doc_rank_band_counts"].items():
            lines.append(f"| `{key}` | {count} | {pct(int(count), evaluated)} |")
        doc_band_labels = [label for label, _start, _end in rank_bands] + ["missing"]
        page_band_labels = [label for label, _start, _end in rank_bands] + ["missing"]
        lines.extend(
            [
                "",
                "## Page Rank Band By Doc Rank Band",
                "",
                "| page band | " + " | ".join(f"`{label}`" for label in doc_band_labels) + " |",
                "|---|" + "|".join("---:" for _label in doc_band_labels) + "|",
            ]
        )
        for page_band_label in page_band_labels:
            row_counts = [
                page_doc_rank_band_counts.get(
                    f"{page_band_label}__doc_{doc_band_label}",
                    0,
                )
                for doc_band_label in doc_band_labels
            ]
            lines.append(
                f"| `{page_band_label}` | "
                + " | ".join(str(value) for value in row_counts)
                + " |"
            )
        lines.extend(
            [
                "",
                "## Gold Count In Top-K",
                "",
                "| gold pages in top-k | qids | rate |",
                "|---:|---:|---:|",
            ]
        )
        for key, count in summary["gold_count_in_topk_counts"].items():
            lines.append(f"| {key} | {count} | {pct(int(count), evaluated)} |")
        lines.extend(
            [
                "",
                "## Top-K Gold Patterns",
                "",
                "`G` means the slot is gold and `_` means it is not gold.",
                "",
                "| pattern | qids | rate |",
                "|---|---:|---:|",
            ]
        )
        for key, count in pattern_counts.most_common(30):
            lines.append(f"| `{key}` | {count} | {pct(count, evaluated)} |")
        lines.extend(
            [
                "",
                "## Boundary Rank Patterns",
                "",
                "| pattern | qids | rate |",
                "|---|---:|---:|",
            ]
        )
        for key, count in boundary_gold_pattern_counts.most_common(30):
            lines.append(f"| `{key}` | {count} | {pct(count, evaluated)} |")
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"saved_md: {args.output_md}")


if __name__ == "__main__":
    main()
