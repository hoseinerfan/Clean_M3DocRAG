#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean, median
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit M3DocVQA/MMQA pseudo page labels. The report focuses on how many "
            "page labels each question has and whether multiple labels come from the "
            "same document, which matters when choosing TOP_PAGES_PER_DOC and "
            "TOP_PAGES_PER_QID for recall-oriented training."
        )
    )
    parser.add_argument(
        "--gold",
        action="append",
        required=True,
        help="Augmented gold JSONL, optionally LABEL=path. Repeatable.",
    )
    parser.add_argument("--output-md", default="", help="Optional Markdown report path.")
    parser.add_argument("--output-json", default="", help="Optional JSON report path.")
    parser.add_argument("--output-csv", default="", help="Optional CSV summary path.")
    parser.add_argument(
        "--example-limit",
        type=int,
        default=12,
        help="Maximum high-label/multi-page examples to include per input.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_labeled_path(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, raw_path = spec.split("=", 1)
        label = label.strip() or Path(raw_path).stem
        return label, Path(raw_path)
    path = Path(spec)
    return path.stem, path


def page_doc(uid: str) -> str:
    value = str(uid).strip()
    if "_page" not in value:
        return value
    return value.rsplit("_page", 1)[0]


def gold_page_uids(row: dict[str, Any]) -> list[str]:
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    raw_values = (
        metadata.get("gold_page_uids")
        or metadata.get("pseudo_gold_page_uids")
        or row.get("gold_page_uids")
        or row.get("pseudo_gold_page_uids")
        or []
    )
    out: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        uid = str(value).strip()
        if uid and uid not in seen:
            seen.add(uid)
            out.append(uid)
    return out


def question_type(row: dict[str, Any]) -> str:
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    return str(metadata.get("type", "UNKNOWN")).strip() or "UNKNOWN"


def supporting_doc_count(row: dict[str, Any]) -> int:
    docs: set[str] = set()
    for ctx in row.get("supporting_context", []) or []:
        if not isinstance(ctx, dict):
            continue
        doc_id = str(ctx.get("doc_id", "")).strip()
        if doc_id:
            docs.add(doc_id)
    return len(docs)


def ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def summarize(label: str, path: Path, *, example_limit: int) -> dict[str, Any]:
    rows = read_jsonl(path)
    label_counts: list[int] = []
    labeled_counts: list[int] = []
    max_pages_per_doc_values: list[int] = []
    page_count_hist: Counter[int] = Counter()
    max_per_doc_hist: Counter[int] = Counter()
    qtype_page_counts: dict[str, list[int]] = defaultdict(list)
    qtype_multi_same_doc: Counter[str] = Counter()
    qtype_total: Counter[str] = Counter()
    supporting_doc_hist: Counter[int] = Counter()
    multi_same_doc_examples: list[dict[str, Any]] = []
    high_label_examples: list[dict[str, Any]] = []

    labeled_qids = 0
    multi_same_doc_qids = 0
    ge_5_qids = 0
    ge_10_qids = 0
    total_labels = 0

    for row in rows:
        qid = str(row.get("qid", "")).strip()
        qtype = question_type(row)
        pages = gold_page_uids(row)
        doc_counts = Counter(page_doc(uid) for uid in pages)
        n_pages = len(pages)
        max_per_doc = max(doc_counts.values(), default=0)
        n_support_docs = supporting_doc_count(row)

        label_counts.append(n_pages)
        page_count_hist[n_pages] += 1
        max_per_doc_hist[max_per_doc] += 1
        qtype_page_counts[qtype].append(n_pages)
        qtype_total[qtype] += 1
        supporting_doc_hist[n_support_docs] += 1
        total_labels += n_pages

        if n_pages > 0:
            labeled_qids += 1
            labeled_counts.append(n_pages)
            max_pages_per_doc_values.append(max_per_doc)
        if n_pages >= 5:
            ge_5_qids += 1
        if n_pages >= 10:
            ge_10_qids += 1
        if max_per_doc > 1:
            multi_same_doc_qids += 1
            qtype_multi_same_doc[qtype] += 1
            if len(multi_same_doc_examples) < example_limit:
                multi_same_doc_examples.append(
                    {
                        "qid": qid,
                        "question_type": qtype,
                        "label_count": n_pages,
                        "max_pages_in_one_doc": max_per_doc,
                        "doc_page_counts": dict(sorted(doc_counts.items())),
                        "gold_page_uids": pages,
                    }
                )
        if n_pages >= 5 and len(high_label_examples) < example_limit:
            high_label_examples.append(
                {
                    "qid": qid,
                    "question_type": qtype,
                    "label_count": n_pages,
                    "max_pages_in_one_doc": max_per_doc,
                    "gold_page_uids": pages,
                }
            )

    qtype_summary = []
    for qtype, counts in sorted(qtype_page_counts.items()):
        labeled = sum(1 for value in counts if value > 0)
        multi = int(qtype_multi_same_doc.get(qtype, 0))
        qtype_summary.append(
            {
                "question_type": qtype,
                "qid_count": len(counts),
                "labeled_qid_count": labeled,
                "mean_pages_per_qid": round(float(fmean(counts)), 4) if counts else 0.0,
                "mean_pages_per_labeled_qid": (
                    round(float(fmean([value for value in counts if value > 0])), 4)
                    if labeled
                    else 0.0
                ),
                "multi_same_doc_qid_count": multi,
                "multi_same_doc_fraction": round(ratio(multi, len(counts)), 4),
            }
        )

    return {
        "label": label,
        "path": str(path),
        "qid_count": len(rows),
        "labeled_qid_count": labeled_qids,
        "labeled_qid_fraction": round(ratio(labeled_qids, len(rows)), 6),
        "pseudo_page_label_count": total_labels,
        "mean_pages_per_qid": round(float(fmean(label_counts)), 6) if label_counts else 0.0,
        "median_pages_per_qid": float(median(label_counts)) if label_counts else 0.0,
        "mean_pages_per_labeled_qid": (
            round(float(fmean(labeled_counts)), 6) if labeled_counts else 0.0
        ),
        "median_pages_per_labeled_qid": float(median(labeled_counts)) if labeled_counts else 0.0,
        "max_pages_per_qid": max(label_counts, default=0),
        "qid_count_with_5plus_pages": ge_5_qids,
        "qid_fraction_with_5plus_pages": round(ratio(ge_5_qids, len(rows)), 6),
        "qid_count_with_10plus_pages": ge_10_qids,
        "qid_fraction_with_10plus_pages": round(ratio(ge_10_qids, len(rows)), 6),
        "qid_count_with_multi_page_same_doc": multi_same_doc_qids,
        "qid_fraction_with_multi_page_same_doc": round(ratio(multi_same_doc_qids, len(rows)), 6),
        "mean_max_pages_per_doc_labeled_qid": (
            round(float(fmean(max_pages_per_doc_values)), 6) if max_pages_per_doc_values else 0.0
        ),
        "max_pages_in_one_doc": max(max_pages_per_doc_values, default=0),
        "page_count_hist": {str(k): v for k, v in sorted(page_count_hist.items())},
        "max_pages_per_doc_hist": {str(k): v for k, v in sorted(max_per_doc_hist.items())},
        "supporting_doc_count_hist": {str(k): v for k, v in sorted(supporting_doc_hist.items())},
        "question_type_summary": qtype_summary,
        "multi_same_doc_examples": multi_same_doc_examples,
        "high_label_examples": high_label_examples,
    }


def write_markdown(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines: list[str] = ["# M3DocVQA Pseudo-Page Label Audit", ""]
    lines.extend(
        [
            "| label | qids | labeled qids | labels | mean labels/labeled qid | max labels/qid | qids >=5 labels | qids >=10 labels | qids with >1 page same doc | max pages in one doc |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in summaries:
        lines.append(
            "| {label} | {qid_count} | {labeled_qid_count} ({labeled_qid_fraction:.1%}) | "
            "{pseudo_page_label_count} | {mean_pages_per_labeled_qid:.3f} | "
            "{max_pages_per_qid} | {qid_count_with_5plus_pages} ({qid_fraction_with_5plus_pages:.1%}) | "
            "{qid_count_with_10plus_pages} ({qid_fraction_with_10plus_pages:.1%}) | "
            "{qid_count_with_multi_page_same_doc} ({qid_fraction_with_multi_page_same_doc:.1%}) | "
            "{max_pages_in_one_doc} |".format(**item)
        )

    for item in summaries:
        lines.extend(["", f"## {item['label']}", "", "### Page Count Histogram", ""])
        lines.extend(["| labels per qid | qid count |", "| ---: | ---: |"])
        for count, qid_count in item["page_count_hist"].items():
            lines.append(f"| {count} | {qid_count} |")

        lines.extend(["", "### Max Pages From One Document", ""])
        lines.extend(["| max pages in one doc | qid count |", "| ---: | ---: |"])
        for count, qid_count in item["max_pages_per_doc_hist"].items():
            lines.append(f"| {count} | {qid_count} |")

        lines.extend(["", "### Question-Type Summary", ""])
        lines.extend(
            [
                "| question type | qids | labeled qids | mean labels/qid | mean labels/labeled qid | qids with >1 page same doc |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in item["question_type_summary"]:
            lines.append(
                "| {question_type} | {qid_count} | {labeled_qid_count} | "
                "{mean_pages_per_qid:.3f} | {mean_pages_per_labeled_qid:.3f} | "
                "{multi_same_doc_qid_count} ({multi_same_doc_fraction:.1%}) |".format(**row)
            )

        if item["multi_same_doc_examples"]:
            lines.extend(["", "### Example QIDs With Multiple Pseudo-Gold Pages From One Document", ""])
            for row in item["multi_same_doc_examples"]:
                lines.append(
                    f"- `{row['qid']}` ({row['question_type']}): "
                    f"{row['label_count']} labels, max {row['max_pages_in_one_doc']} pages/doc, "
                    f"doc_page_counts={row['doc_page_counts']}"
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fields = [
        "label",
        "path",
        "qid_count",
        "labeled_qid_count",
        "labeled_qid_fraction",
        "pseudo_page_label_count",
        "mean_pages_per_labeled_qid",
        "max_pages_per_qid",
        "qid_count_with_5plus_pages",
        "qid_count_with_10plus_pages",
        "qid_count_with_multi_page_same_doc",
        "qid_fraction_with_multi_page_same_doc",
        "max_pages_in_one_doc",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in summaries:
            writer.writerow({key: item.get(key) for key in fields})


def main() -> None:
    args = parse_args()
    summaries = [
        summarize(label, path, example_limit=int(args.example_limit))
        for label, path in (parse_labeled_path(spec) for spec in args.gold)
    ]

    print(
        "| label | qids | labeled_qids | labels | mean_labels_labeled | "
        "max_labels_qid | qids_multi_same_doc | max_pages_one_doc |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in summaries:
        print(
            "| {label} | {qid_count} | {labeled_qid_count} | {pseudo_page_label_count} | "
            "{mean_pages_per_labeled_qid:.3f} | {max_pages_per_qid} | "
            "{qid_count_with_multi_page_same_doc} | {max_pages_in_one_doc} |".format(**item)
        )

    if args.output_md:
        write_markdown(Path(args.output_md), summaries)
        print(f"saved_output_md={args.output_md}")
    if args.output_json:
        out = {"summaries": summaries}
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"saved_output_json={args.output_json}")
    if args.output_csv:
        write_csv(Path(args.output_csv), summaries)
        print(f"saved_output_csv={args.output_csv}")


if __name__ == "__main__":
    main()
