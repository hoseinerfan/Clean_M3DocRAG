#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether an exported M3DocVQA page-text JSONL is aligned with "
            "the current PDF files. The script compares page_text row counts and "
            "page_idx ranges against pdfinfo page counts for each document."
        )
    )
    parser.add_argument("--page-text-jsonl", required=True)
    parser.add_argument("--pdf-dir", required=True)
    parser.add_argument("--pdfinfo-bin", default="pdfinfo")
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-md", default="")
    parser.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Optional limit for quick smoke checks. Default audits all docs in the page-text file.",
    )
    parser.add_argument(
        "--example-limit",
        type=int,
        default=20,
        help="Maximum mismatch examples to include in the Markdown report.",
    )
    return parser.parse_args()


def load_page_text_index(path: Path) -> dict[str, list[int]]:
    pages_by_doc: dict[str, list[int]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            doc_id = str(row.get("doc_id", "")).strip()
            if not doc_id:
                continue
            pages_by_doc[doc_id].append(int(row.get("page_idx", -1)))
    return dict(pages_by_doc)


def pdf_page_count(pdfinfo_bin: str, pdf_path: Path) -> int:
    result = subprocess.run(
        [pdfinfo_bin, str(pdf_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise ValueError(f"Could not parse page count from pdfinfo output for {pdf_path}")


def sorted_counter_values(counter: Counter[int]) -> list[int]:
    return sorted(counter)


def audit_doc(doc_id: str, page_indices: list[int], *, pdf_dir: Path, pdfinfo_bin: str) -> dict[str, Any]:
    pdf_path = pdf_dir / f"{doc_id}.pdf"
    page_counter = Counter(page_indices)
    sorted_indices = sorted(page_counter)
    duplicate_indices = sorted(idx for idx, count in page_counter.items() if count > 1)

    row: dict[str, Any] = {
        "doc_id": doc_id,
        "pdf_path": str(pdf_path),
        "pdf_exists": pdf_path.exists(),
        "pdf_page_count": None,
        "page_text_row_count": len(page_indices),
        "unique_page_text_count": len(page_counter),
        "min_page_idx": sorted_indices[0] if sorted_indices else None,
        "max_page_idx": sorted_indices[-1] if sorted_indices else None,
        "duplicate_page_indices": duplicate_indices,
        "missing_page_indices": [],
        "extra_page_indices": [],
        "status": "unknown",
        "error": "",
    }

    if not pdf_path.exists():
        row["status"] = "missing_pdf"
        return row

    try:
        total_pages = pdf_page_count(pdfinfo_bin, pdf_path)
    except Exception as exc:  # pragma: no cover - depends on system pdfinfo
        row["status"] = "pdfinfo_error"
        row["error"] = str(exc)
        return row

    expected_indices = set(range(total_pages))
    observed_indices = set(page_counter)
    missing = sorted(expected_indices - observed_indices)
    extra = sorted(idx for idx in observed_indices if idx < 0 or idx >= total_pages)

    row["pdf_page_count"] = total_pages
    row["missing_page_indices"] = missing
    row["extra_page_indices"] = extra

    if duplicate_indices:
        row["status"] = "duplicate_page_text_rows"
    elif missing and extra:
        row["status"] = "missing_and_extra_page_text_rows"
    elif missing:
        row["status"] = "missing_page_text_rows"
    elif extra:
        row["status"] = "extra_page_text_rows"
    elif len(page_counter) != total_pages:
        row["status"] = "count_mismatch"
    else:
        row["status"] = "aligned"

    return row


def ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_md(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]], *, example_limit: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# M3DocVQA Page-Text Alignment Audit",
        "",
        "This audit compares exported page-text rows with the current PDF page counts.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key in [
        "doc_count",
        "aligned_doc_count",
        "mismatched_doc_count",
        "missing_pdf_count",
        "pdfinfo_error_count",
        "extra_page_text_row_count",
        "missing_page_text_row_count",
        "duplicate_page_text_row_count",
    ]:
        lines.append(f"| {key} | {summary[key]} |")

    lines.extend(["", "## Status Counts", "", "| status | docs |", "| --- | ---: |"])
    for status, count in summary["status_counts"].items():
        lines.append(f"| {status} | {count} |")

    mismatches = [row for row in rows if row["status"] != "aligned"]
    if mismatches:
        lines.extend(
            [
                "",
                f"## Mismatch Examples (first {min(example_limit, len(mismatches))})",
                "",
                "| doc_id | status | pdf pages | text rows | max text idx | missing | extra | duplicates |",
                "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
            ]
        )
        for row in mismatches[:example_limit]:
            missing = ",".join(map(str, row.get("missing_page_indices", [])[:12]))
            extra = ",".join(map(str, row.get("extra_page_indices", [])[:12]))
            dup = ",".join(map(str, row.get("duplicate_page_indices", [])[:12]))
            lines.append(
                "| {doc_id} | {status} | {pdf_pages} | {text_rows} | {max_idx} | {missing} | {extra} | {dup} |".format(
                    doc_id=row["doc_id"],
                    status=row["status"],
                    pdf_pages=row.get("pdf_page_count"),
                    text_rows=row.get("page_text_row_count"),
                    max_idx=row.get("max_page_idx"),
                    missing=missing,
                    extra=extra,
                    dup=dup,
                )
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    page_text_jsonl = Path(args.page_text_jsonl)
    pdf_dir = Path(args.pdf_dir)

    pages_by_doc = load_page_text_index(page_text_jsonl)
    doc_ids = sorted(pages_by_doc)
    if int(args.max_docs) > 0:
        doc_ids = doc_ids[: int(args.max_docs)]

    rows: list[dict[str, Any]] = []
    for idx, doc_id in enumerate(doc_ids, start=1):
        row = audit_doc(doc_id, pages_by_doc[doc_id], pdf_dir=pdf_dir, pdfinfo_bin=args.pdfinfo_bin)
        rows.append(row)
        if idx % 500 == 0:
            print(f"audited_docs={idx}/{len(doc_ids)}")

    status_counts = Counter(row["status"] for row in rows)
    mismatched = [row for row in rows if row["status"] != "aligned"]
    extra_page_text_row_count = sum(len(row.get("extra_page_indices", [])) for row in rows)
    missing_page_text_row_count = sum(len(row.get("missing_page_indices", [])) for row in rows)
    duplicate_page_text_row_count = sum(len(row.get("duplicate_page_indices", [])) for row in rows)

    summary: dict[str, Any] = {
        "page_text_jsonl": str(page_text_jsonl),
        "pdf_dir": str(pdf_dir),
        "doc_count": len(rows),
        "aligned_doc_count": status_counts.get("aligned", 0),
        "mismatched_doc_count": len(mismatched),
        "mismatched_doc_fraction": round(ratio(len(mismatched), len(rows)), 6),
        "missing_pdf_count": status_counts.get("missing_pdf", 0),
        "pdfinfo_error_count": status_counts.get("pdfinfo_error", 0),
        "extra_page_text_row_count": extra_page_text_row_count,
        "missing_page_text_row_count": missing_page_text_row_count,
        "duplicate_page_text_row_count": duplicate_page_text_row_count,
        "status_counts": dict(sorted(status_counts.items())),
        "output_jsonl": str(args.output_jsonl),
        "output_summary_json": str(args.output_summary_json),
        "output_md": str(args.output_md),
    }

    write_jsonl(Path(args.output_jsonl), rows)
    write_json(Path(args.output_summary_json), summary)
    if args.output_md:
        write_md(Path(args.output_md), summary, rows, example_limit=int(args.example_limit))

    print(f"saved_jsonl={args.output_jsonl}")
    print(f"saved_summary={args.output_summary_json}")
    if args.output_md:
        print(f"saved_md={args.output_md}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
