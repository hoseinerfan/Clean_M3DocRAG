#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from statistics import fmean
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

from m3docrag.datasets.m3_docvqa import M3DocVQADataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export one text record per M3DocVQA page using pdftotext. "
            "This is intended as the page-text source for sparse retrieval experiments "
            "such as SPLADE or BM25 over the full corpus."
        )
    )
    parser.add_argument("--data-name", default="m3-docvqa")
    parser.add_argument("--split", default="dev")
    parser.add_argument(
        "--doc-id-json",
        help=(
            "Optional JSON / JSONL / plain-text file listing doc ids to export. "
            "If omitted, exports all supporting docs from the split."
        ),
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Optional limit on docs to export for smoke tests. Use 0 for all docs.",
    )
    parser.add_argument(
        "--max-pages-per-doc",
        type=int,
        default=0,
        help="Optional limit on pages exported per doc for smoke tests. Use 0 for all pages.",
    )
    parser.add_argument(
        "--pdftotext-bin",
        default="pdftotext",
        help="Path to pdftotext. Default: pdftotext",
    )
    parser.add_argument(
        "--pdfinfo-bin",
        default="pdfinfo",
        help="Path to pdfinfo. Default: pdfinfo",
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-summary-json", required=True)
    return parser.parse_args()


def make_dataset_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        data_name=args.data_name,
        split=args.split,
        loop_unique_doc_ids=False,
        data_len=None,
    )


def load_doc_ids(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected list in JSON file: {path}")
        return dedupe_doc_ids(payload)

    doc_ids: list[object] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                doc_ids.append(json.loads(line))
            except json.JSONDecodeError:
                doc_ids.append(line)
    return dedupe_doc_ids(doc_ids)


def dedupe_doc_ids(values: list[object]) -> list[str]:
    doc_ids: list[str] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, dict):
            raw_doc_id = value.get("doc_id")
        else:
            raw_doc_id = value
        doc_id = str(raw_doc_id or "").strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        doc_ids.append(doc_id)
    return doc_ids


def get_pdf_page_count(*, pdfinfo_bin: str, pdf_path: Path) -> int:
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


def extract_pdf_page_text(*, pdftotext_bin: str, pdf_path: Path, page_idx: int) -> str:
    page_num = int(page_idx) + 1
    result = subprocess.run(
        [
            pdftotext_bin,
            "-f",
            str(page_num),
            "-l",
            str(page_num),
            str(pdf_path),
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def normalize_page_text(text: str) -> str:
    cleaned = str(text).replace("\x0c", " ").replace("\u0000", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def lexical_token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", text))


def main() -> None:
    args = parse_args()

    dataset = M3DocVQADataset(make_dataset_args(args))
    if args.doc_id_json:
        doc_ids = load_doc_ids(Path(args.doc_id_json))
    else:
        doc_ids = [str(doc_id) for doc_id in dataset.all_supporting_doc_ids]

    if int(args.max_docs) > 0:
        doc_ids = doc_ids[: int(args.max_docs)]

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json = Path(args.output_summary_json)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)

    page_count = 0
    empty_text_page_count = 0
    missing_pdf_doc_ids: list[str] = []
    char_counts: list[int] = []
    token_counts: list[int] = []
    exported_doc_page_counts: dict[str, int] = {}

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for doc_idx, doc_id in enumerate(doc_ids, start=1):
            pdf_path = dataset.pdf_dir / f"{doc_id}.pdf"
            if not pdf_path.exists():
                missing_pdf_doc_ids.append(doc_id)
                continue

            total_pages = get_pdf_page_count(
                pdfinfo_bin=args.pdfinfo_bin,
                pdf_path=pdf_path,
            )
            page_limit = total_pages
            if int(args.max_pages_per_doc) > 0:
                page_limit = min(total_pages, int(args.max_pages_per_doc))

            exported_doc_page_counts[doc_id] = page_limit

            for page_idx in range(page_limit):
                raw_text = extract_pdf_page_text(
                    pdftotext_bin=args.pdftotext_bin,
                    pdf_path=pdf_path,
                    page_idx=page_idx,
                )
                text = normalize_page_text(raw_text)
                char_count = len(text)
                token_count = lexical_token_count(text)
                if not text:
                    empty_text_page_count += 1
                char_counts.append(char_count)
                token_counts.append(token_count)
                page_count += 1

                row = {
                    "page_uid": f"{doc_id}_page{page_idx}",
                    "doc_id": doc_id,
                    "page_idx": int(page_idx),
                    "text": text,
                    "char_count": int(char_count),
                    "token_count": int(token_count),
                }
                handle.write(json.dumps(row) + "\n")

            print(
                f"exported_doc: {doc_idx}/{len(doc_ids)} "
                f"doc_id={doc_id} pages={page_limit}/{total_pages}"
            )

    summary = {
        "data_name": args.data_name,
        "split": args.split,
        "doc_id_json": args.doc_id_json,
        "requested_doc_count": len(doc_ids),
        "exported_doc_count": len(exported_doc_page_counts),
        "page_count": page_count,
        "empty_text_page_count": empty_text_page_count,
        "empty_text_page_fraction": (
            float(empty_text_page_count) / float(page_count) if page_count > 0 else None
        ),
        "mean_char_count": float(fmean(char_counts)) if char_counts else None,
        "mean_token_count": float(fmean(token_counts)) if token_counts else None,
        "max_docs": int(args.max_docs),
        "max_pages_per_doc": int(args.max_pages_per_doc),
        "missing_pdf_doc_ids": missing_pdf_doc_ids,
        "exported_doc_page_counts": exported_doc_page_counts,
    }
    output_summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"saved_jsonl: {output_jsonl}")
    print(f"saved_summary: {output_summary_json}")
    print(f"requested_doc_count: {len(doc_ids)}")
    print(f"exported_doc_count: {len(exported_doc_page_counts)}")
    print(f"page_count: {page_count}")
    print(f"empty_text_page_count: {empty_text_page_count}")
    print(f"mean_char_count: {summary['mean_char_count']}")
    print(f"mean_token_count: {summary['mean_token_count']}")


if __name__ == "__main__":
    main()
