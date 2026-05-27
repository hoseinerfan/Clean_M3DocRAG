#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

TRAILING_FRAGMENT_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "his",
    "in",
    "into",
    "is",
    "of",
    "or",
    "our",
    "that",
    "the",
    "their",
    "to",
    "was",
    "were",
    "when",
    "where",
    "which",
    "who",
    "with",
}

PAGE_FURNITURE_PATTERNS = [
    re.compile(r"^www\.", re.IGNORECASE),
    re.compile(r"^https?://", re.IGNORECASE),
    re.compile(r"^page\s+\d+$", re.IGNORECASE),
    re.compile(r"^\d+\s*$"),
    re.compile(r"^pew research center$", re.IGNORECASE),
    re.compile(r"^toyota motor corporation integrated report$", re.IGNORECASE),
]

CODE_LIKE_LINE_PATTERNS = [
    re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_.]*\s*=\s*\S"),
    re.compile(r"^\s*(?:class|def|from|import)\s+\S"),
    re.compile(r"^\s*(?:if|elif|else|for|while|try|except|with|return)\b.*:?$"),
    re.compile(r"^\s*[\])}]+[,]?\s*$"),
]
CODE_COMMENT_LINE_RE = re.compile(r"^\s*#\s+\S")
CODE_FENCE_LINE_RE = re.compile(r"^\s*```")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create controlled doc_pages JSONL variants from PDF-derived markdown. "
            "The variants keep only selected markdown heading lines so heading-breadcrumb "
            "graph ablations can isolate PDF outline headings, heuristic headings, and "
            "noise-reduced heuristic headings."
        )
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--variant",
        action="append",
        choices=[
            "outline_only",
            "heuristic_only",
            "heading_only",
            "strict_heuristic_only",
            "strict_heading",
            "strict_heading_codeguard",
        ],
        help="Variant to write. Repeatable. Defaults to all variants.",
    )
    parser.add_argument("--output-prefix", default="doc_pages_dev_pdf_markdown")
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--strict-max-words", type=int, default=12)
    parser.add_argument("--strict-max-chars", type=int, default=120)
    parser.add_argument(
        "--strict-min-alpha-chars",
        type=int,
        default=3,
        help="Minimum alphabetic characters for retained heuristic heading labels.",
    )
    parser.add_argument(
        "--codeguard-min-code-lines",
        type=int,
        default=4,
        help="Minimum number of code-like body lines required to suppress heuristic headings.",
    )
    parser.add_argument(
        "--codeguard-min-code-line-ratio",
        type=float,
        default=0.10,
        help="Minimum fraction of nonempty body lines that must look like code.",
    )
    return parser.parse_args()


def clean_label(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+#+\s*$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -:\t")


def heading_lines(markdown: str) -> list[tuple[str, str]]:
    lines: list[tuple[str, str]] = []
    for line in str(markdown or "").splitlines():
        match = MARKDOWN_HEADING_RE.match(line)
        if not match:
            continue
        prefix = match.group(1)
        label = clean_label(match.group(2))
        if label:
            lines.append((prefix, label))
    return lines


def split_pdf_heading_lines(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    lines = heading_lines(str(row.get("markdown", "") or ""))
    outline_count = max(0, int(row.get("pdf_outline_heading_count", 0) or 0))
    heuristic_count = max(0, int(row.get("pdf_heuristic_heading_count", 0) or 0))
    outline_lines = [f"{prefix} {label}" for prefix, label in lines[:outline_count]]
    heuristic_lines = [
        f"{prefix} {label}"
        for prefix, label in lines[outline_count : outline_count + heuristic_count]
    ]
    return outline_lines, heuristic_lines


def strict_reject_reason(label: str, *, max_words: int, max_chars: int, min_alpha_chars: int) -> str | None:
    text = clean_label(label)
    if not text:
        return "empty"
    if len(text) > max_chars:
        return "too_long_chars"
    alpha_chars = re.findall(r"[A-Za-z]", text)
    if len(alpha_chars) < min_alpha_chars:
        return "too_few_alpha_chars"
    for pattern in PAGE_FURNITURE_PATTERNS:
        if pattern.search(text):
            return "page_furniture"
    tokens = TOKEN_RE.findall(text)
    if not tokens:
        return "no_tokens"
    if len(tokens) > max_words:
        return "too_many_words"
    first = tokens[0]
    last = tokens[-1].lower()
    if last in TRAILING_FRAGMENT_WORDS:
        return "trailing_fragment_word"
    starts_lower = bool(first[:1].islower())
    alpha_tokens = [token for token in tokens if re.search(r"[A-Za-z]", token)]
    titleish_tokens = sum(1 for token in alpha_tokens if token[:1].isupper() or token.isupper())
    titleish_ratio = titleish_tokens / float(len(alpha_tokens) or 1)
    if starts_lower and titleish_ratio < 0.6:
        return "starts_lower_sentence_fragment"
    if text.count(",") >= 2 and titleish_ratio < 0.6:
        return "comma_heavy_sentence"
    if re.search(r"[.!?]\s+[A-Z]", text) and titleish_ratio < 0.8:
        return "multi_sentence_fragment"
    return None


def filter_heuristic_lines(
    lines: list[str],
    *,
    max_words: int,
    max_chars: int,
    min_alpha_chars: int,
) -> tuple[list[str], Counter[str]]:
    kept: list[str] = []
    reasons: Counter[str] = Counter()
    seen_keys: set[str] = set()
    for line in lines:
        match = MARKDOWN_HEADING_RE.match(line)
        label = clean_label(match.group(2) if match else line)
        reason = strict_reject_reason(
            label,
            max_words=max_words,
            max_chars=max_chars,
            min_alpha_chars=min_alpha_chars,
        )
        if reason is not None:
            reasons[reason] += 1
            continue
        key = re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()
        if not key:
            reasons["empty_key"] += 1
            continue
        if key in seen_keys:
            reasons["duplicate_page_heading"] += 1
            continue
        seen_keys.add(key)
        kept.append(line)
    return kept, reasons


def body_lines_for_codeguard(row: dict[str, Any]) -> list[str]:
    lines = str(row.get("markdown", "") or "").splitlines()
    prefix_heading_count = max(0, int(row.get("pdf_outline_heading_count", 0) or 0)) + max(
        0, int(row.get("pdf_heuristic_heading_count", 0) or 0)
    )
    if prefix_heading_count <= 0:
        return lines
    remaining = prefix_heading_count
    index = 0
    while index < len(lines) and remaining > 0:
        if MARKDOWN_HEADING_RE.match(lines[index]):
            remaining -= 1
        index += 1
    return lines[index:] if remaining == 0 else lines


def codeguard_page_detail(
    row: dict[str, Any],
    *,
    min_code_lines: int,
    min_code_line_ratio: float,
) -> dict[str, Any]:
    source = str(row.get("markdown_source", "") or "")
    backend = str(row.get("markdown_backend", "") or "")
    is_native = backend in {"", "native"} and not source.startswith("pymupdf4llm")
    body_lines = body_lines_for_codeguard(row) if is_native else []
    nonempty_lines = [line for line in body_lines if line.strip()]
    code_line_count = sum(
        any(pattern.search(line) for pattern in CODE_LIKE_LINE_PATTERNS)
        for line in nonempty_lines
    )
    code_comment_line_count = sum(
        bool(CODE_COMMENT_LINE_RE.search(line)) for line in nonempty_lines
    )
    fenced_line_count = sum(bool(CODE_FENCE_LINE_RE.search(line)) for line in nonempty_lines)
    code_line_ratio = code_line_count / float(len(nonempty_lines) or 1)
    code_dense = bool(
        is_native
        and (
            fenced_line_count >= 2
            or (
                code_line_count >= max(1, int(min_code_lines))
                and code_line_ratio >= float(min_code_line_ratio)
                and (
                    code_comment_line_count >= 2
                    or code_line_count >= max(1, int(min_code_lines)) * 2
                )
            )
        )
    )
    return {
        "code_dense": code_dense,
        "body_nonempty_line_count": len(nonempty_lines),
        "code_line_count": code_line_count,
        "code_comment_line_count": code_comment_line_count,
        "fenced_line_count": fenced_line_count,
        "code_line_ratio": code_line_ratio,
    }


def variant_markdown(
    variant: str,
    outline_lines: list[str],
    heuristic_lines: list[str],
    strict_heuristic_lines: list[str],
    codeguard_heuristic_lines: list[str],
) -> str:
    if variant == "outline_only":
        return "\n".join(outline_lines).strip()
    if variant == "heuristic_only":
        return "\n".join(heuristic_lines).strip()
    if variant == "heading_only":
        return "\n".join([*outline_lines, *heuristic_lines]).strip()
    if variant == "strict_heuristic_only":
        return "\n".join(strict_heuristic_lines).strip()
    if variant == "strict_heading":
        return "\n".join([*outline_lines, *strict_heuristic_lines]).strip()
    if variant == "strict_heading_codeguard":
        return "\n".join([*outline_lines, *codeguard_heuristic_lines]).strip()
    raise ValueError(f"Unknown variant: {variant}")


def main() -> None:
    args = parse_args()
    variants = args.variant or [
        "outline_only",
        "heuristic_only",
        "heading_only",
        "strict_heuristic_only",
        "strict_heading",
        "strict_heading_codeguard",
    ]
    input_jsonl = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_json) if args.summary_json else output_dir / "pdf_markdown_variants.summary.json"

    output_paths = {
        variant: output_dir / f"{args.output_prefix}.{variant}.jsonl" for variant in variants
    }
    handles = {
        variant: output_paths[variant].open("w", encoding="utf-8") for variant in variants
    }

    summary: dict[str, Any] = {
        "input_jsonl": str(input_jsonl),
        "variants": {variant: {"output_jsonl": str(path)} for variant, path in output_paths.items()},
        "strict_max_words": int(args.strict_max_words),
        "strict_max_chars": int(args.strict_max_chars),
        "strict_min_alpha_chars": int(args.strict_min_alpha_chars),
        "codeguard_min_code_lines": int(args.codeguard_min_code_lines),
        "codeguard_min_code_line_ratio": float(args.codeguard_min_code_line_ratio),
    }
    variant_counts = {
        variant: Counter() for variant in variants
    }
    source_counts: Counter[str] = Counter()
    strict_reject_counts: Counter[str] = Counter()
    page_count = 0
    raw_outline_line_count = 0
    raw_heuristic_line_count = 0
    strict_heuristic_line_count = 0
    codeguard_heuristic_line_count = 0
    codeguard_code_dense_page_count = 0
    codeguard_suppressed_heuristic_line_count = 0

    try:
        with input_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                page_count += 1
                row = json.loads(line)
                source_counts[str(row.get("markdown_source", "") or "UNKNOWN")] += 1
                outline_lines, heuristic_lines = split_pdf_heading_lines(row)
                strict_lines, reject_reasons = filter_heuristic_lines(
                    heuristic_lines,
                    max_words=int(args.strict_max_words),
                    max_chars=int(args.strict_max_chars),
                    min_alpha_chars=int(args.strict_min_alpha_chars),
                )
                strict_reject_counts.update(reject_reasons)
                codeguard_detail = codeguard_page_detail(
                    row,
                    min_code_lines=int(args.codeguard_min_code_lines),
                    min_code_line_ratio=float(args.codeguard_min_code_line_ratio),
                )
                codeguard_lines = [] if codeguard_detail["code_dense"] else strict_lines
                raw_outline_line_count += len(outline_lines)
                raw_heuristic_line_count += len(heuristic_lines)
                strict_heuristic_line_count += len(strict_lines)
                codeguard_heuristic_line_count += len(codeguard_lines)
                if codeguard_detail["code_dense"]:
                    codeguard_code_dense_page_count += 1
                    codeguard_suppressed_heuristic_line_count += len(strict_lines)

                for variant in variants:
                    output_row = dict(row)
                    output_row["markdown"] = variant_markdown(
                        variant,
                        outline_lines,
                        heuristic_lines,
                        strict_lines,
                        codeguard_lines,
                    )
                    output_row["markdown_variant"] = variant
                    output_row["markdown_original_source"] = row.get("markdown_source", "")
                    output_row["markdown_source"] = f"variant_{variant}"
                    output_row["pdf_variant_outline_heading_count"] = len(outline_lines)
                    if variant == "strict_heading_codeguard":
                        variant_heuristic_line_count = len(codeguard_lines)
                    elif variant in {"strict_heuristic_only", "strict_heading"}:
                        variant_heuristic_line_count = len(strict_lines)
                    else:
                        variant_heuristic_line_count = len(heuristic_lines)
                    output_row["pdf_variant_heuristic_heading_count"] = (
                        variant_heuristic_line_count
                    )
                    if variant == "strict_heading_codeguard":
                        output_row["pdf_variant_codeguard"] = codeguard_detail
                    if output_row["markdown"].strip():
                        variant_counts[variant]["nonempty_page_count"] += 1
                    if outline_lines and variant in {
                        "outline_only",
                        "heading_only",
                        "strict_heading",
                        "strict_heading_codeguard",
                    }:
                        variant_counts[variant]["outline_page_count"] += 1
                    if output_row["pdf_variant_heuristic_heading_count"] > 0 and variant in {
                        "heuristic_only",
                        "heading_only",
                        "strict_heuristic_only",
                        "strict_heading",
                        "strict_heading_codeguard",
                    }:
                        variant_counts[variant]["heuristic_page_count"] += 1
                    handles[variant].write(json.dumps(output_row, ensure_ascii=False) + "\n")
    finally:
        for handle in handles.values():
            handle.close()

    summary.update(
        {
            "page_count": page_count,
            "source_counts": dict(sorted(source_counts.items())),
            "raw_outline_heading_line_count": raw_outline_line_count,
            "raw_heuristic_heading_line_count": raw_heuristic_line_count,
            "strict_heuristic_heading_line_count": strict_heuristic_line_count,
            "strict_reject_counts": dict(sorted(strict_reject_counts.items())),
            "codeguard_heuristic_heading_line_count": codeguard_heuristic_line_count,
            "codeguard_code_dense_page_count": codeguard_code_dense_page_count,
            "codeguard_suppressed_heuristic_heading_line_count": (
                codeguard_suppressed_heuristic_line_count
            ),
        }
    )
    for variant in variants:
        summary["variants"][variant].update(dict(sorted(variant_counts[variant].items())))

    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"saved_summary: {summary_path}")
    for variant, path in output_paths.items():
        print(f"saved_{variant}: {path}")
    print(f"page_count: {page_count}")
    print(f"raw_outline_heading_line_count: {raw_outline_line_count}")
    print(f"raw_heuristic_heading_line_count: {raw_heuristic_line_count}")
    print(f"strict_heuristic_heading_line_count: {strict_heuristic_line_count}")
    print(f"codeguard_heuristic_heading_line_count: {codeguard_heuristic_line_count}")
    print(f"codeguard_code_dense_page_count: {codeguard_code_dense_page_count}")
    print(
        "codeguard_suppressed_heuristic_heading_line_count: "
        f"{codeguard_suppressed_heuristic_line_count}"
    )


if __name__ == "__main__":
    main()
