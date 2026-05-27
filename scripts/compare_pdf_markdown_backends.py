#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")
NON_WORD_RE = re.compile(r"[^a-z0-9]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare native and PyMuPDF4LLM PDF-Markdown outputs without assuming "
            "ground-truth Markdown labels. Reports extraction coverage, paired "
            "heading disagreement samples, and optional downstream retrieval metrics."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--native-jsonl", required=True)
    parser.add_argument("--alternate-jsonl", required=True)
    parser.add_argument("--native-summary", default="")
    parser.add_argument("--alternate-summary", default="")
    parser.add_argument("--native-variant-summary", default="")
    parser.add_argument("--alternate-variant-summary", default="")
    parser.add_argument("--native-full-summary", default="")
    parser.add_argument("--alternate-full-summary", default="")
    parser.add_argument("--native-strict-summary", default="")
    parser.add_argument("--alternate-strict-summary", default="")
    parser.add_argument("--native-safe-summary", default="")
    parser.add_argument("--alternate-safe-summary", default="")
    parser.add_argument("--sample-limit", type=int, default=5)
    parser.add_argument("--excerpt-chars", type=int, default=900)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def read_json(path: str) -> dict[str, Any]:
    if not path:
        return {}
    parsed = Path(path)
    if not parsed.is_file():
        return {}
    with parsed.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def page_uid(row: dict[str, Any]) -> str:
    uid = str(row.get("page_uid", "") or "").strip()
    if uid:
        return uid
    doc_id = str(row.get("doc_id", "") or row.get("doc_name", "") or "").strip()
    page_idx = row.get("page_idx", row.get("page_id", row.get("page")))
    if not doc_id or page_idx is None:
        return ""
    return f"{doc_id}_page{int(page_idx)}"


def read_pages(path: str) -> dict[str, dict[str, Any]]:
    pages: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            uid = page_uid(row)
            if uid:
                pages[uid] = row
    return pages


def clean_heading(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text.strip(" -_:;.,\t")


def heading_labels(row: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for line in str(row.get("markdown", "") or "").splitlines():
        match = MARKDOWN_HEADING_RE.match(line)
        if match:
            label = clean_heading(match.group(1))
            if label:
                labels.append(label)
    return labels


def normalized_heading_set(row: dict[str, Any]) -> set[str]:
    return {
        NON_WORD_RE.sub(" ", label.lower()).strip()
        for label in heading_labels(row)
        if NON_WORD_RE.sub(" ", label.lower()).strip()
    }


def markdown_text(row: dict[str, Any]) -> str:
    return str(row.get("markdown", "") or "").strip()


def extraction_stats(pages: dict[str, dict[str, Any]]) -> dict[str, Any]:
    heading_counts = [len(heading_labels(row)) for row in pages.values()]
    markdown_lengths = [len(markdown_text(row)) for row in pages.values()]
    source_counts = Counter(str(row.get("markdown_source", "") or "UNKNOWN") for row in pages.values())
    return {
        "page_count": len(pages),
        "nonempty_page_count": sum(bool(markdown_text(row)) for row in pages.values()),
        "heading_page_count": sum(bool(heading_labels(row)) for row in pages.values()),
        "heading_line_count": sum(heading_counts),
        "mean_heading_lines_per_page": mean(heading_counts) if heading_counts else 0.0,
        "mean_markdown_chars_per_page": mean(markdown_lengths) if markdown_lengths else 0.0,
        "source_counts": dict(sorted(source_counts.items())),
    }


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / float(len(union)) if union else 1.0


def paired_stats(
    native: dict[str, dict[str, Any]],
    alternate: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    native_uids = set(native)
    alternate_uids = set(alternate)
    aligned = sorted(native_uids & alternate_uids)
    patterns: Counter[str] = Counter()
    jaccards: list[float] = []
    jaccards_with_any_heading: list[float] = []
    identical_heading_sets = 0
    native_only_heading_pages: list[tuple[int, str]] = []
    alternate_only_heading_pages: list[tuple[int, str]] = []
    disagreeing_pages: list[tuple[float, int, str]] = []

    for uid in aligned:
        native_set = normalized_heading_set(native[uid])
        alternate_set = normalized_heading_set(alternate[uid])
        if native_set and alternate_set:
            patterns["both"] += 1
        elif native_set:
            patterns["native_only"] += 1
            native_only_heading_pages.append((len(native_set), uid))
        elif alternate_set:
            patterns["alternate_only"] += 1
            alternate_only_heading_pages.append((len(alternate_set), uid))
        else:
            patterns["neither"] += 1
        score = jaccard(native_set, alternate_set)
        jaccards.append(score)
        if native_set or alternate_set:
            jaccards_with_any_heading.append(score)
        if native_set == alternate_set:
            identical_heading_sets += 1
        elif native_set and alternate_set:
            disagreeing_pages.append((score, -(len(native_set | alternate_set)), uid))

    native_only_heading_pages.sort(key=lambda item: (-item[0], item[1]))
    alternate_only_heading_pages.sort(key=lambda item: (-item[0], item[1]))
    disagreeing_pages.sort()
    stats = {
        "aligned_page_count": len(aligned),
        "native_missing_page_count": len(alternate_uids - native_uids),
        "alternate_missing_page_count": len(native_uids - alternate_uids),
        "heading_presence_counts": dict(patterns),
        "identical_heading_set_count": identical_heading_sets,
        "mean_heading_jaccard_all_pages": mean(jaccards) if jaccards else 0.0,
        "mean_heading_jaccard_pages_with_heading": (
            mean(jaccards_with_any_heading) if jaccards_with_any_heading else 0.0
        ),
    }
    samples = {
        "native_only": [uid for _, uid in native_only_heading_pages],
        "alternate_only": [uid for _, uid in alternate_only_heading_pages],
        "different_both": [uid for _, _, uid in disagreeing_pages],
    }
    return stats, samples


def first_metric(summary: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = summary.get(key)
        if value is not None:
            return value
    return None


def downstream_stats(path: str) -> dict[str, Any]:
    summary = read_json(path)
    if not summary:
        return {}
    rejected = summary.get("rejected_promotion_reason_counts") or {}
    return {
        "page_hit_at_4": first_metric(
            summary,
            "page_hit_at_k_count",
            "page_hit_at_4_count",
            "reranked_top4_page_count",
        ),
        "doc_hit_at_4": first_metric(
            summary,
            "doc_hit_at_k_count",
            "doc_hit_at_4_count",
            "reranked_top4_doc_count",
        ),
        "accepted": summary.get("accepted_count"),
        "base_page_hit_at_4": first_metric(
            summary,
            "base_page_hit_at_k_count",
            "base_page_hit_at_4_count",
        ),
        "candidate_page_hit_at_4": first_metric(
            summary,
            "candidate_page_hit_at_k_count",
            "candidate_page_hit_at_4_count",
        ),
        "recovered": summary.get("recovered"),
        "lost": summary.get("lost"),
        "net": summary.get("net_recovered"),
        "body_rejects": rejected.get("promoted_body_score_not_above_base"),
    }


def display(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def delta(native: Any, alternate: Any) -> str:
    if not isinstance(native, (int, float)) or not isinstance(alternate, (int, float)):
        return "NA"
    value = alternate - native
    return f"{value:+g}"


def excerpt(row: dict[str, Any] | None, max_chars: int) -> str:
    if row is None:
        return "[MISSING]"
    text = markdown_text(row)
    if not text:
        return "[EMPTY MARKDOWN]"
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars].rstrip() + "\n[...]"
    return text


def extraction_summary_fields(summary: dict[str, Any], computed: dict[str, Any]) -> dict[str, Any]:
    return {
        "backend": summary.get("backend"),
        "version": summary.get("pymupdf4llm_version"),
        "pages": first_metric(summary, "page_count") or computed["page_count"],
        "nonempty_pages": first_metric(summary, "nonempty_markdown_page_count") or computed["nonempty_page_count"],
        "heading_pages": first_metric(summary, "heading_page_count") or computed["heading_page_count"],
        "heading_lines": computed["heading_line_count"],
        "mean_chars": computed["mean_markdown_chars_per_page"],
        "backend_errors": summary.get("backend_error_doc_count"),
        "unmatched_docs": summary.get("unmatched_doc_count"),
    }


def add_variant_fields(extraction: dict[str, Any], variants: dict[str, Any]) -> None:
    extraction["raw_outline_lines"] = variants.get("raw_outline_heading_line_count")
    extraction["raw_heuristic_lines"] = variants.get("raw_heuristic_heading_line_count")
    extraction["strict_heuristic_lines"] = variants.get("strict_heuristic_heading_line_count")


def render_samples(
    lines: list[str],
    title: str,
    uids: list[str],
    native: dict[str, dict[str, Any]],
    alternate: dict[str, dict[str, Any]],
    limit: int,
    excerpt_chars: int,
) -> None:
    lines.extend([f"## {title}", ""])
    if not uids:
        lines.extend(["None.", ""])
        return
    for uid in uids[: max(0, limit)]:
        native_labels = heading_labels(native.get(uid, {}))
        alternate_labels = heading_labels(alternate.get(uid, {}))
        lines.extend(
            [
                f"### {uid}",
                "",
                f"- Native headings: {json.dumps(native_labels, ensure_ascii=False)}",
                f"- PyMuPDF4LLM headings: {json.dumps(alternate_labels, ensure_ascii=False)}",
                "",
                "Native:",
                "",
                "````markdown",
                excerpt(native.get(uid), excerpt_chars),
                "````",
                "",
                "PyMuPDF4LLM:",
                "",
                "````markdown",
                excerpt(alternate.get(uid), excerpt_chars),
                "````",
                "",
            ]
        )


def main() -> None:
    args = parse_args()
    native_pages = read_pages(args.native_jsonl)
    alternate_pages = read_pages(args.alternate_jsonl)
    native_computed = extraction_stats(native_pages)
    alternate_computed = extraction_stats(alternate_pages)
    native_summary = extraction_summary_fields(read_json(args.native_summary), native_computed)
    alternate_summary = extraction_summary_fields(read_json(args.alternate_summary), alternate_computed)
    native_variants = read_json(args.native_variant_summary)
    alternate_variants = read_json(args.alternate_variant_summary)
    add_variant_fields(native_summary, native_variants)
    add_variant_fields(alternate_summary, alternate_variants)
    paired, samples = paired_stats(native_pages, alternate_pages)

    downstream = {
        "full_direct": {
            "native": downstream_stats(args.native_full_summary),
            "alternate": downstream_stats(args.alternate_full_summary),
        },
        "strict_direct": {
            "native": downstream_stats(args.native_strict_summary),
            "alternate": downstream_stats(args.alternate_strict_summary),
        },
        "safe_gate": {
            "native": downstream_stats(args.native_safe_summary),
            "alternate": downstream_stats(args.alternate_safe_summary),
        },
    }
    result = {
        "dataset": args.dataset,
        "native_jsonl": args.native_jsonl,
        "alternate_jsonl": args.alternate_jsonl,
        "native_extraction": native_summary,
        "alternate_extraction": alternate_summary,
        "native_variant_summary": native_variants,
        "alternate_variant_summary": alternate_variants,
        "paired": paired,
        "downstream": downstream,
        "samples": {key: values[: max(0, args.sample_limit)] for key, values in samples.items()},
    }

    lines = [
        f"# PDF Markdown Backend Comparison: {args.dataset}",
        "",
        "This report compares extraction coverage and downstream retrieval utility. "
        "There are no gold Markdown or gold heading annotations, so coverage and "
        "heading disagreement are audit signals rather than standalone quality labels.",
        "",
        "## Extraction Coverage",
        "",
        "| backend | pages | nonempty pages | heading pages | raw heuristic heading lines | strict heuristic heading lines | mean markdown chars/page | backend error docs | unmatched docs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| native | {display(native_summary['pages'])} | {display(native_summary['nonempty_pages'])} | "
            f"{display(native_summary['heading_pages'])} | {display(native_summary['raw_heuristic_lines'])} | "
            f"{display(native_summary['strict_heuristic_lines'])} | {display(native_summary['mean_chars'])} | "
            f"{display(native_summary['backend_errors'])} | {display(native_summary['unmatched_docs'])} |"
        ),
        (
            f"| pymupdf4llm | {display(alternate_summary['pages'])} | "
            f"{display(alternate_summary['nonempty_pages'])} | {display(alternate_summary['heading_pages'])} | "
            f"{display(alternate_summary['raw_heuristic_lines'])} | "
            f"{display(alternate_summary['strict_heuristic_lines'])} | {display(alternate_summary['mean_chars'])} | "
            f"{display(alternate_summary['backend_errors'])} | {display(alternate_summary['unmatched_docs'])} |"
        ),
        "",
        "## Paired Heading Agreement",
        "",
        f"- Aligned pages: {paired['aligned_page_count']}",
        f"- Heading pages in both: {(paired['heading_presence_counts']).get('both', 0)}",
        f"- Native-only heading pages: {(paired['heading_presence_counts']).get('native_only', 0)}",
        f"- PyMuPDF4LLM-only heading pages: {(paired['heading_presence_counts']).get('alternate_only', 0)}",
        f"- Pages with identical normalized heading sets: {paired['identical_heading_set_count']}",
        f"- Mean heading-set Jaccard on pages with any heading: {paired['mean_heading_jaccard_pages_with_heading']:.4f}",
        "",
        "## Downstream Utility",
        "",
        "| output | native page hit@4 | pymupdf4llm page hit@4 | delta | native accepted | pymupdf4llm accepted | native net | pymupdf4llm net |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("full_direct", "strict_direct", "safe_gate"):
        native_run = downstream[label]["native"]
        alternate_run = downstream[label]["alternate"]
        native_hit = native_run.get("page_hit_at_4")
        alternate_hit = alternate_run.get("page_hit_at_4")
        lines.append(
            f"| {label} | {display(native_hit)} | {display(alternate_hit)} | "
            f"{delta(native_hit, alternate_hit)} | {display(native_run.get('accepted'))} | "
            f"{display(alternate_run.get('accepted'))} | {display(native_run.get('net'))} | "
            f"{display(alternate_run.get('net'))} |"
        )
    lines.extend(
        [
            "",
            "Interpret the final gate result first; direct heading views diagnose the "
            "quality of the auxiliary signal but are not the delivered method.",
            "",
        ]
    )
    render_samples(
        lines,
        "Native-Only Heading Samples",
        samples["native_only"],
        native_pages,
        alternate_pages,
        args.sample_limit,
        args.excerpt_chars,
    )
    render_samples(
        lines,
        "PyMuPDF4LLM-Only Heading Samples",
        samples["alternate_only"],
        native_pages,
        alternate_pages,
        args.sample_limit,
        args.excerpt_chars,
    )
    render_samples(
        lines,
        "Different Heading Samples",
        samples["different_both"],
        native_pages,
        alternate_pages,
        args.sample_limit,
        args.excerpt_chars,
    )

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"saved_md: {output_md}")
    if args.output_json:
        print(f"saved_json: {args.output_json}")
    print(
        {
            "dataset": args.dataset,
            "native_heading_pages": native_summary["heading_pages"],
            "alternate_heading_pages": alternate_summary["heading_pages"],
            "native_only_heading_pages": paired["heading_presence_counts"].get("native_only", 0),
            "alternate_only_heading_pages": paired["heading_presence_counts"].get("alternate_only", 0),
            "safe_native_page_at_4": downstream["safe_gate"]["native"].get("page_hit_at_4"),
            "safe_alternate_page_at_4": downstream["safe_gate"]["alternate"].get("page_hit_at_4"),
        }
    )


if __name__ == "__main__":
    main()
