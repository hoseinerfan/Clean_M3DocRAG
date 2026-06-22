#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit exact and fuzzy match contributions in pseudo-page labels produced "
            "by scripts/build_mmqa_pseudo_page_labels.py."
        )
    )
    parser.add_argument("--pseudo-labels-jsonl", required=True)
    parser.add_argument(
        "--summary-json",
        default="",
        help=(
            "Optional summary JSON from the same pseudo-label run. If omitted, the "
            "script tries <pseudo-labels-jsonl stem>.summary.json."
        ),
    )
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument(
        "--qid",
        action="append",
        default=[],
        help="Optional qid filter. Can be passed multiple times.",
    )
    parser.add_argument("--example-limit", type=int, default=50)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-jsonl", default="")
    parser.add_argument("--output-csv", default="")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def infer_summary_path(labels_path: Path) -> Path:
    return labels_path.with_suffix(".summary.json")


def load_min_score(labels_path: Path, summary_json: str, explicit_min_score: float | None) -> float | None:
    if explicit_min_score is not None:
        return float(explicit_min_score)

    candidates: list[Path] = []
    if summary_json:
        candidates.append(Path(summary_json))
    candidates.append(infer_summary_path(labels_path))

    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        value = data.get("min_score")
        if value is not None:
            return float(value)
    return None


def pct(count: int | float, total: int | float) -> str:
    if not total:
        return "0.00%"
    return f"{100.0 * float(count) / float(total):.2f}%"


def compact_text(value: Any, limit: int = 110) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def as_pages(row: dict[str, Any], key: str) -> list[dict[str, Any]]:
    pages = row.get(key, [])
    return [page for page in pages if isinstance(page, dict)] if isinstance(pages, list) else []


def match_score(matches: list[dict[str, Any]]) -> float:
    total = 0.0
    for match in matches:
        if not isinstance(match, dict):
            continue
        try:
            total += float(match.get("weight", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
    return total


def match_sources(matches: list[dict[str, Any]]) -> Counter[str]:
    out: Counter[str] = Counter()
    for match in matches:
        if isinstance(match, dict) and match.get("source"):
            out[str(match["source"])] += 1
    return out


def overlap_values(matches: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for match in matches:
        if not isinstance(match, dict) or match.get("overlap") is None:
            continue
        try:
            values.append(float(match["overlap"]))
        except (TypeError, ValueError):
            continue
    return values


def page_score(page: dict[str, Any]) -> float:
    try:
        return float(page.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def page_uid(page: dict[str, Any]) -> str:
    return str(page.get("page_uid", "") or "")


def page_contribution_record(page: dict[str, Any], min_score: float | None) -> dict[str, Any]:
    exact = [match for match in page.get("exact_matches", []) or [] if isinstance(match, dict)]
    fuzzy = [match for match in page.get("fuzzy_matches", []) or [] if isinstance(match, dict)]
    exact_total = match_score(exact)
    fuzzy_total = match_score(fuzzy)
    total = exact_total + fuzzy_total
    recorded_score = page_score(page)
    denominator = recorded_score if recorded_score > 0 else total
    threshold = float(min_score) if min_score is not None else None
    overlaps = overlap_values(fuzzy)

    return {
        "page_uid": page_uid(page),
        "doc_id": str(page.get("doc_id", "") or ""),
        "page_idx": page.get("page_idx"),
        "recorded_score": round(recorded_score, 6),
        "recomputed_score": round(total, 6),
        "score_delta": round(recorded_score - total, 6),
        "confidence": str(page.get("confidence", "") or ""),
        "exact_score": round(exact_total, 6),
        "fuzzy_score": round(fuzzy_total, 6),
        "fuzzy_fraction": round(fuzzy_total / denominator, 6) if denominator else 0.0,
        "exact_match_count": len(exact),
        "fuzzy_match_count": len(fuzzy),
        "has_fuzzy": fuzzy_total > 0.0,
        "exact_only_passes_threshold": bool(threshold is not None and exact_total >= threshold),
        "fuzzy_needed_for_threshold": bool(
            threshold is not None and exact_total < threshold <= recorded_score and fuzzy_total > 0.0
        ),
        "max_fuzzy_overlap": round(max(overlaps), 6) if overlaps else 0.0,
        "mean_fuzzy_overlap": round(float(fmean(overlaps)), 6) if overlaps else 0.0,
        "exact_sources": dict(match_sources(exact).most_common()),
        "fuzzy_sources": dict(match_sources(fuzzy).most_common()),
        "exact_matches": exact,
        "fuzzy_matches": fuzzy,
    }


def qid_record(row: dict[str, Any], min_score: float | None) -> dict[str, Any]:
    selected = [page_contribution_record(page, min_score) for page in as_pages(row, "pseudo_gold_pages")]
    top_scored = [page_contribution_record(page, min_score) for page in as_pages(row, "top_scored_pages")]
    selected_exact = sum(float(page["exact_score"]) for page in selected)
    selected_fuzzy = sum(float(page["fuzzy_score"]) for page in selected)
    qid_total = selected_exact + selected_fuzzy

    return {
        "qid": str(row.get("qid", "") or ""),
        "question": str(row.get("question", "") or ""),
        "question_type": str(row.get("question_type", "") or ""),
        "status": str(row.get("status", "") or ""),
        "gold_doc_count": len(row.get("gold_doc_ids", []) or []),
        "selected_page_count": len(selected),
        "candidate_pages_scored": int(row.get("candidate_pages_scored", 0) or 0),
        "evidence_count": int(row.get("evidence_count", 0) or 0),
        "selected_exact_score_sum": round(selected_exact, 6),
        "selected_fuzzy_score_sum": round(selected_fuzzy, 6),
        "selected_fuzzy_fraction": round(selected_fuzzy / qid_total, 6) if qid_total else 0.0,
        "selected_pages_with_fuzzy": sum(int(page["has_fuzzy"]) for page in selected),
        "selected_pages_fuzzy_needed": sum(int(page["fuzzy_needed_for_threshold"]) for page in selected),
        "qid_has_selected_fuzzy": any(bool(page["has_fuzzy"]) for page in selected),
        "qid_has_fuzzy_needed": any(bool(page["fuzzy_needed_for_threshold"]) for page in selected),
        "qid_has_exact_only_threshold_page": any(
            bool(page["exact_only_passes_threshold"]) for page in selected
        ),
        "selected_pages": selected,
        "top_scored_pages": top_scored,
    }


def summarize(qid_rows: list[dict[str, Any]], min_score: float | None) -> dict[str, Any]:
    qid_count = len(qid_rows)
    labeled = [row for row in qid_rows if int(row["selected_page_count"]) > 0]
    selected_pages = [
        page
        for row in qid_rows
        for page in row.get("selected_pages", [])
        if isinstance(page, dict)
    ]

    exact_source_counts: Counter[str] = Counter()
    fuzzy_source_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    qtype_counts: Counter[str] = Counter()
    qtype_fuzzy_counts: Counter[str] = Counter()
    qtype_fuzzy_needed_counts: Counter[str] = Counter()

    exact_score_total = 0.0
    fuzzy_score_total = 0.0
    pages_with_fuzzy = 0
    pages_fuzzy_needed = 0
    exact_only_pass_pages = 0
    score_deltas: list[float] = []
    fuzzy_fractions: list[float] = []

    for row in qid_rows:
        qtype = str(row.get("question_type") or "UNKNOWN")
        qtype_counts[qtype] += 1
        if row.get("qid_has_selected_fuzzy"):
            qtype_fuzzy_counts[qtype] += 1
        if row.get("qid_has_fuzzy_needed"):
            qtype_fuzzy_needed_counts[qtype] += 1

    for page in selected_pages:
        exact_score_total += float(page["exact_score"])
        fuzzy_score_total += float(page["fuzzy_score"])
        pages_with_fuzzy += int(bool(page["has_fuzzy"]))
        pages_fuzzy_needed += int(bool(page["fuzzy_needed_for_threshold"]))
        exact_only_pass_pages += int(bool(page["exact_only_passes_threshold"]))
        confidence_counts[str(page.get("confidence") or "missing")] += 1
        score_deltas.append(abs(float(page.get("score_delta", 0.0) or 0.0)))
        fuzzy_fractions.append(float(page.get("fuzzy_fraction", 0.0) or 0.0))
        exact_source_counts.update(page.get("exact_sources", {}) or {})
        fuzzy_source_counts.update(page.get("fuzzy_sources", {}) or {})

    score_total = exact_score_total + fuzzy_score_total

    qtype_rows = []
    for qtype, count in qtype_counts.most_common():
        fuzzy_qids = qtype_fuzzy_counts.get(qtype, 0)
        fuzzy_needed_qids = qtype_fuzzy_needed_counts.get(qtype, 0)
        qtype_rows.append(
            {
                "question_type": qtype,
                "qids": int(count),
                "qids_with_selected_fuzzy": int(fuzzy_qids),
                "qids_with_fuzzy_needed": int(fuzzy_needed_qids),
                "selected_fuzzy_rate": round(fuzzy_qids / count, 6) if count else 0.0,
                "fuzzy_needed_rate": round(fuzzy_needed_qids / count, 6) if count else 0.0,
            }
        )

    fuzzy_needed_examples = []
    for row in qid_rows:
        needed_pages = [
            page for page in row["selected_pages"] if page.get("fuzzy_needed_for_threshold")
        ]
        if not needed_pages:
            continue
        best_page = max(needed_pages, key=lambda page: float(page.get("fuzzy_score", 0.0)))
        fuzzy_needed_examples.append(
            {
                "qid": row["qid"],
                "question_type": row["question_type"],
                "page_uid": best_page["page_uid"],
                "score": best_page["recorded_score"],
                "exact_score": best_page["exact_score"],
                "fuzzy_score": best_page["fuzzy_score"],
                "fuzzy_sources": best_page["fuzzy_sources"],
                "question": compact_text(row["question"]),
            }
        )
    fuzzy_needed_examples.sort(
        key=lambda row: (-float(row["fuzzy_score"]), str(row["qid"]), str(row["page_uid"]))
    )

    top_fuzzy_fraction_pages = sorted(
        (
            {
                "qid": row["qid"],
                "question_type": row["question_type"],
                "page_uid": page["page_uid"],
                "score": page["recorded_score"],
                "exact_score": page["exact_score"],
                "fuzzy_score": page["fuzzy_score"],
                "fuzzy_fraction": page["fuzzy_fraction"],
                "fuzzy_needed": page["fuzzy_needed_for_threshold"],
                "question": compact_text(row["question"]),
            }
            for row in qid_rows
            for page in row["selected_pages"]
            if float(page.get("fuzzy_score", 0.0) or 0.0) > 0.0
        ),
        key=lambda row: (
            -float(row["fuzzy_fraction"]),
            -float(row["fuzzy_score"]),
            str(row["qid"]),
        ),
    )

    return {
        "min_score": min_score,
        "qid_count": qid_count,
        "labeled_qids": len(labeled),
        "unlabeled_qids": qid_count - len(labeled),
        "selected_page_count": len(selected_pages),
        "selected_pages_with_fuzzy": pages_with_fuzzy,
        "selected_pages_fuzzy_needed_for_threshold": pages_fuzzy_needed,
        "selected_pages_exact_only_pass_threshold": exact_only_pass_pages,
        "qids_with_selected_fuzzy": sum(int(row["qid_has_selected_fuzzy"]) for row in qid_rows),
        "qids_with_fuzzy_needed": sum(int(row["qid_has_fuzzy_needed"]) for row in qid_rows),
        "qids_with_exact_only_threshold_page": sum(
            int(row["qid_has_exact_only_threshold_page"]) for row in qid_rows
        ),
        "exact_score_total": round(exact_score_total, 6),
        "fuzzy_score_total": round(fuzzy_score_total, 6),
        "fuzzy_score_fraction": round(fuzzy_score_total / score_total, 6) if score_total else 0.0,
        "mean_selected_page_fuzzy_fraction": round(float(fmean(fuzzy_fractions)), 6)
        if fuzzy_fractions
        else 0.0,
        "max_score_recompute_delta": round(max(score_deltas), 6) if score_deltas else 0.0,
        "confidence_counts": dict(confidence_counts.most_common()),
        "exact_source_counts": dict(exact_source_counts.most_common()),
        "fuzzy_source_counts": dict(fuzzy_source_counts.most_common()),
        "question_type_summary": qtype_rows,
        "fuzzy_needed_examples": fuzzy_needed_examples,
        "top_fuzzy_fraction_pages": top_fuzzy_fraction_pages,
    }


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def write_markdown(
    path: Path,
    report: dict[str, Any],
    labels_path: Path,
    qid_rows: list[dict[str, Any]],
    example_limit: int,
) -> None:
    page_total = int(report["selected_page_count"])
    qid_total = int(report["qid_count"])
    lines: list[str] = [
        "# Pseudo-Page Match Contribution Audit",
        "",
        f"- pseudo_labels_jsonl: `{labels_path}`",
        f"- min_score: `{report['min_score']}`",
        f"- qids: `{report['qid_count']}`",
        f"- labeled_qids: `{report['labeled_qids']}`",
        f"- selected_page_count: `{report['selected_page_count']}`",
        "",
        "## Fuzzy Contribution Summary",
        "",
    ]
    lines.extend(
        markdown_table(
            ["metric", "count/value", "rate"],
            [
                [
                    "selected pages with any fuzzy match",
                    report["selected_pages_with_fuzzy"],
                    pct(report["selected_pages_with_fuzzy"], page_total),
                ],
                [
                    "selected pages where fuzzy was needed to pass min_score",
                    report["selected_pages_fuzzy_needed_for_threshold"],
                    pct(report["selected_pages_fuzzy_needed_for_threshold"], page_total),
                ],
                [
                    "qids with any selected fuzzy match",
                    report["qids_with_selected_fuzzy"],
                    pct(report["qids_with_selected_fuzzy"], qid_total),
                ],
                [
                    "qids where fuzzy was needed for at least one selected page",
                    report["qids_with_fuzzy_needed"],
                    pct(report["qids_with_fuzzy_needed"], qid_total),
                ],
                [
                    "fuzzy score fraction across selected labels",
                    report["fuzzy_score_fraction"],
                    "",
                ],
                [
                    "mean selected-page fuzzy fraction",
                    report["mean_selected_page_fuzzy_fraction"],
                    "",
                ],
            ],
        )
    )

    lines.extend(["", "## Score Totals", ""])
    lines.extend(
        markdown_table(
            ["score type", "total"],
            [
                ["exact", report["exact_score_total"]],
                ["fuzzy", report["fuzzy_score_total"]],
            ],
        )
    )

    lines.extend(["", "## Confidence Counts", ""])
    lines.extend(
        markdown_table(
            ["confidence", "selected pages"],
            [[key, value] for key, value in report["confidence_counts"].items()],
        )
    )

    lines.extend(["", "## Exact Source Counts", ""])
    lines.extend(
        markdown_table(
            ["source", "matches"],
            [[key, value] for key, value in list(report["exact_source_counts"].items())[:20]],
        )
    )

    lines.extend(["", "## Fuzzy Source Counts", ""])
    lines.extend(
        markdown_table(
            ["source", "matches"],
            [[key, value] for key, value in list(report["fuzzy_source_counts"].items())[:20]],
        )
    )

    lines.extend(["", "## Question-Type Summary", ""])
    lines.extend(
        markdown_table(
            [
                "question_type",
                "qids",
                "qids_with_fuzzy",
                "qids_fuzzy_needed",
                "fuzzy_needed_rate",
            ],
            [
                [
                    row["question_type"],
                    row["qids"],
                    row["qids_with_selected_fuzzy"],
                    row["qids_with_fuzzy_needed"],
                    row["fuzzy_needed_rate"],
                ]
                for row in report["question_type_summary"][:30]
            ],
        )
    )

    lines.extend(["", "## Examples Where Fuzzy Was Needed", ""])
    lines.extend(
        markdown_table(
            [
                "qid",
                "question_type",
                "page_uid",
                "score",
                "exact",
                "fuzzy",
                "fuzzy_sources",
                "question",
            ],
            [
                [
                    row["qid"],
                    row["question_type"],
                    row["page_uid"],
                    row["score"],
                    row["exact_score"],
                    row["fuzzy_score"],
                    json.dumps(row["fuzzy_sources"], ensure_ascii=False),
                    row["question"],
                ]
                for row in report["fuzzy_needed_examples"][:example_limit]
            ],
        )
    )

    lines.extend(["", "## Highest Fuzzy-Fraction Selected Pages", ""])
    lines.extend(
        markdown_table(
            [
                "qid",
                "question_type",
                "page_uid",
                "score",
                "exact",
                "fuzzy",
                "fuzzy_fraction",
                "fuzzy_needed",
                "question",
            ],
            [
                [
                    row["qid"],
                    row["question_type"],
                    row["page_uid"],
                    row["score"],
                    row["exact_score"],
                    row["fuzzy_score"],
                    row["fuzzy_fraction"],
                    row["fuzzy_needed"],
                    row["question"],
                ]
                for row in report["top_fuzzy_fraction_pages"][:example_limit]
            ],
        )
    )

    if qid_total <= 25:
        lines.extend(["", "## QID Details", ""])
        lines.extend(
            markdown_table(
                [
                    "qid",
                    "question_type",
                    "selected_pages",
                    "pages_with_fuzzy",
                    "pages_fuzzy_needed",
                    "question",
                ],
                [
                    [
                        row["qid"],
                        row["question_type"],
                        row["selected_page_count"],
                        row["selected_pages_with_fuzzy"],
                        row["selected_pages_fuzzy_needed"],
                        compact_text(row["question"]),
                    ]
                    for row in qid_rows
                ],
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_qid_jsonl(path: Path, qid_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in qid_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_page_csv(path: Path, qid_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "qid",
        "question_type",
        "page_uid",
        "doc_id",
        "page_idx",
        "recorded_score",
        "exact_score",
        "fuzzy_score",
        "fuzzy_fraction",
        "exact_match_count",
        "fuzzy_match_count",
        "has_fuzzy",
        "exact_only_passes_threshold",
        "fuzzy_needed_for_threshold",
        "max_fuzzy_overlap",
        "mean_fuzzy_overlap",
        "confidence",
        "exact_sources",
        "fuzzy_sources",
        "question",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in qid_rows:
            for page in row.get("selected_pages", []):
                writer.writerow(
                    {
                        "qid": row["qid"],
                        "question_type": row["question_type"],
                        "page_uid": page["page_uid"],
                        "doc_id": page["doc_id"],
                        "page_idx": page["page_idx"],
                        "recorded_score": page["recorded_score"],
                        "exact_score": page["exact_score"],
                        "fuzzy_score": page["fuzzy_score"],
                        "fuzzy_fraction": page["fuzzy_fraction"],
                        "exact_match_count": page["exact_match_count"],
                        "fuzzy_match_count": page["fuzzy_match_count"],
                        "has_fuzzy": page["has_fuzzy"],
                        "exact_only_passes_threshold": page["exact_only_passes_threshold"],
                        "fuzzy_needed_for_threshold": page["fuzzy_needed_for_threshold"],
                        "max_fuzzy_overlap": page["max_fuzzy_overlap"],
                        "mean_fuzzy_overlap": page["mean_fuzzy_overlap"],
                        "confidence": page["confidence"],
                        "exact_sources": json.dumps(page["exact_sources"], ensure_ascii=False),
                        "fuzzy_sources": json.dumps(page["fuzzy_sources"], ensure_ascii=False),
                        "question": compact_text(row["question"], limit=220),
                    }
                )


def main() -> None:
    args = parse_args()
    labels_path = Path(args.pseudo_labels_jsonl)
    min_score = load_min_score(labels_path, str(args.summary_json or ""), args.min_score)
    rows = read_jsonl(labels_path)

    qid_filter = {str(qid) for qid in args.qid if str(qid).strip()}
    if qid_filter:
        rows = [row for row in rows if str(row.get("qid", "")) in qid_filter]

    qid_rows = [qid_record(row, min_score) for row in rows]
    report = summarize(qid_rows, min_score)
    payload = {
        "pseudo_labels_jsonl": str(labels_path),
        "summary_json": str(args.summary_json or infer_summary_path(labels_path)),
        "report": report,
    }

    write_markdown(
        Path(args.output_md),
        report,
        labels_path,
        qid_rows,
        int(args.example_limit),
    )

    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.output_jsonl:
        write_qid_jsonl(Path(args.output_jsonl), qid_rows)
    if args.output_csv:
        write_page_csv(Path(args.output_csv), qid_rows)

    print(f"saved_output_md={args.output_md}")
    if args.output_json:
        print(f"saved_output_json={args.output_json}")
    if args.output_jsonl:
        print(f"saved_output_jsonl={args.output_jsonl}")
    if args.output_csv:
        print(f"saved_output_csv={args.output_csv}")
    print(f"qid_count={report['qid_count']}")
    print(f"labeled_qids={report['labeled_qids']}")
    print(f"selected_page_count={report['selected_page_count']}")
    print(f"selected_pages_with_fuzzy={report['selected_pages_with_fuzzy']}")
    print(
        "selected_pages_fuzzy_needed_for_threshold="
        f"{report['selected_pages_fuzzy_needed_for_threshold']}"
    )
    print(f"qids_with_fuzzy_needed={report['qids_with_fuzzy_needed']}")


if __name__ == "__main__":
    main()
