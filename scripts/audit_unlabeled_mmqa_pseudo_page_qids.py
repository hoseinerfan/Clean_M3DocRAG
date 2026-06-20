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
            "Audit qids that did not receive MMQA pseudo-page labels. The input is the "
            "JSONL produced by scripts/build_mmqa_pseudo_page_labels.py."
        )
    )
    parser.add_argument("--pseudo-labels-jsonl", required=True)
    parser.add_argument(
        "--summary-json",
        default="",
        help="Optional summary JSON from the same pseudo-label run. If omitted, the script tries <stem>.summary.json.",
    )
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--example-limit", type=int, default=40)
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
    candidates = []
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


def selected_pages(row: dict[str, Any]) -> list[str]:
    pages = row.get("pseudo_gold_page_uids", [])
    return [str(page) for page in pages] if isinstance(pages, list) else []


def evidence_source_counts(row: dict[str, Any]) -> Counter[str]:
    raw = row.get("evidence_source_counts", {})
    out: Counter[str] = Counter()
    if isinstance(raw, dict):
        for key, value in raw.items():
            try:
                out[str(key)] += int(value)
            except (TypeError, ValueError):
                continue
    return out


def top_score(row: dict[str, Any]) -> float:
    top_pages = row.get("top_scored_pages", [])
    if not isinstance(top_pages, list) or not top_pages:
        return 0.0
    try:
        return float(top_pages[0].get("score", 0.0))
    except (AttributeError, TypeError, ValueError):
        return 0.0


def top_page_uid(row: dict[str, Any]) -> str:
    top_pages = row.get("top_scored_pages", [])
    if not isinstance(top_pages, list) or not top_pages or not isinstance(top_pages[0], dict):
        return ""
    return str(top_pages[0].get("page_uid", ""))


def top_match_sources(row: dict[str, Any]) -> list[str]:
    top_pages = row.get("top_scored_pages", [])
    if not isinstance(top_pages, list) or not top_pages or not isinstance(top_pages[0], dict):
        return []
    counter: Counter[str] = Counter()
    for match in (top_pages[0].get("exact_matches", []) or []) + (
        top_pages[0].get("fuzzy_matches", []) or []
    ):
        if isinstance(match, dict) and match.get("source"):
            counter[str(match["source"])] += 1
    return [name for name, _ in counter.most_common(5)]


def support_part_signature(row: dict[str, Any]) -> str:
    parts_by_doc = row.get("supporting_doc_parts", {})
    parts: set[str] = set()
    if isinstance(parts_by_doc, dict):
        for values in parts_by_doc.values():
            if isinstance(values, list):
                parts.update(str(value) for value in values if value)
            elif values:
                parts.add(str(values))
    return "+".join(sorted(parts)) if parts else "none"


def top_score_bucket(value: float, min_score: float | None) -> str:
    if value <= 0:
        return "no_scored_pages"
    if min_score is not None and value < float(min_score):
        return f"below_min_score(<{float(min_score):g})"
    if value < 4:
        return "score_0_to_4"
    if value < 8:
        return "score_4_to_8"
    if value < 14:
        return "score_8_to_14"
    return "score_14_plus"


def classify_root_cause(row: dict[str, Any], min_score: float | None) -> str:
    if selected_pages(row):
        return "matched"

    status = str(row.get("status", "") or "")
    missing_docs = row.get("missing_page_text_doc_ids", [])
    evidence_count = int(row.get("evidence_count", 0) or 0)
    candidate_pages_scored = int(row.get("candidate_pages_scored", 0) or 0)
    gold_docs = row.get("gold_doc_ids", [])
    best_score = top_score(row)

    if not isinstance(gold_docs, list) or not gold_docs:
        return "no_supporting_docs"
    if status == "missing_all_supporting_doc_page_text":
        return "missing_all_supporting_doc_page_text"
    if status == "missing_some_supporting_doc_page_text" or missing_docs:
        return "missing_some_supporting_doc_page_text"
    if evidence_count <= 0:
        return "no_extractable_mmqa_evidence"
    if candidate_pages_scored <= 0:
        return "no_evidence_phrase_matched_page_text"
    if min_score is not None and best_score < float(min_score):
        return "best_match_below_min_score"
    return "selection_policy_or_budget_blocked"


def compact_question(text: Any, limit: int = 130) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def audit(rows: list[dict[str, Any]], min_score: float | None, example_limit: int) -> dict[str, Any]:
    total = len(rows)
    unlabeled_rows = [row for row in rows if not selected_pages(row)]
    labeled_rows = total - len(unlabeled_rows)

    root_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    qtype_counts: Counter[str] = Counter()
    qtype_by_root: dict[str, Counter[str]] = defaultdict(Counter)
    support_signature_counts: Counter[str] = Counter()
    top_score_buckets: Counter[str] = Counter()
    aggregate_evidence_sources: Counter[str] = Counter()
    top_scores: list[float] = []

    examples: list[dict[str, Any]] = []
    examples_by_root: Counter[str] = Counter()

    for row in unlabeled_rows:
        root = classify_root_cause(row, min_score)
        qtype = str(row.get("question_type", "UNKNOWN") or "UNKNOWN")
        status = str(row.get("status", "") or "")
        best_score = top_score(row)

        root_counts[root] += 1
        status_counts[status or "missing"] += 1
        qtype_counts[qtype] += 1
        qtype_by_root[root][qtype] += 1
        support_signature_counts[support_part_signature(row)] += 1
        top_score_buckets[top_score_bucket(best_score, min_score)] += 1
        aggregate_evidence_sources.update(evidence_source_counts(row))
        top_scores.append(best_score)

        if len(examples) < example_limit or examples_by_root[root] < max(3, example_limit // 10):
            examples_by_root[root] += 1
            if len(examples) < max(example_limit, len(root_counts) * 3):
                examples.append(
                    {
                        "qid": str(row.get("qid", "")),
                        "question_type": qtype,
                        "root_cause": root,
                        "status": status,
                        "support_parts": support_part_signature(row),
                        "gold_doc_count": len(row.get("gold_doc_ids", []) or []),
                        "evidence_count": int(row.get("evidence_count", 0) or 0),
                        "candidate_pages_scored": int(row.get("candidate_pages_scored", 0) or 0),
                        "top_score": round(best_score, 6),
                        "top_page_uid": top_page_uid(row),
                        "top_match_sources": top_match_sources(row),
                        "missing_page_text_doc_count": len(row.get("missing_page_text_doc_ids", []) or []),
                        "question": compact_question(row.get("question", "")),
                    }
                )

    qtype_root_rows = []
    for root, counter in sorted(qtype_by_root.items()):
        for qtype, count in counter.most_common():
            qtype_root_rows.append(
                {
                    "root_cause": root,
                    "question_type": qtype,
                    "count": int(count),
                }
            )

    return {
        "total_qids": total,
        "labeled_qids": labeled_rows,
        "unlabeled_qids": len(unlabeled_rows),
        "unlabeled_fraction": round(len(unlabeled_rows) / total, 6) if total else 0.0,
        "min_score": min_score,
        "mean_top_score_unlabeled": round(float(fmean(top_scores)), 6) if top_scores else 0.0,
        "root_cause_counts": dict(root_counts.most_common()),
        "builder_status_counts": dict(status_counts.most_common()),
        "question_type_counts_unlabeled": dict(qtype_counts.most_common()),
        "support_part_signature_counts_unlabeled": dict(support_signature_counts.most_common()),
        "top_score_bucket_counts_unlabeled": dict(top_score_buckets.most_common()),
        "evidence_source_counts_unlabeled": dict(aggregate_evidence_sources.most_common()),
        "question_type_by_root_cause": qtype_root_rows,
        "examples": examples[:example_limit],
    }


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def write_markdown(path: Path, report: dict[str, Any], labels_path: Path) -> None:
    lines: list[str] = [
        "# Unlabeled MMQA Pseudo-Page QID Audit",
        "",
        f"- pseudo_labels_jsonl: `{labels_path}`",
        f"- total_qids: `{report['total_qids']}`",
        f"- labeled_qids: `{report['labeled_qids']}`",
        f"- unlabeled_qids: `{report['unlabeled_qids']}`",
        f"- unlabeled_fraction: `{report['unlabeled_fraction']}`",
        f"- min_score: `{report['min_score']}`",
        "",
        "## Root Cause Counts",
        "",
    ]
    lines.extend(
        markdown_table(
            ["root_cause", "count"],
            [[key, value] for key, value in report["root_cause_counts"].items()],
        )
    )
    lines.extend(["", "## Builder Status Counts", ""])
    lines.extend(
        markdown_table(
            ["status", "count"],
            [[key, value] for key, value in report["builder_status_counts"].items()],
        )
    )
    lines.extend(["", "## Unlabeled Question Types", ""])
    lines.extend(
        markdown_table(
            ["question_type", "count"],
            [[key, value] for key, value in report["question_type_counts_unlabeled"].items()],
        )
    )
    lines.extend(["", "## Support Part Signatures", ""])
    lines.extend(
        markdown_table(
            ["support_parts", "count"],
            [[key, value] for key, value in report["support_part_signature_counts_unlabeled"].items()],
        )
    )
    lines.extend(["", "## Top Score Buckets", ""])
    lines.extend(
        markdown_table(
            ["bucket", "count"],
            [[key, value] for key, value in report["top_score_bucket_counts_unlabeled"].items()],
        )
    )
    lines.extend(["", "## Evidence Sources Available in Unlabeled QIDs", ""])
    lines.extend(
        markdown_table(
            ["source", "count"],
            [[key, value] for key, value in report["evidence_source_counts_unlabeled"].items()],
        )
    )
    lines.extend(["", "## Examples", ""])
    lines.extend(
        markdown_table(
            [
                "qid",
                "question_type",
                "root_cause",
                "support_parts",
                "evidence_count",
                "scored_pages",
                "top_score",
                "top_page",
                "top_match_sources",
                "question",
            ],
            [
                [
                    row["qid"],
                    row["question_type"],
                    row["root_cause"],
                    row["support_parts"],
                    row["evidence_count"],
                    row["candidate_pages_scored"],
                    row["top_score"],
                    row["top_page_uid"],
                    ", ".join(row["top_match_sources"]),
                    row["question"].replace("|", "\\|"),
                ]
                for row in report["examples"]
            ],
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, examples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "qid",
        "question_type",
        "root_cause",
        "status",
        "support_parts",
        "gold_doc_count",
        "evidence_count",
        "candidate_pages_scored",
        "top_score",
        "top_page_uid",
        "top_match_sources",
        "missing_page_text_doc_count",
        "question",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in examples:
            out = dict(row)
            out["top_match_sources"] = ",".join(row.get("top_match_sources", []))
            writer.writerow(out)


def write_unlabeled_jsonl(path: Path, rows: list[dict[str, Any]], min_score: float | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if selected_pages(row):
                continue
            record = {
                "qid": str(row.get("qid", "")),
                "question": row.get("question", ""),
                "question_type": row.get("question_type", "UNKNOWN"),
                "root_cause": classify_root_cause(row, min_score),
                "status": row.get("status", ""),
                "gold_doc_ids": row.get("gold_doc_ids", []),
                "supporting_doc_parts": row.get("supporting_doc_parts", {}),
                "evidence_count": row.get("evidence_count", 0),
                "evidence_source_counts": row.get("evidence_source_counts", {}),
                "candidate_pages_scored": row.get("candidate_pages_scored", 0),
                "top_scored_pages": row.get("top_scored_pages", []),
                "missing_page_text_doc_ids": row.get("missing_page_text_doc_ids", []),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    labels_path = Path(args.pseudo_labels_jsonl)
    rows = read_jsonl(labels_path)
    min_score = load_min_score(labels_path, args.summary_json, args.min_score)
    report = audit(rows, min_score, int(args.example_limit))

    output_md = Path(args.output_md)
    write_markdown(output_md, report, labels_path)

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.output_csv:
        write_csv(Path(args.output_csv), report["examples"])
    if args.output_jsonl:
        write_unlabeled_jsonl(Path(args.output_jsonl), rows, min_score)

    print(f"saved_output_md={output_md}")
    if args.output_json:
        print(f"saved_output_json={args.output_json}")
    if args.output_csv:
        print(f"saved_output_csv={args.output_csv}")
    if args.output_jsonl:
        print(f"saved_output_jsonl={args.output_jsonl}")
    print(f"total_qids={report['total_qids']}")
    print(f"labeled_qids={report['labeled_qids']}")
    print(f"unlabeled_qids={report['unlabeled_qids']}")
    print(f"root_cause_counts={report['root_cause_counts']}")


if __name__ == "__main__":
    main()
