#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

from scripts import build_mmqa_pseudo_page_labels as ppl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit existing MMQA pseudo-page labels against original support documents "
            "and recomputed MMQA evidence/page-text matches."
        )
    )
    parser.add_argument("--gold", required=True, help="Original MMQA_train/dev.jsonl")
    parser.add_argument("--strict-augmented-gold", required=True)
    parser.add_argument("--loose-augmented-gold", default="")
    parser.add_argument("--doc-pages-jsonl", required=True)
    parser.add_argument("--mmqa-texts-jsonl", required=True)
    parser.add_argument("--mmqa-tables-jsonl", required=True)
    parser.add_argument("--mmqa-images-jsonl", required=True)
    parser.add_argument("--id-url-mapping-jsonl", required=True)
    parser.add_argument("--min-token-overlap", type=float, default=0.72)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def load_by_qid(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in ppl.load_jsonl(path):
        qid = str(row.get("qid") or row.get("id") or "").strip()
        if qid:
            rows[qid] = row
    return rows


def metadata(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {}
    return row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}


def gold_page_uids(row: dict[str, Any] | None) -> list[str]:
    meta = metadata(row)
    values = meta.get("gold_page_uids") or meta.get("pseudo_gold_page_uids") or row.get("gold_page_uids") if row else []
    return [str(value).strip() for value in values or [] if str(value).strip()]


def page_doc(page_uid: str) -> str:
    value = str(page_uid)
    return value.rsplit("_page", 1)[0] if "_page" in value else value


def answer_modalities(row: dict[str, Any]) -> set[str]:
    modalities: set[str] = set()
    for answer in row.get("answers", []) or []:
        if isinstance(answer, dict) and answer.get("modality"):
            modalities.add(str(answer["modality"]))
    return modalities or {"unknown"}


def sorted_counter(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in counter.most_common()}


def update_label_stats(
    *,
    label_pages: list[str],
    scored_by_uid: dict[str, ppl.PageScore],
    support_docs: set[str],
    prefix: str,
    stats: dict[str, Any],
) -> None:
    stats[f"{prefix}_label_count_distribution"][len(label_pages)] += 1
    if label_pages:
        stats[f"{prefix}_labeled_qids"] += 1

    label_docs = {page_doc(uid) for uid in label_pages}
    if label_docs.issubset(support_docs):
        stats[f"{prefix}_subset_support_qids"] += 1
    else:
        stats[f"{prefix}_outside_support_qids"] += 1

    for uid in label_pages:
        score = scored_by_uid.get(uid)
        if not score:
            stats[f"{prefix}_missing_score_labels"] += 1
            stats[f"{prefix}_confidence_counts"]["missing_score"] += 1
            continue
        stats[f"{prefix}_label_scores"].append(float(score.score))
        stats[f"{prefix}_confidence_counts"][ppl.confidence(float(score.score))] += 1
        matches = score.exact_matches + score.fuzzy_matches
        if not matches:
            stats[f"{prefix}_no_match_labels"] += 1
        for match in matches:
            source = str(match.get("source", "") or "unknown")
            stats[f"{prefix}_evidence_source_counts"][source] += 1


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(out)


def pct(numer: int, denom: int) -> str:
    return f"{(100.0 * numer / denom):.2f}%" if denom else "n/a"


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines: list[str] = [
        "# MMQA Pseudo-Page Label Quality Audit",
        "",
        "## Coverage and Containment",
        "",
        markdown_table(
            ["policy", "labeled qids", "coverage", "subset of support docs", "outside support docs", "labels", "mean score"],
            [
                [
                    "strict",
                    summary["strict_labeled_qids"],
                    pct(summary["strict_labeled_qids"], summary["qid_count"]),
                    f"{summary['strict_subset_support_qids']} ({pct(summary['strict_subset_support_qids'], summary['qid_count'])})",
                    summary["strict_outside_support_qids"],
                    summary["strict_label_count"],
                    summary["strict_mean_label_score"],
                ],
                [
                    "loose",
                    summary.get("loose_labeled_qids", 0),
                    pct(summary.get("loose_labeled_qids", 0), summary["qid_count"]),
                    f"{summary.get('loose_subset_support_qids', 0)} ({pct(summary.get('loose_subset_support_qids', 0), summary['qid_count'])})",
                    summary.get("loose_outside_support_qids", 0),
                    summary.get("loose_label_count", 0),
                    summary.get("loose_mean_label_score"),
                ],
            ],
        ),
        "",
        "## Confidence Distribution",
        "",
        markdown_table(
            ["policy", "high", "medium", "low", "none/missing"],
            [
                [
                    "strict",
                    summary["strict_confidence_counts"].get("high", 0),
                    summary["strict_confidence_counts"].get("medium", 0),
                    summary["strict_confidence_counts"].get("low", 0),
                    summary["strict_confidence_counts"].get("none", 0)
                    + summary["strict_confidence_counts"].get("missing_score", 0),
                ],
                [
                    "loose",
                    summary.get("loose_confidence_counts", {}).get("high", 0),
                    summary.get("loose_confidence_counts", {}).get("medium", 0),
                    summary.get("loose_confidence_counts", {}).get("low", 0),
                    summary.get("loose_confidence_counts", {}).get("none", 0)
                    + summary.get("loose_confidence_counts", {}).get("missing_score", 0),
                ],
            ],
        ),
        "",
        "## Strict Evidence Source Counts",
        "",
        markdown_table(
            ["source", "count"],
            [[key, value] for key, value in list(summary["strict_evidence_source_counts"].items())[:20]],
        ),
        "",
        "## Strict Unlabeled Question Types",
        "",
        markdown_table(
            ["question type", "count"],
            [[key, value] for key, value in list(summary["strict_unlabeled_question_types"].items())[:20]],
        ),
        "",
        "## Strict Unlabeled Answer Modalities",
        "",
        markdown_table(
            ["answer modality", "count"],
            [[key, value] for key, value in summary["strict_unlabeled_answer_modalities"].items()],
        ),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    gold_rows = load_by_qid(Path(args.gold))
    strict_rows = load_by_qid(Path(args.strict_augmented_gold))
    loose_rows = load_by_qid(Path(args.loose_augmented_gold)) if args.loose_augmented_gold else {}
    pages_by_doc = ppl.load_page_texts(Path(args.doc_pages_jsonl))
    texts = ppl.load_by_id(args.mmqa_texts_jsonl)
    tables = ppl.load_by_id(args.mmqa_tables_jsonl)
    images = ppl.load_by_id(args.mmqa_images_jsonl)
    id_map = ppl.load_id_map(args.id_url_mapping_jsonl)
    url_to_id = ppl.load_url_to_id(args.id_url_mapping_jsonl)

    stats: dict[str, Any] = {
        "strict_labeled_qids": 0,
        "strict_subset_support_qids": 0,
        "strict_outside_support_qids": 0,
        "strict_missing_score_labels": 0,
        "strict_no_match_labels": 0,
        "strict_label_scores": [],
        "strict_label_count_distribution": Counter(),
        "strict_confidence_counts": Counter(),
        "strict_evidence_source_counts": Counter(),
        "loose_labeled_qids": 0,
        "loose_subset_support_qids": 0,
        "loose_outside_support_qids": 0,
        "loose_missing_score_labels": 0,
        "loose_no_match_labels": 0,
        "loose_label_scores": [],
        "loose_label_count_distribution": Counter(),
        "loose_confidence_counts": Counter(),
        "loose_evidence_source_counts": Counter(),
    }
    strict_unlabeled_question_types: Counter[str] = Counter()
    strict_unlabeled_answer_modalities: Counter[str] = Counter()
    question_type_counts: Counter[str] = Counter()
    answer_modality_counts: Counter[str] = Counter()

    for qid, row in gold_rows.items():
        qtype = str(metadata(row).get("type") or row.get("question_type") or "UNKNOWN")
        question_type_counts[qtype] += 1
        for modality in answer_modalities(row):
            answer_modality_counts[modality] += 1

        evidence, _ = ppl.evidence_for_row(
            row,
            tables_by_id=tables,
            texts_by_id=texts,
            images_by_id=images,
            id_map=id_map,
            url_to_id=url_to_id,
        )
        scored, _ = ppl.score_pages(
            row,
            evidence,
            pages_by_doc,
            min_token_overlap=float(args.min_token_overlap),
        )
        scored_by_uid = {item.page_uid: item for item in scored}
        support_docs = set(ppl.supporting_doc_ids(row))

        strict_pages = gold_page_uids(strict_rows.get(qid))
        loose_pages = gold_page_uids(loose_rows.get(qid)) if loose_rows else []
        if not strict_pages:
            strict_unlabeled_question_types[qtype] += 1
            for modality in answer_modalities(row):
                strict_unlabeled_answer_modalities[modality] += 1

        update_label_stats(
            label_pages=strict_pages,
            scored_by_uid=scored_by_uid,
            support_docs=support_docs,
            prefix="strict",
            stats=stats,
        )
        if loose_rows:
            update_label_stats(
                label_pages=loose_pages,
                scored_by_uid=scored_by_uid,
                support_docs=support_docs,
                prefix="loose",
                stats=stats,
            )

    summary = {
        "qid_count": len(gold_rows),
        "strict_labeled_qids": int(stats["strict_labeled_qids"]),
        "strict_coverage": stats["strict_labeled_qids"] / len(gold_rows) if gold_rows else None,
        "strict_subset_support_qids": int(stats["strict_subset_support_qids"]),
        "strict_outside_support_qids": int(stats["strict_outside_support_qids"]),
        "strict_label_count": int(len(stats["strict_label_scores"])),
        "strict_mean_label_score": round(fmean(stats["strict_label_scores"]), 6) if stats["strict_label_scores"] else None,
        "strict_missing_score_labels": int(stats["strict_missing_score_labels"]),
        "strict_no_match_labels": int(stats["strict_no_match_labels"]),
        "strict_label_count_distribution": sorted_counter(stats["strict_label_count_distribution"]),
        "strict_confidence_counts": sorted_counter(stats["strict_confidence_counts"]),
        "strict_evidence_source_counts": sorted_counter(stats["strict_evidence_source_counts"]),
        "strict_unlabeled_question_types": sorted_counter(strict_unlabeled_question_types),
        "strict_unlabeled_answer_modalities": sorted_counter(strict_unlabeled_answer_modalities),
        "question_type_counts": sorted_counter(question_type_counts),
        "answer_modality_counts": sorted_counter(answer_modality_counts),
        "loose_labeled_qids": int(stats["loose_labeled_qids"]),
        "loose_coverage": stats["loose_labeled_qids"] / len(gold_rows) if gold_rows and loose_rows else None,
        "loose_subset_support_qids": int(stats["loose_subset_support_qids"]),
        "loose_outside_support_qids": int(stats["loose_outside_support_qids"]),
        "loose_label_count": int(len(stats["loose_label_scores"])),
        "loose_mean_label_score": round(fmean(stats["loose_label_scores"]), 6) if stats["loose_label_scores"] else None,
        "loose_missing_score_labels": int(stats["loose_missing_score_labels"]),
        "loose_no_match_labels": int(stats["loose_no_match_labels"]),
        "loose_label_count_distribution": sorted_counter(stats["loose_label_count_distribution"]),
        "loose_confidence_counts": sorted_counter(stats["loose_confidence_counts"]),
        "loose_evidence_source_counts": sorted_counter(stats["loose_evidence_source_counts"]),
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(Path(args.output_md), summary)

    print(f"saved_output_md={args.output_md}")
    print(f"saved_output_json={args.output_json}")
    print(f"qid_count={summary['qid_count']}")
    print(f"strict_labeled_qids={summary['strict_labeled_qids']}")
    print(f"strict_coverage={summary['strict_coverage']:.6f}")
    print(f"strict_subset_support_qids={summary['strict_subset_support_qids']}")
    print(f"strict_outside_support_qids={summary['strict_outside_support_qids']}")
    print(f"strict_confidence_counts={summary['strict_confidence_counts']}")
    print(f"loose_labeled_qids={summary['loose_labeled_qids']}")
    if summary["loose_coverage"] is not None:
        print(f"loose_coverage={summary['loose_coverage']:.6f}")


if __name__ == "__main__":
    main()
