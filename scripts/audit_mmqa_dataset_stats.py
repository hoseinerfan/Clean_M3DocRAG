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
            "Summarize MMQA/M3DocVQA dataset structure before choosing page-label and "
            "recall-oriented reranking configs. The audit reports question types, "
            "modalities, supporting document structure, answer evidence structure, and "
            "optional pseudo-page label coverage."
        )
    )
    parser.add_argument(
        "--gold",
        action="append",
        required=True,
        help="MMQA JSONL or augmented gold JSONL, optionally LABEL=path. Repeatable.",
    )
    parser.add_argument(
        "--doc-pages-jsonl",
        action="append",
        default=[],
        help="Optional page-text JSONL, optionally LABEL=path. Used for page-count stats.",
    )
    parser.add_argument("--output-md", default="", help="Optional Markdown report path.")
    parser.add_argument("--output-json", default="", help="Optional JSON report path.")
    parser.add_argument("--output-csv", default="", help="Optional CSV summary path.")
    parser.add_argument("--topn-question-types", type=int, default=40)
    return parser.parse_args()


def parse_labeled_path(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, raw_path = spec.split("=", 1)
        label = label.strip() or Path(raw_path).stem
        return label, Path(raw_path)
    path = Path(spec)
    return path.stem, path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def page_doc(uid: str) -> str:
    value = str(uid).strip()
    if "_page" not in value:
        return value
    return value.rsplit("_page", 1)[0]


def gold_page_uids(row: dict[str, Any]) -> list[str]:
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    raw = (
        metadata.get("gold_page_uids")
        or metadata.get("pseudo_gold_page_uids")
        or row.get("gold_page_uids")
        or row.get("pseudo_gold_page_uids")
        or []
    )
    out: list[str] = []
    seen: set[str] = set()
    for value in raw:
        uid = str(value).strip()
        if uid and uid not in seen:
            seen.add(uid)
            out.append(uid)
    return out


def question_type(row: dict[str, Any]) -> str:
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    return str(metadata.get("type", "UNKNOWN")).strip() or "UNKNOWN"


def modalities(row: dict[str, Any]) -> list[str]:
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    raw = metadata.get("modalities", [])
    if isinstance(raw, list):
        vals = [str(value).strip() for value in raw if str(value).strip()]
        if vals:
            return vals
    qtype = question_type(row)
    vals = []
    for name in ("Text", "Table", "Image"):
        if f"{name}Q" in qtype or f"{name}ListQ" in qtype:
            vals.append(name.lower())
    return vals or ["unknown"]


def supporting_contexts(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [ctx for ctx in row.get("supporting_context", []) or [] if isinstance(ctx, dict)]


def supporting_doc_ids(row: dict[str, Any]) -> list[str]:
    docs: list[str] = []
    seen: set[str] = set()
    for ctx in supporting_contexts(row):
        doc_id = str(ctx.get("doc_id", "")).strip()
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            docs.append(doc_id)
    return docs


def supporting_doc_part_counts(row: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for ctx in supporting_contexts(row):
        part = str(ctx.get("doc_part", "UNKNOWN")).strip() or "UNKNOWN"
        counts[part] += 1
    return counts


def answer_objects(row: dict[str, Any]) -> list[dict[str, Any]]:
    answers: list[dict[str, Any]] = []
    for answer in row.get("answers", []) or []:
        if isinstance(answer, dict):
            answers.append(answer)
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    for group in metadata.get("intermediate_answers", []) or []:
        if isinstance(group, list):
            answers.extend(item for item in group if isinstance(item, dict))
    return answers


def answer_evidence_counts(row: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for answer in answer_objects(row):
        counts["answer_objects"] += 1
        counts[f"answer_type::{str(answer.get('type', 'UNKNOWN')).strip() or 'UNKNOWN'}"] += 1
        counts[f"answer_modality::{str(answer.get('modality', 'UNKNOWN')).strip() or 'UNKNOWN'}"] += 1
        counts["text_instances"] += len(answer.get("text_instances", []) or [])
        counts["image_instances"] += len(answer.get("image_instances", []) or [])
        counts["table_indices"] += len(answer.get("table_indices", []) or [])
    return counts


def ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def summarize_gold(label: str, path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)

    qtype_counts: Counter[str] = Counter()
    modality_counts: Counter[str] = Counter()
    modality_combo_counts: Counter[str] = Counter()
    support_doc_count_hist: Counter[int] = Counter()
    support_ctx_count_hist: Counter[int] = Counter()
    support_part_counts: Counter[str] = Counter()
    answer_counts: Counter[str] = Counter()
    pseudo_label_count_hist: Counter[int] = Counter()
    pseudo_max_per_doc_hist: Counter[int] = Counter()
    qtype_label_counts: dict[str, list[int]] = defaultdict(list)
    qtype_support_doc_counts: dict[str, list[int]] = defaultdict(list)

    support_doc_counts: list[int] = []
    support_ctx_counts: list[int] = []
    pseudo_label_counts: list[int] = []
    pseudo_label_counts_labeled: list[int] = []
    max_pages_per_doc_values: list[int] = []
    examples_multi_same_doc: list[dict[str, Any]] = []
    examples_many_labels: list[dict[str, Any]] = []

    qids_with_multi_support_docs = 0
    qids_with_multi_support_contexts_same_doc = 0
    qids_with_pseudo_labels = 0
    qids_with_multi_page_same_doc = 0
    qids_with_5plus_pseudo_pages = 0
    qids_with_10plus_pseudo_pages = 0

    for row in rows:
        qtype = question_type(row)
        qtype_counts[qtype] += 1

        mods = sorted(set(modalities(row)))
        for mod in mods:
            modality_counts[mod] += 1
        modality_combo_counts["+".join(mods)] += 1

        docs = supporting_doc_ids(row)
        contexts = supporting_contexts(row)
        support_doc_counts.append(len(docs))
        support_ctx_counts.append(len(contexts))
        support_doc_count_hist[len(docs)] += 1
        support_ctx_count_hist[len(contexts)] += 1
        qtype_support_doc_counts[qtype].append(len(docs))
        if len(docs) > 1:
            qids_with_multi_support_docs += 1
        doc_ctx_counts = Counter(str(ctx.get("doc_id", "")).strip() for ctx in contexts)
        doc_ctx_counts.pop("", None)
        if any(value > 1 for value in doc_ctx_counts.values()):
            qids_with_multi_support_contexts_same_doc += 1

        support_part_counts.update(supporting_doc_part_counts(row))
        answer_counts.update(answer_evidence_counts(row))

        pages = gold_page_uids(row)
        n_pages = len(pages)
        pseudo_label_counts.append(n_pages)
        pseudo_label_count_hist[n_pages] += 1
        qtype_label_counts[qtype].append(n_pages)
        doc_page_counts = Counter(page_doc(uid) for uid in pages)
        max_per_doc = max(doc_page_counts.values(), default=0)
        pseudo_max_per_doc_hist[max_per_doc] += 1
        if n_pages:
            qids_with_pseudo_labels += 1
            pseudo_label_counts_labeled.append(n_pages)
            max_pages_per_doc_values.append(max_per_doc)
        if n_pages >= 5:
            qids_with_5plus_pseudo_pages += 1
            if len(examples_many_labels) < 20:
                examples_many_labels.append(
                    {
                        "qid": str(row.get("qid", "")),
                        "question_type": qtype,
                        "label_count": n_pages,
                        "max_pages_in_one_doc": max_per_doc,
                        "gold_page_uids": pages,
                    }
                )
        if n_pages >= 10:
            qids_with_10plus_pseudo_pages += 1
        if max_per_doc > 1:
            qids_with_multi_page_same_doc += 1
            if len(examples_multi_same_doc) < 20:
                examples_multi_same_doc.append(
                    {
                        "qid": str(row.get("qid", "")),
                        "question_type": qtype,
                        "label_count": n_pages,
                        "doc_page_counts": dict(sorted(doc_page_counts.items())),
                        "gold_page_uids": pages,
                    }
                )

    qtype_summary = []
    for qtype, count in sorted(qtype_counts.items()):
        labels = qtype_label_counts.get(qtype, [])
        support_docs = qtype_support_doc_counts.get(qtype, [])
        qtype_summary.append(
            {
                "question_type": qtype,
                "qid_count": count,
                "fraction": round(ratio(count, len(rows)), 6),
                "mean_support_docs": round(float(fmean(support_docs)), 4) if support_docs else 0.0,
                "mean_pseudo_pages": round(float(fmean(labels)), 4) if labels else 0.0,
                "labeled_qid_count": sum(1 for value in labels if value > 0),
                "multi_page_same_doc_qid_count": sum(
                    1
                    for row in rows
                    if question_type(row) == qtype
                    and max(Counter(page_doc(uid) for uid in gold_page_uids(row)).values(), default=0)
                    > 1
                ),
            }
        )

    return {
        "label": label,
        "path": str(path),
        "qid_count": len(rows),
        "question_type_counts": dict(sorted(qtype_counts.items())),
        "modality_counts": dict(sorted(modality_counts.items())),
        "modality_combo_counts": dict(sorted(modality_combo_counts.items())),
        "supporting_doc_count_hist": {str(k): v for k, v in sorted(support_doc_count_hist.items())},
        "supporting_context_count_hist": {
            str(k): v for k, v in sorted(support_ctx_count_hist.items())
        },
        "supporting_doc_part_counts": dict(sorted(support_part_counts.items())),
        "answer_evidence_counts": dict(sorted(answer_counts.items())),
        "mean_supporting_docs": round(float(fmean(support_doc_counts)), 6)
        if support_doc_counts
        else 0.0,
        "median_supporting_docs": float(median(support_doc_counts)) if support_doc_counts else 0.0,
        "max_supporting_docs": max(support_doc_counts, default=0),
        "mean_supporting_contexts": round(float(fmean(support_ctx_counts)), 6)
        if support_ctx_counts
        else 0.0,
        "qids_with_multi_support_docs": qids_with_multi_support_docs,
        "qids_with_multi_support_docs_fraction": round(
            ratio(qids_with_multi_support_docs, len(rows)), 6
        ),
        "qids_with_multi_support_contexts_same_doc": qids_with_multi_support_contexts_same_doc,
        "qids_with_multi_support_contexts_same_doc_fraction": round(
            ratio(qids_with_multi_support_contexts_same_doc, len(rows)), 6
        ),
        "pseudo_page_label_count": sum(pseudo_label_counts),
        "pseudo_labeled_qid_count": qids_with_pseudo_labels,
        "pseudo_labeled_qid_fraction": round(ratio(qids_with_pseudo_labels, len(rows)), 6),
        "mean_pseudo_pages_per_qid": round(float(fmean(pseudo_label_counts)), 6)
        if pseudo_label_counts
        else 0.0,
        "mean_pseudo_pages_per_labeled_qid": round(float(fmean(pseudo_label_counts_labeled)), 6)
        if pseudo_label_counts_labeled
        else 0.0,
        "max_pseudo_pages_per_qid": max(pseudo_label_counts, default=0),
        "qids_with_5plus_pseudo_pages": qids_with_5plus_pseudo_pages,
        "qids_with_5plus_pseudo_pages_fraction": round(
            ratio(qids_with_5plus_pseudo_pages, len(rows)), 6
        ),
        "qids_with_10plus_pseudo_pages": qids_with_10plus_pseudo_pages,
        "qids_with_10plus_pseudo_pages_fraction": round(
            ratio(qids_with_10plus_pseudo_pages, len(rows)), 6
        ),
        "qids_with_multi_page_same_doc": qids_with_multi_page_same_doc,
        "qids_with_multi_page_same_doc_fraction": round(
            ratio(qids_with_multi_page_same_doc, len(rows)), 6
        ),
        "max_pages_in_one_doc": max(max_pages_per_doc_values, default=0),
        "pseudo_page_count_hist": {str(k): v for k, v in sorted(pseudo_label_count_hist.items())},
        "pseudo_max_pages_per_doc_hist": {
            str(k): v for k, v in sorted(pseudo_max_per_doc_hist.items())
        },
        "question_type_summary": qtype_summary,
        "examples_many_labels": examples_many_labels,
        "examples_multi_page_same_doc": examples_multi_same_doc,
    }


def page_idx_from_row(row: dict[str, Any]) -> int | None:
    for key in ("page_idx", "page_id", "page"):
        if row.get(key) is not None:
            try:
                return int(row[key])
            except (TypeError, ValueError):
                return None
    uid = str(row.get("page_uid", ""))
    if "_page" in uid:
        try:
            return int(uid.rsplit("_page", 1)[1])
        except ValueError:
            return None
    return None


def doc_id_from_page_row(row: dict[str, Any]) -> str:
    doc_id = str(row.get("doc_id", "")).strip()
    if doc_id:
        return doc_id
    uid = str(row.get("page_uid", ""))
    if "_page" in uid:
        return uid.rsplit("_page", 1)[0]
    return ""


def summarize_page_text(label: str, path: Path) -> dict[str, Any]:
    pages_per_doc: Counter[str] = Counter()
    empty_text_pages = 0
    total_pages = 0
    for row in read_jsonl(path):
        doc_id = doc_id_from_page_row(row)
        if not doc_id:
            continue
        total_pages += 1
        pages_per_doc[doc_id] += 1
        text = ""
        for key in ("text", "page_text", "ocr_text", "markdown", "vlm_text", "content"):
            value = row.get(key)
            if isinstance(value, str):
                text += value
        if not text.strip():
            empty_text_pages += 1
    counts = list(pages_per_doc.values())
    return {
        "label": label,
        "path": str(path),
        "doc_count": len(pages_per_doc),
        "page_count": total_pages,
        "empty_text_page_count": empty_text_pages,
        "mean_pages_per_doc": round(float(fmean(counts)), 6) if counts else 0.0,
        "median_pages_per_doc": float(median(counts)) if counts else 0.0,
        "max_pages_per_doc": max(counts, default=0),
        "doc_count_ge_5_pages": sum(1 for value in counts if value >= 5),
        "doc_count_ge_10_pages": sum(1 for value in counts if value >= 10),
        "doc_count_ge_20_pages": sum(1 for value in counts if value >= 20),
    }


def write_md(path: Path, gold_summaries: list[dict[str, Any]], page_summaries: list[dict[str, Any]], topn: int) -> None:
    lines: list[str] = ["# MMQA / M3DocVQA Dataset Stats", ""]
    lines.extend(
        [
            "## Split Summary",
            "",
            "| split | qids | mean support docs | multi-doc qids | pseudo-labeled qids | pseudo labels | mean pseudo pages/labeled qid | max pseudo pages/qid | qids with >1 pseudo page in one doc |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in gold_summaries:
        lines.append(
            "| {label} | {qid_count} | {mean_supporting_docs:.3f} | "
            "{qids_with_multi_support_docs} ({qids_with_multi_support_docs_fraction:.1%}) | "
            "{pseudo_labeled_qid_count} ({pseudo_labeled_qid_fraction:.1%}) | "
            "{pseudo_page_label_count} | {mean_pseudo_pages_per_labeled_qid:.3f} | "
            "{max_pseudo_pages_per_qid} | "
            "{qids_with_multi_page_same_doc} ({qids_with_multi_page_same_doc_fraction:.1%}) |".format(
                **item
            )
        )

    if page_summaries:
        lines.extend(
            [
                "",
                "## Page Text Summary",
                "",
                "| split | docs | pages | empty pages | mean pages/doc | median pages/doc | max pages/doc | docs >=10 pages | docs >=20 pages |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in page_summaries:
            lines.append(
                "| {label} | {doc_count} | {page_count} | {empty_text_page_count} | "
                "{mean_pages_per_doc:.3f} | {median_pages_per_doc:.1f} | {max_pages_per_doc} | "
                "{doc_count_ge_10_pages} | {doc_count_ge_20_pages} |".format(**item)
            )

    for item in gold_summaries:
        lines.extend(["", f"## {item['label']}", "", "### Question Types", ""])
        lines.extend(
            [
                "| question type | qids | fraction | mean support docs | mean pseudo pages | labeled qids | multi-page same-doc qids |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        qtypes = sorted(item["question_type_summary"], key=lambda row: (-row["qid_count"], row["question_type"]))
        for row in qtypes[:topn]:
            lines.append(
                "| {question_type} | {qid_count} | {fraction:.1%} | {mean_support_docs:.3f} | "
                "{mean_pseudo_pages:.3f} | {labeled_qid_count} | {multi_page_same_doc_qid_count} |".format(
                    **row
                )
            )

        lines.extend(["", "### Modality Combinations", ""])
        lines.extend(["| modalities | qids |", "| --- | ---: |"])
        for key, value in sorted(item["modality_combo_counts"].items(), key=lambda pair: (-pair[1], pair[0])):
            lines.append(f"| {key} | {value} |")

        lines.extend(["", "### Supporting Document Count", ""])
        lines.extend(["| support docs/qid | qids |", "| ---: | ---: |"])
        for key, value in item["supporting_doc_count_hist"].items():
            lines.append(f"| {key} | {value} |")

        if item["pseudo_page_count_hist"]:
            lines.extend(["", "### Pseudo-Page Label Count", ""])
            lines.extend(["| pseudo pages/qid | qids |", "| ---: | ---: |"])
            for key, value in item["pseudo_page_count_hist"].items():
                lines.append(f"| {key} | {value} |")

        if item["examples_multi_page_same_doc"]:
            lines.extend(["", "### Examples With Multiple Pseudo Pages From One Doc", ""])
            for row in item["examples_multi_page_same_doc"][:10]:
                lines.append(
                    f"- `{row['qid']}` ({row['question_type']}): "
                    f"{row['label_count']} labels, doc_page_counts={row['doc_page_counts']}"
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fields = [
        "label",
        "qid_count",
        "mean_supporting_docs",
        "qids_with_multi_support_docs",
        "qids_with_multi_support_docs_fraction",
        "pseudo_labeled_qid_count",
        "pseudo_page_label_count",
        "mean_pseudo_pages_per_labeled_qid",
        "max_pseudo_pages_per_qid",
        "qids_with_multi_page_same_doc",
        "qids_with_multi_page_same_doc_fraction",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in summaries:
            writer.writerow({key: item.get(key) for key in fields})


def main() -> None:
    args = parse_args()
    gold_summaries = [summarize_gold(label, path) for label, path in map(parse_labeled_path, args.gold)]
    page_summaries = [
        summarize_page_text(label, path) for label, path in map(parse_labeled_path, args.doc_pages_jsonl)
    ]

    print(
        "| split | qids | mean_support_docs | multi_doc_qids | pseudo_labeled_qids | "
        "pseudo_labels | mean_pseudo_pages_labeled | multi_page_same_doc |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in gold_summaries:
        print(
            "| {label} | {qid_count} | {mean_supporting_docs:.3f} | "
            "{qids_with_multi_support_docs} | {pseudo_labeled_qid_count} | "
            "{pseudo_page_label_count} | {mean_pseudo_pages_per_labeled_qid:.3f} | "
            "{qids_with_multi_page_same_doc} |".format(**item)
        )

    if args.output_md:
        write_md(Path(args.output_md), gold_summaries, page_summaries, int(args.topn_question_types))
        print(f"saved_output_md={args.output_md}")
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"gold_summaries": gold_summaries, "page_summaries": page_summaries}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        print(f"saved_output_json={args.output_json}")
    if args.output_csv:
        write_csv(Path(args.output_csv), gold_summaries)
        print(f"saved_output_csv={args.output_csv}")


if __name__ == "__main__":
    main()
