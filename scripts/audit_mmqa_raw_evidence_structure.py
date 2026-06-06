#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


PAGE_KEY_RE = re.compile(r"(^|_)(page|pageid|pageidx|page_id|page_idx|page_uid)($|_)", re.I)
PAGE_VALUE_RE = re.compile(r"_page\d+\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit raw MMQA train/dev JSONL files for page-like fields and evidence-object "
            "structure. Raw MMQA usually has supporting documents/evidence objects rather "
            "than official page labels; this script makes that explicit and measures how "
            "often multiple evidence objects attach to the same document."
        )
    )
    parser.add_argument("--gold", action="append", required=True, help="LABEL=MMQA_*.jsonl")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--example-limit", type=int, default=15)
    return parser.parse_args()


def parse_labeled_path(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, raw_path = spec.split("=", 1)
        return label.strip() or Path(raw_path).stem, Path(raw_path)
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


def question_type(row: dict[str, Any]) -> str:
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    return str(metadata.get("type", "UNKNOWN")).strip() or "UNKNOWN"


def supporting_doc_ids(row: dict[str, Any]) -> list[str]:
    docs: list[str] = []
    seen: set[str] = set()
    for ctx in row.get("supporting_context", []) or []:
        if not isinstance(ctx, dict):
            continue
        doc_id = str(ctx.get("doc_id", "")).strip()
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            docs.append(doc_id)
    return docs


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


def scan_page_like_fields(obj: Any, prefix: str = "") -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if PAGE_KEY_RE.search(str(key)):
                hits.append({"path": path, "kind": "key", "value": str(value)[:120]})
            hits.extend(scan_page_like_fields(value, path))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            hits.extend(scan_page_like_fields(value, f"{prefix}[{idx}]"))
    elif isinstance(obj, str) and PAGE_VALUE_RE.search(obj):
        hits.append({"path": prefix, "kind": "value", "value": obj[:120]})
    return hits


def evidence_doc_counts(row: dict[str, Any]) -> tuple[Counter[str], Counter[str]]:
    counts: Counter[str] = Counter()
    typed_counts: Counter[str] = Counter()
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    table_id = str(metadata.get("table_id", "")).strip()

    for answer in answer_objects(row):
        for instance in answer.get("text_instances", []) or []:
            if not isinstance(instance, dict):
                continue
            doc_id = str(instance.get("doc_id", "")).strip()
            if doc_id:
                counts[doc_id] += 1
                typed_counts[f"text_instance::{doc_id}"] += 1
        for instance in answer.get("image_instances", []) or []:
            if not isinstance(instance, dict):
                continue
            doc_id = str(instance.get("doc_id", "")).strip()
            if doc_id:
                counts[doc_id] += 1
                typed_counts[f"image_instance::{doc_id}"] += 1
        table_indices = answer.get("table_indices", []) or []
        if table_id and table_indices:
            counts[table_id] += len(table_indices)
            typed_counts[f"table_index::{table_id}"] += len(table_indices)

    return counts, typed_counts


def ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def summarize(label: str, path: Path, *, example_limit: int) -> dict[str, Any]:
    rows = read_jsonl(path)
    page_key_paths: Counter[str] = Counter()
    page_value_paths: Counter[str] = Counter()
    qids_with_page_like_fields = 0
    evidence_count_hist: Counter[int] = Counter()
    max_evidence_per_doc_hist: Counter[int] = Counter()
    support_doc_count_hist: Counter[int] = Counter()
    qtype_summary_counts: dict[str, list[int]] = defaultdict(list)
    qtype_multi_evidence_same_doc: Counter[str] = Counter()
    examples_page_like: list[dict[str, Any]] = []
    examples_multi_evidence_same_doc: list[dict[str, Any]] = []

    max_evidence_per_doc_values: list[int] = []
    qids_with_multi_evidence_same_doc = 0
    qids_with_3plus_evidence_same_doc = 0
    qids_with_5plus_evidence_same_doc = 0

    for row in rows:
        qid = str(row.get("qid", "")).strip()
        qtype = question_type(row)
        docs = supporting_doc_ids(row)
        support_doc_count_hist[len(docs)] += 1

        page_hits = scan_page_like_fields(row)
        if page_hits:
            qids_with_page_like_fields += 1
            for hit in page_hits:
                if hit["kind"] == "key":
                    page_key_paths[hit["path"]] += 1
                else:
                    page_value_paths[hit["path"]] += 1
            if len(examples_page_like) < example_limit:
                examples_page_like.append({"qid": qid, "question_type": qtype, "hits": page_hits[:10]})

        counts, typed_counts = evidence_doc_counts(row)
        total_evidence = sum(counts.values())
        max_per_doc = max(counts.values(), default=0)
        evidence_count_hist[total_evidence] += 1
        max_evidence_per_doc_hist[max_per_doc] += 1
        max_evidence_per_doc_values.append(max_per_doc)
        qtype_summary_counts[qtype].append(max_per_doc)

        if max_per_doc > 1:
            qids_with_multi_evidence_same_doc += 1
            qtype_multi_evidence_same_doc[qtype] += 1
            if len(examples_multi_evidence_same_doc) < example_limit:
                examples_multi_evidence_same_doc.append(
                    {
                        "qid": qid,
                        "question_type": qtype,
                        "supporting_doc_ids": docs,
                        "evidence_doc_counts": dict(sorted(counts.items())),
                        "typed_evidence_counts": dict(sorted(typed_counts.items())),
                    }
                )
        if max_per_doc >= 3:
            qids_with_3plus_evidence_same_doc += 1
        if max_per_doc >= 5:
            qids_with_5plus_evidence_same_doc += 1

    qtype_summary = []
    for qtype, values in sorted(qtype_summary_counts.items()):
        multi = int(qtype_multi_evidence_same_doc.get(qtype, 0))
        qtype_summary.append(
            {
                "question_type": qtype,
                "qid_count": len(values),
                "mean_max_evidence_per_doc": round(float(fmean(values)), 4) if values else 0.0,
                "max_evidence_per_doc": max(values, default=0),
                "qids_with_multi_evidence_same_doc": multi,
                "multi_evidence_same_doc_fraction": round(ratio(multi, len(values)), 6),
            }
        )

    return {
        "label": label,
        "path": str(path),
        "qid_count": len(rows),
        "qids_with_page_like_fields": qids_with_page_like_fields,
        "qids_with_page_like_fields_fraction": round(ratio(qids_with_page_like_fields, len(rows)), 6),
        "page_like_key_paths": dict(page_key_paths.most_common(50)),
        "page_like_value_paths": dict(page_value_paths.most_common(50)),
        "support_doc_count_hist": {str(k): v for k, v in sorted(support_doc_count_hist.items())},
        "evidence_object_count_hist": {str(k): v for k, v in sorted(evidence_count_hist.items())},
        "max_evidence_objects_per_doc_hist": {
            str(k): v for k, v in sorted(max_evidence_per_doc_hist.items())
        },
        "mean_max_evidence_objects_per_doc": round(float(fmean(max_evidence_per_doc_values)), 6)
        if max_evidence_per_doc_values
        else 0.0,
        "max_evidence_objects_per_doc": max(max_evidence_per_doc_values, default=0),
        "qids_with_multi_evidence_same_doc": qids_with_multi_evidence_same_doc,
        "qids_with_multi_evidence_same_doc_fraction": round(
            ratio(qids_with_multi_evidence_same_doc, len(rows)), 6
        ),
        "qids_with_3plus_evidence_same_doc": qids_with_3plus_evidence_same_doc,
        "qids_with_3plus_evidence_same_doc_fraction": round(
            ratio(qids_with_3plus_evidence_same_doc, len(rows)), 6
        ),
        "qids_with_5plus_evidence_same_doc": qids_with_5plus_evidence_same_doc,
        "qids_with_5plus_evidence_same_doc_fraction": round(
            ratio(qids_with_5plus_evidence_same_doc, len(rows)), 6
        ),
        "question_type_summary": qtype_summary,
        "examples_page_like": examples_page_like,
        "examples_multi_evidence_same_doc": examples_multi_evidence_same_doc,
    }


def write_md(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines: list[str] = ["# Raw MMQA Evidence Structure Audit", ""]
    lines.extend(
        [
            "| split | qids | qids with page-like raw fields | qids with >1 evidence object same doc | qids with >=3 evidence objects same doc | qids with >=5 evidence objects same doc | max evidence objects/doc |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in summaries:
        lines.append(
            "| {label} | {qid_count} | {qids_with_page_like_fields} ({qids_with_page_like_fields_fraction:.1%}) | "
            "{qids_with_multi_evidence_same_doc} ({qids_with_multi_evidence_same_doc_fraction:.1%}) | "
            "{qids_with_3plus_evidence_same_doc} ({qids_with_3plus_evidence_same_doc_fraction:.1%}) | "
            "{qids_with_5plus_evidence_same_doc} ({qids_with_5plus_evidence_same_doc_fraction:.1%}) | "
            "{max_evidence_objects_per_doc} |".format(**item)
        )

    for item in summaries:
        lines.extend(["", f"## {item['label']}", ""])
        lines.extend(["### Max Evidence Objects Per Document", ""])
        lines.extend(["| max evidence objects/doc | qids |", "| ---: | ---: |"])
        for key, value in item["max_evidence_objects_per_doc_hist"].items():
            lines.append(f"| {key} | {value} |")

        lines.extend(["", "### Page-Like Raw Fields", ""])
        if item["page_like_key_paths"] or item["page_like_value_paths"]:
            lines.extend(["| path | count | type |", "| --- | ---: | --- |"])
            for key, value in item["page_like_key_paths"].items():
                lines.append(f"| `{key}` | {value} | key |")
            for key, value in item["page_like_value_paths"].items():
                lines.append(f"| `{key}` | {value} | value |")
        else:
            lines.append("No page-like keys or page UID values were found in the raw rows.")

        lines.extend(["", "### Question Types With Same-Doc Evidence Multiplicity", ""])
        lines.extend(
            [
                "| question type | qids | mean max evidence/doc | max evidence/doc | qids with >1 evidence same doc |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in sorted(
            item["question_type_summary"],
            key=lambda value: (-value["qids_with_multi_evidence_same_doc"], value["question_type"]),
        )[:30]:
            lines.append(
                "| {question_type} | {qid_count} | {mean_max_evidence_per_doc:.3f} | "
                "{max_evidence_per_doc} | {qids_with_multi_evidence_same_doc} "
                "({multi_evidence_same_doc_fraction:.1%}) |".format(**row)
            )

        if item["examples_multi_evidence_same_doc"]:
            lines.extend(["", "### Example QIDs With Multiple Evidence Objects in One Doc", ""])
            for row in item["examples_multi_evidence_same_doc"][:10]:
                lines.append(
                    f"- `{row['qid']}` ({row['question_type']}): "
                    f"evidence_doc_counts={row['evidence_doc_counts']}"
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fields = [
        "label",
        "qid_count",
        "qids_with_page_like_fields",
        "qids_with_page_like_fields_fraction",
        "qids_with_multi_evidence_same_doc",
        "qids_with_multi_evidence_same_doc_fraction",
        "qids_with_3plus_evidence_same_doc",
        "qids_with_3plus_evidence_same_doc_fraction",
        "qids_with_5plus_evidence_same_doc",
        "qids_with_5plus_evidence_same_doc_fraction",
        "max_evidence_objects_per_doc",
        "mean_max_evidence_objects_per_doc",
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
        for label, path in map(parse_labeled_path, args.gold)
    ]

    print(
        "| split | qids | page_like_raw_qids | multi_evidence_same_doc | "
        "3plus_same_doc | 5plus_same_doc | max_evidence_doc |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in summaries:
        print(
            "| {label} | {qid_count} | {qids_with_page_like_fields} | "
            "{qids_with_multi_evidence_same_doc} | {qids_with_3plus_evidence_same_doc} | "
            "{qids_with_5plus_evidence_same_doc} | {max_evidence_objects_per_doc} |".format(
                **item
            )
        )

    if args.output_md:
        write_md(Path(args.output_md), summaries)
        print(f"saved_output_md={args.output_md}")
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"summaries": summaries}, indent=2) + "\n", encoding="utf-8")
        print(f"saved_output_json={args.output_json}")
    if args.output_csv:
        write_csv(Path(args.output_csv), summaries)
        print(f"saved_output_csv={args.output_csv}")


if __name__ == "__main__":
    main()
