#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_mmqa_pseudo_page_labels as ppl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export per-question MMQA evidence metadata for M3DocVQA. The output "
            "records support-document counts, document-grounded evidence units, "
            "evidence units per document, and optional pseudo-page label counts."
        )
    )
    parser.add_argument("--gold", action="append", required=True, help="LABEL=MMQA_*.jsonl")
    parser.add_argument("--mmqa-texts-jsonl", default="")
    parser.add_argument("--mmqa-tables-jsonl", default="")
    parser.add_argument("--mmqa-images-jsonl", default="")
    parser.add_argument("--id-url-mapping-jsonl", default="")
    parser.add_argument(
        "--augmented-gold",
        action="append",
        default=[],
        help=(
            "Optional LABEL=augmented_gold.jsonl with metadata.gold_page_uids. "
            "LABEL should match a --gold label, e.g. dev=..."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", default="mmqa_evidence_metadata")
    parser.add_argument("--write-pretty-json", action="store_true")
    return parser.parse_args()


def parse_labeled_path(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, raw_path = spec.split("=", 1)
        return label.strip() or Path(raw_path).stem, Path(raw_path)
    path = Path(spec)
    return path.stem, path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_by_qid(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        qid = str(row.get("qid", "")).strip()
        if qid:
            out[qid] = row
    return out


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}


def question_type(row: dict[str, Any]) -> str:
    return str(metadata(row).get("type", "UNKNOWN")).strip() or "UNKNOWN"


def answer_objects(row: dict[str, Any]) -> list[tuple[str, int, int | None, dict[str, Any]]]:
    out: list[tuple[str, int, int | None, dict[str, Any]]] = []
    for idx, answer in enumerate(row.get("answers", []) or []):
        if isinstance(answer, dict):
            out.append(("answer", idx, None, answer))

    meta = metadata(row)
    for group_idx, group in enumerate(meta.get("intermediate_answers", []) or []):
        if not isinstance(group, list):
            continue
        for idx, answer in enumerate(group):
            if isinstance(answer, dict):
                out.append(("intermediate_answer", idx, group_idx, answer))
    return out


def support_doc_rows(
    row: dict[str, Any],
    *,
    texts_by_id: dict[str, dict[str, Any]],
    tables_by_id: dict[str, dict[str, Any]],
    images_by_id: dict[str, dict[str, Any]],
    id_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ctx in row.get("supporting_context", []) or []:
        if not isinstance(ctx, dict):
            continue
        doc_id = str(ctx.get("doc_id", "")).strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        side_row = texts_by_id.get(doc_id) or tables_by_id.get(doc_id) or images_by_id.get(doc_id) or {}
        rows.append(
            {
                "doc_id": doc_id,
                "doc_part": str(ctx.get("doc_part", "") or ""),
                "title": ppl.doc_title_from_map(doc_id, side_row, id_map) if id_map else str(side_row.get("title", "") or ""),
                "url": str(id_map.get(doc_id, {}).get("url", "") if id_map else side_row.get("url", "") or ""),
            }
        )
    return rows


def table_cell_text(table: dict[str, Any], row_idx: int, col_idx: int) -> str:
    try:
        return ppl.table_cell_text(table, row_idx, col_idx)
    except Exception:
        return ""


def add_unit(
    units: list[dict[str, Any]],
    *,
    qid: str,
    unit_type: str,
    doc_id: str,
    answer_scope: str,
    answer_index: int,
    intermediate_group_index: int | None,
    text: str = "",
    row_idx: int | None = None,
    col_idx: int | None = None,
    modality: str = "",
) -> None:
    doc_id = str(doc_id or "").strip()
    if not doc_id:
        return
    unit_id = f"{qid}::evidence{len(units)}"
    unit = {
        "unit_id": unit_id,
        "type": unit_type,
        "doc_id": doc_id,
        "answer_scope": answer_scope,
        "answer_index": int(answer_index),
        "intermediate_group_index": intermediate_group_index,
        "modality": modality,
        "text": str(text or ""),
    }
    if row_idx is not None:
        unit["row_idx"] = int(row_idx)
    if col_idx is not None:
        unit["col_idx"] = int(col_idx)
    units.append(unit)


def evidence_units_for_row(
    row: dict[str, Any],
    *,
    tables_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    qid = str(row.get("qid", "")).strip()
    meta = metadata(row)
    table_id = str(meta.get("table_id", "") or "").strip()
    table = tables_by_id.get(table_id, {}) if table_id else {}
    units: list[dict[str, Any]] = []

    for answer_scope, answer_index, intermediate_group_index, answer in answer_objects(row):
        modality = str(answer.get("modality", "") or "")
        for instance in answer.get("text_instances", []) or []:
            if not isinstance(instance, dict):
                continue
            add_unit(
                units,
                qid=qid,
                unit_type="text_instance",
                doc_id=str(instance.get("doc_id", "") or ""),
                answer_scope=answer_scope,
                answer_index=answer_index,
                intermediate_group_index=intermediate_group_index,
                text=str(instance.get("text", "") or ""),
                modality=modality,
            )

        for instance in answer.get("image_instances", []) or []:
            if not isinstance(instance, dict):
                continue
            add_unit(
                units,
                qid=qid,
                unit_type="image_instance",
                doc_id=str(instance.get("doc_id", "") or ""),
                answer_scope=answer_scope,
                answer_index=answer_index,
                intermediate_group_index=intermediate_group_index,
                text=str(instance.get("text", "") or instance.get("title", "") or ""),
                modality=modality,
            )

        for pair in answer.get("table_indices", []) or []:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2 or not table_id:
                continue
            try:
                row_idx, col_idx = int(pair[0]), int(pair[1])
            except (TypeError, ValueError):
                continue
            add_unit(
                units,
                qid=qid,
                unit_type="table_cell",
                doc_id=table_id,
                answer_scope=answer_scope,
                answer_index=answer_index,
                intermediate_group_index=intermediate_group_index,
                text=table_cell_text(table, row_idx, col_idx),
                row_idx=row_idx,
                col_idx=col_idx,
                modality=modality,
            )

    return units


def pseudo_page_uids(row: dict[str, Any] | None) -> list[str]:
    if not row:
        return []
    meta = metadata(row)
    values = (
        meta.get("gold_page_uids")
        or meta.get("pseudo_gold_page_uids")
        or row.get("gold_page_uids")
        or row.get("pseudo_gold_page_uids")
        or []
    )
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        uid = str(value).strip()
        if uid and uid not in seen:
            seen.add(uid)
            out.append(uid)
    return out


def page_doc_id(page_uid: str) -> str:
    return str(page_uid).rsplit("_page", 1)[0] if "_page" in str(page_uid) else str(page_uid)


def hist_update(counter: Counter[int], value: int) -> None:
    counter[int(value)] += 1


def split_summary(rows: list[dict[str, Any]], split_label: str) -> dict[str, Any]:
    support_counts = [int(row["support_doc_count"]) for row in rows]
    evidence_counts = [int(row["evidence_unit_count"]) for row in rows]
    max_units_per_doc = [int(row["max_evidence_units_per_doc"]) for row in rows]
    pseudo_counts = [
        int(row["pseudo_page_label_count"])
        for row in rows
        if row.get("pseudo_page_label_count") is not None
    ]

    qtype_counts = Counter(str(row["question_type"]) for row in rows)
    modality_counts = Counter()
    support_hist: Counter[int] = Counter()
    evidence_hist: Counter[int] = Counter()
    max_per_doc_hist: Counter[int] = Counter()
    pseudo_hist: Counter[int] = Counter()
    qids_multi_evidence_same_doc = 0
    qids_evidence_count_gt_support_docs = 0
    qids_pseudo_count_matches_support_doc_count = 0
    qids_pseudo_count_matches_evidence_unit_count = 0
    qids_with_pseudo = 0

    for row in rows:
        for modality in row.get("answer_modalities", []):
            modality_counts[str(modality)] += 1
        hist_update(support_hist, int(row["support_doc_count"]))
        hist_update(evidence_hist, int(row["evidence_unit_count"]))
        hist_update(max_per_doc_hist, int(row["max_evidence_units_per_doc"]))
        if int(row["max_evidence_units_per_doc"]) > 1:
            qids_multi_evidence_same_doc += 1
        if int(row["evidence_unit_count"]) > int(row["support_doc_count"]):
            qids_evidence_count_gt_support_docs += 1
        pseudo_count = row.get("pseudo_page_label_count")
        if pseudo_count is not None:
            pseudo_count = int(pseudo_count)
            hist_update(pseudo_hist, pseudo_count)
            if pseudo_count > 0:
                qids_with_pseudo += 1
            if pseudo_count == int(row["support_doc_count"]):
                qids_pseudo_count_matches_support_doc_count += 1
            if pseudo_count == int(row["evidence_unit_count"]):
                qids_pseudo_count_matches_evidence_unit_count += 1

    return {
        "split": split_label,
        "qid_count": len(rows),
        "mean_support_doc_count": round(float(fmean(support_counts)), 6) if support_counts else 0.0,
        "mean_evidence_unit_count": round(float(fmean(evidence_counts)), 6) if evidence_counts else 0.0,
        "mean_max_evidence_units_per_doc": round(float(fmean(max_units_per_doc)), 6)
        if max_units_per_doc
        else 0.0,
        "max_support_doc_count": max(support_counts, default=0),
        "max_evidence_unit_count": max(evidence_counts, default=0),
        "max_evidence_units_per_doc": max(max_units_per_doc, default=0),
        "qids_with_multi_evidence_same_doc": qids_multi_evidence_same_doc,
        "qids_with_evidence_count_gt_support_doc_count": qids_evidence_count_gt_support_docs,
        "support_doc_count_hist": {str(k): v for k, v in sorted(support_hist.items())},
        "evidence_unit_count_hist": {str(k): v for k, v in sorted(evidence_hist.items())},
        "max_evidence_units_per_doc_hist": {str(k): v for k, v in sorted(max_per_doc_hist.items())},
        "question_type_counts": dict(sorted(qtype_counts.items())),
        "answer_modality_counts": dict(sorted(modality_counts.items())),
        "pseudo_page_label_count_hist": {str(k): v for k, v in sorted(pseudo_hist.items())},
        "pseudo_labeled_qid_count": qids_with_pseudo,
        "mean_pseudo_page_label_count": round(float(fmean(pseudo_counts)), 6)
        if pseudo_counts
        else None,
        "qids_pseudo_count_matches_support_doc_count": qids_pseudo_count_matches_support_doc_count
        if pseudo_counts
        else None,
        "qids_pseudo_count_matches_evidence_unit_count": qids_pseudo_count_matches_evidence_unit_count
        if pseudo_counts
        else None,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_pretty_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_summary_md(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines: list[str] = [
        "# MMQA Evidence Metadata Summary",
        "",
        "| split | qids | mean support docs | mean evidence units | qids with >1 evidence unit in same doc | evidence units > support docs | max evidence units/doc | pseudo count = support doc count | pseudo count = evidence unit count |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        pseudo_support_match = item.get("qids_pseudo_count_matches_support_doc_count")
        pseudo_support_match_text = "" if pseudo_support_match is None else str(pseudo_support_match)
        pseudo_evidence_match = item.get("qids_pseudo_count_matches_evidence_unit_count")
        pseudo_evidence_match_text = "" if pseudo_evidence_match is None else str(pseudo_evidence_match)
        lines.append(
            "| {split} | {qid_count} | {mean_support_doc_count:.3f} | "
            "{mean_evidence_unit_count:.3f} | {qids_with_multi_evidence_same_doc} | "
            "{qids_with_evidence_count_gt_support_doc_count} | {max_evidence_units_per_doc} | "
            "{pseudo_support_match_text} | {pseudo_evidence_match_text} |".format(
                **item,
                pseudo_support_match_text=pseudo_support_match_text,
                pseudo_evidence_match_text=pseudo_evidence_match_text,
            )
        )

    for item in summaries:
        lines.extend(
            [
                "",
                f"## {item['split']}",
                "",
                "### Support Document Count Histogram",
                "",
                "| support docs | qids |",
                "| ---: | ---: |",
            ]
        )
        for key, value in item["support_doc_count_hist"].items():
            lines.append(f"| {key} | {value} |")
        lines.extend(["", "### Evidence Unit Count Histogram", "", "| evidence units | qids |", "| ---: | ---: |"])
        for key, value in item["evidence_unit_count_hist"].items():
            lines.append(f"| {key} | {value} |")
        lines.extend(
            [
                "",
                "### Max Evidence Units Per Document Histogram",
                "",
                "| max units/doc | qids |",
                "| ---: | ---: |",
            ]
        )
        for key, value in item["max_evidence_units_per_doc_hist"].items():
            lines.append(f"| {key} | {value} |")
        if item["pseudo_page_label_count_hist"]:
            lines.extend(
                [
                    "",
                    "### Pseudo-Page Label Count Histogram",
                    "",
                    "| pseudo pages | qids |",
                    "| ---: | ---: |",
                ]
            )
            for key, value in item["pseudo_page_label_count_hist"].items():
                lines.append(f"| {key} | {value} |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_split_rows(
    *,
    split_label: str,
    gold_path: Path,
    augmented_rows_by_qid: dict[str, dict[str, Any]],
    texts_by_id: dict[str, dict[str, Any]],
    tables_by_id: dict[str, dict[str, Any]],
    images_by_id: dict[str, dict[str, Any]],
    id_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in load_jsonl(gold_path):
        qid = str(row.get("qid", "")).strip()
        if not qid:
            continue
        meta = metadata(row)
        supports = support_doc_rows(
            row,
            texts_by_id=texts_by_id,
            tables_by_id=tables_by_id,
            images_by_id=images_by_id,
            id_map=id_map,
        )
        support_doc_ids = [item["doc_id"] for item in supports]
        units = evidence_units_for_row(row, tables_by_id=tables_by_id)
        units_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for unit in units:
            units_by_doc[str(unit["doc_id"])].append(unit)
        units_per_doc = {doc_id: len(items) for doc_id, items in sorted(units_by_doc.items())}
        support_docs_with_no_units = [doc_id for doc_id in support_doc_ids if doc_id not in units_by_doc]
        non_support_evidence_docs = [
            doc_id for doc_id in sorted(units_by_doc) if doc_id not in set(support_doc_ids)
        ]

        pseudo_uids = pseudo_page_uids(augmented_rows_by_qid.get(qid))
        pseudo_docs = sorted({page_doc_id(uid) for uid in pseudo_uids})
        pseudo_page_count = len(pseudo_uids) if augmented_rows_by_qid else None
        support_count = len(support_doc_ids)
        evidence_count = len(units)
        max_units_per_doc = max(units_per_doc.values(), default=0)
        answer_modalities = sorted(
            {
                str(answer.get("modality", "")).strip()
                for _scope, _idx, _group, answer in answer_objects(row)
                if str(answer.get("modality", "")).strip()
            }
        )

        rows.append(
            {
                "split": split_label,
                "qid": qid,
                "question": row.get("question", ""),
                "question_type": question_type(row),
                "answer_modalities": answer_modalities,
                "support_doc_count": support_count,
                "support_docs": supports,
                "support_doc_ids": support_doc_ids,
                "evidence_unit_count": evidence_count,
                "evidence_units": units,
                "evidence_units_per_doc": units_per_doc,
                "max_evidence_units_per_doc": max_units_per_doc,
                "docs_with_multiple_evidence_units": [
                    doc_id for doc_id, count in units_per_doc.items() if count > 1
                ],
                "support_docs_with_no_document_grounded_evidence_units": support_docs_with_no_units,
                "non_support_evidence_doc_ids": non_support_evidence_docs,
                "evidence_count_minus_support_doc_count": evidence_count - support_count,
                "pseudo_page_label_count": pseudo_page_count,
                "pseudo_page_uids": pseudo_uids,
                "pseudo_page_doc_ids": pseudo_docs,
                "pseudo_page_count_matches_support_doc_count": (
                    pseudo_page_count == support_count if pseudo_page_count is not None else None
                ),
                "pseudo_page_count_matches_evidence_unit_count": (
                    pseudo_page_count == evidence_count if pseudo_page_count is not None else None
                ),
                "metadata_table_id": str(meta.get("table_id", "") or ""),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    texts_by_id = ppl.load_by_id(args.mmqa_texts_jsonl) if args.mmqa_texts_jsonl else {}
    tables_by_id = ppl.load_by_id(args.mmqa_tables_jsonl) if args.mmqa_tables_jsonl else {}
    images_by_id = ppl.load_by_id(args.mmqa_images_jsonl) if args.mmqa_images_jsonl else {}
    id_map = ppl.load_id_map(Path(args.id_url_mapping_jsonl)) if args.id_url_mapping_jsonl else {}

    augmented_by_label: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in args.augmented_gold:
        label, path = parse_labeled_path(spec)
        augmented_by_label[label] = load_by_qid(path)

    summaries: list[dict[str, Any]] = []
    for spec in args.gold:
        label, gold_path = parse_labeled_path(spec)
        rows = build_split_rows(
            split_label=label,
            gold_path=gold_path,
            augmented_rows_by_qid=augmented_by_label.get(label, {}),
            texts_by_id=texts_by_id,
            tables_by_id=tables_by_id,
            images_by_id=images_by_id,
            id_map=id_map,
        )
        out_jsonl = output_dir / f"{args.output_prefix}_{label}.jsonl"
        out_summary = output_dir / f"{args.output_prefix}_{label}.summary.json"
        write_jsonl(out_jsonl, rows)
        if args.write_pretty_json:
            write_pretty_json(output_dir / f"{args.output_prefix}_{label}.pretty.json", rows)
        summary = split_summary(rows, label)
        summary["output_jsonl"] = str(out_jsonl)
        out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        summaries.append(summary)
        print(f"saved_output_jsonl={out_jsonl}")
        print(f"saved_summary={out_summary}")
        print(
            f"{label}: qids={summary['qid_count']} "
            f"mean_support_docs={summary['mean_support_doc_count']:.3f} "
            f"mean_evidence_units={summary['mean_evidence_unit_count']:.3f} "
            f"multi_evidence_same_doc={summary['qids_with_multi_evidence_same_doc']}"
        )

    all_summary = {
        "gold": args.gold,
        "augmented_gold": args.augmented_gold,
        "mmqa_texts_jsonl": args.mmqa_texts_jsonl,
        "mmqa_tables_jsonl": args.mmqa_tables_jsonl,
        "mmqa_images_jsonl": args.mmqa_images_jsonl,
        "id_url_mapping_jsonl": args.id_url_mapping_jsonl,
        "splits": summaries,
    }
    all_summary_path = output_dir / f"{args.output_prefix}.summary.json"
    all_summary_path.write_text(json.dumps(all_summary, indent=2) + "\n", encoding="utf-8")
    write_summary_md(output_dir / f"{args.output_prefix}.summary.md", summaries)
    print(f"saved_combined_summary={all_summary_path}")
    print(f"saved_combined_summary_md={output_dir / f'{args.output_prefix}.summary.md'}")


if __name__ == "__main__":
    main()
