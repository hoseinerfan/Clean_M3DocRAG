#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_RECALL_KS = [1, 4, 20, 100, 1000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare exact/compact MaxSim retrieval predictions by question type or other "
            "metadata fields. Works with converted MMDocIR-style MMQA_dev.jsonl files."
        )
    )
    parser.add_argument("--gold", required=True, help="Converted MMQA_dev.jsonl")
    parser.add_argument("--exact-pred", default="", help="Exact/full MaxSim prediction JSON.")
    parser.add_argument("--compact-pred", default="", help="Compact MaxSim prediction JSON.")
    parser.add_argument("--exact-label", default="exactmaxsim")
    parser.add_argument("--compact-label", default="compactmaxsim")
    parser.add_argument(
        "--group-field",
        action="append",
        default=[],
        help=(
            "Metadata path to group by. Repeat for several dimensions. Examples: "
            "metadata.type, metadata.domain, metadata.query_types, metadata.repo_slug. "
            "Default: auto."
        ),
    )
    parser.add_argument("--hit-k", type=int, default=4, help="K used for hit/miss struggle rates.")
    parser.add_argument("--recall-k", dest="recall_ks", type=int, nargs="+", default=DEFAULT_RECALL_KS)
    parser.add_argument("--min-count", type=int, default=5, help="Hide groups with fewer qids.")
    parser.add_argument("--top-groups", type=int, default=40)
    parser.add_argument(
        "--sort-by",
        default="auto",
        help=(
            "Metric key to sort by, or auto. Useful keys include "
            "compactmaxsim_page_hit_at_4, exactmaxsim_page_hit_at_4, "
            "both_page_miss_at_4, compact_loses_page_at_4, compact_recovers_page_at_4."
        ),
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Sort descending. Default auto sorts hit rates ascending and miss/recover/loss rates descending.",
    )
    parser.add_argument("--examples-per-group", type=int, default=3)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--output-md", default="")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_prediction(path: str) -> dict[str, dict]:
    if not path:
        return {}
    pred_path = Path(path)
    text = pred_path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    if text[0] == "{":
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise TypeError(f"Prediction JSON must be an object keyed by qid: {pred_path}")
        return {str(qid): row for qid, row in payload.items()}

    rows: dict[str, dict] = {}
    with pred_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("qid", "")).strip()
            if qid:
                rows[qid] = row
    return rows


def get_path(row: dict, path: str) -> Any:
    current: Any = row
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def normalize_group_values(value: Any) -> list[str]:
    if value is None:
        return ["UNKNOWN"]
    if isinstance(value, list):
        values = [str(item).strip() for item in value if str(item).strip()]
        return values or ["UNKNOWN"]
    value_text = str(value).strip()
    return [value_text or "UNKNOWN"]


def auto_group_fields(row: dict) -> list[str]:
    metadata = row.get("metadata", {})
    source = str(metadata.get("source", metadata.get("type", ""))).lower()
    if "vidore" in source:
        if metadata.get("query_types"):
            return ["metadata.query_types"]
        if metadata.get("query_type_for_generation"):
            return ["metadata.query_type_for_generation"]
        if metadata.get("query_format"):
            return ["metadata.query_format"]
        return ["metadata.repo_slug"]
    if metadata.get("type"):
        return ["metadata.type"]
    return ["metadata.source"]


def group_keys(row: dict, requested_fields: list[str]) -> list[tuple[str, str]]:
    fields = requested_fields or ["auto"]
    keys: list[tuple[str, str]] = []
    for field in fields:
        expanded_fields = auto_group_fields(row) if field == "auto" else [field]
        for expanded_field in expanded_fields:
            for value in normalize_group_values(get_path(row, expanded_field)):
                keys.append((expanded_field, value))
    return keys or [("UNKNOWN", "UNKNOWN")]


def gold_page_uids(row: dict) -> set[str]:
    metadata = row.get("metadata", {})
    uids = {
        str(value).strip()
        for value in metadata.get("gold_page_uids", [])
        if str(value).strip()
    }
    for ctx in row.get("supporting_context", []):
        doc_id = str(ctx.get("doc_id", "")).strip()
        page_idx = ctx.get("page_idx", ctx.get("page_id"))
        if doc_id and page_idx is not None:
            uids.add(f"{doc_id}_page{int(page_idx)}")
    return uids


def gold_doc_ids(row: dict) -> set[str]:
    return {
        str(ctx.get("doc_id", "")).strip()
        for ctx in row.get("supporting_context", [])
        if str(ctx.get("doc_id", "")).strip()
    }


def ranked_pages(pred_row: dict) -> list[str]:
    rows = pred_row.get("page_retrieval_results", [])
    ranked: list[str] = []
    for item in rows:
        if not isinstance(item, list) or len(item) < 2:
            continue
        ranked.append(f"{str(item[0])}_page{int(item[1])}")
    return ranked


def ranked_docs(pred_row: dict) -> list[str]:
    docs: list[str] = []
    seen: set[str] = set()
    for item in pred_row.get("page_retrieval_results", []):
        if not isinstance(item, list) or not item:
            continue
        doc_id = str(item[0])
        if doc_id not in seen:
            seen.add(doc_id)
            docs.append(doc_id)
    return docs


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / len(gold)


def first_rank(ranked: list[str], gold: set[str]) -> int | None:
    for idx, item in enumerate(ranked, start=1):
        if item in gold:
            return idx
    return None


def score_prediction(pred_row: dict | None, gold_row: dict, recall_ks: list[int], hit_k: int) -> dict:
    if pred_row is None:
        return {
            "present": False,
            "page_first_rank": None,
            "doc_first_rank": None,
            "page_hit_at_k": False,
            "doc_hit_at_k": False,
            "page_recall_at_k": {str(k): 0.0 for k in recall_ks},
            "doc_recall_at_k": {str(k): 0.0 for k in recall_ks},
        }

    page_gold = gold_page_uids(gold_row)
    doc_gold = gold_doc_ids(gold_row)
    pages = ranked_pages(pred_row)
    docs = ranked_docs(pred_row)
    page_rank = first_rank(pages, page_gold)
    doc_rank = first_rank(docs, doc_gold)
    return {
        "present": True,
        "page_first_rank": page_rank,
        "doc_first_rank": doc_rank,
        "page_hit_at_k": page_rank is not None and page_rank <= hit_k,
        "doc_hit_at_k": doc_rank is not None and doc_rank <= hit_k,
        "page_recall_at_k": {str(k): recall_at_k(pages, page_gold, k) for k in recall_ks},
        "doc_recall_at_k": {str(k): recall_at_k(docs, doc_gold, k) for k in recall_ks},
    }


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def summarize_group(
    *,
    group_by: str,
    group: str,
    rows: list[dict],
    labels: list[str],
    hit_k: int,
    recall_ks: list[int],
    examples_per_group: int,
) -> dict:
    summary: dict[str, Any] = {
        "group_by": group_by,
        "group": group,
        "n_qids": len(rows),
    }
    for label in labels:
        label_rows = [row for row in rows if row["runs"][label]["present"]]
        summary[f"{label}_present_qids"] = len(label_rows)
        summary[f"{label}_page_hit_at_{hit_k}"] = mean(
            [float(row["runs"][label]["page_hit_at_k"]) for row in label_rows]
        )
        summary[f"{label}_doc_hit_at_{hit_k}"] = mean(
            [float(row["runs"][label]["doc_hit_at_k"]) for row in label_rows]
        )
        for k in recall_ks:
            key = str(k)
            summary[f"{label}_page_recall_at_{k}"] = mean(
                [float(row["runs"][label]["page_recall_at_k"][key]) for row in label_rows]
            )
            summary[f"{label}_doc_recall_at_{k}"] = mean(
                [float(row["runs"][label]["doc_recall_at_k"][key]) for row in label_rows]
            )

    if len(labels) == 2:
        exact_label, compact_label = labels
        paired = [
            row
            for row in rows
            if row["runs"][exact_label]["present"] and row["runs"][compact_label]["present"]
        ]
        if paired:
            exact_page_hits = [row["runs"][exact_label]["page_hit_at_k"] for row in paired]
            compact_page_hits = [row["runs"][compact_label]["page_hit_at_k"] for row in paired]
            exact_doc_hits = [row["runs"][exact_label]["doc_hit_at_k"] for row in paired]
            compact_doc_hits = [row["runs"][compact_label]["doc_hit_at_k"] for row in paired]
            summary[f"{compact_label}_minus_{exact_label}_page_hit_at_{hit_k}"] = (
                mean([float(value) for value in compact_page_hits])
                - mean([float(value) for value in exact_page_hits])
            )
            summary[f"{compact_label}_minus_{exact_label}_doc_hit_at_{hit_k}"] = (
                mean([float(value) for value in compact_doc_hits])
                - mean([float(value) for value in exact_doc_hits])
            )
            summary[f"both_page_miss_at_{hit_k}"] = mean(
                [float((not e_hit) and (not c_hit)) for e_hit, c_hit in zip(exact_page_hits, compact_page_hits)]
            )
            summary[f"both_doc_miss_at_{hit_k}"] = mean(
                [float((not e_hit) and (not c_hit)) for e_hit, c_hit in zip(exact_doc_hits, compact_doc_hits)]
            )
            summary[f"{compact_label}_loses_page_at_{hit_k}"] = mean(
                [float(e_hit and not c_hit) for e_hit, c_hit in zip(exact_page_hits, compact_page_hits)]
            )
            summary[f"{compact_label}_recovers_page_at_{hit_k}"] = mean(
                [float((not e_hit) and c_hit) for e_hit, c_hit in zip(exact_page_hits, compact_page_hits)]
            )

            both_page_miss_examples = [
                row
                for row in paired
                if not row["runs"][exact_label]["page_hit_at_k"]
                and not row["runs"][compact_label]["page_hit_at_k"]
            ]
            both_page_miss_examples.sort(
                key=lambda row: (
                    row["runs"][compact_label]["page_first_rank"] is not None,
                    row["runs"][compact_label]["page_first_rank"] or 10**9,
                    row["runs"][exact_label]["page_first_rank"] or 10**9,
                )
            )
            summary["example_both_page_misses"] = [
                {
                    "qid": row["qid"],
                    "question": row["question"],
                    f"{exact_label}_page_rank": row["runs"][exact_label]["page_first_rank"],
                    f"{compact_label}_page_rank": row["runs"][compact_label]["page_first_rank"],
                    f"{exact_label}_doc_rank": row["runs"][exact_label]["doc_first_rank"],
                    f"{compact_label}_doc_rank": row["runs"][compact_label]["doc_first_rank"],
                }
                for row in both_page_miss_examples[:examples_per_group]
            ]
    return summary


def make_markdown_table(summaries: list[dict], labels: list[str], hit_k: int) -> str:
    headers = ["group_by", "group", "n"]
    for label in labels:
        headers.extend([f"{label} page@{hit_k}", f"{label} doc@{hit_k}"])
    if len(labels) == 2:
        exact_label, compact_label = labels
        headers.extend(
            [
                f"{compact_label}-{exact_label} page@{hit_k}",
                f"both page miss@{hit_k}",
                f"{compact_label} loses page@{hit_k}",
                f"{compact_label} recovers page@{hit_k}",
            ]
        )

    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for item in summaries:
        row = [item["group_by"], item["group"], item["n_qids"]]
        for label in labels:
            row.extend(
                [
                    item.get(f"{label}_page_hit_at_{hit_k}", 0.0),
                    item.get(f"{label}_doc_hit_at_{hit_k}", 0.0),
                ]
            )
        if len(labels) == 2:
            exact_label, compact_label = labels
            row.extend(
                [
                    item.get(f"{compact_label}_minus_{exact_label}_page_hit_at_{hit_k}", 0.0),
                    item.get(f"both_page_miss_at_{hit_k}", 0.0),
                    item.get(f"{compact_label}_loses_page_at_{hit_k}", 0.0),
                    item.get(f"{compact_label}_recovers_page_at_{hit_k}", 0.0),
                ]
            )
        lines.append("| " + " | ".join(fmt(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def choose_sort(args: argparse.Namespace, labels: list[str]) -> tuple[str, bool]:
    if args.sort_by != "auto":
        return args.sort_by, args.descending
    if len(labels) == 2:
        return f"both_page_miss_at_{args.hit_k}", True
    return f"{labels[0]}_page_hit_at_{args.hit_k}", False


def main() -> None:
    args = parse_args()
    if not args.exact_pred and not args.compact_pred:
        raise ValueError("Provide --exact-pred, --compact-pred, or both.")
    if args.hit_k not in args.recall_ks:
        args.recall_ks = sorted(set(args.recall_ks + [args.hit_k]))

    gold_rows = read_jsonl(Path(args.gold))
    exact_pred = read_prediction(args.exact_pred)
    compact_pred = read_prediction(args.compact_pred)

    labels: list[str] = []
    predictions: dict[str, dict[str, dict]] = {}
    if exact_pred:
        labels.append(args.exact_label)
        predictions[args.exact_label] = exact_pred
    if compact_pred:
        labels.append(args.compact_label)
        predictions[args.compact_label] = compact_pred

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for gold_row in gold_rows:
        qid = str(gold_row["qid"])
        runs = {
            label: score_prediction(predictions[label].get(qid), gold_row, args.recall_ks, args.hit_k)
            for label in labels
        }
        if not any(run["present"] for run in runs.values()):
            continue
        item = {
            "qid": qid,
            "question": gold_row.get("question", ""),
            "runs": runs,
        }
        for group_by, group in group_keys(gold_row, args.group_field):
            groups[(group_by, group)].append(item)

    summaries = [
        summarize_group(
            group_by=group_by,
            group=group,
            rows=rows,
            labels=labels,
            hit_k=args.hit_k,
            recall_ks=args.recall_ks,
            examples_per_group=args.examples_per_group,
        )
        for (group_by, group), rows in groups.items()
        if len(rows) >= args.min_count
    ]
    sort_key, descending = choose_sort(args, labels)
    summaries.sort(key=lambda item: item.get(sort_key, 0.0), reverse=descending)
    if args.top_groups > 0:
        summaries = summaries[: args.top_groups]

    payload = {
        "gold": args.gold,
        "exact_pred": args.exact_pred,
        "compact_pred": args.compact_pred,
        "labels": labels,
        "hit_k": args.hit_k,
        "recall_ks": args.recall_ks,
        "group_fields": args.group_field or ["auto"],
        "sort_by": sort_key,
        "descending": descending,
        "groups": summaries,
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"saved_json={output_path}")

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for item in summaries for key in item if key != "example_both_page_misses"})
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for item in summaries:
                writer.writerow({key: item.get(key, "") for key in fieldnames})
        print(f"saved_csv={output_path}")

    markdown = make_markdown_table(summaries, labels, args.hit_k)
    if args.output_md:
        output_path = Path(args.output_md)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
        print(f"saved_md={output_path}")

    print(markdown)


if __name__ == "__main__":
    main()
