#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

from m3docrag.retrieval import ColPaliRetrievalModel
from scripts.rerank_target_docs_visual_aware import (
    QUERY_TOKEN_FILTER_CHOICES,
    clean_token_label,
    load_splice_query_axis_classes,
    resolve_model_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two query-token label exports after aligning them onto the exact "
            "ColPali query tokenization used by the visual-aware reranker."
        )
    )
    parser.add_argument("--gold", required=True, help="Path to MMQA_<split>.jsonl.")
    parser.add_argument(
        "--query-label-a",
        required=True,
        help="First query-token label file, e.g. union_relaxed_v2.",
    )
    parser.add_argument(
        "--query-label-b",
        required=True,
        help="Second query-token label file, e.g. union_relaxed_v6_fulltrainlex_v2.",
    )
    parser.add_argument(
        "--name-a",
        default="a",
        help="Display name for --query-label-a in the output.",
    )
    parser.add_argument(
        "--name-b",
        default="b",
        help="Display name for --query-label-b in the output.",
    )
    parser.add_argument(
        "--question-type",
        action="append",
        default=[],
        help="Optional MMQA question type filter. Pass multiple times. Defaults to all qtypes.",
    )
    parser.add_argument(
        "--query_token_filter",
        default="full",
        choices=QUERY_TOKEN_FILTER_CHOICES,
        help="ColPali query-token filter used before label alignment.",
    )
    parser.add_argument("--retrieval_model_name_or_path", default="colpaligemma-3b-pt-448-base")
    parser.add_argument("--retrieval_adapter_model_name_or_path", default="colpali-v1.2")
    parser.add_argument(
        "--max-qids",
        type=int,
        default=0,
        help="Optional cap for smoke tests.",
    )
    parser.add_argument(
        "--report-topn",
        type=int,
        default=25,
        help="How many changed qids to keep in the JSON summary.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional JSON summary output path.",
    )
    return parser.parse_args()


def load_gold_rows(path: Path, allowed_qtypes: set[str], max_qids: int) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            qtype = str(row.get("metadata", {}).get("type", "")).strip()
            if allowed_qtypes and qtype not in allowed_qtypes:
                continue
            rows.append(row)
            if max_qids > 0 and len(rows) >= max_qids:
                break
    if not rows:
        raise ValueError("No matching qids found in the gold file.")
    return rows


def visual_indices(classes: list[str]) -> list[int]:
    return [idx for idx, value in enumerate(classes) if value == "visual"]


def visual_tokens(classes: list[str], query_token_labels: list[str]) -> list[str]:
    return [query_token_labels[idx] for idx, value in enumerate(classes) if value == "visual"]


def compare_class_lists(classes_a: list[str], classes_b: list[str]) -> dict:
    visual_a = set(visual_indices(classes_a))
    visual_b = set(visual_indices(classes_b))
    return {
        "identical_class_sequence": classes_a == classes_b,
        "identical_visual_index_set": visual_a == visual_b,
        "only_in_a": sorted(visual_a - visual_b),
        "only_in_b": sorted(visual_b - visual_a),
    }


def main() -> None:
    args = parse_args()

    gold_rows = load_gold_rows(
        path=Path(args.gold),
        allowed_qtypes=set(args.question_type),
        max_qids=args.max_qids,
    )

    retrieval_model = ColPaliRetrievalModel(
        backbone_name_or_path=resolve_model_path(args.retrieval_model_name_or_path),
        adapter_name_or_path=resolve_model_path(args.retrieval_adapter_model_name_or_path),
    )

    summary_counts = Counter()
    qtype_counts = Counter()
    qtype_changed_counts = Counter()
    qtype_gained_counts = Counter()
    qtype_lost_counts = Counter()
    changed_rows: list[dict] = []

    for row_idx, row in enumerate(gold_rows, start=1):
        qid = str(row["qid"]).strip()
        qtype = str(row.get("metadata", {}).get("type", "")).strip()
        question = str(row["question"])

        qtype_counts[qtype] += 1
        summary_counts["total_qids"] += 1

        query_meta = retrieval_model.encode_query_with_metadata(
            query=question,
            to_cpu=True,
            query_token_filter=args.query_token_filter,
        )
        query_raw_tokens = query_meta["raw_tokens"]
        query_token_labels = [clean_token_label(tok) for tok in query_raw_tokens]

        classes_a = load_splice_query_axis_classes(
            query_labels_path=args.query_label_a,
            qid=qid,
            query_token_labels=query_token_labels,
            query_raw_tokens=query_raw_tokens,
        )
        classes_b = load_splice_query_axis_classes(
            query_labels_path=args.query_label_b,
            qid=qid,
            query_token_labels=query_token_labels,
            query_raw_tokens=query_raw_tokens,
        )

        diff = compare_class_lists(classes_a, classes_b)
        visual_count_a = len(visual_indices(classes_a))
        visual_count_b = len(visual_indices(classes_b))

        if diff["identical_class_sequence"]:
            summary_counts["identical_class_sequence_count"] += 1
        else:
            summary_counts["different_class_sequence_count"] += 1
            qtype_changed_counts[qtype] += 1

        if diff["identical_visual_index_set"]:
            summary_counts["identical_visual_index_set_count"] += 1
        else:
            summary_counts["different_visual_index_set_count"] += 1

        if visual_count_b > visual_count_a:
            summary_counts["gained_visual_qid_count"] += 1
            qtype_gained_counts[qtype] += 1
        elif visual_count_b < visual_count_a:
            summary_counts["lost_visual_qid_count"] += 1
            qtype_lost_counts[qtype] += 1
        else:
            summary_counts["same_visual_count_qid_count"] += 1

        if not diff["identical_class_sequence"]:
            changed_rows.append(
                {
                    "qid": qid,
                    "question_type": qtype,
                    "question": question,
                    "visual_indices_" + args.name_a: visual_indices(classes_a),
                    "visual_indices_" + args.name_b: visual_indices(classes_b),
                    "visual_tokens_" + args.name_a: visual_tokens(classes_a, query_token_labels),
                    "visual_tokens_" + args.name_b: visual_tokens(classes_b, query_token_labels),
                    "only_in_" + args.name_a: diff["only_in_a"],
                    "only_in_" + args.name_b: diff["only_in_b"],
                }
            )

        print(
            f"[{row_idx}/{len(gold_rows)}] {qid} "
            f"{args.name_a}_visual={visual_count_a} {args.name_b}_visual={visual_count_b} "
            f"same_classes={diff['identical_class_sequence']}"
        )

    changed_rows.sort(
        key=lambda item: (
            len(item["only_in_" + args.name_b]) + len(item["only_in_" + args.name_a]),
            item["qid"],
        ),
        reverse=True,
    )

    qtype_summary = []
    for qtype in sorted(qtype_counts):
        qtype_summary.append(
            {
                "question_type": qtype,
                "total_qids": qtype_counts[qtype],
                "changed_class_sequence_count": qtype_changed_counts[qtype],
                "gained_visual_qid_count": qtype_gained_counts[qtype],
                "lost_visual_qid_count": qtype_lost_counts[qtype],
            }
        )

    summary = {
        "query_token_filter": args.query_token_filter,
        "query_label_a": args.query_label_a,
        "query_label_b": args.query_label_b,
        "name_a": args.name_a,
        "name_b": args.name_b,
        "question_types": args.question_type,
        "counts": dict(summary_counts),
        "question_type_summary": qtype_summary,
        "changed_qids_topn": changed_rows[: args.report_topn],
    }

    print("summary_counts:", json.dumps(summary["counts"], ensure_ascii=False, sort_keys=True))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("saved_summary:", output_path)


if __name__ == "__main__":
    main()
