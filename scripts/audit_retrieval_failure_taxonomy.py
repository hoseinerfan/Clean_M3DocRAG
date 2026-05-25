#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from analyze_layout_evidence_gate import (
    DEFAULT_RECALL_KS,
    first_rank,
    gold_doc_ids,
    gold_page_uids,
    load_prediction,
    mean,
    metric_scores,
    page_doc,
    ranked_docs,
    ranked_pages,
    read_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit retrieval failures for one or more MMQA-style prediction files. "
            "The script categorizes page-hit failures into doc-miss, right-doc/wrong-page, "
            "boundary recoverable, same-document sibling, adjacent-page, and missing-pool buckets."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=3,
        metavar=("LABEL", "GOLD", "PREDICTION"),
        default=[],
        help="Run tuple. Repeat to compare datasets or methods.",
    )
    parser.add_argument("--gold", default="", help="Single-run gold JSONL.")
    parser.add_argument("--prediction", default="", help="Single-run prediction JSON.")
    parser.add_argument("--run-label", default="run")
    parser.add_argument("--hit-k", type=int, default=4)
    parser.add_argument(
        "--boundary-k",
        type=int,
        default=10,
        help="Upper rank treated as near-boundary/recoverable for page failures.",
    )
    parser.add_argument(
        "--adjacent-window",
        type=int,
        default=2,
        help="Same-document page distance used for adjacent sibling diagnostics.",
    )
    parser.add_argument("--recall-k", dest="recall_ks", type=int, nargs="+", default=DEFAULT_RECALL_KS)
    parser.add_argument("--topn", type=int, default=25)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-csv", default="")
    return parser.parse_args()


def page_index(uid: str) -> int | None:
    if "_page" not in uid:
        return None
    try:
        return int(uid.rsplit("_page", 1)[1])
    except ValueError:
        return None


def question_text(row: dict[str, Any]) -> str:
    for key in ("question", "query", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def query_cues(question: str) -> list[str]:
    q = question.lower()
    cues: list[str] = []
    if re.search(r"\d|\b(how many|how much|percentage|percent|ratio|total|sum|difference|average)\b", q):
        cues.append("numeric")
    if re.search(r"\b(table|figure|chart|graph|image|picture|plot|diagram|visual)\b", q):
        cues.append("visual_table")
    if re.search(r"\b(page|section|appendix|slide|chapter)\b", q):
        cues.append("page_locator")
    if re.search(r"\b(compare|versus|vs\.?|difference|between|higher|lower|more|less)\b", q):
        cues.append("comparison")
    if re.search(r"\b(why|how|explain|describe)\b", q):
        cues.append("reasoning")
    return cues or ["uncued"]


def rank_bucket(rank: int | None, hit_k: int, boundary_k: int) -> str:
    if rank is None:
        return "missing"
    if rank <= hit_k:
        return f"top{hit_k}"
    if rank <= boundary_k:
        return f"boundary_{hit_k + 1}_{boundary_k}"
    if rank <= 20:
        return "rank_11_20"
    if rank <= 50:
        return "rank_21_50"
    if rank <= 100:
        return "rank_51_100"
    return "rank_gt100"


def score_at(row: dict[str, Any], rank: int) -> float | None:
    items = row.get("page_retrieval_results", [])
    idx = rank - 1
    if idx < 0 or idx >= len(items):
        return None
    item = items[idx]
    if not isinstance(item, (list, tuple)) or len(item) < 3:
        return None
    try:
        return float(item[2])
    except (TypeError, ValueError):
        return None


def score_margin(row: dict[str, Any], left_rank: int, right_rank: int) -> float | None:
    left = score_at(row, left_rank)
    right = score_at(row, right_rank)
    if left is None or right is None:
        return None
    return left - right


def same_doc_top_pages(pred_pages: list[str], gold_docs: set[str], hit_k: int) -> list[str]:
    return [uid for uid in pred_pages[:hit_k] if page_doc(uid) in gold_docs]


def adjacent_top_pages(
    pred_pages: list[str],
    gold_pages: set[str],
    hit_k: int,
    adjacent_window: int,
) -> list[str]:
    gold_by_doc: dict[str, list[int]] = defaultdict(list)
    for uid in gold_pages:
        idx = page_index(uid)
        if idx is not None:
            gold_by_doc[page_doc(uid)].append(idx)
    adjacent: list[str] = []
    for uid in pred_pages[:hit_k]:
        idx = page_index(uid)
        if idx is None:
            continue
        if any(abs(idx - gold_idx) <= adjacent_window for gold_idx in gold_by_doc.get(page_doc(uid), [])):
            adjacent.append(uid)
    return adjacent


def primary_failure_category(
    *,
    page_rank: int | None,
    doc_rank: int | None,
    hit_k: int,
    boundary_k: int,
    same_doc_topk: list[str],
    adjacent_topk: list[str],
) -> str:
    page_hit = page_rank is not None and page_rank <= hit_k
    if page_hit:
        return "page_hit"
    doc_hit = doc_rank is not None and doc_rank <= hit_k
    if not doc_hit:
        if doc_rank is None:
            return "doc_missing_from_pool"
        return "doc_miss_topk"
    if page_rank is not None and hit_k < page_rank <= boundary_k:
        return "right_doc_boundary_page"
    if adjacent_topk:
        return "right_doc_adjacent_page"
    if same_doc_topk:
        return "right_doc_same_doc_sibling"
    if page_rank is None:
        return "right_doc_gold_page_missing_from_pool"
    return "right_doc_late_page"


def secondary_tags(
    *,
    page_rank: int | None,
    doc_rank: int | None,
    hit_k: int,
    boundary_k: int,
    same_doc_topk: list[str],
    adjacent_topk: list[str],
    gold_page_count: int,
    gold_doc_count: int,
) -> list[str]:
    tags: list[str] = []
    if gold_page_count > 1:
        tags.append("multi_gold_page")
    if gold_doc_count > 1:
        tags.append("multi_gold_doc")
    if doc_rank is not None and doc_rank <= hit_k:
        tags.append("doc_hit_topk")
    elif doc_rank is None:
        tags.append("doc_missing")
    else:
        tags.append("doc_late")
    tags.append(f"page_{rank_bucket(page_rank, hit_k, boundary_k)}")
    if same_doc_topk:
        tags.append("same_doc_in_topk")
    if adjacent_topk:
        tags.append("adjacent_page_in_topk")
    return tags


def metadata_value(row: dict[str, Any], key: str) -> str:
    value: Any = row.get("metadata", {})
    for part in key.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            value = None
            break
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def audit_run(
    *,
    run_label: str,
    gold_path: Path,
    prediction_path: Path,
    hit_k: int,
    boundary_k: int,
    adjacent_window: int,
    recall_ks: list[int],
) -> dict[str, Any]:
    gold = {str(row["qid"]): row for row in read_jsonl(gold_path)}
    prediction = load_prediction(prediction_path)
    cases: list[dict[str, Any]] = []
    page_recall: dict[int, list[float]] = defaultdict(list)
    doc_recall: dict[int, list[float]] = defaultdict(list)
    qids = sorted(set(gold) & set(prediction))
    for qid in qids:
        gold_row = gold[qid]
        pred_row = prediction[qid]
        gold_pages = gold_page_uids(gold_row)
        gold_docs = gold_doc_ids(gold_row)
        pages = ranked_pages(pred_row)
        docs = ranked_docs(pred_row)
        page_rank = first_rank(pages, gold_pages)
        doc_rank = first_rank(docs, gold_docs)
        scores = metric_scores(pred_row, gold_pages, gold_docs, recall_ks, hit_k)
        for k in recall_ks:
            page_recall[int(k)].append(float(scores.get(f"page_recall@{k}", 0.0)))
            doc_recall[int(k)].append(float(scores.get(f"doc_recall@{k}", 0.0)))
        same_doc_topk = same_doc_top_pages(pages, gold_docs, hit_k)
        adjacent_topk = adjacent_top_pages(pages, gold_pages, hit_k, adjacent_window)
        category = primary_failure_category(
            page_rank=page_rank,
            doc_rank=doc_rank,
            hit_k=hit_k,
            boundary_k=boundary_k,
            same_doc_topk=same_doc_topk,
            adjacent_topk=adjacent_topk,
        )
        question = question_text(gold_row)
        cues = query_cues(question)
        tags = secondary_tags(
            page_rank=page_rank,
            doc_rank=doc_rank,
            hit_k=hit_k,
            boundary_k=boundary_k,
            same_doc_topk=same_doc_topk,
            adjacent_topk=adjacent_topk,
            gold_page_count=len(gold_pages),
            gold_doc_count=len(gold_docs),
        )
        case = {
            "run": run_label,
            "qid": qid,
            "question": question,
            "category": category,
            "tags": tags,
            "query_cues": cues,
            "page_first_rank": page_rank,
            "doc_first_rank": doc_rank,
            "page_rank_bucket": rank_bucket(page_rank, hit_k, boundary_k),
            "page_hit_at_k": page_rank is not None and page_rank <= hit_k,
            "doc_hit_at_k": doc_rank is not None and doc_rank <= hit_k,
            "gold_page_count": len(gold_pages),
            "gold_doc_count": len(gold_docs),
            "gold_page_uids": sorted(gold_pages),
            "gold_doc_ids": sorted(gold_docs),
            "top_page_uids": pages[:hit_k],
            "top_doc_ids": docs[:hit_k],
            "same_doc_topk": same_doc_topk,
            "adjacent_topk": adjacent_topk,
            "score_margin_4_5": score_margin(pred_row, 4, 5),
            "metadata_type": metadata_value(gold_row, "type"),
            "metadata_domain": metadata_value(gold_row, "domain"),
        }
        cases.append(case)

    category_counts = Counter(case["category"] for case in cases)
    rank_bucket_counts = Counter(case["page_rank_bucket"] for case in cases)
    tag_counts = Counter(tag for case in cases for tag in case["tags"])
    cue_counts = Counter(cue for case in cases for cue in case["query_cues"])
    failures = [case for case in cases if not case["page_hit_at_k"]]
    failure_category_counts = Counter(case["category"] for case in failures)
    by_cue_failure: dict[str, Counter[str]] = defaultdict(Counter)
    for case in failures:
        for cue in case["query_cues"]:
            by_cue_failure[cue][case["category"]] += 1
    return {
        "run": run_label,
        "gold": str(gold_path),
        "prediction": str(prediction_path),
        "n": len(cases),
        "hit_k": hit_k,
        "boundary_k": boundary_k,
        "adjacent_window": adjacent_window,
        "page_hit_at_k_count": sum(1 for case in cases if case["page_hit_at_k"]),
        "doc_hit_at_k_count": sum(1 for case in cases if case["doc_hit_at_k"]),
        "page_failure_count": len(failures),
        "category_counts": dict(sorted(category_counts.items())),
        "failure_category_counts": dict(sorted(failure_category_counts.items())),
        "page_rank_bucket_counts": dict(sorted(rank_bucket_counts.items())),
        "tag_counts": dict(sorted(tag_counts.items())),
        "query_cue_counts": dict(sorted(cue_counts.items())),
        "failure_by_query_cue": {
            cue: dict(sorted(counter.items())) for cue, counter in sorted(by_cue_failure.items())
        },
        "page_recall_at_k": {str(k): mean(values) for k, values in sorted(page_recall.items())},
        "doc_recall_at_k": {str(k): mean(values) for k, values in sorted(doc_recall.items())},
        "top_failures": sorted(
            failures,
            key=lambda case: (
                case["doc_first_rank"] is None,
                case["doc_first_rank"] if case["doc_first_rank"] is not None else 10**9,
                case["page_first_rank"] if case["page_first_rank"] is not None else 10**9,
                case["qid"],
            ),
        ),
        "cases": cases,
    }


def table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return lines


def render_md(payload: dict[str, Any], topn: int) -> str:
    lines = ["# Retrieval Failure Taxonomy", ""]
    lines.append(
        "This is an audit only. Categories use gold labels to explain failures and must not be used as routing features."
    )
    lines.append("")
    headers = [
        "run",
        "n",
        "page_hit@k",
        "doc_hit@k",
        "page_failures",
        "page_recall@4",
        "doc_recall@4",
    ]
    rows = []
    for run in payload["runs"]:
        rows.append(
            [
                run["run"],
                run["n"],
                run["page_hit_at_k_count"],
                run["doc_hit_at_k_count"],
                run["page_failure_count"],
                f"{run['page_recall_at_k'].get('4', 0.0):.4f}",
                f"{run['doc_recall_at_k'].get('4', 0.0):.4f}",
            ]
        )
    lines.extend(table(headers, rows))
    for run in payload["runs"]:
        lines.extend(["", f"## {run['run']}", ""])
        cat_rows = [
            [category, count]
            for category, count in sorted(
                run["failure_category_counts"].items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]
        lines.extend(table(["failure_category", "count"], cat_rows))
        lines.extend(["", "### Page Rank Buckets", ""])
        bucket_rows = [
            [bucket, count]
            for bucket, count in sorted(
                run["page_rank_bucket_counts"].items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]
        lines.extend(table(["page_rank_bucket", "count"], bucket_rows))
        lines.extend(["", "### Failure By Query Cue", ""])
        cue_rows = []
        for cue, counts in run["failure_by_query_cue"].items():
            cue_rows.append([cue, json.dumps(counts, sort_keys=True)])
        lines.extend(table(["query_cue", "failure_categories"], cue_rows))
        lines.extend(["", "### Example Failures", ""])
        for case in run["top_failures"][:topn]:
            lines.append(
                f"- `{case['qid']}` category=`{case['category']}` "
                f"page_rank={case['page_first_rank']} doc_rank={case['doc_first_rank']} "
                f"bucket=`{case['page_rank_bucket']}` cues={case['query_cues']}"
            )
            lines.append(f"  - question: {case['question']}")
            lines.append(f"  - gold_pages: {case['gold_page_uids']}")
            lines.append(f"  - top_pages: {case['top_page_uids']}")
            if case["same_doc_topk"]:
                lines.append(f"  - same_doc_topk: {case['same_doc_topk']}")
            if case["adjacent_topk"]:
                lines.append(f"  - adjacent_topk: {case['adjacent_topk']}")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, payload: dict[str, Any]) -> None:
    fieldnames = [
        "run",
        "qid",
        "category",
        "page_first_rank",
        "doc_first_rank",
        "page_rank_bucket",
        "page_hit_at_k",
        "doc_hit_at_k",
        "gold_page_count",
        "gold_doc_count",
        "query_cues",
        "tags",
        "metadata_type",
        "metadata_domain",
        "score_margin_4_5",
        "question",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run in payload["runs"]:
            for case in run["cases"]:
                row = {key: case.get(key) for key in fieldnames}
                row["query_cues"] = ",".join(case["query_cues"])
                row["tags"] = ",".join(case["tags"])
                writer.writerow(row)


def main() -> None:
    args = parse_args()
    run_specs = list(args.run)
    if not run_specs:
        if not args.gold or not args.prediction:
            raise ValueError("Provide either repeated --run LABEL GOLD PREDICTION or --gold/--prediction.")
        run_specs = [(args.run_label, args.gold, args.prediction)]
    runs = [
        audit_run(
            run_label=str(label),
            gold_path=Path(gold_path),
            prediction_path=Path(prediction_path),
            hit_k=int(args.hit_k),
            boundary_k=int(args.boundary_k),
            adjacent_window=int(args.adjacent_window),
            recall_ks=[int(k) for k in args.recall_ks],
        )
        for label, gold_path, prediction_path in run_specs
    ]
    payload = {
        "hit_k": int(args.hit_k),
        "boundary_k": int(args.boundary_k),
        "adjacent_window": int(args.adjacent_window),
        "runs": runs,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_md(payload, int(args.topn)), encoding="utf-8")
    if args.output_csv:
        write_csv(Path(args.output_csv), payload)
    print(f"saved_json: {output_json}")
    print(f"saved_md: {output_md}")
    if args.output_csv:
        print(f"saved_csv: {args.output_csv}")
    for run in runs:
        print(
            run["run"],
            "n",
            run["n"],
            "page_hit_at_k",
            run["page_hit_at_k_count"],
            "doc_hit_at_k",
            run["doc_hit_at_k_count"],
            "page_failures",
            run["page_failure_count"],
            "failure_categories",
            run["failure_category_counts"],
        )


if __name__ == "__main__":
    main()
