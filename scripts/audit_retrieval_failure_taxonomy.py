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
    parser.add_argument(
        "--metadata-field",
        action="append",
        default=[
            "metadata.type",
            "metadata.domain",
            "metadata.repo_slug",
            "metadata.query_types",
            "metadata.query_format",
            "metadata.content_type",
            "metadata.category",
            "metadata.source",
        ],
        help=(
            "Metadata path to slice failures by. Repeatable. Defaults cover MMDocIR, "
            "ViDoRe, and common converted MMQA metadata fields."
        ),
    )
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


def doc_rank_bucket(rank: int | None, hit_k: int) -> str:
    if rank is None:
        return "missing"
    if rank <= hit_k:
        return f"top{hit_k}"
    if rank <= 10:
        return f"doc_{hit_k + 1}_10"
    if rank <= 20:
        return "doc_11_20"
    if rank <= 50:
        return "doc_21_50"
    if rank <= 100:
        return "doc_51_100"
    return "doc_gt100"


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


def score_for_rank(row: dict[str, Any], rank: int | None) -> float | None:
    if rank is None:
        return None
    return score_at(row, rank)


def uid_ranks(ranked: list[str], gold: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, uid in enumerate(ranked, start=1):
        if uid in gold:
            out.append({"uid": uid, "rank": idx})
    for uid in sorted(gold - set(ranked)):
        out.append({"uid": uid, "rank": None})
    return out


def same_doc_details(
    pred_pages: list[str],
    gold_pages: set[str],
    gold_docs: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    gold_by_doc: dict[str, list[int]] = defaultdict(list)
    for uid in gold_pages:
        idx = page_index(uid)
        if idx is not None:
            gold_by_doc[page_doc(uid)].append(idx)
    details: list[dict[str, Any]] = []
    for rank, uid in enumerate(pred_pages[:limit], start=1):
        doc_id = page_doc(uid)
        if doc_id not in gold_docs:
            continue
        idx = page_index(uid)
        distances = (
            [abs(idx - gold_idx) for gold_idx in gold_by_doc.get(doc_id, [])]
            if idx is not None
            else []
        )
        details.append(
            {
                "uid": uid,
                "rank": rank,
                "page_idx": idx,
                "min_gold_page_distance": min(distances) if distances else None,
            }
        )
    return details


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


def limitation_group(category: str) -> str:
    if category == "page_hit":
        return "solved_page_hit"
    if category in {"doc_miss_topk", "doc_missing_from_pool"}:
        return "document_retrieval_gap"
    if category == "right_doc_boundary_page":
        return "rank_boundary_localization"
    if category in {"right_doc_adjacent_page", "right_doc_same_doc_sibling"}:
        return "same_document_page_confusion"
    if category in {"right_doc_late_page", "right_doc_gold_page_missing_from_pool"}:
        return "right_doc_deep_or_missing_page"
    return "other"


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
    if key.startswith("metadata."):
        value: Any = row
    else:
        value = row.get("metadata", {})
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


def top_counter(counter: Counter[str], limit: int = 20) -> dict[str, int]:
    return dict(counter.most_common(limit))


def nested_counter_to_dict(value: dict[str, Counter[str]], limit: int = 20) -> dict[str, dict[str, int]]:
    return {key: top_counter(counter, limit) for key, counter in sorted(value.items())}


def audit_run(
    *,
    run_label: str,
    gold_path: Path,
    prediction_path: Path,
    hit_k: int,
    boundary_k: int,
    adjacent_window: int,
    recall_ks: list[int],
    metadata_fields: list[str],
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
        gold_page_rank_rows = uid_ranks(pages, gold_pages)
        gold_doc_rank_rows = uid_ranks(docs, gold_docs)
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
        group = limitation_group(category)
        top_boundary_page = pages[hit_k] if len(pages) > hit_k else None
        top_boundary_doc = page_doc(top_boundary_page) if top_boundary_page else None
        same_doc_detail_rows = same_doc_details(pages, gold_pages, gold_docs, max(boundary_k, hit_k))
        closest_same_doc_distance = min(
            [
                int(row["min_gold_page_distance"])
                for row in same_doc_detail_rows
                if row.get("min_gold_page_distance") is not None
            ],
            default=None,
        )
        metadata_fields_out = {
            field: metadata_value(gold_row, field)
            for field in metadata_fields
            if metadata_value(gold_row, field)
        }
        case = {
            "run": run_label,
            "qid": qid,
            "question": question,
            "category": category,
            "limitation_group": group,
            "tags": tags,
            "query_cues": cues,
            "page_first_rank": page_rank,
            "doc_first_rank": doc_rank,
            "page_rank_bucket": rank_bucket(page_rank, hit_k, boundary_k),
            "doc_rank_bucket": doc_rank_bucket(doc_rank, hit_k),
            "page_hit_at_k": page_rank is not None and page_rank <= hit_k,
            "doc_hit_at_k": doc_rank is not None and doc_rank <= hit_k,
            "gold_page_in_pool": page_rank is not None,
            "gold_doc_in_pool": doc_rank is not None,
            "page_rank_gap_from_topk": (
                page_rank - hit_k if page_rank is not None and page_rank > hit_k else 0
            ),
            "gold_page_count": len(gold_pages),
            "gold_doc_count": len(gold_docs),
            "gold_page_uids": sorted(gold_pages),
            "gold_doc_ids": sorted(gold_docs),
            "gold_page_ranks": gold_page_rank_rows,
            "gold_doc_ranks": gold_doc_rank_rows,
            "top_page_uids": pages[:hit_k],
            "top10_page_uids": pages[:10],
            "top20_page_uids": pages[:20],
            "top_doc_ids": docs[:hit_k],
            "top10_doc_ids": docs[:10],
            "top1_page_uid": pages[0] if pages else "",
            "top1_doc_id": docs[0] if docs else "",
            "rank5_page_uid": top_boundary_page or "",
            "rank5_doc_id": top_boundary_doc or "",
            "rank5_is_gold_page": bool(top_boundary_page and top_boundary_page in gold_pages),
            "rank5_is_gold_doc": bool(top_boundary_doc and top_boundary_doc in gold_docs),
            "same_doc_topk": same_doc_topk,
            "adjacent_topk": adjacent_topk,
            "same_doc_details_top_boundary": same_doc_detail_rows,
            "same_doc_topk_count": len(same_doc_topk),
            "adjacent_topk_count": len(adjacent_topk),
            "closest_same_doc_page_distance": closest_same_doc_distance,
            "score_rank1": score_at(pred_row, 1),
            "score_rank4": score_at(pred_row, hit_k),
            "score_rank5": score_at(pred_row, hit_k + 1),
            "score_best_gold_page": score_for_rank(pred_row, page_rank),
            "score_margin_4_5": score_margin(pred_row, 4, 5),
            "metadata_type": metadata_value(gold_row, "type"),
            "metadata_domain": metadata_value(gold_row, "domain"),
            "metadata_fields": metadata_fields_out,
        }
        cases.append(case)

    category_counts = Counter(case["category"] for case in cases)
    limitation_group_counts = Counter(case["limitation_group"] for case in cases)
    rank_bucket_counts = Counter(case["page_rank_bucket"] for case in cases)
    doc_rank_bucket_counts = Counter(case["doc_rank_bucket"] for case in cases)
    tag_counts = Counter(tag for case in cases for tag in case["tags"])
    cue_counts = Counter(cue for case in cases for cue in case["query_cues"])
    failures = [case for case in cases if not case["page_hit_at_k"]]
    failure_category_counts = Counter(case["category"] for case in failures)
    failure_limitation_group_counts = Counter(case["limitation_group"] for case in failures)
    failure_rank_bucket_counts = Counter(case["page_rank_bucket"] for case in failures)
    failure_doc_rank_bucket_counts = Counter(case["doc_rank_bucket"] for case in failures)
    exact_rank_failure_counts = Counter(
        str(case["page_first_rank"]) if case["page_first_rank"] is not None else "missing"
        for case in failures
    )
    rank5_gold_count = sum(1 for case in failures if case["rank5_is_gold_page"])
    right_doc_failures = [case for case in failures if case["doc_hit_at_k"]]
    by_cue_failure: dict[str, Counter[str]] = defaultdict(Counter)
    by_cue_limitation: dict[str, Counter[str]] = defaultdict(Counter)
    by_metadata_failure: dict[str, Counter[str]] = defaultdict(Counter)
    by_metadata_limitation: dict[str, Counter[str]] = defaultdict(Counter)
    metadata_value_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for case in failures:
        for cue in case["query_cues"]:
            by_cue_failure[cue][case["category"]] += 1
            by_cue_limitation[cue][case["limitation_group"]] += 1
        for field, value in case["metadata_fields"].items():
            key = f"{field}={value}"
            by_metadata_failure[key][case["category"]] += 1
            by_metadata_limitation[key][case["limitation_group"]] += 1
            metadata_value_counts[field][value] += 1
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
        "limitation_group_counts": dict(sorted(limitation_group_counts.items())),
        "failure_category_counts": dict(sorted(failure_category_counts.items())),
        "failure_limitation_group_counts": dict(sorted(failure_limitation_group_counts.items())),
        "page_rank_bucket_counts": dict(sorted(rank_bucket_counts.items())),
        "failure_page_rank_bucket_counts": dict(sorted(failure_rank_bucket_counts.items())),
        "doc_rank_bucket_counts": dict(sorted(doc_rank_bucket_counts.items())),
        "failure_doc_rank_bucket_counts": dict(sorted(failure_doc_rank_bucket_counts.items())),
        "failure_exact_page_rank_counts": dict(sorted(exact_rank_failure_counts.items())),
        "tag_counts": dict(sorted(tag_counts.items())),
        "query_cue_counts": dict(sorted(cue_counts.items())),
        "failure_by_query_cue": {
            cue: dict(sorted(counter.items())) for cue, counter in sorted(by_cue_failure.items())
        },
        "failure_limitation_by_query_cue": nested_counter_to_dict(by_cue_limitation),
        "failure_by_metadata_value": nested_counter_to_dict(by_metadata_failure),
        "failure_limitation_by_metadata_value": nested_counter_to_dict(by_metadata_limitation),
        "failure_metadata_value_counts": {
            field: top_counter(counter) for field, counter in sorted(metadata_value_counts.items())
        },
        "boundary_rank5_gold_count": rank5_gold_count,
        "right_doc_failure_count": len(right_doc_failures),
        "right_doc_failure_rank5_gold_count": sum(1 for case in right_doc_failures if case["rank5_is_gold_page"]),
        "right_doc_failure_adjacent_or_same_doc_count": sum(
            1 for case in right_doc_failures if case["same_doc_topk"] or case["adjacent_topk"]
        ),
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


def pct(count: int | float, denom: int | float) -> str:
    if not denom:
        return "0.0%"
    return f"{100.0 * float(count) / float(denom):.1f}%"


def limitation_interpretation(group: str) -> str:
    return {
        "document_retrieval_gap": "gold document is not available in the top-k document set; page-local reranking cannot fix this alone",
        "rank_boundary_localization": "gold page is near the top-k boundary; local verifier/reranker can plausibly help",
        "same_document_page_confusion": "right document is present but a nearby/sibling page is preferred; needs page-local visual/text evidence",
        "right_doc_deep_or_missing_page": "right document is present but gold page is deep or absent from the returned page pool",
        "solved_page_hit": "already solved at page level",
        "other": "uncategorized residual",
    }.get(group, "")


def limitation_next_step(group: str) -> str:
    return {
        "document_retrieval_gap": "improve document discovery or support graph recall before boundary reranking",
        "rank_boundary_localization": "test exact MaxSim/content verifier on top4-vs-rank5/rank10",
        "same_document_page_confusion": "audit page-local content, OCR/layout regions, table/figure evidence, and adjacent-page traps",
        "right_doc_deep_or_missing_page": "expand candidate page pool inside the right document or improve intra-document page propagation",
        "solved_page_hit": "keep base ranking",
        "other": "inspect examples manually",
    }.get(group, "")


def render_md(payload: dict[str, Any], topn: int) -> str:
    lines = ["# Retrieval Limitation Report", ""]
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
        failure_n = int(run["page_failure_count"])
        solved_n = int(run["page_hit_at_k_count"])
        lines.append(
            f"Page-hit failures: **{failure_n} / {run['n']} ({pct(failure_n, run['n'])})**. "
            f"Already solved at page@{run['hit_k']}: **{solved_n} / {run['n']} ({pct(solved_n, run['n'])})**."
        )
        lines.append("")
        lines.append("### Limitation Map")
        lines.append("")
        group_rows = []
        for group, count in sorted(
            run["failure_limitation_group_counts"].items(),
            key=lambda item: (-item[1], item[0]),
        ):
            group_rows.append(
                [
                    group,
                    count,
                    pct(count, failure_n),
                    limitation_interpretation(group),
                    limitation_next_step(group),
                ]
            )
        lines.extend(table(["limitation", "failures", "failure_frac", "interpretation", "next_test"], group_rows))
        lines.extend(["", "### Primary Categories", ""])
        cat_rows = [
            [category, count]
            for category, count in sorted(
                run["failure_category_counts"].items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]
        lines.extend(table(["failure_category", "count"], cat_rows))
        lines.extend(["", "### Rank Position Diagnostics", ""])
        lines.append(
            f"Rank-5 gold failures: **{run['boundary_rank5_gold_count']} / {failure_n} "
            f"({pct(run['boundary_rank5_gold_count'], failure_n)})**. "
            f"Right-doc failures: **{run['right_doc_failure_count']}**; among those, rank-5 gold is "
            f"**{run['right_doc_failure_rank5_gold_count']} ({pct(run['right_doc_failure_rank5_gold_count'], run['right_doc_failure_count'])})**."
        )
        lines.append("")
        bucket_rows = [
            [bucket, count]
            for bucket, count in sorted(
                run["failure_page_rank_bucket_counts"].items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]
        lines.extend(table(["failed_page_rank_bucket", "count"], bucket_rows))
        exact_rank_rows = [
            [rank, count]
            for rank, count in sorted(
                run["failure_exact_page_rank_counts"].items(),
                key=lambda item: (
                    item[0] == "missing",
                    int(item[0]) if str(item[0]).isdigit() else 10**9,
                ),
            )[:20]
        ]
        lines.extend(["", "Exact failed gold-page rank histogram, first 20 ranks:", ""])
        lines.extend(table(["gold_page_rank", "count"], exact_rank_rows))
        doc_bucket_rows = [
            [bucket, count]
            for bucket, count in sorted(
                run["failure_doc_rank_bucket_counts"].items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]
        lines.extend(["", "Failed gold-document rank buckets:", ""])
        lines.extend(table(["gold_doc_rank_bucket", "count"], doc_bucket_rows))
        lines.extend(["", "### Failure By Query Cue", ""])
        cue_rows = []
        for cue, counts in run["failure_limitation_by_query_cue"].items():
            cue_rows.append([cue, json.dumps(counts, sort_keys=True)])
        lines.extend(table(["query_cue", "limitation_groups"], cue_rows))
        if run["failure_limitation_by_metadata_value"]:
            lines.extend(["", "### Metadata Hotspots", ""])
            metadata_rows = []
            for key, counts in run["failure_limitation_by_metadata_value"].items():
                metadata_rows.append([key, sum(counts.values()), json.dumps(counts, sort_keys=True)])
            metadata_rows.sort(key=lambda row: (-int(row[1]), str(row[0])))
            lines.extend(table(["metadata_value", "failures", "limitation_groups"], metadata_rows[:25]))
        lines.extend(["", "### Example Failures By Limitation", ""])
        for group, _count in sorted(
            run["failure_limitation_group_counts"].items(),
            key=lambda item: (-item[1], item[0]),
        ):
            examples = [case for case in run["top_failures"] if case["limitation_group"] == group][: max(1, topn // 5)]
            if not examples:
                continue
            lines.extend(["", f"#### {group}", ""])
            for case in examples:
                lines.append(
                    f"- `{case['qid']}` category=`{case['category']}` "
                    f"page_rank={case['page_first_rank']} doc_rank={case['doc_first_rank']} "
                    f"page_bucket=`{case['page_rank_bucket']}` doc_bucket=`{case['doc_rank_bucket']}` "
                    f"rank5_gold={case['rank5_is_gold_page']} cues={case['query_cues']}"
                )
                lines.append(f"  - question: {case['question']}")
                lines.append(f"  - gold_pages: {case['gold_page_uids']}")
                lines.append(f"  - top_pages: {case['top_page_uids']}")
                lines.append(f"  - rank5_page: {case['rank5_page_uid']}")
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
        "limitation_group",
        "page_first_rank",
        "doc_first_rank",
        "page_rank_bucket",
        "doc_rank_bucket",
        "page_hit_at_k",
        "doc_hit_at_k",
        "gold_page_in_pool",
        "gold_doc_in_pool",
        "page_rank_gap_from_topk",
        "gold_page_count",
        "gold_doc_count",
        "top1_page_uid",
        "top1_doc_id",
        "rank5_page_uid",
        "rank5_doc_id",
        "rank5_is_gold_page",
        "rank5_is_gold_doc",
        "same_doc_topk_count",
        "adjacent_topk_count",
        "closest_same_doc_page_distance",
        "query_cues",
        "tags",
        "metadata_type",
        "metadata_domain",
        "metadata_fields",
        "score_rank1",
        "score_rank4",
        "score_rank5",
        "score_best_gold_page",
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
                row["metadata_fields"] = json.dumps(case.get("metadata_fields", {}), ensure_ascii=False, sort_keys=True)
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
            metadata_fields=[str(field) for field in args.metadata_field],
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
