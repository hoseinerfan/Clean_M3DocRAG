#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit page-retrieval failures by checking whether gold pages exist deeper in a "
            "prediction list and/or in dense/sparse source predictions."
        )
    )
    parser.add_argument("--gold", required=True, help="Converted MMQA_dev.jsonl-style gold file.")
    parser.add_argument("--prediction", required=True, help="Prediction JSON to audit.")
    parser.add_argument("--prediction-label", default="candidate")
    parser.add_argument("--dense-pred", default="", help="Optional dense prediction JSON.")
    parser.add_argument("--sparse-pred", default="", help="Optional sparse prediction JSON.")
    parser.add_argument("--hit-k", type=int, default=4)
    parser.add_argument(
        "--rank-bins",
        type=int,
        nargs="+",
        default=[5, 10, 20, 50, 100, 200, 500, 1000],
        help="Rank cutoffs used to summarize missed-but-present gold pages.",
    )
    parser.add_argument("--top-rows", type=int, default=5, help="Top retrieved rows saved per failure.")
    parser.add_argument("--examples-per-bucket", type=int, default=10)
    parser.add_argument("--output-md", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-jsonl", default="")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_prediction(path: str) -> dict[str, Any]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("predictions"), dict):
        payload = payload["predictions"]
    if not isinstance(payload, dict):
        raise TypeError(f"Prediction must be a JSON object keyed by qid: {path}")
    return {str(qid): row for qid, row in payload.items()}


def page_uid(doc_id: str, page_idx: int) -> str:
    return f"{doc_id}_page{int(page_idx)}"


def parse_retrieval_row(row: Any) -> tuple[str, int, float | None] | None:
    if isinstance(row, (list, tuple)) and len(row) >= 2:
        try:
            score = float(row[2]) if len(row) >= 3 and row[2] is not None else None
            return str(row[0]), int(row[1]), score
        except (TypeError, ValueError):
            return None
    if isinstance(row, dict):
        doc_id = row.get("doc_id", row.get("docid", row.get("document_id")))
        page_idx = row.get("page_idx", row.get("page_id", row.get("page")))
        if doc_id is None or page_idx is None:
            uid = row.get("page_uid")
            if isinstance(uid, str) and "_page" in uid:
                doc, page = uid.rsplit("_page", 1)
                try:
                    return doc, int(page), float(row["score"]) if row.get("score") is not None else None
                except (TypeError, ValueError):
                    return None
            return None
        try:
            return str(doc_id), int(page_idx), float(row["score"]) if row.get("score") is not None else None
        except (TypeError, ValueError):
            return None
    return None


def prediction_rows(pred_row: Any) -> list[Any]:
    if not isinstance(pred_row, dict):
        return []
    rows = pred_row.get("page_retrieval_results")
    if rows is None:
        rows = pred_row.get("retrieval_results", pred_row.get("results", []))
    return rows if isinstance(rows, list) else []


def ranked_pages(pred_row: Any) -> list[str]:
    pages = []
    seen = set()
    for row in prediction_rows(pred_row):
        parsed = parse_retrieval_row(row)
        if parsed is None:
            continue
        uid = page_uid(parsed[0], parsed[1])
        if uid not in seen:
            seen.add(uid)
            pages.append(uid)
    return pages


def ranked_docs(pred_row: Any) -> list[str]:
    docs = []
    seen = set()
    for row in prediction_rows(pred_row):
        parsed = parse_retrieval_row(row)
        if parsed is None:
            continue
        doc_id = parsed[0]
        if doc_id not in seen:
            seen.add(doc_id)
            docs.append(doc_id)
    return docs


def top_rows(pred_row: Any, limit: int) -> list[dict[str, Any]]:
    out = []
    for rank, row in enumerate(prediction_rows(pred_row), start=1):
        if len(out) >= limit:
            break
        parsed = parse_retrieval_row(row)
        if parsed is None:
            continue
        doc_id, page_idx, score = parsed
        out.append(
            {
                "rank": rank,
                "doc_id": doc_id,
                "page_idx": page_idx,
                "page_uid": page_uid(doc_id, page_idx),
                "score": score,
            }
        )
    return out


def gold_page_uids(row: dict[str, Any]) -> list[str]:
    metadata = row.get("metadata", {})
    uids = []
    seen = set()
    for value in metadata.get("gold_page_uids", []):
        uid = str(value).strip()
        if uid and uid not in seen:
            seen.add(uid)
            uids.append(uid)
    for ctx in row.get("supporting_context", []):
        doc_id = str(ctx.get("doc_id", "")).strip()
        page_idx = ctx.get("page_idx", ctx.get("page_id"))
        if doc_id and page_idx is not None:
            uid = page_uid(doc_id, int(page_idx))
            if uid not in seen:
                seen.add(uid)
                uids.append(uid)
    return uids


def gold_doc_ids(row: dict[str, Any]) -> list[str]:
    docs = []
    seen = set()
    for ctx in row.get("supporting_context", []):
        doc_id = str(ctx.get("doc_id", "")).strip()
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            docs.append(doc_id)
    if docs:
        return docs
    for uid in gold_page_uids(row):
        if "_page" in uid:
            doc_id = uid.rsplit("_page", 1)[0]
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                docs.append(doc_id)
    return docs


def first_rank(ranked: list[str], gold: set[str]) -> int | None:
    for idx, item in enumerate(ranked, start=1):
        if item in gold:
            return idx
    return None


def rank_map(ranked: list[str], gold: set[str]) -> dict[str, int | None]:
    ranks = {item: idx for idx, item in enumerate(ranked, start=1)}
    return {item: ranks.get(item) for item in sorted(gold)}


def metadata_value(row: dict[str, Any], field: str) -> str:
    current: Any = row
    for part in field.split("."):
        if not isinstance(current, dict):
            return "UNKNOWN"
        current = current.get(part)
    if current is None:
        return "UNKNOWN"
    if isinstance(current, list):
        return str(current)
    text = str(current).strip()
    return text or "UNKNOWN"


def rank_bucket(rank: int | None, hit_k: int, rank_bins: list[int]) -> str:
    if rank is None:
        return "missing_from_prediction"
    if rank <= hit_k:
        return f"hit_at_{hit_k}"
    for cutoff in sorted(rank_bins):
        if rank <= cutoff:
            return f"rank_{hit_k + 1}_to_{cutoff}"
    return f"rank_gt_{max(rank_bins)}"


def classify_failure(pred_rank: int | None, dense_rank: int | None, sparse_rank: int | None, hit_k: int) -> str:
    if pred_rank is not None and pred_rank <= hit_k:
        return "hit"
    if pred_rank is not None:
        return "rankable_in_prediction_beyond_hit_k"
    if dense_rank is not None or sparse_rank is not None:
        return "missing_from_prediction_but_source_has"
    return "missing_from_all_checked_predictions"


def pct(count: int, denom: int) -> str:
    return f"{(100.0 * count / denom):.2f}%" if denom else "0.00%"


def md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def sort_failure(row: dict[str, Any]) -> tuple[int, int, str]:
    ranks = [
        row.get("prediction_first_gold_page_rank"),
        row.get("dense_first_gold_page_rank"),
        row.get("sparse_first_gold_page_rank"),
    ]
    best_rank = min([int(rank) for rank in ranks if rank is not None], default=10**9)
    pred_rank = row.get("prediction_first_gold_page_rank")
    return best_rank, 10**9 if pred_rank is None else int(pred_rank), str(row["qid"])


def main() -> None:
    args = parse_args()
    gold_rows = read_jsonl(Path(args.gold))
    prediction = read_prediction(args.prediction)
    dense = read_prediction(args.dense_pred)
    sparse = read_prediction(args.sparse_pred)
    rank_bins = sorted(set(args.rank_bins))

    summary_counter: Counter[str] = Counter()
    bucket_counter: Counter[str] = Counter()
    type_counter: dict[str, Counter[str]] = defaultdict(Counter)
    domain_counter: dict[str, Counter[str]] = defaultdict(Counter)
    failure_rows: list[dict[str, Any]] = []
    per_qid: list[dict[str, Any]] = []

    for gold_row in gold_rows:
        qid = str(gold_row.get("qid", "")).strip()
        if not qid:
            continue
        gold_pages = set(gold_page_uids(gold_row))
        gold_docs = set(gold_doc_ids(gold_row))
        if not gold_pages:
            summary_counter["skipped_no_gold_page"] += 1
            continue

        pred_row = prediction.get(qid)
        dense_row = dense.get(qid)
        sparse_row = sparse.get(qid)
        pred_pages = ranked_pages(pred_row)
        dense_pages = ranked_pages(dense_row) if dense else []
        sparse_pages = ranked_pages(sparse_row) if sparse else []
        pred_docs = ranked_docs(pred_row)
        dense_docs = ranked_docs(dense_row) if dense else []
        sparse_docs = ranked_docs(sparse_row) if sparse else []
        pred_rank = first_rank(pred_pages, gold_pages)
        dense_rank = first_rank(dense_pages, gold_pages) if dense else None
        sparse_rank = first_rank(sparse_pages, gold_pages) if sparse else None
        pred_doc_rank = first_rank(pred_docs, gold_docs)
        dense_doc_rank = first_rank(dense_docs, gold_docs) if dense else None
        sparse_doc_rank = first_rank(sparse_docs, gold_docs) if sparse else None

        classification = classify_failure(pred_rank, dense_rank, sparse_rank, args.hit_k)
        bucket = rank_bucket(pred_rank, args.hit_k, rank_bins)
        page_hit = pred_rank is not None and pred_rank <= args.hit_k
        doc_hit = pred_doc_rank is not None and pred_doc_rank <= args.hit_k

        summary_counter["evaluated_qids"] += 1
        summary_counter[classification] += 1
        summary_counter["page_hit_at_k"] += int(page_hit)
        summary_counter["page_miss_at_k"] += int(not page_hit)
        summary_counter["doc_hit_at_k"] += int(doc_hit)
        summary_counter["dense_has_gold_page"] += int(dense_rank is not None)
        summary_counter["sparse_has_gold_page"] += int(sparse_rank is not None)
        summary_counter["either_source_has_gold_page"] += int(dense_rank is not None or sparse_rank is not None)
        if not page_hit:
            bucket_counter[bucket] += 1
            type_counter[metadata_value(gold_row, "metadata.type")][classification] += 1
            domain_counter[metadata_value(gold_row, "metadata.domain")][classification] += 1

        item = {
            "qid": qid,
            "question": gold_row.get("question", ""),
            "metadata_type": metadata_value(gold_row, "metadata.type"),
            "metadata_domain": metadata_value(gold_row, "metadata.domain"),
            "gold_page_uids": sorted(gold_pages),
            "gold_doc_ids": sorted(gold_docs),
            "prediction_gold_page_ranks": rank_map(pred_pages, gold_pages),
            "dense_gold_page_ranks": rank_map(dense_pages, gold_pages) if dense else {},
            "sparse_gold_page_ranks": rank_map(sparse_pages, gold_pages) if sparse else {},
            "prediction_first_gold_page_rank": pred_rank,
            "dense_first_gold_page_rank": dense_rank,
            "sparse_first_gold_page_rank": sparse_rank,
            "prediction_first_gold_doc_rank": pred_doc_rank,
            "dense_first_gold_doc_rank": dense_doc_rank,
            "sparse_first_gold_doc_rank": sparse_doc_rank,
            "failure_class": classification,
            "rank_bucket": bucket,
        }
        per_qid.append(item)

        if not page_hit:
            item["top_prediction_pages"] = top_rows(pred_row, args.top_rows)
            item["top_dense_pages"] = top_rows(dense_row, args.top_rows) if dense else []
            item["top_sparse_pages"] = top_rows(sparse_row, args.top_rows) if sparse else []
            failure_rows.append(item)

    failure_rows.sort(key=sort_failure)
    summary = {
        "prediction_label": args.prediction_label,
        "hit_k": args.hit_k,
        "gold_path": args.gold,
        "prediction_path": args.prediction,
        "dense_prediction_path": args.dense_pred,
        "sparse_prediction_path": args.sparse_pred,
        **dict(summary_counter),
        "page_hit_at_k_fraction": summary_counter["page_hit_at_k"] / summary_counter["evaluated_qids"]
        if summary_counter["evaluated_qids"]
        else 0.0,
        "doc_hit_at_k_fraction": summary_counter["doc_hit_at_k"] / summary_counter["evaluated_qids"]
        if summary_counter["evaluated_qids"]
        else 0.0,
        "miss_rank_bucket_counts": dict(bucket_counter),
    }
    payload = {"summary": summary, "failures": failure_rows, "per_qid": per_qid}

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.output_jsonl:
        with Path(args.output_jsonl).open("w", encoding="utf-8") as handle:
            for row in failure_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.output_md:
        lines = render_markdown(args, summary, bucket_counter, type_counter, domain_counter, failure_rows)
        Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"prediction_label {args.prediction_label}")
    print(f"evaluated_qids {summary_counter['evaluated_qids']}")
    print(f"page_hit_at_{args.hit_k} {summary_counter['page_hit_at_k']} ({pct(summary_counter['page_hit_at_k'], summary_counter['evaluated_qids'])})")
    print(f"page_miss_at_{args.hit_k} {summary_counter['page_miss_at_k']} ({pct(summary_counter['page_miss_at_k'], summary_counter['evaluated_qids'])})")
    print(f"doc_hit_at_{args.hit_k} {summary_counter['doc_hit_at_k']} ({pct(summary_counter['doc_hit_at_k'], summary_counter['evaluated_qids'])})")
    print("failure_class_counts", dict((key, summary_counter[key]) for key in sorted(summary_counter) if key in {
        "rankable_in_prediction_beyond_hit_k",
        "missing_from_prediction_but_source_has",
        "missing_from_all_checked_predictions",
    }))
    print("miss_rank_bucket_counts", dict(bucket_counter))
    if args.output_md:
        print(f"saved_md {args.output_md}")
    if args.output_json:
        print(f"saved_json {args.output_json}")
    if args.output_jsonl:
        print(f"saved_jsonl {args.output_jsonl}")


def render_markdown(
    args: argparse.Namespace,
    summary: dict[str, Any],
    bucket_counter: Counter[str],
    type_counter: dict[str, Counter[str]],
    domain_counter: dict[str, Counter[str]],
    failure_rows: list[dict[str, Any]],
) -> list[str]:
    total = int(summary.get("evaluated_qids", 0))
    miss_total = int(summary.get("page_miss_at_k", 0))
    lines = [
        "# Gold Page Candidate Coverage Audit",
        "",
        f"- prediction: `{args.prediction_label}`",
        f"- hit_k: `{args.hit_k}`",
        f"- evaluated_qids: `{total}`",
        "",
        "## Summary",
        "",
    ]
    lines.extend(
        md_table(
            ["metric", "count", "fraction"],
            [
                [f"page_hit@{args.hit_k}", summary.get("page_hit_at_k", 0), pct(int(summary.get("page_hit_at_k", 0)), total)],
                [f"page_miss@{args.hit_k}", summary.get("page_miss_at_k", 0), pct(miss_total, total)],
                [f"doc_hit@{args.hit_k}", summary.get("doc_hit_at_k", 0), pct(int(summary.get("doc_hit_at_k", 0)), total)],
                [
                    "rankable_in_prediction_beyond_hit_k",
                    summary.get("rankable_in_prediction_beyond_hit_k", 0),
                    pct(int(summary.get("rankable_in_prediction_beyond_hit_k", 0)), miss_total),
                ],
                [
                    "missing_from_prediction_but_source_has",
                    summary.get("missing_from_prediction_but_source_has", 0),
                    pct(int(summary.get("missing_from_prediction_but_source_has", 0)), miss_total),
                ],
                [
                    "missing_from_all_checked_predictions",
                    summary.get("missing_from_all_checked_predictions", 0),
                    pct(int(summary.get("missing_from_all_checked_predictions", 0)), miss_total),
                ],
            ],
        )
    )
    lines.extend(["", "## Miss Rank Buckets", ""])
    lines.extend(
        md_table(
            ["bucket", "count", "fraction_of_misses"],
            [[bucket, count, pct(count, miss_total)] for bucket, count in bucket_counter.most_common()],
        )
    )
    lines.extend(["", "## Failure Classes By Domain", ""])
    lines.extend(render_group_table("metadata.domain", domain_counter))
    lines.extend(["", "## Failure Classes By Type", ""])
    lines.extend(render_group_table("metadata.type", type_counter))
    lines.extend(["", "## Example Failures", ""])

    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in failure_rows:
        by_bucket[row["failure_class"]].append(row)
    for bucket, rows in sorted(by_bucket.items()):
        lines.extend(["", f"### {bucket}", ""])
        for row in rows[: args.examples_per_bucket]:
            lines.extend(
                [
                    f"- `{row['qid']}`",
                    f"  - question: {row.get('question', '')}",
                    f"  - metadata: type=`{row.get('metadata_type')}`, domain=`{row.get('metadata_domain')}`",
                    (
                        "  - ranks: "
                        f"{args.prediction_label}_page={row.get('prediction_first_gold_page_rank')}, "
                        f"dense_page={row.get('dense_first_gold_page_rank')}, "
                        f"sparse_page={row.get('sparse_first_gold_page_rank')}, "
                        f"{args.prediction_label}_doc={row.get('prediction_first_gold_doc_rank')}"
                    ),
                    f"  - gold_pages: `{', '.join(row.get('gold_page_uids', []))}`",
                    f"  - top_{args.prediction_label}: {format_top_pages(row.get('top_prediction_pages', []))}",
                ]
            )
    return lines


def render_group_table(group_label: str, counter_by_group: dict[str, Counter[str]]) -> list[str]:
    rows = []
    for group, counter in counter_by_group.items():
        total = sum(counter.values())
        rows.append(
            [
                group_label,
                group,
                total,
                counter.get("rankable_in_prediction_beyond_hit_k", 0),
                counter.get("missing_from_prediction_but_source_has", 0),
                counter.get("missing_from_all_checked_predictions", 0),
            ]
        )
    rows.sort(key=lambda row: (-int(row[2]), str(row[1])))
    return md_table(
        [
            "group_by",
            "group",
            "miss_count",
            "rankable_beyond_hit_k",
            "candidate_missing_source_has",
            "missing_all_checked",
        ],
        rows,
    )


def format_top_pages(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "N/A"
    return "; ".join(f"{row['rank']}:{row['page_uid']}" for row in rows)


if __name__ == "__main__":
    main()
