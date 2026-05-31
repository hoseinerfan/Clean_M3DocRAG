#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect per-dataset gold-page candidate coverage audits into one "
            "discovery-vs-promotion report."
        )
    )
    parser.add_argument("--input-json", action="append", required=True)
    parser.add_argument("--pool-k", type=int, default=1000)
    parser.add_argument("--hit-k", type=int, default=4)
    parser.add_argument("--coverage-ks", type=int, nargs="+", default=[4, 20, 50, 100, 200, 500, 1000])
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-csv", default="")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: int | float, denom: int | float) -> str:
    if not denom:
        return "0.00%"
    return f"{100.0 * float(value) / float(denom):.2f}%"


def md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def is_hit(rank: Any, k: int) -> bool:
    return rank is not None and int(rank) <= int(k)


def rank_values(row: dict[str, Any], key: str, fallback_rank: Any) -> list[Any]:
    ranks = row.get(key)
    if isinstance(ranks, dict) and ranks:
        return list(ranks.values())
    return [fallback_rank] if fallback_rank is not None else []


def all_ranks_hit(ranks: list[Any], k: int) -> bool:
    return bool(ranks) and all(is_hit(rank, k) for rank in ranks)


def summarize_payload(path: Path, payload: dict[str, Any], *, pool_k: int, hit_k: int, coverage_ks: list[int]) -> dict[str, Any]:
    summary = payload.get("summary", {})
    per_qid = payload.get("per_qid", [])
    dataset = str(summary.get("prediction_label") or path.stem).strip()
    total = len(per_qid)
    miss_at_hit_k = 0
    page_hits = {int(k): 0 for k in coverage_ks}
    doc_hits = {int(k): 0 for k in coverage_ks}
    dense_pool_hits = 0
    sparse_pool_hits = 0
    source_union_pool_hits = 0
    source_only_pool_hits = 0
    missing_from_prediction_pool = 0
    rankable_beyond_hit_k = 0
    all_gold_page_hit_at_hit_k = 0
    all_gold_page_pool_hits = 0
    gold_page_instance_total = 0
    gold_page_instance_hit_at_hit_k = 0
    gold_page_instance_pool_hits = 0

    for row in per_qid:
        page_rank = row.get("prediction_first_gold_page_rank")
        doc_rank = row.get("prediction_first_gold_doc_rank")
        dense_rank = row.get("dense_first_gold_page_rank")
        sparse_rank = row.get("sparse_first_gold_page_rank")
        page_ranks = rank_values(row, "prediction_gold_page_ranks", page_rank)

        if not is_hit(page_rank, hit_k):
            miss_at_hit_k += 1
        if is_hit(page_rank, pool_k) and not is_hit(page_rank, hit_k):
            rankable_beyond_hit_k += 1
        if not is_hit(page_rank, pool_k):
            missing_from_prediction_pool += 1

        if is_hit(dense_rank, pool_k):
            dense_pool_hits += 1
        if is_hit(sparse_rank, pool_k):
            sparse_pool_hits += 1
        if is_hit(dense_rank, pool_k) or is_hit(sparse_rank, pool_k):
            source_union_pool_hits += 1
        if not is_hit(page_rank, pool_k) and (is_hit(dense_rank, pool_k) or is_hit(sparse_rank, pool_k)):
            source_only_pool_hits += 1
        if all_ranks_hit(page_ranks, hit_k):
            all_gold_page_hit_at_hit_k += 1
        if all_ranks_hit(page_ranks, pool_k):
            all_gold_page_pool_hits += 1
        gold_page_instance_total += len(page_ranks)
        gold_page_instance_hit_at_hit_k += sum(1 for rank in page_ranks if is_hit(rank, hit_k))
        gold_page_instance_pool_hits += sum(1 for rank in page_ranks if is_hit(rank, pool_k))

        for k in page_hits:
            page_hits[k] += int(is_hit(page_rank, k))
            doc_hits[k] += int(is_hit(doc_rank, k))

    promotion_share_of_misses = (
        float(rankable_beyond_hit_k) / float(miss_at_hit_k) if miss_at_hit_k else 0.0
    )
    discovery_share_of_misses = (
        float(missing_from_prediction_pool) / float(miss_at_hit_k) if miss_at_hit_k else 0.0
    )
    return {
        "dataset": dataset,
        "path": str(path),
        "total": total,
        "hit_k": hit_k,
        "pool_k": pool_k,
        "page_hits": page_hits,
        "doc_hits": doc_hits,
        "page_hit_at_hit_k": page_hits.get(hit_k, 0),
        "page_pool_hits": page_hits.get(pool_k, 0),
        "miss_at_hit_k": miss_at_hit_k,
        "rankable_beyond_hit_k": rankable_beyond_hit_k,
        "missing_from_prediction_pool": missing_from_prediction_pool,
        "all_gold_page_hit_at_hit_k": all_gold_page_hit_at_hit_k,
        "all_gold_page_pool_hits": all_gold_page_pool_hits,
        "gold_page_instance_total": gold_page_instance_total,
        "gold_page_instance_hit_at_hit_k": gold_page_instance_hit_at_hit_k,
        "gold_page_instance_pool_hits": gold_page_instance_pool_hits,
        "dense_pool_hits": dense_pool_hits,
        "sparse_pool_hits": sparse_pool_hits,
        "source_union_pool_hits": source_union_pool_hits,
        "source_only_pool_hits": source_only_pool_hits,
        "promotion_share_of_misses": promotion_share_of_misses,
        "discovery_share_of_misses": discovery_share_of_misses,
        "statement_supported": promotion_share_of_misses > discovery_share_of_misses,
    }


def render_markdown(rows: list[dict[str, Any]], *, pool_k: int, hit_k: int, coverage_ks: list[int]) -> str:
    supported = sum(1 for row in rows if row["statement_supported"])
    lines = [
        "# Gold Page Pool Coverage Report",
        "",
        "## Question",
        "",
        f"Can we support the statement: if gold pages are already in the top-{pool_k} pool, the bottleneck is promotion rather than discovery?",
        "",
        "## Conclusion",
        "",
        (
            f"The statement is supported for {supported}/{len(rows)} audited prediction pools "
            f"when the share of top-{hit_k} misses that are still present by top-{pool_k} "
            "is larger than the share missing from the pool."
        ),
        "",
        "Interpretation: `rankable beyond top-k` means the gold page was discovered in the candidate pool but not promoted high enough. "
        "`missing from pool` means reranking the fixed pool cannot recover the gold page without adding candidates.",
        "",
        "## Dataset Summary",
        "",
    ]
    summary_rows = []
    for row in rows:
        total = int(row["total"])
        miss = int(row["miss_at_hit_k"])
        summary_rows.append(
            [
                row["dataset"],
                total,
                f"{row['page_hit_at_hit_k']} ({pct(row['page_hit_at_hit_k'], total)})",
                f"{row['page_pool_hits']} ({pct(row['page_pool_hits'], total)})",
                f"{row['rankable_beyond_hit_k']} ({pct(row['rankable_beyond_hit_k'], miss)})",
                f"{row['missing_from_prediction_pool']} ({pct(row['missing_from_prediction_pool'], miss)})",
                "yes" if row["statement_supported"] else "no/mixed",
            ]
        )
    lines.extend(
        md_table(
            [
                "dataset",
                "qids",
                f"page@{hit_k}",
                f"page@{pool_k}",
                f"misses rankable {hit_k + 1}-{pool_k}",
                f"misses missing >{pool_k}",
                "promotion bottleneck?",
            ],
            summary_rows,
        )
    )
    lines.extend(["", "## Page Recall by Rank Cutoff", ""])
    recall_rows = []
    for row in rows:
        total = int(row["total"])
        recall_rows.append(
            [row["dataset"]]
            + [f"{row['page_hits'].get(int(k), 0)} ({pct(row['page_hits'].get(int(k), 0), total)})" for k in coverage_ks]
        )
    lines.extend(md_table(["dataset"] + [f"page@{k}" for k in coverage_ks], recall_rows))

    lines.extend(["", "## Multi-Gold Page Completeness", ""])
    complete_rows = []
    for row in rows:
        total = int(row["total"])
        instance_total = int(row["gold_page_instance_total"])
        complete_rows.append(
            [
                row["dataset"],
                f"{row['all_gold_page_hit_at_hit_k']} ({pct(row['all_gold_page_hit_at_hit_k'], total)})",
                f"{row['all_gold_page_pool_hits']} ({pct(row['all_gold_page_pool_hits'], total)})",
                f"{row['gold_page_instance_hit_at_hit_k']} ({pct(row['gold_page_instance_hit_at_hit_k'], instance_total)})",
                f"{row['gold_page_instance_pool_hits']} ({pct(row['gold_page_instance_pool_hits'], instance_total)})",
            ]
        )
    lines.extend(
        md_table(
            [
                "dataset",
                f"all gold pages @ {hit_k}",
                f"all gold pages @ {pool_k}",
                f"gold page instances @ {hit_k}",
                f"gold page instances @ {pool_k}",
            ],
            complete_rows,
        )
    )

    lines.extend(["", "## Dense/Sparse Source Pool Coverage", ""])
    source_rows = []
    for row in rows:
        total = int(row["total"])
        source_rows.append(
            [
                row["dataset"],
                f"{row['dense_pool_hits']} ({pct(row['dense_pool_hits'], total)})",
                f"{row['sparse_pool_hits']} ({pct(row['sparse_pool_hits'], total)})",
                f"{row['source_union_pool_hits']} ({pct(row['source_union_pool_hits'], total)})",
                row["source_only_pool_hits"],
            ]
        )
    lines.extend(
        md_table(
            [
                "dataset",
                f"dense page@{pool_k}",
                f"sparse page@{pool_k}",
                f"dense-or-sparse page@{pool_k}",
                "source has but audited prediction misses",
            ],
            source_rows,
        )
    )

    lines.extend(["", "## Report Rule", ""])
    lines.extend(
        [
            f"- If `page@{pool_k}` is high and many top-{hit_k} misses are `rankable {hit_k + 1}-{pool_k}`, the problem is mostly promotion.",
            f"- If many top-{hit_k} misses are missing beyond top-{pool_k}, the problem is discovery and fixed-pool reranking cannot solve those cases.",
            "- If the dense-or-sparse source union has higher pool coverage than the audited prediction, a better fusion step may recover additional cases without external expansion.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_csv(rows: list[dict[str, Any]], path: Path, coverage_ks: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "total",
        "hit_k",
        "pool_k",
        "miss_at_hit_k",
        "rankable_beyond_hit_k",
        "missing_from_prediction_pool",
        "promotion_share_of_misses",
        "discovery_share_of_misses",
        "statement_supported",
        "dense_pool_hits",
        "sparse_pool_hits",
        "source_union_pool_hits",
        "source_only_pool_hits",
        "all_gold_page_hit_at_hit_k",
        "all_gold_page_pool_hits",
        "gold_page_instance_total",
        "gold_page_instance_hit_at_hit_k",
        "gold_page_instance_pool_hits",
    ] + [f"page@{k}" for k in coverage_ks] + [f"doc@{k}" for k in coverage_ks]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in fieldnames if key in row}
            for k in coverage_ks:
                flat[f"page@{k}"] = row["page_hits"].get(int(k), 0)
                flat[f"doc@{k}"] = row["doc_hits"].get(int(k), 0)
            writer.writerow(flat)


def main() -> None:
    args = parse_args()
    coverage_ks = sorted(set(int(k) for k in args.coverage_ks + [args.hit_k, args.pool_k]))
    rows = [
        summarize_payload(Path(path), read_json(Path(path)), pool_k=int(args.pool_k), hit_k=int(args.hit_k), coverage_ks=coverage_ks)
        for path in args.input_json
    ]
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(rows, pool_k=int(args.pool_k), hit_k=int(args.hit_k), coverage_ks=coverage_ks), encoding="utf-8")
    if args.output_csv:
        write_csv(rows, Path(args.output_csv), coverage_ks)
    print(f"saved_report_md={output_md}")
    if args.output_csv:
        print(f"saved_report_csv={args.output_csv}")
    print(f"dataset_count={len(rows)}")


if __name__ == "__main__":
    main()
