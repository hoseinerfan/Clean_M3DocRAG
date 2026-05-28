#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SCRIPT_PATH = Path(__file__).resolve().with_name("graph_rerank_page_retrieval_predictions.py")
SPEC = importlib.util.spec_from_file_location("graph_rerank_page_retrieval_predictions", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load graph reranker module: {SCRIPT_PATH}")
GRAPH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GRAPH
SPEC.loader.exec_module(GRAPH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a dense_sparse_agreement doc-doc ablation summary by recomputing "
            "selected docs, agreement docs, raw agreement pairs, and retained edge counts "
            "from the original dense/sparse prediction files."
        )
    )
    parser.add_argument("summary_json", help="dense_sparse_agreement summary JSON.")
    parser.add_argument(
        "--sample",
        type=int,
        default=5,
        help="Number of sample qids to print.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any mismatch is found.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def per_qid_by_qid(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in summary.get("per_qid", []):
        if not isinstance(row, dict):
            continue
        qid = str(row.get("qid", "")).strip()
        if qid:
            rows[qid] = row
    return rows


def build_records_and_doc_ranks(
    dense_row: dict[str, Any],
    sparse_row: dict[str, Any],
    *,
    dense_top_pages: int,
    sparse_top_pages: int,
) -> tuple[dict[str, Any], dict[str, int], dict[str, int]]:
    dense_pages = GRAPH.ranked_unique_pages(
        dense_row.get("page_retrieval_results", []),
        dense_top_pages,
    )
    sparse_pages = GRAPH.ranked_unique_pages(
        sparse_row.get("page_retrieval_results", []),
        sparse_top_pages,
    )
    dense_doc_ranks = GRAPH.first_doc_rank_map(dense_pages)
    sparse_doc_ranks = GRAPH.first_doc_rank_map(sparse_pages)

    records: dict[str, Any] = {}
    for source_name, source_pages in [("dense", dense_pages), ("sparse", sparse_pages)]:
        for doc_id, page_idx, score, rank in source_pages:
            uid = GRAPH.page_uid(doc_id, page_idx)
            record = records.get(uid)
            if record is None:
                record = GRAPH.PageRecord(doc_id=doc_id, page_idx=page_idx)
                records[uid] = record
            if source_name == "dense":
                record.dense_rank = rank
                record.dense_score = score
            else:
                record.sparse_rank = rank
                record.sparse_score = score
    return records, dense_doc_ranks, sparse_doc_ranks


def make_doc_doc_args(summary: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        rrf_k=float(summary.get("rrf_k", 10.0)),
        doc_doc_top_docs=int(summary.get("doc_doc_top_docs", 20)),
        doc_doc_max_edges_per_doc=int(summary.get("doc_doc_max_edges_per_doc", 8)),
        doc_doc_edge_weight=float(summary.get("doc_doc_edge_weight", 0.1)),
    )


def check_summary_mode(summary: dict[str, Any], problems: list[str]) -> None:
    if summary.get("doc_doc_edge_mode") != "dense_sparse_agreement":
        problems.append(
            "summary doc_doc_edge_mode is "
            f"{summary.get('doc_doc_edge_mode')!r}, expected 'dense_sparse_agreement'"
        )
    if float(summary.get("doc_doc_edge_weight", 0.0)) <= 0:
        problems.append("summary doc_doc_edge_weight is not positive")
    if float(summary.get("doc_seed_weight", 0.0)) != 0.0:
        problems.append("summary doc_seed_weight should be 0.0 for doc-doc edge ablation")


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_json)
    summary = load_json(summary_path)
    if not isinstance(summary, dict):
        raise TypeError(f"Expected summary JSON object: {summary_path}")

    problems: list[str] = []
    check_summary_mode(summary, problems)

    dense_path = Path(str(summary.get("dense_prediction_json", "")))
    sparse_path = Path(str(summary.get("sparse_prediction_json", "")))
    if not dense_path.is_file():
        problems.append(f"dense prediction missing: {dense_path}")
    if not sparse_path.is_file():
        problems.append(f"sparse prediction missing: {sparse_path}")
    if problems:
        for problem in problems:
            print(f"ERROR {problem}")
        raise SystemExit(1)

    dense_pred = GRAPH.load_prediction(dense_path)
    sparse_pred = GRAPH.load_prediction(sparse_path)
    qid_rows = per_qid_by_qid(summary)
    qids = sorted(qid_rows)
    rerank_args = make_doc_doc_args(summary)
    source_weights = GRAPH.SourceWeights(
        float(summary.get("dense_weight", 1.0)),
        float(summary.get("sparse_weight", 1.0)),
        {},
    )

    dense_top_pages = int(summary.get("dense_top_pages", 1000))
    sparse_top_pages = int(summary.get("sparse_top_pages", 1000))

    selected_counts: list[int] = []
    agreement_doc_counts: list[int] = []
    raw_pair_counts: list[int] = []
    kept_pair_counts: list[int] = []
    directed_edge_counts: list[int] = []
    cap_reduced_qids = 0
    bad_pair_qids = 0
    mismatches: list[str] = []
    samples: list[dict[str, Any]] = []

    for qid in qids:
        dense_row = dense_pred.get(qid)
        sparse_row = sparse_pred.get(qid)
        if dense_row is None or sparse_row is None:
            mismatches.append(f"{qid}: missing dense or sparse prediction row")
            continue

        records, dense_doc_ranks, sparse_doc_ranks = build_records_and_doc_ranks(
            dense_row,
            sparse_row,
            dense_top_pages=dense_top_pages,
            sparse_top_pages=sparse_top_pages,
        )
        selected_doc_ids, doc_best_ranks = GRAPH.selected_doc_doc_ids(
            records=records,
            dense_doc_ranks=dense_doc_ranks,
            sparse_doc_ranks=sparse_doc_ranks,
            args=rerank_args,
        )
        agreement_docs = sorted(
            doc_id
            for doc_id in selected_doc_ids
            if doc_id in dense_doc_ranks and doc_id in sparse_doc_ranks
        )
        pair_scores = GRAPH.dense_sparse_agreement_doc_doc_scores(
            selected_doc_ids=selected_doc_ids,
            dense_doc_ranks=dense_doc_ranks,
            sparse_doc_ranks=sparse_doc_ranks,
            source_weights=source_weights,
            args=rerank_args,
        )
        bad_pairs = [
            pair
            for pair in pair_scores
            if pair[0] not in agreement_docs or pair[1] not in agreement_docs
        ]
        if bad_pairs:
            bad_pair_qids += 1

        graph: dict[str, dict[str, float]] = defaultdict(dict)
        directed_edges, kept_pairs, mean_weight = GRAPH.add_doc_doc_pair_edges(
            graph=graph,
            pair_scores=pair_scores,
            args=rerank_args,
        )

        graph_metadata = qid_rows[qid].get("graph", {})
        observed_selected = int(graph_metadata.get("doc_doc_selected_doc_count", -1))
        observed_raw_pairs = int(
            graph_metadata.get("doc_doc_dense_sparse_agreement_pair_count", -1)
        )
        observed_kept_pairs = int(graph_metadata.get("doc_doc_edge_pair_count", -1))
        observed_directed = int(graph_metadata.get("doc_doc_edge_count_directed", -1))

        expected_values = {
            "selected": len(selected_doc_ids),
            "raw_pairs": len(pair_scores),
            "kept_pairs": kept_pairs,
            "directed_edges": directed_edges,
        }
        observed_values = {
            "selected": observed_selected,
            "raw_pairs": observed_raw_pairs,
            "kept_pairs": observed_kept_pairs,
            "directed_edges": observed_directed,
        }
        for key, expected in expected_values.items():
            observed = observed_values[key]
            if observed != expected:
                mismatches.append(f"{qid}: {key} observed={observed} expected={expected}")

        selected_counts.append(len(selected_doc_ids))
        agreement_doc_counts.append(len(agreement_docs))
        raw_pair_counts.append(len(pair_scores))
        kept_pair_counts.append(kept_pairs)
        directed_edge_counts.append(directed_edges)
        if len(pair_scores) > kept_pairs:
            cap_reduced_qids += 1
        if len(samples) < int(args.sample):
            samples.append(
                {
                    "qid": qid,
                    "selected_docs": len(selected_doc_ids),
                    "agreement_docs": len(agreement_docs),
                    "raw_pairs": len(pair_scores),
                    "kept_pairs": kept_pairs,
                    "directed_edges": directed_edges,
                    "mean_edge_weight": mean_weight,
                    "first_agreement_docs": agreement_docs[:5],
                    "best_ranks": {
                        doc_id: doc_best_ranks.get(doc_id)
                        for doc_id in agreement_docs[:5]
                    },
                }
            )

    def mean(values: list[int]) -> float:
        return statistics.fmean(values) if values else 0.0

    print(f"summary_json: {summary_path}")
    print(f"qids_checked: {len(qids)}")
    print(f"mismatch_count: {len(mismatches)}")
    print(f"bad_pair_qids: {bad_pair_qids}")
    print(f"cap_reduced_qids: {cap_reduced_qids}")
    print(f"mean_selected_docs: {mean(selected_counts):.3f}")
    print(f"mean_agreement_docs: {mean(agreement_doc_counts):.3f}")
    print(f"mean_raw_agreement_pairs: {mean(raw_pair_counts):.3f}")
    print(f"mean_retained_edge_pairs: {mean(kept_pair_counts):.3f}")
    print(f"mean_directed_edges: {mean(directed_edge_counts):.3f}")
    print()
    print("| qid | selected docs | agreement docs | raw pairs | kept pairs | directed edges | sample agreement docs |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for sample in samples:
        docs = ", ".join(sample["first_agreement_docs"])
        print(
            f"| {sample['qid']} | {sample['selected_docs']} | "
            f"{sample['agreement_docs']} | {sample['raw_pairs']} | "
            f"{sample['kept_pairs']} | {sample['directed_edges']} | {docs} |"
        )

    if mismatches or bad_pair_qids:
        print()
        print("Problems")
        for mismatch in mismatches[:50]:
            print(f"- {mismatch}")
        if len(mismatches) > 50:
            print(f"- ... {len(mismatches) - 50} more mismatches")
        if bad_pair_qids:
            print(
                "- Found pairs whose docs were not both in dense and sparse ranks "
                f"for {bad_pair_qids} qids"
            )
        if args.strict:
            raise SystemExit(1)
    else:
        print()
        print("Dense/sparse agreement audit passed.")


if __name__ == "__main__":
    main()
