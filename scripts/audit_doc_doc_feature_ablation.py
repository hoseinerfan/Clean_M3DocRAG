#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import math
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
            "Audit shared_entity_title_topic or semantic_similarity doc-doc ablation "
            "summaries by recomputing feature pairs from saved inputs."
        )
    )
    parser.add_argument(
        "--feature",
        choices=["shared_entity_title_topic", "semantic_similarity"],
        required=True,
    )
    parser.add_argument("summary_json", help="Feature ablation summary JSON.")
    parser.add_argument("--sample", type=int, default=5)
    parser.add_argument("--strict", action="store_true")
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


def build_records_doc_ranks_and_page_seed(
    dense_row: dict[str, Any],
    sparse_row: dict[str, Any],
    *,
    dense_top_pages: int,
    sparse_top_pages: int,
    source_weights: Any,
    rrf_k: float,
    score_seed_weight: float,
) -> tuple[dict[str, Any], dict[str, int], dict[str, int], dict[str, float]]:
    dense_pages = GRAPH.ranked_unique_pages(
        dense_row.get("page_retrieval_results", []),
        dense_top_pages,
    )
    sparse_pages = GRAPH.ranked_unique_pages(
        sparse_row.get("page_retrieval_results", []),
        sparse_top_pages,
    )
    dense_score_norm = GRAPH.minmax_by_uid(dense_pages)
    sparse_score_norm = GRAPH.minmax_by_uid(sparse_pages)
    dense_doc_ranks = GRAPH.first_doc_rank_map(dense_pages)
    sparse_doc_ranks = GRAPH.first_doc_rank_map(sparse_pages)

    records: dict[str, Any] = {}
    page_seed: dict[str, float] = defaultdict(float)
    for source_name, source_weight, source_pages, source_score_norm in [
        ("dense", source_weights.dense_weight, dense_pages, dense_score_norm),
        ("sparse", source_weights.sparse_weight, sparse_pages, sparse_score_norm),
    ]:
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
            page_seed[uid] += float(source_weight) / (float(rrf_k) + float(rank))
            page_seed[uid] += (
                float(source_weight)
                * float(score_seed_weight)
                * float(source_score_norm.get(uid, 0.0))
            )
    return records, dense_doc_ranks, sparse_doc_ranks, dict(page_seed)


def make_feature_args(summary: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        rrf_k=float(summary.get("rrf_k", 10.0)),
        doc_doc_top_docs=int(summary.get("doc_doc_top_docs", 20)),
        doc_doc_max_edges_per_doc=int(summary.get("doc_doc_max_edges_per_doc", 8)),
        doc_doc_edge_weight=float(summary.get("doc_doc_edge_weight", 0.1)),
        doc_doc_min_shared_signals=int(summary.get("doc_doc_min_shared_signals", 1)),
        doc_doc_max_signal_doc_matches=int(
            summary.get("doc_doc_max_signal_doc_matches", 8)
        ),
        doc_doc_min_semantic_similarity=float(
            summary.get("doc_doc_min_semantic_similarity", 0.35)
        ),
        doc_doc_semantic_top_terms=int(summary.get("doc_doc_semantic_top_terms", 64)),
        heading_breadcrumb_min_token_len=int(summary.get("heading_breadcrumb_min_token_len", 3)),
    )


def as_list(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [part for part in value.split() if part]
    return list(default)


def load_catalog(summary: dict[str, Any]) -> Any:
    doc_pages = Path(str(summary.get("doc_pages_jsonl", "")))
    if not doc_pages.is_file():
        raise FileNotFoundError(f"doc_pages_jsonl missing: {doc_pages}")
    return GRAPH.load_doc_page_catalog(
        doc_pages,
        heading_fields=as_list(summary.get("heading_breadcrumb_field"), ["markdown", "text"]),
        load_page_breadcrumbs=True,
        heading_max_per_page=int(summary.get("heading_breadcrumb_max_headings_per_page", 8)),
        heading_min_tokens=int(summary.get("heading_breadcrumb_min_tokens", 1)),
        heading_min_token_len=int(summary.get("heading_breadcrumb_min_token_len", 3)),
        entity_fields=as_list(
            summary.get("entity_alias_field"),
            ["ocr_text", "vlm_text", "markdown", "text", "page_text", "content"],
        ),
        load_page_entities=True,
        entity_max_per_page=int(summary.get("entity_alias_max_entities_per_page", 24)),
        entity_min_token_len=int(summary.get("entity_alias_min_token_len", 2)),
    )


def maybe_sparse_index(summary: dict[str, Any], feature: str) -> Any | None:
    if feature != "semantic_similarity":
        return None
    sparse_index_path = Path(str(summary.get("splade_index_pt", "")))
    if not sparse_index_path.is_file():
        raise FileNotFoundError(f"splade_index_pt missing: {sparse_index_path}")
    return GRAPH.SparsePageIndex(sparse_index_path, load_postings=False)


def compare_int(
    *,
    mismatches: list[str],
    qid: str,
    name: str,
    observed: Any,
    expected: int,
) -> None:
    try:
        observed_int = int(observed)
    except (TypeError, ValueError):
        observed_int = -1
    if observed_int != int(expected):
        mismatches.append(f"{qid}: {name} observed={observed_int} expected={expected}")


def compare_bool(
    *,
    mismatches: list[str],
    qid: str,
    name: str,
    observed: Any,
    expected: bool,
) -> None:
    if bool(observed) is not bool(expected):
        mismatches.append(f"{qid}: {name} observed={observed!r} expected={expected!r}")


def mean(values: list[int]) -> float:
    return statistics.fmean(values) if values else 0.0


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_json)
    summary = load_json(summary_path)
    if not isinstance(summary, dict):
        raise TypeError(f"Expected summary JSON object: {summary_path}")
    if summary.get("doc_doc_edge_mode") != args.feature:
        raise ValueError(
            f"summary doc_doc_edge_mode={summary.get('doc_doc_edge_mode')!r}, "
            f"expected {args.feature!r}"
        )
    if float(summary.get("doc_seed_weight", 0.0)) != 0.0:
        raise ValueError("doc_seed_weight should be 0.0 for doc-doc feature ablations")

    dense_path = Path(str(summary.get("dense_prediction_json", "")))
    sparse_path = Path(str(summary.get("sparse_prediction_json", "")))
    if not dense_path.is_file():
        raise FileNotFoundError(f"dense prediction missing: {dense_path}")
    if not sparse_path.is_file():
        raise FileNotFoundError(f"sparse prediction missing: {sparse_path}")
    dense_pred = GRAPH.load_prediction(dense_path)
    sparse_pred = GRAPH.load_prediction(sparse_path)
    catalog = load_catalog(summary) if args.feature == "shared_entity_title_topic" else None
    sparse_index = maybe_sparse_index(summary, args.feature)

    qid_rows = per_qid_by_qid(summary)
    qids = sorted(qid_rows)
    rerank_args = make_feature_args(summary)
    source_weights = GRAPH.SourceWeights(
        float(summary.get("dense_weight", 1.0)),
        float(summary.get("sparse_weight", 1.0)),
        {},
    )
    dense_top_pages = int(summary.get("dense_top_pages", 1000))
    sparse_top_pages = int(summary.get("sparse_top_pages", 1000))
    rrf_k = float(summary.get("rrf_k", 10.0))
    score_seed_weight = float(summary.get("score_seed_weight", 0.0))

    selected_counts: list[int] = []
    raw_pair_counts: list[int] = []
    kept_pair_counts: list[int] = []
    directed_edge_counts: list[int] = []
    signal_doc_counts: list[int] = []
    signal_counts: list[int] = []
    vector_doc_counts: list[int] = []
    active_qids = 0
    cap_reduced_qids = 0
    mismatches: list[str] = []
    samples: list[dict[str, Any]] = []

    for qid in qids:
        dense_row = dense_pred.get(qid)
        sparse_row = sparse_pred.get(qid)
        if dense_row is None or sparse_row is None:
            mismatches.append(f"{qid}: missing dense or sparse prediction row")
            continue
        records, dense_doc_ranks, sparse_doc_ranks, page_seed = (
            build_records_doc_ranks_and_page_seed(
                dense_row,
                sparse_row,
                dense_top_pages=dense_top_pages,
                sparse_top_pages=sparse_top_pages,
                source_weights=source_weights,
                rrf_k=rrf_k,
                score_seed_weight=score_seed_weight,
            )
        )
        selected_doc_ids, _doc_best_ranks = GRAPH.selected_doc_doc_ids(
            records=records,
            dense_doc_ranks=dense_doc_ranks,
            sparse_doc_ranks=sparse_doc_ranks,
            args=rerank_args,
        )

        if args.feature == "shared_entity_title_topic":
            assert catalog is not None
            pair_scores, feature_metadata = GRAPH.shared_entity_title_topic_doc_doc_scores(
                selected_doc_ids=selected_doc_ids,
                records=records,
                page_breadcrumbs=catalog.page_breadcrumbs,
                page_entities=catalog.page_entities,
                args=rerank_args,
            )
            expected_metadata = {
                "doc_doc_shared_signal_doc_count": feature_metadata[
                    "doc_doc_shared_signal_doc_count"
                ],
                "doc_doc_shared_signal_count": feature_metadata["doc_doc_shared_signal_count"],
                "doc_doc_shared_dropped_broad_signal_count": feature_metadata[
                    "doc_doc_shared_dropped_broad_signal_count"
                ],
                "doc_doc_shared_pair_count": feature_metadata["doc_doc_shared_pair_count"],
            }
            signal_doc_counts.append(int(feature_metadata["doc_doc_shared_signal_doc_count"]))
            signal_counts.append(int(feature_metadata["doc_doc_shared_signal_count"]))
        else:
            pair_scores, feature_metadata = GRAPH.semantic_similarity_doc_doc_scores(
                selected_doc_ids=selected_doc_ids,
                records=records,
                page_seed=page_seed,
                sparse_index=sparse_index,
                args=rerank_args,
            )
            expected_metadata = {
                "doc_doc_semantic_available": feature_metadata["doc_doc_semantic_available"],
                "doc_doc_semantic_vector_doc_count": feature_metadata[
                    "doc_doc_semantic_vector_doc_count"
                ],
                "doc_doc_semantic_pair_count": feature_metadata["doc_doc_semantic_pair_count"],
                "doc_doc_semantic_missing_sparse_index": feature_metadata[
                    "doc_doc_semantic_missing_sparse_index"
                ],
            }
            vector_doc_counts.append(int(feature_metadata["doc_doc_semantic_vector_doc_count"]))

        graph: dict[str, dict[str, float]] = defaultdict(dict)
        directed_edges, kept_pairs, mean_weight = GRAPH.add_doc_doc_pair_edges(
            graph=graph,
            pair_scores=pair_scores,
            args=rerank_args,
        )
        graph_metadata = qid_rows[qid].get("graph", {})
        compare_int(
            mismatches=mismatches,
            qid=qid,
            name="doc_doc_selected_doc_count",
            observed=graph_metadata.get("doc_doc_selected_doc_count"),
            expected=len(selected_doc_ids),
        )
        for key, expected in expected_metadata.items():
            if isinstance(expected, bool):
                compare_bool(
                    mismatches=mismatches,
                    qid=qid,
                    name=key,
                    observed=graph_metadata.get(key),
                    expected=expected,
                )
            else:
                compare_int(
                    mismatches=mismatches,
                    qid=qid,
                    name=key,
                    observed=graph_metadata.get(key),
                    expected=int(expected),
                )
        compare_int(
            mismatches=mismatches,
            qid=qid,
            name="doc_doc_edge_pair_count",
            observed=graph_metadata.get("doc_doc_edge_pair_count"),
            expected=kept_pairs,
        )
        compare_int(
            mismatches=mismatches,
            qid=qid,
            name="doc_doc_edge_count_directed",
            observed=graph_metadata.get("doc_doc_edge_count_directed"),
            expected=directed_edges,
        )

        selected_counts.append(len(selected_doc_ids))
        raw_pair_counts.append(len(pair_scores))
        kept_pair_counts.append(kept_pairs)
        directed_edge_counts.append(directed_edges)
        if kept_pairs > 0:
            active_qids += 1
        if len(pair_scores) > kept_pairs:
            cap_reduced_qids += 1
        if len(samples) < int(args.sample):
            nonzero_pairs = sorted(pair_scores.items(), key=lambda item: (-item[1], item[0]))[:5]
            samples.append(
                {
                    "qid": qid,
                    "selected_docs": len(selected_doc_ids),
                    "raw_pairs": len(pair_scores),
                    "kept_pairs": kept_pairs,
                    "directed_edges": directed_edges,
                    "mean_edge_weight": mean_weight,
                    "pairs": nonzero_pairs,
                    "feature_metadata": feature_metadata,
                }
            )

    print(f"summary_json: {summary_path}")
    print(f"feature: {args.feature}")
    print(f"qids_checked: {len(qids)}")
    print(f"mismatch_count: {len(mismatches)}")
    print(f"active_edge_qids: {active_qids}")
    print(f"cap_reduced_qids: {cap_reduced_qids}")
    print(f"mean_selected_docs: {mean(selected_counts):.3f}")
    if args.feature == "shared_entity_title_topic":
        print(f"mean_signal_doc_count: {mean(signal_doc_counts):.3f}")
        print(f"mean_kept_signal_count: {mean(signal_counts):.3f}")
    else:
        print(f"mean_vector_doc_count: {mean(vector_doc_counts):.3f}")
    print(f"mean_raw_pairs: {mean(raw_pair_counts):.3f}")
    print(f"mean_retained_edge_pairs: {mean(kept_pair_counts):.3f}")
    print(f"mean_directed_edges: {mean(directed_edge_counts):.3f}")
    print()
    print("| qid | selected docs | raw pairs | kept pairs | directed edges | sample pairs |")
    print("|---|---:|---:|---:|---:|---|")
    for sample in samples:
        pair_text = ", ".join(
            f"{left}-{right}:{score:.3f}"
            for (left, right), score in sample["pairs"]
        )
        print(
            f"| {sample['qid']} | {sample['selected_docs']} | "
            f"{sample['raw_pairs']} | {sample['kept_pairs']} | "
            f"{sample['directed_edges']} | {pair_text} |"
        )

    if mismatches:
        print()
        print("Problems")
        for mismatch in mismatches[:50]:
            print(f"- {mismatch}")
        if len(mismatches) > 50:
            print(f"- ... {len(mismatches) - 50} more mismatches")
        if args.strict:
            raise SystemExit(1)
    else:
        print()
        print(f"{args.feature} audit passed.")


if __name__ == "__main__":
    main()
