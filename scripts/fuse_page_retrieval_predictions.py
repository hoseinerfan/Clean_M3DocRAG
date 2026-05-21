#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse dense and sparse page-retrieval predictions into a doc-ordered shortlist "
            "suitable for cue-verifier experiments."
        )
    )
    parser.add_argument("--dense-prediction-json", required=True)
    parser.add_argument("--sparse-prediction-json", required=True)
    parser.add_argument("--gold", help="Optional MMQA_<split>.jsonl for summary metrics.")
    parser.add_argument(
        "--fusion-mode",
        default="dense_sparse_doc_union",
        choices=["dense_sparse_doc_union", "doc_rrf"],
        help=(
            "How to combine dense and sparse doc shortlists. "
            "'dense_sparse_doc_union' keeps a dense head and appends sparse docs; "
            "'doc_rrf' uses reciprocal-rank fusion at the doc level."
        ),
    )
    parser.add_argument("--dense-top-docs", type=int, default=20)
    parser.add_argument("--sparse-top-docs", type=int, default=20)
    parser.add_argument("--dense-keep-docs", type=int, default=8)
    parser.add_argument("--sparse-add-docs", type=int, default=4)
    parser.add_argument(
        "--final-top-docs",
        type=int,
        default=0,
        help=(
            "Optional final fused shortlist size. Use 0 to default to "
            "dense_keep_docs + sparse_add_docs."
        ),
    )
    parser.add_argument(
        "--rrf-k",
        type=float,
        default=10.0,
        help="RRF smoothing constant used when --fusion-mode=doc_rrf. Default: 10.",
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=1.0,
        help="Dense branch weight used when --fusion-mode=doc_rrf. Default: 1.0.",
    )
    parser.add_argument(
        "--sparse-weight",
        type=float,
        default=1.0,
        help="Sparse branch weight used when --fusion-mode=doc_rrf. Default: 1.0.",
    )
    parser.add_argument("--output-prediction-json", required=True)
    parser.add_argument("--output-summary-json", required=True)
    return parser.parse_args()


def load_prediction(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Prediction JSON must be an object: {path}")
    return payload


def dedupe_doc_best_rows(rows: list[list[object]], top_docs: int) -> list[list[object]]:
    result: list[list[object]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, list) or len(row) < 3:
            continue
        doc_id = str(row[0]).strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        result.append([doc_id, int(row[1]), float(row[2])])
        if len(result) >= top_docs:
            break
    return result


def first_doc_rows(rows: list[list[object]], top_docs: int) -> tuple[list[list[object]], dict[str, int]]:
    deduped = dedupe_doc_best_rows(rows, top_docs)
    ranks = {str(row[0]): idx for idx, row in enumerate(deduped, start=1)}
    return deduped, ranks


def pick_output_row_for_doc(
    *,
    doc_id: str,
    dense_row_map: dict[str, list[object]],
    sparse_row_map: dict[str, list[object]],
    dense_rank_map: dict[str, int],
    sparse_rank_map: dict[str, int],
) -> list[object]:
    dense_row = dense_row_map.get(doc_id)
    sparse_row = sparse_row_map.get(doc_id)
    if dense_row is None and sparse_row is None:
        raise KeyError(f"Doc id missing from both row maps: {doc_id}")
    if dense_row is None:
        return sparse_row  # type: ignore[return-value]
    if sparse_row is None:
        return dense_row
    dense_rank = dense_rank_map.get(doc_id, 10**9)
    sparse_rank = sparse_rank_map.get(doc_id, 10**9)
    if dense_rank <= sparse_rank:
        return dense_row
    return sparse_row


def fuse_doc_union(
    *,
    dense_docs: list[list[object]],
    sparse_docs: list[list[object]],
    dense_keep_docs: int,
    sparse_add_docs: int,
) -> list[str]:
    selected_doc_ids: list[str] = []
    seen_docs: set[str] = set()
    for row in dense_docs[:dense_keep_docs]:
        doc_id = str(row[0])
        if doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)
        selected_doc_ids.append(doc_id)
    for row in sparse_docs:
        if len(selected_doc_ids) >= dense_keep_docs + sparse_add_docs:
            break
        doc_id = str(row[0])
        if doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)
        selected_doc_ids.append(doc_id)
    return selected_doc_ids


def fuse_doc_rrf(
    *,
    dense_rank_map: dict[str, int],
    sparse_rank_map: dict[str, int],
    dense_weight: float,
    sparse_weight: float,
    rrf_k: float,
    final_top_docs: int,
) -> tuple[list[str], dict[str, float]]:
    doc_ids = sorted(set(dense_rank_map) | set(sparse_rank_map))
    fused_scores: dict[str, float] = {}
    for doc_id in doc_ids:
        score = 0.0
        dense_rank = dense_rank_map.get(doc_id)
        sparse_rank = sparse_rank_map.get(doc_id)
        if dense_rank is not None:
            score += float(dense_weight) / (float(rrf_k) + float(dense_rank))
        if sparse_rank is not None:
            score += float(sparse_weight) / (float(rrf_k) + float(sparse_rank))
        fused_scores[doc_id] = score
    ranked_doc_ids = sorted(
        doc_ids,
        key=lambda doc_id: (
            -fused_scores[doc_id],
            min(dense_rank_map.get(doc_id, 10**9), sparse_rank_map.get(doc_id, 10**9)),
            dense_rank_map.get(doc_id, 10**9),
            sparse_rank_map.get(doc_id, 10**9),
            doc_id,
        ),
    )
    return ranked_doc_ids[:final_top_docs], fused_scores


def load_gold_rows(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("qid", "")).strip()
            if qid:
                rows[qid] = row
    return rows


def first_gold_doc_rank(rows: list[list[object]], gold_doc_ids: set[str]) -> int | None:
    seen: set[str] = set()
    rank = 0
    for row in rows:
        doc_id = str(row[0])
        if doc_id in seen:
            continue
        seen.add(doc_id)
        rank += 1
        if doc_id in gold_doc_ids:
            return rank
    return None


def main() -> None:
    args = parse_args()

    dense_pred = load_prediction(Path(args.dense_prediction_json))
    sparse_pred = load_prediction(Path(args.sparse_prediction_json))
    if set(dense_pred) != set(sparse_pred):
        missing_in_sparse = sorted(set(dense_pred) - set(sparse_pred))
        missing_in_dense = sorted(set(sparse_pred) - set(dense_pred))
        raise ValueError(
            "Dense/sparse qid sets differ: "
            f"missing_in_sparse={missing_in_sparse[:10]} "
            f"missing_in_dense={missing_in_dense[:10]}"
        )

    fused_payload: dict[str, dict] = {}
    summary_rows: list[dict] = []
    gold_rows = load_gold_rows(Path(args.gold)) if args.gold else {}
    final_top_docs = int(args.final_top_docs) if int(args.final_top_docs) > 0 else int(args.dense_keep_docs) + int(args.sparse_add_docs)

    for qid in sorted(dense_pred):
        dense_rows = dense_pred[qid].get("page_retrieval_results", [])
        sparse_rows = sparse_pred[qid].get("page_retrieval_results", [])
        dense_docs, dense_rank_map = first_doc_rows(dense_rows, int(args.dense_top_docs))
        sparse_docs, sparse_rank_map = first_doc_rows(sparse_rows, int(args.sparse_top_docs))
        dense_row_map = {str(row[0]): row for row in dense_docs}
        sparse_row_map = {str(row[0]): row for row in sparse_docs}

        fused_score_map: dict[str, float] = {}
        if args.fusion_mode == "dense_sparse_doc_union":
            selected_doc_ids = fuse_doc_union(
                dense_docs=dense_docs,
                sparse_docs=sparse_docs,
                dense_keep_docs=int(args.dense_keep_docs),
                sparse_add_docs=int(args.sparse_add_docs),
            )
        elif args.fusion_mode == "doc_rrf":
            selected_doc_ids, fused_score_map = fuse_doc_rrf(
                dense_rank_map=dense_rank_map,
                sparse_rank_map=sparse_rank_map,
                dense_weight=float(args.dense_weight),
                sparse_weight=float(args.sparse_weight),
                rrf_k=float(args.rrf_k),
                final_top_docs=final_top_docs,
            )
        else:
            raise ValueError(f"Unsupported fusion_mode: {args.fusion_mode}")

        selected = [
            pick_output_row_for_doc(
                doc_id=doc_id,
                dense_row_map=dense_row_map,
                sparse_row_map=sparse_row_map,
                dense_rank_map=dense_rank_map,
                sparse_rank_map=sparse_rank_map,
            )
            for doc_id in selected_doc_ids
        ]

        fused_payload[qid] = {
            "pred_answer": dense_pred[qid].get("pred_answer", ""),
            "page_retrieval_results": selected,
            "qid": qid,
            "question": dense_pred[qid].get("question", ""),
            "top_retrieved_docs": [row[0] for row in selected[:10]],
            "reranker_metadata": {
                "fusion_method": args.fusion_mode,
                "dense_prediction_json": args.dense_prediction_json,
                "sparse_prediction_json": args.sparse_prediction_json,
                "dense_keep_docs": int(args.dense_keep_docs),
                "sparse_add_docs": int(args.sparse_add_docs),
                "dense_top_docs": int(args.dense_top_docs),
                "sparse_top_docs": int(args.sparse_top_docs),
                "final_top_docs": int(final_top_docs),
                "rrf_k": float(args.rrf_k),
                "dense_weight": float(args.dense_weight),
                "sparse_weight": float(args.sparse_weight),
            },
        }

        row_summary = {
            "qid": qid,
            "fused_doc_count": len(selected),
            "fused_top_doc_ids": [row[0] for row in selected],
        }
        if fused_score_map:
            row_summary["fused_doc_scores_top10"] = [
                {"doc_id": doc_id, "score": float(fused_score_map[doc_id])}
                for doc_id in selected_doc_ids[:10]
            ]
        if gold_rows:
            gold_doc_ids = {
                str(item["doc_id"]).strip() for item in gold_rows[qid].get("supporting_context", [])
            }
            row_summary["reranked_first_gold_doc_rank"] = first_gold_doc_rank(selected, gold_doc_ids)
        summary_rows.append(row_summary)

    output_prediction_json = Path(args.output_prediction_json)
    output_prediction_json.parent.mkdir(parents=True, exist_ok=True)
    output_prediction_json.write_text(json.dumps(fused_payload, indent=2) + "\n", encoding="utf-8")

    summary = {
        "fusion_method": args.fusion_mode,
        "qid_count": len(summary_rows),
        "dense_keep_docs": int(args.dense_keep_docs),
        "sparse_add_docs": int(args.sparse_add_docs),
        "dense_top_docs": int(args.dense_top_docs),
        "sparse_top_docs": int(args.sparse_top_docs),
        "final_top_docs": int(final_top_docs),
        "rrf_k": float(args.rrf_k),
        "dense_weight": float(args.dense_weight),
        "sparse_weight": float(args.sparse_weight),
        "mean_fused_doc_count": (
            sum(int(row["fused_doc_count"]) for row in summary_rows) / len(summary_rows)
            if summary_rows
            else None
        ),
        "per_qid": summary_rows,
    }
    if gold_rows:
        summary["reranked_top4_doc_count"] = sum(
            1
            for row in summary_rows
            if row.get("reranked_first_gold_doc_rank") is not None
            and int(row["reranked_first_gold_doc_rank"]) <= 4
        )
        summary["reranked_top20_doc_count"] = sum(
            1
            for row in summary_rows
            if row.get("reranked_first_gold_doc_rank") is not None
            and int(row["reranked_first_gold_doc_rank"]) <= 20
        )

    output_summary_json = Path(args.output_summary_json)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"saved_prediction: {output_prediction_json}")
    print(f"saved_summary: {output_summary_json}")
    print(f"qid_count: {len(summary_rows)}")
    if gold_rows:
        print(f"reranked_top4_doc_count: {summary['reranked_top4_doc_count']}")
        print(f"reranked_top20_doc_count: {summary['reranked_top20_doc_count']}")


if __name__ == "__main__":
    main()
