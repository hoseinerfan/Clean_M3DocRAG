#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_prediction(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "predictions" in payload and isinstance(
        payload["predictions"],
        (dict, list),
    ):
        payload = payload["predictions"]

    rows_by_qid: dict[str, dict[str, Any]] = {}
    iterable: Any
    if isinstance(payload, list):
        iterable = enumerate(payload)
    elif isinstance(payload, dict):
        iterable = payload.items()
    else:
        raise TypeError(f"Prediction JSON must be a list or object: {path}")

    for raw_key, row in iterable:
        if not isinstance(row, dict):
            raise TypeError(f"Prediction row must be an object: {path} key={raw_key!r}")
        qid = str(row.get("qid", "")).strip() or str(raw_key).strip()
        if not qid:
            raise ValueError(f"Prediction row is missing qid: {path} key={raw_key!r}")
        if qid in rows_by_qid:
            raise ValueError(f"Duplicate qid after normalization: {qid} ({path})")
        rows_by_qid[qid] = row
    return rows_by_qid


def page_uid(doc_id: object, page_idx: object) -> str:
    return f"{doc_id}_page{int(page_idx)}"


def ranked_page_uids(rows: list[Any], limit: int | None = None) -> list[str]:
    pages: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, list) or len(row) < 2:
            continue
        try:
            uid = page_uid(row[0], row[1])
        except (TypeError, ValueError):
            continue
        if uid in seen:
            continue
        seen.add(uid)
        pages.append(uid)
        if limit is not None and len(pages) >= limit:
            break
    return pages


def ranked_doc_ids(rows: list[Any], limit: int | None = None) -> list[str]:
    doc_ids: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, list) or not row:
            continue
        doc_id = str(row[0])
        if doc_id in seen:
            continue
        seen.add(doc_id)
        doc_ids.append(doc_id)
        if limit is not None and len(doc_ids) >= limit:
            break
    return doc_ids


def first_rank(ranked: list[str], gold: set[str]) -> int | None:
    for idx, value in enumerate(ranked, start=1):
        if value in gold:
            return idx
    return None


def hit_at(rank: int | None, topk: int) -> bool:
    return rank is not None and int(rank) <= int(topk)


def movement_for_hit(
    baseline_rank: int | None,
    candidate_rank: int | None,
    topk: int,
) -> str:
    baseline_hit = hit_at(baseline_rank, topk)
    candidate_hit = hit_at(candidate_rank, topk)
    if not baseline_hit and candidate_hit:
        return "recovered"
    if baseline_hit and not candidate_hit:
        return "lost"
    if baseline_rank is None and candidate_rank is None:
        return "missing_in_both"
    if baseline_rank is not None and candidate_rank is not None and candidate_rank < baseline_rank:
        return "improved_rank"
    if baseline_rank is not None and candidate_rank is not None and candidate_rank > baseline_rank:
        return "worsened_rank"
    return "unchanged"


def gold_page_uids(row: dict[str, Any]) -> set[str]:
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
            uids.add(page_uid(doc_id, page_idx))
    return uids


def gold_doc_ids(row: dict[str, Any]) -> set[str]:
    metadata = row.get("metadata", {})
    doc_ids = {
        str(value).strip()
        for value in metadata.get("gold_doc_ids", [])
        if str(value).strip()
    }
    for ctx in row.get("supporting_context", []):
        doc_id = str(ctx.get("doc_id", "")).strip()
        if doc_id:
            doc_ids.add(doc_id)
    return doc_ids


def recall_at_k(ranked: list[str], gold: set[str], topk: int) -> float | None:
    if not gold:
        return None
    return len(set(ranked[:topk]) & gold) / float(len(gold))


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.fmean(values)


def format_float(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.replace(",", " ").split() if item.strip()]


def prediction_rows(row: dict[str, Any]) -> list[Any]:
    rows = row.get("page_retrieval_results", [])
    return rows if isinstance(rows, list) else []


def support_vote_threshold(variant_count: int, support_policy: str) -> int:
    if variant_count <= 0:
        return 1
    if support_policy == "unanimity":
        return variant_count
    return variant_count // 2 + 1


def parse_labeled_path(value: str) -> tuple[str, Path]:
    raw = str(value).strip()
    if not raw:
        raise ValueError("Empty prediction path")
    if "=" in raw:
        raw_label, raw_path = raw.split("=", 1)
        label = raw_label.strip()
        path = Path(raw_path.strip())
    else:
        path = Path(raw)
        label = path.stem
    if not label:
        label = path.stem
    return label, path


def dedupe_labeled_paths(values: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    deduped: list[tuple[str, Path]] = []
    seen: set[str] = set()
    label_counts: Counter[str] = Counter()
    for label, path in values:
        key = str(path.expanduser())
        if key in seen:
            continue
        seen.add(key)
        label_counts[label] += 1
        unique_label = label if label_counts[label] == 1 else f"{label}_{label_counts[label]}"
        deduped.append((unique_label, path))
    return deduped


def choose_prediction(
    *,
    reference_row: dict[str, Any],
    primary_row: dict[str, Any],
    support_view_rows: list[tuple[str, dict[str, Any]]],
    topk: int,
    support_topk: int,
    selection_mode: str,
    support_policy: str,
    min_variant_count: int,
) -> dict[str, Any]:
    reference_top = ranked_page_uids(prediction_rows(reference_row), topk)
    primary_top = ranked_page_uids(prediction_rows(primary_row), topk)
    reference_top_set = set(reference_top)
    primary_top_set = set(primary_top)
    promoted_pages = [uid for uid in primary_top if uid not in reference_top_set]
    demoted_pages = [uid for uid in reference_top if uid not in primary_top_set]

    support_view_top_sets = [
        (label, set(ranked_page_uids(prediction_rows(row), support_topk)))
        for label, row in support_view_rows
    ]
    variant_count = len(support_view_top_sets)
    required_votes = support_vote_threshold(variant_count, support_policy)
    promoted_vote_counts = {
        uid: sum(1 for _label, top_set in support_view_top_sets if uid in top_set)
        for uid in promoted_pages
    }
    promoted_vote_sources = {
        uid: [label for label, top_set in support_view_top_sets if uid in top_set]
        for uid in promoted_pages
    }
    stable_promoted_pages = [
        uid for uid in promoted_pages if promoted_vote_counts.get(uid, 0) >= required_votes
    ]
    unstable_promoted_pages = [
        uid for uid in promoted_pages if promoted_vote_counts.get(uid, 0) < required_votes
    ]

    if variant_count < min_variant_count:
        selected_source = "reference"
        reason = "insufficient_variant_count"
    elif not promoted_pages:
        selected_source = "reference"
        reason = "no_candidate_promotions"
    elif selection_mode == "any_promoted_stable" and stable_promoted_pages:
        selected_source = "candidate"
        reason = "stable_candidate_promotion"
    elif selection_mode == "all_promoted_stable" and not unstable_promoted_pages:
        selected_source = "candidate"
        reason = "all_candidate_promotions_stable"
    else:
        selected_source = "reference"
        reason = "unstable_candidate_promotions"

    return {
        "selected_source": selected_source,
        "selection_reason": reason,
        "reference_topk_pages": reference_top,
        "candidate_topk_pages": primary_top,
        "promoted_pages": promoted_pages,
        "demoted_pages": demoted_pages,
        "stable_promoted_pages": stable_promoted_pages,
        "unstable_promoted_pages": unstable_promoted_pages,
        "promoted_page_vote_counts": promoted_vote_counts,
        "promoted_page_vote_sources": promoted_vote_sources,
        "variant_count": variant_count,
        "required_votes": required_votes,
        "promoted_page_count": len(promoted_pages),
        "stable_promoted_page_count": len(stable_promoted_pages),
        "unstable_promoted_page_count": len(unstable_promoted_pages),
        "min_promoted_vote_fraction": (
            min(promoted_vote_counts.values()) / float(variant_count)
            if promoted_vote_counts and variant_count > 0
            else None
        ),
        "mean_promoted_vote_fraction": (
            statistics.fmean(promoted_vote_counts.values()) / float(variant_count)
            if promoted_vote_counts and variant_count > 0
            else None
        ),
    }


def summarize_prediction_rows(rows: list[Any], gold_row: dict[str, Any] | None) -> dict[str, Any]:
    ranked_docs = ranked_doc_ids(rows)
    ranked_pages = ranked_page_uids(rows)
    summary: dict[str, Any] = {
        "doc_count": len(ranked_docs),
        "page_count": len(ranked_pages),
        "top_doc_ids": ranked_docs[:20],
    }
    if gold_row is not None:
        doc_gold = gold_doc_ids(gold_row)
        page_gold = gold_page_uids(gold_row)
        summary["gold_doc_ids"] = sorted(doc_gold)
        summary["gold_page_uids"] = sorted(page_gold)
        summary["reranked_first_gold_doc_rank"] = first_rank(ranked_docs, doc_gold)
        summary["reranked_first_gold_page_rank"] = first_rank(ranked_pages, page_gold)
        summary["contains_gold_doc"] = summary["reranked_first_gold_doc_rank"] is not None
        summary["contains_gold_page"] = summary["reranked_first_gold_page_rank"] is not None
    return summary


def add_selector_metadata(
    *,
    source_row: dict[str, Any],
    qid: str,
    decision: dict[str, Any],
    selector_metadata: dict[str, Any],
) -> dict[str, Any]:
    row = copy.deepcopy(source_row)
    row["qid"] = qid
    rows = prediction_rows(row)
    row["top_retrieved_docs"] = ranked_doc_ids(rows, 10)
    metadata = row.get("reranker_metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    metadata = copy.deepcopy(metadata)
    metadata["stability_selector"] = {
        **selector_metadata,
        **decision,
    }
    row["reranker_metadata"] = metadata
    return row


def evaluate_recall(
    *,
    selected: dict[str, dict[str, Any]],
    gold_by_qid: dict[str, dict[str, Any]],
    qids: list[str],
    k_values: list[int],
) -> dict[str, dict[str, float | None]]:
    page_values: dict[str, list[float]] = {str(k): [] for k in k_values}
    doc_values: dict[str, list[float]] = {str(k): [] for k in k_values}
    for qid in qids:
        gold_row = gold_by_qid.get(qid)
        if gold_row is None:
            continue
        rows = prediction_rows(selected[qid])
        ranked_pages = ranked_page_uids(rows)
        ranked_docs = ranked_doc_ids(rows)
        page_gold = gold_page_uids(gold_row)
        doc_gold = gold_doc_ids(gold_row)
        for k in k_values:
            page_recall = recall_at_k(ranked_pages, page_gold, k)
            doc_recall = recall_at_k(ranked_docs, doc_gold, k)
            if page_recall is not None:
                page_values[str(k)].append(page_recall)
            if doc_recall is not None:
                doc_values[str(k)].append(doc_recall)
    return {
        "page_recall_at_k": {key: mean_or_none(values) for key, values in page_values.items()},
        "doc_recall_at_k": {key: mean_or_none(values) for key, values in doc_values.items()},
    }


def evaluate_movements(
    *,
    baseline: dict[str, dict[str, Any]],
    selected: dict[str, dict[str, Any]],
    gold_by_qid: dict[str, dict[str, Any]],
    qids: list[str],
    topk: int,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for qid in qids:
        if qid not in baseline or qid not in gold_by_qid:
            continue
        gold_pages = gold_page_uids(gold_by_qid[qid])
        baseline_rank = first_rank(
            ranked_page_uids(prediction_rows(baseline[qid])),
            gold_pages,
        )
        selected_rank = first_rank(
            ranked_page_uids(prediction_rows(selected[qid])),
            gold_pages,
        )
        counts[movement_for_hit(baseline_rank, selected_rank, topk)] += 1
    return counts


def build_top_examples(
    *,
    baseline: dict[str, dict[str, Any]] | None,
    selected: dict[str, dict[str, Any]],
    gold_by_qid: dict[str, dict[str, Any]],
    qids: list[str],
    topk: int,
    topn: int,
) -> list[dict[str, Any]]:
    if baseline is None:
        return []
    examples: list[dict[str, Any]] = []
    for qid in qids:
        if qid not in baseline or qid not in gold_by_qid:
            continue
        gold_pages = gold_page_uids(gold_by_qid[qid])
        baseline_rank = first_rank(ranked_page_uids(prediction_rows(baseline[qid])), gold_pages)
        selected_rank = first_rank(ranked_page_uids(prediction_rows(selected[qid])), gold_pages)
        movement = movement_for_hit(baseline_rank, selected_rank, topk)
        if movement != "recovered":
            continue
        examples.append(
            {
                "qid": qid,
                "baseline_first_gold_page_rank": baseline_rank,
                "candidate_first_gold_page_rank": selected_rank,
                "gold_page_uids": sorted(gold_pages),
                "question": gold_by_qid[qid].get("question", ""),
            }
        )
    examples.sort(
        key=lambda row: (
            row["candidate_first_gold_page_rank"]
            if row["candidate_first_gold_page_rank"] is not None
            else 10**9,
            row["baseline_first_gold_page_rank"]
            if row["baseline_first_gold_page_rank"] is not None
            else 10**9,
            row["qid"],
        )
    )
    return examples[:topn]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select between a reference prediction and a candidate prediction using label-free "
            "final-output stability across support graph views. Gold labels are optional and "
            "are used only for reporting, never for selection."
        )
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Prediction JSON for the stable reference method, e.g. old constraint competition.",
    )
    parser.add_argument(
        "--primary-candidate",
        required=True,
        help="Prediction JSON to use when the query-local candidate is selected.",
    )
    parser.add_argument(
        "--candidate-variant",
        action="append",
        default=[],
        help=(
            "Backward-compatible alias for --support-prediction. Additional prediction JSON "
            "used only for support votes."
        ),
    )
    parser.add_argument(
        "--support-prediction",
        action="append",
        default=[],
        help=(
            "Independent graph-view prediction JSON used for consensus support votes. "
            "Use either path or label=path. Repeat for multiple views."
        ),
    )
    parser.add_argument("--baseline", default="", help="Optional baseline prediction for movement reporting.")
    parser.add_argument("--gold", default="", help="Optional gold JSONL for recall and movement reporting.")
    parser.add_argument("--topk", type=int, default=4, help="Top-k boundary used by the selector.")
    parser.add_argument(
        "--support-topk",
        type=int,
        default=0,
        help=(
            "Rank window used when counting support votes from graph views. "
            "Default 0 means use --topk."
        ),
    )
    parser.add_argument("--topn", type=int, default=20, help="Number of recovered examples to store.")
    parser.add_argument(
        "--recall-k-values",
        default="1 2 4 5 10 20 50 100",
        help="Space- or comma-separated k values for optional recall reporting.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["all_promoted_stable", "any_promoted_stable"],
        default="all_promoted_stable",
        help=(
            "all_promoted_stable is conservative: every candidate-promoted top-k page must be "
            "stable. any_promoted_stable is more permissive and meant for diagnostics."
        ),
    )
    parser.add_argument(
        "--support-policy",
        choices=["strict_majority", "unanimity"],
        default="strict_majority",
        help="Vote requirement across support graph views.",
    )
    parser.add_argument(
        "--min-variant-count",
        type=int,
        default=2,
        help=(
            "Minimum available support views required before the selector can choose the candidate. "
            "By default this includes the primary candidate as one support view."
        ),
    )
    parser.add_argument(
        "--exclude-primary-from-support",
        action="store_true",
        help=(
            "Do not count the primary candidate as a support vote. This is stricter and useful "
            "when support files are independent graph views rather than perturbations."
        ),
    )
    parser.add_argument("--output-prediction-json", required=True)
    parser.add_argument("--output-summary-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = load_prediction(Path(args.reference))
    primary = load_prediction(Path(args.primary_candidate))
    support_inputs: list[tuple[str, Path]] = []
    if not bool(args.exclude_primary_from_support):
        support_inputs.append(("primary_candidate", Path(args.primary_candidate)))
    support_inputs.extend(parse_labeled_path(value) for value in args.candidate_variant)
    support_inputs.extend(parse_labeled_path(value) for value in args.support_prediction)
    support_paths = dedupe_labeled_paths(support_inputs)
    support_views = [(label, path, load_prediction(path)) for label, path in support_paths]
    baseline = load_prediction(Path(args.baseline)) if args.baseline else None
    gold_by_qid = (
        {str(row["qid"]): row for row in read_jsonl(Path(args.gold))}
        if args.gold
        else {}
    )

    qid_sets = [set(reference), set(primary)]
    if gold_by_qid:
        qid_sets.append(set(gold_by_qid))
    if baseline is not None:
        qid_sets.append(set(baseline))
    qids = sorted(set.intersection(*qid_sets))
    if not qids:
        raise ValueError("No qids remain after intersecting required prediction inputs.")

    selector_metadata = {
        "selection_method": "ppr_output_stability",
        "selection_mode": args.selection_mode,
        "support_policy": args.support_policy,
        "topk": int(args.topk),
        "support_topk": int(args.support_topk) if int(args.support_topk) > 0 else int(args.topk),
        "min_variant_count": int(args.min_variant_count),
        "include_primary_candidate_in_support": not bool(args.exclude_primary_from_support),
        "reference_prediction_json": args.reference,
        "primary_candidate_prediction_json": args.primary_candidate,
        "support_prediction_jsons": [
            {"label": label, "path": str(path)} for label, path in support_paths
        ],
    }
    support_topk = int(selector_metadata["support_topk"])

    selected: dict[str, dict[str, Any]] = {}
    per_qid: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    promoted_counts: list[float] = []
    stable_promoted_counts: list[float] = []
    min_vote_fractions: list[float] = []

    for qid in qids:
        support_view_rows = [
            (label, prediction[qid])
            for label, _path, prediction in support_views
            if qid in prediction
        ]
        decision = choose_prediction(
            reference_row=reference[qid],
            primary_row=primary[qid],
            support_view_rows=support_view_rows,
            topk=int(args.topk),
            support_topk=support_topk,
            selection_mode=args.selection_mode,
            support_policy=args.support_policy,
            min_variant_count=int(args.min_variant_count),
        )
        source_row = primary[qid] if decision["selected_source"] == "candidate" else reference[qid]
        selected[qid] = add_selector_metadata(
            source_row=source_row,
            qid=qid,
            decision=decision,
            selector_metadata=selector_metadata,
        )
        reason_counts[str(decision["selection_reason"])] += 1
        source_counts[str(decision["selected_source"])] += 1
        promoted_counts.append(float(decision["promoted_page_count"]))
        stable_promoted_counts.append(float(decision["stable_promoted_page_count"]))
        if decision["min_promoted_vote_fraction"] is not None:
            min_vote_fractions.append(float(decision["min_promoted_vote_fraction"]))

        gold_row = gold_by_qid.get(qid)
        row_summary = {
            "qid": qid,
            "question": (
                selected[qid].get("question")
                or primary[qid].get("question")
                or reference[qid].get("question", "")
            ),
            **summarize_prediction_rows(prediction_rows(selected[qid]), gold_row),
            "graph": {
                "selector_selected_source": decision["selected_source"],
                "selector_selection_reason": decision["selection_reason"],
                "selector_variant_count": decision["variant_count"],
                "selector_support_view_count": decision["variant_count"],
                "selector_required_votes": decision["required_votes"],
                "selector_support_topk": support_topk,
                "selector_promoted_page_count": decision["promoted_page_count"],
                "selector_stable_promoted_page_count": decision["stable_promoted_page_count"],
                "selector_unstable_promoted_page_count": decision["unstable_promoted_page_count"],
                "selector_min_promoted_vote_fraction": decision["min_promoted_vote_fraction"],
                "selector_mean_promoted_vote_fraction": decision["mean_promoted_vote_fraction"],
                "selector_promoted_pages": decision["promoted_pages"],
                "selector_stable_promoted_pages": decision["stable_promoted_pages"],
                "selector_unstable_promoted_pages": decision["unstable_promoted_pages"],
                "selector_promoted_page_vote_counts": decision["promoted_page_vote_counts"],
                "selector_promoted_page_vote_sources": decision["promoted_page_vote_sources"],
            },
        }
        per_qid.append(row_summary)

    k_values = parse_int_list(args.recall_k_values)
    summary: dict[str, Any] = {
        **selector_metadata,
        "qid_count": len(qids),
        "selected_candidate_count": source_counts.get("candidate", 0),
        "selected_reference_count": source_counts.get("reference", 0),
        "selection_reason_counts": dict(reason_counts),
        "mean_selector_promoted_page_count": mean_or_none(promoted_counts),
        "mean_selector_stable_promoted_page_count": mean_or_none(stable_promoted_counts),
        "mean_selector_min_promoted_vote_fraction": mean_or_none(min_vote_fractions),
        "per_qid": per_qid,
    }

    if gold_by_qid:
        doc_ranks = [row.get("reranked_first_gold_doc_rank") for row in per_qid]
        page_ranks = [row.get("reranked_first_gold_page_rank") for row in per_qid]
        summary.update(
            {
                "reranked_top4_doc_count": sum(hit_at(rank, 4) for rank in doc_ranks),
                "reranked_top20_doc_count": sum(hit_at(rank, 20) for rank in doc_ranks),
                "reranked_top4_page_count": sum(hit_at(rank, 4) for rank in page_ranks),
                "reranked_top20_page_count": sum(hit_at(rank, 20) for rank in page_ranks),
                **evaluate_recall(
                    selected=selected,
                    gold_by_qid=gold_by_qid,
                    qids=qids,
                    k_values=k_values,
                ),
            }
        )
        if baseline is not None:
            movement_counts = evaluate_movements(
                baseline=baseline,
                selected=selected,
                gold_by_qid=gold_by_qid,
                qids=qids,
                topk=int(args.topk),
            )
            summary["selected_vs_baseline_counts"] = dict(movement_counts)
            summary["top_recovered"] = build_top_examples(
                baseline=baseline,
                selected=selected,
                gold_by_qid=gold_by_qid,
                qids=qids,
                topk=int(args.topk),
                topn=int(args.topn),
            )

    output_prediction_json = Path(args.output_prediction_json)
    output_prediction_json.parent.mkdir(parents=True, exist_ok=True)
    output_prediction_json.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")

    output_summary_json = Path(args.output_summary_json)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"saved_prediction: {output_prediction_json}")
    print(f"saved_summary: {output_summary_json}")
    print(f"qid_count: {len(qids)}")
    print(f"selected_candidate_count: {summary['selected_candidate_count']}")
    print(f"selected_reference_count: {summary['selected_reference_count']}")
    print(f"selection_reason_counts: {summary['selection_reason_counts']}")
    if gold_by_qid:
        print(f"reranked_top4_doc_count: {summary['reranked_top4_doc_count']}")
        print(f"reranked_top20_doc_count: {summary['reranked_top20_doc_count']}")
        print(f"reranked_top4_page_count: {summary['reranked_top4_page_count']}")
        print(f"reranked_top20_page_count: {summary['reranked_top20_page_count']}")
        print(f"page_recall_at_k: {summary['page_recall_at_k']}")
        print(f"doc_recall_at_k: {summary['doc_recall_at_k']}")
        if "selected_vs_baseline_counts" in summary:
            for key in [
                "recovered",
                "lost",
                "improved_rank",
                "worsened_rank",
                "unchanged",
                "missing_in_both",
            ]:
                print(f"{key}: {summary['selected_vs_baseline_counts'].get(key, 0)}")


if __name__ == "__main__":
    main()
