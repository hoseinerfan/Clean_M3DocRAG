#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


NUMERIC_FEATURES = [
    "query_local_evidence_atom_count",
    "query_local_evidence_matched_atom_count",
    "query_local_evidence_selected_atom_count",
    "query_local_evidence_page_match_count",
    "mean_query_local_evidence_atom_specificity",
    "mean_query_local_evidence_doc_confidence",
    "mean_query_local_evidence_active_page_frac",
    "mean_query_local_evidence_doc_alpha",
    "constraint_competition_active_doc_count",
    "constraint_competition_active_page_count",
    "mean_constraint_competition_doc_to_page_multiplier",
    "max_constraint_competition_doc_to_page_multiplier",
    "query_local_evidence_skipped_low_confidence_doc_count",
    "query_local_evidence_skipped_broad_doc_count",
    "mean_query_local_evidence_boundary_weight",
    "mean_query_local_evidence_active_boundary_weight",
    "min_query_local_evidence_boundary_weight",
    "max_query_local_evidence_boundary_weight",
    "query_local_selector_feature_value",
    "query_local_stability_variant_count",
    "query_local_stability_support_vote_count",
    "query_local_stability_vote_fraction",
    "query_local_stability_promoted_page_count",
    "selector_variant_count",
    "selector_support_view_count",
    "selector_required_votes",
    "selector_support_topk",
    "selector_promoted_page_count",
    "selector_stable_promoted_page_count",
    "selector_unstable_promoted_page_count",
    "selector_min_promoted_vote_fraction",
    "selector_mean_promoted_vote_fraction",
    "query_anchor_constraint_bundle_page_match_count",
    "query_anchor_constraint_active_slot_count",
    "query_anchor_constraint_matched_slot_count",
    "query_anchor_constraint_mean_slot_specificity",
    "query_anchor_constraint_mean_value_specificity",
]


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
        rows_by_qid[qid] = row
    return rows_by_qid


def load_summary_per_qid(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    per_qid = payload.get("per_qid", [])
    if not isinstance(per_qid, list):
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for row in per_qid:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("qid", "")).strip()
        if qid:
            rows[qid] = row
    return rows


def page_uid(doc_id: object, page_idx: object) -> str:
    return f"{doc_id}_page{int(page_idx)}"


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


def first_rank(rows: list[list[Any]], gold_pages: set[str]) -> int | None:
    for rank, row in enumerate(rows, start=1):
        if not isinstance(row, list) or len(row) < 2:
            continue
        try:
            uid = page_uid(row[0], row[1])
        except (TypeError, ValueError):
            continue
        if uid in gold_pages:
            return rank
    return None


def hit_at(rank: int | None, topk: int) -> bool:
    return rank is not None and int(rank) <= int(topk)


def movement_for_hit(
    baseline_rank: int | None,
    candidate_rank: int | None,
    topk: int,
) -> str:
    baseline_hit = baseline_rank is not None and baseline_rank <= topk
    candidate_hit = candidate_rank is not None and candidate_rank <= topk
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


def metadata_value(row: dict[str, Any], key: str) -> Any:
    metadata = row.get("metadata", {})
    if key in metadata:
        return metadata.get(key)
    return row.get(key)


def category_values(value: Any) -> list[str]:
    if value is None:
        return ["<missing>"]
    if isinstance(value, list):
        if not value:
            return ["<empty>"]
        return [str(item) for item in value]
    return [str(value)]


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def quantile(sorted_values: list[float], frac: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = frac * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def numeric_stats(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    if not ordered:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {
        "n": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p10": quantile(ordered, 0.10),
        "p90": quantile(ordered, 0.90),
    }


def format_float(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)


def print_table(rows: list[list[Any]]) -> None:
    if not rows:
        return
    widths = [0] * len(rows[0])
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))
    for row_idx, row in enumerate(rows):
        print("  ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row)))
        if row_idx == 0:
            print("  ".join("-" * width for width in widths))


def build_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    baseline = load_prediction(Path(args.baseline))
    candidate = load_prediction(Path(args.candidate))
    summary_by_qid = load_summary_per_qid(Path(args.candidate_summary_json))
    reference = load_prediction(Path(args.reference_candidate)) if args.reference_candidate else {}
    gold_rows = read_jsonl(Path(args.gold))
    gold_by_qid = {str(row["qid"]): row for row in gold_rows}

    qids = sorted(set(baseline) & set(candidate) & set(gold_by_qid))
    rows: list[dict[str, Any]] = []
    for qid in qids:
        gold_row = gold_by_qid[qid]
        gold_pages = gold_page_uids(gold_row)
        baseline_rank = first_rank(baseline[qid].get("page_retrieval_results", []), gold_pages)
        candidate_rank = first_rank(candidate[qid].get("page_retrieval_results", []), gold_pages)
        movement = movement_for_hit(baseline_rank, candidate_rank, int(args.topk))
        graph = summary_by_qid.get(qid, {}).get("graph", {})
        row: dict[str, Any] = {
            "qid": qid,
            "question": gold_row.get("question", ""),
            "movement": movement,
            "baseline_first_gold_page_rank": baseline_rank,
            "candidate_first_gold_page_rank": candidate_rank,
            "rank_delta": (
                None
                if baseline_rank is None or candidate_rank is None
                else int(baseline_rank) - int(candidate_rank)
            ),
            "repo_slug": metadata_value(gold_row, "repo_slug"),
            "source_query_id": metadata_value(gold_row, "source_query_id"),
            "query_types": metadata_value(gold_row, "query_types"),
            "query_format": metadata_value(gold_row, "query_format"),
            "query_type_for_generation": metadata_value(
                gold_row,
                "query_type_for_generation",
            ),
            "content_type": metadata_value(gold_row, "content_type"),
        }
        if qid in reference:
            reference_rank = first_rank(reference[qid].get("page_retrieval_results", []), gold_pages)
            row["reference_first_gold_page_rank"] = reference_rank
            row["candidate_vs_reference_movement"] = movement_for_hit(
                reference_rank,
                candidate_rank,
                int(args.topk),
            )
        for feature in NUMERIC_FEATURES:
            row[feature] = graph.get(feature)
        rows.append(row)
    return rows


def group_cases(cases: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        groups[str(row["movement"])].append(row)
    return groups


def summarize_categories(cases: list[dict[str, Any]], groups: dict[str, list[dict[str, Any]]]) -> None:
    for key in ["repo_slug", "query_types", "query_format", "query_type_for_generation", "content_type"]:
        print(f"\n{key}")
        table = [["group", "top_values"]]
        for group_name in ["recovered", "lost", "worsened_rank", "unchanged"]:
            counter: Counter[str] = Counter()
            for row in groups.get(group_name, []):
                for value in category_values(row.get(key)):
                    counter[value] += 1
            top_values = ", ".join(f"{value}:{count}" for value, count in counter.most_common(6))
            table.append([group_name, top_values or "-"])
        print_table(table)


def summarize_features(groups: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, dict[str, Any]]]:
    feature_summary: dict[str, dict[str, dict[str, Any]]] = {}
    for feature in NUMERIC_FEATURES:
        feature_summary[feature] = {}
        for group_name in ["recovered", "lost", "worsened_rank", "unchanged"]:
            values = [
                number
                for row in groups.get(group_name, [])
                if (number := as_float(row.get(feature))) is not None
            ]
            feature_summary[feature][group_name] = numeric_stats(values)
    return feature_summary


def print_feature_summary(feature_summary: dict[str, dict[str, dict[str, Any]]]) -> None:
    rows = [["feature", "recovered_mean", "lost_mean", "worsened_mean", "unchanged_mean"]]
    for feature, by_group in feature_summary.items():
        rows.append(
            [
                feature,
                format_float(by_group["recovered"]["mean"]),
                format_float(by_group["lost"]["mean"]),
                format_float(by_group["worsened_rank"]["mean"]),
                format_float(by_group["unchanged"]["mean"]),
            ]
        )
    print("\nnumeric_feature_means")
    print_table(rows)


def scan_qid_gate(
    cases: list[dict[str, Any]],
    *,
    confidence_thresholds: list[float],
    max_multiplier_thresholds: list[float],
    active_page_thresholds: list[float],
    topn: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for min_conf in confidence_thresholds:
        for max_mult in max_multiplier_thresholds:
            for max_pages in active_page_thresholds:
                use_candidate = []
                for row in cases:
                    confidence = as_float(row.get("mean_query_local_evidence_doc_confidence"))
                    max_multiplier = as_float(row.get("max_constraint_competition_doc_to_page_multiplier"))
                    active_pages = as_float(row.get("constraint_competition_active_page_count"))
                    if confidence is None or max_multiplier is None or active_pages is None:
                        use_candidate.append(False)
                        continue
                    use_candidate.append(
                        confidence >= min_conf
                        and max_multiplier <= max_mult
                        and active_pages <= max_pages
                    )
                recovered = sum(
                    row["movement"] in {"recovered", "improved_rank"}
                    for row, keep in zip(cases, use_candidate)
                    if keep
                )
                harmed = sum(
                    row["movement"] in {"lost", "worsened_rank"}
                    for row, keep in zip(cases, use_candidate)
                    if keep
                )
                topk_hits = 0
                for row, keep in zip(cases, use_candidate):
                    rank_key = (
                        "candidate_first_gold_page_rank"
                        if keep
                        else "baseline_first_gold_page_rank"
                    )
                    rank = row.get(rank_key)
                    if rank is not None and int(rank) <= 4:
                        topk_hits += 1
                candidates.append(
                    {
                        "min_confidence": min_conf,
                        "max_multiplier": max_mult,
                        "max_active_pages": max_pages,
                        "applied_qids": sum(use_candidate),
                        "benefit_events": recovered,
                        "harm_events": harmed,
                        "benefit_minus_harm": recovered - harmed,
                        "estimated_top4_hits": topk_hits,
                    }
                )
    candidates.sort(
        key=lambda row: (
            -int(row["estimated_top4_hits"]),
            -int(row["benefit_minus_harm"]),
            int(row["harm_events"]),
            int(row["applied_qids"]),
        )
    )
    return candidates[:topn]


def selector_movement_counts(
    cases: list[dict[str, Any]],
    *,
    use_candidate: list[bool],
    topk: int,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row, keep_candidate in zip(cases, use_candidate):
        chosen_rank = (
            row.get("candidate_first_gold_page_rank")
            if keep_candidate
            else row.get("reference_first_gold_page_rank")
        )
        counts[
            movement_for_hit(
                row.get("baseline_first_gold_page_rank"),
                chosen_rank,
                topk,
            )
        ] += 1
    return counts


def selector_topk_hit_count(
    cases: list[dict[str, Any]],
    *,
    use_candidate: list[bool],
    topk: int,
) -> int:
    hits = 0
    for row, keep_candidate in zip(cases, use_candidate):
        chosen_rank = (
            row.get("candidate_first_gold_page_rank")
            if keep_candidate
            else row.get("reference_first_gold_page_rank")
        )
        if hit_at(chosen_rank, topk):
            hits += 1
    return hits


def rank_is_better(left: int | None, right: int | None) -> bool:
    if left is None:
        return False
    if right is None:
        return True
    return int(left) < int(right)


def scan_reference_selector(
    cases: list[dict[str, Any]],
    *,
    topk: int,
    topn: int,
) -> list[dict[str, Any]]:
    comparable_cases = [row for row in cases if "reference_first_gold_page_rank" in row]
    if not comparable_cases:
        return []

    reference_hits = sum(
        1 for row in comparable_cases if hit_at(row.get("reference_first_gold_page_rank"), topk)
    )
    candidate_hits = sum(
        1 for row in comparable_cases if hit_at(row.get("candidate_first_gold_page_rank"), topk)
    )
    baseline_hits = sum(
        1 for row in comparable_cases if hit_at(row.get("baseline_first_gold_page_rank"), topk)
    )

    rows: list[dict[str, Any]] = []
    for feature in NUMERIC_FEATURES:
        values = sorted(
            {
                number
                for row in comparable_cases
                if (number := as_float(row.get(feature))) is not None
            }
        )
        if not values:
            continue
        thresholds = sorted(
            {
                values[0],
                values[-1],
                *(
                    quantile(values, frac)
                    for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                ),
            }
        )
        for direction in ["<=", ">="]:
            for threshold in thresholds:
                if threshold is None:
                    continue
                use_candidate: list[bool] = []
                for row in comparable_cases:
                    value = as_float(row.get(feature))
                    if value is None:
                        use_candidate.append(False)
                    elif direction == "<=":
                        use_candidate.append(value <= threshold)
                    else:
                        use_candidate.append(value >= threshold)

                selected_count = sum(use_candidate)
                if selected_count <= 0:
                    continue
                selector_hits = selector_topk_hit_count(
                    comparable_cases,
                    use_candidate=use_candidate,
                    topk=topk,
                )
                movement_counts = selector_movement_counts(
                    comparable_cases,
                    use_candidate=use_candidate,
                    topk=topk,
                )
                candidate_wins = 0
                reference_wins = 0
                rank_ties = 0
                for row, keep_candidate in zip(comparable_cases, use_candidate):
                    if not keep_candidate:
                        continue
                    candidate_rank = row.get("candidate_first_gold_page_rank")
                    reference_rank = row.get("reference_first_gold_page_rank")
                    if rank_is_better(candidate_rank, reference_rank):
                        candidate_wins += 1
                    elif rank_is_better(reference_rank, candidate_rank):
                        reference_wins += 1
                    else:
                        rank_ties += 1

                rows.append(
                    {
                        "feature": feature,
                        "direction": direction,
                        "threshold": threshold,
                        "selected_candidate_qids": selected_count,
                        "baseline_topk_hits": baseline_hits,
                        "reference_topk_hits": reference_hits,
                        "candidate_topk_hits": candidate_hits,
                        "selector_topk_hits": selector_hits,
                        "selector_minus_reference": selector_hits - reference_hits,
                        "selector_recovered": movement_counts.get("recovered", 0),
                        "selector_lost": movement_counts.get("lost", 0),
                        "selector_worsened_rank": movement_counts.get("worsened_rank", 0),
                        "candidate_wins_inside_selection": candidate_wins,
                        "reference_wins_inside_selection": reference_wins,
                        "rank_ties_inside_selection": rank_ties,
                    }
                )

    rows.sort(
        key=lambda row: (
            -int(row["selector_topk_hits"]),
            -int(row["selector_minus_reference"]),
            int(row["selector_lost"]),
            int(row["selector_worsened_rank"]),
            int(row["selected_candidate_qids"]),
        )
    )
    return rows[:topn]


def parse_float_list(value: str, default: list[float]) -> list[float]:
    if not value:
        return default
    return [float(item) for item in value.replace(",", " ").split()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit query-local softmax improvements/losses and the graph diagnostics "
            "that separate them."
        )
    )
    parser.add_argument("--gold", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--candidate-summary-json", required=True)
    parser.add_argument(
        "--reference-candidate",
        default="",
        help="Optional competing prediction JSON, e.g. cb_comp_s100, for movement labels.",
    )
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--topn", type=int, default=20)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-jsonl", default="")
    parser.add_argument("--confidence-grid", default="0 0.05 0.10 0.15 0.20 0.30")
    parser.add_argument("--max-multiplier-grid", default="2 4 6 8 12 20 40")
    parser.add_argument("--active-pages-grid", default="25 50 100 200 400 800 2000")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = build_cases(args)
    groups = group_cases(cases)
    counts = Counter(row["movement"] for row in cases)

    print(f"n_qids {len(cases)}")
    print(f"topk {int(args.topk)}")
    for key in ["recovered", "lost", "improved_rank", "worsened_rank", "unchanged", "missing_in_both"]:
        print(f"{key} {counts.get(key, 0)}")

    summarize_categories(cases, groups)
    feature_summary = summarize_features(groups)
    print_feature_summary(feature_summary)

    gate_rows = scan_qid_gate(
        cases,
        confidence_thresholds=parse_float_list(args.confidence_grid, []),
        max_multiplier_thresholds=parse_float_list(args.max_multiplier_grid, []),
        active_page_thresholds=parse_float_list(args.active_pages_grid, []),
        topn=int(args.topn),
    )
    print("\nqid_level_gate_grid_estimates")
    print_table(
        [
            [
                "min_conf",
                "max_mult",
                "max_active_pages",
                "applied",
                "benefit",
                "harm",
                "benefit-harm",
                "est_top4_hits",
            ],
            *[
                [
                    format_float(row["min_confidence"]),
                    format_float(row["max_multiplier"]),
                    format_float(row["max_active_pages"]),
                    row["applied_qids"],
                    row["benefit_events"],
                    row["harm_events"],
                    row["benefit_minus_harm"],
                    row["estimated_top4_hits"],
                ]
                for row in gate_rows
            ],
        ]
    )

    selector_rows = scan_reference_selector(
        cases,
        topk=int(args.topk),
        topn=int(args.topn),
    )
    if selector_rows:
        print("\nreference_selector_grid_estimates")
        print_table(
            [
                [
                    "feature",
                    "op",
                    "threshold",
                    "selected",
                    "base_hits",
                    "ref_hits",
                    "cand_hits",
                    "selector_hits",
                    "selector-ref",
                    "sel_recovered",
                    "sel_lost",
                    "sel_worse",
                    "cand_wins",
                    "ref_wins",
                ],
                *[
                    [
                        row["feature"],
                        row["direction"],
                        format_float(row["threshold"]),
                        row["selected_candidate_qids"],
                        row["baseline_topk_hits"],
                        row["reference_topk_hits"],
                        row["candidate_topk_hits"],
                        row["selector_topk_hits"],
                        row["selector_minus_reference"],
                        row["selector_recovered"],
                        row["selector_lost"],
                        row["selector_worsened_rank"],
                        row["candidate_wins_inside_selection"],
                        row["reference_wins_inside_selection"],
                    ]
                    for row in selector_rows
                ],
            ]
        )

    payload = {
        "n_qids": len(cases),
        "topk": int(args.topk),
        "counts": dict(counts),
        "feature_summary": feature_summary,
        "gate_grid_estimates": gate_rows,
        "reference_selector_grid_estimates": selector_rows,
        "cases": cases,
    }
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.output_jsonl:
        with Path(args.output_jsonl).open("w", encoding="utf-8") as handle:
            for row in cases:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
