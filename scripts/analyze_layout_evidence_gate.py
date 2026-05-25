#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


DEFAULT_RECALL_KS = [1, 2, 4, 5, 10, 20, 50, 100]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Learn an interpretable, query-adaptive gate for deciding whether to accept a "
            "layout/OCR evidence reranker or keep the base prediction. Rules use only "
            "observable query-time features, such as top-doc preservation, evidence density, "
            "and rank displacement."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=5,
        metavar=("LABEL", "GOLD", "BASELINE", "CANDIDATE", "CASE_JSON"),
        default=[],
        help=(
            "Dataset/run tuple. Repeat to learn one rule across datasets. CASE_JSON should "
            "be the *_cases.json emitted by run_layout_evidence_graph_track.sh."
        ),
    )
    parser.add_argument("--gold", default="", help="Single-run gold JSONL.")
    parser.add_argument("--baseline", default="", help="Single-run base prediction JSON.")
    parser.add_argument("--candidate", default="", help="Single-run candidate prediction JSON.")
    parser.add_argument("--case-json", default="", help="Single-run candidate case JSON.")
    parser.add_argument("--run-label", default="run")
    parser.add_argument("--hit-k", type=int, default=4)
    parser.add_argument("--recall-k", dest="recall_ks", type=int, nargs="+", default=DEFAULT_RECALL_KS)
    parser.add_argument("--min-accept", type=int, default=5)
    parser.add_argument(
        "--max-doc-hit-loss",
        type=int,
        default=0,
        help="Discard rules whose gated doc-hit count is worse than base by more than this.",
    )
    parser.add_argument("--pair-source-rules", type=int, default=80)
    parser.add_argument("--max-rules", type=int, default=30)
    parser.add_argument("--output-md", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--output-rule-json", default="")
    parser.add_argument(
        "--rule-json",
        default="",
        help="Optional previously learned rule JSON to apply/report instead of selecting best.",
    )
    parser.add_argument(
        "--output-gated-dir",
        default="",
        help="Optional directory for per-run gated prediction and summary JSON files.",
    )
    return parser.parse_args()


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
        payload["predictions"], (dict, list)
    ):
        payload = payload["predictions"]
    rows: dict[str, dict[str, Any]] = {}
    if isinstance(payload, dict):
        iterable = payload.items()
    elif isinstance(payload, list):
        iterable = enumerate(payload)
    else:
        raise TypeError(f"Prediction JSON must be list or object: {path}")
    for raw_key, row in iterable:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("qid", "")).strip() or str(raw_key).strip()
        if qid:
            rows[qid] = row
    return rows


def load_case_json(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        cases = payload.get("cases", payload.get("rows", []))
    elif isinstance(payload, list):
        cases = payload
    else:
        cases = []
    out: dict[str, dict[str, Any]] = {}
    for row in cases:
        if isinstance(row, dict) and str(row.get("qid", "")).strip():
            out[str(row["qid"])] = row
    return out


def page_uid(doc_id: Any, page_idx: Any) -> str:
    return f"{doc_id}_page{int(page_idx)}"


def page_doc(uid: str) -> str:
    return uid.rsplit("_page", 1)[0]


def ranked_pages(pred_row: dict[str, Any] | None, limit: int = 0) -> list[str]:
    if pred_row is None:
        return []
    pages: list[str] = []
    seen: set[str] = set()
    for item in pred_row.get("page_retrieval_results", []):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            uid = page_uid(str(item[0]), int(item[1]))
        except (TypeError, ValueError):
            continue
        if uid in seen:
            continue
        seen.add(uid)
        pages.append(uid)
        if limit > 0 and len(pages) >= limit:
            break
    return pages


def ranked_docs(pred_row: dict[str, Any] | None, limit: int = 0) -> list[str]:
    docs: list[str] = []
    seen: set[str] = set()
    for uid in ranked_pages(pred_row):
        doc_id = page_doc(uid)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        docs.append(doc_id)
        if limit > 0 and len(docs) >= limit:
            break
    return docs


def prediction_scores(pred_row: dict[str, Any] | None, limit: int = 0) -> list[float]:
    if pred_row is None:
        return []
    scores: list[float] = []
    for item in pred_row.get("page_retrieval_results", []):
        if not isinstance(item, (list, tuple)):
            continue
        try:
            scores.append(float(item[2]) if len(item) >= 3 else 0.0)
        except (TypeError, ValueError):
            scores.append(0.0)
        if limit > 0 and len(scores) >= limit:
            break
    return scores


def first_rank(ranked: list[str], gold: set[str]) -> int | None:
    for idx, item in enumerate(ranked, start=1):
        if item in gold:
            return idx
    return None


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / float(len(gold))


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
            uids.add(page_uid(doc_id, int(page_idx)))
    return uids


def gold_doc_ids(row: dict[str, Any]) -> set[str]:
    metadata = row.get("metadata", {})
    docs = {
        str(value).strip()
        for value in metadata.get("gold_doc_ids", [])
        if str(value).strip()
    }
    for ctx in row.get("supporting_context", []):
        doc_id = str(ctx.get("doc_id", "")).strip()
        if doc_id:
            docs.add(doc_id)
    return docs


def metric_scores(
    pred_row: dict[str, Any] | None,
    gold_pages: set[str],
    gold_docs: set[str],
    recall_ks: list[int],
    hit_k: int,
) -> dict[str, Any]:
    pages = ranked_pages(pred_row)
    docs = ranked_docs(pred_row)
    page_rank = first_rank(pages, gold_pages)
    doc_rank = first_rank(docs, gold_docs)
    out: dict[str, Any] = {
        "page_first_rank": page_rank,
        "doc_first_rank": doc_rank,
        f"page_hit@{hit_k}": page_rank is not None and page_rank <= hit_k,
        f"doc_hit@{hit_k}": doc_rank is not None and doc_rank <= hit_k,
    }
    for k in recall_ks:
        out[f"page_recall@{k}"] = recall_at_k(pages, gold_pages, k)
        out[f"doc_recall@{k}"] = recall_at_k(docs, gold_docs, k)
    return out


def metric_value(scores: dict[str, Any], key: str) -> float:
    value = scores.get(key)
    if isinstance(value, bool):
        return float(value)
    if value is None:
        return 0.0
    return float(value)


def jaccard(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / float(len(union))


def overlap_fraction(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    if not left_set:
        return 0.0
    return len(left_set & set(right)) / float(len(left_set))


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def score_margin(scores: list[float], left_rank: int, right_rank: int) -> float:
    left_idx = left_rank - 1
    right_idx = right_rank - 1
    if left_idx >= len(scores) or right_idx >= len(scores):
        return 0.0
    return float(scores[left_idx]) - float(scores[right_idx])


def query_features(question: str) -> dict[str, Any]:
    q = question.lower()
    return {
        "query_len": len(question.split()),
        "query_has_number": bool(re.search(r"\d", question)),
        "query_has_page_cue": bool(re.search(r"\b(page|section|appendix)\b", q)),
        "query_has_visual_cue": bool(re.search(r"\b(table|figure|chart|graph|image|picture)\b", q)),
        "query_has_numeric_cue": bool(re.search(r"\b(how many|percentage|percent|ratio|total|sum)\b", q)),
    }


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def build_features(
    *,
    gold_row: dict[str, Any],
    baseline_row: dict[str, Any],
    candidate_row: dict[str, Any],
    case_row: dict[str, Any],
    hit_k: int,
) -> dict[str, Any]:
    base_pages4 = ranked_pages(baseline_row, hit_k)
    cand_pages4 = ranked_pages(candidate_row, hit_k)
    base_pages10 = ranked_pages(baseline_row, 10)
    cand_pages10 = ranked_pages(candidate_row, 10)
    base_docs4 = ranked_docs(baseline_row, hit_k)
    cand_docs4 = ranked_docs(candidate_row, hit_k)
    base_rank = {uid: rank for rank, uid in enumerate(ranked_pages(baseline_row), start=1)}
    cand_scores = prediction_scores(candidate_row, 6)
    base_scores = prediction_scores(baseline_row, 6)

    cand_base_ranks = [base_rank.get(uid, 10**6) for uid in cand_pages4]
    candidate_page_count = safe_float(case_row.get("candidate_page_count"))
    region_count = safe_float(case_row.get("region_count"))
    positive_query_region_count = safe_float(case_row.get("positive_query_region_count"))
    positive_evidence_page_count = safe_float(case_row.get("positive_evidence_page_count"))

    features: dict[str, Any] = {
        **query_features(str(gold_row.get("question", ""))),
        "candidate_page_count": candidate_page_count,
        "region_count": region_count,
        "positive_query_region_count": positive_query_region_count,
        "positive_evidence_page_count": positive_evidence_page_count,
        "positive_evidence_frac": (
            positive_evidence_page_count / candidate_page_count if candidate_page_count else 0.0
        ),
        "query_region_frac": positive_query_region_count / region_count if region_count else 0.0,
        "query_regions_per_candidate_page": (
            positive_query_region_count / candidate_page_count if candidate_page_count else 0.0
        ),
        "regions_per_candidate_page": region_count / candidate_page_count if candidate_page_count else 0.0,
        "page_overlap_top4": jaccard(base_pages4, cand_pages4),
        "page_overlap_top10": jaccard(base_pages10, cand_pages10),
        "candidate_top4_in_base_top4_frac": overlap_fraction(cand_pages4, base_pages4),
        "candidate_top4_new_page_count": len(set(cand_pages4) - set(base_pages4)),
        "doc_overlap_top4": jaccard(base_docs4, cand_docs4),
        "candidate_top4_doc_in_base_top4_frac": overlap_fraction(cand_docs4, base_docs4),
        "candidate_top4_new_doc_count": len(set(cand_docs4) - set(base_docs4)),
        "candidate_top4_doc_subset_base_top4": set(cand_docs4).issubset(set(base_docs4)),
        "candidate_top1_doc_in_base_top4": bool(cand_docs4 and cand_docs4[0] in set(base_docs4)),
        "candidate_top1_page_in_base_top4": bool(cand_pages4 and cand_pages4[0] in set(base_pages4)),
        "candidate_top4_mean_base_rank": mean([float(rank) for rank in cand_base_ranks]) if cand_base_ranks else 0.0,
        "candidate_top4_max_base_rank": max(cand_base_ranks) if cand_base_ranks else 0.0,
        "candidate_promoted_from_below4_count": sum(rank > hit_k for rank in cand_base_ranks),
        "candidate_score_margin_1_2": score_margin(cand_scores, 1, 2),
        "candidate_score_margin_4_5": score_margin(cand_scores, 4, 5),
        "base_score_margin_1_2": score_margin(base_scores, 1, 2),
        "base_score_margin_4_5": score_margin(base_scores, 4, 5),
    }
    for key in [
        "pages_with_regions",
        "explicit_region_pages",
        "fallback_region_pages",
    ]:
        if key in case_row:
            features[key] = safe_float(case_row.get(key))
    return features


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


def load_runs(args: argparse.Namespace) -> list[tuple[str, Path, Path, Path, Path]]:
    runs: list[tuple[str, Path, Path, Path, Path]] = []
    for label, gold, baseline, candidate, case_json in args.run:
        runs.append((label, Path(gold), Path(baseline), Path(candidate), Path(case_json)))
    single_fields = [args.gold, args.baseline, args.candidate, args.case_json]
    if any(single_fields):
        if not all(single_fields):
            raise ValueError("--gold, --baseline, --candidate, and --case-json must be set together.")
        runs.append(
            (
                str(args.run_label),
                Path(args.gold),
                Path(args.baseline),
                Path(args.candidate),
                Path(args.case_json),
            )
        )
    if not runs:
        raise ValueError("Provide at least one --run or the single-run input arguments.")
    return runs


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_label, gold_path, base_path, cand_path, case_path in load_runs(args):
        gold = {str(row["qid"]): row for row in read_jsonl(gold_path)}
        baseline = load_prediction(base_path)
        candidate = load_prediction(cand_path)
        cases = load_case_json(case_path)
        qids = sorted(set(gold) & set(baseline) & set(candidate))
        for qid in qids:
            gold_row = gold[qid]
            base_row = baseline[qid]
            cand_row = candidate[qid]
            case_row = cases.get(qid, {})
            gold_pages = gold_page_uids(gold_row)
            gold_docs = gold_doc_ids(gold_row)
            scores = {
                "base": metric_scores(base_row, gold_pages, gold_docs, args.recall_ks, int(args.hit_k)),
                "candidate": metric_scores(cand_row, gold_pages, gold_docs, args.recall_ks, int(args.hit_k)),
            }
            movement = movement_for_hit(
                scores["base"]["page_first_rank"],
                scores["candidate"]["page_first_rank"],
                int(args.hit_k),
            )
            rows.append(
                {
                    "run": run_label,
                    "qid": qid,
                    "question": gold_row.get("question", ""),
                    "gold": gold_row,
                    "baseline_row": base_row,
                    "candidate_row": cand_row,
                    "case": case_row,
                    "features": build_features(
                        gold_row=gold_row,
                        baseline_row=base_row,
                        candidate_row=cand_row,
                        case_row=case_row,
                        hit_k=int(args.hit_k),
                    ),
                    "scores": scores,
                    "movement": movement,
                }
            )
    return rows


def condition_label(condition: dict[str, Any]) -> str:
    value = condition["value"]
    if isinstance(value, float):
        value_text = f"{value:.4g}"
    else:
        value_text = str(value)
    return f"{condition['feature']} {condition['op']} {value_text}"


def rule_label(rule: dict[str, Any]) -> str:
    return " AND ".join(condition_label(cond) for cond in rule["conditions"]) or "accept_all"


def compare_values(left: Any, op: str, right: Any) -> bool:
    if op == "==":
        return bool(left) == bool(right) if isinstance(right, bool) else left == right
    value = safe_float(left, default=math.nan)
    threshold = safe_float(right, default=math.nan)
    if math.isnan(value) or math.isnan(threshold):
        return False
    if op == "<=":
        return value <= threshold
    if op == ">=":
        return value >= threshold
    raise ValueError(f"Unsupported op: {op}")


def rule_accepts(row: dict[str, Any], rule: dict[str, Any]) -> bool:
    return all(
        compare_values(row["features"].get(cond["feature"]), cond["op"], cond["value"])
        for cond in rule.get("conditions", [])
    )


def numeric_thresholds(values: list[float]) -> list[float]:
    finite = sorted({float(value) for value in values if math.isfinite(float(value))})
    if not finite:
        return []
    if len(finite) <= 16:
        return finite
    quantiles = [0.05, 0.10, 0.20, 0.25, 0.33, 0.50, 0.67, 0.75, 0.80, 0.90, 0.95]
    thresholds = {finite[0], finite[-1]}
    for q in quantiles:
        idx = min(len(finite) - 1, max(0, int(round(q * (len(finite) - 1)))))
        thresholds.add(finite[idx])
    return sorted(thresholds)


def generate_single_condition_rules(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    feature_values: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        for key, value in row["features"].items():
            feature_values[key].append(value)

    rules: list[dict[str, Any]] = [{"conditions": []}]
    for key, values in sorted(feature_values.items()):
        if all(isinstance(value, bool) for value in values):
            for target in [True, False]:
                rules.append({"conditions": [{"feature": key, "op": "==", "value": target}]})
            continue
        numeric_values: list[float] = []
        for value in values:
            if isinstance(value, bool):
                continue
            try:
                numeric_values.append(float(value))
            except (TypeError, ValueError):
                pass
        if not numeric_values:
            continue
        for threshold in numeric_thresholds(numeric_values):
            rules.append({"conditions": [{"feature": key, "op": "<=", "value": threshold}]})
            rules.append({"conditions": [{"feature": key, "op": ">=", "value": threshold}]})
    return rules


def selected_scores(row: dict[str, Any], accept_candidate: bool) -> dict[str, Any]:
    return row["scores"]["candidate" if accept_candidate else "base"]


def summarize_selection(
    rows: list[dict[str, Any]],
    selector: Callable[[dict[str, Any]], bool],
    recall_ks: list[int],
    hit_k: int,
) -> dict[str, Any]:
    page_hit_key = f"page_hit@{hit_k}"
    doc_hit_key = f"doc_hit@{hit_k}"
    movement_counts: Counter[str] = Counter()
    accepted = 0
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        accept = bool(selector(row))
        accepted += int(accept)
        scores = selected_scores(row, accept)
        base_rank = row["scores"]["base"]["page_first_rank"]
        chosen_rank = scores["page_first_rank"]
        movement_counts[movement_for_hit(base_rank, chosen_rank, hit_k)] += 1
        by_run[row["run"]].append({"row": row, "accept": accept, "scores": scores})

    item: dict[str, Any] = {
        "n": len(rows),
        "accept_count": accepted,
        "accept_frac": accepted / float(len(rows)) if rows else 0.0,
        "page_hit_count": int(
            sum(metric_value(selected_scores(row, selector(row)), page_hit_key) for row in rows)
        ),
        "doc_hit_count": int(
            sum(metric_value(selected_scores(row, selector(row)), doc_hit_key) for row in rows)
        ),
        "movement_counts": dict(sorted(movement_counts.items())),
        "recovered": movement_counts.get("recovered", 0),
        "lost": movement_counts.get("lost", 0),
        "net_recovered": movement_counts.get("recovered", 0) - movement_counts.get("lost", 0),
    }
    for k in recall_ks:
        item[f"page_recall@{k}"] = mean(
            [metric_value(selected_scores(row, selector(row)), f"page_recall@{k}") for row in rows]
        )
        item[f"doc_recall@{k}"] = mean(
            [metric_value(selected_scores(row, selector(row)), f"doc_recall@{k}") for row in rows]
        )
    item[page_hit_key] = item["page_hit_count"] / float(len(rows)) if rows else 0.0
    item[doc_hit_key] = item["doc_hit_count"] / float(len(rows)) if rows else 0.0
    for run_label, run_rows in sorted(by_run.items()):
        if not run_rows:
            continue
        item[f"{run_label}.accept_count"] = sum(int(payload["accept"]) for payload in run_rows)
        item[f"{run_label}.page_hit_count"] = int(
            sum(metric_value(payload["scores"], page_hit_key) for payload in run_rows)
        )
        item[f"{run_label}.doc_hit_count"] = int(
            sum(metric_value(payload["scores"], doc_hit_key) for payload in run_rows)
        )
    return item


def evaluate_rule(
    rows: list[dict[str, Any]],
    rule: dict[str, Any],
    recall_ks: list[int],
    hit_k: int,
) -> dict[str, Any]:
    summary = summarize_selection(
        rows,
        selector=lambda row: rule_accepts(row, rule),
        recall_ks=recall_ks,
        hit_k=hit_k,
    )
    summary["rule"] = rule_label(rule)
    summary["conditions"] = rule["conditions"]
    return summary


def oracle_accepts(row: dict[str, Any], hit_k: int) -> bool:
    base = row["scores"]["base"]
    cand = row["scores"]["candidate"]
    base_hit = bool(base[f"page_hit@{hit_k}"])
    cand_hit = bool(cand[f"page_hit@{hit_k}"])
    if cand_hit and not base_hit:
        return True
    if base_hit and not cand_hit:
        return False
    base_rank = base["page_first_rank"] or 10**9
    cand_rank = cand["page_first_rank"] or 10**9
    cand_doc_hit = bool(cand[f"doc_hit@{hit_k}"])
    base_doc_hit = bool(base[f"doc_hit@{hit_k}"])
    return cand_rank < base_rank and (cand_doc_hit or not base_doc_hit)


def learn_rules(
    rows: list[dict[str, Any]],
    recall_ks: list[int],
    hit_k: int,
    min_accept: int,
    max_doc_hit_loss: int,
    pair_source_rules: int,
    max_rules: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    base_summary = summarize_selection(rows, lambda _row: False, recall_ks, hit_k)
    cand_summary = summarize_selection(rows, lambda _row: True, recall_ks, hit_k)
    oracle_summary = summarize_selection(rows, lambda row: oracle_accepts(row, hit_k), recall_ks, hit_k)

    singles = generate_single_condition_rules(rows)
    single_evals = [evaluate_rule(rows, rule, recall_ks, hit_k) for rule in singles]
    filtered_singles = [
        item
        for item in single_evals
        if item["accept_count"] >= min_accept
        and item["doc_hit_count"] >= base_summary["doc_hit_count"] - max_doc_hit_loss
    ]
    filtered_singles.sort(
        key=lambda item: (
            item["page_hit_count"],
            item["net_recovered"],
            item["doc_hit_count"],
            -item["accept_count"],
            item["rule"],
        ),
        reverse=True,
    )

    candidate_rules = [item for item in filtered_singles]
    source_rules = filtered_singles[: max(0, pair_source_rules)]
    seen_rule_keys = {
        tuple((cond["feature"], cond["op"], json.dumps(cond["value"], sort_keys=True)) for cond in item["conditions"])
        for item in candidate_rules
    }
    for idx, left in enumerate(source_rules):
        for right in source_rules[idx + 1 :]:
            conditions = left["conditions"] + right["conditions"]
            key = tuple(
                sorted((cond["feature"], cond["op"], json.dumps(cond["value"], sort_keys=True)) for cond in conditions)
            )
            if key in seen_rule_keys:
                continue
            seen_rule_keys.add(key)
            rule = {"conditions": conditions}
            item = evaluate_rule(rows, rule, recall_ks, hit_k)
            if item["accept_count"] < min_accept:
                continue
            if item["doc_hit_count"] < base_summary["doc_hit_count"] - max_doc_hit_loss:
                continue
            candidate_rules.append(item)

    candidate_rules.sort(
        key=lambda item: (
            item["page_hit_count"],
            item["net_recovered"],
            item["doc_hit_count"],
            -len(item["conditions"]),
            -item["accept_count"],
            item["rule"],
        ),
        reverse=True,
    )
    return candidate_rules[:max_rules], base_summary, cand_summary, oracle_summary


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def flatten_feature_rows(rows: list[dict[str, Any]], hit_k: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        flat: dict[str, Any] = {
            "run": row["run"],
            "qid": row["qid"],
            "question": row["question"],
            "movement": row["movement"],
            "base_page_rank": row["scores"]["base"]["page_first_rank"],
            "candidate_page_rank": row["scores"]["candidate"]["page_first_rank"],
            f"base_page_hit@{hit_k}": row["scores"]["base"][f"page_hit@{hit_k}"],
            f"candidate_page_hit@{hit_k}": row["scores"]["candidate"][f"page_hit@{hit_k}"],
            f"base_doc_hit@{hit_k}": row["scores"]["base"][f"doc_hit@{hit_k}"],
            f"candidate_doc_hit@{hit_k}": row["scores"]["candidate"][f"doc_hit@{hit_k}"],
        }
        for key, value in row["features"].items():
            flat[f"feature.{key}"] = value
        out.append(flat)
    return out


def write_gated_outputs(
    *,
    rows: list[dict[str, Any]],
    rule: dict[str, Any],
    output_dir: Path,
    recall_ks: list[int],
    hit_k: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_run[row["run"]].append(row)
    for run_label, run_rows in rows_by_run.items():
        predictions: dict[str, dict[str, Any]] = {}
        accepted = 0
        for row in run_rows:
            accept = rule_accepts(row, rule)
            accepted += int(accept)
            pred_row = dict(row["candidate_row"] if accept else row["baseline_row"])
            pred_row["qid"] = row["qid"]
            predictions[row["qid"]] = pred_row
        summary = summarize_selection(
            run_rows,
            selector=lambda row: rule_accepts(row, rule),
            recall_ks=recall_ks,
            hit_k=hit_k,
        )
        summary["rule"] = rule_label(rule)
        summary["conditions"] = rule["conditions"]
        summary["accepted_candidate_count"] = accepted
        pred_path = output_dir / f"{run_label}_gated.prediction.json"
        summary_path = output_dir / f"{run_label}_gated.summary.json"
        pred_path.write_text(json.dumps({"predictions": predictions}, ensure_ascii=False), encoding="utf-8")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"saved_gated_prediction: {pred_path}")
        print(f"saved_gated_summary: {summary_path}")


def main() -> None:
    args = parse_args()
    if int(args.hit_k) not in args.recall_ks:
        args.recall_ks = sorted(set(args.recall_ks + [int(args.hit_k)]))
    rows = build_rows(args)
    if not rows:
        raise ValueError("No rows loaded for gate analysis.")

    best_rules, base_summary, cand_summary, oracle_summary = learn_rules(
        rows=rows,
        recall_ks=args.recall_ks,
        hit_k=int(args.hit_k),
        min_accept=int(args.min_accept),
        max_doc_hit_loss=int(args.max_doc_hit_loss),
        pair_source_rules=int(args.pair_source_rules),
        max_rules=int(args.max_rules),
    )
    selected_rule = json.loads(Path(args.rule_json).read_text(encoding="utf-8")) if args.rule_json else None
    if selected_rule is None and best_rules:
        selected_rule = {"conditions": best_rules[0]["conditions"]}
    selected_summary = (
        evaluate_rule(rows, selected_rule, args.recall_ks, int(args.hit_k)) if selected_rule else {}
    )

    overall_rows = [
        {"label": "base", **base_summary},
        {"label": "candidate", **cand_summary},
        {"label": "oracle", **oracle_summary},
    ]
    if selected_summary:
        overall_rows.append({"label": "selected_gate", **selected_summary})

    rule_rows = [
        {
            "rank": idx,
            "rule": item["rule"],
            "accept_count": item["accept_count"],
            "page_hit_count": item["page_hit_count"],
            "doc_hit_count": item["doc_hit_count"],
            "recovered": item["recovered"],
            "lost": item["lost"],
            "net_recovered": item["net_recovered"],
            f"page_recall@{int(args.hit_k)}": item.get(f"page_recall@{int(args.hit_k)}", 0.0),
            f"doc_recall@{int(args.hit_k)}": item.get(f"doc_recall@{int(args.hit_k)}", 0.0),
            **{
                key: value
                for key, value in item.items()
                if key.endswith(".page_hit_count")
                or key.endswith(".doc_hit_count")
                or key.endswith(".accept_count")
            },
        }
        for idx, item in enumerate(best_rules, start=1)
    ]

    overall_columns = [
        "label",
        "n",
        "accept_count",
        "page_hit_count",
        "doc_hit_count",
        "recovered",
        "lost",
        "net_recovered",
        f"page_recall@{int(args.hit_k)}",
        f"doc_recall@{int(args.hit_k)}",
    ]
    rule_columns = [
        "rank",
        "rule",
        "accept_count",
        "page_hit_count",
        "doc_hit_count",
        "recovered",
        "lost",
        "net_recovered",
        f"page_recall@{int(args.hit_k)}",
        f"doc_recall@{int(args.hit_k)}",
    ]
    run_metric_columns = sorted(
        {
            key
            for row in rule_rows
            for key in row
            if key.endswith(".page_hit_count")
            or key.endswith(".doc_hit_count")
            or key.endswith(".accept_count")
        }
    )
    rule_columns.extend(run_metric_columns)

    markdown = "\n".join(
        [
            "# Layout Evidence Gate Analysis",
            "",
            "## Overall",
            "",
            markdown_table(overall_rows, overall_columns),
            "## Candidate Rules",
            "",
            markdown_table(rule_rows, rule_columns),
        ]
    )

    payload = {
        "qid_count": len(rows),
        "hit_k": int(args.hit_k),
        "recall_ks": args.recall_ks,
        "base": base_summary,
        "candidate": cand_summary,
        "oracle": oracle_summary,
        "selected_rule": selected_rule,
        "selected_summary": selected_summary,
        "rules": best_rules,
    }

    if args.output_md:
        path = Path(args.output_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
        print(f"saved_md: {path}")
    else:
        print(markdown)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"saved_json: {path}")
    if args.output_csv:
        write_csv(Path(args.output_csv), flatten_feature_rows(rows, int(args.hit_k)))
        print(f"saved_csv: {args.output_csv}")
    if args.output_rule_json and selected_rule is not None:
        path = Path(args.output_rule_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(selected_rule, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"saved_rule_json: {path}")
    if args.output_gated_dir and selected_rule is not None:
        write_gated_outputs(
            rows=rows,
            rule=selected_rule,
            output_dir=Path(args.output_gated_dir),
            recall_ks=args.recall_ks,
            hit_k=int(args.hit_k),
        )


if __name__ == "__main__":
    main()
