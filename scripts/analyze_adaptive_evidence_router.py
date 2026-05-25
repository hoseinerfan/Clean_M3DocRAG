#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from analyze_layout_evidence_gate import (
    DEFAULT_RECALL_KS,
    build_features,
    compare_values,
    generate_single_condition_rules,
    gold_doc_ids,
    gold_page_uids,
    load_case_json,
    load_prediction,
    mean,
    metric_scores,
    metric_value,
    movement_for_hit,
    read_jsonl,
    rule_label,
)


BASE_LABEL = "base"
SELF_CALIBRATED_METHODS = [
    "robust_z",
    "robust_z_evidence_gain",
    "robust_z_qpp_veto",
    "percentile",
    "consensus",
    "pareto",
    "qpp",
]
SELF_CALIBRATED_PROFILES = SELF_CALIBRATED_METHODS + ["conservative", "balanced"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Learn an interpretable adaptive router over multiple evidence rerankers. "
            "Each rule is candidate-specific, for example: choose query_anchor only when "
            "its promoted pages stay near the base ranking; otherwise keep the base graph "
            "page-preserving prediction."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=3,
        metavar=("LABEL", "GOLD", "BASELINE"),
        default=[],
        help="Dataset/run tuple. Repeat for cross-dataset rule learning.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=4,
        metavar=("RUN_LABEL", "CANDIDATE_LABEL", "PREDICTION", "CASE_JSON"),
        default=[],
        help=(
            "Candidate prediction for a run. CASE_JSON may be '-' when the candidate has "
            "no per-query evidence case file."
        ),
    )
    parser.add_argument("--hit-k", type=int, default=4)
    parser.add_argument("--recall-k", dest="recall_ks", type=int, nargs="+", default=DEFAULT_RECALL_KS)
    parser.add_argument("--min-accept", type=int, default=5)
    parser.add_argument(
        "--max-doc-hit-loss",
        type=int,
        default=0,
        help="Discard rules/routers whose doc-hit count is worse than base by more than this.",
    )
    parser.add_argument(
        "--max-page-hit-loss",
        type=int,
        default=-1,
        help=(
            "Optional cap on page-hit losses versus base. Use 0 for a no-page-loss router; "
            "negative disables this constraint."
        ),
    )
    parser.add_argument("--pair-source-rules", type=int, default=80)
    parser.add_argument(
        "--max-rules",
        type=int,
        default=40,
        help="Maximum candidate rules retained per candidate label.",
    )
    parser.add_argument(
        "--router-source-rules",
        type=int,
        default=120,
        help="Top candidate-specific rules considered by the greedy multi-candidate router.",
    )
    parser.add_argument(
        "--max-router-rules",
        type=int,
        default=3,
        help="Maximum number of ordered rules in the selected router.",
    )
    parser.add_argument("--output-md", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--output-router-json", default="")
    parser.add_argument(
        "--router-json",
        default="",
        help="Optional previously learned router JSON to apply/report instead of selecting best.",
    )
    parser.add_argument(
        "--router-mode",
        choices=["learned", "self_calibrated"],
        default="learned",
        help=(
            "learned searches threshold rules on the provided runs. self_calibrated uses a "
            "fixed training-free per-query rule based on document preservation, local base "
            "uncertainty, and query-local evidence coverage."
        ),
    )
    parser.add_argument(
        "--self-calibrated-profile",
        choices=SELF_CALIBRATED_PROFILES + ["all"],
        default="conservative",
        help="Training-free router profile used when --router-mode self_calibrated.",
    )
    parser.add_argument(
        "--output-routed-dir",
        default="",
        help="Optional directory for per-run routed prediction and summary JSON files.",
    )
    return parser.parse_args()


def clean_label(raw: str) -> str:
    label = str(raw).strip()
    if not label:
        raise ValueError("Empty label is not allowed.")
    if label == BASE_LABEL:
        raise ValueError("'base' is reserved and cannot be used as a candidate label.")
    return label


def load_cases_optional(raw_path: str) -> dict[str, dict[str, Any]]:
    path_text = str(raw_path).strip()
    if not path_text or path_text == "-":
        return {}
    return load_case_json(Path(path_text))


def require_existing_file(path: Path, role: str) -> None:
    if path.is_file():
        return
    hint = ""
    if str(path).startswith("/path/to/"):
        hint = (
            " This is still a placeholder path. Replace it with a real prediction file, "
            "or remove that --candidate line. Use '-' only for CASE_JSON, not for the "
            "candidate prediction JSON."
        )
    raise FileNotFoundError(f"Missing {role}: {path}.{hint}")


def adaptive_query_features(question: str) -> dict[str, Any]:
    q = question.lower()
    return {
        "query_has_quantity_cue": bool(
            re.search(
                r"\d|"
                r"\b(how many|how much|percentage|percent|ratio|total|sum|difference)\b|"
                r"\b(combien|pourcentage|proportion|total|somme|difference)\b|"
                r"\b(cu[aá]nt[oa]s?|porcentaje|proporci[oó]n|total|suma|diferencia)\b|"
                r"\b(quanto[sa]?|percentual|porcentagem|propor[cç][aã]o|total|soma|diferen[cç]a)\b|"
                r"\b(percentuale|percento|proporzione|totale|somma|differenza)\b|"
                r"\b(wie hoch|wie viel|wie viele|prozentsatz|prozent|verh[aä]ltnis|summe|gesamt)\b",
                q,
            )
        ),
        "query_has_visual_or_table_cue": bool(
            re.search(
                r"\b(table|figure|chart|graph|image|picture|plot)\b|"
                r"\b(tableau|figure|graphique|image)\b|"
                r"\b(tabla|figura|gr[aá]fico|imagen)\b|"
                r"\b(tabela|figura|gr[aá]fico|imagem)\b|"
                r"\b(tabella|figura|grafico|immagine)\b|"
                r"\b(tabelle|abbildung|bild|diagramm|grafik)\b",
                q,
            )
        ),
        "query_has_page_locator_cue": bool(
            re.search(
                r"\b(page|section|appendix|slide)\b|"
                r"\b(page|section|annexe|diapositive)\b|"
                r"\b(p[aá]gina|secci[oó]n|ap[eé]ndice|diapositiva)\b|"
                r"\b(p[aá]gina|se[cç][aã]o|ap[eê]ndice|slide)\b|"
                r"\b(pagina|sezione|appendice|diapositiva)\b|"
                r"\b(seite|abschnitt|anhang|folie)\b",
                q,
            )
        ),
    }


def load_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, tuple[Path, Path]], dict[str, list[tuple[str, Path, str]]]]:
    runs: dict[str, tuple[Path, Path]] = {}
    for raw_label, gold, baseline in args.run:
        label = str(raw_label).strip()
        if not label:
            raise ValueError("Run label cannot be empty.")
        if label in runs:
            raise ValueError(f"Duplicate run label: {label}")
        gold_path = Path(gold)
        baseline_path = Path(baseline)
        require_existing_file(gold_path, f"gold JSONL for run '{label}'")
        require_existing_file(baseline_path, f"baseline prediction for run '{label}'")
        runs[label] = (gold_path, baseline_path)
    if not runs:
        raise ValueError("Provide at least one --run.")

    candidates: dict[str, list[tuple[str, Path, str]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for run_label, raw_candidate_label, prediction, case_json in args.candidate:
        if run_label not in runs:
            raise ValueError(f"Candidate references unknown run label: {run_label}")
        candidate_label = clean_label(raw_candidate_label)
        key = (run_label, candidate_label)
        if key in seen:
            raise ValueError(f"Duplicate candidate for run {run_label}: {candidate_label}")
        seen.add(key)
        prediction_path = Path(prediction)
        require_existing_file(
            prediction_path,
            f"candidate prediction for run '{run_label}' candidate '{candidate_label}'",
        )
        if str(case_json).strip() not in {"", "-"}:
            require_existing_file(
                Path(case_json),
                f"case JSON for run '{run_label}' candidate '{candidate_label}'",
            )
        candidates[run_label].append((candidate_label, prediction_path, case_json))
    missing = sorted(label for label in runs if not candidates.get(label))
    if missing:
        raise ValueError(f"Every run needs at least one --candidate. Missing: {', '.join(missing)}")
    return runs, candidates


def build_bundles(args: argparse.Namespace) -> list[dict[str, Any]]:
    runs, candidates_by_run = load_inputs(args)
    bundles: list[dict[str, Any]] = []
    for run_label, (gold_path, baseline_path) in runs.items():
        gold = {str(row["qid"]): row for row in read_jsonl(gold_path)}
        baseline = load_prediction(baseline_path)
        candidate_payloads: dict[str, tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]] = {}
        for candidate_label, prediction_path, case_path in candidates_by_run[run_label]:
            candidate_payloads[candidate_label] = (
                load_prediction(prediction_path),
                load_cases_optional(case_path),
            )

        qids = sorted(set(gold) & set(baseline))
        for qid in qids:
            gold_row = gold[qid]
            base_row = baseline[qid]
            gold_pages = gold_page_uids(gold_row)
            gold_docs = gold_doc_ids(gold_row)
            base_scores = metric_scores(base_row, gold_pages, gold_docs, args.recall_ks, int(args.hit_k))
            pairs: dict[str, dict[str, Any]] = {}
            for candidate_label, (candidate, cases) in candidate_payloads.items():
                candidate_row = candidate.get(qid)
                if candidate_row is None:
                    continue
                case_row = cases.get(qid, {})
                candidate_scores = metric_scores(
                    candidate_row,
                    gold_pages,
                    gold_docs,
                    args.recall_ks,
                    int(args.hit_k),
                )
                movement = movement_for_hit(
                    base_scores["page_first_rank"],
                    candidate_scores["page_first_rank"],
                    int(args.hit_k),
                )
                features = build_features(
                    gold_row=gold_row,
                    baseline_row=base_row,
                    candidate_row=candidate_row,
                    case_row=case_row,
                    hit_k=int(args.hit_k),
                )
                features.update(adaptive_query_features(str(gold_row.get("question", ""))))
                pairs[candidate_label] = {
                    "run": run_label,
                    "qid": qid,
                    "candidate_label": candidate_label,
                    "question": gold_row.get("question", ""),
                    "gold": gold_row,
                    "baseline_row": base_row,
                    "candidate_row": candidate_row,
                    "case": case_row,
                    "features": features,
                    "scores": {
                        BASE_LABEL: base_scores,
                        "candidate": candidate_scores,
                    },
                    "movement": movement,
                }
            bundles.append(
                {
                    "run": run_label,
                    "qid": qid,
                    "question": gold_row.get("question", ""),
                    "gold": gold_row,
                    "baseline_row": base_row,
                    "base_scores": base_scores,
                    "pairs": pairs,
                }
            )
    return bundles


def candidate_pair_rows(bundles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        rows.extend(bundle["pairs"].values())
    return rows


def selected_scores(bundle: dict[str, Any], selected_label: str) -> dict[str, Any]:
    if selected_label == BASE_LABEL:
        return bundle["base_scores"]
    pair = bundle["pairs"].get(selected_label)
    if pair is None:
        return bundle["base_scores"]
    return pair["scores"]["candidate"]


def summary_for_selector(
    bundles: list[dict[str, Any]],
    selector: Callable[[dict[str, Any]], str],
    recall_ks: list[int],
    hit_k: int,
) -> dict[str, Any]:
    page_hit_key = f"page_hit@{hit_k}"
    doc_hit_key = f"doc_hit@{hit_k}"
    movement_counts: Counter[str] = Counter()
    selection_counts: Counter[str] = Counter()
    available_counts: Counter[str] = Counter()
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for bundle in bundles:
        for candidate_label in bundle["pairs"]:
            available_counts[candidate_label] += 1
        selected_label = selector(bundle)
        if selected_label != BASE_LABEL and selected_label not in bundle["pairs"]:
            selected_label = BASE_LABEL
        scores = selected_scores(bundle, selected_label)
        selection_counts[selected_label] += 1
        movement_counts[
            movement_for_hit(bundle["base_scores"]["page_first_rank"], scores["page_first_rank"], hit_k)
        ] += 1
        by_run[bundle["run"]].append(
            {
                "bundle": bundle,
                "selected_label": selected_label,
                "scores": scores,
            }
        )

    item: dict[str, Any] = {
        "n": len(bundles),
        "accept_count": len(bundles) - selection_counts.get(BASE_LABEL, 0),
        "accept_frac": (
            (len(bundles) - selection_counts.get(BASE_LABEL, 0)) / float(len(bundles))
            if bundles
            else 0.0
        ),
        "candidate_available_counts": dict(sorted(available_counts.items())),
        "selection_counts": dict(sorted(selection_counts.items())),
        "page_hit_count": int(
            sum(metric_value(selected_scores(bundle, selector(bundle)), page_hit_key) for bundle in bundles)
        ),
        "doc_hit_count": int(
            sum(metric_value(selected_scores(bundle, selector(bundle)), doc_hit_key) for bundle in bundles)
        ),
        "movement_counts": dict(sorted(movement_counts.items())),
        "recovered": movement_counts.get("recovered", 0),
        "lost": movement_counts.get("lost", 0),
        "net_recovered": movement_counts.get("recovered", 0) - movement_counts.get("lost", 0),
    }
    for k in recall_ks:
        item[f"page_recall@{k}"] = mean(
            [metric_value(selected_scores(bundle, selector(bundle)), f"page_recall@{k}") for bundle in bundles]
        )
        item[f"doc_recall@{k}"] = mean(
            [metric_value(selected_scores(bundle, selector(bundle)), f"doc_recall@{k}") for bundle in bundles]
        )
    item[page_hit_key] = item["page_hit_count"] / float(len(bundles)) if bundles else 0.0
    item[doc_hit_key] = item["doc_hit_count"] / float(len(bundles)) if bundles else 0.0

    for run_label, run_rows in sorted(by_run.items()):
        if not run_rows:
            continue
        item[f"{run_label}.accept_count"] = sum(
            int(payload["selected_label"] != BASE_LABEL) for payload in run_rows
        )
        item[f"{run_label}.page_hit_count"] = int(
            sum(metric_value(payload["scores"], page_hit_key) for payload in run_rows)
        )
        item[f"{run_label}.doc_hit_count"] = int(
            sum(metric_value(payload["scores"], doc_hit_key) for payload in run_rows)
        )
        run_selection_counts = Counter(payload["selected_label"] for payload in run_rows)
        for label, count in sorted(run_selection_counts.items()):
            item[f"{run_label}.select.{label}"] = count
    return item


def option_rank_key(scores: dict[str, Any], hit_k: int, prefer_base: bool) -> tuple[Any, ...]:
    page_rank = scores["page_first_rank"] if scores["page_first_rank"] is not None else 10**9
    doc_rank = scores["doc_first_rank"] if scores["doc_first_rank"] is not None else 10**9
    return (
        bool(scores[f"page_hit@{hit_k}"]),
        metric_value(scores, f"page_recall@{hit_k}"),
        -int(page_rank),
        bool(scores[f"doc_hit@{hit_k}"]),
        -int(doc_rank),
        prefer_base,
    )


def oracle_select(bundle: dict[str, Any], hit_k: int) -> str:
    options: list[tuple[str, dict[str, Any]]] = [(BASE_LABEL, bundle["base_scores"])]
    for candidate_label, pair in bundle["pairs"].items():
        options.append((candidate_label, pair["scores"]["candidate"]))

    base_doc_hit = bool(bundle["base_scores"][f"doc_hit@{hit_k}"])
    doc_preserving = [
        (label, scores)
        for label, scores in options
        if label == BASE_LABEL or bool(scores[f"doc_hit@{hit_k}"]) or not base_doc_hit
    ]
    search_space = doc_preserving or options
    return max(
        search_space,
        key=lambda item: option_rank_key(item[1], hit_k, prefer_base=item[0] == BASE_LABEL),
    )[0]


def compare_rule_values(left: Any, op: str, right: Any) -> bool:
    return compare_values(left, op, right)


def pair_rule_accepts(pair: dict[str, Any], rule: dict[str, Any]) -> bool:
    if rule.get("mode") == "self_calibrated":
        return self_calibrated_pair_accepts(pair, rule)
    if pair["candidate_label"] != rule["candidate_label"]:
        return False
    return all(
        compare_rule_values(pair["features"].get(cond["feature"]), cond["op"], cond["value"])
        for cond in rule.get("conditions", [])
    )


def self_calibrated_pair_accepts(pair: dict[str, Any], rule: dict[str, Any]) -> bool:
    if pair["candidate_label"] != rule["candidate_label"]:
        return False
    f = pair["features"]
    hit_k = int(rule.get("hit_k", 4))
    profile = str(rule.get("profile", "conservative"))
    if profile == "all":
        return False

    candidate_page_count = float(f.get("candidate_page_count", 0.0))
    positive_evidence_page_count = float(f.get("positive_evidence_page_count", 0.0))
    promoted_count = float(f.get("candidate_promoted_from_below4_count", 0.0))
    base_boundary_margin = float(f.get("base_score_margin_4_5", 0.0))
    base_head_margin = float(f.get("base_score_margin_1_2", 0.0))
    candidate_head_margin = float(f.get("candidate_score_margin_1_2", 0.0))
    candidate_boundary_margin = float(f.get("candidate_score_margin_4_5", 0.0))
    doc_overlap = float(f.get("candidate_top4_doc_in_base_top4_frac", 0.0))
    top4_positive_evidence_count = float(f.get("candidate_top4_positive_evidence_count", 0.0))
    top4_evidence_percentile = float(f.get("candidate_top4_max_evidence_percentile", 0.0))
    top4_evidence_robust_z = float(f.get("candidate_top4_max_evidence_robust_z", 0.0))
    max_evidence_gain = float(f.get("candidate_top4_max_evidence_gain_vs_base", 0.0))
    mean_evidence_gain = float(f.get("candidate_top4_mean_evidence_gain_vs_base", 0.0))
    positive_evidence_count_gain = float(
        f.get("candidate_top4_positive_evidence_count_gain_vs_base", 0.0)
    )

    top_doc_safe = bool(f.get("candidate_top1_doc_in_base_top4"))
    doc_subset = bool(f.get("candidate_top4_doc_subset_base_top4"))
    doc_safe = top_doc_safe and doc_subset
    promotes_pages = promoted_count > 0
    evidence_nontrivial = positive_evidence_page_count >= max(1.0, min(float(hit_k), promoted_count))
    evidence_selective = 0 < positive_evidence_page_count < candidate_page_count
    base_locally_uncertain = base_boundary_margin <= base_head_margin
    top4_has_evidence = top4_positive_evidence_count > 0
    robust_z_accept = bool(
        doc_safe
        and promotes_pages
        and evidence_selective
        and top4_has_evidence
        and top4_evidence_robust_z >= 1.0
    )

    if profile == "robust_z":
        return robust_z_accept

    if profile == "robust_z_evidence_gain":
        return bool(
            robust_z_accept
            and max_evidence_gain > 0
            and (mean_evidence_gain >= 0 or positive_evidence_count_gain >= 0)
        )

    if profile == "robust_z_qpp_veto":
        relative_boundary_ok = (
            candidate_boundary_margin >= 0.5 * base_boundary_margin
            if base_boundary_margin > 0
            else candidate_boundary_margin >= 0
        )
        candidate_self_commitment_ok = (
            candidate_boundary_margin >= 0.25 * candidate_head_margin
            if candidate_head_margin > 0
            else candidate_boundary_margin >= 0
        )
        return bool(
            robust_z_accept
            and base_locally_uncertain
            and relative_boundary_ok
            and candidate_self_commitment_ok
        )

    if profile == "percentile":
        return bool(
            doc_safe
            and promotes_pages
            and evidence_selective
            and top4_has_evidence
            and top4_evidence_percentile >= 0.75
        )

    if profile == "consensus":
        return bool(
            doc_safe
            and promotes_pages
            and evidence_nontrivial
            and evidence_selective
        )

    if profile == "pareto":
        return bool(
            doc_safe
            and promotes_pages
            and evidence_selective
            and top4_has_evidence
            and float(f.get("candidate_top4_in_base_top4_frac", 0.0)) > 0
            and candidate_boundary_margin >= 0
        )

    if profile == "qpp":
        return bool(
            top_doc_safe
            and doc_overlap > 0
            and promotes_pages
            and evidence_nontrivial
            and evidence_selective
            and base_locally_uncertain
            and candidate_boundary_margin >= base_boundary_margin
        )

    if profile == "conservative":
        return bool(
            doc_safe
            and promotes_pages
            and evidence_nontrivial
            and evidence_selective
            and base_locally_uncertain
        )

    # Balanced stays training-free but relaxes the same-doc requirement from all candidate
    # top-docs to the candidate top document, while still requiring selective evidence.
    evidence_coverage = positive_evidence_page_count >= float(hit_k)
    return bool(
        top_doc_safe
        and doc_overlap > 0
        and promotes_pages
        and evidence_coverage
        and evidence_selective
        and base_locally_uncertain
    )


def feature_conditions_accept(pair: dict[str, Any], conditions: list[dict[str, Any]]) -> bool:
    return all(
        compare_rule_values(pair["features"].get(cond["feature"]), cond["op"], cond["value"])
        for cond in conditions
    )


def conditions_split_rows(rows: list[dict[str, Any]], conditions: list[dict[str, Any]]) -> bool:
    if not conditions:
        return True
    accept_count = sum(int(feature_conditions_accept(row, conditions)) for row in rows)
    return 0 < accept_count < len(rows)


def router_select(bundle: dict[str, Any], rules: list[dict[str, Any]]) -> str:
    for rule in rules:
        candidate_label = rule["candidate_label"]
        pair = bundle["pairs"].get(candidate_label)
        if pair is not None and pair_rule_accepts(pair, rule):
            return candidate_label
    return BASE_LABEL


def evaluate_router(
    bundles: list[dict[str, Any]],
    rules: list[dict[str, Any]],
    recall_ks: list[int],
    hit_k: int,
) -> dict[str, Any]:
    summary = summary_for_selector(
        bundles,
        selector=lambda bundle: router_select(bundle, rules),
        recall_ks=recall_ks,
        hit_k=hit_k,
    )
    summary["rules"] = rules
    summary["router"] = router_label(rules)
    return summary


def router_label(rules: list[dict[str, Any]]) -> str:
    if not rules:
        return BASE_LABEL
    parts = []
    for rule in rules:
        parts.append(f"{rule['candidate_label']} IF {display_rule_label(rule)}")
    return " ELSE ".join(parts) + f" ELSE {BASE_LABEL}"


def display_rule_label(rule: dict[str, Any]) -> str:
    if rule.get("mode") == "self_calibrated":
        profile = str(rule.get("profile", "conservative"))
        if profile == "robust_z":
            return (
                "robust_z("
                "same_base_docs AND promoted_pages AND selective_query_evidence "
                "AND top4_evidence_robust_z >= 1)"
            )
        if profile == "robust_z_evidence_gain":
            return (
                "robust_z_evidence_gain("
                "robust_z AND candidate_top4_max_evidence > base_top4_max_evidence "
                "AND candidate_mean_or_count_evidence >= base)"
            )
        if profile == "robust_z_qpp_veto":
            return (
                "robust_z_qpp_veto("
                "robust_z AND base_boundary_margin <= base_head_margin "
                "AND candidate_boundary >= 0.5 * base_boundary "
                "AND candidate_boundary >= 0.25 * candidate_head_margin)"
            )
        if profile == "percentile":
            return (
                "percentile("
                "same_base_docs AND promoted_pages AND selective_query_evidence "
                "AND top4_evidence_percentile >= 0.75)"
            )
        if profile == "consensus":
            return (
                "consensus("
                "same_base_docs AND promoted_pages AND enough_query_evidence)"
            )
        if profile == "pareto":
            return (
                "pareto("
                "same_base_docs AND promoted_pages AND keeps_some_base_top4_page "
                "AND top4_has_query_evidence)"
            )
        if profile == "qpp":
            return (
                "qpp("
                "base_top_doc_preserved AND base_boundary_margin <= base_head_margin "
                "AND candidate_margin >= base_boundary_margin)"
            )
        if profile == "conservative":
            return (
                "self_calibrated_conservative("
                "same_base_docs AND promoted_pages AND selective_query_evidence "
                "AND base_boundary_margin <= base_head_margin)"
            )
        return (
            "self_calibrated_balanced("
            "base_top_doc_preserved AND promoted_pages AND selective_query_evidence "
            "AND base_boundary_margin <= base_head_margin)"
        )
    return rule_label(rule)


def evaluate_candidate_rule(
    bundles: list[dict[str, Any]],
    rule: dict[str, Any],
    recall_ks: list[int],
    hit_k: int,
) -> dict[str, Any]:
    summary = evaluate_router(bundles, [rule], recall_ks, hit_k)
    summary["candidate_label"] = rule["candidate_label"]
    summary["rule"] = display_rule_label(rule)
    summary["conditions"] = rule.get("conditions", [])
    return summary


def rule_key(rule: dict[str, Any]) -> tuple[Any, ...]:
    return (
        rule["candidate_label"],
        tuple(
            sorted(
                (
                    cond["feature"],
                    cond["op"],
                    json.dumps(cond["value"], sort_keys=True),
                )
                for cond in rule.get("conditions", [])
            )
        ),
    )


def learn_candidate_rules(
    bundles: list[dict[str, Any]],
    recall_ks: list[int],
    hit_k: int,
    min_accept: int,
    max_doc_hit_loss: int,
    max_page_hit_loss: int,
    pair_source_rules: int,
    max_rules: int,
) -> list[dict[str, Any]]:
    base_summary = summary_for_selector(bundles, lambda _bundle: BASE_LABEL, recall_ks, hit_k)
    pair_rows_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in candidate_pair_rows(bundles):
        pair_rows_by_label[pair["candidate_label"]].append(pair)

    candidates: list[dict[str, Any]] = []
    for candidate_label, rows in sorted(pair_rows_by_label.items()):
        singles = generate_single_condition_rules(rows)
        single_evals: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        for single in singles:
            rule = {"candidate_label": candidate_label, "conditions": single["conditions"]}
            key = rule_key(rule)
            if key in seen:
                continue
            if not conditions_split_rows(rows, rule["conditions"]):
                continue
            seen.add(key)
            item = evaluate_candidate_rule(bundles, rule, recall_ks, hit_k)
            single_evals.append(item)

        filtered_singles = [
            item
            for item in single_evals
            if item["accept_count"] >= min_accept
            and item["doc_hit_count"] >= base_summary["doc_hit_count"] - max_doc_hit_loss
            and (max_page_hit_loss < 0 or item["lost"] <= max_page_hit_loss)
        ]
        label_candidates = list(filtered_singles)
        filtered_singles.sort(key=rank_summary_key, reverse=True)

        source_rules = filtered_singles[: max(0, pair_source_rules)]
        for idx, left in enumerate(source_rules):
            for right in source_rules[idx + 1 :]:
                conditions = left["conditions"] + right["conditions"]
                rule = {"candidate_label": candidate_label, "conditions": conditions}
                key = rule_key(rule)
                if key in seen:
                    continue
                if not conditions_split_rows(rows, conditions):
                    continue
                seen.add(key)
                item = evaluate_candidate_rule(bundles, rule, recall_ks, hit_k)
                if item["accept_count"] < min_accept:
                    continue
                if item["doc_hit_count"] < base_summary["doc_hit_count"] - max_doc_hit_loss:
                    continue
                if max_page_hit_loss >= 0 and item["lost"] > max_page_hit_loss:
                    continue
                label_candidates.append(item)

        label_candidates.sort(key=rank_summary_key, reverse=True)
        candidates.extend(label_candidates[:max_rules])

    candidates.sort(key=rank_summary_key, reverse=True)
    return candidates


def rank_summary_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        item["page_hit_count"],
        item["net_recovered"],
        item["doc_hit_count"],
        -len(item.get("conditions", [])),
        -item["accept_count"],
        item.get("candidate_label", ""),
        item.get("rule", ""),
    )


def greedy_router(
    bundles: list[dict[str, Any]],
    candidate_rules: list[dict[str, Any]],
    recall_ks: list[int],
    hit_k: int,
    max_doc_hit_loss: int,
    max_page_hit_loss: int,
    max_router_rules: int,
    router_source_rules: int,
) -> dict[str, Any]:
    base_summary = summary_for_selector(bundles, lambda _bundle: BASE_LABEL, recall_ks, hit_k)
    selected_rules: list[dict[str, Any]] = []
    selected_summary = evaluate_router(bundles, selected_rules, recall_ks, hit_k)
    used: set[tuple[Any, ...]] = set()
    source = candidate_rules[: max(0, router_source_rules)]

    for _step in range(max(0, max_router_rules)):
        best_rule: dict[str, Any] | None = None
        best_summary: dict[str, Any] | None = None
        for item in source:
            rule = {"candidate_label": item["candidate_label"], "conditions": item["conditions"]}
            key = rule_key(rule)
            if key in used:
                continue
            trial_rules = selected_rules + [rule]
            trial_summary = evaluate_router(bundles, trial_rules, recall_ks, hit_k)
            if trial_summary["doc_hit_count"] < base_summary["doc_hit_count"] - max_doc_hit_loss:
                continue
            if max_page_hit_loss >= 0 and trial_summary["lost"] > max_page_hit_loss:
                continue
            if best_summary is None or rank_summary_key(trial_summary) > rank_summary_key(best_summary):
                best_rule = rule
                best_summary = trial_summary
        if best_rule is None or best_summary is None:
            break
        if rank_summary_key(best_summary) <= rank_summary_key(selected_summary):
            break
        selected_rules.append(best_rule)
        used.add(rule_key(best_rule))
        selected_summary = best_summary

    selected_summary["rules"] = selected_rules
    selected_summary["router"] = router_label(selected_rules)
    return selected_summary


def load_router_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rules = payload
    elif isinstance(payload, dict):
        rules = payload.get("rules", [])
    else:
        rules = []
    if not isinstance(rules, list):
        raise ValueError(f"Router JSON must contain a list of rules: {path}")
    out: list[dict[str, Any]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        normalized = {
            "candidate_label": clean_label(str(rule.get("candidate_label", ""))),
            "conditions": list(rule.get("conditions", [])),
        }
        for key in ["mode", "profile", "hit_k"]:
            if key in rule:
                normalized[key] = rule[key]
        out.append(normalized)
    return out


def self_calibrated_rules(
    candidate_labels: list[str],
    hit_k: int,
    profile: str,
) -> list[dict[str, Any]]:
    return [
        {
            "candidate_label": candidate_label,
            "conditions": [],
            "mode": "self_calibrated",
            "profile": profile,
            "hit_k": int(hit_k),
        }
        for candidate_label in candidate_labels
    ]


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def flatten_feature_rows(bundles: list[dict[str, Any]], hit_k: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for pair in candidate_pair_rows(bundles):
        flat: dict[str, Any] = {
            "run": pair["run"],
            "qid": pair["qid"],
            "candidate_label": pair["candidate_label"],
            "question": pair["question"],
            "movement": pair["movement"],
            "base_page_rank": pair["scores"][BASE_LABEL]["page_first_rank"],
            "candidate_page_rank": pair["scores"]["candidate"]["page_first_rank"],
            f"base_page_hit@{hit_k}": pair["scores"][BASE_LABEL][f"page_hit@{hit_k}"],
            f"candidate_page_hit@{hit_k}": pair["scores"]["candidate"][f"page_hit@{hit_k}"],
            f"base_doc_hit@{hit_k}": pair["scores"][BASE_LABEL][f"doc_hit@{hit_k}"],
            f"candidate_doc_hit@{hit_k}": pair["scores"]["candidate"][f"doc_hit@{hit_k}"],
        }
        for key, value in pair["features"].items():
            flat[f"feature.{key}"] = value
        out.append(flat)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def write_routed_outputs(
    *,
    bundles: list[dict[str, Any]],
    rules: list[dict[str, Any]],
    output_dir: Path,
    recall_ks: list[int],
    hit_k: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundles_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for bundle in bundles:
        bundles_by_run[bundle["run"]].append(bundle)

    for run_label, run_bundles in bundles_by_run.items():
        predictions: dict[str, dict[str, Any]] = {}
        for bundle in run_bundles:
            selected_label = router_select(bundle, rules)
            if selected_label == BASE_LABEL:
                pred_row = dict(bundle["baseline_row"])
            else:
                pred_row = dict(bundle["pairs"][selected_label]["candidate_row"])
            pred_row["qid"] = bundle["qid"]
            predictions[bundle["qid"]] = pred_row

        summary = evaluate_router(run_bundles, rules, recall_ks, hit_k)
        summary["router_rules"] = rules
        pred_path = output_dir / f"{run_label}_routed.prediction.json"
        summary_path = output_dir / f"{run_label}_routed.summary.json"
        pred_path.write_text(json.dumps({"predictions": predictions}, ensure_ascii=False), encoding="utf-8")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"saved_routed_prediction: {pred_path}")
        print(f"saved_routed_summary: {summary_path}")


def profile_candidates(raw_profile: str) -> list[str]:
    if raw_profile == "all":
        return list(SELF_CALIBRATED_METHODS)
    return [raw_profile]


def summary_respects_loss_caps(
    summary: dict[str, Any],
    base_summary: dict[str, Any],
    *,
    max_doc_hit_loss: int,
    max_page_hit_loss: int,
) -> bool:
    if summary["doc_hit_count"] < base_summary["doc_hit_count"] - max_doc_hit_loss:
        return False
    if max_page_hit_loss >= 0 and summary["lost"] > max_page_hit_loss:
        return False
    return True


def main() -> None:
    args = parse_args()
    if int(args.hit_k) not in args.recall_ks:
        args.recall_ks = sorted(set(args.recall_ks + [int(args.hit_k)]))

    bundles = build_bundles(args)
    if not bundles:
        raise ValueError("No qids loaded for router analysis.")
    pair_rows = candidate_pair_rows(bundles)
    if not pair_rows:
        raise ValueError("No candidate predictions loaded for router analysis.")

    base_summary = summary_for_selector(bundles, lambda _bundle: BASE_LABEL, args.recall_ks, int(args.hit_k))
    candidate_labels = sorted({pair["candidate_label"] for pair in pair_rows})
    candidate_summaries = {
        label: summary_for_selector(
            bundles,
            selector=lambda bundle, candidate_label=label: (
                candidate_label if candidate_label in bundle["pairs"] else BASE_LABEL
            ),
            recall_ks=args.recall_ks,
            hit_k=int(args.hit_k),
        )
        for label in candidate_labels
    }
    oracle_summary = summary_for_selector(
        bundles,
        selector=lambda bundle: oracle_select(bundle, int(args.hit_k)),
        recall_ks=args.recall_ks,
        hit_k=int(args.hit_k),
    )

    best_rules: list[dict[str, Any]] = []
    self_calibrated_summaries: dict[str, dict[str, Any]] = {}
    selected_self_calibrated_profile = ""
    if args.router_mode == "learned" and not args.router_json:
        best_rules = learn_candidate_rules(
            bundles=bundles,
            recall_ks=args.recall_ks,
            hit_k=int(args.hit_k),
            min_accept=int(args.min_accept),
            max_doc_hit_loss=int(args.max_doc_hit_loss),
            max_page_hit_loss=int(args.max_page_hit_loss),
            pair_source_rules=int(args.pair_source_rules),
            max_rules=int(args.max_rules),
        )
    if args.router_json:
        selected_rules = load_router_json(Path(args.router_json))
        selected_summary = evaluate_router(bundles, selected_rules, args.recall_ks, int(args.hit_k))
    elif args.router_mode == "self_calibrated":
        for profile in profile_candidates(str(args.self_calibrated_profile)):
            rules = self_calibrated_rules(
                candidate_labels,
                hit_k=int(args.hit_k),
                profile=profile,
            )
            summary = evaluate_router(bundles, rules, args.recall_ks, int(args.hit_k))
            summary["profile"] = profile
            self_calibrated_summaries[profile] = summary

        eligible_profiles = {
            profile: summary
            for profile, summary in self_calibrated_summaries.items()
            if summary_respects_loss_caps(
                summary,
                base_summary,
                max_doc_hit_loss=int(args.max_doc_hit_loss),
                max_page_hit_loss=int(args.max_page_hit_loss),
            )
        }
        ranking_pool = eligible_profiles or self_calibrated_summaries
        selected_self_calibrated_profile, selected_summary = max(
            ranking_pool.items(),
            key=lambda item: rank_summary_key(item[1]),
        )
        selected_rules = list(selected_summary.get("rules", []))
    else:
        selected_summary = greedy_router(
            bundles=bundles,
            candidate_rules=best_rules,
            recall_ks=args.recall_ks,
            hit_k=int(args.hit_k),
            max_doc_hit_loss=int(args.max_doc_hit_loss),
            max_page_hit_loss=int(args.max_page_hit_loss),
            max_router_rules=int(args.max_router_rules),
            router_source_rules=int(args.router_source_rules),
        )
        selected_rules = selected_summary.get("rules", [])

    overall_rows = [{"label": BASE_LABEL, **base_summary}]
    for label in candidate_labels:
        overall_rows.append({"label": f"candidate:{label}", **candidate_summaries[label]})
    for profile, summary in self_calibrated_summaries.items():
        overall_rows.append({"label": f"self:{profile}", **summary})
    overall_rows.extend(
        [
            {"label": "oracle", **oracle_summary},
            {"label": "selected_router", **selected_summary},
        ]
    )

    if self_calibrated_summaries:
        ranked_self_rows = sorted(
            self_calibrated_summaries.values(),
            key=rank_summary_key,
            reverse=True,
        )
        rule_rows = [
            {
                "rank": idx,
                "candidate": ",".join(candidate_labels),
                "rule": display_rule_label(item["rules"][0]) if item.get("rules") else BASE_LABEL,
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
                    or ".select." in key
                },
            }
            for idx, item in enumerate(ranked_self_rows, start=1)
        ]
    else:
        rule_rows = [
            {
                "rank": idx,
                "candidate": item["candidate_label"],
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
                    or ".select." in key
                },
            }
            for idx, item in enumerate(best_rules, start=1)
        ]
    router_rows = [
        {
            "step": idx,
            "candidate": rule["candidate_label"],
            "rule": display_rule_label(rule),
        }
        for idx, rule in enumerate(selected_rules, start=1)
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
        "selection_counts",
    ]
    rule_columns = [
        "rank",
        "candidate",
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
            or ".select." in key
        }
    )
    rule_columns.extend(run_metric_columns)

    markdown_parts = [
        "# Adaptive Evidence Router Analysis",
        "",
        "## Overall",
        "",
        markdown_table(overall_rows, overall_columns),
        "## Selected Router",
        "",
        markdown_table(router_rows, ["step", "candidate", "rule"]) if router_rows else "No rule selected.\n",
        "## Candidate Rules",
        "",
        markdown_table(rule_rows, rule_columns),
    ]
    markdown = "\n".join(markdown_parts)

    payload = {
        "qid_count": len(bundles),
        "candidate_pair_count": len(pair_rows),
        "hit_k": int(args.hit_k),
        "recall_ks": args.recall_ks,
        "max_doc_hit_loss": int(args.max_doc_hit_loss),
        "max_page_hit_loss": int(args.max_page_hit_loss),
        "router_mode": str(args.router_mode),
        "self_calibrated_profile": str(args.self_calibrated_profile),
        "selected_self_calibrated_profile": selected_self_calibrated_profile,
        "base": base_summary,
        "candidates": candidate_summaries,
        "self_calibrated": self_calibrated_summaries,
        "oracle": oracle_summary,
        "selected_router": {
            "rules": selected_rules,
            "summary": selected_summary,
        },
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
        write_csv(Path(args.output_csv), flatten_feature_rows(bundles, int(args.hit_k)))
        print(f"saved_csv: {args.output_csv}")
    if args.output_router_json:
        path = Path(args.output_router_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "hit_k": int(args.hit_k),
                    "router_mode": str(args.router_mode),
                    "self_calibrated_profile": str(args.self_calibrated_profile),
                    "selected_self_calibrated_profile": selected_self_calibrated_profile,
                    "rules": selected_rules,
                    "summary": selected_summary,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"saved_router_json: {path}")
    if args.output_routed_dir:
        write_routed_outputs(
            bundles=bundles,
            rules=selected_rules,
            output_dir=Path(args.output_routed_dir),
            recall_ks=args.recall_ks,
            hit_k=int(args.hit_k),
        )


if __name__ == "__main__":
    main()
