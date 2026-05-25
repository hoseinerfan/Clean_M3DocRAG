#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist, median
from typing import Any

from analyze_layout_evidence_gate import (
    DEFAULT_RECALL_KS,
    first_rank,
    gold_doc_ids,
    gold_page_uids,
    load_prediction,
    mean,
    metric_scores,
    movement_for_hit,
    page_doc,
    page_uid,
    ranked_docs,
    ranked_pages,
    read_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a local boundary graph reasoner over base top-k and near-boundary pages. "
            "The method builds a small per-query page graph, runs PPR, then applies a "
            "Gaussian/robust-z swap test to decide whether a boundary page should enter top-k."
        )
    )
    parser.add_argument("--gold", required=True, help="Gold MMQA-style JSONL.")
    parser.add_argument("--base-prediction", required=True, help="Base prediction JSON.")
    parser.add_argument(
        "--support",
        action="append",
        nargs=2,
        metavar=("LABEL", "PREDICTION"),
        default=[],
        help="Optional support prediction view, such as SPLADE graph-view or query-anchor.",
    )
    parser.add_argument("--hit-k", type=int, default=4)
    parser.add_argument("--boundary-top-pages", type=int, default=10)
    parser.add_argument("--recall-k", dest="recall_ks", type=int, nargs="+", default=DEFAULT_RECALL_KS)
    parser.add_argument("--restart-prob", type=float, default=0.30)
    parser.add_argument("--ppr-iters", type=int, default=20)
    parser.add_argument("--base-seed-weight", type=float, default=0.50)
    parser.add_argument("--support-seed-weight", type=float, default=1.00)
    parser.add_argument("--base-rrf-k", type=float, default=60.0)
    parser.add_argument("--support-rrf-k", type=float, default=60.0)
    parser.add_argument("--support-top-pages", type=int, default=100)
    parser.add_argument("--same-doc-window", type=int, default=2)
    parser.add_argument("--same-doc-edge-weight", type=float, default=1.0)
    parser.add_argument("--base-rank-window", type=int, default=1)
    parser.add_argument("--base-rank-edge-weight", type=float, default=0.15)
    parser.add_argument("--support-rank-window", type=int, default=4)
    parser.add_argument("--support-rank-edge-weight", type=float, default=0.50)
    parser.add_argument(
        "--same-top-docs-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only allow swaps from boundary pages whose document appears in base top-k docs.",
    )
    parser.add_argument(
        "--decision-test",
        choices=(
            "relative_z",
            "paired_gaussian",
            "posterior_rerank",
            "posterior_mixture",
            "posterior_disagreement",
            "posterior_temperature",
        ),
        default="relative_z",
        help=(
            "Swap decision. relative_z keeps the original graph-z boundary test. "
            "paired_gaussian uses a one-sided Gaussian test over standardized paired evidence. "
            "posterior_rerank sorts the local top pages by the graph posterior directly. "
            "posterior_mixture sorts by an entropy-weighted base-prior/graph-posterior mixture. "
            "posterior_disagreement weights graph by JS(base, graph) times graph confidence. "
            "posterior_temperature uses a rank prior plus temperature-scaled graph z-likelihood."
        ),
    )
    parser.add_argument(
        "--paired-confidence",
        type=float,
        default=0.95,
        help="One-sided confidence required by --decision-test paired_gaussian.",
    )
    parser.add_argument(
        "--paired-evidence",
        nargs="+",
        choices=("graph", "support", "base"),
        default=["graph", "support"],
        help="Self-normalized evidence sources combined by Stouffer z in paired_gaussian mode.",
    )
    parser.add_argument("--z-margin", type=float, default=0.0)
    parser.add_argument(
        "--min-boundary-z",
        type=float,
        default=-1e9,
        help=(
            "Optional absolute robust-z floor for the boundary page. The default disables "
            "the floor and only requires the boundary page to beat the weakest top-k page."
        ),
    )
    parser.add_argument("--output-prediction-json", required=True)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument("--output-case-json", default="")
    return parser.parse_args()


def parse_uid(uid: str) -> tuple[str, int | None]:
    doc_id = page_doc(uid)
    try:
        return doc_id, int(uid.rsplit("_page", 1)[1])
    except (IndexError, ValueError):
        return doc_id, None


def normalize(values: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, value) for value in values.values())
    if total <= 0:
        if not values:
            return {}
        uniform = 1.0 / float(len(values))
        return {key: uniform for key in values}
    return {key: max(0.0, value) / total for key, value in values.items()}


def add_undirected_edge(edges: dict[str, dict[str, float]], left: str, right: str, weight: float) -> None:
    if left == right or weight <= 0:
        return
    edges[left][right] += weight
    edges[right][left] += weight


def robust_z_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    center = median(values)
    deviations = [abs(value - center) for value in values]
    mad = median(deviations)
    scale = 1.4826 * mad
    if scale <= 1e-12:
        mean_value = sum(values) / float(len(values))
        variance = sum((value - mean_value) ** 2 for value in values) / float(len(values))
        scale = math.sqrt(variance)
        center = mean_value
    if scale <= 1e-12:
        return {key: 0.0 for key in scores}
    return {key: (value - center) / scale for key, value in scores.items()}


def prediction_item_by_uid(pred_row: dict[str, Any]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for item in pred_row.get("page_retrieval_results", []):
        if not isinstance(item, list) or len(item) < 2:
            continue
        try:
            uid = page_uid(str(item[0]), int(item[1]))
        except (TypeError, ValueError):
            continue
        out.setdefault(uid, item)
    return out


def support_rank_maps(
    qid: str,
    support_predictions: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, int]]:
    maps: dict[str, dict[str, int]] = {}
    for label, prediction in support_predictions.items():
        pages = ranked_pages(prediction.get(qid))
        maps[label] = {uid: rank for rank, uid in enumerate(pages, start=1)}
    return maps


def min_support_rank(uid: str, support_ranks: dict[str, dict[str, int]]) -> int | None:
    ranks = [
        rank_map[uid]
        for rank_map in support_ranks.values()
        if uid in rank_map
    ]
    if not ranks:
        return None
    return min(ranks)


def evidence_z_maps(
    *,
    local_pages: list[str],
    base_rank: dict[str, int],
    support_ranks: dict[str, dict[str, int]],
    graph_scores: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, dict[str, float]]:
    base_values = {
        uid: 1.0 / (float(args.base_rrf_k) + float(base_rank.get(uid, 10**9)))
        for uid in local_pages
    }
    support_values: dict[str, float] = {}
    for uid in local_pages:
        support_value = 0.0
        for rank_map in support_ranks.values():
            rank = rank_map.get(uid)
            if rank is not None and rank <= int(args.support_top_pages):
                support_value = max(
                    support_value,
                    1.0 / (float(args.support_rrf_k) + float(rank)),
                )
        support_values[uid] = support_value
    return {
        "base": robust_z_scores(base_values),
        "graph": robust_z_scores({uid: graph_scores.get(uid, 0.0) for uid in local_pages}),
        "support": robust_z_scores(support_values),
    }


def paired_gaussian_score(
    *,
    top_page: str,
    boundary_page: str,
    evidence_maps: dict[str, dict[str, float]],
    evidence_sources: list[str],
) -> tuple[float, dict[str, float]]:
    components: dict[str, float] = {}
    for source in evidence_sources:
        z_map = evidence_maps.get(source, {})
        components[source] = z_map.get(boundary_page, 0.0) - z_map.get(top_page, 0.0)
    if not components:
        return -math.inf, components
    z_value = sum(components.values()) / math.sqrt(float(len(components)))
    return z_value, components


def best_paired_swap(
    *,
    top_pages: list[str],
    boundary_pages: list[str],
    evidence_maps: dict[str, dict[str, float]],
    evidence_sources: list[str],
) -> tuple[str, str, float, dict[str, float]]:
    best_top = top_pages[0]
    best_boundary = boundary_pages[0]
    best_z = -math.inf
    best_components: dict[str, float] = {}
    for top_page in top_pages:
        for boundary_page in boundary_pages:
            paired_z, components = paired_gaussian_score(
                top_page=top_page,
                boundary_page=boundary_page,
                evidence_maps=evidence_maps,
                evidence_sources=evidence_sources,
            )
            if paired_z > best_z:
                best_top = top_page
                best_boundary = boundary_page
                best_z = paired_z
                best_components = components
    return best_top, best_boundary, best_z, best_components


def distribution_concentration(probabilities: dict[str, float]) -> float:
    if len(probabilities) <= 1:
        return 0.0
    values = [max(0.0, value) for value in probabilities.values()]
    total = sum(values)
    if total <= 0:
        return 0.0
    normalized = [value / total for value in values if value > 0]
    entropy = -sum(value * math.log(value) for value in normalized)
    max_entropy = math.log(float(len(probabilities)))
    if max_entropy <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (entropy / max_entropy)))


def kl_divergence(left: dict[str, float], right: dict[str, float]) -> float:
    total = 0.0
    for key, left_value in left.items():
        if left_value <= 0:
            continue
        right_value = right.get(key, 0.0)
        if right_value <= 0:
            return math.inf
        total += left_value * math.log(left_value / right_value)
    return total


def normalized_js_divergence(left: dict[str, float], right: dict[str, float]) -> float:
    left_prob = normalize(left)
    right_prob = normalize(right)
    keys = set(left_prob) | set(right_prob)
    if len(keys) <= 1:
        return 0.0
    mixture = {
        key: 0.5 * left_prob.get(key, 0.0) + 0.5 * right_prob.get(key, 0.0)
        for key in keys
    }
    js_value = 0.5 * kl_divergence(left_prob, mixture) + 0.5 * kl_divergence(right_prob, mixture)
    if not math.isfinite(js_value):
        return 1.0
    return max(0.0, min(1.0, js_value / math.log(2.0)))


def posterior_mixture_scores(
    *,
    local_pages: list[str],
    base_rank: dict[str, int],
    graph_scores: dict[str, float],
) -> tuple[dict[str, float], dict[str, Any]]:
    base_prior = normalize({
        uid: 1.0 / float(base_rank.get(uid, 10**9))
        for uid in local_pages
    })
    graph_posterior = normalize({
        uid: graph_scores.get(uid, 0.0)
        for uid in local_pages
    })
    base_concentration = distribution_concentration(base_prior)
    graph_concentration = distribution_concentration(graph_posterior)
    total_concentration = base_concentration + graph_concentration
    if total_concentration <= 1e-12:
        base_weight = 0.5
    else:
        base_weight = base_concentration / total_concentration
    mixture_scores = {
        uid: base_weight * base_prior.get(uid, 0.0)
        + (1.0 - base_weight) * graph_posterior.get(uid, 0.0)
        for uid in local_pages
    }
    diagnostics = {
        "base_weight": base_weight,
        "graph_weight": 1.0 - base_weight,
        "base_concentration": base_concentration,
        "graph_concentration": graph_concentration,
        "base_prior": base_prior,
        "graph_posterior": graph_posterior,
    }
    return mixture_scores, diagnostics


def posterior_disagreement_scores(
    *,
    local_pages: list[str],
    base_rank: dict[str, int],
    graph_scores: dict[str, float],
) -> tuple[dict[str, float], dict[str, Any]]:
    base_prior = normalize({
        uid: 1.0 / float(base_rank.get(uid, 10**9))
        for uid in local_pages
    })
    graph_posterior = normalize({
        uid: graph_scores.get(uid, 0.0)
        for uid in local_pages
    })
    base_concentration = distribution_concentration(base_prior)
    graph_concentration = distribution_concentration(graph_posterior)
    movement_strength = normalized_js_divergence(base_prior, graph_posterior)
    graph_weight = max(0.0, min(1.0, graph_concentration * movement_strength))
    base_weight = 1.0 - graph_weight
    mixture_scores = {
        uid: base_weight * base_prior.get(uid, 0.0)
        + graph_weight * graph_posterior.get(uid, 0.0)
        for uid in local_pages
    }
    diagnostics = {
        "base_weight": base_weight,
        "graph_weight": graph_weight,
        "base_concentration": base_concentration,
        "graph_concentration": graph_concentration,
        "base_graph_js": movement_strength,
        "base_prior": base_prior,
        "graph_posterior": graph_posterior,
    }
    return mixture_scores, diagnostics


def posterior_temperature_scores(
    *,
    local_pages: list[str],
    base_rank: dict[str, int],
    graph_scores: dict[str, float],
) -> tuple[dict[str, float], dict[str, Any]]:
    base_prior = normalize({
        uid: 1.0 / float(base_rank.get(uid, 10**9))
        for uid in local_pages
    })
    graph_posterior = normalize({
        uid: graph_scores.get(uid, 0.0)
        for uid in local_pages
    })
    graph_z = robust_z_scores({
        uid: graph_scores.get(uid, 0.0)
        for uid in local_pages
    })
    graph_concentration = distribution_concentration(graph_posterior)
    graph_entropy = max(1e-12, 1.0 - graph_concentration)
    graph_beta = 1.0 / graph_entropy
    scores = {
        uid: math.log(max(base_prior.get(uid, 0.0), 1e-300))
        + graph_beta * graph_z.get(uid, 0.0)
        for uid in local_pages
    }
    diagnostics = {
        "base_weight": None,
        "graph_weight": None,
        "base_concentration": distribution_concentration(base_prior),
        "graph_concentration": graph_concentration,
        "base_graph_js": normalized_js_divergence(base_prior, graph_posterior),
        "temperature": graph_entropy,
        "graph_beta": graph_beta,
        "base_prior": base_prior,
        "graph_posterior": graph_posterior,
    }
    return scores, diagnostics


def reorder_local_by_scores(
    *,
    base_pages: list[str],
    local_pages: list[str],
    scores: dict[str, float],
    base_rank: dict[str, int],
) -> list[str]:
    local_set = set(local_pages)
    reordered_local = sorted(
        local_pages,
        key=lambda uid: (-scores.get(uid, 0.0), base_rank.get(uid, 10**9)),
    )
    return reordered_local + [uid for uid in base_pages if uid not in local_set]


def local_graph_scores(
    *,
    local_pages: list[str],
    base_rank: dict[str, int],
    support_ranks: dict[str, dict[str, int]],
    args: argparse.Namespace,
) -> tuple[dict[str, float], dict[str, Any]]:
    seed: dict[str, float] = {}
    for uid in local_pages:
        base_value = 1.0 / (float(args.base_rrf_k) + float(base_rank.get(uid, 10**9)))
        support_value = 0.0
        for rank_map in support_ranks.values():
            rank = rank_map.get(uid)
            if rank is not None and rank <= int(args.support_top_pages):
                support_value = max(
                    support_value,
                    1.0 / (float(args.support_rrf_k) + float(rank)),
                )
        seed[uid] = float(args.base_seed_weight) * base_value + float(args.support_seed_weight) * support_value
    restart = normalize(seed)

    edges: dict[str, dict[str, float]] = {uid: defaultdict(float) for uid in local_pages}

    parsed = {uid: parse_uid(uid) for uid in local_pages}
    for i, left in enumerate(local_pages):
        left_doc, left_page = parsed[left]
        for right in local_pages[i + 1 :]:
            right_doc, right_page = parsed[right]
            if left_doc == right_doc and left_page is not None and right_page is not None:
                page_distance = abs(left_page - right_page)
                if 0 < page_distance <= int(args.same_doc_window):
                    add_undirected_edge(
                        edges,
                        left,
                        right,
                        float(args.same_doc_edge_weight) / float(page_distance),
                    )

            rank_distance = abs(base_rank.get(left, 10**9) - base_rank.get(right, 10**9))
            if 0 < rank_distance <= int(args.base_rank_window):
                add_undirected_edge(
                    edges,
                    left,
                    right,
                    float(args.base_rank_edge_weight) / float(rank_distance),
                )

    for rank_map in support_ranks.values():
        support_local = [
            uid
            for uid in local_pages
            if rank_map.get(uid, 10**9) <= int(args.support_top_pages)
        ]
        for i, left in enumerate(support_local):
            for right in support_local[i + 1 :]:
                rank_distance = abs(rank_map.get(left, 10**9) - rank_map.get(right, 10**9))
                if 0 < rank_distance <= int(args.support_rank_window):
                    add_undirected_edge(
                        edges,
                        left,
                        right,
                        float(args.support_rank_edge_weight) / float(rank_distance),
                    )

    transitions = {uid: normalize(dict(neighbors)) for uid, neighbors in edges.items()}
    scores = dict(restart)
    for _ in range(int(args.ppr_iters)):
        next_scores = {uid: float(args.restart_prob) * restart.get(uid, 0.0) for uid in local_pages}
        for source, source_score in scores.items():
            neighbors = transitions.get(source, {})
            if not neighbors:
                for uid, restart_weight in restart.items():
                    next_scores[uid] += (1.0 - float(args.restart_prob)) * source_score * restart_weight
                continue
            for target, weight in neighbors.items():
                next_scores[target] += (1.0 - float(args.restart_prob)) * source_score * weight
        scores = next_scores

    diagnostics = {
        "restart": restart,
        "edge_count": sum(len(neighbors) for neighbors in edges.values()) // 2,
    }
    return scores, diagnostics


def rerank_one(
    *,
    qid: str,
    base_row: dict[str, Any],
    support_predictions: dict[str, dict[str, dict[str, Any]]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_pages = ranked_pages(base_row)
    if len(base_pages) <= int(args.hit_k):
        return base_row, {"qid": qid, "accepted": False, "reason": "not_enough_pages"}

    posterior_modes = {
        "posterior_rerank",
        "posterior_mixture",
        "posterior_disagreement",
        "posterior_temperature",
    }
    boundary_top = max(int(args.boundary_top_pages), int(args.hit_k) + 1)
    local_pages = base_pages[:boundary_top]
    top_pages = base_pages[: int(args.hit_k)]
    boundary_pages = base_pages[int(args.hit_k) : boundary_top]
    if args.same_top_docs_only and args.decision_test not in posterior_modes:
        top_docs = {page_doc(uid) for uid in top_pages}
        boundary_pages = [uid for uid in boundary_pages if page_doc(uid) in top_docs]
    if not boundary_pages:
        return base_row, {"qid": qid, "accepted": False, "reason": "empty_boundary"}

    base_rank = {uid: rank for rank, uid in enumerate(base_pages, start=1)}
    support_ranks = support_rank_maps(qid, support_predictions)
    graph_scores, graph_diag = local_graph_scores(
        local_pages=local_pages,
        base_rank=base_rank,
        support_ranks=support_ranks,
        args=args,
    )
    evidence_maps = evidence_z_maps(
        local_pages=local_pages,
        base_rank=base_rank,
        support_ranks=support_ranks,
        graph_scores=graph_scores,
        args=args,
    )
    z_scores = evidence_maps["graph"]
    paired_z: float | None = None
    paired_p: float | None = None
    paired_z_threshold: float | None = None
    paired_components: dict[str, float] = {}
    posterior_scores = graph_scores
    posterior_diag: dict[str, Any] = {}

    if args.decision_test in posterior_modes:
        if args.decision_test == "posterior_mixture":
            posterior_scores, posterior_diag = posterior_mixture_scores(
                local_pages=local_pages,
                base_rank=base_rank,
                graph_scores=graph_scores,
            )
        elif args.decision_test == "posterior_disagreement":
            posterior_scores, posterior_diag = posterior_disagreement_scores(
                local_pages=local_pages,
                base_rank=base_rank,
                graph_scores=graph_scores,
            )
        elif args.decision_test == "posterior_temperature":
            posterior_scores, posterior_diag = posterior_temperature_scores(
                local_pages=local_pages,
                base_rank=base_rank,
                graph_scores=graph_scores,
            )
        else:
            posterior_diag = {
                "base_weight": 0.0,
                "graph_weight": 1.0,
                "base_concentration": None,
                "graph_concentration": distribution_concentration(normalize({
                    uid: graph_scores.get(uid, 0.0)
                    for uid in local_pages
                })),
            }
        weakest_top = min(top_pages, key=lambda uid: z_scores.get(uid, -10**9))
        best_boundary = max(boundary_pages, key=lambda uid: z_scores.get(uid, -10**9))
        reordered_pages = reorder_local_by_scores(
            base_pages=base_pages,
            local_pages=local_pages,
            scores=posterior_scores,
            base_rank=base_rank,
        )
        accepted = reordered_pages != base_pages
    elif args.decision_test == "paired_gaussian":
        confidence = float(args.paired_confidence)
        if not 0.5 < confidence < 1.0:
            raise ValueError("--paired-confidence must be between 0.5 and 1.0")
        weakest_top, best_boundary, paired_z, paired_components = best_paired_swap(
            top_pages=top_pages,
            boundary_pages=boundary_pages,
            evidence_maps=evidence_maps,
            evidence_sources=[str(source) for source in args.paired_evidence],
        )
        paired_p = NormalDist().cdf(float(paired_z))
        paired_z_threshold = NormalDist().inv_cdf(confidence)
    else:
        weakest_top = min(top_pages, key=lambda uid: z_scores.get(uid, -10**9))
        best_boundary = max(boundary_pages, key=lambda uid: z_scores.get(uid, -10**9))

    weakest_top_z = z_scores.get(weakest_top, 0.0)
    best_boundary_z = z_scores.get(best_boundary, 0.0)
    if args.decision_test in posterior_modes:
        pass
    elif args.decision_test == "paired_gaussian":
        accepted = bool(
            best_boundary_z >= float(args.min_boundary_z)
            and paired_z is not None
            and paired_z_threshold is not None
            and paired_z >= paired_z_threshold
        )
    else:
        accepted = bool(
            best_boundary_z >= float(args.min_boundary_z)
            and best_boundary_z > weakest_top_z + float(args.z_margin)
        )

    case = {
        "qid": qid,
        "accepted": accepted,
        "weakest_top_page": weakest_top,
        "weakest_top_base_rank": base_rank.get(weakest_top),
        "weakest_top_graph_score": graph_scores.get(weakest_top, 0.0),
        "weakest_top_z": weakest_top_z,
        "best_boundary_page": best_boundary,
        "best_boundary_base_rank": base_rank.get(best_boundary),
        "best_boundary_graph_score": graph_scores.get(best_boundary, 0.0),
        "best_boundary_z": best_boundary_z,
        "z_margin": best_boundary_z - weakest_top_z,
        "local_page_count": len(local_pages),
        "posterior_rerank_changed_count": (
            sum(1 for left, right in zip(local_pages, reordered_pages[: len(local_pages)]) if left != right)
            if args.decision_test in posterior_modes
            else None
        ),
        "posterior_top_pages": (
            reordered_pages[: min(10, len(local_pages))]
            if args.decision_test in posterior_modes
            else []
        ),
        "posterior_base_weight": posterior_diag.get("base_weight"),
        "posterior_graph_weight": posterior_diag.get("graph_weight"),
        "posterior_base_concentration": posterior_diag.get("base_concentration"),
        "posterior_graph_concentration": posterior_diag.get("graph_concentration"),
        "posterior_base_graph_js": posterior_diag.get("base_graph_js"),
        "posterior_temperature": posterior_diag.get("temperature"),
        "posterior_graph_beta": posterior_diag.get("graph_beta"),
        "boundary_page_count": len(boundary_pages),
        "graph_edge_count": graph_diag["edge_count"],
        "decision_test": str(args.decision_test),
        "selection_mode": str(args.decision_test),
        "paired_evidence": [str(source) for source in args.paired_evidence],
        "paired_z": paired_z,
        "paired_p": paired_p,
        "paired_confidence": float(args.paired_confidence),
        "paired_z_threshold": paired_z_threshold,
        "paired_components": paired_components,
        "support_min_rank_best_boundary": min_support_rank(best_boundary, support_ranks),
        "support_min_rank_weakest_top": min_support_rank(weakest_top, support_ranks),
    }
    if not accepted:
        return base_row, case

    if args.decision_test not in posterior_modes:
        reordered_pages = list(base_pages)
        top_idx = reordered_pages.index(weakest_top)
        boundary_idx = reordered_pages.index(best_boundary)
        reordered_pages[top_idx], reordered_pages[boundary_idx] = (
            reordered_pages[boundary_idx],
            reordered_pages[top_idx],
        )

    item_map = prediction_item_by_uid(base_row)
    output_items = [item_map[uid] for uid in reordered_pages if uid in item_map]
    output_row = dict(base_row)
    output_row["qid"] = str(base_row.get("qid", qid))
    output_row["page_retrieval_results"] = output_items
    return output_row, case


def write_prediction(path: Path, rows: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"predictions": rows}, ensure_ascii=False),
        encoding="utf-8",
    )


def evaluate(
    *,
    gold: dict[str, dict[str, Any]],
    base: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    cases: list[dict[str, Any]],
    recall_ks: list[int],
    hit_k: int,
) -> dict[str, Any]:
    page_recall: dict[int, list[float]] = defaultdict(list)
    doc_recall: dict[int, list[float]] = defaultdict(list)
    movement_counts: Counter[str] = Counter()
    page_hit_count = 0
    doc_hit_count = 0
    base_page_hit_count = 0
    base_doc_hit_count = 0
    oracle_boundary_recoverable_count = 0

    case_by_qid = {str(row["qid"]): row for row in cases}
    qids = sorted(set(gold) & set(base) & set(candidate))
    for qid in qids:
        gold_pages = gold_page_uids(gold[qid])
        gold_docs = gold_doc_ids(gold[qid])
        base_scores = metric_scores(base[qid], gold_pages, gold_docs, recall_ks, hit_k)
        candidate_scores = metric_scores(candidate[qid], gold_pages, gold_docs, recall_ks, hit_k)
        base_page_rank = first_rank(ranked_pages(base[qid]), gold_pages)
        cand_page_rank = first_rank(ranked_pages(candidate[qid]), gold_pages)
        base_doc_rank = first_rank(ranked_docs(base[qid]), gold_docs)
        cand_doc_rank = first_rank(ranked_docs(candidate[qid]), gold_docs)

        base_page_hit_count += int(base_page_rank is not None and base_page_rank <= hit_k)
        base_doc_hit_count += int(base_doc_rank is not None and base_doc_rank <= hit_k)
        page_hit_count += int(cand_page_rank is not None and cand_page_rank <= hit_k)
        doc_hit_count += int(cand_doc_rank is not None and cand_doc_rank <= hit_k)
        movement_counts[movement_for_hit(base_page_rank, cand_page_rank, hit_k)] += 1

        if (
            (base_page_rank is None or base_page_rank > hit_k)
            and base_page_rank is not None
            and base_page_rank <= 10
        ):
            oracle_boundary_recoverable_count += 1

        for k in recall_ks:
            page_recall[int(k)].append(float(candidate_scores.get(f"page_recall@{k}", 0.0)))
            doc_recall[int(k)].append(float(candidate_scores.get(f"doc_recall@{k}", 0.0)))
        case_by_qid[qid]["base_first_gold_page_rank"] = base_page_rank
        case_by_qid[qid]["candidate_first_gold_page_rank"] = cand_page_rank

    accepted_count = sum(1 for row in cases if row.get("accepted"))
    recovered = movement_counts.get("recovered", 0)
    lost = movement_counts.get("lost", 0)
    return {
        "n": len(qids),
        "accepted_count": accepted_count,
        "accept_frac": accepted_count / float(len(qids)) if qids else 0.0,
        "base_page_hit_at_k_count": base_page_hit_count,
        "page_hit_at_k_count": page_hit_count,
        "base_doc_hit_at_k_count": base_doc_hit_count,
        "doc_hit_at_k_count": doc_hit_count,
        "movement_counts": dict(sorted(movement_counts.items())),
        "recovered": recovered,
        "lost": lost,
        "net_recovered": recovered - lost,
        "oracle_boundary_recoverable_count": oracle_boundary_recoverable_count,
        "page_recall_at_k": {str(k): mean(values) for k, values in sorted(page_recall.items())},
        "doc_recall_at_k": {str(k): mean(values) for k, values in sorted(doc_recall.items())},
    }


def main() -> None:
    args = parse_args()
    gold_rows = {str(row["qid"]): row for row in read_jsonl(Path(args.gold))}
    base_prediction = load_prediction(Path(args.base_prediction))
    support_predictions = {
        str(label): load_prediction(Path(path))
        for label, path in args.support
    }

    output_rows: dict[str, dict[str, Any]] = {}
    cases: list[dict[str, Any]] = []
    for qid in sorted(set(gold_rows) & set(base_prediction)):
        output_row, case = rerank_one(
            qid=qid,
            base_row=base_prediction[qid],
            support_predictions=support_predictions,
            args=args,
        )
        output_rows[qid] = output_row
        cases.append(case)

    output_prediction = Path(args.output_prediction_json)
    write_prediction(output_prediction, output_rows)

    summary = evaluate(
        gold=gold_rows,
        base=base_prediction,
        candidate=output_rows,
        cases=cases,
        recall_ks=[int(k) for k in args.recall_ks],
        hit_k=int(args.hit_k),
    )
    summary.update(
        {
            "gold": str(args.gold),
            "base_prediction": str(args.base_prediction),
            "supports": {str(label): str(path) for label, path in args.support},
            "hit_k": int(args.hit_k),
            "boundary_top_pages": int(args.boundary_top_pages),
            "restart_prob": float(args.restart_prob),
            "ppr_iters": int(args.ppr_iters),
            "base_seed_weight": float(args.base_seed_weight),
            "support_seed_weight": float(args.support_seed_weight),
            "same_top_docs_only": bool(args.same_top_docs_only),
            "decision_test": str(args.decision_test),
            "paired_confidence": float(args.paired_confidence),
            "paired_evidence": [str(source) for source in args.paired_evidence],
            "z_margin": float(args.z_margin),
            "min_boundary_z": float(args.min_boundary_z),
        }
    )

    output_summary = Path(args.output_summary_json)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.output_case_json:
        output_case = Path(args.output_case_json)
        output_case.parent.mkdir(parents=True, exist_ok=True)
        output_case.write_text(json.dumps({"cases": cases}, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"saved_prediction: {output_prediction}")
    print(f"saved_summary: {output_summary}")
    if args.output_case_json:
        print(f"saved_cases: {args.output_case_json}")
    print(f"n {summary['n']}")
    print(f"accepted {summary['accepted_count']}")
    print(f"page_hit_at_{int(args.hit_k)}_count {summary['page_hit_at_k_count']}")
    print(f"base_page_hit_at_{int(args.hit_k)}_count {summary['base_page_hit_at_k_count']}")
    print(f"recovered {summary['recovered']}")
    print(f"lost {summary['lost']}")
    print(f"net_recovered {summary['net_recovered']}")
    print(f"page_recall_at_k {summary['page_recall_at_k']}")
    print(f"doc_recall_at_k {summary['doc_recall_at_k']}")


if __name__ == "__main__":
    main()
