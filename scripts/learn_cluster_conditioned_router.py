#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from analyze_layout_evidence_gate import (
    DEFAULT_RECALL_KS,
    first_rank,
    gold_doc_ids,
    gold_page_uids,
    load_case_json,
    load_prediction,
    mean,
    movement_for_hit,
    ranked_docs,
    ranked_pages,
    read_jsonl,
)
from learn_query_subtype_router import (
    BASE_LABEL,
    NON_OBSERVABLE_CASE_PATTERNS,
    candidate_features,
    evaluate_routed,
    write_prediction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Learn a cluster-conditioned retrieval router. The script clusters queries using "
            "only observable base/candidate/case features, estimates candidate utility per "
            "cluster on training folds, and routes held-out queries only when the matched "
            "cluster has positive expected page-hit utility."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=3,
        metavar=("LABEL", "GOLD", "BASE"),
        default=[],
        help="Run tuple. Repeat for cross-run validation.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=4,
        metavar=("RUN_LABEL", "CANDIDATE_LABEL", "PREDICTION", "CASE_JSON"),
        default=[],
        help="Candidate prediction for a run. Use '-' for CASE_JSON when unavailable.",
    )
    parser.add_argument("--hit-k", type=int, default=4)
    parser.add_argument("--recall-k", dest="recall_ks", type=int, nargs="+", default=DEFAULT_RECALL_KS)
    parser.add_argument(
        "--cv-mode",
        choices=("leave_run_out", "qid_kfold", "fit_all"),
        default="leave_run_out",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--run-weighting",
        choices=("none", "equal_run"),
        default="equal_run",
        help=(
            "Training weight policy. equal_run gives each run equal total training weight "
            "per candidate, so large full-dev datasets do not dominate cluster utility."
        ),
    )
    parser.add_argument(
        "--cluster-count",
        type=int,
        default=0,
        help="Fixed cluster count. Use 0 to select cluster count by BIC.",
    )
    parser.add_argument("--min-clusters", type=int, default=1)
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=0,
        help="Maximum cluster count for BIC search. Use 0 for an automatic sqrt(n) cap.",
    )
    parser.add_argument("--kmeans-iters", type=int, default=60)
    parser.add_argument(
        "--min-cluster-n",
        type=int,
        default=5,
        help="Minimum raw training examples required for a cluster to be eligible.",
    )
    parser.add_argument(
        "--doc-policy",
        choices=("ignore", "nonnegative"),
        default="nonnegative",
        help="With nonnegative, a cluster can route only when expected doc-hit delta is >= 0.",
    )
    parser.add_argument(
        "--reliability-mode",
        choices=("mean", "hoeffding_lcb"),
        default="mean",
        help="How to score page-hit utility inside a cluster.",
    )
    parser.add_argument(
        "--doc-reliability-mode",
        choices=("mean", "hoeffding_lcb"),
        default="mean",
        help="How to score doc-hit utility for --doc-policy nonnegative.",
    )
    parser.add_argument("--confidence", type=float, default=0.90)
    parser.add_argument("--top-clusters", type=int, default=20)
    parser.add_argument("--output-md", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-routed-dir", default="")
    return parser.parse_args()


def require_file(path: Path, role: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {role}: {path}")


def doc_delta(
    *,
    gold_row: dict[str, Any],
    base_row: dict[str, Any],
    candidate_row: dict[str, Any],
    hit_k: int,
) -> int:
    gold_docs = gold_doc_ids(gold_row)
    base_rank = first_rank(ranked_docs(base_row), gold_docs)
    candidate_rank = first_rank(ranked_docs(candidate_row), gold_docs)
    base_hit = base_rank is not None and base_rank <= hit_k
    candidate_hit = candidate_rank is not None and candidate_rank <= hit_k
    return int(candidate_hit) - int(base_hit)


def page_delta_and_movement(
    *,
    gold_row: dict[str, Any],
    base_row: dict[str, Any],
    candidate_row: dict[str, Any],
    hit_k: int,
) -> tuple[int, str]:
    gold_pages = gold_page_uids(gold_row)
    base_rank = first_rank(ranked_pages(base_row), gold_pages)
    candidate_rank = first_rank(ranked_pages(candidate_row), gold_pages)
    base_hit = base_rank is not None and base_rank <= hit_k
    candidate_hit = candidate_rank is not None and candidate_rank <= hit_k
    return int(candidate_hit) - int(base_hit), movement_for_hit(base_rank, candidate_rank, hit_k)


@dataclass
class ClusterExample:
    run_label: str
    qid: str
    candidate_label: str
    features: dict[str, float]
    page_delta: int
    doc_delta: int
    movement: str
    weight: float = 1.0


@dataclass
class ClusterStats:
    n: int = 0
    weight_sum: float = 0.0
    weight_sq_sum: float = 0.0
    page_delta_sum: float = 0.0
    doc_delta_sum: float = 0.0
    movement_counts: Counter[str] = field(default_factory=Counter)

    def add(self, example: ClusterExample) -> None:
        weight = max(0.0, float(example.weight))
        self.n += 1
        self.weight_sum += weight
        self.weight_sq_sum += weight * weight
        self.page_delta_sum += weight * float(example.page_delta)
        self.doc_delta_sum += weight * float(example.doc_delta)
        self.movement_counts[example.movement] += 1

    @property
    def effective_n(self) -> float:
        if self.weight_sq_sum <= 0:
            return 0.0
        return (self.weight_sum * self.weight_sum) / self.weight_sq_sum

    @property
    def page_mean(self) -> float:
        return self.page_delta_sum / self.weight_sum if self.weight_sum > 0 else 0.0

    @property
    def doc_mean(self) -> float:
        return self.doc_delta_sum / self.weight_sum if self.weight_sum > 0 else 0.0


@dataclass
class Scaler:
    feature_names: list[str]
    means: list[float]
    scales: list[float]

    def transform(self, features: dict[str, float]) -> list[float]:
        values: list[float] = []
        for name, center, scale in zip(self.feature_names, self.means, self.scales):
            raw_value = features.get(name, 0.0)
            if not math.isfinite(raw_value):
                raw_value = 0.0
            values.append((raw_value - center) / scale)
        return values


@dataclass
class ClusterModel:
    candidate_label: str
    scaler: Scaler
    centroids: list[list[float]]
    stats_by_cluster: dict[int, ClusterStats]
    selected_k: int
    bic_by_k: dict[int, float]
    train_n: int
    train_weight_sum: float


def bounded_lcb(
    mean_value: float,
    effective_n: float,
    *,
    confidence: float,
    lower: float = -1.0,
    upper: float = 1.0,
) -> float:
    if effective_n <= 0:
        return float("-inf")
    clipped_confidence = min(max(float(confidence), 1e-9), 1.0 - 1e-9)
    delta = 1.0 - clipped_confidence
    radius = (upper - lower) * math.sqrt(math.log(1.0 / delta) / (2.0 * effective_n))
    return max(lower, mean_value - radius)


def utility_score(mean_value: float, effective_n: float, *, mode: str, confidence: float) -> float:
    if mode == "mean":
        return mean_value
    if mode == "hoeffding_lcb":
        return bounded_lcb(mean_value, effective_n, confidence=confidence)
    raise ValueError(f"Unknown reliability mode: {mode}")


def feature_names(examples: list[ClusterExample]) -> list[str]:
    return sorted({key for example in examples for key in example.features})


def fit_scaler(examples: list[ClusterExample], names: list[str]) -> Scaler:
    weights = [max(0.0, example.weight) for example in examples]
    total_weight = sum(weights)
    if total_weight <= 0:
        total_weight = float(len(examples))
        weights = [1.0 for _ in examples]
    means: list[float] = []
    scales: list[float] = []
    for name in names:
        values = [
            example.features.get(name, 0.0)
            if math.isfinite(example.features.get(name, 0.0))
            else 0.0
            for example in examples
        ]
        center = sum(weight * value for weight, value in zip(weights, values)) / total_weight
        variance = sum(weight * (value - center) ** 2 for weight, value in zip(weights, values)) / total_weight
        scale = math.sqrt(max(variance, 0.0))
        if scale <= 1e-12:
            scale = 1.0
        means.append(center)
        scales.append(scale)
    return Scaler(feature_names=names, means=means, scales=scales)


def squared_distance(left: list[float], right: list[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right))


def weighted_centroid(vectors: list[list[float]], weights: list[float]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    total_weight = sum(max(0.0, weight) for weight in weights)
    if total_weight <= 0:
        total_weight = float(len(vectors))
        weights = [1.0 for _ in vectors]
    return [
        sum(max(0.0, weight) * vector[idx] for vector, weight in zip(vectors, weights)) / total_weight
        for idx in range(dim)
    ]


def initialize_centroids(vectors: list[list[float]], weights: list[float], k: int) -> list[list[float]]:
    if not vectors:
        return []
    global_center = weighted_centroid(vectors, weights)
    first_idx = min(range(len(vectors)), key=lambda idx: squared_distance(vectors[idx], global_center))
    selected = [first_idx]
    while len(selected) < k:
        selected_set = set(selected)
        best_idx = None
        best_score = -1.0
        for idx, vector in enumerate(vectors):
            if idx in selected_set:
                continue
            min_dist = min(squared_distance(vector, vectors[centroid_idx]) for centroid_idx in selected)
            score = min_dist * max(1e-12, weights[idx])
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
    return [list(vectors[idx]) for idx in selected]


def assign_clusters(vectors: list[list[float]], centroids: list[list[float]]) -> list[int]:
    if not centroids:
        return [0 for _ in vectors]
    return [
        min(range(len(centroids)), key=lambda cluster_id: squared_distance(vector, centroids[cluster_id]))
        for vector in vectors
    ]


def fit_kmeans(
    vectors: list[list[float]],
    weights: list[float],
    *,
    k: int,
    max_iters: int,
) -> tuple[list[list[float]], list[int], float]:
    if not vectors:
        return [], [], 0.0
    k = max(1, min(int(k), len(vectors)))
    centroids = initialize_centroids(vectors, weights, k)
    if len(centroids) < k:
        k = len(centroids)
    assignments: list[int] = [-1 for _ in vectors]
    for _ in range(max(1, int(max_iters))):
        next_assignments = assign_clusters(vectors, centroids)
        if next_assignments == assignments:
            break
        assignments = next_assignments
        next_centroids: list[list[float]] = []
        for cluster_id in range(k):
            cluster_vectors = [vector for vector, assigned in zip(vectors, assignments) if assigned == cluster_id]
            cluster_weights = [weight for weight, assigned in zip(weights, assignments) if assigned == cluster_id]
            if cluster_vectors:
                next_centroids.append(weighted_centroid(cluster_vectors, cluster_weights))
            else:
                farthest_idx = max(
                    range(len(vectors)),
                    key=lambda idx: squared_distance(vectors[idx], centroids[assignments[idx]]),
                )
                next_centroids.append(list(vectors[farthest_idx]))
        centroids = next_centroids
    assignments = assign_clusters(vectors, centroids)
    sse = sum(
        max(0.0, weight) * squared_distance(vector, centroids[cluster_id])
        for vector, weight, cluster_id in zip(vectors, weights, assignments)
    )
    return centroids, assignments, sse


def bic_for_kmeans(*, sse: float, n: int, dim: int, k: int) -> float:
    if n <= 0:
        return float("inf")
    variance = max(float(sse) / float(n), 1e-12)
    param_count = k * max(1, dim) + k
    return float(n) * math.log(variance) + float(param_count) * math.log(float(n))


def candidate_cluster_range(n: int, args: argparse.Namespace) -> list[int]:
    if n <= 0:
        return []
    if int(args.cluster_count) > 0:
        return [max(1, min(int(args.cluster_count), n))]
    min_k = max(1, int(args.min_clusters))
    if int(args.max_clusters) > 0:
        max_k = int(args.max_clusters)
    else:
        max_k = min(12, max(min_k, int(math.sqrt(max(1, n)))))
    max_k = max(min_k, min(max_k, n))
    return list(range(min_k, max_k + 1))


def fit_cluster_model(
    candidate_label: str,
    examples: list[ClusterExample],
    args: argparse.Namespace,
) -> ClusterModel | None:
    if not examples:
        return None
    names = feature_names(examples)
    scaler = fit_scaler(examples, names)
    vectors = [scaler.transform(example.features) for example in examples]
    weights = [max(0.0, example.weight) for example in examples]
    if not names:
        vectors = [[] for _ in examples]
    best: tuple[float, int, list[list[float]], list[int]] | None = None
    bic_by_k: dict[int, float] = {}
    for k in candidate_cluster_range(len(examples), args):
        centroids, assignments, sse = fit_kmeans(
            vectors,
            weights,
            k=k,
            max_iters=int(args.kmeans_iters),
        )
        actual_k = max(1, len(centroids))
        bic = bic_for_kmeans(sse=sse, n=len(examples), dim=len(names), k=actual_k)
        bic_by_k[actual_k] = bic
        if best is None or bic < best[0]:
            best = (bic, actual_k, centroids, assignments)
    if best is None:
        return None
    _, selected_k, centroids, assignments = best
    stats_by_cluster: dict[int, ClusterStats] = defaultdict(ClusterStats)
    for example, cluster_id in zip(examples, assignments):
        stats_by_cluster[int(cluster_id)].add(example)
    return ClusterModel(
        candidate_label=candidate_label,
        scaler=scaler,
        centroids=centroids,
        stats_by_cluster=dict(stats_by_cluster),
        selected_k=selected_k,
        bic_by_k=dict(sorted(bic_by_k.items())),
        train_n=len(examples),
        train_weight_sum=sum(weights),
    )


def cluster_id_for_example(model: ClusterModel, example: ClusterExample) -> int:
    vector = model.scaler.transform(example.features)
    if not model.centroids:
        return 0
    return min(
        range(len(model.centroids)),
        key=lambda cluster_id: squared_distance(vector, model.centroids[cluster_id]),
    )


def score_stats(stats: ClusterStats, args: argparse.Namespace) -> tuple[float, float]:
    page_score = utility_score(
        stats.page_mean,
        stats.effective_n,
        mode=str(args.reliability_mode),
        confidence=float(args.confidence),
    )
    doc_score = utility_score(
        stats.doc_mean,
        stats.effective_n,
        mode=str(args.doc_reliability_mode),
        confidence=float(args.confidence),
    )
    return page_score, doc_score


def stats_to_dict(stats: ClusterStats, args: argparse.Namespace) -> dict[str, Any]:
    page_score, doc_score = score_stats(stats, args)
    return {
        "n": stats.n,
        "weight_sum": stats.weight_sum,
        "effective_n": stats.effective_n,
        "page_delta_mean": stats.page_mean,
        "page_delta_score": page_score,
        "doc_delta_mean": stats.doc_mean,
        "doc_delta_score": doc_score,
        "movement_counts": dict(sorted(stats.movement_counts.items())),
    }


def cluster_is_eligible(stats: ClusterStats, args: argparse.Namespace) -> bool:
    if stats.n < int(args.min_cluster_n):
        return False
    page_score, doc_score = score_stats(stats, args)
    if page_score <= 0:
        return False
    if args.doc_policy == "nonnegative" and doc_score < 0:
        return False
    return True


def load_all(args: argparse.Namespace) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, dict[str, dict[str, Any]]]],
    dict[str, list[ClusterExample]],
]:
    gold_by_run: dict[str, dict[str, dict[str, Any]]] = {}
    base_by_run: dict[str, dict[str, dict[str, Any]]] = {}
    candidates_by_run: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(dict)
    cases_by_run: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(dict)

    for run_label, gold_path_raw, base_path_raw in args.run:
        gold_path = Path(gold_path_raw)
        base_path = Path(base_path_raw)
        require_file(gold_path, f"gold for run {run_label}")
        require_file(base_path, f"base prediction for run {run_label}")
        gold_by_run[run_label] = {str(row["qid"]): row for row in read_jsonl(gold_path)}
        base_by_run[run_label] = load_prediction(base_path)

    for run_label, candidate_label, prediction_raw, case_raw in args.candidate:
        if run_label not in gold_by_run:
            raise ValueError(f"Unknown run label in candidate: {run_label}")
        if candidate_label == BASE_LABEL:
            raise ValueError("'base' is reserved.")
        prediction_path = Path(prediction_raw)
        require_file(prediction_path, f"candidate prediction {candidate_label} for {run_label}")
        candidates_by_run[run_label][candidate_label] = load_prediction(prediction_path)
        if case_raw == "-":
            cases_by_run[run_label][candidate_label] = {}
        else:
            case_path = Path(case_raw)
            require_file(case_path, f"case JSON {candidate_label} for {run_label}")
            cases_by_run[run_label][candidate_label] = load_case_json(case_path)

    examples_by_candidate: dict[str, list[ClusterExample]] = defaultdict(list)
    for run_label, gold in gold_by_run.items():
        base = base_by_run[run_label]
        for candidate_label, candidate in candidates_by_run[run_label].items():
            cases = cases_by_run[run_label].get(candidate_label, {})
            for qid in sorted(set(gold) & set(base) & set(candidate)):
                page_delta, movement = page_delta_and_movement(
                    gold_row=gold[qid],
                    base_row=base[qid],
                    candidate_row=candidate[qid],
                    hit_k=int(args.hit_k),
                )
                features = candidate_features(
                    gold_row=gold[qid],
                    base_row=base[qid],
                    candidate_row=candidate[qid],
                    case_row=cases.get(qid, {}),
                    hit_k=int(args.hit_k),
                )
                examples_by_candidate[candidate_label].append(
                    ClusterExample(
                        run_label=run_label,
                        qid=qid,
                        candidate_label=candidate_label,
                        features=features,
                        page_delta=page_delta,
                        doc_delta=doc_delta(
                            gold_row=gold[qid],
                            base_row=base[qid],
                            candidate_row=candidate[qid],
                            hit_k=int(args.hit_k),
                        ),
                        movement=movement,
                    )
                )
    apply_run_weights(examples_by_candidate, str(args.run_weighting))
    return gold_by_run, base_by_run, candidates_by_run, examples_by_candidate


def apply_run_weights(examples_by_candidate: dict[str, list[ClusterExample]], run_weighting: str) -> None:
    for examples in examples_by_candidate.values():
        if run_weighting == "none":
            for example in examples:
                example.weight = 1.0
            continue
        if run_weighting != "equal_run":
            raise ValueError(f"Unknown run weighting: {run_weighting}")
        counts: Counter[str] = Counter(example.run_label for example in examples)
        run_count = max(1, len(counts))
        total_examples = len(examples)
        for example in examples:
            count = counts.get(example.run_label, 0)
            example.weight = float(total_examples) / float(run_count * count) if count > 0 else 1.0


def folds_for_examples(
    examples: list[ClusterExample],
    args: argparse.Namespace,
) -> list[tuple[str, set[tuple[str, str]]]]:
    keys = sorted({(example.run_label, example.qid) for example in examples})
    if args.cv_mode == "fit_all":
        return [("fit_all", set(keys))]
    if args.cv_mode == "leave_run_out":
        run_labels = sorted({run_label for run_label, _ in keys})
        if len(run_labels) <= 1:
            raise ValueError("leave_run_out requires at least two runs; use --cv-mode qid_kfold.")
        return [
            (f"heldout:{run_label}", {(run, qid) for run, qid in keys if run == run_label})
            for run_label in run_labels
        ]
    fold_count = max(2, int(args.folds))
    return [
        (
            f"fold:{fold_idx}",
            {key for idx, key in enumerate(keys) if idx % fold_count == fold_idx},
        )
        for fold_idx in range(fold_count)
    ]


def model_to_report(model: ClusterModel, args: argparse.Namespace) -> dict[str, Any]:
    cluster_rows = []
    for cluster_id, stats in sorted(model.stats_by_cluster.items()):
        row = {"cluster": cluster_id}
        row.update(stats_to_dict(stats, args))
        cluster_rows.append(row)
    cluster_rows.sort(key=lambda row: (row["page_delta_score"], row["n"]), reverse=True)
    return {
        "candidate_label": model.candidate_label,
        "selected_k": model.selected_k,
        "train_n": model.train_n,
        "train_weight_sum": model.train_weight_sum,
        "feature_count": len(model.scaler.feature_names),
        "feature_names": model.scaler.feature_names,
        "bic_by_k": {str(k): value for k, value in model.bic_by_k.items()},
        "clusters": cluster_rows,
    }


def run_router(
    *,
    args: argparse.Namespace,
    gold_by_run: dict[str, dict[str, dict[str, Any]]],
    base_by_run: dict[str, dict[str, dict[str, Any]]],
    candidates_by_run: dict[str, dict[str, dict[str, dict[str, Any]]]],
    examples_by_candidate: dict[str, list[ClusterExample]],
) -> dict[str, Any]:
    all_examples = [example for values in examples_by_candidate.values() for example in values]
    folds = folds_for_examples(all_examples, args)
    example_lookup = {
        (example.candidate_label, example.run_label, example.qid): example
        for example in all_examples
    }
    routed_by_run: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    selected_by_run: dict[str, dict[str, str]] = defaultdict(dict)
    selected_meta_by_run: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    fold_reports: list[dict[str, Any]] = []
    model_reports_by_fold: dict[str, dict[str, Any]] = {}

    for fold_label, heldout_keys in folds:
        models: dict[str, ClusterModel] = {}
        for candidate_label, examples in sorted(examples_by_candidate.items()):
            train_examples = [
                example
                for example in examples
                if args.cv_mode == "fit_all" or (example.run_label, example.qid) not in heldout_keys
            ]
            model = fit_cluster_model(candidate_label, train_examples, args)
            if model is not None:
                models[candidate_label] = model
        model_reports_by_fold[fold_label] = {
            candidate_label: model_to_report(model, args)
            for candidate_label, model in sorted(models.items())
        }

        routed_fold: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        selected_fold: dict[str, dict[str, str]] = defaultdict(dict)
        for run_label, gold in gold_by_run.items():
            base = base_by_run[run_label]
            for qid in sorted(set(gold) & set(base)):
                key = (run_label, qid)
                if args.cv_mode != "fit_all" and key not in heldout_keys:
                    continue
                best_label = BASE_LABEL
                best_score = 0.0
                best_meta: dict[str, Any] = {
                    "selected": BASE_LABEL,
                    "score": 0.0,
                    "cluster": None,
                    "cluster_stats": None,
                }
                for candidate_label, candidate in candidates_by_run.get(run_label, {}).items():
                    if qid not in candidate:
                        continue
                    model = models.get(candidate_label)
                    example = example_lookup.get((candidate_label, run_label, qid))
                    if model is None or example is None:
                        continue
                    cluster_id = cluster_id_for_example(model, example)
                    stats = model.stats_by_cluster.get(cluster_id)
                    if stats is None or not cluster_is_eligible(stats, args):
                        continue
                    page_score, _doc_score = score_stats(stats, args)
                    if page_score > best_score:
                        best_score = page_score
                        best_label = candidate_label
                        best_meta = {
                            "selected": candidate_label,
                            "score": page_score,
                            "cluster": cluster_id,
                            "cluster_stats": stats_to_dict(stats, args),
                        }
                selected_fold[run_label][qid] = best_label
                selected_meta_by_run[run_label][qid] = best_meta
                routed_fold[run_label][qid] = (
                    base[qid]
                    if best_label == BASE_LABEL
                    else candidates_by_run[run_label][best_label][qid]
                )

        run_summaries = {}
        for run_label, routed in routed_fold.items():
            run_summaries[run_label] = evaluate_routed(
                gold=gold_by_run[run_label],
                base=base_by_run[run_label],
                routed=routed,
                selected=selected_fold[run_label],
                recall_ks=[int(k) for k in args.recall_ks],
                hit_k=int(args.hit_k),
            )
            routed_by_run[run_label].update(routed)
            selected_by_run[run_label].update(selected_fold[run_label])
        fold_reports.append({"fold": fold_label, "runs": run_summaries})

    final_summaries = {}
    for run_label, routed in routed_by_run.items():
        final_summaries[run_label] = evaluate_routed(
            gold=gold_by_run[run_label],
            base=base_by_run[run_label],
            routed=routed,
            selected=selected_by_run[run_label],
            recall_ks=[int(k) for k in args.recall_ks],
            hit_k=int(args.hit_k),
        )
    return {
        "folds": fold_reports,
        "models": model_reports_by_fold,
        "final_summaries": final_summaries,
        "routed_predictions": routed_by_run,
        "selected": selected_by_run,
        "selected_meta": selected_meta_by_run,
    }


def render_md(report: dict[str, Any]) -> str:
    lines = ["# Cluster-Conditioned Router", ""]
    hit_k = int(report.get("hit_k", 4))
    lines.append(
        "Queries are clustered with only observable base/candidate/case features. Gold labels are "
        "used only to estimate held-out cluster utility, never as clustering features."
    )
    lines.append("")
    lines.append(
        f"Clustering: deterministic weighted k-means; cluster count mode: {report['cluster_count_mode']}; "
        f"run weighting: `{report['run_weighting']}`."
    )
    lines.append("")
    lines.append("Excluded non-observable case-field patterns: " + ", ".join(f"`{x}`" for x in NON_OBSERVABLE_CASE_PATTERNS) + ".")
    lines.append("")
    lines.append("## Final Cross-Validated Summary")
    lines.append("")
    headers = ["run", "n", f"base_hit@{hit_k}", f"routed_hit@{hit_k}", "recovered", "lost", "net", "selections"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for run_label, summary in sorted(report["final_summaries"].items()):
        row = [
            run_label,
            summary["n"],
            summary["base_page_hit_at_k_count"],
            summary["page_hit_at_k_count"],
            summary["recovered"],
            summary["lost"],
            summary["net_recovered"],
            json.dumps(summary["selection_counts"], sort_keys=True),
        ]
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    lines.append("")
    lines.append("## Fold Cluster Policies")
    lines.append("")
    for fold_label, models in sorted(report["models"].items()):
        lines.append(f"### {fold_label}")
        lines.append("")
        if not models:
            lines.append("No trainable candidate models.")
            lines.append("")
            continue
        for candidate_label, model_report in sorted(models.items()):
            lines.append(f"#### {candidate_label}")
            lines.append("")
            lines.append(
                f"selected_k: `{model_report['selected_k']}`, train_n: `{model_report['train_n']}`, "
                f"feature_count: `{model_report['feature_count']}`"
            )
            lines.append("")
            cluster_headers = [
                "cluster",
                "n",
                "eff_n",
                "page_mean",
                "page_score",
                "doc_mean",
                "doc_score",
                "movement",
            ]
            lines.append("| " + " | ".join(cluster_headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(cluster_headers)) + " |")
            for row in model_report["clusters"][: int(report["top_clusters"])]:
                values = [
                    row["cluster"],
                    row["n"],
                    f"{row['effective_n']:.2f}",
                    f"{row['page_delta_mean']:.4f}",
                    f"{row['page_delta_score']:.4f}",
                    f"{row['doc_delta_mean']:.4f}",
                    f"{row['doc_delta_score']:.4f}",
                    json.dumps(row["movement_counts"], sort_keys=True),
                ]
                lines.append("| " + " | ".join(str(value) for value in values) + " |")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    gold_by_run, base_by_run, candidates_by_run, examples_by_candidate = load_all(args)
    report = run_router(
        args=args,
        gold_by_run=gold_by_run,
        base_by_run=base_by_run,
        candidates_by_run=candidates_by_run,
        examples_by_candidate=examples_by_candidate,
    )
    serializable_report = {
        "folds": report["folds"],
        "models": report["models"],
        "final_summaries": report["final_summaries"],
        "hit_k": int(args.hit_k),
        "cv_mode": str(args.cv_mode),
        "run_weighting": str(args.run_weighting),
        "cluster_count": int(args.cluster_count),
        "cluster_count_mode": "fixed" if int(args.cluster_count) > 0 else "bic",
        "min_clusters": int(args.min_clusters),
        "max_clusters": int(args.max_clusters),
        "min_cluster_n": int(args.min_cluster_n),
        "doc_policy": str(args.doc_policy),
        "reliability_mode": str(args.reliability_mode),
        "doc_reliability_mode": str(args.doc_reliability_mode),
        "confidence": float(args.confidence),
        "top_clusters": int(args.top_clusters),
        "excluded_case_feature_patterns": list(NON_OBSERVABLE_CASE_PATTERNS),
    }
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(serializable_report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved_json: {path}")
    md = render_md(serializable_report)
    if args.output_md:
        path = Path(args.output_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(md, encoding="utf-8")
        print(f"saved_md: {path}")
    else:
        print(md)
    if args.output_routed_dir:
        out_dir = Path(args.output_routed_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for run_label, routed in report["routed_predictions"].items():
            prediction_path = out_dir / f"{run_label}_routed.prediction.json"
            summary_path = out_dir / f"{run_label}_routed.summary.json"
            selected_path = out_dir / f"{run_label}_selected.json"
            write_prediction(prediction_path, routed)
            summary_path.write_text(
                json.dumps(report["final_summaries"][run_label], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            selected_path.write_text(
                json.dumps(report["selected_meta"][run_label], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"saved_routed_prediction: {prediction_path}")
            print(f"saved_routed_summary: {summary_path}")
            print(f"saved_selected: {selected_path}")


if __name__ == "__main__":
    main()
