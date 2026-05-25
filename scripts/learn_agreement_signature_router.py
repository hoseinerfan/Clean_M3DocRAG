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
    load_prediction,
    mean,
    metric_scores,
    movement_for_hit,
    page_doc,
    ranked_docs,
    ranked_pages,
    read_jsonl,
)


BASE_LABEL = "base"
SIGNATURE_LEVELS = ("full", "rank", "coarse")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Learn a complete Agreement/Disagreement Signature router. The method builds "
            "discrete observable signatures from base/candidate/support rank agreement, "
            "estimates held-out expected utility for each signature, and routes only when "
            "the matched signature has positive expected page-hit utility."
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
        help=(
            "Candidate prediction for a run. CASE_JSON is accepted for command compatibility "
            "but is intentionally ignored; signatures use only observable rankings."
        ),
    )
    parser.add_argument(
        "--support",
        action="append",
        nargs=3,
        metavar=("RUN_LABEL", "SUPPORT_LABEL", "PREDICTION"),
        default=[],
        help="Optional independent support ranking view used only for agreement signatures.",
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
        "--doc-policy",
        choices=("ignore", "nonnegative"),
        default="nonnegative",
        help="With nonnegative, a signature is eligible only if expected doc-hit delta is >= 0.",
    )
    parser.add_argument(
        "--reliability-mode",
        choices=("mean", "hoeffding_lcb"),
        default="hoeffding_lcb",
        help=(
            "How to score page-hit utility. hoeffding_lcb uses a one-sided lower confidence "
            "bound for bounded utility in [-1, 1]; mean reproduces the raw empirical router."
        ),
    )
    parser.add_argument(
        "--doc-reliability-mode",
        choices=("mean", "hoeffding_lcb"),
        default="mean",
        help=(
            "How to score doc-hit utility for --doc-policy nonnegative. The default keeps the "
            "previous empirical nonnegative-doc policy; set hoeffding_lcb for a stricter bound."
        ),
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.90,
        help="One-sided confidence level used by hoeffding_lcb.",
    )
    parser.add_argument(
        "--min-signature-n",
        type=int,
        default=1,
        help="Minimum training examples required for a matched signature or candidate prior.",
    )
    parser.add_argument(
        "--allow-candidate-prior",
        action="store_true",
        help="If no signature level was observed in training, fall back to candidate-level utility.",
    )
    parser.add_argument("--top-signatures", type=int, default=25)
    parser.add_argument("--output-md", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-routed-dir", default="")
    return parser.parse_args()


def require_file(path: Path, role: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {role}: {path}")


def write_prediction(path: Path, rows: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"predictions": rows}, ensure_ascii=False), encoding="utf-8")


def rank_map(pages: list[str]) -> dict[str, int]:
    return {uid: idx for idx, uid in enumerate(pages, start=1)}


def overlap_count(left: list[str], right: list[str], k: int) -> int:
    return len(set(left[:k]) & set(right[:k]))


def zone_for_rank(rank: int | None, hit_k: int) -> str:
    if rank is None:
        return "missing"
    if rank <= hit_k:
        return "topk"
    if rank <= 2 * hit_k:
        return "boundary"
    return "below_boundary"


def min_support_rank(uid: str, support_ranks: dict[str, dict[str, int]]) -> int | None:
    ranks = [rank_map.get(uid) for rank_map in support_ranks.values() if uid in rank_map]
    return min(ranks) if ranks else None


def support_rank_maps(
    qid: str,
    support_by_label: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for label, prediction in support_by_label.items():
        row = prediction.get(qid)
        if row is not None:
            out[label] = rank_map(ranked_pages(row))
    return out


def signature_levels(
    *,
    base_row: dict[str, Any],
    candidate_row: dict[str, Any],
    support_ranks: dict[str, dict[str, int]],
    hit_k: int,
) -> dict[str, str]:
    base_pages = ranked_pages(base_row)
    candidate_pages = ranked_pages(candidate_row)
    base_docs = ranked_docs(base_row)
    candidate_docs = ranked_docs(candidate_row)
    base_page_rank = rank_map(base_pages)
    top_candidate = candidate_pages[:hit_k]
    top_base = base_pages[:hit_k]
    top_candidate_docs = [page_doc(uid) for uid in top_candidate]
    top_base_docs = [page_doc(uid) for uid in top_base]
    candidate_base_ranks = [base_page_rank.get(uid) for uid in top_candidate]
    promoted = [
        uid
        for uid, rank in zip(top_candidate, candidate_base_ranks)
        if rank is None or rank > hit_k
    ]
    boundary_promoted = [
        uid
        for uid, rank in zip(top_candidate, candidate_base_ranks)
        if rank is not None and hit_k < rank <= 2 * hit_k
    ]
    below_boundary_promoted = [
        uid
        for uid, rank in zip(top_candidate, candidate_base_ranks)
        if rank is None or rank > 2 * hit_k
    ]
    support_topk_promoted = 0
    support_boundary_promoted = 0
    support_best_top1_zone = "none"
    if support_ranks:
        support_topk_promoted = sum(
            1
            for uid in promoted
            if (rank := min_support_rank(uid, support_ranks)) is not None and rank <= hit_k
        )
        support_boundary_promoted = sum(
            1
            for uid in promoted
            if (rank := min_support_rank(uid, support_ranks)) is not None and rank <= 2 * hit_k
        )
        support_best_top1_zone = zone_for_rank(
            min_support_rank(top_candidate[0], support_ranks) if top_candidate else None,
            hit_k,
        )

    page_overlap = overlap_count(base_pages, candidate_pages, hit_k)
    doc_overlap = overlap_count(base_docs, candidate_docs, hit_k)
    base_top1_kept = int(bool(base_pages and base_pages[0] in set(top_candidate)))
    candidate_top1_base_zone = zone_for_rank(candidate_base_ranks[0] if candidate_base_ranks else None, hit_k)
    new_doc_count = len(set(top_candidate_docs) - set(top_base_docs))
    unique_candidate_doc_count = len(set(top_candidate_docs))

    full_parts = (
        f"page_overlap={page_overlap}",
        f"doc_overlap={doc_overlap}",
        f"base_top1_kept={base_top1_kept}",
        f"candidate_top1_base_zone={candidate_top1_base_zone}",
        f"promoted={len(promoted)}",
        f"boundary_promoted={len(boundary_promoted)}",
        f"below_boundary_promoted={len(below_boundary_promoted)}",
        f"new_doc_count={new_doc_count}",
        f"unique_candidate_doc_count={unique_candidate_doc_count}",
        f"support_views={len(support_ranks)}",
        f"support_topk_promoted={support_topk_promoted}",
        f"support_boundary_promoted={support_boundary_promoted}",
        f"support_top1_zone={support_best_top1_zone}",
    )
    rank_parts = (
        f"page_overlap={page_overlap}",
        f"doc_overlap={doc_overlap}",
        f"candidate_top1_base_zone={candidate_top1_base_zone}",
        f"promoted={len(promoted)}",
        f"boundary_promoted={len(boundary_promoted)}",
        f"support_topk_promoted={support_topk_promoted}",
        f"support_boundary_promoted={support_boundary_promoted}",
    )
    coarse_parts = (
        f"page_overlap={page_overlap}",
        f"doc_overlap={doc_overlap}",
        f"promoted={len(promoted)}",
        f"boundary_promoted={len(boundary_promoted)}",
    )
    return {
        "full": "|".join(full_parts),
        "rank": "|".join(rank_parts),
        "coarse": "|".join(coarse_parts),
    }


def deltas_and_movement(
    *,
    gold_row: dict[str, Any],
    base_row: dict[str, Any],
    candidate_row: dict[str, Any],
    hit_k: int,
) -> tuple[int, int, str]:
    gold_pages = gold_page_uids(gold_row)
    gold_docs = gold_doc_ids(gold_row)
    base_page_rank = first_rank(ranked_pages(base_row), gold_pages)
    candidate_page_rank = first_rank(ranked_pages(candidate_row), gold_pages)
    base_doc_rank = first_rank(ranked_docs(base_row), gold_docs)
    candidate_doc_rank = first_rank(ranked_docs(candidate_row), gold_docs)
    base_page_hit = base_page_rank is not None and base_page_rank <= hit_k
    candidate_page_hit = candidate_page_rank is not None and candidate_page_rank <= hit_k
    base_doc_hit = base_doc_rank is not None and base_doc_rank <= hit_k
    candidate_doc_hit = candidate_doc_rank is not None and candidate_doc_rank <= hit_k
    page_delta = int(candidate_page_hit) - int(base_page_hit)
    doc_delta = int(candidate_doc_hit) - int(base_doc_hit)
    movement = movement_for_hit(base_page_rank, candidate_page_rank, hit_k)
    return page_delta, doc_delta, movement


@dataclass
class SignatureExample:
    run_label: str
    qid: str
    candidate_label: str
    signatures: dict[str, str]
    page_delta: int
    doc_delta: int
    movement: str


@dataclass
class SignatureStats:
    n: int = 0
    page_delta_sum: float = 0.0
    doc_delta_sum: float = 0.0
    movement_counts: Counter[str] = field(default_factory=Counter)

    def add(self, example: SignatureExample) -> None:
        self.n += 1
        self.page_delta_sum += float(example.page_delta)
        self.doc_delta_sum += float(example.doc_delta)
        self.movement_counts[example.movement] += 1

    @property
    def page_mean(self) -> float:
        return self.page_delta_sum / float(self.n) if self.n else 0.0

    @property
    def doc_mean(self) -> float:
        return self.doc_delta_sum / float(self.n) if self.n else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "page_delta_mean": self.page_mean,
            "doc_delta_mean": self.doc_mean,
            "movement_counts": dict(sorted(self.movement_counts.items())),
        }


def bounded_lcb(
    mean_value: float,
    n: int,
    *,
    confidence: float,
    lower: float = -1.0,
    upper: float = 1.0,
) -> float:
    if n <= 0:
        return float("-inf")
    clipped_confidence = min(max(float(confidence), 1e-9), 1.0 - 1e-9)
    delta = 1.0 - clipped_confidence
    radius = (upper - lower) * math.sqrt(math.log(1.0 / delta) / (2.0 * float(n)))
    return max(lower, mean_value - radius)


def utility_score(
    mean_value: float,
    n: int,
    *,
    mode: str,
    confidence: float,
) -> float:
    if mode == "mean":
        return mean_value
    if mode == "hoeffding_lcb":
        return bounded_lcb(mean_value, n, confidence=confidence)
    raise ValueError(f"Unknown reliability mode: {mode}")


def stats_summary(
    stats: SignatureStats,
    *,
    reliability_mode: str,
    doc_reliability_mode: str,
    confidence: float,
) -> dict[str, Any]:
    out = stats.to_dict()
    out.update(
        {
            "page_delta_score": utility_score(
                stats.page_mean,
                stats.n,
                mode=reliability_mode,
                confidence=confidence,
            ),
            "doc_delta_score": utility_score(
                stats.doc_mean,
                stats.n,
                mode=doc_reliability_mode,
                confidence=confidence,
            ),
        }
    )
    return out


def load_all(args: argparse.Namespace) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, dict[str, dict[str, Any]]]],
    dict[str, dict[str, dict[str, dict[str, Any]]]],
    list[SignatureExample],
]:
    gold_by_run: dict[str, dict[str, dict[str, Any]]] = {}
    base_by_run: dict[str, dict[str, dict[str, Any]]] = {}
    candidates_by_run: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(dict)
    supports_by_run: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(dict)

    for run_label, gold_path_raw, base_path_raw in args.run:
        gold_path = Path(gold_path_raw)
        base_path = Path(base_path_raw)
        require_file(gold_path, f"gold for run {run_label}")
        require_file(base_path, f"base prediction for run {run_label}")
        gold_by_run[run_label] = {str(row["qid"]): row for row in read_jsonl(gold_path)}
        base_by_run[run_label] = load_prediction(base_path)

    for run_label, support_label, prediction_raw in args.support:
        if run_label not in gold_by_run:
            raise ValueError(f"Unknown run label in support: {run_label}")
        prediction_path = Path(prediction_raw)
        require_file(prediction_path, f"support prediction {support_label} for run {run_label}")
        supports_by_run[run_label][support_label] = load_prediction(prediction_path)

    for run_label, candidate_label, prediction_raw, _case_raw in args.candidate:
        if run_label not in gold_by_run:
            raise ValueError(f"Unknown run label in candidate: {run_label}")
        if candidate_label == BASE_LABEL:
            raise ValueError("'base' is reserved.")
        prediction_path = Path(prediction_raw)
        require_file(prediction_path, f"candidate prediction {candidate_label} for run {run_label}")
        candidates_by_run[run_label][candidate_label] = load_prediction(prediction_path)

    examples: list[SignatureExample] = []
    for run_label, gold in gold_by_run.items():
        base = base_by_run[run_label]
        supports = supports_by_run.get(run_label, {})
        for candidate_label, candidate in candidates_by_run[run_label].items():
            for qid in sorted(set(gold) & set(base) & set(candidate)):
                support_ranks = support_rank_maps(qid, supports)
                page_delta, doc_delta, movement = deltas_and_movement(
                    gold_row=gold[qid],
                    base_row=base[qid],
                    candidate_row=candidate[qid],
                    hit_k=int(args.hit_k),
                )
                examples.append(
                    SignatureExample(
                        run_label=run_label,
                        qid=qid,
                        candidate_label=candidate_label,
                        signatures=signature_levels(
                            base_row=base[qid],
                            candidate_row=candidate[qid],
                            support_ranks=support_ranks,
                            hit_k=int(args.hit_k),
                        ),
                        page_delta=page_delta,
                        doc_delta=doc_delta,
                        movement=movement,
                    )
                )
    return gold_by_run, base_by_run, candidates_by_run, supports_by_run, examples


def folds_for_examples(
    examples: list[SignatureExample],
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


def build_signature_stats(
    train_examples: list[SignatureExample],
) -> tuple[
    dict[str, dict[str, dict[str, SignatureStats]]],
    dict[str, SignatureStats],
]:
    by_level: dict[str, dict[str, dict[str, SignatureStats]]] = {
        level: defaultdict(lambda: defaultdict(SignatureStats))
        for level in SIGNATURE_LEVELS
    }
    candidate_prior: dict[str, SignatureStats] = defaultdict(SignatureStats)
    for example in train_examples:
        candidate_prior[example.candidate_label].add(example)
        for level in SIGNATURE_LEVELS:
            by_level[level][example.candidate_label][example.signatures[level]].add(example)
    return by_level, candidate_prior


def lookup_stats(
    example: SignatureExample,
    by_level: dict[str, dict[str, dict[str, SignatureStats]]],
    candidate_prior: dict[str, SignatureStats],
    allow_candidate_prior: bool,
) -> tuple[str, str, SignatureStats | None]:
    for level in SIGNATURE_LEVELS:
        signature = example.signatures[level]
        stats = by_level.get(level, {}).get(example.candidate_label, {}).get(signature)
        if stats is not None:
            return level, signature, stats
    if allow_candidate_prior:
        stats = candidate_prior.get(example.candidate_label)
        if stats is not None:
            return "candidate_prior", example.candidate_label, stats
    return "unseen", "", None


def eligible(
    stats: SignatureStats | None,
    *,
    doc_policy: str,
    reliability_mode: str,
    doc_reliability_mode: str,
    confidence: float,
    min_signature_n: int,
) -> bool:
    if stats is None:
        return False
    if stats.n < min_signature_n:
        return False
    page_score = utility_score(
        stats.page_mean,
        stats.n,
        mode=reliability_mode,
        confidence=confidence,
    )
    if page_score <= 0.0:
        return False
    doc_score = utility_score(
        stats.doc_mean,
        stats.n,
        mode=doc_reliability_mode,
        confidence=confidence,
    )
    if doc_policy == "nonnegative" and doc_score < 0.0:
        return False
    return True


def evaluate_routed(
    *,
    gold: dict[str, dict[str, Any]],
    base: dict[str, dict[str, Any]],
    routed: dict[str, dict[str, Any]],
    selected: dict[str, str],
    recall_ks: list[int],
    hit_k: int,
) -> dict[str, Any]:
    movement_counts: Counter[str] = Counter()
    selection_counts = Counter(selected.values())
    page_hit_count = 0
    doc_hit_count = 0
    base_page_hit_count = 0
    base_doc_hit_count = 0
    page_recall: dict[int, list[float]] = defaultdict(list)
    doc_recall: dict[int, list[float]] = defaultdict(list)
    qids = sorted(set(gold) & set(base) & set(routed))
    for qid in qids:
        gold_pages = gold_page_uids(gold[qid])
        gold_docs = gold_doc_ids(gold[qid])
        base_page_rank = first_rank(ranked_pages(base[qid]), gold_pages)
        routed_page_rank = first_rank(ranked_pages(routed[qid]), gold_pages)
        base_doc_rank = first_rank(ranked_docs(base[qid]), gold_docs)
        routed_doc_rank = first_rank(ranked_docs(routed[qid]), gold_docs)
        movement_counts[movement_for_hit(base_page_rank, routed_page_rank, hit_k)] += 1
        base_page_hit_count += int(base_page_rank is not None and base_page_rank <= hit_k)
        page_hit_count += int(routed_page_rank is not None and routed_page_rank <= hit_k)
        base_doc_hit_count += int(base_doc_rank is not None and base_doc_rank <= hit_k)
        doc_hit_count += int(routed_doc_rank is not None and routed_doc_rank <= hit_k)
        scores = metric_scores(routed[qid], gold_pages, gold_docs, recall_ks, hit_k)
        for k in recall_ks:
            page_recall[int(k)].append(float(scores.get(f"page_recall@{k}", 0.0)))
            doc_recall[int(k)].append(float(scores.get(f"doc_recall@{k}", 0.0)))
    recovered = movement_counts.get("recovered", 0)
    lost = movement_counts.get("lost", 0)
    return {
        "n": len(qids),
        "base_page_hit_at_k_count": base_page_hit_count,
        "page_hit_at_k_count": page_hit_count,
        "base_doc_hit_at_k_count": base_doc_hit_count,
        "doc_hit_at_k_count": doc_hit_count,
        "movement_counts": dict(sorted(movement_counts.items())),
        "selection_counts": dict(sorted(selection_counts.items())),
        "recovered": recovered,
        "lost": lost,
        "net_recovered": recovered - lost,
        "page_recall_at_k": {str(k): mean(values) for k, values in sorted(page_recall.items())},
        "doc_recall_at_k": {str(k): mean(values) for k, values in sorted(doc_recall.items())},
    }


def positive_signature_rows(
    by_level: dict[str, dict[str, dict[str, SignatureStats]]],
    top_n: int,
    *,
    reliability_mode: str,
    doc_reliability_mode: str,
    confidence: float,
    min_signature_n: int,
    doc_policy: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for level in SIGNATURE_LEVELS:
        for candidate_label, stats_by_signature in by_level.get(level, {}).items():
            for signature, stats in stats_by_signature.items():
                if not eligible(
                    stats,
                    doc_policy=doc_policy,
                    reliability_mode=reliability_mode,
                    doc_reliability_mode=doc_reliability_mode,
                    confidence=confidence,
                    min_signature_n=min_signature_n,
                ):
                    continue
                rows.append(
                    {
                        "level": level,
                        "candidate": candidate_label,
                        "signature": signature,
                        **stats_summary(
                            stats,
                            reliability_mode=reliability_mode,
                            doc_reliability_mode=doc_reliability_mode,
                            confidence=confidence,
                        ),
                    }
                )
    rows.sort(key=lambda row: (row["page_delta_score"], row["page_delta_mean"], row["n"]), reverse=True)
    return rows[:top_n]


def run_router(
    *,
    args: argparse.Namespace,
    gold_by_run: dict[str, dict[str, dict[str, Any]]],
    base_by_run: dict[str, dict[str, dict[str, Any]]],
    candidates_by_run: dict[str, dict[str, dict[str, dict[str, Any]]]],
    examples: list[SignatureExample],
) -> dict[str, Any]:
    folds = folds_for_examples(examples, args)
    example_lookup = {
        (example.candidate_label, example.run_label, example.qid): example
        for example in examples
    }
    routed_by_run: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    selected_by_run: dict[str, dict[str, str]] = defaultdict(dict)
    fold_reports: list[dict[str, Any]] = []
    policy_reports: dict[str, Any] = {}

    for fold_label, heldout_keys in folds:
        train_examples = [
            example
            for example in examples
            if args.cv_mode == "fit_all" or (example.run_label, example.qid) not in heldout_keys
        ]
        by_level, candidate_prior = build_signature_stats(train_examples)
        policy_reports[fold_label] = {
            "train_n": len(train_examples),
            "positive_signatures": positive_signature_rows(
                by_level,
                int(args.top_signatures),
                reliability_mode=str(args.reliability_mode),
                doc_reliability_mode=str(args.doc_reliability_mode),
                confidence=float(args.confidence),
                min_signature_n=int(args.min_signature_n),
                doc_policy=str(args.doc_policy),
            ),
            "candidate_priors": {
                candidate: stats_summary(
                    stats,
                    reliability_mode=str(args.reliability_mode),
                    doc_reliability_mode=str(args.doc_reliability_mode),
                    confidence=float(args.confidence),
                )
                for candidate, stats in sorted(candidate_prior.items())
            },
        }

        routed_fold: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        selected_fold: dict[str, dict[str, str]] = defaultdict(dict)
        matched_level_counts: Counter[str] = Counter()
        for run_label, gold in gold_by_run.items():
            base = base_by_run[run_label]
            qids = sorted(set(gold) & set(base))
            for qid in qids:
                key = (run_label, qid)
                if args.cv_mode != "fit_all" and key not in heldout_keys:
                    continue
                best_label = BASE_LABEL
                best_score = 0.0
                best_level = "base"
                for candidate_label, candidate in candidates_by_run.get(run_label, {}).items():
                    if qid not in candidate:
                        continue
                    example = example_lookup.get((candidate_label, run_label, qid))
                    if example is None:
                        continue
                    level, _signature, stats = lookup_stats(
                        example,
                        by_level,
                        candidate_prior,
                        bool(args.allow_candidate_prior),
                    )
                    if not eligible(
                        stats,
                        doc_policy=str(args.doc_policy),
                        reliability_mode=str(args.reliability_mode),
                        doc_reliability_mode=str(args.doc_reliability_mode),
                        confidence=float(args.confidence),
                        min_signature_n=int(args.min_signature_n),
                    ):
                        continue
                    assert stats is not None
                    page_score = utility_score(
                        stats.page_mean,
                        stats.n,
                        mode=str(args.reliability_mode),
                        confidence=float(args.confidence),
                    )
                    if page_score > best_score:
                        best_score = page_score
                        best_label = candidate_label
                        best_level = level
                matched_level_counts[best_level] += 1
                selected_fold[run_label][qid] = best_label
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
        fold_reports.append(
            {
                "fold": fold_label,
                "runs": run_summaries,
                "matched_level_counts": dict(sorted(matched_level_counts.items())),
            }
        )

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
        "policies": policy_reports,
        "final_summaries": final_summaries,
        "routed_predictions": routed_by_run,
        "selected": selected_by_run,
    }


def render_md(report: dict[str, Any]) -> str:
    hit_k = int(report.get("hit_k", 4))
    lines = [
        "# Agreement/Disagreement Signature Router",
        "",
        (
            "The router uses only observable rank signatures from base, candidate, and optional "
            "support predictions. CASE_JSON inputs are ignored to avoid oracle/evaluation leakage."
        ),
        "",
        f"Decision: choose the candidate with positive page-hit@{hit_k} utility score "
        "under the most specific observed signature; otherwise keep base.",
        "",
    ]
    reliability_mode = str(report.get("reliability_mode", "mean"))
    doc_reliability_mode = str(report.get("doc_reliability_mode", "mean"))
    confidence = float(report.get("confidence", 0.90))
    min_signature_n = int(report.get("min_signature_n", 1))
    if reliability_mode == "hoeffding_lcb" or doc_reliability_mode == "hoeffding_lcb":
        lines.extend(
            [
                (
                    "Reliability: utility scores use one-sided Hoeffding lower confidence bounds "
                    f"where configured, confidence={confidence:.3f}, min_signature_n={min_signature_n}."
                ),
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"Reliability: raw empirical means, min_signature_n={min_signature_n}.",
                "",
            ]
        )
    if report.get("doc_policy") == "nonnegative":
        lines.extend(
            [
                "Doc policy: candidate signatures must also have nonnegative doc-hit utility score.",
                "",
            ]
        )

    lines.append("## Final Cross-Validated Summary")
    lines.append("")
    headers = ["run", "n", f"base_hit@{hit_k}", f"routed_hit@{hit_k}", "recovered", "lost", "net", "selections"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for run_label, summary in sorted(report["final_summaries"].items()):
        values = [
            run_label,
            summary["n"],
            summary["base_page_hit_at_k_count"],
            summary["page_hit_at_k_count"],
            summary["recovered"],
            summary["lost"],
            summary["net_recovered"],
            json.dumps(summary["selection_counts"], sort_keys=True),
        ]
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    lines.append("")

    lines.append("## Fold Policies")
    lines.append("")
    for fold_label, policy in sorted(report["policies"].items()):
        lines.append(f"### {fold_label}")
        lines.append("")
        priors = policy.get("candidate_priors", {})
        if priors:
            lines.append("| candidate | n | page_mean | page_score | doc_mean | doc_score |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for candidate, stats in sorted(priors.items()):
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            candidate,
                            str(stats["n"]),
                            f"{float(stats['page_delta_mean']):.4f}",
                            f"{float(stats['page_delta_score']):.4f}",
                            f"{float(stats['doc_delta_mean']):.4f}",
                            f"{float(stats['doc_delta_score']):.4f}",
                        ]
                    )
                    + " |"
                )
            lines.append("")
        rows = policy.get("positive_signatures", [])
        if rows:
            lines.append("| level | candidate | n | page_mean | page_score | doc_mean | doc_score | signature |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
            for row in rows:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(row["level"]),
                            str(row["candidate"]),
                            str(row["n"]),
                            f"{float(row['page_delta_mean']):.4f}",
                            f"{float(row['page_delta_score']):.4f}",
                            f"{float(row['doc_delta_mean']):.4f}",
                            f"{float(row['doc_delta_score']):.4f}",
                            str(row["signature"]).replace("|", "<br>"),
                        ]
                    )
                    + " |"
                )
        else:
            lines.append("No positive signatures learned for this fold.")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    gold_by_run, base_by_run, candidates_by_run, _supports_by_run, examples = load_all(args)
    report = run_router(
        args=args,
        gold_by_run=gold_by_run,
        base_by_run=base_by_run,
        candidates_by_run=candidates_by_run,
        examples=examples,
    )
    serializable_report = {
        "hit_k": int(args.hit_k),
        "cv_mode": args.cv_mode,
        "signature_levels": list(SIGNATURE_LEVELS),
        "doc_policy": args.doc_policy,
        "reliability_mode": args.reliability_mode,
        "doc_reliability_mode": args.doc_reliability_mode,
        "confidence": float(args.confidence),
        "min_signature_n": int(args.min_signature_n),
        "allow_candidate_prior": bool(args.allow_candidate_prior),
        "folds": report["folds"],
        "policies": report["policies"],
        "final_summaries": report["final_summaries"],
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
            write_prediction(prediction_path, routed)
            summary_path.write_text(
                json.dumps(report["final_summaries"][run_label], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"saved_routed_prediction: {prediction_path}")
            print(f"saved_routed_summary: {summary_path}")


if __name__ == "__main__":
    main()
