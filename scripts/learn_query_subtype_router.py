#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
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
    metric_scores,
    movement_for_hit,
    page_doc,
    ranked_docs,
    ranked_pages,
    read_jsonl,
)


BASE_LABEL = "base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Learn an interpretable query-subtype router from observable retrieval features. "
            "The learner predicts per-query hit@k utility for each candidate reranker and "
            "routes to the highest positive expected utility, otherwise keeping base."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=3,
        metavar=("LABEL", "GOLD", "BASE"),
        default=[],
        help="Run tuple. Repeat for cross-run calibration.",
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
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-leaf", type=int, default=8)
    parser.add_argument(
        "--cv-mode",
        choices=("leave_run_out", "qid_kfold", "fit_all"),
        default="leave_run_out",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output-md", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-routed-dir", default="")
    return parser.parse_args()


def question_text(row: dict[str, Any]) -> str:
    for key in ("question", "query", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def bool_float(value: bool) -> float:
    return 1.0 if value else 0.0


def query_features(question: str) -> dict[str, float]:
    q = question.lower()
    tokens = re.findall(r"\w+", q)
    return {
        "query_len": float(len(tokens)),
        "query_char_len": float(len(q)),
        "query_digit_count": float(sum(ch.isdigit() for ch in q)),
        "query_has_numeric_cue": bool_float(
            bool(re.search(r"\d|\b(how many|how much|percentage|percent|ratio|total|sum|difference)\b", q))
        ),
        "query_has_visual_table_cue": bool_float(
            bool(re.search(r"\b(table|figure|chart|graph|image|picture|plot|diagram)\b", q))
        ),
        "query_has_page_cue": bool_float(
            bool(re.search(r"\b(page|section|appendix|slide)\b", q))
        ),
    }


def score_at_rank(pred_row: dict[str, Any] | None, rank: int) -> float:
    if pred_row is None:
        return 0.0
    items = pred_row.get("page_retrieval_results", [])
    idx = rank - 1
    if idx < 0 or idx >= len(items):
        return 0.0
    item = items[idx]
    if not isinstance(item, (list, tuple)) or len(item) < 3:
        return 0.0
    try:
        return float(item[2])
    except (TypeError, ValueError):
        return 0.0


def rank_map(pages: list[str]) -> dict[str, int]:
    return {uid: idx for idx, uid in enumerate(pages, start=1)}


def overlap_frac(left: list[str], right: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(left[:k]) & set(right[:k])) / float(k)


def candidate_features(
    *,
    gold_row: dict[str, Any],
    base_row: dict[str, Any],
    candidate_row: dict[str, Any],
    case_row: dict[str, Any],
    hit_k: int,
) -> dict[str, float]:
    question = question_text(gold_row)
    features = query_features(question)
    base_pages = ranked_pages(base_row)
    cand_pages = ranked_pages(candidate_row)
    base_docs = ranked_docs(base_row)
    cand_docs = ranked_docs(candidate_row)
    base_page_rank = rank_map(base_pages)
    candidate_top_pages = cand_pages[:hit_k]
    candidate_top_base_ranks = [
        float(base_page_rank.get(uid, len(base_pages) + 1))
        for uid in candidate_top_pages
    ]
    promoted = [rank for rank in candidate_top_base_ranks if rank > hit_k]
    boundary_promoted = [rank for rank in candidate_top_base_ranks if hit_k < rank <= 10]

    features.update(
        {
            "page_overlap_top4": overlap_frac(base_pages, cand_pages, hit_k),
            "page_overlap_top10": overlap_frac(base_pages, cand_pages, 10),
            "doc_overlap_top4": overlap_frac(base_docs, cand_docs, hit_k),
            "doc_overlap_top10": overlap_frac(base_docs, cand_docs, 10),
            "candidate_top4_mean_base_rank": mean(candidate_top_base_ranks),
            "candidate_top4_max_base_rank": max(candidate_top_base_ranks) if candidate_top_base_ranks else 0.0,
            "candidate_top4_min_base_rank": min(candidate_top_base_ranks) if candidate_top_base_ranks else 0.0,
            "candidate_promoted_from_below4_count": float(len(promoted)),
            "candidate_promoted_from_boundary_count": float(len(boundary_promoted)),
            "candidate_top4_unique_doc_frac": (
                len({page_doc(uid) for uid in candidate_top_pages}) / float(len(candidate_top_pages))
                if candidate_top_pages
                else 0.0
            ),
            "base_score_margin_4_5": score_at_rank(base_row, 4) - score_at_rank(base_row, 5),
            "candidate_score_margin_4_5": score_at_rank(candidate_row, 4) - score_at_rank(candidate_row, 5),
        }
    )

    for key, value in case_row.items():
        if isinstance(value, bool):
            features[f"case_{key}"] = bool_float(value)
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            features[f"case_{key}"] = float(value)
    return features


def hit_delta(
    *,
    gold_row: dict[str, Any],
    base_row: dict[str, Any],
    candidate_row: dict[str, Any],
    hit_k: int,
) -> tuple[int, str]:
    gold_pages = gold_page_uids(gold_row)
    base_rank = first_rank(ranked_pages(base_row), gold_pages)
    cand_rank = first_rank(ranked_pages(candidate_row), gold_pages)
    base_hit = base_rank is not None and base_rank <= hit_k
    cand_hit = cand_rank is not None and cand_rank <= hit_k
    return int(cand_hit) - int(base_hit), movement_for_hit(base_rank, cand_rank, hit_k)


@dataclass
class Example:
    run_label: str
    qid: str
    candidate_label: str
    features: dict[str, float]
    utility: float
    movement: str


@dataclass
class TreeNode:
    value: float
    n: int
    feature: str | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    def predict(self, features: dict[str, float]) -> float:
        if self.feature is None or self.threshold is None or self.left is None or self.right is None:
            return self.value
        if features.get(self.feature, 0.0) <= self.threshold:
            return self.left.predict(features)
        return self.right.predict(features)

    def to_dict(self) -> dict[str, Any]:
        out = {"value": self.value, "n": self.n}
        if self.feature is not None:
            out.update(
                {
                    "feature": self.feature,
                    "threshold": self.threshold,
                    "left": self.left.to_dict() if self.left else None,
                    "right": self.right.to_dict() if self.right else None,
                }
            )
        return out


def sse(values: list[float]) -> float:
    if not values:
        return 0.0
    avg = sum(values) / float(len(values))
    return sum((value - avg) ** 2 for value in values)


def train_tree(
    examples: list[Example],
    features: list[str],
    *,
    max_depth: int,
    min_leaf: int,
    depth: int = 0,
) -> TreeNode:
    values = [example.utility for example in examples]
    node = TreeNode(value=sum(values) / float(len(values)) if values else 0.0, n=len(examples))
    if depth >= max_depth or len(examples) < 2 * min_leaf:
        return node
    parent_sse = sse(values)
    best: tuple[float, str, float, list[Example], list[Example]] | None = None
    for feature in features:
        pairs = sorted((example.features.get(feature, 0.0), idx, example) for idx, example in enumerate(examples))
        unique_values = sorted({value for value, _, _ in pairs})
        if len(unique_values) <= 1:
            continue
        thresholds = [
            (left + right) / 2.0
            for left, right in zip(unique_values, unique_values[1:])
        ]
        for threshold in thresholds:
            left_examples = [example for value, _, example in pairs if value <= threshold]
            right_examples = [example for value, _, example in pairs if value > threshold]
            if len(left_examples) < min_leaf or len(right_examples) < min_leaf:
                continue
            child_sse = sse([example.utility for example in left_examples]) + sse(
                [example.utility for example in right_examples]
            )
            gain = parent_sse - child_sse
            if best is None or gain > best[0]:
                best = (gain, feature, threshold, left_examples, right_examples)
    if best is None or best[0] <= 1e-12:
        return node
    _, feature, threshold, left_examples, right_examples = best
    node.feature = feature
    node.threshold = threshold
    node.left = train_tree(
        left_examples,
        features,
        max_depth=max_depth,
        min_leaf=min_leaf,
        depth=depth + 1,
    )
    node.right = train_tree(
        right_examples,
        features,
        max_depth=max_depth,
        min_leaf=min_leaf,
        depth=depth + 1,
    )
    return node


def tree_lines(node: TreeNode, indent: str = "") -> list[str]:
    if node.feature is None or node.threshold is None or node.left is None or node.right is None:
        return [f"{indent}predict {node.value:.4f}  # n={node.n}"]
    lines = [f"{indent}if {node.feature} <= {node.threshold:.4g}:  # n={node.n}, mean={node.value:.4f}"]
    lines.extend(tree_lines(node.left, indent + "  "))
    lines.append(f"{indent}else:")
    lines.extend(tree_lines(node.right, indent + "  "))
    return lines


def require_file(path: Path, role: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {role}: {path}")


def load_all(args: argparse.Namespace) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, dict[str, dict[str, Any]]]],
    dict[str, list[Example]],
]:
    gold_by_run: dict[str, dict[str, dict[str, Any]]] = {}
    base_by_run: dict[str, dict[str, dict[str, Any]]] = {}
    cand_by_run: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(dict)
    case_by_run: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(dict)

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
        cand_by_run[run_label][candidate_label] = load_prediction(prediction_path)
        if case_raw == "-":
            case_by_run[run_label][candidate_label] = {}
        else:
            case_path = Path(case_raw)
            require_file(case_path, f"case JSON {candidate_label} for {run_label}")
            case_by_run[run_label][candidate_label] = load_case_json(case_path)

    examples_by_candidate: dict[str, list[Example]] = defaultdict(list)
    for run_label, gold in gold_by_run.items():
        base = base_by_run[run_label]
        for candidate_label, candidate in cand_by_run[run_label].items():
            cases = case_by_run[run_label].get(candidate_label, {})
            for qid in sorted(set(gold) & set(base) & set(candidate)):
                utility, movement = hit_delta(
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
                    Example(
                        run_label=run_label,
                        qid=qid,
                        candidate_label=candidate_label,
                        features=features,
                        utility=float(utility),
                        movement=movement,
                    )
                )
    return gold_by_run, base_by_run, cand_by_run, examples_by_candidate


def feature_names(examples: list[Example]) -> list[str]:
    names = sorted({key for example in examples for key in example.features})
    return names


def folds_for_examples(examples: list[Example], args: argparse.Namespace) -> list[tuple[str, set[tuple[str, str]]]]:
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
    folds: list[tuple[str, set[tuple[str, str]]]] = []
    fold_count = max(2, int(args.folds))
    for fold_idx in range(fold_count):
        heldout = {
            key
            for idx, key in enumerate(keys)
            if idx % fold_count == fold_idx
        }
        folds.append((f"fold:{fold_idx}", heldout))
    return folds


def write_prediction(path: Path, rows: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"predictions": rows}, ensure_ascii=False), encoding="utf-8")


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


def run_router(
    *,
    args: argparse.Namespace,
    gold_by_run: dict[str, dict[str, dict[str, Any]]],
    base_by_run: dict[str, dict[str, dict[str, Any]]],
    cand_by_run: dict[str, dict[str, dict[str, dict[str, Any]]]],
    examples_by_candidate: dict[str, list[Example]],
) -> dict[str, Any]:
    all_examples = [example for values in examples_by_candidate.values() for example in values]
    folds = folds_for_examples(all_examples, args)
    fold_reports: list[dict[str, Any]] = []
    routed_by_run: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    selected_by_run: dict[str, dict[str, str]] = defaultdict(dict)
    tree_reports_by_fold: dict[str, dict[str, Any]] = {}
    example_lookup = {
        (example.candidate_label, example.run_label, example.qid): example
        for example in all_examples
    }

    for fold_label, heldout_keys in folds:
        trees: dict[str, TreeNode] = {}
        tree_reports: dict[str, Any] = {}
        for candidate_label, examples in examples_by_candidate.items():
            train_examples = [
                example
                for example in examples
                if args.cv_mode == "fit_all" or (example.run_label, example.qid) not in heldout_keys
            ]
            if not train_examples:
                continue
            names = feature_names(train_examples)
            tree = train_tree(
                train_examples,
                names,
                max_depth=int(args.max_depth),
                min_leaf=int(args.min_leaf),
            )
            trees[candidate_label] = tree
            tree_reports[candidate_label] = {
                "tree": tree.to_dict(),
                "text": tree_lines(tree),
                "train_n": len(train_examples),
                "train_mean_utility": mean([example.utility for example in train_examples]),
            }
        tree_reports_by_fold[fold_label] = tree_reports

        routed_fold: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        selected_fold: dict[str, dict[str, str]] = defaultdict(dict)
        for run_label, gold in gold_by_run.items():
            base = base_by_run[run_label]
            qids = sorted(set(gold) & set(base))
            for qid in qids:
                key = (run_label, qid)
                if args.cv_mode != "fit_all" and key not in heldout_keys:
                    continue
                best_label = BASE_LABEL
                best_score = 0.0
                for candidate_label, candidate in cand_by_run.get(run_label, {}).items():
                    tree = trees.get(candidate_label)
                    if tree is None or qid not in candidate:
                        continue
                    example = example_lookup.get((candidate_label, run_label, qid))
                    if example is None:
                        continue
                    score = tree.predict(example.features)
                    if score > best_score:
                        best_score = score
                        best_label = candidate_label
                selected_fold[run_label][qid] = best_label
                routed_fold[run_label][qid] = (
                    base[qid]
                    if best_label == BASE_LABEL
                    else cand_by_run[run_label][best_label][qid]
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
        "final_summaries": final_summaries,
        "trees": tree_reports_by_fold,
        "routed_predictions": routed_by_run,
        "selected": selected_by_run,
    }


def render_md(report: dict[str, Any]) -> str:
    lines = ["# Query Subtype Router", ""]
    hit_k = int(report.get("hit_k", 4))
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
    lines.append("## Learned Trees")
    lines.append("")
    for fold_label, trees in sorted(report["trees"].items()):
        lines.append(f"### {fold_label}")
        lines.append("")
        for candidate_label, tree_report in sorted(trees.items()):
            lines.append(f"#### {candidate_label}")
            lines.append("")
            lines.append("```text")
            lines.extend(tree_report["text"])
            lines.append("```")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    gold_by_run, base_by_run, cand_by_run, examples_by_candidate = load_all(args)
    report = run_router(
        args=args,
        gold_by_run=gold_by_run,
        base_by_run=base_by_run,
        cand_by_run=cand_by_run,
        examples_by_candidate=examples_by_candidate,
    )

    serializable_report = {
        "folds": report["folds"],
        "final_summaries": report["final_summaries"],
        "trees": report["trees"],
        "hit_k": int(args.hit_k),
        "cv_mode": args.cv_mode,
        "max_depth": int(args.max_depth),
        "min_leaf": int(args.min_leaf),
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
