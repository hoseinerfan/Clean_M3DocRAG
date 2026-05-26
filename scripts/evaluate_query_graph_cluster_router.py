#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_query_semantic_graph import (
    DEFAULT_RECALL_KS,
    first_rank,
    gold_doc_ids,
    gold_page_uids,
    load_prediction,
    movement_for_hit,
    page_doc,
    ranked_docs,
    ranked_pages,
    read_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate query-semantic-graph clusters as a non-oracle specialist router. "
            "Clusters are fixed from graph features. Gold labels are used only on train folds "
            "to select positive clusters, then the selected cluster rule is evaluated on held-out qids."
        )
    )
    parser.add_argument("--gold", required=True)
    parser.add_argument("--base-prediction", required=True)
    parser.add_argument("--qid-clusters-json", required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Candidate prediction. Repeat for multiple specialists.",
    )
    parser.add_argument("--hit-k", type=int, default=4)
    parser.add_argument("--recall-k", dest="recall_ks", type=int, nargs="+", default=DEFAULT_RECALL_KS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-train-cluster-n", type=int, default=3)
    parser.add_argument("--min-train-page-delta", type=int, default=1)
    parser.add_argument("--doc-policy", choices=["ignore", "nonnegative"], default="nonnegative")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-routed-dir", default="")
    return parser.parse_args()


def parse_labeled_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Expected LABEL=PATH, got {raw!r}")
    label, path = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Empty label in {raw!r}")
    return label, Path(path)


def load_cluster_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"Expected query cluster JSON list: {path}")
    rows: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("qid", "")).strip()
        cluster = str(row.get("cluster", "")).strip()
        if qid and cluster:
            rows[qid] = row
    if not rows:
        raise ValueError(f"No qid cluster rows found in {path}")
    return rows


def write_prediction(path: Path, rows: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"predictions": rows}, ensure_ascii=False), encoding="utf-8")


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / float(len(gold))


def metric_scores(row: dict[str, Any], gold_pages: set[str], gold_docs: set[str], recall_ks: list[int]) -> dict[str, Any]:
    pages = ranked_pages(row)
    docs = ranked_docs(row)
    out: dict[str, Any] = {}
    for k in recall_ks:
        out[f"page_recall@{k}"] = recall_at_k(pages, gold_pages, k)
        out[f"doc_recall@{k}"] = recall_at_k(docs, gold_docs, k)
    return out


def mean(values: list[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def page_and_doc_delta(
    *,
    gold_row: dict[str, Any],
    base_row: dict[str, Any],
    candidate_row: dict[str, Any],
    hit_k: int,
) -> tuple[int, int, str]:
    gold_pages = gold_page_uids(gold_row)
    gold_docs = gold_doc_ids(gold_row)
    base_page_rank = first_rank(ranked_pages(base_row), gold_pages)
    cand_page_rank = first_rank(ranked_pages(candidate_row), gold_pages)
    base_doc_rank = first_rank(ranked_docs(base_row), gold_docs)
    cand_doc_rank = first_rank(ranked_docs(candidate_row), gold_docs)
    base_page_hit = base_page_rank is not None and base_page_rank <= hit_k
    cand_page_hit = cand_page_rank is not None and cand_page_rank <= hit_k
    base_doc_hit = base_doc_rank is not None and base_doc_rank <= hit_k
    cand_doc_hit = cand_doc_rank is not None and cand_doc_rank <= hit_k
    return (
        int(cand_page_hit) - int(base_page_hit),
        int(cand_doc_hit) - int(base_doc_hit),
        movement_for_hit(base_page_rank, cand_page_rank, hit_k),
    )


def fold_id(qid: str, fold_count: int) -> int:
    # Stable enough across Python versions and independent of hash randomization.
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(qid)) % fold_count


def cluster_stats_for_qids(
    *,
    qids: list[str],
    clusters: dict[str, dict[str, Any]],
    gold: dict[str, dict[str, Any]],
    base: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    hit_k: int,
) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "n": 0,
            "page_delta": 0,
            "doc_delta": 0,
            "movement_counts": Counter(),
            "qids": [],
        }
    )
    for qid in qids:
        if qid not in clusters or qid not in gold or qid not in base or qid not in candidate:
            continue
        cluster = str(clusters[qid]["cluster"])
        page_delta, doc_delta, movement = page_and_doc_delta(
            gold_row=gold[qid],
            base_row=base[qid],
            candidate_row=candidate[qid],
            hit_k=hit_k,
        )
        row = stats[cluster]
        row["n"] += 1
        row["page_delta"] += page_delta
        row["doc_delta"] += doc_delta
        row["movement_counts"][movement] += 1
        row["qids"].append(qid)
    out: dict[str, dict[str, Any]] = {}
    for cluster, row in stats.items():
        out[cluster] = {
            "n": int(row["n"]),
            "page_delta": int(row["page_delta"]),
            "doc_delta": int(row["doc_delta"]),
            "page_delta_mean": float(row["page_delta"]) / float(row["n"]) if row["n"] else 0.0,
            "doc_delta_mean": float(row["doc_delta"]) / float(row["n"]) if row["n"] else 0.0,
            "movement_counts": dict(sorted(row["movement_counts"].items())),
            "qids": sorted(row["qids"]),
        }
    return out


def eligible_cluster(row: dict[str, Any], args: argparse.Namespace) -> bool:
    if int(row.get("n", 0)) < int(args.min_train_cluster_n):
        return False
    if int(row.get("page_delta", 0)) < int(args.min_train_page_delta):
        return False
    if args.doc_policy == "nonnegative" and int(row.get("doc_delta", 0)) < 0:
        return False
    return True


def evaluate_routed(
    *,
    qids: list[str],
    gold: dict[str, dict[str, Any]],
    base: dict[str, dict[str, Any]],
    routed: dict[str, dict[str, Any]],
    selected: dict[str, str],
    recall_ks: list[int],
    hit_k: int,
) -> dict[str, Any]:
    movement_counts: Counter[str] = Counter()
    selection_counts = Counter(selected.values())
    base_page_hit = 0
    routed_page_hit = 0
    base_doc_hit = 0
    routed_doc_hit = 0
    page_recall: dict[int, list[float]] = defaultdict(list)
    doc_recall: dict[int, list[float]] = defaultdict(list)
    for qid in qids:
        if qid not in gold or qid not in base or qid not in routed:
            continue
        gold_pages = gold_page_uids(gold[qid])
        gold_docs = gold_doc_ids(gold[qid])
        base_page_rank = first_rank(ranked_pages(base[qid]), gold_pages)
        routed_page_rank = first_rank(ranked_pages(routed[qid]), gold_pages)
        base_doc_rank = first_rank(ranked_docs(base[qid]), gold_docs)
        routed_doc_rank = first_rank(ranked_docs(routed[qid]), gold_docs)
        movement_counts[movement_for_hit(base_page_rank, routed_page_rank, hit_k)] += 1
        base_page_hit += int(base_page_rank is not None and base_page_rank <= hit_k)
        routed_page_hit += int(routed_page_rank is not None and routed_page_rank <= hit_k)
        base_doc_hit += int(base_doc_rank is not None and base_doc_rank <= hit_k)
        routed_doc_hit += int(routed_doc_rank is not None and routed_doc_rank <= hit_k)
        scores = metric_scores(routed[qid], gold_pages, gold_docs, recall_ks)
        for k in recall_ks:
            page_recall[k].append(float(scores[f"page_recall@{k}"]))
            doc_recall[k].append(float(scores[f"doc_recall@{k}"]))
    recovered = int(movement_counts.get("recovered", 0))
    lost = int(movement_counts.get("lost", 0))
    return {
        "n": len(qids),
        "base_page_hit_at_k_count": base_page_hit,
        "page_hit_at_k_count": routed_page_hit,
        "base_doc_hit_at_k_count": base_doc_hit,
        "doc_hit_at_k_count": routed_doc_hit,
        "recovered": recovered,
        "lost": lost,
        "net_recovered": recovered - lost,
        "movement_counts": dict(sorted(movement_counts.items())),
        "selection_counts": dict(sorted(selection_counts.items())),
        "page_recall_at_k": {str(k): mean(values) for k, values in sorted(page_recall.items())},
        "doc_recall_at_k": {str(k): mean(values) for k, values in sorted(doc_recall.items())},
    }


def main() -> None:
    args = parse_args()
    if not args.candidate:
        raise ValueError("Provide at least one --candidate LABEL=PATH.")
    fold_count = max(2, int(args.folds))
    gold = {str(row["qid"]): row for row in read_jsonl(Path(args.gold))}
    base = load_prediction(Path(args.base_prediction))
    clusters = load_cluster_rows(Path(args.qid_clusters_json))
    candidates = {
        label: load_prediction(path)
        for label, path in (parse_labeled_path(raw) for raw in args.candidate)
    }
    qids = sorted(set(gold) & set(base) & set(clusters))

    routed: dict[str, dict[str, Any]] = {}
    selected: dict[str, str] = {}
    fold_reports: list[dict[str, Any]] = []
    selected_clusters_by_fold: dict[str, Any] = {}

    for fold in range(fold_count):
        heldout = [qid for qid in qids if fold_id(qid, fold_count) == fold]
        train = [qid for qid in qids if fold_id(qid, fold_count) != fold]
        selected_clusters: dict[str, dict[str, Any]] = {}
        for label, candidate in candidates.items():
            train_stats = cluster_stats_for_qids(
                qids=train,
                clusters=clusters,
                gold=gold,
                base=base,
                candidate=candidate,
                hit_k=int(args.hit_k),
            )
            selected_clusters[label] = {
                cluster: row
                for cluster, row in train_stats.items()
                if eligible_cluster(row, args)
            }

        heldout_selected: dict[str, str] = {}
        heldout_routed: dict[str, dict[str, Any]] = {}
        for qid in heldout:
            cluster = str(clusters[qid]["cluster"])
            best_label = "base"
            best_score = 0.0
            for label, cluster_rows in selected_clusters.items():
                row = cluster_rows.get(cluster)
                if row is None or qid not in candidates[label]:
                    continue
                score = float(row.get("page_delta_mean", 0.0))
                if score > best_score:
                    best_score = score
                    best_label = label
            heldout_selected[qid] = best_label
            heldout_routed[qid] = base[qid] if best_label == "base" else candidates[best_label][qid]
        routed.update(heldout_routed)
        selected.update(heldout_selected)

        fold_summary = evaluate_routed(
            qids=heldout,
            gold=gold,
            base=base,
            routed=heldout_routed,
            selected=heldout_selected,
            recall_ks=[int(k) for k in args.recall_ks],
            hit_k=int(args.hit_k),
        )
        fold_label = f"fold:{fold}"
        fold_reports.append({"fold": fold_label, "summary": fold_summary})
        selected_clusters_by_fold[fold_label] = {
            label: {
                cluster: {
                    key: value
                    for key, value in row.items()
                    if key != "qids"
                }
                for cluster, row in sorted(cluster_rows.items())
            }
            for label, cluster_rows in sorted(selected_clusters.items())
        }

    final_summary = evaluate_routed(
        qids=qids,
        gold=gold,
        base=base,
        routed=routed,
        selected=selected,
        recall_ks=[int(k) for k in args.recall_ks],
        hit_k=int(args.hit_k),
    )
    report = {
        "args": vars(args),
        "qid_count": len(qids),
        "final_summary": final_summary,
        "folds": fold_reports,
        "selected_clusters_by_fold": selected_clusters_by_fold,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved_json: {output_json}")
    print("final_summary:", json.dumps(final_summary, sort_keys=True))

    if args.output_routed_dir:
        out_dir = Path(args.output_routed_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_prediction(out_dir / "routed.prediction.json", routed)
        (out_dir / "routed.summary.json").write_text(
            json.dumps(final_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (out_dir / "selected.json").write_text(
            json.dumps(selected, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"saved_routed_prediction: {out_dir / 'routed.prediction.json'}")
        print(f"saved_routed_summary: {out_dir / 'routed.summary.json'}")
        print(f"saved_selected: {out_dir / 'selected.json'}")


if __name__ == "__main__":
    main()
