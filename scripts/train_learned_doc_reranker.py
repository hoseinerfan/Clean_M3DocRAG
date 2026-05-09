#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - runtime dependency check
    torch = None
    F = None

from scripts.rerank_target_docs_visual_aware import LEARNED_DOC_RERANKER_FEATURE_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a lightweight learned doc reranker from exported per-doc feature JSONL records."
        )
    )
    parser.add_argument("--doc-features-jsonl", required=True)
    parser.add_argument("--eval-doc-features-jsonl")
    parser.add_argument("--output-model-json", required=True)
    parser.add_argument("--output-summary-json")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-negatives-per-qid", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_grouped_rows(path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            grouped[str(row["qid"])].append(row)
    if not grouped:
        raise ValueError(f"No rows found in {path}")
    return grouped


def feature_tensor(rows: list[dict], means: torch.Tensor | None = None, stds: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vectors = []
    labels = []
    stage1_ranks = []
    for row in rows:
        feature_values = row["feature_values"]
        vectors.append([float(feature_values[name]) for name in LEARNED_DOC_RERANKER_FEATURE_NAMES])
        labels.append(1.0 if row.get("label_is_gold") else 0.0)
        stage1_ranks.append(float(row.get("stage1_base_doc_rank", 10**6)))
    x = torch.tensor(vectors, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    stage1 = torch.tensor(stage1_ranks, dtype=torch.float32)
    if means is None:
        means = x.mean(dim=0)
    if stds is None:
        stds = x.std(dim=0, unbiased=False)
    stds = torch.where(stds > 1e-6, stds, torch.ones_like(stds))
    return (x - means) / stds, y, stage1


def evaluate_grouped_rows(
    grouped_rows: dict[str, list[dict]],
    *,
    weights: torch.Tensor,
    bias: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor,
) -> dict:
    top1 = 0
    top4 = 0
    improved_vs_stage1_top4 = 0
    total = 0
    for rows in grouped_rows.values():
        x, y, stage1_ranks = feature_tensor(rows, means=means, stds=stds)
        scores = (x @ weights) + bias
        ranked_indices = sorted(
            range(len(rows)),
            key=lambda idx: (float(scores[idx]), -float(stage1_ranks[idx])),
            reverse=True,
        )
        gold_positions = [rank + 1 for rank, idx in enumerate(ranked_indices) if float(y[idx]) > 0.5]
        stage1_gold_ranks = [int(stage1_ranks[idx].item()) for idx in range(len(rows)) if float(y[idx]) > 0.5]
        if not gold_positions:
            continue
        total += 1
        first_gold_rank = gold_positions[0]
        if first_gold_rank == 1:
            top1 += 1
        if first_gold_rank <= 4:
            top4 += 1
        if stage1_gold_ranks and min(stage1_gold_ranks) > 4 and first_gold_rank <= 4:
            improved_vs_stage1_top4 += 1
    return {
        "num_qids": total,
        "top1_doc_count": top1,
        "top4_doc_count": top4,
        "top1_doc_rate": (top1 / total) if total else 0.0,
        "top4_doc_rate": (top4 / total) if total else 0.0,
        "rescued_into_top4_from_stage1_count": improved_vs_stage1_top4,
    }


def main() -> None:
    args = parse_args()
    if torch is None or F is None:
        raise ImportError(
            "train_learned_doc_reranker.py requires torch in the active environment."
        )
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_grouped = load_grouped_rows(Path(args.doc_features_jsonl))
    eval_grouped = (
        load_grouped_rows(Path(args.eval_doc_features_jsonl))
        if args.eval_doc_features_jsonl
        else train_grouped
    )

    all_train_rows = [row for rows in train_grouped.values() for row in rows]
    train_x, _train_y, _train_stage1 = feature_tensor(all_train_rows)
    feature_means = torch.tensor(
        [[float(row["feature_values"][name]) for name in LEARNED_DOC_RERANKER_FEATURE_NAMES] for row in all_train_rows],
        dtype=torch.float32,
    ).mean(dim=0)
    feature_stds = torch.tensor(
        [[float(row["feature_values"][name]) for name in LEARNED_DOC_RERANKER_FEATURE_NAMES] for row in all_train_rows],
        dtype=torch.float32,
    ).std(dim=0, unbiased=False)
    feature_stds = torch.where(feature_stds > 1e-6, feature_stds, torch.ones_like(feature_stds))

    weights = torch.nn.Parameter(torch.zeros(len(LEARNED_DOC_RERANKER_FEATURE_NAMES), dtype=torch.float32))
    bias = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
    optimizer = torch.optim.Adam([weights, bias], lr=args.learning_rate, weight_decay=args.weight_decay)

    qids = list(train_grouped.keys())
    epoch_history = []
    for epoch in range(args.epochs):
        random.shuffle(qids)
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, dtype=torch.float32)
        pair_count = 0

        for qid in qids:
            rows = train_grouped[qid]
            x, y, stage1_ranks = feature_tensor(rows, means=feature_means, stds=feature_stds)
            scores = (x @ weights) + bias

            positive_indices = [idx for idx, label in enumerate(y.tolist()) if label > 0.5]
            negative_indices = [idx for idx, label in enumerate(y.tolist()) if label <= 0.5]
            if not positive_indices or not negative_indices:
                continue
            negative_indices = sorted(
                negative_indices,
                key=lambda idx: float(stage1_ranks[idx]),
            )[: args.max_negatives_per_qid]
            if not negative_indices:
                continue

            pos_scores = scores[positive_indices]
            neg_scores = scores[negative_indices]
            pairwise_diffs = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
            total_loss = total_loss + F.softplus(-pairwise_diffs).mean()
            pair_count += 1

        if pair_count == 0:
            raise ValueError("No positive/negative training pairs were formed.")
        loss = total_loss / pair_count
        loss.backward()
        optimizer.step()
        epoch_history.append({"epoch": epoch + 1, "loss": float(loss.item())})

    train_metrics = evaluate_grouped_rows(
        train_grouped,
        weights=weights.detach(),
        bias=bias.detach(),
        means=feature_means,
        stds=feature_stds,
    )
    eval_metrics = evaluate_grouped_rows(
        eval_grouped,
        weights=weights.detach(),
        bias=bias.detach(),
        means=feature_means,
        stds=feature_stds,
    )

    model_payload = {
        "model_type": "linear_pairwise_doc_reranker",
        "feature_names": list(LEARNED_DOC_RERANKER_FEATURE_NAMES),
        "feature_means": [float(value) for value in feature_means.tolist()],
        "feature_stds": [float(value) for value in feature_stds.tolist()],
        "weights": [float(value) for value in weights.detach().tolist()],
        "bias": float(bias.detach().item()),
        "training_args": {
            "doc_features_jsonl": args.doc_features_jsonl,
            "eval_doc_features_jsonl": args.eval_doc_features_jsonl,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_negatives_per_qid": args.max_negatives_per_qid,
            "seed": args.seed,
        },
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
    }

    output_model_json = Path(args.output_model_json)
    output_model_json.parent.mkdir(parents=True, exist_ok=True)
    output_model_json.write_text(json.dumps(model_payload, indent=2) + "\n", encoding="utf-8")

    summary_payload = {
        "output_model_json": str(output_model_json),
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "final_loss": epoch_history[-1]["loss"] if epoch_history else None,
        "epoch_history_tail": epoch_history[-10:],
    }
    if args.output_summary_json:
        output_summary_json = Path(args.output_summary_json)
        output_summary_json.parent.mkdir(parents=True, exist_ok=True)
        output_summary_json.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
        print(f"saved_summary: {output_summary_json}")

    print(f"saved_model: {output_model_json}")
    print(f"train_top4_doc_count: {train_metrics['top4_doc_count']}")
    print(f"train_top4_doc_rate: {train_metrics['top4_doc_rate']:.4f}")
    print(f"eval_top4_doc_count: {eval_metrics['top4_doc_count']}")
    print(f"eval_top4_doc_rate: {eval_metrics['top4_doc_rate']:.4f}")


if __name__ == "__main__":
    main()
