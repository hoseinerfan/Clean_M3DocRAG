#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import sys
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

from scripts.rerank_target_docs_visual_aware import (
    LEARNED_TOKEN_SELECTOR_FEATURE_NAMES,
    QUERY_TOKEN_FILTER_CHOICES,
    build_page_id_metadata,
    build_page_token_classes,
    compute_exact_token_winner_counts,
    compute_token_selector_feature_matrix,
    load_doc_embeddings_for_doc_ids,
    load_patch_axis_classes_for_pages,
    make_query_score_mask,
    resolve_model_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a lightweight learned page-token selector from exact MaxSim winner tokens."
        )
    )
    parser.add_argument("--qid-jsonl", required=True)
    parser.add_argument("--qid-field", default="qid")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--baseline-pred", required=True)
    parser.add_argument("--from-baseline-top-pages", type=int, default=50)
    parser.add_argument("--embedding-name", default="colpali-v1.2_m3-docvqa_dev")
    parser.add_argument(
        "--query-token-filter",
        default="full",
        choices=QUERY_TOKEN_FILTER_CHOICES,
    )
    parser.add_argument("--approx-base-page-token-topk", type=int, default=256)
    parser.add_argument(
        "--approx-base-page-token-coarse-dtype",
        default="fp32",
        choices=("fp32", "bf16", "fp16"),
    )
    parser.add_argument(
        "--nonspatial-token-position",
        default="suffix",
        choices=["prefix", "suffix"],
    )
    parser.add_argument("--retrieval-model-name-or-path", default="colpaligemma-3b-pt-448-base")
    parser.add_argument("--retrieval-adapter-model-name-or-path", default="colpali-v1.2")
    parser.add_argument("--splice-patch-labels-jsonl", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-negatives-per-page", type=int, default=128)
    parser.add_argument("--max-qids", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-model-json", required=True)
    parser.add_argument("--output-summary-json")
    return parser.parse_args()


def load_qids(path: Path, qid_field: str, max_qids: int) -> list[str]:
    qids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get(qid_field, "")).strip()
            if not qid:
                continue
            qids.append(qid)
            if max_qids > 0 and len(qids) >= max_qids:
                break
    if not qids:
        raise ValueError(f"No qids found in {path} using field {qid_field!r}")
    return qids


def load_gold_rows(path: Path, qids: set[str]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            qid = str(row.get("qid", "")).strip()
            if qid in qids:
                rows[qid] = row
    missing = sorted(qids - set(rows))
    if missing:
        raise KeyError(f"Missing {len(missing)} qids in gold file: {missing[:10]}")
    return rows


def load_baseline_payload(path: Path, qids: set[str]) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = sorted(qids - set(payload))
    if missing:
        raise KeyError(f"Missing {len(missing)} qids in baseline prediction: {missing[:10]}")
    return {qid: payload[qid] for qid in qids}


def build_baseline_pool(
    rows: list[list[object]],
    top_pages: int,
) -> tuple[list[str], list[str]]:
    baseline_page_uids: list[str] = []
    candidate_doc_ids: list[str] = []
    seen_docs: set[str] = set()

    for row_rank, row in enumerate(rows, start=1):
        if row_rank > top_pages:
            break
        doc_id = str(row[0]).strip()
        page_idx = int(row[1])
        page_uid = f"{doc_id}_page{page_idx}"
        baseline_page_uids.append(page_uid)
        if doc_id not in seen_docs:
            seen_docs.add(doc_id)
            candidate_doc_ids.append(doc_id)
    return candidate_doc_ids, baseline_page_uids


def evaluate_selector_records(
    records: list[dict],
    *,
    weights: torch.Tensor,
    bias: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor,
    topk: int,
    device: torch.device,
) -> dict:
    baseline_weighted_hits = 0.0
    learned_weighted_hits = 0.0
    total_positive_wins = 0.0
    improved_pages = 0
    worsened_pages = 0
    tied_pages = 0

    coarse_idx = LEARNED_TOKEN_SELECTOR_FEATURE_NAMES.index("coarse_query_mean_score")

    with torch.no_grad():
        for record in records:
            features = record["features"].to(device=device, dtype=torch.float32)
            winner_counts = record["winner_counts"].to(device=device, dtype=torch.float32)
            total_wins = float(winner_counts.sum().item())
            if total_wins <= 0.0:
                continue
            k = min(topk, int(features.shape[0]))
            standardized = (features - means) / stds
            learned_scores = (standardized @ weights) + bias
            learned_indices = torch.topk(learned_scores, k=k, dim=0).indices
            baseline_indices = torch.topk(features[:, coarse_idx], k=k, dim=0).indices
            learned_hits = float(winner_counts[learned_indices].sum().item())
            baseline_hits = float(winner_counts[baseline_indices].sum().item())

            learned_weighted_hits += learned_hits
            baseline_weighted_hits += baseline_hits
            total_positive_wins += total_wins

            if learned_hits > baseline_hits:
                improved_pages += 1
            elif learned_hits < baseline_hits:
                worsened_pages += 1
            else:
                tied_pages += 1

    denom = total_positive_wins if total_positive_wins > 0.0 else 1.0
    return {
        "num_pages": len(records),
        "positive_win_total": total_positive_wins,
        "baseline_weighted_positive_recall_at_k": baseline_weighted_hits / denom,
        "learned_weighted_positive_recall_at_k": learned_weighted_hits / denom,
        "delta_weighted_positive_recall_at_k": (learned_weighted_hits - baseline_weighted_hits) / denom,
        "improved_pages": improved_pages,
        "worsened_pages": worsened_pages,
        "tied_pages": tied_pages,
    }


def main() -> None:
    args = parse_args()
    if torch is None or F is None:
        raise ImportError(
            "train_learned_token_selector.py requires torch in the active environment."
        )

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from m3docrag.retrieval import ColPaliRetrievalModel

    qids = load_qids(Path(args.qid_jsonl), args.qid_field, args.max_qids)
    gold_rows = load_gold_rows(Path(args.gold), set(qids))
    baseline_payload = load_baseline_payload(Path(args.baseline_pred), set(qids))

    retrieval_model = ColPaliRetrievalModel(
        backbone_name_or_path=resolve_model_path(args.retrieval_model_name_or_path),
        adapter_name_or_path=resolve_model_path(args.retrieval_adapter_model_name_or_path),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_sum = torch.zeros(len(LEARNED_TOKEN_SELECTOR_FEATURE_NAMES), dtype=torch.float64)
    feature_sumsq = torch.zeros(len(LEARNED_TOKEN_SELECTOR_FEATURE_NAMES), dtype=torch.float64)
    total_token_count = 0
    total_positive_win_count = 0.0
    records: list[dict] = []

    with torch.inference_mode():
        for idx, qid in enumerate(qids, start=1):
            gold_row = gold_rows[qid]
            query_text = str(gold_row["question"])
            baseline_rows = baseline_payload[qid].get("page_retrieval_results", [])
            candidate_doc_ids, baseline_page_uids = build_baseline_pool(
                baseline_rows,
                args.from_baseline_top_pages,
            )
            if not candidate_doc_ids or not baseline_page_uids:
                continue

            docid2embs = load_doc_embeddings_for_doc_ids(candidate_doc_ids, args.embedding_name)
            page_specs, page_meta = build_page_id_metadata(
                docid2embs=docid2embs,
                explicit_page_uids=set(baseline_page_uids),
                nonspatial_token_position=args.nonspatial_token_position,
            )
            patch_axis_classes_by_uid = load_patch_axis_classes_for_pages(
                labels_jsonl=args.splice_patch_labels_jsonl,
                page_meta=page_meta,
            )
            page_token_classes_by_uid = {
                page_uid: build_page_token_classes(
                    page_meta=page_meta[page_uid],
                    patch_axis_classes=patch_axis_classes,
                )
                for page_uid, patch_axis_classes in patch_axis_classes_by_uid.items()
            }

            query_meta = retrieval_model.encode_query_with_metadata(
                query=query_text,
                to_cpu=True,
                query_token_filter=args.query_token_filter,
            )
            query_emb = query_meta["embeddings"].float().to(device=device, dtype=torch.float32)
            query_raw_tokens = query_meta.get("raw_tokens", [])
            query_score_mask = make_query_score_mask(
                query_raw_tokens=query_raw_tokens,
                ignore_pad_scores_in_final_ranking=False,
            )

            page_spec_set = set(page_specs)
            for page_uid in baseline_page_uids:
                doc_id, page_suffix = page_uid.rsplit("_page", 1)
                page_idx = int(page_suffix)
                if (doc_id, page_idx) not in page_spec_set:
                    continue
                page_emb = docid2embs[doc_id][page_idx].view(
                    -1,
                    docid2embs[doc_id][page_idx].shape[-1],
                ).to(device=device, dtype=torch.float32)
                features = compute_token_selector_feature_matrix(
                    page_emb=page_emb,
                    query_emb=query_emb,
                    query_score_mask=query_score_mask,
                    page_meta=page_meta[page_uid],
                    page_token_classes=page_token_classes_by_uid[page_uid],
                    coarse_score_dtype=args.approx_base_page_token_coarse_dtype,
                ).cpu()
                winner_counts = compute_exact_token_winner_counts(
                    page_emb=page_emb,
                    query_emb=query_emb,
                    query_score_mask=query_score_mask,
                ).cpu().to(dtype=torch.float32)
                if int(winner_counts.shape[0]) != int(features.shape[0]):
                    raise ValueError(
                        f"Feature/token count mismatch for {page_uid}: "
                        f"{features.shape[0]} vs {winner_counts.shape[0]}"
                    )
                if float(winner_counts.sum().item()) <= 0.0:
                    continue

                feature_sum += features.to(dtype=torch.float64).sum(dim=0)
                feature_sumsq += (features.to(dtype=torch.float64) ** 2).sum(dim=0)
                total_token_count += int(features.shape[0])
                total_positive_win_count += float(winner_counts.sum().item())
                records.append(
                    {
                        "qid": qid,
                        "page_uid": page_uid,
                        "features": features,
                        "winner_counts": winner_counts,
                    }
                )

            print(f"[{idx}/{len(qids)}] {qid} pages={len(baseline_page_uids)} collected={len(records)}")

    if not records or total_token_count <= 0:
        raise ValueError("No token-selector training records were collected.")

    feature_means = (feature_sum / total_token_count).to(dtype=torch.float32)
    feature_vars = (feature_sumsq / total_token_count).to(dtype=torch.float32) - (feature_means ** 2)
    feature_vars = torch.clamp(feature_vars, min=1e-6)
    feature_stds = torch.sqrt(feature_vars)

    means_device = feature_means.to(device=device, dtype=torch.float32)
    stds_device = feature_stds.to(device=device, dtype=torch.float32)
    weights = torch.nn.Parameter(
        torch.zeros(len(LEARNED_TOKEN_SELECTOR_FEATURE_NAMES), dtype=torch.float32, device=device)
    )
    bias = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(
        [weights, bias],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    epoch_history: list[dict] = []
    for epoch in range(args.epochs):
        random.shuffle(records)
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        page_count = 0

        for record in records:
            features = record["features"].to(device=device, dtype=torch.float32)
            winner_counts = record["winner_counts"].to(device=device, dtype=torch.float32)
            positive_indices = torch.nonzero(winner_counts > 0.0, as_tuple=False).squeeze(-1)
            negative_indices = torch.nonzero(winner_counts <= 0.0, as_tuple=False).squeeze(-1)
            if int(positive_indices.numel()) == 0 or int(negative_indices.numel()) == 0:
                continue
            if (
                args.max_negatives_per_page > 0
                and int(negative_indices.numel()) > args.max_negatives_per_page
            ):
                sampled = torch.randperm(int(negative_indices.numel()), device=device)[
                    : args.max_negatives_per_page
                ]
                negative_indices = negative_indices[sampled]

            standardized = (features - means_device) / stds_device
            scores = (standardized @ weights) + bias
            pos_scores = scores[positive_indices]
            neg_scores = scores[negative_indices]
            pos_weights = winner_counts[positive_indices]
            pos_weights = pos_weights / pos_weights.sum().clamp_min(1e-6)
            pairwise_diffs = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
            page_loss = (F.softplus(-pairwise_diffs) * pos_weights.unsqueeze(1)).mean()
            total_loss = total_loss + page_loss
            page_count += 1

        if page_count == 0:
            raise ValueError("No positive/negative page-token training pairs were formed.")
        loss = total_loss / page_count
        loss.backward()
        optimizer.step()
        epoch_history.append({"epoch": epoch + 1, "loss": float(loss.item())})

    train_metrics = evaluate_selector_records(
        records,
        weights=weights.detach(),
        bias=bias.detach(),
        means=means_device,
        stds=stds_device,
        topk=args.approx_base_page_token_topk,
        device=device,
    )

    model_payload = {
        "model_type": "linear_token_selector",
        "feature_names": list(LEARNED_TOKEN_SELECTOR_FEATURE_NAMES),
        "feature_means": [float(value) for value in feature_means.tolist()],
        "feature_stds": [float(value) for value in feature_stds.tolist()],
        "weights": [float(value) for value in weights.detach().cpu().tolist()],
        "bias": float(bias.detach().cpu().item()),
        "training_args": {
            "qid_jsonl": args.qid_jsonl,
            "gold": args.gold,
            "baseline_pred": args.baseline_pred,
            "from_baseline_top_pages": args.from_baseline_top_pages,
            "embedding_name": args.embedding_name,
            "query_token_filter": args.query_token_filter,
            "approx_base_page_token_topk": args.approx_base_page_token_topk,
            "approx_base_page_token_coarse_dtype": args.approx_base_page_token_coarse_dtype,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_negatives_per_page": args.max_negatives_per_page,
            "seed": args.seed,
        },
        "train_metrics": train_metrics,
    }

    output_model_json = Path(args.output_model_json)
    output_model_json.parent.mkdir(parents=True, exist_ok=True)
    output_model_json.write_text(json.dumps(model_payload, indent=2) + "\n", encoding="utf-8")

    summary_payload = {
        "output_model_json": str(output_model_json),
        "num_qids": len(qids),
        "num_pages": len(records),
        "total_token_count": total_token_count,
        "total_positive_win_count": total_positive_win_count,
        "train_metrics": train_metrics,
        "final_loss": epoch_history[-1]["loss"] if epoch_history else None,
        "epoch_history_tail": epoch_history[-10:],
    }
    if args.output_summary_json:
        output_summary_json = Path(args.output_summary_json)
        output_summary_json.parent.mkdir(parents=True, exist_ok=True)
        output_summary_json.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
        print(f"saved_summary: {output_summary_json}")

    print(f"saved_model: {output_model_json}")
    print(f"num_pages: {len(records)}")
    print(
        "baseline_weighted_positive_recall_at_k:",
        f"{train_metrics['baseline_weighted_positive_recall_at_k']:.4f}",
    )
    print(
        "learned_weighted_positive_recall_at_k:",
        f"{train_metrics['learned_weighted_positive_recall_at_k']:.4f}",
    )
    print(
        "delta_weighted_positive_recall_at_k:",
        f"{train_metrics['delta_weighted_positive_recall_at_k']:+.4f}",
    )


if __name__ == "__main__":
    main()
