#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

import train_content_aware_pseudo_page_reranker as ca
import zero_shot_content_aware_page_reranker as zsc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a trained M3DocVQA/MMQA content-aware page promotion model to a new "
            "dataset without using that dataset's labels for training."
        )
    )
    parser.add_argument("--model-json", required=True)
    parser.add_argument("--base-pred", required=True)
    parser.add_argument("--page-text-jsonl", required=True)
    parser.add_argument("--gold", default="", help="Optional gold JSONL for evaluation only.")
    parser.add_argument("--source", action="append", default=[], help="Optional LABEL=prediction.json source.")
    parser.add_argument("--candidate-top-k", type=int, default=0, help="Defaults to the model setting.")
    parser.add_argument("--inference-mode", default="", choices=["", "blend_rerank", "full_rerank", "safe_promote", "doc_head_blend", "doc_slot_blend"])
    parser.add_argument("--blend-alpha", type=float, default=-1.0, help="Defaults to the model setting.")
    parser.add_argument("--anchor-top-k", type=int, default=4)
    parser.add_argument("--promotion-rank-min", type=int, default=5)
    parser.add_argument("--promotion-rank-max", type=int, default=200)
    parser.add_argument("--max-promotions-per-qid", type=int, default=2)
    parser.add_argument("--promotion-margin", type=float, default=0.05)
    parser.add_argument("--recall-k", type=int, nargs="+", default=zsc.DEFAULT_RECALL_KS)
    parser.add_argument("--output-prediction-json", required=True)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument("--output-table-md", default="")
    parser.add_argument("--output-prior-jsonl", default="")
    return parser.parse_args()


def load_model(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    feature_names = payload.get("feature_names")
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError("Model feature_names must be a non-empty list.")
    unknown = [name for name in feature_names if name not in ca.FEATURE_NAMES]
    if unknown:
        raise ValueError(f"Model contains unknown feature_names: {unknown}")
    return payload


def resolved_args(args: argparse.Namespace, model: dict[str, Any]) -> argparse.Namespace:
    model_args = model.get("args", {}) if isinstance(model.get("args"), dict) else {}
    if int(args.candidate_top_k) <= 0:
        args.candidate_top_k = int(model_args.get("candidate_top_k", 1000))
    if not args.inference_mode:
        args.inference_mode = str(model_args.get("inference_mode", "blend_rerank"))
    if float(args.blend_alpha) < 0.0:
        args.blend_alpha = float(model_args.get("blend_alpha", 0.30))
    adaptive_config = model.get("adaptive_alpha_config")
    if not isinstance(adaptive_config, dict):
        adaptive_config = model_args.get("adaptive_alpha_config")
    if not isinstance(adaptive_config, dict):
        train_meta = model.get("train_metadata", {})
        if isinstance(train_meta, dict):
            adaptive_config = train_meta.get("adaptive_alpha_config")
    if not isinstance(adaptive_config, dict):
        adaptive_config = None
    args.adaptive_alpha_config = adaptive_config
    mode = str(adaptive_config.get("mode", "")) if adaptive_config else ""
    args.query_adaptive_alpha = bool(model_args.get("query_adaptive_alpha", False) or mode == "confidence_bins")
    args.learned_query_alpha = bool(model_args.get("learned_query_alpha", False) or mode == "learned_query_regressor")
    args.learned_alpha_action = bool(model_args.get("learned_alpha_action", False) or mode == "learned_alpha_action")
    args.learned_alpha_utility_gate = bool(
        model_args.get("learned_alpha_utility_gate", False) or mode == "learned_alpha_utility_gate"
    )
    args.base_aware_alpha_utility_gate = bool(
        model_args.get("base_aware_alpha_utility_gate", False) or mode == "base_aware_alpha_utility_gate"
    )
    return args


def apply_model(
    *,
    model: dict[str, Any],
    base_pred: dict[str, dict[str, Any]],
    gold: dict[str, dict[str, Any]],
    page_features: dict[str, dict[str, Any]],
    source_maps_by_label: dict[str, dict[str, dict[str, float]]],
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    mean = np.asarray(model["mean"], dtype=np.float32)
    std = np.asarray(model["std"], dtype=np.float32)
    weights = np.asarray(model["weights"], dtype=np.float32)
    bias = float(model["bias"])
    feature_names = list(model.get("feature_names") or ca.FEATURE_NAMES)

    output: dict[str, dict[str, Any]] = {}
    prior_rows: list[dict[str, Any]] = []
    total_records = 0
    scored_records = 0

    for qid, base_row in base_pred.items():
        records = ca.ranked_page_records(base_row, int(args.candidate_top_k))
        if not records:
            output[qid] = dict(base_row)
            continue
        gold_row = gold.get(qid, {"qid": qid, "question": base_row.get("question", "")})
        if not str(gold_row.get("question", "")).strip() and str(base_row.get("question", "")).strip():
            gold_row = dict(gold_row)
            gold_row["question"] = base_row.get("question", "")

        for record in records:
            total_records += 1
            if str(record["uid"]) in page_features:
                scored_records += 1

        scored_records_qid = ca.score_records(
            qid=qid,
            gold_row=gold_row,
            records=records,
            page_features=page_features,
            source_maps_by_label=source_maps_by_label,
            mean=mean,
            std=std,
            weights=weights,
            bias=bias,
            feature_names=feature_names,
        )
        apply_args = args
        query_alpha = float(args.blend_alpha)
        query_alpha_info: dict[str, Any] = {}
        query_alpha_confidence: float | None = None
        adaptive_config = getattr(args, "adaptive_alpha_config", None)
        if bool(getattr(args, "learned_alpha_utility_gate", False)):
            query_alpha_confidence = ca.query_confidence_score(scored_records_qid, source_maps_by_label, qid)
            query_alpha, query_alpha_info = ca.predict_learned_alpha_utility_gate(
                records=scored_records_qid,
                source_maps_by_label=source_maps_by_label,
                qid=qid,
                adaptive_config=adaptive_config,
                fallback_alpha=float(args.blend_alpha),
            )
            apply_args = copy.copy(args)
            apply_args.blend_alpha = float(query_alpha)
        elif bool(getattr(args, "base_aware_alpha_utility_gate", False)):
            query_alpha_confidence = ca.query_confidence_score(scored_records_qid, source_maps_by_label, qid)
            query_alpha, query_alpha_info = ca.predict_base_aware_alpha_utility_gate(
                records=scored_records_qid,
                source_maps_by_label=source_maps_by_label,
                qid=qid,
                adaptive_config=adaptive_config,
                fallback_alpha=float(args.blend_alpha),
            )
            apply_args = copy.copy(args)
            apply_args.blend_alpha = float(query_alpha)
        elif bool(getattr(args, "learned_alpha_action", False)):
            query_alpha_confidence = ca.query_confidence_score(scored_records_qid, source_maps_by_label, qid)
            query_alpha, query_alpha_info = ca.predict_learned_alpha_action(
                records=scored_records_qid,
                source_maps_by_label=source_maps_by_label,
                qid=qid,
                adaptive_config=adaptive_config,
                fallback_alpha=float(args.blend_alpha),
            )
            apply_args = copy.copy(args)
            apply_args.blend_alpha = float(query_alpha)
        elif bool(getattr(args, "learned_query_alpha", False)):
            query_alpha_confidence = ca.query_confidence_score(scored_records_qid, source_maps_by_label, qid)
            query_alpha, query_alpha_info = ca.predict_learned_query_alpha(
                records=scored_records_qid,
                source_maps_by_label=source_maps_by_label,
                qid=qid,
                adaptive_config=adaptive_config,
                fallback_alpha=float(args.blend_alpha),
            )
            apply_args = copy.copy(args)
            apply_args.blend_alpha = float(query_alpha)
        elif bool(getattr(args, "query_adaptive_alpha", False)):
            query_alpha_confidence = ca.query_confidence_score(scored_records_qid, source_maps_by_label, qid)
            query_alpha, query_alpha_bin = ca.alpha_for_query_confidence(
                query_alpha_confidence,
                adaptive_config,
                float(args.blend_alpha),
            )
            query_alpha_info = {"mode": "confidence_bins", "query_alpha_bin": query_alpha_bin}
            apply_args = copy.copy(args)
            apply_args.blend_alpha = float(query_alpha)

        reranked = ca.rerank_records(scored_records_qid, apply_args)
        reranked_uids = {str(row["uid"]) for row in reranked}
        raw_by_uid = {str(row["uid"]): row["raw"] for row in records}
        output_rows = [raw_by_uid[str(row["uid"])] for row in reranked if str(row["uid"]) in raw_by_uid]
        output_rows.extend(
            raw_by_uid[str(row["uid"])]
            for row in records
            if str(row["uid"]) not in reranked_uids and str(row["uid"]) in raw_by_uid
        )

        out_row = dict(base_row)
        if "page_retrieval_results" in out_row:
            out_row["page_retrieval_results"] = output_rows
        elif "retrieval_results" in out_row:
            out_row["retrieval_results"] = output_rows
        else:
            out_row["page_retrieval_results"] = output_rows
        out_row["trained_content_aware_transfer_metadata"] = {
            "model_json": args.model_json,
            "inference_mode": args.inference_mode,
            "blend_alpha": float(query_alpha),
            "global_blend_alpha": float(args.blend_alpha),
            "query_alpha_mode": query_alpha_info.get("mode"),
            "query_alpha_confidence": query_alpha_confidence,
            "query_alpha_info": query_alpha_info,
            "base_aware_alpha_utility_gate": bool(getattr(args, "base_aware_alpha_utility_gate", False)),
            "candidate_top_k": int(args.candidate_top_k),
        }
        output[qid] = out_row

        for row in scored_records_qid[:50]:
            prior_rows.append(
                {
                    "qid": qid,
                    "page_uid": row["uid"],
                    "doc_id": row["doc_id"],
                    "page_idx": int(row["page_idx"]),
                    "base_rank": int(row["base_rank"]),
                    "learned_score": float(row.get("learned_score", 0.0)),
                    "rerank_score": float(row.get("rerank_score", row.get("learned_score", 0.0))),
                    "selected_blend_alpha": float(query_alpha),
                }
            )

    return output, prior_rows, {
        "total_candidate_records": int(total_records),
        "candidate_records_with_page_text": int(scored_records),
        "candidate_page_text_coverage": zsc.safe_div(scored_records, total_records),
    }


def main() -> None:
    args = parse_args()
    model = load_model(Path(args.model_json))
    model_feature_names = list(model.get("feature_names") or ca.FEATURE_NAMES)
    args = resolved_args(args, model)

    base_pred = ca.load_prediction(Path(args.base_pred))
    gold = zsc.load_gold(Path(args.gold)) if args.gold else {}
    page_features = ca.load_page_features(Path(args.page_text_jsonl))
    source_maps_by_label = {
        label: ca.source_maps(ca.load_prediction(path), int(args.candidate_top_k))
        for label, path in map(ca.parse_labeled_path, args.source)
    }

    output_pred, prior_rows, metadata = apply_model(
        model=model,
        base_pred=base_pred,
        gold=gold,
        page_features=page_features,
        source_maps_by_label=source_maps_by_label,
        args=args,
    )

    out_pred = Path(args.output_prediction_json)
    out_pred.parent.mkdir(parents=True, exist_ok=True)
    out_pred.write_text(json.dumps(output_pred) + "\n", encoding="utf-8")

    metrics = []
    if gold:
        metrics = [
            zsc.evaluate_run("base", base_pred, gold, list(args.recall_k)),
            zsc.evaluate_run("trained_content_aware_transfer", output_pred, gold, list(args.recall_k)),
        ]
    summary = {
        "model_json": args.model_json,
        "feature_names": model_feature_names,
        "feature_set": model.get("feature_set", ""),
        "base_pred": args.base_pred,
        "page_text_jsonl": args.page_text_jsonl,
        "source_count": len(source_maps_by_label),
        "candidate_top_k": int(args.candidate_top_k),
        "inference_mode": args.inference_mode,
        "blend_alpha": float(args.blend_alpha),
        "adaptive_alpha_config": getattr(args, "adaptive_alpha_config", None),
        "metrics": metrics,
        "movement_vs_base": zsc.movement_vs_base(base_pred, output_pred, gold) if gold else {},
        **metadata,
    }
    out_summary = Path(args.output_summary_json)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.output_table_md and metrics:
        zsc.write_table(Path(args.output_table_md), metrics, list(args.recall_k))
    if args.output_prior_jsonl:
        out_prior = Path(args.output_prior_jsonl)
        out_prior.parent.mkdir(parents=True, exist_ok=True)
        with out_prior.open("w", encoding="utf-8") as handle:
            for row in prior_rows:
                handle.write(json.dumps(row) + "\n")

    print(f"saved_prediction={out_pred}")
    print(f"saved_summary={out_summary}")
    if args.output_table_md and metrics:
        print(f"saved_table={args.output_table_md}")
    if args.output_prior_jsonl:
        print(f"saved_prior={args.output_prior_jsonl}")
    print(f"candidate_page_text_coverage={metadata['candidate_page_text_coverage']:.6f}")
    if metrics:
        for row in metrics:
            print(row)
        print(f"movement_vs_base={summary['movement_vs_base']}")


if __name__ == "__main__":
    main()
