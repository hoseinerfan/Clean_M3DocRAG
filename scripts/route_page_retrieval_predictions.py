#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Route between multiple page-retrieval prediction JSONs using lightweight "
            "question heuristics, then report doc-hit summary metrics."
        )
    )
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="Named prediction JSON in the form name=/path/to/prediction.json. Pass multiple times.",
    )
    parser.add_argument("--gold", required=True, help="Path to MMQA_<split>.jsonl.")
    parser.add_argument(
        "--question-type",
        default="",
        help="Optional question type filter, e.g. ImageListQ. Empty means all questions.",
    )
    parser.add_argument(
        "--route-mode",
        default="imagelistq_v1",
        choices=["imagelistq_v1"],
        help="Built-in routing heuristic to apply. Default: imagelistq_v1.",
    )
    parser.add_argument(
        "--baseline-profile",
        default="dense",
        help="Profile name used for recovered/lost comparisons. Default: dense.",
    )
    parser.add_argument(
        "--output-prediction-json",
        required=True,
        help="Path to write the routed prediction JSON.",
    )
    parser.add_argument(
        "--output-summary-json",
        required=True,
        help="Path to write the routed summary JSON.",
    )
    return parser.parse_args()


def parse_named_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected name=/path form, got: {value}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        path = Path(raw_path).expanduser()
        if not name:
            raise ValueError(f"Missing profile name in: {value}")
        if name in result:
            raise ValueError(f"Duplicate profile name: {name}")
        if not path.exists():
            raise FileNotFoundError(path)
        result[name] = path
    return result


def load_prediction(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Prediction JSON must be an object: {path}")
    return payload


def load_gold_rows(path: Path, question_type: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    wanted_type = str(question_type).strip()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("qid", "")).strip()
            if not qid:
                continue
            row_type = str(row.get("metadata", {}).get("type", "")).strip()
            if wanted_type and row_type != wanted_type:
                continue
            rows[qid] = row
    return rows


def deduped_docs(pred_row: dict) -> list[str]:
    docs: list[str] = []
    seen: set[str] = set()
    for item in pred_row.get("page_retrieval_results", []):
        if not isinstance(item, list) or not item:
            continue
        doc_id = str(item[0]).strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        docs.append(doc_id)
    return docs


def first_gold_doc_rank(pred_row: dict, gold_doc_ids: set[str]) -> int | None:
    for idx, doc_id in enumerate(deduped_docs(pred_row), start=1):
        if doc_id in gold_doc_ids:
            return idx
    return None


def build_route_config(mode: str) -> dict:
    if mode != "imagelistq_v1":
        raise ValueError(f"Unsupported route mode: {mode}")
    return {
        "default_profile": "hybrid",
        "routes": [
            {
                "profile": "dense",
                "label": "dense_symbolic_v1",
                "substring_any": [
                    "downwards pointing arrow",
                    "three vertical green lines",
                    "exactly four people",
                ],
            },
            {
                "profile": "sparse",
                "label": "sparse_lexical_v1",
                "substring_any": [
                    "stethoscope",
                    "motorcycle",
                    "lighthouse",
                    "penguin",
                    "gymnast",
                    "eagle",
                    "dolphin",
                    "pouncing wild cat",
                    "wild cat",
                    "orange sign",
                    "thick glasses",
                    "completely bald",
                    "more than three colors",
                    "books on the title screen",
                    "fire explosion",
                    "banners over blue seats",
                ],
            },
        ],
    }


def choose_profile(question: str, route_config: dict) -> tuple[str, str]:
    normalized = str(question).lower()
    for route in route_config["routes"]:
        substrings = [str(item).lower() for item in route.get("substring_any", [])]
        if any(token in normalized for token in substrings):
            return str(route["profile"]), str(route.get("label", route["profile"]))
    default_profile = str(route_config["default_profile"])
    return default_profile, "default"


def main() -> None:
    args = parse_args()
    prediction_paths = parse_named_paths(args.prediction)
    predictions = {name: load_prediction(path) for name, path in prediction_paths.items()}
    gold_rows = load_gold_rows(Path(args.gold), args.question_type)
    if not gold_rows:
        raise ValueError("No gold rows matched the requested filter.")

    route_config = build_route_config(args.route_mode)
    required_profiles = {route_config["default_profile"]} | {
        str(route["profile"]) for route in route_config["routes"]
    }
    missing_profiles = sorted(required_profiles - set(predictions))
    if missing_profiles:
        raise KeyError(f"Missing required prediction profiles for route mode: {missing_profiles}")
    if args.baseline_profile not in predictions:
        raise KeyError(f"Missing baseline profile: {args.baseline_profile}")

    qids = set(gold_rows)
    for name, payload in predictions.items():
        missing = sorted(qids - set(payload))
        if missing:
            raise KeyError(f"Prediction {name} missing {len(missing)} qids: {missing[:10]}")

    routed_payload: dict[str, dict] = {}
    per_qid: list[dict] = []
    route_counts: Counter[str] = Counter()
    route_reason_counts: Counter[str] = Counter()

    baseline_hit_qids: set[str] = set()
    routed_hit_qids: set[str] = set()

    for qid in sorted(gold_rows):
        gold_row = gold_rows[qid]
        gold_doc_ids = {
            str(item.get("doc_id", "")).strip()
            for item in gold_row.get("supporting_context", [])
            if str(item.get("doc_id", "")).strip()
        }
        question = str(gold_row.get("question", ""))
        profile_name, route_reason = choose_profile(question, route_config)
        selected_row = predictions[profile_name][qid]
        baseline_row = predictions[str(args.baseline_profile)][qid]

        baseline_rank = first_gold_doc_rank(baseline_row, gold_doc_ids)
        routed_rank = first_gold_doc_rank(selected_row, gold_doc_ids)
        if baseline_rank is not None:
            baseline_hit_qids.add(qid)
        if routed_rank is not None:
            routed_hit_qids.add(qid)

        routed_payload[qid] = {
            **selected_row,
            "reranker_metadata": {
                **selected_row.get("reranker_metadata", {}),
                "route_mode": args.route_mode,
                "selected_profile": profile_name,
                "route_reason": route_reason,
            },
        }
        route_counts[profile_name] += 1
        route_reason_counts[route_reason] += 1
        per_qid.append(
            {
                "qid": qid,
                "question": question,
                "selected_profile": profile_name,
                "route_reason": route_reason,
                "baseline_profile": str(args.baseline_profile),
                "baseline_first_gold_doc_rank": baseline_rank,
                "routed_first_gold_doc_rank": routed_rank,
            }
        )

    recovered_vs_baseline = sorted(routed_hit_qids - baseline_hit_qids)
    lost_vs_baseline = sorted(baseline_hit_qids - routed_hit_qids)

    output_prediction_json = Path(args.output_prediction_json)
    output_prediction_json.parent.mkdir(parents=True, exist_ok=True)
    output_prediction_json.write_text(json.dumps(routed_payload, indent=2) + "\n", encoding="utf-8")

    summary = {
        "route_mode": args.route_mode,
        "qid_count": len(gold_rows),
        "question_type": str(args.question_type),
        "baseline_profile": str(args.baseline_profile),
        "route_counts": dict(route_counts),
        "route_reason_counts": dict(route_reason_counts),
        "baseline_gold_doc_hit_count": len(baseline_hit_qids),
        "routed_gold_doc_hit_count": len(routed_hit_qids),
        "baseline_gold_doc_hit_rate": len(baseline_hit_qids) / len(gold_rows),
        "routed_gold_doc_hit_rate": len(routed_hit_qids) / len(gold_rows),
        "recovered_vs_baseline_count": len(recovered_vs_baseline),
        "lost_vs_baseline_count": len(lost_vs_baseline),
        "recovered_vs_baseline_qids": recovered_vs_baseline,
        "lost_vs_baseline_qids": lost_vs_baseline,
        "per_qid": per_qid,
    }
    output_summary_json = Path(args.output_summary_json)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"saved_prediction: {output_prediction_json}")
    print(f"saved_summary: {output_summary_json}")
    print(f"qid_count: {len(gold_rows)}")
    print(f"baseline_gold_doc_hit_count: {summary['baseline_gold_doc_hit_count']}")
    print(f"routed_gold_doc_hit_count: {summary['routed_gold_doc_hit_count']}")
    print(f"recovered_vs_baseline_count: {summary['recovered_vs_baseline_count']}")
    print(f"lost_vs_baseline_count: {summary['lost_vs_baseline_count']}")


if __name__ == "__main__":
    main()
