#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one routed prediction JSON by selecting among named prediction files "
            "according to qid-group text files. This is useful for oracle or learned "
            "route experiments where each qid should use a different specialist run."
        )
    )
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="Named prediction JSON as LABEL=path/to/prediction.json. Can be repeated.",
    )
    parser.add_argument(
        "--default-label",
        required=True,
        help="Prediction label to use for qids not covered by any route group.",
    )
    parser.add_argument(
        "--route",
        action="append",
        default=[],
        help=(
            "Route as GROUP=path/to/qids.txt:LABEL. Qids in the group are selected "
            "from LABEL. Can be repeated. Groups must not overlap unless "
            "--allow-overlap is set."
        ),
    )
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Allow later --route entries to override earlier ones for overlapping qids.",
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Fail if the selected prediction is missing a routed qid instead of falling back.",
    )
    parser.add_argument("--output-prediction-json", required=True)
    parser.add_argument("--output-summary-json", required=True)
    return parser.parse_args()


def parse_labeled_path(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected LABEL=path, got: {spec!r}")
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    path = Path(raw_path.strip())
    if not label or not str(path):
        raise ValueError(f"Invalid prediction spec: {spec!r}")
    return label, path


def parse_route(spec: str) -> tuple[str, Path, str]:
    if "=" not in spec or ":" not in spec:
        raise ValueError(f"Expected GROUP=qids.txt:LABEL, got: {spec!r}")
    group, rest = spec.split("=", 1)
    qids_path_raw, label = rest.rsplit(":", 1)
    group = group.strip()
    qids_path = Path(qids_path_raw.strip())
    label = label.strip()
    if not group or not str(qids_path) or not label:
        raise ValueError(f"Invalid route spec: {spec!r}")
    return group, qids_path, label


def load_qids(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    qids: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        qid = line.strip()
        if not qid or qid.startswith("#") or qid in seen:
            continue
        seen.add(qid)
        qids.append(qid)
    return qids


def load_prediction(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("predictions"), (dict, list)):
        payload = payload["predictions"]
    if isinstance(payload, dict):
        iterator = payload.items()
    elif isinstance(payload, list):
        iterator = enumerate(payload)
    else:
        raise TypeError(f"Prediction JSON must be object or list: {path}")

    out: dict[str, dict[str, Any]] = {}
    for raw_key, row in iterator:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("qid") or raw_key).strip()
        if not qid:
            continue
        if qid in out:
            raise ValueError(f"Duplicate qid after normalization: {qid} ({path})")
        out[qid] = row
    return out


def main() -> None:
    args = parse_args()

    prediction_paths = dict(parse_labeled_path(spec) for spec in args.prediction)
    if args.default_label not in prediction_paths:
        raise KeyError(f"--default-label is not among --prediction labels: {args.default_label}")
    route_specs = [parse_route(spec) for spec in args.route]
    unknown_route_labels = sorted({label for _, _, label in route_specs} - set(prediction_paths))
    if unknown_route_labels:
        raise KeyError(f"Route labels missing from --prediction labels: {unknown_route_labels}")

    predictions = {label: load_prediction(path) for label, path in prediction_paths.items()}
    default_pred = predictions[args.default_label]

    qid_to_route: dict[str, tuple[str, str]] = {}
    route_group_counts: Counter[str] = Counter()
    for group, qids_path, label in route_specs:
        for qid in load_qids(qids_path):
            if qid in qid_to_route and not args.allow_overlap:
                old_group, old_label = qid_to_route[qid]
                raise ValueError(
                    f"Qid {qid!r} appears in multiple route groups: "
                    f"{old_group}:{old_label} and {group}:{label}"
                )
            qid_to_route[qid] = (group, label)
            route_group_counts[group] += 1

    all_qids = sorted(set(default_pred) | set(qid_to_route))
    routed: dict[str, dict[str, Any]] = {}
    selected_label_counts: Counter[str] = Counter()
    selected_group_counts: Counter[str] = Counter()
    fallback_count = 0
    missing_default_qids: list[str] = []
    missing_selected_qids: list[str] = []

    for qid in all_qids:
        route_group, selected_label = qid_to_route.get(qid, ("__default__", args.default_label))
        selected_pred = predictions[selected_label]
        selected_row = selected_pred.get(qid)
        used_fallback = False
        if selected_row is None:
            missing_selected_qids.append(qid)
            if args.strict_missing:
                raise KeyError(f"Selected prediction {selected_label!r} missing qid {qid!r}")
            selected_row = default_pred.get(qid)
            selected_label = args.default_label
            route_group = "__fallback__"
            used_fallback = True
        if selected_row is None:
            missing_default_qids.append(qid)
            continue

        row = copy.deepcopy(selected_row)
        metadata = row.get("reranker_metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["qid_group_route"] = {
            "route_group": route_group,
            "selected_label": selected_label,
            "used_fallback": used_fallback,
        }
        row["reranker_metadata"] = metadata
        routed[qid] = row
        selected_label_counts[selected_label] += 1
        selected_group_counts[route_group] += 1
        if used_fallback:
            fallback_count += 1

    summary: dict[str, Any] = {
        "prediction_paths": {label: str(path) for label, path in prediction_paths.items()},
        "default_label": args.default_label,
        "route_specs": [
            {"group": group, "qids_path": str(path), "label": label}
            for group, path, label in route_specs
        ],
        "output_qid_count": len(routed),
        "input_default_qid_count": len(default_pred),
        "route_qid_count": len(qid_to_route),
        "route_group_counts": dict(route_group_counts),
        "selected_label_counts": dict(selected_label_counts),
        "selected_group_counts": dict(selected_group_counts),
        "fallback_count": fallback_count,
        "missing_selected_qid_count": len(missing_selected_qids),
        "missing_selected_qid_sample": missing_selected_qids[:20],
        "missing_default_qid_count": len(missing_default_qids),
        "missing_default_qid_sample": missing_default_qids[:20],
    }

    output_prediction = Path(args.output_prediction_json)
    output_summary = Path(args.output_summary_json)
    output_prediction.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_prediction.write_text(json.dumps(routed, separators=(",", ":")) + "\n", encoding="utf-8")
    output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved_prediction={output_prediction}")
    print(f"saved_summary={output_summary}")
    print(f"output_qid_count={len(routed)}")
    print(f"selected_label_counts={dict(selected_label_counts)}")
    print(f"fallback_count={fallback_count}")


if __name__ == "__main__":
    main()
