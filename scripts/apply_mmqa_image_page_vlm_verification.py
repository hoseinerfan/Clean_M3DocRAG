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
            "Attach image-page VLM verification decisions to evidence-unit-aware pseudo labels, "
            "or create a verified-only training label set."
        )
    )
    parser.add_argument("--labels-jsonl", required=True)
    parser.add_argument("--augmented-gold-jsonl", required=True)
    parser.add_argument("--verification-jsonl", required=True)
    parser.add_argument("--policy", choices=("annotate", "verified_only"), default="annotate")
    parser.add_argument("--output-labels-jsonl", required=True)
    parser.add_argument("--output-augmented-gold-jsonl", required=True)
    parser.add_argument("--output-summary-json", required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def decision_tier(decision: str) -> str:
    return {
        "verified": "direct_visual_verified",
        "rejected": "visual_proxy_rejected",
        "unclear": "visual_proxy_unclear",
    }.get(decision, "visual_proxy_unverified")


def page_tier(units: list[dict[str, Any]]) -> str:
    tiers = {str(unit.get("supervision_tier", "")) for unit in units}
    if tiers == {"direct"}:
        return "direct"
    if tiers == {"direct_visual_verified"}:
        return "direct_visual_verified"
    if tiers.issubset({"direct", "direct_visual_verified"}):
        return "mixed_direct_visual_verified"
    if "visual_proxy_rejected" in tiers:
        return "contains_rejected_visual_proxy"
    if "visual_proxy_unclear" in tiers or "visual_proxy_unverified" in tiers:
        return "contains_unverified_visual_proxy"
    return "mixed"


def supporting_doc_ids(row: dict[str, Any]) -> set[str]:
    return {
        str(item.get("doc_id", "")).strip()
        for item in row.get("supporting_context", []) or []
        if isinstance(item, dict) and str(item.get("doc_id", "")).strip()
    }


def verification_map(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        unit_id = str(row.get("unit_id", "")).strip()
        page_uid = str(row.get("page_uid", "")).strip()
        if unit_id and page_uid:
            out[(unit_id, page_uid)] = row
    return out


def transform_label_row(
    row: dict[str, Any],
    decisions: dict[tuple[str, str], dict[str, Any]],
    policy: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    out = copy.deepcopy(row)
    transformed_pages: list[dict[str, Any]] = []
    stats = Counter()
    original_proxy_units = 0
    verified_proxy_units = 0

    for page in out.get("pseudo_gold_pages", []) or []:
        page_uid = str(page.get("page_uid", ""))
        kept_units: list[dict[str, Any]] = []
        for unit in page.get("evidence_units", []) or []:
            unit = copy.deepcopy(unit)
            if str(unit.get("supervision_tier", "")) != "visual_proxy":
                kept_units.append(unit)
                continue
            original_proxy_units += 1
            key = (str(unit.get("unit_id", "")), page_uid)
            verification = decisions.get(key, {})
            decision = str(verification.get("decision", "unverified"))
            unit["vlm_image_page_verification"] = verification
            unit["supervision_tier"] = decision_tier(decision)
            stats[f"proxy_decision_{decision}"] += 1
            if decision == "verified":
                verified_proxy_units += 1
                kept_units.append(unit)
            elif policy == "annotate":
                kept_units.append(unit)
            else:
                stats["proxy_units_removed"] += 1

        if not kept_units:
            stats["pages_removed"] += 1
            continue
        page["evidence_units"] = kept_units
        page["score"] = round(sum(float(unit.get("score", 0.0) or 0.0) for unit in kept_units), 6)
        page["supervision_tier"] = page_tier(kept_units)
        transformed_pages.append(page)

    out["pseudo_gold_pages"] = transformed_pages
    out["pseudo_gold_page_uids"] = [str(page.get("page_uid", "")) for page in transformed_pages]
    selected_docs = {str(page.get("doc_id", "")) for page in transformed_pages}
    gold_docs = {str(value) for value in out.get("gold_doc_ids", []) or []}
    support_complete = not gold_docs or gold_docs.issubset(selected_docs)

    if not transformed_pages:
        training_tier = "exclude_unlabeled"
    elif original_proxy_units == 0:
        training_tier = "direct_only"
    elif verified_proxy_units == original_proxy_units and support_complete:
        training_tier = "complete_vlm_verified"
    elif verified_proxy_units > 0:
        training_tier = "partial_vlm_verified_positive_only"
    else:
        training_tier = "direct_remainder_positive_only"
    out["vlm_training_tier"] = training_tier
    out["vlm_verified_visual_unit_count"] = verified_proxy_units
    out["original_visual_proxy_unit_count"] = original_proxy_units
    out["support_doc_coverage_complete"] = support_complete
    stats[f"qid_tier_{training_tier}"] += 1
    return out, dict(stats)


def update_augmented_row(label_row: dict[str, Any], augmented_row: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(augmented_row)
    metadata = out.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        out["metadata"] = metadata
    pages = label_row.get("pseudo_gold_pages", []) or []
    uids = [str(page.get("page_uid", "")) for page in pages if str(page.get("page_uid", ""))]
    metadata["gold_page_uids"] = uids
    metadata["pseudo_gold_page_uids"] = uids
    metadata["pseudo_gold_page_supervision_tiers"] = {
        str(page.get("page_uid", "")): str(page.get("supervision_tier", "")) for page in pages
    }
    metadata["pseudo_gold_qid_vlm_training_tier"] = str(label_row.get("vlm_training_tier", ""))
    metadata["pseudo_gold_vlm_verified_visual_unit_count"] = int(
        label_row.get("vlm_verified_visual_unit_count", 0) or 0
    )
    return out


def main() -> None:
    args = parse_args()
    label_rows = load_jsonl(Path(args.labels_jsonl))
    augmented_rows = load_jsonl(Path(args.augmented_gold_jsonl))
    augmented_by_qid = {str(row.get("qid", "")): row for row in augmented_rows}
    decisions = verification_map(load_jsonl(Path(args.verification_jsonl)))

    transformed: list[dict[str, Any]] = []
    transformed_augmented: list[dict[str, Any]] = []
    totals = Counter()
    for row in label_rows:
        output_row, row_stats = transform_label_row(row, decisions, str(args.policy))
        totals.update(row_stats)
        transformed.append(output_row)
        qid = str(row.get("qid", ""))
        transformed_augmented.append(update_augmented_row(output_row, augmented_by_qid.get(qid, row)))

    write_jsonl(Path(args.output_labels_jsonl), transformed)
    write_jsonl(Path(args.output_augmented_gold_jsonl), transformed_augmented)
    summary = {
        "policy": str(args.policy),
        "qid_count": len(transformed),
        "labeled_qid_count": sum(bool(row.get("pseudo_gold_page_uids")) for row in transformed),
        "page_label_count": sum(len(row.get("pseudo_gold_page_uids", []) or []) for row in transformed),
        "verification_case_count": len(decisions),
        "counts": dict(sorted(totals.items())),
        "output_labels_jsonl": str(args.output_labels_jsonl),
        "output_augmented_gold_jsonl": str(args.output_augmented_gold_jsonl),
    }
    summary_path = Path(args.output_summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"saved_labels={args.output_labels_jsonl}")
    print(f"saved_augmented_gold={args.output_augmented_gold_jsonl}")
    print(f"saved_summary={summary_path}")
    print(f"labeled_qid_count={summary['labeled_qid_count']}")
    print(f"page_label_count={summary['page_label_count']}")
    print(f"counts={summary['counts']}")


if __name__ == "__main__":
    main()
