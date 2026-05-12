#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize plain top-k failure buckets by coarse cue type and export a compact "
            "pruning-sensitive subset for follow-up analysis."
        )
    )
    parser.add_argument(
        "--hard-fail-jsonl",
        required=True,
        help="JSONL from audit_plain_topk_failures.py bucket hard_fail_even_exact.jsonl",
    )
    parser.add_argument(
        "--pruning-recover-jsonl",
        required=True,
        help="JSONL from audit_plain_topk_failures.py bucket pruning_damage_exact_recovers_topk.jsonl",
    )
    parser.add_argument(
        "--pruning-improves-jsonl",
        required=True,
        help="JSONL from audit_plain_topk_failures.py bucket pruning_damage_exact_improves_but_still_fails.jsonl",
    )
    parser.add_argument(
        "--top-pruning-improves",
        type=int,
        default=4,
        help="How many strongest pruning-damage-improves qids to include in the exported subset.",
    )
    parser.add_argument(
        "--output-summary-json",
        help="Optional path to write the full summary JSON.",
    )
    parser.add_argument(
        "--output-pruning-sensitive-jsonl",
        help="Optional JSONL export of the pruning-sensitive follow-up subset.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def normalized_question(row: dict) -> str:
    return str(row.get("question", "")).strip().lower()


def classify_cue_type(row: dict) -> str:
    q = normalized_question(row)

    logo_keywords = (
        " logo",
        "logo ",
        "logo?",
        "logo.",
        "logo,",
        "symbol",
        "jersey",
        "sponsored by",
        "sponsor",
        "crest",
    )
    scene_keywords = (
        "building",
        "airport",
        "venue",
        "city",
        "location",
        "lawn",
        "entrance",
        "pier",
        "inlet",
        "seat",
        "seats",
        "arch",
        "stadium",
        "ground",
    )
    appearance_keywords = (
        " hair",
        "beard",
        "beards",
        "bald",
        "helmet",
        "glasses",
        "wearing",
        "descent",
        "african american",
        "african heritage",
        "red-haired",
        "blonde",
        "short dark hair",
    )
    cover_keywords = (
        "poster",
        "cover",
        "title screen",
        "title poster",
        "film cover",
    )
    cover_object_keywords = (
        "penguin",
        "gorilla",
        "dolphin",
        "motorcycle",
        "train",
        "torch",
        "books",
        "stethoscope",
        "guitar",
        "face",
        "woman",
        "man",
        "old man",
        "gymnast",
        "fire explosion",
        "shirtless",
        "boxing gloves",
        "single old man",
        "one woman",
        "four people",
    )

    if any(keyword in q for keyword in logo_keywords):
        return "logo_symbol"
    if any(keyword in q for keyword in scene_keywords):
        return "scene_building_location"
    if any(keyword in q for keyword in cover_keywords) and any(keyword in q for keyword in cover_object_keywords):
        return "animal_or_object_on_cover"
    if any(keyword in q for keyword in appearance_keywords):
        return "face_hair_body_attribute"
    if any(keyword in q for keyword in cover_keywords):
        return "poster_or_cover_object"
    return "other"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    hard_fail_rows = load_jsonl(Path(args.hard_fail_jsonl))
    pruning_recover_rows = load_jsonl(Path(args.pruning_recover_jsonl))
    pruning_improves_rows = load_jsonl(Path(args.pruning_improves_jsonl))

    cue_type_counts: dict[str, int] = {}
    cue_type_examples: dict[str, list[dict]] = {}
    for row in hard_fail_rows:
        cue_type = classify_cue_type(row)
        cue_type_counts[cue_type] = cue_type_counts.get(cue_type, 0) + 1
        cue_type_examples.setdefault(cue_type, [])
        if len(cue_type_examples[cue_type]) < 5:
            cue_type_examples[cue_type].append(
                {
                    "qid": row["qid"],
                    "question": row.get("question"),
                    "baseline": row.get("baseline_first_gold_doc_rank"),
                    "top224": row.get("topk_first_gold_doc_rank"),
                    "exact": row.get("exact_first_gold_doc_rank"),
                }
            )

    pruning_improves_rows.sort(
        key=lambda item: (
            -(int(item.get("exact_rank_gain_vs_topk", -10**9))),
            int(item.get("exact_first_gold_doc_rank") or 10**9),
            item["qid"],
        )
    )
    pruning_sensitive = [
        {
            "qid": row["qid"],
            "question": row.get("question"),
            "source_bucket": "pruning_damage_exact_recovers_topk",
            "baseline_first_gold_doc_rank": row.get("baseline_first_gold_doc_rank"),
            "topk_first_gold_doc_rank": row.get("topk_first_gold_doc_rank"),
            "exact_first_gold_doc_rank": row.get("exact_first_gold_doc_rank"),
            "exact_rank_gain_vs_topk": row.get("exact_rank_gain_vs_topk"),
        }
        for row in pruning_recover_rows
    ]
    pruning_sensitive.extend(
        {
            "qid": row["qid"],
            "question": row.get("question"),
            "source_bucket": "pruning_damage_exact_improves_but_still_fails",
            "baseline_first_gold_doc_rank": row.get("baseline_first_gold_doc_rank"),
            "topk_first_gold_doc_rank": row.get("topk_first_gold_doc_rank"),
            "exact_first_gold_doc_rank": row.get("exact_first_gold_doc_rank"),
            "exact_rank_gain_vs_topk": row.get("exact_rank_gain_vs_topk"),
        }
        for row in pruning_improves_rows[: max(int(args.top_pruning_improves), 0)]
    )

    summary = {
        "hard_fail_qid_count": len(hard_fail_rows),
        "hard_fail_cue_type_counts": dict(sorted(cue_type_counts.items(), key=lambda item: (-item[1], item[0]))),
        "hard_fail_cue_type_examples": cue_type_examples,
        "pruning_sensitive_qid_count": len(pruning_sensitive),
        "pruning_sensitive_qids": pruning_sensitive,
    }

    if args.output_summary_json:
        out_path = Path(args.output_summary_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.output_pruning_sensitive_jsonl:
        write_jsonl(Path(args.output_pruning_sensitive_jsonl), pruning_sensitive)

    print("hard_fail_qid_count:", summary["hard_fail_qid_count"])
    for label, count in summary["hard_fail_cue_type_counts"].items():
        print(f"{label}: {count}")
    print("pruning_sensitive_qid_count:", summary["pruning_sensitive_qid_count"])


if __name__ == "__main__":
    main()
