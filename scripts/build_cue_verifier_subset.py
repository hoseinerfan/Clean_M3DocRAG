#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


GENERIC_CUE_TOKENS = {
    "logo",
    "poster",
    "cover",
    "title",
    "team",
    "club",
    "player",
    "players",
    "driver",
    "drivers",
    "rider",
    "riders",
    "person",
    "people",
    "film",
    "movie",
    "album",
    "show",
    "series",
    "man",
    "woman",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "included",
    "is",
    "it",
    "its",
    "listed",
    "middle",
    "of",
    "on",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "to",
    "which",
    "with",
}

LOW_SPECIFICITY_TOKENS = {
    "soccer",
    "ball",
    "hair",
    "face",
    "poster",
    "logo",
    "cover",
    "woman",
    "man",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a cue-verifier subset from a failure JSONL by filtering to qids whose "
            "gold doc rank falls in a target range and auto-suggesting cue tokens."
        )
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--min-gold-doc-rank", type=int, default=5)
    parser.add_argument("--max-gold-doc-rank", type=int, default=20)
    parser.add_argument("--question-type", default="ImageListQ")
    parser.add_argument("--require-single-gold-doc", action="store_true")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-manual-overrides-jsonl", required=True)
    parser.add_argument("--output-summary-json", required=True)
    return parser.parse_args()


def normalize_spaces(text: str) -> str:
    return " ".join(str(text).strip().split())


def extract_cue_phrase(question: str) -> tuple[str | None, str]:
    text = normalize_spaces(question)
    lowered = text.lower()
    patterns: list[tuple[str, str]] = [
        (r"there is (?:an?|the) (.+?) on (?:the|its) (?:logo|poster|cover|jersey|shirt|symbol)\b", "there_is_on_object"),
        (r"there is (?:an?|the) (.+?) in (?:the|its) (?:logo|middle of the logo|symbol|coat of arms)\b", "there_is_in_object"),
        (r"logo featuring (.+?)(?:\b in\b|\b of\b|\b had\b|\b has\b|\?)", "logo_featuring"),
        (r"has (?:an?|the) (.+?) on (?:its|the) (?:logo|poster|cover|jersey|shirt|symbol)\b", "has_on_object"),
        (r"has (?:an?|the) (.+?) in (?:its|the) (?:logo|middle of the logo|symbol|coat of arms)\b", "has_in_object"),
        (r"features (?:an?|the) (.+?) on (?:its|the) (?:logo|poster|cover|jersey|shirt|symbol)\b", "features_on_object"),
        (r"features (?:an?|the) (.+?) in (?:its|the) (?:logo|middle of the logo|symbol|coat of arms)\b", "features_in_object"),
        (r"which .* has (.+?)\?$", "which_has_suffix"),
        (r"which .* is (?:a|an) (.+?)\?$", "which_is_suffix"),
        (r"which .* with (.+?)\?$", "which_with_suffix"),
    ]
    for pattern, source in patterns:
        match = re.search(pattern, lowered)
        if not match:
            continue
        phrase = normalize_spaces(match.group(1).strip(" ,.?"))
        if phrase:
            return phrase, source
    return None, "none"


def tokenize_phrase(phrase: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", phrase.lower())
    return [token for token in tokens if token and token not in STOPWORDS]


def suggest_cue_tokens(question: str) -> tuple[list[str], str | None, str, str]:
    cue_phrase, cue_source = extract_cue_phrase(question)
    if cue_phrase is None:
        return [], None, cue_source, "none"

    tokens = tokenize_phrase(cue_phrase)
    if not tokens:
        return [], cue_phrase, cue_source, "none"

    non_generic_tokens = [token for token in tokens if token not in GENERIC_CUE_TOKENS]
    if non_generic_tokens:
        suggested = non_generic_tokens
    else:
        suggested = tokens

    if len(suggested) > 3:
        suggested = suggested[-3:]

    low_specificity_hits = sum(token in LOW_SPECIFICITY_TOKENS for token in suggested)
    if low_specificity_hits == 0 and len(suggested) >= 1:
        specificity = "high"
    elif low_specificity_hits < len(suggested):
        specificity = "medium"
    else:
        specificity = "low"

    return suggested, cue_phrase, cue_source, specificity


def main() -> None:
    args = parse_args()

    selected_rows: list[dict] = []
    manual_override_rows: list[dict] = []

    with Path(args.input_jsonl).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question_type = str(row.get("question_type", "")).strip()
            if args.question_type and question_type != args.question_type:
                continue
            rank_value = row.get("baseline_first_gold_doc_rank")
            if rank_value is None:
                continue
            gold_doc_rank = int(rank_value)
            if gold_doc_rank < int(args.min_gold_doc_rank) or gold_doc_rank > int(args.max_gold_doc_rank):
                continue
            gold_doc_ids = [str(value) for value in row.get("gold_doc_ids", [])]
            if args.require_single_gold_doc and len(gold_doc_ids) != 1:
                continue

            cue_tokens, cue_phrase, cue_source, cue_specificity = suggest_cue_tokens(str(row["question"]))
            suggested_positive_page_uid = (
                f"{gold_doc_ids[0]}_page0" if len(gold_doc_ids) == 1 else None
            )

            output_row = {
                "qid": row["qid"],
                "question": row["question"],
                "question_type": question_type,
                "baseline_first_gold_doc_rank": gold_doc_rank,
                "gold_doc_ids": gold_doc_ids,
                "suggested_positive_page_uid": suggested_positive_page_uid,
                "cue_phrase": cue_phrase,
                "cue_source": cue_source,
                "cue_token_substrings": cue_tokens,
                "cue_specificity": cue_specificity,
            }
            selected_rows.append(output_row)

            manual_override_rows.append(
                {
                    "qid": row["qid"],
                    "positive_page_uid": suggested_positive_page_uid,
                    "cue_token_substrings": cue_tokens,
                    "notes": (
                        f"auto-suggested from cue_phrase={cue_phrase!r}; "
                        f"cue_source={cue_source}; cue_specificity={cue_specificity}. "
                        "Manual page and cue review still recommended."
                    ),
                }
            )

    selected_rows.sort(key=lambda item: (item["baseline_first_gold_doc_rank"], item["qid"]))

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in selected_rows:
            handle.write(json.dumps(row) + "\n")

    output_override_jsonl = Path(args.output_manual_overrides_jsonl)
    output_override_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_override_jsonl.open("w", encoding="utf-8") as handle:
        for row in manual_override_rows:
            handle.write(json.dumps(row) + "\n")

    specificity_counts: dict[str, int] = {}
    single_gold_doc_count = 0
    for row in selected_rows:
        specificity_counts[row["cue_specificity"]] = specificity_counts.get(row["cue_specificity"], 0) + 1
        if len(row["gold_doc_ids"]) == 1:
            single_gold_doc_count += 1

    summary = {
        "input_jsonl": args.input_jsonl,
        "question_type": args.question_type,
        "min_gold_doc_rank": int(args.min_gold_doc_rank),
        "max_gold_doc_rank": int(args.max_gold_doc_rank),
        "require_single_gold_doc": bool(args.require_single_gold_doc),
        "selected_qid_count": len(selected_rows),
        "single_gold_doc_count": single_gold_doc_count,
        "cue_specificity_counts": specificity_counts,
    }
    output_summary_json = Path(args.output_summary_json)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"saved_jsonl: {output_jsonl}")
    print(f"saved_manual_overrides_jsonl: {output_override_jsonl}")
    print(f"saved_summary: {output_summary_json}")
    print(f"count: {len(selected_rows)}")
    for row in selected_rows[:20]:
        print(
            f"{row['baseline_first_gold_doc_rank']} {row['qid']} | "
            f"cue={row['cue_token_substrings']} | specificity={row['cue_specificity']} | "
            f"question={row['question']}"
        )


if __name__ == "__main__":
    main()
