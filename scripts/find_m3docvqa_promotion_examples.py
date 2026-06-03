#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find thesis-friendly M3DocVQA/MMQA qualitative examples where a baseline "
            "run has a pseudo-gold evidence page in a deeper rank window and a candidate "
            "run promotes that page into the target top-k."
        )
    )
    parser.add_argument("--gold", required=True, help="Augmented gold JSONL with metadata.gold_page_uids.")
    parser.add_argument("--baseline", required=True, help="Baseline run as LABEL=prediction.json.")
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate run as LABEL=prediction.json. Can be repeated.",
    )
    parser.add_argument(
        "--qa",
        action="append",
        default=[],
        help=(
            "Optional QA prediction as LABEL=prediction.json. LABEL should match the baseline "
            "or a candidate label. Can be repeated."
        ),
    )
    parser.add_argument("--baseline-min-rank", type=int, default=6)
    parser.add_argument("--baseline-max-rank", type=int, default=20)
    parser.add_argument("--candidate-hit-k", type=int, default=4)
    parser.add_argument("--top-pages", type=int, default=10, help="How many top pages to show per run.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum detailed examples per candidate.")
    parser.add_argument("--output-md", default="", help="Optional Markdown report path.")
    parser.add_argument("--output-jsonl", default="", help="Optional JSONL path with every matched example.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_gold(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        qid = str(row.get("qid", "")).strip()
        if qid:
            out[qid] = row
    return out


def load_prediction(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("predictions"), (dict, list)):
        payload = payload["predictions"]
    if isinstance(payload, dict):
        iterator = payload.items()
    elif isinstance(payload, list):
        iterator = enumerate(payload)
    else:
        raise TypeError(f"Prediction JSON must be an object or list: {path}")

    rows: dict[str, dict[str, Any]] = {}
    for raw_key, row in iterator:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("qid") or raw_key).strip()
        if qid:
            rows[qid] = row
    return rows


def parse_labeled_path(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected LABEL=path, got {spec!r}")
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    path = Path(raw_path.strip())
    if not label or not str(path):
        raise ValueError(f"Invalid labeled path: {spec!r}")
    return label, path


def parse_page_uid(uid: str) -> tuple[str, int] | None:
    if "_page" not in uid:
        return None
    doc_id, raw_page = uid.rsplit("_page", 1)
    if not doc_id:
        return None
    try:
        return doc_id, int(raw_page)
    except ValueError:
        return None


def page_uid(doc_id: str, page_idx: int) -> str:
    return f"{doc_id}_page{int(page_idx)}"


def parse_prediction_item(item: Any) -> tuple[str, int, float | None] | None:
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        doc_id = str(item[0]).strip()
        if not doc_id:
            return None
        try:
            page_idx = int(item[1])
        except (TypeError, ValueError):
            return None
        score = None
        if len(item) >= 3 and item[2] is not None:
            try:
                score = float(item[2])
            except (TypeError, ValueError):
                score = None
        return doc_id, page_idx, score

    if isinstance(item, dict):
        raw_uid = str(item.get("page_uid", "")).strip()
        if raw_uid:
            parsed = parse_page_uid(raw_uid)
            if parsed is None:
                return None
            doc_id, page_idx = parsed
        else:
            doc_id = str(item.get("doc_id", item.get("docid", item.get("document_id", "")))).strip()
            raw_page = item.get("page_idx", item.get("page_id", item.get("page")))
            if not doc_id or raw_page is None:
                return None
            try:
                page_idx = int(raw_page)
            except (TypeError, ValueError):
                return None
        raw_score = item.get("score", item.get("retrieval_score", item.get("fused_page_score")))
        score = None
        if raw_score is not None:
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                score = None
        return doc_id, page_idx, score

    return None


def prediction_items(row: dict[str, Any] | None) -> list[Any]:
    if not isinstance(row, dict):
        return []
    for key in ("page_retrieval_results", "retrieval_results", "results", "pages"):
        value = row.get(key)
        if isinstance(value, list):
            return value
    return []


def ranked_pages(row: dict[str, Any] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in prediction_items(row):
        parsed = parse_prediction_item(item)
        if parsed is None:
            continue
        doc_id, page_idx, score = parsed
        uid = page_uid(doc_id, page_idx)
        if uid in seen:
            continue
        seen.add(uid)
        out.append(
            {
                "rank": len(out) + 1,
                "doc_id": doc_id,
                "page_idx": page_idx,
                "page_uid": uid,
                "score": score,
            }
        )
    return out


def gold_page_uids(row: dict[str, Any]) -> set[str]:
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    values = (
        metadata.get("gold_page_uids")
        or metadata.get("pseudo_gold_page_uids")
        or row.get("gold_page_uids")
        or row.get("pseudo_gold_page_uids")
        or []
    )
    pages = {str(value).strip() for value in values if str(value).strip()}
    for ctx in row.get("supporting_context", []):
        if not isinstance(ctx, dict):
            continue
        doc_id = str(ctx.get("doc_id", ctx.get("doc_name", ""))).strip()
        raw_page = ctx.get("page_idx", ctx.get("page_id", ctx.get("page")))
        if doc_id and raw_page is not None:
            try:
                pages.add(page_uid(doc_id, int(raw_page)))
            except (TypeError, ValueError):
                pass
    return pages


def first_gold_page(ranked: list[dict[str, Any]], gold: set[str]) -> tuple[int | None, str | None]:
    for row in ranked:
        uid = str(row["page_uid"])
        if uid in gold:
            return int(row["rank"]), uid
    return None, None


def question_type(row: dict[str, Any]) -> str:
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    return str(metadata.get("type") or row.get("question_type") or "").strip() or "UNKNOWN"


def answer_strings(row: dict[str, Any]) -> list[str]:
    answers: list[str] = []
    for item in row.get("answers", []):
        value: Any
        if isinstance(item, dict):
            value = item.get("answer")
        else:
            value = item
        text = str(value).strip()
        if text:
            answers.append(text)
    return answers


def qa_answer(row: dict[str, Any] | None) -> str:
    if not isinstance(row, dict):
        return ""
    for key in ("pred_answer", "generated_answer", "answer", "prediction", "pred", "response"):
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            text = json.dumps(value, ensure_ascii=False)
        else:
            text = str(value)
        text = text.strip()
        if text:
            return text
    return ""


def fmt_score(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def md_escape(value: Any) -> str:
    text = str(value).replace("\n", " ").replace("\r", " ").strip()
    text = text.replace("|", "\\|")
    return text


def short_text(value: Any, limit: int = 220) -> str:
    text = md_escape(value)
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


def top_page_rows(ranked: list[dict[str, Any]], gold: set[str], limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in ranked[: max(int(limit), 0)]:
        copied = dict(row)
        copied["is_gold"] = str(row["page_uid"]) in gold
        rows.append(copied)
    return rows


def build_case(
    *,
    qid: str,
    gold_row: dict[str, Any],
    baseline_label: str,
    candidate_label: str,
    baseline_row: dict[str, Any],
    candidate_row: dict[str, Any],
    baseline_qa_row: dict[str, Any] | None,
    candidate_qa_row: dict[str, Any] | None,
    top_pages: int,
) -> dict[str, Any]:
    gold = gold_page_uids(gold_row)
    baseline_ranked = ranked_pages(baseline_row)
    candidate_ranked = ranked_pages(candidate_row)
    baseline_rank, baseline_gold_uid = first_gold_page(baseline_ranked, gold)
    candidate_rank, candidate_gold_uid = first_gold_page(candidate_ranked, gold)
    return {
        "qid": qid,
        "question": str(gold_row.get("question", "")).strip(),
        "question_type": question_type(gold_row),
        "answers": answer_strings(gold_row),
        "gold_page_uids": sorted(gold),
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "baseline_first_gold_page_rank": baseline_rank,
        "baseline_first_gold_page_uid": baseline_gold_uid,
        "candidate_first_gold_page_rank": candidate_rank,
        "candidate_first_gold_page_uid": candidate_gold_uid,
        "rank_improvement": None if baseline_rank is None or candidate_rank is None else baseline_rank - candidate_rank,
        "baseline_qa_answer": qa_answer(baseline_qa_row),
        "candidate_qa_answer": qa_answer(candidate_qa_row),
        "baseline_top_pages": top_page_rows(baseline_ranked, gold, top_pages),
        "candidate_top_pages": top_page_rows(candidate_ranked, gold, top_pages),
    }


def render_page_table(title: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = [f"**{title}**", "", "| rank | page_uid | score | gold? |", "| --- | --- | --- | --- |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["rank"]),
                    md_escape(row["page_uid"]),
                    fmt_score(row.get("score")),
                    "yes" if row.get("is_gold") else "",
                ]
            )
            + " |"
        )
    return lines


def render_markdown(
    *,
    baseline_label: str,
    baseline_min_rank: int,
    baseline_max_rank: int,
    candidate_hit_k: int,
    summary_rows: list[dict[str, Any]],
    examples_by_candidate: dict[str, list[dict[str, Any]]],
    limit: int,
) -> str:
    lines: list[str] = [
        "# M3DocVQA Qualitative Promotion Examples",
        "",
        (
            "Selection rule: baseline first pseudo-gold page rank is between "
            f"{baseline_min_rank} and {baseline_max_rank}, and candidate first pseudo-gold "
            f"page rank is at most {candidate_hit_k}."
        ),
        "",
        "| baseline | candidate | evaluated_qids | matched_examples | mean_rank_gain |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    md_escape(baseline_label),
                    md_escape(row["candidate_label"]),
                    str(row["evaluated_qids"]),
                    str(row["matched_examples"]),
                    "" if row["mean_rank_gain"] is None else f"{row['mean_rank_gain']:.2f}",
                ]
            )
            + " |"
        )

    for candidate_label, examples in examples_by_candidate.items():
        lines.extend(["", f"## {md_escape(candidate_label)}", ""])
        if not examples:
            lines.append("No examples matched the selection rule.")
            continue
        lines.extend(
            [
                "| qid | question_type | baseline_rank | candidate_rank | rank_gain | question |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for case in examples[:limit]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        md_escape(case["qid"]),
                        md_escape(case["question_type"]),
                        str(case["baseline_first_gold_page_rank"]),
                        str(case["candidate_first_gold_page_rank"]),
                        str(case["rank_improvement"]),
                        short_text(case["question"], 140),
                    ]
                )
                + " |"
            )

        for idx, case in enumerate(examples[:limit], start=1):
            lines.extend(
                [
                    "",
                    f"### Example {idx}: {md_escape(case['qid'])}",
                    "",
                    f"- question_type: `{md_escape(case['question_type'])}`",
                    f"- question: {md_escape(case['question'])}",
                    f"- gold_answers: {md_escape(case['answers'])}",
                    f"- gold_page_uids: {md_escape(case['gold_page_uids'])}",
                    (
                        f"- rank movement: {md_escape(case['baseline_label'])} rank "
                        f"{case['baseline_first_gold_page_rank']} -> "
                        f"{md_escape(case['candidate_label'])} rank "
                        f"{case['candidate_first_gold_page_rank']}"
                    ),
                ]
            )
            if case.get("baseline_qa_answer") or case.get("candidate_qa_answer"):
                lines.extend(
                    [
                        f"- {md_escape(case['baseline_label'])} QA answer: {md_escape(case.get('baseline_qa_answer', ''))}",
                        f"- {md_escape(case['candidate_label'])} QA answer: {md_escape(case.get('candidate_qa_answer', ''))}",
                    ]
                )
            lines.append("")
            lines.extend(render_page_table(f"{case['baseline_label']} top pages", case["baseline_top_pages"]))
            lines.append("")
            lines.extend(render_page_table(f"{case['candidate_label']} top pages", case["candidate_top_pages"]))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    baseline_label, baseline_path = parse_labeled_path(args.baseline)
    candidate_specs = [parse_labeled_path(spec) for spec in args.candidate]
    qa_specs = dict(parse_labeled_path(spec) for spec in args.qa)

    gold = load_gold(Path(args.gold))
    baseline = load_prediction(baseline_path)
    candidates = {label: load_prediction(path) for label, path in candidate_specs}
    qa_predictions = {label: load_prediction(path) for label, path in qa_specs.items()}

    examples_by_candidate: dict[str, list[dict[str, Any]]] = {}
    all_matches: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for candidate_label, candidate in candidates.items():
        matched: list[dict[str, Any]] = []
        evaluated = 0
        qids = sorted(set(gold) & set(baseline) & set(candidate))
        for qid in qids:
            gold_row = gold[qid]
            if not gold_page_uids(gold_row):
                continue
            evaluated += 1
            case = build_case(
                qid=qid,
                gold_row=gold_row,
                baseline_label=baseline_label,
                candidate_label=candidate_label,
                baseline_row=baseline[qid],
                candidate_row=candidate[qid],
                baseline_qa_row=qa_predictions.get(baseline_label, {}).get(qid),
                candidate_qa_row=qa_predictions.get(candidate_label, {}).get(qid),
                top_pages=int(args.top_pages),
            )
            baseline_rank = case["baseline_first_gold_page_rank"]
            candidate_rank = case["candidate_first_gold_page_rank"]
            if (
                baseline_rank is not None
                and candidate_rank is not None
                and int(args.baseline_min_rank) <= int(baseline_rank) <= int(args.baseline_max_rank)
                and int(candidate_rank) <= int(args.candidate_hit_k)
            ):
                matched.append(case)

        matched.sort(
            key=lambda row: (
                -(row["rank_improvement"] or 0),
                row["candidate_first_gold_page_rank"] or 10**9,
                row["baseline_first_gold_page_rank"] or 10**9,
                row["qid"],
            )
        )
        examples_by_candidate[candidate_label] = matched
        all_matches.extend(matched)
        gains = [row["rank_improvement"] for row in matched if row["rank_improvement"] is not None]
        summary_rows.append(
            {
                "candidate_label": candidate_label,
                "evaluated_qids": evaluated,
                "matched_examples": len(matched),
                "mean_rank_gain": mean(gains) if gains else None,
            }
        )

    if args.output_jsonl:
        path = Path(args.output_jsonl)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in all_matches:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    markdown = render_markdown(
        baseline_label=baseline_label,
        baseline_min_rank=int(args.baseline_min_rank),
        baseline_max_rank=int(args.baseline_max_rank),
        candidate_hit_k=int(args.candidate_hit_k),
        summary_rows=summary_rows,
        examples_by_candidate=examples_by_candidate,
        limit=int(args.limit),
    )
    if args.output_md:
        path = Path(args.output_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
        print(f"saved_output_md={path}")
    else:
        print(markdown)
    if args.output_jsonl:
        print(f"saved_output_jsonl={args.output_jsonl}")


if __name__ == "__main__":
    main()
