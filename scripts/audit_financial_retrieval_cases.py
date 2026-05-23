#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_TEXT_FIELDS = ["ocr_text", "vlm_text", "markdown", "text", "page_text", "content"]


def load_financial_helpers() -> Any:
    module_path = Path(__file__).with_name("rerank_financial_evidence_pages.py")
    spec = importlib.util.spec_from_file_location("_financial_evidence_helpers", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load financial helper module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_financial_evidence_helpers"] = module
    spec.loader.exec_module(module)
    return module


FIN = load_financial_helpers()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect financial-domain retrieval successes and failures, including "
            "metric/year/entity evidence scores for gold and top candidate pages."
        )
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--doc-pages-jsonl", required=True)
    parser.add_argument("--candidate-summary-json", default="")
    parser.add_argument("--graph-summary-json", default="")
    parser.add_argument("--text-field", nargs="*", default=DEFAULT_TEXT_FIELDS)
    parser.add_argument("--filter-field", default="metadata.domain")
    parser.add_argument("--filter-value", default="Financial report")
    parser.add_argument("--hit-k", type=int, default=4)
    parser.add_argument("--show-top-pages", type=int, default=8)
    parser.add_argument("--topn", type=int, default=20)
    parser.add_argument("--line-window", type=int, default=3)
    parser.add_argument(
        "--financial-scoring-mode",
        choices=["soft", "strict"],
        default="soft",
        help="Evidence scorer mode passed to scripts/rerank_financial_evidence_pages.py.",
    )
    parser.add_argument("--broad-positive-threshold", type=int, default=500)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_prediction(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "predictions" in payload and isinstance(
        payload["predictions"], (dict, list)
    ):
        payload = payload["predictions"]
    if isinstance(payload, dict):
        iterable = payload.items()
    elif isinstance(payload, list):
        iterable = enumerate(payload)
    else:
        raise TypeError(f"Unsupported prediction payload: {path}")
    rows = {}
    for raw_key, row in iterable:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("qid", "")).strip() or str(raw_key).strip()
        if qid:
            rows[qid] = row
    return rows


def load_summary_per_qid(path: str) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    per_qid = payload.get("per_qid", []) if isinstance(payload, dict) else []
    rows = {}
    for row in per_qid:
        if isinstance(row, dict) and str(row.get("qid", "")).strip():
            rows[str(row["qid"]).strip()] = row
    return rows


def metadata_value(row: dict[str, Any], dotted_key: str) -> str:
    current: Any = row
    for part in dotted_key.split("."):
        if not isinstance(current, dict):
            return ""
        current = current.get(part)
    return str(current).strip()


def page_uid(doc_id: str, page_idx: int) -> str:
    return f"{doc_id}_page{int(page_idx)}"


def parse_page_row(row: Any) -> tuple[str, int, float | None] | None:
    if not isinstance(row, (list, tuple)) or len(row) < 2:
        return None
    doc_id = str(row[0]).strip()
    if not doc_id:
        return None
    try:
        page_idx = int(row[1])
    except (TypeError, ValueError):
        return None
    score = None
    if len(row) >= 3:
        try:
            score = float(row[2])
        except (TypeError, ValueError):
            score = None
    return doc_id, page_idx, score


def ranked_pages(pred_row: dict[str, Any]) -> list[str]:
    pages = []
    for row in pred_row.get("page_retrieval_results", []):
        parsed = parse_page_row(row)
        if parsed is not None:
            pages.append(page_uid(parsed[0], parsed[1]))
    return pages


def ranked_docs(pred_row: dict[str, Any]) -> list[str]:
    docs = []
    seen = set()
    for row in pred_row.get("page_retrieval_results", []):
        parsed = parse_page_row(row)
        if parsed is None:
            continue
        if parsed[0] not in seen:
            seen.add(parsed[0])
            docs.append(parsed[0])
    return docs


def top_pages(pred_row: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    out = []
    for rank, row in enumerate(pred_row.get("page_retrieval_results", [])[:limit], start=1):
        parsed = parse_page_row(row)
        if parsed is None:
            continue
        out.append(
            {
                "rank": rank,
                "page_uid": page_uid(parsed[0], parsed[1]),
                "doc_id": parsed[0],
                "page_idx": parsed[1],
                "score": parsed[2],
            }
        )
    return out


def first_rank(items: list[str], gold: set[str]) -> int | None:
    for idx, item in enumerate(items, start=1):
        if item in gold:
            return idx
    return None


def gold_page_uids(row: dict[str, Any]) -> set[str]:
    metadata = row.get("metadata", {})
    pages = {
        str(value).strip()
        for value in metadata.get("gold_page_uids", [])
        if str(value).strip()
    }
    for ctx in row.get("supporting_context", []):
        doc_id = str(ctx.get("doc_id", "")).strip()
        page_idx = ctx.get("page_idx", ctx.get("page_id"))
        if doc_id and page_idx is not None:
            pages.add(page_uid(doc_id, int(page_idx)))
    return pages


def gold_doc_ids(row: dict[str, Any]) -> set[str]:
    docs = {
        str(ctx.get("doc_id", "")).strip()
        for ctx in row.get("supporting_context", [])
        if str(ctx.get("doc_id", "")).strip()
    }
    for uid in gold_page_uids(row):
        if "_page" in uid:
            docs.add(uid.rsplit("_page", 1)[0])
    return docs


def movement_for_hit(base_rank: int | None, cand_rank: int | None, hit_k: int) -> str:
    base_hit = base_rank is not None and base_rank <= hit_k
    cand_hit = cand_rank is not None and cand_rank <= hit_k
    if not base_hit and cand_hit:
        return "recovered"
    if base_hit and not cand_hit:
        return "lost"
    if base_rank is None and cand_rank is None:
        return "missing_in_both"
    if base_rank is not None and cand_rank is not None and cand_rank < base_rank:
        return "improved_rank"
    if base_rank is not None and cand_rank is not None and cand_rank > base_rank:
        return "worsened_rank"
    return "unchanged"


def score_page(
    question: str,
    page_uid_value: str,
    page_texts: dict[str, str],
    line_window: int,
    scoring_mode: str,
) -> dict[str, Any]:
    score, meta = FIN.financial_evidence_score(
        question,
        page_texts.get(page_uid_value, ""),
        int(line_window),
        str(scoring_mode),
    )
    slots = meta.get("slots", {})
    return {
        "page_uid": page_uid_value,
        "evidence_score": score,
        "active": bool(meta.get("active", False)),
        "reason": str(meta.get("reason", "")),
        "slots": slots,
        "best_line_score": meta.get("best_line_score"),
        "best_window_score": meta.get("best_window_score"),
        "best_strict_line_score": meta.get("best_strict_line_score"),
        "best_strict_window_score": meta.get("best_strict_window_score"),
        "table_score": meta.get("table_score"),
        "page_slot_score": meta.get("page_slot_score"),
        "scoring_mode": meta.get("scoring_mode"),
    }


def score_page_list(
    question: str,
    page_uids: list[str],
    page_texts: dict[str, str],
    line_window: int,
    scoring_mode: str,
) -> list[dict[str, Any]]:
    scored = [
        score_page(question, uid, page_texts, line_window, scoring_mode)
        for uid in page_uids
    ]
    return sorted(scored, key=lambda row: (-float(row["evidence_score"]), row["page_uid"]))


def classify_limitation(
    row: dict[str, Any],
    broad_positive_threshold: int,
) -> str:
    if row["movement"] in {"recovered", "improved_rank"}:
        if row["gold_best_evidence_score"] > 0 and row["gold_evidence_rank_among_candidate_top"] is not None:
            return "gold_page_has_evidence_and_moves_up"
        if row["gold_best_evidence_score"] > 0:
            return "gold_page_has_evidence_but_not_in_candidate_head"
        return "rank_improved_without_gold_financial_evidence"
    if row["movement"] == "missing_in_both":
        return "gold_page_missing_from_candidate_pool"
    if row["candidate_rank"] is None:
        return "gold_page_not_in_candidate_prediction"
    if row["positive_page_count"] >= broad_positive_threshold:
        return "evidence_too_broad"
    if row["gold_best_evidence_score"] <= 0:
        return "gold_page_no_financial_evidence_match"
    if row["candidate_top_best_evidence_score"] >= row["gold_best_evidence_score"]:
        return "competing_top_pages_score_as_high_or_higher"
    return "gold_page_has_signal_but_weight_or_base_rank_insufficient"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def compact_page_rows(rows: list[dict[str, Any]]) -> str:
    parts = []
    for row in rows:
        score = row.get("evidence_score", row.get("score"))
        if score is None:
            parts.append(f"{row.get('rank', '?')}:{row['page_uid']}")
        else:
            parts.append(f"{row.get('rank', '?')}:{row['page_uid']}({float(score):.3f})")
    return "; ".join(parts)


def main() -> None:
    args = parse_args()
    baseline = load_prediction(Path(args.baseline))
    candidate = load_prediction(Path(args.candidate))
    gold_rows = {str(row["qid"]): row for row in read_jsonl(Path(args.gold))}
    page_texts = FIN.load_page_texts(Path(args.doc_pages_jsonl), list(args.text_field))
    candidate_summary = load_summary_per_qid(args.candidate_summary_json)
    graph_summary = load_summary_per_qid(args.graph_summary_json)

    rows = []
    hit_k = int(args.hit_k)
    show_top_pages = int(args.show_top_pages)
    for qid in sorted(set(baseline) & set(candidate) & set(gold_rows)):
        gold_row = gold_rows[qid]
        if args.filter_field:
            if metadata_value(gold_row, args.filter_field) != str(args.filter_value):
                continue
        question = str(gold_row.get("question", ""))
        gold_pages = gold_page_uids(gold_row)
        gold_docs = gold_doc_ids(gold_row)
        base_pages = ranked_pages(baseline[qid])
        cand_pages = ranked_pages(candidate[qid])
        base_rank = first_rank(base_pages, gold_pages)
        cand_rank = first_rank(cand_pages, gold_pages)
        base_doc_rank = first_rank(ranked_docs(baseline[qid]), gold_docs)
        cand_doc_rank = first_rank(ranked_docs(candidate[qid]), gold_docs)
        movement = movement_for_hit(base_rank, cand_rank, hit_k)

        cand_top = top_pages(candidate[qid], show_top_pages)
        cand_top_uids = [row["page_uid"] for row in cand_top]
        scored_cand_top = {
            row["page_uid"]: row
            for row in score_page_list(
                question,
                cand_top_uids,
                page_texts,
                int(args.line_window),
                str(args.financial_scoring_mode),
            )
        }
        for top_row in cand_top:
            top_row.update(scored_cand_top.get(top_row["page_uid"], {}))

        gold_scores = score_page_list(
            question,
            sorted(gold_pages),
            page_texts,
            int(args.line_window),
            str(args.financial_scoring_mode),
        )
        gold_best = gold_scores[0] if gold_scores else {"evidence_score": 0.0}
        gold_rank_among_cand_top = None
        for index, top_uid in enumerate(
            [
                row["page_uid"]
                for row in sorted(
                    cand_top,
                    key=lambda value: (
                        -float(value.get("evidence_score", 0.0) or 0.0),
                        int(value["rank"]),
                    ),
                )
            ],
            start=1,
        ):
            if top_uid in gold_pages:
                gold_rank_among_cand_top = index
                break

        cand_summary = candidate_summary.get(qid, {})
        graph_row = graph_summary.get(qid, {}).get("graph", {})
        row = {
            "qid": qid,
            "question": question,
            "metadata_type": metadata_value(gold_row, "metadata.type"),
            "metadata_domain": metadata_value(gold_row, "metadata.domain"),
            "gold_page_uids": sorted(gold_pages),
            "baseline_rank": base_rank,
            "candidate_rank": cand_rank,
            "baseline_doc_rank": base_doc_rank,
            "candidate_doc_rank": cand_doc_rank,
            "movement": movement,
            "rank_delta": None if base_rank is None or cand_rank is None else base_rank - cand_rank,
            "gold_best_evidence_score": float(gold_best.get("evidence_score", 0.0) or 0.0),
            "gold_best_evidence_page": gold_best.get("page_uid"),
            "gold_best_evidence_detail": gold_best,
            "candidate_top_best_evidence_score": max(
                (float(row.get("evidence_score", 0.0) or 0.0) for row in cand_top),
                default=0.0,
            ),
            "gold_evidence_rank_among_candidate_top": gold_rank_among_cand_top,
            "candidate_top_pages": cand_top,
            "gold_evidence_pages": gold_scores,
            "positive_page_count": int(cand_summary.get("positive_page_count", 0) or 0),
            "mean_evidence_score": float(cand_summary.get("mean_evidence_score", 0.0) or 0.0),
            "max_evidence_score": float(cand_summary.get("max_evidence_score", 0.0) or 0.0),
            "top_evidence_pages": cand_summary.get("top_evidence_pages", []),
            "graph_financial_reasoning_active": bool(
                graph_row.get("query_anchor_financial_reasoning_active", False)
            ),
            "graph_financial_reasoning_reason": str(
                graph_row.get("query_anchor_financial_reasoning_reason", "")
            ),
            "graph_financial_bundle_page_match_count": int(
                graph_row.get("query_anchor_financial_bundle_page_match_count", 0) or 0
            ),
            "graph_financial_bundle_label": str(
                graph_row.get("query_anchor_financial_bundle_label", "")
            ),
        }
        row["limitation"] = classify_limitation(row, int(args.broad_positive_threshold))
        rows.append(row)

    movement_counts = Counter(row["movement"] for row in rows)
    limitation_counts = Counter(row["limitation"] for row in rows)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row["metadata_type"]].append(row)

    def group_summary(group: list[dict[str, Any]]) -> dict[str, Any]:
        moves = Counter(row["movement"] for row in group)
        return {
            "n": len(group),
            "recovered": moves.get("recovered", 0),
            "lost": moves.get("lost", 0),
            "improved_rank": moves.get("improved_rank", 0),
            "worsened_rank": moves.get("worsened_rank", 0),
            "unchanged": moves.get("unchanged", 0),
            "missing_in_both": moves.get("missing_in_both", 0),
            "mean_gold_evidence_score": (
                statistics.fmean(row["gold_best_evidence_score"] for row in group)
                if group
                else 0.0
            ),
            "mean_positive_page_count": (
                statistics.fmean(row["positive_page_count"] for row in group) if group else 0.0
            ),
        }

    def top_cases(movement: str, topn: int) -> list[dict[str, Any]]:
        selected = [row for row in rows if row["movement"] == movement]
        if movement == "improved_rank":
            return sorted(
                selected,
                key=lambda row: (
                    -(row["rank_delta"] or 0),
                    row["candidate_rank"] if row["candidate_rank"] is not None else 10**9,
                    row["qid"],
                ),
            )[:topn]
        if movement == "worsened_rank":
            return sorted(
                selected,
                key=lambda row: (
                    row["rank_delta"] or 0,
                    row["candidate_rank"] if row["candidate_rank"] is not None else 10**9,
                    row["qid"],
                ),
            )[:topn]
        return sorted(
            selected,
            key=lambda row: (
                row["candidate_rank"] if row["candidate_rank"] is not None else 10**9,
                row["baseline_rank"] if row["baseline_rank"] is not None else 10**9,
                row["qid"],
            ),
        )[:topn]

    payload = {
        "n_qids": len(rows),
        "hit_k": hit_k,
        "financial_scoring_mode": str(args.financial_scoring_mode),
        "line_window": int(args.line_window),
        "movement_counts": dict(movement_counts),
        "limitation_counts": dict(limitation_counts),
        "by_type": {key: group_summary(group) for key, group in by_type.items()},
        "overall": group_summary(rows),
        "top_recovered": top_cases("recovered", int(args.topn)),
        "top_improved_rank": top_cases("improved_rank", int(args.topn)),
        "top_worsened_rank": top_cases("worsened_rank", int(args.topn)),
        "top_missing_in_both": top_cases("missing_in_both", int(args.topn)),
        "all_cases": rows,
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Financial Retrieval Case Audit",
        "",
        f"- qids: `{len(rows)}`",
        f"- hit_k: `{hit_k}`",
        f"- filter: `{args.filter_field} == {args.filter_value}`",
        f"- page_text_count: `{len(page_texts)}`",
        f"- financial_scoring_mode: `{args.financial_scoring_mode}`",
        f"- line_window: `{args.line_window}`",
        "",
        "## Movement",
        "",
        markdown_table(
            ["movement", "count"],
            [[key, movement_counts.get(key, 0)] for key in sorted(movement_counts)],
        ),
        "",
        "## Limitation Classes",
        "",
        markdown_table(
            ["class", "count"],
            [
                [key, value]
                for key, value in sorted(limitation_counts.items(), key=lambda item: (-item[1], item[0]))
            ],
        ),
        "",
        "## By Type",
        "",
        markdown_table(
            [
                "type",
                "n",
                "recovered",
                "improved",
                "worsened",
                "missing",
                "mean_gold_evidence",
                "mean_positive_pages",
            ],
            [
                [
                    key,
                    summary["n"],
                    summary["recovered"],
                    summary["improved_rank"],
                    summary["worsened_rank"],
                    summary["missing_in_both"],
                    f"{summary['mean_gold_evidence_score']:.3f}",
                    f"{summary['mean_positive_page_count']:.1f}",
                ]
                for key, summary in sorted(
                    payload["by_type"].items(),
                    key=lambda item: (-(item[1]["recovered"] + item[1]["improved_rank"]), item[0]),
                )
            ],
        ),
    ]

    def add_case_section(title: str, cases: list[dict[str, Any]]) -> None:
        lines.extend(["", f"## {title}", ""])
        for row in cases:
            lines.extend(
                [
                    f"- `{row['qid']}` base={row['baseline_rank']} cand={row['candidate_rank']} "
                    f"doc_base={row['baseline_doc_rank']} doc_cand={row['candidate_doc_rank']} "
                    f"type=`{row['metadata_type']}`",
                    f"  - question: {row['question']}",
                    f"  - gold: {row['gold_page_uids']}",
                    f"  - limitation: {row['limitation']}",
                    f"  - gold_evidence: page={row['gold_best_evidence_page']} "
                    f"score={row['gold_best_evidence_score']:.3f} "
                    f"line={row['gold_best_evidence_detail'].get('best_line_score')} "
                    f"window={row['gold_best_evidence_detail'].get('best_window_score')} "
                    f"table={row['gold_best_evidence_detail'].get('table_score')}",
                    f"  - positive_pages={row['positive_page_count']} "
                    f"mean_evidence={row['mean_evidence_score']:.3f} "
                    f"max_evidence={row['max_evidence_score']:.3f}",
                    f"  - graph_financial: active={row['graph_financial_reasoning_active']} "
                    f"reason={row['graph_financial_reasoning_reason']} "
                    f"matches={row['graph_financial_bundle_page_match_count']} "
                    f"bundle={row['graph_financial_bundle_label']}",
                    f"  - candidate_top: {compact_page_rows(row['candidate_top_pages'])}",
                    f"  - top_evidence: {compact_page_rows(row['top_evidence_pages'][:5])}",
                ]
            )

    add_case_section("Recovered", payload["top_recovered"])
    add_case_section("Top Improved Rank", payload["top_improved_rank"])
    add_case_section("Top Worsened Rank", payload["top_worsened_rank"])
    add_case_section("Top Missing In Both", payload["top_missing_in_both"])

    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("n_qids", payload["n_qids"])
    print("movement_counts", payload["movement_counts"])
    print("limitation_counts", payload["limitation_counts"])
    print("saved_md", args.output_md)
    print("saved_json", args.output_json)


if __name__ == "__main__":
    main()
