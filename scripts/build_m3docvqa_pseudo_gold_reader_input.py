#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an oracle-style M3DocVQA reader input where MMQA-derived "
            "pseudo-gold pages are placed at the top of page_retrieval_results."
        )
    )
    parser.add_argument(
        "--augmented-gold",
        required=True,
        help="Augmented MMQA JSONL with metadata.gold_page_uids.",
    )
    parser.add_argument(
        "--original-gold",
        default="",
        help=(
            "Optional original MMQA JSONL used to read supporting_context. "
            "If omitted, supporting_context is read from --augmented-gold."
        ),
    )
    parser.add_argument(
        "--base-prediction",
        default="",
        help=(
            "Optional base retrieval prediction JSON. If supplied with --fill-from-base, "
            "non-gold base pages are appended after pseudo-gold pages up to --top-pages."
        ),
    )
    parser.add_argument(
        "--top-pages",
        type=int,
        default=4,
        help="Reader page budget to construct. Default: 4.",
    )
    parser.add_argument(
        "--fill-from-base",
        action="store_true",
        help="Fill remaining reader slots from the base prediction after pseudo-gold pages.",
    )
    parser.add_argument(
        "--include-unlabeled-with-base",
        action="store_true",
        help=(
            "For unlabeled QIDs, keep base top pages when --fill-from-base is used. "
            "By default unlabeled QIDs are excluded."
        ),
    )
    parser.add_argument(
        "--require-all-support-docs-covered",
        action="store_true",
        help=(
            "Keep only QIDs where the pseudo-page labels include at least one page "
            "from every original MMQA supporting document."
        ),
    )
    parser.add_argument(
        "--require-pseudo-pages-match-support-docs",
        action="store_true",
        help=(
            "Keep only QIDs where pseudo-page labels cover exactly the original "
            "support-document set and the number of pseudo pages equals the number "
            "of original supporting documents. This is stricter than "
            "--require-all-support-docs-covered."
        ),
    )
    parser.add_argument(
        "--output-prediction-json",
        required=True,
        help="Output prediction JSON for run_m3docvqa_external_retrieval_qa.py.",
    )
    parser.add_argument(
        "--output-filtered-gold",
        default="",
        help="Optional JSONL gold file containing only the QIDs written to the prediction JSON.",
    )
    parser.add_argument(
        "--output-summary",
        default="",
        help="Optional summary JSON path.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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

    out: dict[str, dict[str, Any]] = {}
    for raw_key, row in iterator:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("qid") or raw_key).strip()
        if qid:
            out[qid] = row
    return out


def gold_page_uids(row: dict[str, Any]) -> list[str]:
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    values = (
        metadata.get("gold_page_uids")
        or metadata.get("pseudo_gold_page_uids")
        or row.get("gold_page_uids")
        or row.get("pseudo_gold_page_uids")
        or []
    )
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        uid = str(value).strip()
        if uid and uid not in seen:
            seen.add(uid)
            out.append(uid)
    return out


def support_doc_ids(row: dict[str, Any]) -> set[str]:
    docs: set[str] = set()
    for ctx in row.get("supporting_context", []):
        if isinstance(ctx, dict) and ctx.get("doc_id"):
            docs.add(str(ctx["doc_id"]).strip())
    return {doc_id for doc_id in docs if doc_id}


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


def parse_prediction_row(row: Any) -> tuple[str, int, float | None] | None:
    if isinstance(row, (list, tuple)) and len(row) >= 2:
        doc_id = str(row[0]).strip()
        if not doc_id:
            return None
        try:
            page_idx = int(row[1])
        except (TypeError, ValueError):
            return None
        score = None
        if len(row) >= 3 and row[2] is not None:
            try:
                score = float(row[2])
            except (TypeError, ValueError):
                score = None
        return doc_id, page_idx, score

    if isinstance(row, dict):
        uid = str(row.get("page_uid", "")).strip()
        if uid:
            parsed = parse_page_uid(uid)
            if parsed is None:
                return None
            doc_id, page_idx = parsed
        else:
            doc_id = str(row.get("doc_id", row.get("docid", row.get("document_id", "")))).strip()
            page_value = row.get("page_idx", row.get("page_id", row.get("page")))
            if not doc_id or page_value is None:
                return None
            try:
                page_idx = int(page_value)
            except (TypeError, ValueError):
                return None
        score = None
        score_value = row.get("score", row.get("retrieval_score", row.get("fused_page_score")))
        if score_value is not None:
            try:
                score = float(score_value)
            except (TypeError, ValueError):
                score = None
        return doc_id, page_idx, score

    return None


def prediction_rows(row: dict[str, Any] | None) -> list[Any]:
    if not isinstance(row, dict):
        return []
    rows = row.get("page_retrieval_results", row.get("retrieval_results", row.get("results", [])))
    return rows if isinstance(rows, list) else []


def ranked_docs(rows: list[list[Any]], limit: int = 10) -> list[str]:
    docs: list[str] = []
    seen: set[str] = set()
    for row in rows:
        doc_id = str(row[0]).strip() if row else ""
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            docs.append(doc_id)
            if len(docs) >= limit:
                break
    return docs


def make_gold_rows(gold_uids: list[str], top_pages: int) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for rank, uid in enumerate(gold_uids[: max(top_pages, 0)], start=1):
        parsed = parse_page_uid(uid)
        if parsed is None:
            continue
        doc_id, page_idx = parsed
        # The reader uses row order, not score. A large descending score makes
        # the oracle ranking explicit if the file is audited later.
        rows.append([doc_id, page_idx, float(1_000_000 - rank)])
    return rows


def append_base_fill(
    *,
    rows: list[list[Any]],
    base_row: dict[str, Any] | None,
    top_pages: int,
) -> list[list[Any]]:
    filled = list(rows)
    seen = {page_uid(str(row[0]), int(row[1])) for row in filled}
    for raw in prediction_rows(base_row):
        parsed = parse_prediction_row(raw)
        if parsed is None:
            continue
        doc_id, page_idx, score = parsed
        uid = page_uid(doc_id, page_idx)
        if uid in seen:
            continue
        seen.add(uid)
        filled.append([doc_id, page_idx, float(score) if score is not None else 0.0])
        if len(filled) >= top_pages:
            break
    return filled


def write_filtered_gold(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.top_pages < 1:
        raise ValueError("--top-pages must be >= 1")
    if args.fill_from_base and not args.base_prediction:
        raise ValueError("--fill-from-base requires --base-prediction")
    if args.include_unlabeled_with_base and not args.fill_from_base:
        raise ValueError("--include-unlabeled-with-base requires --fill-from-base")

    gold_rows = load_jsonl(Path(args.augmented_gold))
    support_rows_by_qid = {
        str(row.get("qid", "")).strip(): row
        for row in (load_jsonl(Path(args.original_gold)) if args.original_gold else gold_rows)
        if str(row.get("qid", "")).strip()
    }
    base = load_prediction(Path(args.base_prediction)) if args.base_prediction else {}

    output: dict[str, dict[str, Any]] = {}
    filtered_gold_rows: list[dict[str, Any]] = []
    stats = {
        "gold_qids": len(gold_rows),
        "labeled_qids": 0,
        "written_qids": 0,
        "support_complete_labeled_qids": 0,
        "support_exact_match_labeled_qids": 0,
        "skipped_incomplete_support_doc_coverage": 0,
        "skipped_support_doc_count_mismatch": 0,
        "unlabeled_written_with_base": 0,
        "gold_page_count": 0,
        "mean_gold_pages_per_written_qid": 0.0,
        "filled_from_base_qids": 0,
        "shorter_than_top_pages_qids": 0,
        "top_pages": int(args.top_pages),
        "fill_from_base": bool(args.fill_from_base),
        "include_unlabeled_with_base": bool(args.include_unlabeled_with_base),
        "require_all_support_docs_covered": bool(args.require_all_support_docs_covered),
        "require_pseudo_pages_match_support_docs": bool(args.require_pseudo_pages_match_support_docs),
        "support_doc_count_hist": {},
        "pseudo_page_count_hist": {},
        "written_support_doc_count_hist": {},
        "written_pseudo_page_count_hist": {},
    }

    gold_counts: list[int] = []
    support_doc_count_hist: dict[int, int] = {}
    pseudo_page_count_hist: dict[int, int] = {}
    written_support_doc_count_hist: dict[int, int] = {}
    written_pseudo_page_count_hist: dict[int, int] = {}
    for row in gold_rows:
        qid = str(row.get("qid", "")).strip()
        if not qid:
            continue
        gold_uids = gold_page_uids(row)
        support_row = support_rows_by_qid.get(qid, row)
        support_docs = support_doc_ids(support_row)
        pseudo_docs = {
            parsed[0]
            for uid in gold_uids
            if (parsed := parse_page_uid(uid)) is not None
        }
        support_doc_count_hist[len(support_docs)] = support_doc_count_hist.get(len(support_docs), 0) + 1
        pseudo_page_count_hist[len(gold_uids)] = pseudo_page_count_hist.get(len(gold_uids), 0) + 1
        if gold_uids:
            stats["labeled_qids"] += 1
            stats["gold_page_count"] += len(gold_uids)
            if support_docs and support_docs.issubset(pseudo_docs):
                stats["support_complete_labeled_qids"] += 1
            if support_docs and support_docs == pseudo_docs and len(gold_uids) == len(support_docs):
                stats["support_exact_match_labeled_qids"] += 1
        elif not args.include_unlabeled_with_base:
            continue

        if args.require_all_support_docs_covered and not support_docs.issubset(pseudo_docs):
            stats["skipped_incomplete_support_doc_coverage"] += 1
            continue
        if args.require_pseudo_pages_match_support_docs and not (
            support_docs and support_docs == pseudo_docs and len(gold_uids) == len(support_docs)
        ):
            stats["skipped_support_doc_count_mismatch"] += 1
            continue

        page_rows = make_gold_rows(gold_uids, args.top_pages)
        if args.fill_from_base:
            before_fill_count = len(page_rows)
            page_rows = append_base_fill(rows=page_rows, base_row=base.get(qid), top_pages=args.top_pages)
            if len(page_rows) > before_fill_count:
                stats["filled_from_base_qids"] += 1

        if not page_rows:
            continue
        if len(page_rows) < args.top_pages:
            stats["shorter_than_top_pages_qids"] += 1

        gold_counts.append(len(gold_uids))
        written_support_doc_count_hist[len(support_docs)] = (
            written_support_doc_count_hist.get(len(support_docs), 0) + 1
        )
        written_pseudo_page_count_hist[len(gold_uids)] = (
            written_pseudo_page_count_hist.get(len(gold_uids), 0) + 1
        )
        filtered_gold_rows.append(row)
        output[qid] = {
            "qid": qid,
            "question": str(row.get("question", "")).strip(),
            "pred_answer": "",
            "page_retrieval_results": page_rows,
            "selected_page_retrieval_results": page_rows[: args.top_pages],
            "source_page_retrieval_results": page_rows,
            "top_retrieved_docs": ranked_docs(page_rows),
            "source_top_retrieved_docs": ranked_docs(page_rows),
            "time_retrieval": None,
            "time_qa": None,
            "oracle_pseudo_gold_page_uids": gold_uids,
        }

        if not gold_uids:
            stats["unlabeled_written_with_base"] += 1

    stats["written_qids"] = len(output)
    if gold_counts:
        stats["mean_gold_pages_per_written_qid"] = sum(gold_counts) / len(gold_counts)
    stats["support_doc_count_hist"] = {str(k): v for k, v in sorted(support_doc_count_hist.items())}
    stats["pseudo_page_count_hist"] = {str(k): v for k, v in sorted(pseudo_page_count_hist.items())}
    stats["written_support_doc_count_hist"] = {
        str(k): v for k, v in sorted(written_support_doc_count_hist.items())
    }
    stats["written_pseudo_page_count_hist"] = {
        str(k): v for k, v in sorted(written_pseudo_page_count_hist.items())
    }

    output_path = Path(args.output_prediction_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    if args.output_filtered_gold:
        write_filtered_gold(Path(args.output_filtered_gold), filtered_gold_rows)

    if args.output_summary:
        summary_path = Path(args.output_summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    print(f"saved_prediction={output_path}")
    if args.output_filtered_gold:
        print(f"saved_filtered_gold={args.output_filtered_gold}")
    if args.output_summary:
        print(f"saved_summary={args.output_summary}")
    for key, value in stats.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
