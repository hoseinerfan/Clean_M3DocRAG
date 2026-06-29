#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_mmqa_pseudo_page_labels as ppl
from scripts import export_mmqa_evidence_metadata as em


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a direct_plus_context_weak ablation label set by appending at most "
            "one weak page per bridge/context support document to an existing direct "
            "pseudo-page label set. Weak pages are selected from a base ranking."
        )
    )
    parser.add_argument("--gold", required=True, help="Original MMQA split JSONL.")
    parser.add_argument("--direct-labels-jsonl", required=True)
    parser.add_argument("--direct-augmented-gold-jsonl", required=True)
    parser.add_argument("--base-prediction-json", required=True)
    parser.add_argument("--mmqa-tables-jsonl", required=True)
    parser.add_argument("--output-labels-jsonl", required=True)
    parser.add_argument("--output-augmented-gold-jsonl", required=True)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument("--context-tier", default="context_weak")
    parser.add_argument("--context-score", type=float, default=0.25)
    parser.add_argument("--candidate-top-k", type=int, default=1000)
    parser.add_argument("--max-context-pages-per-doc", type=int, default=1)
    parser.add_argument(
        "--include-unlabeled-qids",
        action="store_true",
        help=(
            "Allow qids without direct pseudo labels to receive weak context pages. "
            "Default is off to keep this ablation anchored to direct evidence labels."
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_by_qid(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
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
        raise TypeError(f"Prediction JSON must be object or list: {path}")
    out: dict[str, dict[str, Any]] = {}
    for raw_key, row in iterator:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("qid") or raw_key).strip()
        if qid:
            out[qid] = row
    return out


def support_doc_ids(row: dict[str, Any]) -> list[str]:
    docs: list[str] = []
    seen: set[str] = set()
    for ctx in row.get("supporting_context", []) or []:
        if not isinstance(ctx, dict):
            continue
        doc_id = str(ctx.get("doc_id", "") or "").strip()
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            docs.append(doc_id)
    return docs


def page_uid(doc_id: str, page_idx: int) -> str:
    return f"{doc_id}_page{int(page_idx)}"


def prediction_records(pred_row: dict[str, Any] | None, *, top_k: int) -> list[dict[str, Any]]:
    if not isinstance(pred_row, dict):
        return []
    rows = pred_row.get("page_retrieval_results", pred_row.get("retrieval_results", pred_row.get("results", [])))
    records: list[dict[str, Any]] = []
    for rank, item in enumerate(rows[: int(top_k)], start=1):
        doc_id = ""
        page_idx: int | None = None
        score = 0.0
        uid = ""
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            doc_id = str(item[0])
            try:
                page_idx = int(item[1])
            except (TypeError, ValueError):
                page_idx = None
            if len(item) >= 3:
                try:
                    score = float(item[2])
                except (TypeError, ValueError):
                    score = 0.0
        elif isinstance(item, dict):
            uid = str(item.get("page_uid") or item.get("uid") or "")
            doc_id = str(item.get("doc_id") or item.get("docid") or "")
            raw_page = item.get("page_idx", item.get("page_id", item.get("page")))
            if raw_page is not None:
                try:
                    page_idx = int(raw_page)
                except (TypeError, ValueError):
                    page_idx = None
            if not doc_id and "_page" in uid:
                doc_id = uid.rsplit("_page", 1)[0]
            if page_idx is None and "_page" in uid:
                try:
                    page_idx = int(uid.rsplit("_page", 1)[1])
                except ValueError:
                    page_idx = None
            try:
                score = float(item.get("score", item.get("retrieval_score", item.get("base_score", 0.0))))
            except (TypeError, ValueError):
                score = 0.0
        if doc_id and page_idx is not None:
            uid = uid or page_uid(doc_id, page_idx)
            records.append(
                {
                    "page_uid": uid,
                    "doc_id": doc_id,
                    "page_idx": int(page_idx),
                    "base_rank": int(rank),
                    "base_score": float(score),
                }
            )
    return records


def direct_page_uids(row: dict[str, Any] | None) -> list[str]:
    if not row:
        return []
    values = row.get("pseudo_gold_page_uids") or row.get("gold_page_uids") or []
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        uid = str(value).strip()
        if uid and uid not in seen:
            seen.add(uid)
            out.append(uid)
    return out


def augmented_page_uids(row: dict[str, Any] | None) -> list[str]:
    if not row:
        return []
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    values = metadata.get("gold_page_uids") or metadata.get("pseudo_gold_page_uids") or []
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        uid = str(value).strip()
        if uid and uid not in seen:
            seen.add(uid)
            out.append(uid)
    return out


def append_unique(values: list[str], additions: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values + additions:
        uid = str(value).strip()
        if uid and uid not in seen:
            seen.add(uid)
            out.append(uid)
    return out


def context_page_record(record: dict[str, Any], *, context_tier: str, context_score: float) -> dict[str, Any]:
    return {
        "page_uid": record["page_uid"],
        "doc_id": record["doc_id"],
        "page_idx": int(record["page_idx"]),
        "score": float(context_score),
        "confidence": "weak",
        "supervision_tier": context_tier,
        "weak_context_source": "bridge_support_doc_ranked_candidate",
        "base_rank": int(record["base_rank"]),
        "base_score": float(record["base_score"]),
        "exact_matches": [],
        "fuzzy_matches": [],
        "positive_exact_matches": [],
        "positive_fuzzy_matches": [],
        "diagnostic_exact_matches": [],
        "diagnostic_fuzzy_matches": [],
        "verification_matches": [],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    gold_rows = read_jsonl(Path(args.gold))
    direct_labels = load_by_qid(Path(args.direct_labels_jsonl))
    direct_augmented = load_by_qid(Path(args.direct_augmented_gold_jsonl))
    base_pred = load_prediction(Path(args.base_prediction_json))
    tables_by_id = ppl.load_by_id(args.mmqa_tables_jsonl)

    output_label_rows: list[dict[str, Any]] = []
    output_aug_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    weak_pages_by_qid_hist: Counter[int] = Counter()

    for gold_row in gold_rows:
        qid = str(gold_row.get("qid", "")).strip()
        if not qid:
            continue
        direct_label = json.loads(json.dumps(direct_labels.get(qid, {"qid": qid})))
        aug_row = json.loads(json.dumps(direct_augmented.get(qid, gold_row)))
        metadata = aug_row.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
            aug_row["metadata"] = metadata

        direct_uids = direct_page_uids(direct_label) or augmented_page_uids(aug_row)
        direct_uid_set = set(direct_uids)
        if direct_uids:
            counts["direct_labeled_qids"] += 1
        elif not bool(args.include_unlabeled_qids):
            counts["unlabeled_qids_skipped_for_context"] += 1
            output_label_rows.append(direct_label)
            output_aug_rows.append(aug_row)
            weak_pages_by_qid_hist[0] += 1
            continue

        units = em.evidence_units_for_row(gold_row, tables_by_id=tables_by_id)
        evidence_doc_ids = {str(unit.get("doc_id", "") or "").strip() for unit in units if str(unit.get("doc_id", "") or "").strip()}
        bridge_docs = [doc_id for doc_id in support_doc_ids(gold_row) if doc_id not in evidence_doc_ids]
        counts["bridge_context_doc_refs"] += len(bridge_docs)

        records_by_doc: dict[str, list[dict[str, Any]]] = {}
        for record in prediction_records(base_pred.get(qid), top_k=int(args.candidate_top_k)):
            records_by_doc.setdefault(str(record["doc_id"]), []).append(record)

        context_records: list[dict[str, Any]] = []
        for doc_id in bridge_docs:
            available = [record for record in records_by_doc.get(doc_id, []) if record["page_uid"] not in direct_uid_set]
            if not available:
                counts["bridge_context_doc_refs_missing_candidate"] += 1
                continue
            limit = max(0, int(args.max_context_pages_per_doc))
            for record in available[:limit]:
                context_records.append(record)
                direct_uid_set.add(str(record["page_uid"]))
                counts["bridge_context_doc_refs_with_candidate"] += 1

        context_uids = [str(record["page_uid"]) for record in context_records]
        if context_uids:
            counts["context_augmented_qids"] += 1
            counts["context_weak_page_count"] += len(context_uids)
        weak_pages_by_qid_hist[len(context_uids)] += 1

        combined_uids = append_unique(direct_uids, context_uids)
        direct_label["pseudo_gold_page_uids"] = combined_uids
        direct_label["context_weak_page_uids"] = context_uids
        direct_label["context_weak_bridge_doc_ids"] = bridge_docs
        direct_label["context_weak_added_count"] = len(context_uids)
        pages = list(direct_label.get("pseudo_gold_pages") or [])
        existing_page_uids = {str(page.get("page_uid", "")) for page in pages if isinstance(page, dict)}
        for record in context_records:
            if record["page_uid"] not in existing_page_uids:
                pages.append(
                    context_page_record(
                        record,
                        context_tier=str(args.context_tier),
                        context_score=float(args.context_score),
                    )
                )
        direct_label["pseudo_gold_pages"] = pages

        metadata["pseudo_gold_page_uids"] = combined_uids
        metadata["gold_page_uids"] = combined_uids
        metadata["pseudo_gold_direct_page_uids"] = direct_uids
        metadata["pseudo_gold_context_weak_page_uids"] = context_uids
        metadata["pseudo_gold_page_label_source"] = "mmqa_direct_evidence_plus_context_weak"
        metadata["pseudo_gold_context_weak_source"] = "bridge_support_doc_ranked_candidate"
        metadata["pseudo_gold_qid_supervision_tier"] = (
            "direct_plus_context_weak" if context_uids else "direct"
        )
        tiers = dict(metadata.get("pseudo_gold_page_supervision_tiers") or {})
        for uid in direct_uids:
            tiers[str(uid)] = tiers.get(str(uid), "direct")
        for uid in context_uids:
            tiers[str(uid)] = str(args.context_tier)
        metadata["pseudo_gold_page_supervision_tiers"] = tiers
        scores = dict(metadata.get("pseudo_gold_page_label_scores") or {})
        for uid in context_uids:
            scores[str(uid)] = float(args.context_score)
        metadata["pseudo_gold_page_label_scores"] = scores

        output_label_rows.append(direct_label)
        output_aug_rows.append(aug_row)

    summary = {
        "gold": str(args.gold),
        "direct_labels_jsonl": str(args.direct_labels_jsonl),
        "direct_augmented_gold_jsonl": str(args.direct_augmented_gold_jsonl),
        "base_prediction_json": str(args.base_prediction_json),
        "qid_count": len(output_aug_rows),
        "context_tier": str(args.context_tier),
        "context_score": float(args.context_score),
        "max_context_pages_per_doc": int(args.max_context_pages_per_doc),
        "candidate_top_k": int(args.candidate_top_k),
        "counts": dict(sorted(counts.items())),
        "context_weak_pages_per_qid_hist": {str(k): v for k, v in sorted(weak_pages_by_qid_hist.items())},
        "output_labels_jsonl": str(args.output_labels_jsonl),
        "output_augmented_gold_jsonl": str(args.output_augmented_gold_jsonl),
    }

    write_jsonl(Path(args.output_labels_jsonl), output_label_rows)
    write_jsonl(Path(args.output_augmented_gold_jsonl), output_aug_rows)
    Path(args.output_summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_summary_json).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"saved_labels={args.output_labels_jsonl}")
    print(f"saved_augmented_gold={args.output_augmented_gold_jsonl}")
    print(f"saved_summary={args.output_summary_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
