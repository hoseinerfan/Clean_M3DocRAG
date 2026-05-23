#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PAGE_REFERENCE_TOKEN_RE = r"(?:pages?|pg\.?|p\.|slides?)"
PAGE_REFERENCE_RANGE_RE = (
    rf"\b{PAGE_REFERENCE_TOKEN_RE}\s*(?:no\.?|number|#)?\s*"
    r"(\d{1,4})\s*(?:-|to|and)\s*(\d{1,4})\b"
)
PAGE_REFERENCE_SINGLE_RE = (
    rf"\b{PAGE_REFERENCE_TOKEN_RE}\s*(?:no\.?|number|#)?\s*(\d{{1,4}})\b"
)
DOCUMENT_COVER_TARGET_RE = (
    r"(?:document|report|paper|article|brochure|guidebook|manual|book|chapter|"
    r"newspaper|presentation|slides?|deck)"
)
FIRST_COVER_REFERENCE_RE = (
    r"\b(?:first|opening)\s+page\b"
    r"|\bfront\s+(?:page|cover)\b"
    r"|\btitle\s+page\b"
    r"|\bcover\s+page\b"
    rf"|\b{DOCUMENT_COVER_TARGET_RE}(?:'s)?\s+cover\b"
    rf"|\bcover\s+of\s+(?:the\s+|this\s+|that\s+|each\s+|a\s+|an\s+)?{DOCUMENT_COVER_TARGET_RE}\b"
)


@dataclass
class DocPageCatalog:
    page_counts: dict[str, int]
    page_number_indices: dict[str, dict[int, set[int]]]


@dataclass
class StructuralPage:
    doc_id: str
    page_idx: int
    weight: float
    reason: str

    @property
    def page_uid(self) -> str:
        return page_uid(self.doc_id, self.page_idx)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Structure-aware page reranking for metadata/navigation queries. The script uses "
            "only corpus-side doc_pages metadata to add or boost first/last/explicit-page "
            "candidates inside high-confidence retrieved documents."
        )
    )
    parser.add_argument("--prediction", required=True, help="Baseline prediction JSON.")
    parser.add_argument("--doc-pages-jsonl", required=True, help="Converted doc_pages JSONL.")
    parser.add_argument("--gold", default="", help="Optional MMQA_dev.jsonl for evaluation.")
    parser.add_argument(
        "--strategy",
        choices=["score_boost", "doc_anchor_insert"],
        default="score_boost",
    )
    parser.add_argument(
        "--top-docs",
        type=int,
        default=4,
        help="Top retrieved docs eligible for structural page generation.",
    )
    parser.add_argument("--base-weight", type=float, default=1.0)
    parser.add_argument("--structure-weight", type=float, default=0.65)
    parser.add_argument(
        "--doc-support-power",
        type=float,
        default=0.5,
        help="Power applied to 1/doc_rank support. Smaller values keep lower top-docs active.",
    )
    parser.add_argument(
        "--enable-intents",
        nargs="*",
        default=[
            "explicit_page",
            "page_count",
            "first_cover",
            "last_final",
            "signature",
            "date_release",
            "appendix_reference",
        ],
    )
    parser.add_argument(
        "--final-top-pages",
        type=int,
        default=0,
        help="Cap output page rows. Use 0 to keep all rows plus generated structural pages.",
    )
    parser.add_argument("--hit-k", type=int, default=4)
    parser.add_argument("--output-prediction-json", required=True)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument("--output-md", default="")
    return parser.parse_args()


def load_prediction(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "predictions" in payload and isinstance(
        payload["predictions"], (dict, list)
    ):
        payload = payload["predictions"]
    if isinstance(payload, list):
        rows = {str(row.get("qid", idx)): row for idx, row in enumerate(payload)}
    elif isinstance(payload, dict):
        rows = {str(qid): row for qid, row in payload.items()}
    else:
        raise TypeError(f"Unsupported prediction payload: {path}")
    return rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def maybe_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_doc_page_catalog(path: Path) -> DocPageCatalog:
    counts: dict[str, int] = {}
    page_number_indices: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            doc_id = str(row.get("doc_id", "")).strip()
            page_idx = maybe_int(row.get("page_idx", row.get("page_id", row.get("page_number"))))
            if not doc_id or page_idx is None:
                continue
            counts[doc_id] = max(counts.get(doc_id, 0), page_idx + 1)
            for field_name in ("page_number", "source_page_number"):
                page_number = maybe_int(row.get(field_name))
                if page_number is not None:
                    page_number_indices[doc_id][page_number].add(page_idx)
    return DocPageCatalog(
        page_counts=counts,
        page_number_indices={doc_id: dict(values) for doc_id, values in page_number_indices.items()},
    )


def page_uid(doc_id: str, page_idx: int) -> str:
    return f"{doc_id}_page{int(page_idx)}"


def parse_page_row(row: Any) -> tuple[str, int, float] | None:
    if not isinstance(row, (list, tuple)) or len(row) < 2:
        return None
    doc_id = str(row[0]).strip()
    if not doc_id:
        return None
    page_idx = maybe_int(row[1])
    if page_idx is None:
        return None
    score = 0.0
    if len(row) >= 3:
        try:
            score = float(row[2])
        except (TypeError, ValueError):
            score = 0.0
    return doc_id, page_idx, score


def prediction_rows(row: dict[str, Any]) -> list[list[Any]]:
    rows = row.get("page_retrieval_results", [])
    return rows if isinstance(rows, list) else []


def ranked_pages(rows: list[list[Any]]) -> list[str]:
    out = []
    for row in rows:
        parsed = parse_page_row(row)
        if parsed is None:
            continue
        out.append(page_uid(parsed[0], parsed[1]))
    return out


def ranked_docs(rows: list[list[Any]]) -> list[str]:
    docs = []
    seen = set()
    for row in rows:
        parsed = parse_page_row(row)
        if parsed is None:
            continue
        doc_id = parsed[0]
        if doc_id not in seen:
            seen.add(doc_id)
            docs.append(doc_id)
    return docs


def first_rank(items: list[str], gold: set[str]) -> int | None:
    for idx, item in enumerate(items, start=1):
        if item in gold:
            return idx
    return None


def gold_page_uids(row: dict[str, Any]) -> set[str]:
    metadata = row.get("metadata", {})
    uids = {
        str(value).strip()
        for value in metadata.get("gold_page_uids", [])
        if str(value).strip()
    }
    for ctx in row.get("supporting_context", []):
        doc_id = str(ctx.get("doc_id", "")).strip()
        page_idx = ctx.get("page_idx", ctx.get("page_id"))
        if doc_id and page_idx is not None:
            uids.add(page_uid(doc_id, int(page_idx)))
    return uids


def metadata_value(row: dict[str, Any], field: str) -> str:
    current: Any = row
    for part in field.split("."):
        if not isinstance(current, dict):
            return "UNKNOWN"
        current = current.get(part)
    if current is None:
        return "UNKNOWN"
    return str(current).strip() or "UNKNOWN"


def add_intent(intents: dict[str, float], name: str, weight: float) -> None:
    intents[name] = max(intents.get(name, 0.0), weight)


def add_page_number(raw_pages: list[int], page_number: int) -> None:
    if page_number >= 0 and page_number not in raw_pages:
        raw_pages.append(page_number)


def detect_structural_intents(question: str) -> tuple[dict[str, float], list[int]]:
    query = str(question or "").lower()
    intents: dict[str, float] = {}
    raw_pages: list[int] = []

    for match in re.finditer(PAGE_REFERENCE_RANGE_RE, query):
        start_page = int(match.group(1))
        end_page = int(match.group(2))
        lo, hi = sorted((start_page, end_page))
        if hi - lo <= 20:
            for page_number in range(lo, hi + 1):
                add_page_number(raw_pages, page_number)
    for match in re.finditer(PAGE_REFERENCE_SINGLE_RE, query):
        add_page_number(raw_pages, int(match.group(1)))
    if raw_pages:
        add_intent(intents, "explicit_page", 1.0)

    if re.search(r"\b(how\s+many\s+pages|number\s+of\s+pages|page\s+count|total\s+pages)\b", query):
        add_intent(intents, "page_count", 1.0)
    if re.search(FIRST_COVER_REFERENCE_RE, query):
        add_intent(intents, "first_cover", 0.9)
    if re.search(
        r"\b(last|final|ending|back)\s+page\b|\bend\s+of\s+(the\s+)?(document|report|paper)\b",
        query,
    ):
        add_intent(intents, "last_final", 0.9)
    if re.search(r"\b(signature|signatures|signed|leadership\s+signature)\b", query):
        add_intent(intents, "signature", 1.0)
    if re.search(r"\b(released|release date|published|publication date|dated)\b", query):
        add_intent(intents, "date_release", 0.8)
    if re.search(r"\b(references|bibliography|appendix|appendices|acknowledg(e)?ments?)\b", query):
        add_intent(intents, "appendix_reference", 0.7)
    return intents, raw_pages


def resolve_page_number(catalog: DocPageCatalog, doc_id: str, raw_page: int) -> set[int]:
    page_count = catalog.page_counts.get(doc_id, 0)
    resolved = set(catalog.page_number_indices.get(doc_id, {}).get(raw_page, set()))
    one_based_idx = raw_page - 1 if raw_page > 0 else 0
    if 0 <= one_based_idx < page_count:
        resolved.add(one_based_idx)
    if 0 <= raw_page < page_count:
        resolved.add(raw_page)
    return resolved


def add_structural_page(
    pages: dict[int, StructuralPage],
    doc_id: str,
    page_idx: int,
    weight: float,
    reason: str,
) -> None:
    existing = pages.get(page_idx)
    if existing is None or weight > existing.weight:
        pages[page_idx] = StructuralPage(doc_id=doc_id, page_idx=page_idx, weight=weight, reason=reason)


def structural_pages_for_doc(
    *,
    catalog: DocPageCatalog,
    doc_id: str,
    intents: dict[str, float],
    raw_pages: list[int],
    enabled_intents: set[str],
) -> list[StructuralPage]:
    page_count = catalog.page_counts.get(doc_id, 0)
    if page_count <= 0:
        return []
    pages: dict[int, StructuralPage] = {}

    if "explicit_page" in intents and "explicit_page" in enabled_intents:
        for raw_page in raw_pages:
            for page_idx in resolve_page_number(catalog, doc_id, raw_page):
                add_structural_page(pages, doc_id, page_idx, 1.0, f"explicit_page:{raw_page}")

    if "page_count" in intents and "page_count" in enabled_intents:
        add_structural_page(pages, doc_id, page_count - 1, intents["page_count"], "page_count:last")

    if "first_cover" in intents and "first_cover" in enabled_intents:
        add_structural_page(pages, doc_id, 0, intents["first_cover"], "first_cover")

    if "date_release" in intents and "date_release" in enabled_intents:
        add_structural_page(pages, doc_id, 0, intents["date_release"], "date_release:first")

    if "last_final" in intents and "last_final" in enabled_intents:
        add_structural_page(pages, doc_id, page_count - 1, intents["last_final"], "last_final:last")

    if "signature" in intents and "signature" in enabled_intents:
        for offset, multiplier in [(0, 1.0), (1, 0.7), (2, 0.5)]:
            page_idx = page_count - 1 - offset
            if page_idx >= 0:
                add_structural_page(
                    pages,
                    doc_id,
                    page_idx,
                    intents["signature"] * multiplier,
                    f"signature:late{offset}",
                )

    if "appendix_reference" in intents and "appendix_reference" in enabled_intents:
        late_count = min(5, max(1, int(round(page_count * 0.1))))
        for offset in range(late_count):
            page_idx = page_count - 1 - offset
            if page_idx >= 0:
                add_structural_page(
                    pages,
                    doc_id,
                    page_idx,
                    intents["appendix_reference"] * (1.0 / (offset + 1)),
                    f"appendix_reference:late{offset}",
                )

    return sorted(pages.values(), key=lambda page: (-page.weight, page.page_idx))


def doc_supports(rows: list[list[Any]], top_docs: int, power: float) -> dict[str, float]:
    supports: dict[str, float] = {}
    for rank, doc_id in enumerate(ranked_docs(rows), start=1):
        if top_docs > 0 and rank > top_docs:
            break
        supports[doc_id] = (1.0 / float(rank)) ** max(0.0, float(power))
    return supports


def rerank_score_boost(
    *,
    rows: list[list[Any]],
    structural_by_doc: dict[str, list[StructuralPage]],
    supports: dict[str, float],
    base_weight: float,
    structure_weight: float,
    final_top_pages: int,
) -> tuple[list[list[Any]], list[dict[str, Any]]]:
    scored: dict[str, dict[str, Any]] = {}
    for rank, row in enumerate(rows, start=1):
        parsed = parse_page_row(row)
        if parsed is None:
            continue
        doc_id, page_idx, _score = parsed
        uid = page_uid(doc_id, page_idx)
        scored[uid] = {
            "doc_id": doc_id,
            "page_idx": page_idx,
            "score": float(base_weight) / float(rank),
            "base_rank": rank,
            "structural_reason": "",
            "structural_boost": 0.0,
        }

    generated: list[dict[str, Any]] = []
    for doc_id, pages in structural_by_doc.items():
        doc_support = supports.get(doc_id, 0.0)
        if doc_support <= 0:
            continue
        for page in pages:
            uid = page.page_uid
            boost = float(structure_weight) * doc_support * float(page.weight)
            if boost <= 0:
                continue
            if uid not in scored:
                scored[uid] = {
                    "doc_id": page.doc_id,
                    "page_idx": int(page.page_idx),
                    "score": 0.0,
                    "base_rank": 10**9,
                    "structural_reason": page.reason,
                    "structural_boost": 0.0,
                }
            scored[uid]["score"] += boost
            scored[uid]["structural_boost"] += boost
            if page.reason:
                scored[uid]["structural_reason"] = page.reason
            generated.append(
                {
                    "page_uid": uid,
                    "doc_id": page.doc_id,
                    "page_idx": int(page.page_idx),
                    "reason": page.reason,
                    "boost": boost,
                }
            )

    ranked = sorted(
        scored.values(),
        key=lambda item: (
            -float(item["score"]),
            int(item["base_rank"]),
            str(item["doc_id"]),
            int(item["page_idx"]),
        ),
    )
    if final_top_pages > 0:
        ranked = ranked[:final_top_pages]
    return (
        [[item["doc_id"], int(item["page_idx"]), float(item["score"])] for item in ranked],
        generated,
    )


def rerank_doc_anchor_insert(
    *,
    rows: list[list[Any]],
    structural_by_doc: dict[str, list[StructuralPage]],
    final_top_pages: int,
) -> tuple[list[list[Any]], list[dict[str, Any]]]:
    out: list[list[Any]] = []
    seen_pages: set[str] = set()
    docs_seen: set[str] = set()
    generated: list[dict[str, Any]] = []

    def append_page(doc_id: str, page_idx: int, score: float, *, reason: str = "") -> None:
        uid = page_uid(doc_id, page_idx)
        if uid in seen_pages:
            return
        seen_pages.add(uid)
        out.append([doc_id, int(page_idx), float(score)])
        if reason:
            generated.append(
                {
                    "page_uid": uid,
                    "doc_id": doc_id,
                    "page_idx": int(page_idx),
                    "reason": reason,
                    "boost": score,
                }
            )

    for rank, row in enumerate(rows, start=1):
        parsed = parse_page_row(row)
        if parsed is None:
            continue
        doc_id, page_idx, _score = parsed
        append_page(doc_id, page_idx, 1.0 / float(len(out) + 1))
        if doc_id in docs_seen:
            continue
        docs_seen.add(doc_id)
        for page in structural_by_doc.get(doc_id, []):
            append_page(page.doc_id, page.page_idx, 1.0 / float(len(out) + 1), reason=page.reason)

    if final_top_pages > 0:
        out = out[:final_top_pages]
    return out, generated


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


def summarize_by_intent(per_qid: list[dict[str, Any]], hit_k: int) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_qid:
        intents = row.get("intents") or ["none"]
        for intent in intents:
            grouped[str(intent)].append(row)

    summary = {}
    for intent, rows in sorted(grouped.items()):
        n = len(rows)
        summary[intent] = {
            "n": n,
            f"baseline_page_hit@{hit_k}": sum(
                1
                for row in rows
                if row.get("baseline_first_gold_page_rank") is not None
                and int(row["baseline_first_gold_page_rank"]) <= hit_k
            ),
            f"candidate_page_hit@{hit_k}": sum(
                1
                for row in rows
                if row.get("candidate_first_gold_page_rank") is not None
                and int(row["candidate_first_gold_page_rank"]) <= hit_k
            ),
            "recovered": sum(1 for row in rows if row.get("movement") == "recovered"),
            "lost": sum(1 for row in rows if row.get("movement") == "lost"),
            "improved_rank": sum(1 for row in rows if row.get("movement") == "improved_rank"),
            "worsened_rank": sum(1 for row in rows if row.get("movement") == "worsened_rank"),
        }
    return summary


def md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def write_md(path: Path, summary: dict[str, Any]) -> None:
    lines = ["# Structural Metadata Rerank Summary", ""]
    lines.extend(
        md_table(
            ["metric", "value"],
            [
                ["qid_count", summary["qid_count"]],
                ["structural_qid_count", summary["structural_qid_count"]],
                ["recovered", summary["movement_counts"].get("recovered", 0)],
                ["lost", summary["movement_counts"].get("lost", 0)],
                ["improved_rank", summary["movement_counts"].get("improved_rank", 0)],
                ["worsened_rank", summary["movement_counts"].get("worsened_rank", 0)],
            ],
        )
    )
    lines.extend(["", "## By Intent", ""])
    rows = []
    hit_k = summary["hit_k"]
    for intent, item in summary["by_intent"].items():
        n = item["n"]
        base = item[f"baseline_page_hit@{hit_k}"]
        cand = item[f"candidate_page_hit@{hit_k}"]
        rows.append(
            [
                intent,
                n,
                f"{base / n:.4f}" if n else "0.0000",
                f"{cand / n:.4f}" if n else "0.0000",
                item["recovered"],
                item["lost"],
                item["improved_rank"],
                item["worsened_rank"],
            ]
        )
    lines.extend(
        md_table(
            [
                "intent",
                "n",
                f"baseline page_hit@{hit_k}",
                f"candidate page_hit@{hit_k}",
                "recovered",
                "lost",
                "improved_rank",
                "worsened_rank",
            ],
            rows,
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    prediction = load_prediction(Path(args.prediction))
    catalog = load_doc_page_catalog(Path(args.doc_pages_jsonl))
    gold_by_qid = {}
    if args.gold:
        gold_by_qid = {str(row["qid"]): row for row in read_jsonl(Path(args.gold))}

    enabled_intents = {str(value) for value in args.enable_intents}
    output: dict[str, dict[str, Any]] = {}
    per_qid: list[dict[str, Any]] = []
    movement_counts: Counter[str] = Counter()
    intent_counts: Counter[str] = Counter()
    structural_qid_count = 0
    generated_page_counts: list[int] = []

    for qid, row in sorted(prediction.items()):
        base_rows = prediction_rows(row)
        question = str(row.get("question", ""))
        intents_with_weights, raw_pages = detect_structural_intents(question)
        intents = sorted(intent for intent in intents_with_weights if intent in enabled_intents)
        if intents:
            structural_qid_count += 1
        for intent in intents or ["none"]:
            intent_counts[intent] += 1

        supports = doc_supports(
            base_rows,
            top_docs=int(args.top_docs),
            power=float(args.doc_support_power),
        )
        structural_by_doc = {
            doc_id: structural_pages_for_doc(
                catalog=catalog,
                doc_id=doc_id,
                intents={intent: intents_with_weights[intent] for intent in intents},
                raw_pages=raw_pages,
                enabled_intents=enabled_intents,
            )
            for doc_id in supports
        }
        structural_by_doc = {doc_id: pages for doc_id, pages in structural_by_doc.items() if pages}

        if args.strategy == "doc_anchor_insert":
            candidate_rows, generated = rerank_doc_anchor_insert(
                rows=base_rows,
                structural_by_doc=structural_by_doc,
                final_top_pages=int(args.final_top_pages),
            )
        else:
            candidate_rows, generated = rerank_score_boost(
                rows=base_rows,
                structural_by_doc=structural_by_doc,
                supports=supports,
                base_weight=float(args.base_weight),
                structure_weight=float(args.structure_weight),
                final_top_pages=int(args.final_top_pages),
            )
        generated_page_counts.append(len(generated))

        out_row = dict(row)
        out_row["page_retrieval_results"] = candidate_rows
        out_row["top_retrieved_docs"] = ranked_docs(candidate_rows)[:10]
        out_row["reranker_metadata"] = {
            **(row.get("reranker_metadata", {}) if isinstance(row.get("reranker_metadata"), dict) else {}),
            "structural_metadata_rerank": {
                "strategy": args.strategy,
                "top_docs": int(args.top_docs),
                "base_weight": float(args.base_weight),
                "structure_weight": float(args.structure_weight),
                "doc_support_power": float(args.doc_support_power),
                "intents": intents,
                "raw_page_numbers": raw_pages,
                "generated_page_count": len(generated),
                "generated_pages_preview": generated[:20],
            },
        }
        output[qid] = out_row

        gold_row = gold_by_qid.get(qid)
        if gold_row is None:
            continue
        gold_pages = gold_page_uids(gold_row)
        base_rank = first_rank(ranked_pages(base_rows), gold_pages)
        cand_rank = first_rank(ranked_pages(candidate_rows), gold_pages)
        movement = movement_for_hit(base_rank, cand_rank, int(args.hit_k))
        movement_counts[movement] += 1
        per_qid.append(
            {
                "qid": qid,
                "question": question,
                "metadata_type": metadata_value(gold_row, "metadata.type"),
                "metadata_domain": metadata_value(gold_row, "metadata.domain"),
                "intents": intents,
                "raw_page_numbers": raw_pages,
                "generated_page_count": len(generated),
                "baseline_first_gold_page_rank": base_rank,
                "candidate_first_gold_page_rank": cand_rank,
                "movement": movement,
                "gold_page_uids": sorted(gold_pages),
                "generated_pages_preview": generated[:10],
            }
        )

    summary: dict[str, Any] = {
        "prediction": args.prediction,
        "doc_pages_jsonl": args.doc_pages_jsonl,
        "strategy": args.strategy,
        "top_docs": int(args.top_docs),
        "base_weight": float(args.base_weight),
        "structure_weight": float(args.structure_weight),
        "doc_support_power": float(args.doc_support_power),
        "enabled_intents": sorted(enabled_intents),
        "hit_k": int(args.hit_k),
        "qid_count": len(prediction),
        "structural_qid_count": structural_qid_count,
        "intent_counts": dict(intent_counts),
        "movement_counts": dict(movement_counts),
        "mean_generated_page_count": (
            statistics.fmean(generated_page_counts) if generated_page_counts else None
        ),
        "by_intent": summarize_by_intent(per_qid, int(args.hit_k)) if per_qid else {},
        "per_qid": per_qid,
    }

    output_path = Path(args.output_prediction_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path = Path(args.output_summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_md:
        write_md(Path(args.output_md), summary)

    print(f"saved_prediction: {output_path}")
    print(f"saved_summary: {summary_path}")
    if args.output_md:
        print(f"saved_md: {args.output_md}")
    print(f"qid_count: {len(prediction)}")
    print(f"structural_qid_count: {summary['structural_qid_count']}")
    print(f"intent_counts: {dict(intent_counts)}")
    print(f"movement_counts: {dict(movement_counts)}")


if __name__ == "__main__":
    main()
