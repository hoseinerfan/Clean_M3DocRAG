#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build qid groups from M3DocVQA/MMQA gold annotations. Page-gold groups "
            "are emitted when page_idx/page_id or metadata.gold_page_uids are present; "
            "doc-level groups are always emitted from supporting_context doc_id labels."
        )
    )
    parser.add_argument("--gold", required=True, help="Gold MMQA_<split>.jsonl")
    parser.add_argument("--out-dir", required=True, help="Directory for qid group files")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_qid(row: dict[str, Any]) -> str:
    qid = str(row.get("qid", "")).strip()
    if not qid:
        raise ValueError(f"Gold row missing qid: {row}")
    return qid


def gold_doc_ids(row: dict[str, Any]) -> set[str]:
    docs: set[str] = set()
    for ctx in row.get("supporting_context", []):
        if not isinstance(ctx, dict):
            continue
        doc_id = str(ctx.get("doc_id", "")).strip()
        if doc_id:
            docs.add(doc_id)
    return docs


def page_uid_from_parts(doc_id: str, page_idx: Any) -> str | None:
    if not doc_id or page_idx is None:
        return None
    try:
        return f"{doc_id}_page{int(page_idx)}"
    except (TypeError, ValueError):
        return None


def doc_id_from_page_uid(uid: str) -> str:
    marker = "_page"
    if marker not in uid:
        return ""
    return uid.rsplit(marker, 1)[0]


def gold_page_uids(row: dict[str, Any]) -> set[str]:
    pages: set[str] = set()
    metadata = row.get("metadata", {})
    if isinstance(metadata, dict):
        for value in metadata.get("gold_page_uids", []):
            uid = str(value).strip()
            if uid:
                pages.add(uid)
    for ctx in row.get("supporting_context", []):
        if not isinstance(ctx, dict):
            continue
        doc_id = str(ctx.get("doc_id", "")).strip()
        page_idx = ctx.get("page_idx", ctx.get("page_id"))
        uid = page_uid_from_parts(doc_id, page_idx)
        if uid:
            pages.add(uid)
    return pages


def write_qids(path: Path, qids: list[str]) -> None:
    path.write_text("".join(f"{qid}\n" for qid in sorted(qids)), encoding="utf-8")


def main() -> None:
    args = parse_args()
    gold_path = Path(args.gold)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list[str]] = {
        "single_gold_doc": [],
        "multi_gold_doc": [],
        "no_gold_doc": [],
        "same_doc_page_gold": [],
        "cross_doc_page_gold": [],
        "no_page_gold": [],
    }
    page_count_dist: Counter[int] = Counter()
    page_doc_count_dist: Counter[int] = Counter()
    gold_doc_count_dist: Counter[int] = Counter()

    for row in load_jsonl(gold_path):
        qid = row_qid(row)
        docs = gold_doc_ids(row)
        pages = gold_page_uids(row)
        page_docs = {doc_id_from_page_uid(uid) for uid in pages}
        page_docs.discard("")

        gold_doc_count_dist[len(docs)] += 1
        page_count_dist[len(pages)] += 1
        page_doc_count_dist[len(page_docs)] += 1

        if len(docs) == 0:
            groups["no_gold_doc"].append(qid)
        elif len(docs) == 1:
            groups["single_gold_doc"].append(qid)
        else:
            groups["multi_gold_doc"].append(qid)

        if not pages:
            groups["no_page_gold"].append(qid)
        elif len(page_docs) <= 1:
            groups["same_doc_page_gold"].append(qid)
        else:
            groups["cross_doc_page_gold"].append(qid)

    for name, qids in groups.items():
        write_qids(out_dir / f"{name}.qids.txt", qids)

    summary = {
        "gold_file": str(gold_path),
        "qid_count": sum(len(qids) for qids in [groups["single_gold_doc"], groups["multi_gold_doc"], groups["no_gold_doc"]]),
        "groups": {name: len(qids) for name, qids in groups.items()},
        "gold_doc_count_dist": dict(sorted(gold_doc_count_dist.items())),
        "gold_page_count_dist": dict(sorted(page_count_dist.items())),
        "gold_page_doc_count_dist": dict(sorted(page_doc_count_dist.items())),
        "note": (
            "Use page groups for true same-doc/cross-doc page-gold evaluation when page gold exists. "
            "If no_page_gold equals qid_count, use single_gold_doc vs multi_gold_doc as a doc-level proxy only."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved_qid_groups={out_dir}")


if __name__ == "__main__":
    main()
