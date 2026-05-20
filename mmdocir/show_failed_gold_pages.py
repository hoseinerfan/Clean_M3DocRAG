#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Show qids and gold page images for failed exact/compact MaxSim retrieval cases. "
            "Use --where filters over MMQA metadata fields, for example "
            "metadata.type=meta-data or metadata.content_type~=Table."
        )
    )
    parser.add_argument("--data-root", required=True, help="Converted dataset root containing MMQA_dev.jsonl.")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--exact-pred", default="", help="Exact/full MaxSim prediction JSON.")
    parser.add_argument("--compact-pred", default="", help="Compact MaxSim prediction JSON.")
    parser.add_argument("--exact-label", default="exactmaxsim")
    parser.add_argument("--compact-label", default="compactmaxsim")
    parser.add_argument(
        "--where",
        action="append",
        default=[],
        help=(
            "Filter expression. FIELD=VALUE requires exact normalized value match; "
            "FIELD~=VALUE requires case-insensitive substring match. Repeat for AND filters."
        ),
    )
    parser.add_argument(
        "--fail-mode",
        default="both_page_miss",
        choices=[
            "both_page_miss",
            "exact_page_miss",
            "compact_page_miss",
            "either_page_miss",
            "compact_loses_page",
            "compact_recovers_page",
        ],
    )
    parser.add_argument("--hit-k", type=int, default=4)
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--top-retrieved", type=int, default=3)
    parser.add_argument("--output-md", default="")
    parser.add_argument(
        "--copy-gold-pages-dir",
        default="",
        help="Optional directory where gold page images are copied with qid-prefixed filenames.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_prediction(path: str) -> dict[str, dict]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Prediction must be a JSON object keyed by qid: {path}")
    return {str(qid): row for qid, row in payload.items()}


def get_path(row: dict, path: str) -> Any:
    current: Any = row
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def normalize_values(value: Any) -> list[str]:
    if value is None:
        return ["UNKNOWN"]
    if isinstance(value, list):
        values = [str(item).strip() for item in value if str(item).strip()]
        return values or ["UNKNOWN"]
    text = str(value).strip()
    return [text or "UNKNOWN"]


def parse_filter(raw: str) -> tuple[str, str, str]:
    if "~=" in raw:
        field, value = raw.split("~=", 1)
        op = "~="
    elif "=" in raw:
        field, value = raw.split("=", 1)
        op = "="
    else:
        raise ValueError(f"--where must contain '=' or '~=': {raw!r}")
    field = field.strip()
    value = value.strip()
    if not field or not value:
        raise ValueError(f"Invalid --where filter: {raw!r}")
    return field, op, value


def row_matches(row: dict, filters: list[tuple[str, str, str]]) -> bool:
    for field, op, expected in filters:
        values = normalize_values(get_path(row, field))
        if op == "=":
            if expected not in values:
                return False
        elif op == "~=":
            expected_lower = expected.lower()
            if not any(expected_lower in value.lower() for value in values):
                return False
        else:
            raise ValueError(op)
    return True


def page_uid(doc_id: str, page_idx: int) -> str:
    return f"{doc_id}_page{int(page_idx)}"


def gold_page_uids(row: dict) -> list[str]:
    metadata = row.get("metadata", {})
    uids = [
        str(value).strip()
        for value in metadata.get("gold_page_uids", [])
        if str(value).strip()
    ]
    seen = set(uids)
    for ctx in row.get("supporting_context", []):
        doc_id = str(ctx.get("doc_id", "")).strip()
        page_idx = ctx.get("page_idx", ctx.get("page_id"))
        if doc_id and page_idx is not None:
            uid = page_uid(doc_id, int(page_idx))
            if uid not in seen:
                seen.add(uid)
                uids.append(uid)
    return uids


def ranked_pages(pred_row: dict | None) -> list[str]:
    if pred_row is None:
        return []
    pages = []
    for item in pred_row.get("page_retrieval_results", []):
        if isinstance(item, list) and len(item) >= 2:
            pages.append(page_uid(str(item[0]), int(item[1])))
    return pages


def ranked_docs(pred_row: dict | None) -> list[str]:
    docs = []
    seen = set()
    if pred_row is None:
        return docs
    for item in pred_row.get("page_retrieval_results", []):
        if not isinstance(item, list) or not item:
            continue
        doc_id = str(item[0])
        if doc_id not in seen:
            seen.add(doc_id)
            docs.append(doc_id)
    return docs


def first_rank(ranked: list[str], gold: set[str]) -> int | None:
    for idx, item in enumerate(ranked, start=1):
        if item in gold:
            return idx
    return None


def score(pred_row: dict | None, gold_row: dict, hit_k: int) -> dict:
    gold_pages = set(gold_page_uids(gold_row))
    gold_docs = {
        str(ctx.get("doc_id", "")).strip()
        for ctx in gold_row.get("supporting_context", [])
        if str(ctx.get("doc_id", "")).strip()
    }
    page_rank = first_rank(ranked_pages(pred_row), gold_pages)
    doc_rank = first_rank(ranked_docs(pred_row), gold_docs)
    return {
        "page_rank": page_rank,
        "doc_rank": doc_rank,
        "page_hit": page_rank is not None and page_rank <= hit_k,
        "doc_hit": doc_rank is not None and doc_rank <= hit_k,
    }


def is_failure(mode: str, exact: dict, compact: dict) -> bool:
    if mode == "both_page_miss":
        return not exact["page_hit"] and not compact["page_hit"]
    if mode == "exact_page_miss":
        return not exact["page_hit"]
    if mode == "compact_page_miss":
        return not compact["page_hit"]
    if mode == "either_page_miss":
        return not exact["page_hit"] or not compact["page_hit"]
    if mode == "compact_loses_page":
        return exact["page_hit"] and not compact["page_hit"]
    if mode == "compact_recovers_page":
        return not exact["page_hit"] and compact["page_hit"]
    raise ValueError(mode)


def rank_sort_value(rank: int | None) -> int:
    return 10**9 if rank is None else int(rank)


def load_page_map(path: Path) -> dict[str, dict]:
    pages = {}
    for row in read_jsonl(path):
        uid = str(row.get("page_uid", "")).strip()
        if uid:
            pages[uid] = row
    return pages


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_")[:180] or "item"


def page_entry(data_root: Path, page_map: dict[str, dict], uid: str) -> dict:
    row = page_map.get(uid, {})
    image_path = str(row.get("image_path", ""))
    abs_path = data_root / image_path if image_path else None
    return {
        "page_uid": uid,
        "doc_id": row.get("doc_id", uid.rsplit("_page", 1)[0]),
        "page_idx": row.get("page_idx"),
        "page_number": row.get("page_number"),
        "image_path": str(abs_path) if abs_path else "",
        "row": row,
    }


def copy_gold_pages(copy_dir: Path, qid: str, pages: list[dict]) -> dict[str, str]:
    copy_dir.mkdir(parents=True, exist_ok=True)
    copied = {}
    for idx, page in enumerate(pages, start=1):
        src = Path(page["image_path"])
        if not src.exists():
            continue
        suffix = src.suffix or ".jpg"
        dest = copy_dir / f"{safe_name(qid)}__gold{idx:02d}__{safe_name(page['page_uid'])}{suffix}"
        shutil.copy2(src, dest)
        copied[page["page_uid"]] = str(dest)
    return copied


def retrieved_rows(pred_row: dict | None, data_root: Path, page_map: dict[str, dict], top_n: int) -> list[dict]:
    if pred_row is None or top_n <= 0:
        return []
    rows = []
    for rank, item in enumerate(pred_row.get("page_retrieval_results", [])[:top_n], start=1):
        if not isinstance(item, list) or len(item) < 2:
            continue
        uid = page_uid(str(item[0]), int(item[1]))
        entry = page_entry(data_root, page_map, uid)
        entry["rank"] = rank
        entry["score"] = item[2] if len(item) > 2 else None
        rows.append(entry)
    return rows


def display_metadata(metadata: dict) -> dict:
    keep_keys = [
        "type",
        "domain",
        "repo_slug",
        "query_types",
        "query_format",
        "content_type",
        "language",
        "source",
        "source_query_id",
        "query_type_for_generation",
    ]
    return {key: metadata[key] for key in keep_keys if key in metadata}


def markdown_for_examples(examples: list[dict], exact_label: str, compact_label: str) -> str:
    lines = [f"# Failed Gold Page Examples", ""]
    for idx, item in enumerate(examples, start=1):
        lines.extend(
            [
                f"## {idx}. `{item['qid']}`",
                "",
                f"Question: {item['question']}",
                "",
                f"Metadata: `{json.dumps(display_metadata(item['metadata']), ensure_ascii=False)}`",
                "",
                (
                    f"Ranks: {exact_label} page={item[exact_label]['page_rank']} "
                    f"doc={item[exact_label]['doc_rank']}; "
                    f"{compact_label} page={item[compact_label]['page_rank']} "
                    f"doc={item[compact_label]['doc_rank']}"
                ),
                "",
                "Gold pages:",
            ]
        )
        for page in item["gold_pages"]:
            lines.append(
                f"- `{page['page_uid']}` page_number={page.get('page_number')} image={page['image_path']}"
            )
            if page.get("copied_path"):
                lines.append(f"  copied={page['copied_path']}")
        if item.get("top_retrieved"):
            lines.extend(["", "Top retrieved pages:"])
            for label, rows in item["top_retrieved"].items():
                lines.append(f"- {label}:")
                for row in rows:
                    lines.append(
                        f"  - rank={row['rank']} `{row['page_uid']}` score={row.get('score')} image={row['image_path']}"
                    )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    filters = [parse_filter(raw) for raw in args.where]
    exact_pred = read_prediction(args.exact_pred)
    compact_pred = read_prediction(args.compact_pred)
    gold_rows = read_jsonl(data_root / f"MMQA_{args.split}.jsonl")
    page_map = load_page_map(data_root / f"doc_pages_{args.split}.jsonl")
    copy_dir = Path(args.copy_gold_pages_dir) if args.copy_gold_pages_dir else None

    candidates = []
    for gold_row in gold_rows:
        if filters and not row_matches(gold_row, filters):
            continue
        qid = str(gold_row["qid"])
        exact_score = score(exact_pred.get(qid), gold_row, args.hit_k)
        compact_score = score(compact_pred.get(qid), gold_row, args.hit_k)
        if not is_failure(args.fail_mode, exact_score, compact_score):
            continue
        candidates.append((gold_row, exact_score, compact_score))

    candidates.sort(
        key=lambda item: (
            rank_sort_value(item[2]["page_rank"]),
            rank_sort_value(item[0].get("metadata", {}).get("gold_page_ids", [10**9])[0] if item[0].get("metadata", {}).get("gold_page_ids") else None),
            str(item[0]["qid"]),
        ),
        reverse=True,
    )
    if args.max_examples > 0:
        candidates = candidates[: args.max_examples]

    examples = []
    for gold_row, exact_score, compact_score in candidates:
        qid = str(gold_row["qid"])
        gold_pages = [page_entry(data_root, page_map, uid) for uid in gold_page_uids(gold_row)]
        if copy_dir is not None:
            copied = copy_gold_pages(copy_dir, qid, gold_pages)
            for page in gold_pages:
                if page["page_uid"] in copied:
                    page["copied_path"] = copied[page["page_uid"]]
        examples.append(
            {
                "qid": qid,
                "question": gold_row.get("question", ""),
                "metadata": gold_row.get("metadata", {}),
                args.exact_label: exact_score,
                args.compact_label: compact_score,
                "gold_pages": gold_pages,
                "top_retrieved": {
                    args.exact_label: retrieved_rows(
                        exact_pred.get(qid), data_root, page_map, args.top_retrieved
                    ),
                    args.compact_label: retrieved_rows(
                        compact_pred.get(qid), data_root, page_map, args.top_retrieved
                    ),
                },
            }
        )

    markdown = markdown_for_examples(examples, args.exact_label, args.compact_label)
    if args.output_md:
        output_path = Path(args.output_md)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
        print(f"saved_md={output_path}")
    print(markdown)


if __name__ == "__main__":
    main()
