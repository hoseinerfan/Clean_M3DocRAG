#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_prediction(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "predictions" in payload and isinstance(
        payload["predictions"], (dict, list)
    ):
        payload = payload["predictions"]

    rows_by_qid: dict[str, dict] = {}
    if isinstance(payload, list):
        iterable = enumerate(payload)
    elif isinstance(payload, dict):
        iterable = payload.items()
    else:
        raise TypeError(f"Prediction JSON must be a list or object of prediction rows: {path}")

    for raw_key, row in iterable:
        if not isinstance(row, dict):
            raise TypeError(f"Prediction row must be an object: {path} key={raw_key!r}")
        qid = str(row.get("qid", "")).strip() or str(raw_key).strip()
        if not qid:
            raise ValueError(f"Prediction row is missing qid and key is empty: {path} key={raw_key!r}")
        if qid in rows_by_qid:
            raise ValueError(f"Duplicate qid after normalization: {qid} ({path})")
        rows_by_qid[qid] = row
    return rows_by_qid


def gold_page_uids(row: dict) -> list[str]:
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
            uids.add(f"{doc_id}_page{int(page_idx)}")
    return sorted(uids)


def first_rank(ranked: list[str], gold: set[str]) -> int | None:
    for idx, item in enumerate(ranked, start=1):
        if item in gold:
            return idx
    return None


def top_pages(retrieval_rows: list[list], limit: int) -> list[dict]:
    rows = []
    for rank, row in enumerate(retrieval_rows[:limit], start=1):
        rows.append(
            {
                "rank": rank,
                "doc_id": str(row[0]),
                "page_idx": int(row[1]),
                "page_uid": f"{row[0]}_page{int(row[1])}",
                "score": float(row[2]) if len(row) >= 3 else None,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare page-labeled retrieval predictions and extract recovered / worsened cases "
            "at a chosen top-k threshold."
        )
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--show-top-pages", type=int, default=8)
    parser.add_argument("--topn", type=int, default=20)
    parser.add_argument("--qid", dest="qids", action="append", default=[])
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-jsonl", default="")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def movement_for_hit(baseline_rank: int | None, candidate_rank: int | None, topk: int) -> str:
    baseline_hit = baseline_rank is not None and baseline_rank <= topk
    candidate_hit = candidate_rank is not None and candidate_rank <= topk
    if not baseline_hit and candidate_hit:
        return "recovered"
    if baseline_hit and not candidate_hit:
        return "lost"
    if baseline_rank is None and candidate_rank is None:
        return "missing_in_both"
    if baseline_rank is not None and candidate_rank is not None and candidate_rank < baseline_rank:
        return "improved_rank"
    if baseline_rank is not None and candidate_rank is not None and candidate_rank > baseline_rank:
        return "worsened_rank"
    return "unchanged"


def main() -> None:
    args = parse_args()

    baseline = load_prediction(Path(args.baseline))
    candidate = load_prediction(Path(args.candidate))
    gold_rows = read_jsonl(Path(args.gold))
    gold_by_qid = {str(row["qid"]): row for row in gold_rows}

    qids = args.qids if args.qids else sorted(set(baseline.keys()) & set(candidate.keys()) & set(gold_by_qid.keys()))

    cases: list[dict] = []
    for qid in qids:
        gold_row = gold_by_qid[qid]
        gold_pages = set(gold_page_uids(gold_row))
        baseline_rows = baseline[qid].get("page_retrieval_results", [])
        candidate_rows = candidate[qid].get("page_retrieval_results", [])
        baseline_ranked_pages = [f"{row[0]}_page{int(row[1])}" for row in baseline_rows]
        candidate_ranked_pages = [f"{row[0]}_page{int(row[1])}" for row in candidate_rows]
        baseline_rank = first_rank(baseline_ranked_pages, gold_pages)
        candidate_rank = first_rank(candidate_ranked_pages, gold_pages)
        movement = movement_for_hit(baseline_rank, candidate_rank, int(args.topk))
        cases.append(
            {
                "qid": qid,
                "question": gold_row.get("question", ""),
                "gold_page_uids": sorted(gold_pages),
                "baseline_first_gold_page_rank": baseline_rank,
                "candidate_first_gold_page_rank": candidate_rank,
                "rank_delta": (
                    None
                    if baseline_rank is None or candidate_rank is None
                    else int(baseline_rank) - int(candidate_rank)
                ),
                "movement": movement,
                "baseline_top_pages": top_pages(baseline_rows, int(args.show_top_pages)),
                "candidate_top_pages": top_pages(candidate_rows, int(args.show_top_pages)),
            }
        )

    recovered = sorted(
        [case for case in cases if case["movement"] == "recovered"],
        key=lambda case: (
            case["candidate_first_gold_page_rank"] if case["candidate_first_gold_page_rank"] is not None else 10**9,
            case["baseline_first_gold_page_rank"] if case["baseline_first_gold_page_rank"] is not None else 10**9,
            case["qid"],
        ),
    )
    lost = sorted(
        [case for case in cases if case["movement"] == "lost"],
        key=lambda case: (
            case["baseline_first_gold_page_rank"] if case["baseline_first_gold_page_rank"] is not None else 10**9,
            case["candidate_first_gold_page_rank"] if case["candidate_first_gold_page_rank"] is not None else 10**9,
            case["qid"],
        ),
    )
    improved_rank = sorted(
        [case for case in cases if case["movement"] == "improved_rank"],
        key=lambda case: (
            -(case["rank_delta"] or 0),
            case["candidate_first_gold_page_rank"] if case["candidate_first_gold_page_rank"] is not None else 10**9,
            case["qid"],
        ),
    )
    worsened_rank = sorted(
        [case for case in cases if case["movement"] == "worsened_rank"],
        key=lambda case: (
            case["rank_delta"] or 0,
            case["candidate_first_gold_page_rank"] if case["candidate_first_gold_page_rank"] is not None else 10**9,
            case["qid"],
        ),
    )

    payload = {
        "n_qids": len(cases),
        "topk": int(args.topk),
        "counts": {
            "recovered": len(recovered),
            "lost": len(lost),
            "improved_rank": len(improved_rank),
            "worsened_rank": len(worsened_rank),
            "unchanged": sum(case["movement"] == "unchanged" for case in cases),
            "missing_in_both": sum(case["movement"] == "missing_in_both" for case in cases),
        },
        "top_recovered": recovered[: int(args.topn)],
        "top_lost": lost[: int(args.topn)],
        "top_improved_rank": improved_rank[: int(args.topn)],
        "top_worsened_rank": worsened_rank[: int(args.topn)],
    }

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.output_jsonl:
        with Path(args.output_jsonl).open("w", encoding="utf-8") as handle:
            for row in recovered:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(f"n_qids {payload['n_qids']}")
    print(f"topk {payload['topk']}")
    print(f"recovered {payload['counts']['recovered']}")
    print(f"lost {payload['counts']['lost']}")
    print(f"improved_rank {payload['counts']['improved_rank']}")
    print(f"worsened_rank {payload['counts']['worsened_rank']}")
    print(f"unchanged {payload['counts']['unchanged']}")
    print(f"missing_in_both {payload['counts']['missing_in_both']}")
    print(f"top_recovered {len(payload['top_recovered'])}")
    for row in payload["top_recovered"]:
        print(
            json.dumps(
                {
                    "qid": row["qid"],
                    "baseline_first_gold_page_rank": row["baseline_first_gold_page_rank"],
                    "candidate_first_gold_page_rank": row["candidate_first_gold_page_rank"],
                    "gold_page_uids": row["gold_page_uids"],
                    "question": row["question"],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
