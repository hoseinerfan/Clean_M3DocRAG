#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_RECALL_KS = [1, 2, 4, 5, 10, 20, 50, 100]
DEFAULT_GROUP_FIELDS = [
    "metadata.domain",
    "metadata.type",
    "feature.page_overlap20_bin",
    "feature.doc_overlap20_bin",
    "feature.dense_top1_in_sparse1000",
    "feature.dense_top1_demoted_by_reference",
    "feature.dense_top3_demoted_by_reference",
    "feature.query_structural_cue",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze dense/SPLADE reliability features against multiple page-retrieval "
            "predictions. This is intended for designing query-adaptive Graph-PPR weights."
        )
    )
    parser.add_argument("--gold", required=True, help="MMQA-style gold JSONL.")
    parser.add_argument("--dense-pred", required=True, help="Dense/plain_top224 prediction JSON.")
    parser.add_argument("--sparse-pred", required=True, help="SPLADE prediction JSON.")
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="Candidate prediction in name=/path/to/prediction.json form. Repeat.",
    )
    parser.add_argument(
        "--reference-label",
        default="",
        help=(
            "Optional candidate label used for demotion features, usually graph_best. "
            "Must match one --prediction name."
        ),
    )
    parser.add_argument(
        "--group-field",
        action="append",
        default=[],
        help=(
            "Group field. Supports metadata.* and feature.*. Repeat. "
            f"Default: {', '.join(DEFAULT_GROUP_FIELDS)}"
        ),
    )
    parser.add_argument("--recall-k", dest="recall_ks", type=int, nargs="+", default=DEFAULT_RECALL_KS)
    parser.add_argument("--hit-k", type=int, default=4)
    parser.add_argument("--oracle-metric", default="page_recall@4")
    parser.add_argument("--baseline-label", default="", help="Optional label for delta columns.")
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--top-groups", type=int, default=200)
    parser.add_argument("--output-md", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-csv", default="")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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

    rows: dict[str, dict[str, Any]] = {}
    if isinstance(payload, dict):
        for qid, row in payload.items():
            if isinstance(row, dict):
                rows[str(qid)] = row
    elif isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict) and str(row.get("qid", "")).strip():
                rows[str(row["qid"])] = row
    else:
        raise TypeError(f"Prediction JSON must be an object or list: {path}")
    return rows


def parse_named_paths(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected name=/path form: {value}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Missing prediction name: {value}")
        if name in parsed:
            raise ValueError(f"Duplicate prediction label: {name}")
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        parsed[name] = path
    return parsed


def page_uid(doc_id: str, page_idx: int) -> str:
    return f"{doc_id}_page{int(page_idx)}"


def ranked_pages(pred_row: dict[str, Any] | None, limit: int = 0) -> list[str]:
    if pred_row is None:
        return []
    pages: list[str] = []
    seen: set[str] = set()
    for item in pred_row.get("page_retrieval_results", []):
        if not isinstance(item, list) or len(item) < 2:
            continue
        uid = page_uid(str(item[0]), int(item[1]))
        if uid in seen:
            continue
        seen.add(uid)
        pages.append(uid)
        if limit > 0 and len(pages) >= limit:
            break
    return pages


def ranked_docs(pred_row: dict[str, Any] | None, limit: int = 0) -> list[str]:
    if pred_row is None:
        return []
    docs: list[str] = []
    seen: set[str] = set()
    for item in pred_row.get("page_retrieval_results", []):
        if not isinstance(item, list) or not item:
            continue
        doc_id = str(item[0])
        if doc_id in seen:
            continue
        seen.add(doc_id)
        docs.append(doc_id)
        if limit > 0 and len(docs) >= limit:
            break
    return docs


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


def gold_doc_ids(row: dict[str, Any]) -> set[str]:
    return {
        str(ctx.get("doc_id", "")).strip()
        for ctx in row.get("supporting_context", [])
        if str(ctx.get("doc_id", "")).strip()
    }


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / len(gold)


def first_rank(ranked: list[str], gold: set[str]) -> int | None:
    for idx, item in enumerate(ranked, start=1):
        if item in gold:
            return idx
    return None


def score_row(
    pred_row: dict[str, Any] | None,
    gold_pages: set[str],
    gold_docs: set[str],
    recall_ks: list[int],
    hit_k: int,
) -> dict[str, Any]:
    pages = ranked_pages(pred_row)
    docs = ranked_docs(pred_row)
    page_rank = first_rank(pages, gold_pages)
    doc_rank = first_rank(docs, gold_docs)
    scores: dict[str, Any] = {
        "page_first_rank": page_rank,
        "doc_first_rank": doc_rank,
        f"page_hit@{hit_k}": page_rank is not None and page_rank <= hit_k,
        f"doc_hit@{hit_k}": doc_rank is not None and doc_rank <= hit_k,
    }
    for k in recall_ks:
        scores[f"page_recall@{k}"] = recall_at_k(pages, gold_pages, k)
        scores[f"doc_recall@{k}"] = recall_at_k(docs, gold_docs, k)
    return scores


def jaccard(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


def overlap_fraction(left: list[str], right: list[str]) -> float:
    if not left:
        return 0.0
    return len(set(left) & set(right)) / len(set(left))


def bin_float(value: float, bins: list[tuple[float, str]]) -> str:
    for upper, label in bins:
        if value <= upper:
            return label
    return bins[-1][1]


def query_structural_cue(question: str) -> str:
    q = question.lower()
    cues: list[str] = []
    if re.search(r"\bpage\s+\d+\b", q):
        cues.append("page_number")
    if re.search(r"\b(last|final|end)\b", q):
        cues.append("last_final")
    if re.search(r"\b(how many pages|number of pages|total pages)\b", q):
        cues.append("page_count")
    if re.search(r"\b(signature|signed|leadership|approval|certification)\b", q):
        cues.append("signature")
    if re.search(r"\b(table|chart|figure|image|graph)\b", q):
        cues.append("visual_token")
    return "+".join(cues) if cues else "none"


def get_path(row: dict[str, Any], path: str) -> Any:
    current: Any = row
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def normalize_group_value(value: Any) -> str:
    if value is None:
        return "UNKNOWN"
    if isinstance(value, list):
        return str(value)
    text = str(value).strip()
    return text if text else "UNKNOWN"


def doc_concentration(docs: list[str], pages: list[str], top_pages: int) -> float:
    if not pages[:top_pages]:
        return 0.0
    page_docs = [uid.rsplit("_page", 1)[0] for uid in pages[:top_pages]]
    count = Counter(page_docs)
    return max(count.values()) / len(page_docs)


def build_features(
    *,
    gold_row: dict[str, Any],
    dense_row: dict[str, Any] | None,
    sparse_row: dict[str, Any] | None,
    reference_row: dict[str, Any] | None,
) -> dict[str, Any]:
    dense_pages_4 = ranked_pages(dense_row, 4)
    dense_pages_20 = ranked_pages(dense_row, 20)
    dense_pages_100 = ranked_pages(dense_row, 100)
    dense_pages_1000 = ranked_pages(dense_row, 1000)
    sparse_pages_4 = ranked_pages(sparse_row, 4)
    sparse_pages_20 = ranked_pages(sparse_row, 20)
    sparse_pages_100 = ranked_pages(sparse_row, 100)
    sparse_pages_1000 = ranked_pages(sparse_row, 1000)
    dense_docs_4 = ranked_docs(dense_row, 4)
    dense_docs_20 = ranked_docs(dense_row, 20)
    dense_docs_100 = ranked_docs(dense_row, 100)
    sparse_docs_4 = ranked_docs(sparse_row, 4)
    sparse_docs_20 = ranked_docs(sparse_row, 20)
    sparse_docs_100 = ranked_docs(sparse_row, 100)
    reference_pages_4 = ranked_pages(reference_row, 4)
    question = str(gold_row.get("question", ""))

    dense_top1 = dense_pages_1000[0] if dense_pages_1000 else None
    dense_top3 = set(dense_pages_1000[:3])
    reference_top4 = set(reference_pages_4)

    page_overlap20 = jaccard(dense_pages_20, sparse_pages_20)
    doc_overlap20 = jaccard(dense_docs_20, sparse_docs_20)
    features: dict[str, Any] = {
        "query_len_bin": bin_float(
            len(question.split()),
            [(5, "q_len_<=5"), (10, "q_len_6_10"), (20, "q_len_11_20"), (10**9, "q_len_>20")],
        ),
        "query_has_number": bool(re.search(r"\d", question)),
        "query_structural_cue": query_structural_cue(question),
        "page_overlap4": jaccard(dense_pages_4, sparse_pages_4),
        "page_overlap20": page_overlap20,
        "page_overlap100": jaccard(dense_pages_100, sparse_pages_100),
        "doc_overlap4": jaccard(dense_docs_4, sparse_docs_4),
        "doc_overlap20": doc_overlap20,
        "doc_overlap100": jaccard(dense_docs_100, sparse_docs_100),
        "dense_top1_in_sparse20": dense_top1 in set(sparse_pages_20) if dense_top1 else False,
        "dense_top1_in_sparse100": dense_top1 in set(sparse_pages_100) if dense_top1 else False,
        "dense_top1_in_sparse1000": dense_top1 in set(sparse_pages_1000) if dense_top1 else False,
        "dense_top1_doc_in_sparse20": (
            dense_docs_20[0] in set(sparse_docs_20) if dense_docs_20 else False
        ),
        "dense_top1_doc_in_sparse100": (
            dense_docs_20[0] in set(sparse_docs_100) if dense_docs_20 else False
        ),
        "dense_doc_concentration20": doc_concentration(dense_docs_20, dense_pages_20, 20),
        "sparse_doc_concentration20": doc_concentration(sparse_docs_20, sparse_pages_20, 20),
        "page_overlap20_bin": bin_float(
            page_overlap20,
            [(0.0, "page_overlap20=0"), (0.05, "page_overlap20<=.05"), (0.15, "page_overlap20<=.15"), (1, "page_overlap20>.15")],
        ),
        "doc_overlap20_bin": bin_float(
            doc_overlap20,
            [(0.0, "doc_overlap20=0"), (0.05, "doc_overlap20<=.05"), (0.15, "doc_overlap20<=.15"), (1, "doc_overlap20>.15")],
        ),
        "dense_top1_demoted_by_reference": (
            bool(dense_top1 and reference_row is not None and dense_top1 not in reference_top4)
        ),
        "dense_top3_demoted_by_reference": (
            bool(reference_row is not None and not dense_top3.issubset(reference_top4))
        ),
    }
    return features


def metric_value(scores: dict[str, Any], metric: str) -> float:
    value = scores.get(metric)
    if isinstance(value, bool):
        return float(value)
    if value is None:
        return 0.0
    return float(value)


def choose_oracle_label(
    row: dict[str, Any],
    labels: list[str],
    metric: str,
    baseline_label: str,
) -> tuple[str, str]:
    values = {label: metric_value(row["scores"][label], metric) for label in labels}
    best_value = max(values.values())
    best_labels = [label for label in labels if math.isclose(values[label], best_value, abs_tol=1e-12)]
    if len(best_labels) == 1:
        return best_labels[0], best_labels[0]
    if baseline_label and baseline_label in best_labels:
        score_label = baseline_label
    else:
        score_label = best_labels[0]
    return score_label, f"tie/{len(best_labels)}"


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def summarize_label(rows: list[dict[str, Any]], labels: list[str], recall_ks: list[int], hit_k: int) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for label in labels:
        item: dict[str, Any] = {"label": label, "n": len(rows)}
        for k in recall_ks:
            item[f"page_recall@{k}"] = mean([metric_value(row["scores"][label], f"page_recall@{k}") for row in rows])
            item[f"doc_recall@{k}"] = mean([metric_value(row["scores"][label], f"doc_recall@{k}") for row in rows])
        item[f"page_hit@{hit_k}"] = mean([metric_value(row["scores"][label], f"page_hit@{hit_k}") for row in rows])
        item[f"doc_hit@{hit_k}"] = mean([metric_value(row["scores"][label], f"doc_hit@{hit_k}") for row in rows])
        summaries.append(item)
    return summaries


def group_rows(rows: list[dict[str, Any]], group_field: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if group_field.startswith("feature."):
            value = row["features"].get(group_field.split(".", 1)[1])
        else:
            value = get_path(row["gold"], group_field)
        groups[normalize_group_value(value)].append(row)
    return groups


def summarize_groups(
    rows: list[dict[str, Any]],
    labels: list[str],
    group_fields: list[str],
    oracle_metric: str,
    baseline_label: str,
    min_count: int,
    top_groups: int,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for group_field in group_fields:
        for group, group_items in group_rows(rows, group_field).items():
            if len(group_items) < min_count:
                continue
            winner_counts = Counter(row["oracle_report_label"] for row in group_items)
            mean_by_label = {
                label: mean([metric_value(row["scores"][label], oracle_metric) for row in group_items])
                for label in labels
            }
            best_label = max(labels, key=lambda label: (mean_by_label[label], label))
            item: dict[str, Any] = {
                "group_by": group_field,
                "group": group,
                "n": len(group_items),
                "best_label_by_mean": best_label,
                f"best_mean_{oracle_metric}": mean_by_label[best_label],
                "oracle_winner_top": winner_counts.most_common(1)[0][0] if winner_counts else "",
                "oracle_winner_top_count": winner_counts.most_common(1)[0][1] if winner_counts else 0,
                "oracle_winner_dist": ", ".join(f"{label}:{count}" for label, count in winner_counts.most_common(8)),
            }
            for label in labels:
                item[f"{label}_{oracle_metric}"] = mean_by_label[label]
            if baseline_label and baseline_label in labels:
                item[f"best_minus_{baseline_label}_{oracle_metric}"] = (
                    mean_by_label[best_label] - mean_by_label[baseline_label]
                )
            summaries.append(item)
    summaries.sort(
        key=lambda item: (
            item.get(f"best_minus_{baseline_label}_{oracle_metric}", 0.0)
            if baseline_label
            else item.get(f"best_mean_{oracle_metric}", 0.0),
            item["n"],
        ),
        reverse=True,
    )
    return summaries[:top_groups]


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def main() -> None:
    args = parse_args()
    if args.hit_k not in args.recall_ks:
        args.recall_ks = sorted(set(args.recall_ks + [args.hit_k]))

    prediction_paths = parse_named_paths(args.prediction)
    labels = list(prediction_paths)
    if args.reference_label and args.reference_label not in labels:
        raise KeyError(f"--reference-label must match one --prediction name: {args.reference_label}")
    if args.baseline_label and args.baseline_label not in labels:
        raise KeyError(f"--baseline-label must match one --prediction name: {args.baseline_label}")

    gold_rows = {str(row["qid"]): row for row in read_jsonl(Path(args.gold))}
    dense_pred = load_prediction(Path(args.dense_pred))
    sparse_pred = load_prediction(Path(args.sparse_pred))
    predictions = {label: load_prediction(path) for label, path in prediction_paths.items()}

    qids = set(gold_rows) & set(dense_pred) & set(sparse_pred)
    for label, pred in predictions.items():
        qids &= set(pred)
    if not qids:
        raise ValueError("No qids remain after intersecting gold/dense/sparse/predictions.")

    rows: list[dict[str, Any]] = []
    for qid in sorted(qids):
        gold = gold_rows[qid]
        reference_row = predictions[args.reference_label].get(qid) if args.reference_label else None
        features = build_features(
            gold_row=gold,
            dense_row=dense_pred.get(qid),
            sparse_row=sparse_pred.get(qid),
            reference_row=reference_row,
        )
        gold_pages = gold_page_uids(gold)
        gold_docs = gold_doc_ids(gold)
        scores = {
            label: score_row(predictions[label].get(qid), gold_pages, gold_docs, args.recall_ks, args.hit_k)
            for label in labels
        }
        row = {
            "qid": qid,
            "question": gold.get("question", ""),
            "gold": gold,
            "features": features,
            "scores": scores,
        }
        oracle_label, oracle_report_label = choose_oracle_label(
            row,
            labels,
            args.oracle_metric,
            args.baseline_label,
        )
        row["oracle_label"] = oracle_label
        row["oracle_report_label"] = oracle_report_label
        rows.append(row)

    overall = summarize_label(rows, labels, args.recall_ks, args.hit_k)
    oracle_scores: dict[str, Any] = {"label": f"oracle_by_{args.oracle_metric}", "n": len(rows)}
    for k in args.recall_ks:
        oracle_scores[f"page_recall@{k}"] = mean(
            [
                metric_value(row["scores"][row["oracle_label"]], f"page_recall@{k}")
                for row in rows
            ]
        )
        oracle_scores[f"doc_recall@{k}"] = mean(
            [
                metric_value(row["scores"][row["oracle_label"]], f"doc_recall@{k}")
                for row in rows
            ]
        )
    oracle_scores[f"page_hit@{args.hit_k}"] = mean(
        [metric_value(row["scores"][row["oracle_label"]], f"page_hit@{args.hit_k}") for row in rows]
    )
    oracle_scores[f"doc_hit@{args.hit_k}"] = mean(
        [metric_value(row["scores"][row["oracle_label"]], f"doc_hit@{args.hit_k}") for row in rows]
    )
    overall_with_oracle = overall + [oracle_scores]

    group_fields = args.group_field or DEFAULT_GROUP_FIELDS
    group_summaries = summarize_groups(
        rows=rows,
        labels=labels,
        group_fields=group_fields,
        oracle_metric=args.oracle_metric,
        baseline_label=args.baseline_label,
        min_count=int(args.min_count),
        top_groups=int(args.top_groups),
    )

    winner_counts = Counter(row["oracle_report_label"] for row in rows)
    feature_rows: list[dict[str, Any]] = []
    for row in rows:
        flat: dict[str, Any] = {
            "qid": row["qid"],
            "question": row["question"],
            "oracle_label": row["oracle_label"],
            "oracle_report_label": row["oracle_report_label"],
        }
        metadata = row["gold"].get("metadata", {})
        for key in ["domain", "type", "source"]:
            flat[f"metadata.{key}"] = metadata.get(key, "")
        for key, value in row["features"].items():
            flat[f"feature.{key}"] = value
        for label in labels:
            for metric in [args.oracle_metric, f"page_hit@{args.hit_k}", f"doc_hit@{args.hit_k}"]:
                flat[f"{label}.{metric}"] = metric_value(row["scores"][label], metric)
        feature_rows.append(flat)

    payload = {
        "gold": args.gold,
        "dense_pred": args.dense_pred,
        "sparse_pred": args.sparse_pred,
        "prediction_paths": {label: str(path) for label, path in prediction_paths.items()},
        "labels": labels,
        "reference_label": args.reference_label,
        "baseline_label": args.baseline_label,
        "oracle_metric": args.oracle_metric,
        "hit_k": args.hit_k,
        "recall_ks": args.recall_ks,
        "qid_count": len(rows),
        "overall": overall_with_oracle,
        "oracle_winner_counts": dict(winner_counts),
        "groups": group_summaries,
    }

    columns = [
        "label",
        "n",
        "page_recall@1",
        "page_recall@4",
        "page_recall@20",
        f"page_hit@{args.hit_k}",
        "doc_recall@4",
        "doc_recall@20",
        f"doc_hit@{args.hit_k}",
    ]
    group_columns = [
        "group_by",
        "group",
        "n",
        "best_label_by_mean",
        f"best_mean_{args.oracle_metric}",
        "oracle_winner_top",
        "oracle_winner_top_count",
        "oracle_winner_dist",
    ]
    if args.baseline_label:
        group_columns.append(f"best_minus_{args.baseline_label}_{args.oracle_metric}")

    markdown = "\n".join(
        [
            "# Reliability-Adaptive Graph Analysis",
            "",
            "## Overall",
            "",
            markdown_table(overall_with_oracle, columns),
            "## Oracle Winner Counts",
            "",
            markdown_table(
                [{"label": label, "count": count} for label, count in winner_counts.most_common()],
                ["label", "count"],
            ),
            "## Feature/Metadata Groups",
            "",
            markdown_table(group_summaries, group_columns),
        ]
    )

    if args.output_md:
        path = Path(args.output_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
        print(f"saved_md={path}")
    else:
        print(markdown)

    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"saved_json={path}")

    if args.output_csv:
        write_csv(Path(args.output_csv), feature_rows)
        print(f"saved_csv={args.output_csv}")


if __name__ == "__main__":
    main()
