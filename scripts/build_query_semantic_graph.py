#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "please",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
    "write",
    "your",
}

DEFAULT_RECALL_KS = [1, 2, 4, 5, 10, 20, 50, 100]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a non-oracle query semantic graph. QID nodes connect to query-token, "
            "query-cue, and optional observable rank-signature nodes. Gold labels are used "
            "only for cluster reporting and candidate utility, never as graph features."
        )
    )
    parser.add_argument("--gold", required=True, help="MMQA-style gold JSONL.")
    parser.add_argument("--base-prediction", default="", help="Optional base prediction JSON for rank-state nodes.")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Optional candidate prediction for per-cluster utility and observable candidate-signature nodes.",
    )
    parser.add_argument("--hit-k", type=int, default=4)
    parser.add_argument("--max-tokens-per-query", type=int, default=12)
    parser.add_argument("--min-token-df", type=int, default=2)
    parser.add_argument("--max-token-df-frac", type=float, default=0.35)
    parser.add_argument("--max-feature-df-frac", type=float, default=0.45)
    parser.add_argument("--neighbor-threshold", type=float, default=0.08)
    parser.add_argument("--max-neighbors", type=int, default=30)
    parser.add_argument("--label-prop-iters", type=int, default=20)
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--top-clusters", type=int, default=30)
    parser.add_argument("--output-dir", required=True)
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
    if isinstance(payload, dict) and "predictions" in payload:
        payload = payload["predictions"]
    rows: dict[str, dict[str, Any]] = {}
    if isinstance(payload, dict):
        iterable = payload.items()
    elif isinstance(payload, list):
        iterable = enumerate(payload)
    else:
        raise TypeError(f"Prediction JSON must be a list or object: {path}")
    for raw_key, row in iterable:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("qid", "")).strip() or str(raw_key).strip()
        if qid:
            rows[qid] = row
    return rows


def parse_labeled_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Expected LABEL=PATH for --candidate, got {raw!r}")
    label, path = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Empty candidate label in {raw!r}")
    return label, Path(path)


def question_text(row: dict[str, Any]) -> str:
    for key in ("question", "query", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def page_uid(doc_id: Any, page_idx: Any) -> str:
    return f"{doc_id}_page{int(page_idx)}"


def page_doc(uid: str) -> str:
    return uid.rsplit("_page", 1)[0]


def ranked_pages(row: dict[str, Any] | None, limit: int = 0) -> list[str]:
    if row is None:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in row.get("page_retrieval_results", []):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            uid = page_uid(str(item[0]), int(item[1]))
        except (TypeError, ValueError):
            continue
        if uid in seen:
            continue
        seen.add(uid)
        out.append(uid)
        if limit > 0 and len(out) >= limit:
            break
    return out


def ranked_docs(row: dict[str, Any] | None, limit: int = 0) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for uid in ranked_pages(row):
        doc_id = page_doc(uid)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        out.append(doc_id)
        if limit > 0 and len(out) >= limit:
            break
    return out


def first_rank(items: list[str], gold: set[str]) -> int | None:
    for idx, item in enumerate(items, start=1):
        if item in gold:
            return idx
    return None


def gold_page_uids(row: dict[str, Any]) -> set[str]:
    out = {
        str(value).strip()
        for value in row.get("metadata", {}).get("gold_page_uids", [])
        if str(value).strip()
    }
    for ctx in row.get("supporting_context", []):
        doc_id = str(ctx.get("doc_id", "")).strip()
        page_idx = ctx.get("page_idx", ctx.get("page_id"))
        if doc_id and page_idx is not None:
            out.add(page_uid(doc_id, page_idx))
    return out


def gold_doc_ids(row: dict[str, Any]) -> set[str]:
    out = {
        str(value).strip()
        for value in row.get("metadata", {}).get("gold_doc_ids", [])
        if str(value).strip()
    }
    for ctx in row.get("supporting_context", []):
        doc_id = str(ctx.get("doc_id", "")).strip()
        if doc_id:
            out.add(doc_id)
    return out


def hit_rank(row: dict[str, Any] | None, gold_pages: set[str], gold_docs: set[str]) -> tuple[int | None, int | None]:
    return first_rank(ranked_pages(row), gold_pages), first_rank(ranked_docs(row), gold_docs)


def movement_for_hit(base_rank: int | None, candidate_rank: int | None, hit_k: int) -> str:
    base_hit = base_rank is not None and base_rank <= hit_k
    candidate_hit = candidate_rank is not None and candidate_rank <= hit_k
    if not base_hit and candidate_hit:
        return "recovered"
    if base_hit and not candidate_hit:
        return "lost"
    if base_rank is None and candidate_rank is None:
        return "missing_in_both"
    if base_rank is not None and candidate_rank is not None and candidate_rank < base_rank:
        return "improved_rank"
    if base_rank is not None and candidate_rank is not None and candidate_rank > base_rank:
        return "worsened_rank"
    return "unchanged"


def tokenize(question: str) -> list[str]:
    tokens = []
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9_.-]*|\d+(?:\.\d+)?%?", question.lower()):
        token = raw.strip("._-")
        if not token or token in STOPWORDS:
            continue
        if len(token) < 3 and not any(ch.isdigit() for ch in token):
            continue
        tokens.append(token)
    return tokens


def query_cues(question: str) -> list[str]:
    q = question.lower()
    cues: list[str] = []
    patterns = {
        "quantity": r"\d|\b(how many|how much|percentage|percent|ratio|total|sum|difference|average|increase|decrease)\b",
        "page_locator": r"\b(page|section|appendix|slide|start|begin|mentions?)\b",
        "visual": r"\b(image|picture|photo|figure|diagram|map|screen|interface)\b",
        "table": r"\b(table|row|column|cell)\b",
        "chart": r"\b(chart|graph|plot|bar|axis|trend|distribution)\b",
        "comparison": r"\b(compare|comparison|between|higher|lower|largest|smallest|least|most|maximum|minimum)\b",
        "yes_no": r"^(is|are|was|were|does|do|did|has|have|can|could|should)\b|\byes\b|\bno\b",
        "finance": r"\b(fy|fiscal|liabilities|assets|equity|revenue|income|cash|expenses|debt|ratio|shareholders?)\b",
        "academic_metric": r"\b(accuracy|f1|bleu|rouge|hit rate|dataset|model|baseline|method)\b",
    }
    for cue, pattern in patterns.items():
        if re.search(pattern, q):
            cues.append(cue)
    length = len(re.findall(r"\w+", q))
    if length <= 8:
        cues.append("short_query")
    elif length >= 24:
        cues.append("long_query")
    else:
        cues.append("medium_query")
    return cues


def score_at(row: dict[str, Any] | None, rank: int) -> float | None:
    if row is None:
        return None
    items = row.get("page_retrieval_results", [])
    idx = rank - 1
    if idx < 0 or idx >= len(items):
        return None
    item = items[idx]
    if not isinstance(item, (list, tuple)) or len(item) < 3:
        return None
    try:
        return float(item[2])
    except (TypeError, ValueError):
        return None


def bin_float(value: float | None, bins: list[tuple[float, str]]) -> str:
    if value is None or not math.isfinite(value):
        return "missing"
    for threshold, label in bins:
        if value <= threshold:
            return label
    return bins[-1][1]


def rank_state_features(base_row: dict[str, Any] | None) -> list[str]:
    if base_row is None:
        return []
    pages4 = ranked_pages(base_row, 4)
    docs4 = ranked_docs(base_row, 4)
    margin = None
    s4 = score_at(base_row, 4)
    s5 = score_at(base_row, 5)
    if s4 is not None and s5 is not None:
        margin = s4 - s5
    features = [
        f"base_top4_doc_count::{len(docs4)}",
        f"base_top4_same_doc::{str(len({page_doc(uid) for uid in pages4}) <= 2).lower()}",
        f"base_margin_4_5::{bin_float(margin, [(0.001, 'tiny'), (0.01, 'small'), (0.05, 'medium'), (1e9, 'large')])}",
    ]
    return features


def candidate_signature_features(
    *,
    label: str,
    base_row: dict[str, Any] | None,
    candidate_row: dict[str, Any] | None,
    hit_k: int,
) -> list[str]:
    if base_row is None or candidate_row is None:
        return []
    base_pages = ranked_pages(base_row)
    cand_pages = ranked_pages(candidate_row)
    base_docs = ranked_docs(base_row)
    cand_docs = ranked_docs(candidate_row)
    base_rank = {uid: idx for idx, uid in enumerate(base_pages, start=1)}
    promoted_ranks = [base_rank.get(uid, len(base_pages) + 1) for uid in cand_pages[:hit_k]]
    promoted = [rank for rank in promoted_ranks if rank > hit_k]
    overlap4 = len(set(base_pages[:hit_k]) & set(cand_pages[:hit_k]))
    doc_overlap4 = len(set(base_docs[:hit_k]) & set(cand_docs[:hit_k]))
    features = [
        f"candidate::{label}::page_overlap4::{overlap4}",
        f"candidate::{label}::doc_overlap4::{doc_overlap4}",
        f"candidate::{label}::promoted_count::{len(promoted)}",
    ]
    if promoted:
        first = min(promoted)
        if first <= hit_k + 2:
            features.append(f"candidate::{label}::promotion_window::boundary")
        elif first <= 10:
            features.append(f"candidate::{label}::promotion_window::near")
        else:
            features.append(f"candidate::{label}::promotion_window::deep")
    else:
        features.append(f"candidate::{label}::promotion_window::none")
    return features


def build_feature_maps(
    *,
    gold_rows: list[dict[str, Any]],
    base: dict[str, dict[str, Any]],
    candidates: dict[str, dict[str, dict[str, Any]]],
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, Any]]]:
    qid_to_tokens: dict[str, Counter[str]] = {}
    qid_metadata: dict[str, dict[str, Any]] = {}
    token_df: Counter[str] = Counter()
    for row in gold_rows:
        qid = str(row.get("qid", "")).strip()
        if not qid:
            continue
        counts = Counter(tokenize(question_text(row)))
        qid_to_tokens[qid] = counts
        token_df.update(counts.keys())
        metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
        qid_metadata[qid] = {
            "question": question_text(row),
            "type": metadata.get("type", "UNKNOWN"),
            "domain": metadata.get("domain", "UNKNOWN"),
        }

    n_qids = max(1, len(qid_to_tokens))
    max_df = max(1, int(float(args.max_token_df_frac) * n_qids))
    allowed_tokens = {
        token
        for token, df in token_df.items()
        if int(args.min_token_df) <= df <= max_df
    }
    idf = {
        token: math.log((1.0 + n_qids) / (1.0 + token_df[token])) + 1.0
        for token in allowed_tokens
    }

    feature_maps: dict[str, dict[str, float]] = {}
    for row in gold_rows:
        qid = str(row.get("qid", "")).strip()
        if not qid:
            continue
        question = question_text(row)
        features: dict[str, float] = {}
        token_scores = [
            (token, count * idf[token])
            for token, count in qid_to_tokens.get(qid, {}).items()
            if token in allowed_tokens
        ]
        token_scores.sort(key=lambda item: (-item[1], item[0]))
        for token, weight in token_scores[: max(0, int(args.max_tokens_per_query))]:
            features[f"token::{token}"] = float(weight)
        for cue in query_cues(question):
            features[f"cue::{cue}"] = 1.0
        base_row = base.get(qid)
        for feature in rank_state_features(base_row):
            features[f"rank::{feature}"] = 1.0
        for label, candidate_rows in candidates.items():
            for feature in candidate_signature_features(
                label=label,
                base_row=base_row,
                candidate_row=candidate_rows.get(qid),
                hit_k=int(args.hit_k),
            ):
                features[f"rank::{feature}"] = 1.0
        feature_maps[qid] = features
    return feature_maps, qid_metadata


def write_edges(path: Path, feature_maps: dict[str, dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for qid, features in sorted(feature_maps.items()):
            for feature, weight in sorted(features.items()):
                handle.write(
                    json.dumps(
                        {
                            "source": f"qid::{qid}",
                            "target": feature,
                            "weight": weight,
                            "kind": feature.split("::", 1)[0],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def feature_projection(
    feature_maps: dict[str, dict[str, float]],
    *,
    max_feature_df_frac: float,
    neighbor_threshold: float,
    max_neighbors: int,
) -> dict[str, dict[str, float]]:
    qids = sorted(feature_maps)
    norms = {
        qid: math.sqrt(sum(weight * weight for weight in features.values())) or 1.0
        for qid, features in feature_maps.items()
    }
    feature_to_qids: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for qid, features in feature_maps.items():
        for feature, weight in features.items():
            feature_to_qids[feature].append((qid, weight / norms[qid]))
    max_df = max(2, int(max_feature_df_frac * max(1, len(qids))))
    neighbors: dict[str, Counter[str]] = {qid: Counter() for qid in qids}
    for feature, postings in feature_to_qids.items():
        if len(postings) < 2 or len(postings) > max_df:
            continue
        for idx, (left_qid, left_weight) in enumerate(postings):
            for right_qid, right_weight in postings[idx + 1 :]:
                score = left_weight * right_weight
                if score <= 0:
                    continue
                neighbors[left_qid][right_qid] += score
                neighbors[right_qid][left_qid] += score
    pruned: dict[str, dict[str, float]] = {}
    for qid, counter in neighbors.items():
        items = [
            (other, score)
            for other, score in counter.most_common(max(0, int(max_neighbors)))
            if score >= float(neighbor_threshold)
        ]
        pruned[qid] = dict(items)
    return pruned


def label_propagation(neighbors: dict[str, dict[str, float]], iters: int) -> dict[str, str]:
    labels = {qid: qid for qid in sorted(neighbors)}
    for _ in range(max(1, int(iters))):
        changed = False
        for qid in sorted(neighbors):
            scores: Counter[str] = Counter()
            scores[labels[qid]] += 1e-6
            for other, weight in neighbors[qid].items():
                scores[labels[other]] += weight
            if not scores:
                continue
            best_label, _ = max(scores.items(), key=lambda item: (item[1], -len(item[0]), item[0]))
            if best_label != labels[qid]:
                labels[qid] = best_label
                changed = True
        if not changed:
            break
    remap: dict[str, str] = {}
    clusters: dict[str, list[str]] = defaultdict(list)
    for qid, label in labels.items():
        clusters[label].append(qid)
    for idx, (_label, members) in enumerate(
        sorted(clusters.items(), key=lambda item: (-len(item[1]), item[0])),
        start=1,
    ):
        for qid in members:
            remap[qid] = f"cluster_{idx:04d}"
    return remap


def summarize_cluster(
    *,
    qids: list[str],
    qid_metadata: dict[str, dict[str, Any]],
    feature_maps: dict[str, dict[str, float]],
    gold_by_qid: dict[str, dict[str, Any]],
    base: dict[str, dict[str, Any]],
    candidates: dict[str, dict[str, dict[str, Any]]],
    hit_k: int,
) -> dict[str, Any]:
    type_counts = Counter(str(qid_metadata.get(qid, {}).get("type", "UNKNOWN")) for qid in qids)
    domain_counts = Counter(str(qid_metadata.get(qid, {}).get("domain", "UNKNOWN")) for qid in qids)
    feature_scores: Counter[str] = Counter()
    for qid in qids:
        feature_scores.update(feature_maps.get(qid, {}))
    candidate_summaries: dict[str, Any] = {}
    for label, rows in candidates.items():
        movement_counts: Counter[str] = Counter()
        page_delta = 0
        doc_delta = 0
        for qid in qids:
            gold_row = gold_by_qid.get(qid, {})
            gold_pages = gold_page_uids(gold_row)
            gold_docs = gold_doc_ids(gold_row)
            base_page_rank, base_doc_rank = hit_rank(base.get(qid), gold_pages, gold_docs)
            cand_page_rank, cand_doc_rank = hit_rank(rows.get(qid), gold_pages, gold_docs)
            base_page_hit = base_page_rank is not None and base_page_rank <= hit_k
            cand_page_hit = cand_page_rank is not None and cand_page_rank <= hit_k
            base_doc_hit = base_doc_rank is not None and base_doc_rank <= hit_k
            cand_doc_hit = cand_doc_rank is not None and cand_doc_rank <= hit_k
            page_delta += int(cand_page_hit) - int(base_page_hit)
            doc_delta += int(cand_doc_hit) - int(base_doc_hit)
            movement_counts[movement_for_hit(base_page_rank, cand_page_rank, hit_k)] += 1
        candidate_summaries[label] = {
            "page_delta": page_delta,
            "doc_delta": doc_delta,
            "movement_counts": dict(sorted(movement_counts.items())),
        }
    return {
        "n": len(qids),
        "top_types": type_counts.most_common(10),
        "top_domains": domain_counts.most_common(10),
        "top_features": feature_scores.most_common(20),
        "candidate_summaries": candidate_summaries,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gold_rows = read_jsonl(Path(args.gold))
    gold_by_qid = {str(row.get("qid", "")).strip(): row for row in gold_rows if str(row.get("qid", "")).strip()}
    base = load_prediction(Path(args.base_prediction)) if args.base_prediction else {}
    candidates = {
        label: load_prediction(path)
        for label, path in (parse_labeled_path(raw) for raw in args.candidate)
    }

    feature_maps, qid_metadata = build_feature_maps(
        gold_rows=gold_rows,
        base=base,
        candidates=candidates,
        args=args,
    )
    write_edges(out_dir / "query_semantic_graph.edges.jsonl", feature_maps)
    (out_dir / "query_semantic_graph.features.json").write_text(
        json.dumps(feature_maps, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    neighbors = feature_projection(
        feature_maps,
        max_feature_df_frac=float(args.max_feature_df_frac),
        neighbor_threshold=float(args.neighbor_threshold),
        max_neighbors=int(args.max_neighbors),
    )
    labels = label_propagation(neighbors, int(args.label_prop_iters))
    clusters: dict[str, list[str]] = defaultdict(list)
    for qid, cluster_id in labels.items():
        clusters[cluster_id].append(qid)

    qid_cluster_rows = [
        {
            "qid": qid,
            "cluster": labels[qid],
            "question": qid_metadata.get(qid, {}).get("question", ""),
            "type": qid_metadata.get(qid, {}).get("type", "UNKNOWN"),
            "domain": qid_metadata.get(qid, {}).get("domain", "UNKNOWN"),
        }
        for qid in sorted(labels)
    ]
    (out_dir / "query_semantic_graph.qid_clusters.json").write_text(
        json.dumps(qid_cluster_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (out_dir / "query_semantic_graph.qid_clusters.jsonl").open("w", encoding="utf-8") as handle:
        for row in qid_cluster_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    cluster_summaries = []
    for cluster_id, qids in sorted(clusters.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(qids) < int(args.min_cluster_size):
            continue
        summary = summarize_cluster(
            qids=sorted(qids),
            qid_metadata=qid_metadata,
            feature_maps=feature_maps,
            gold_by_qid=gold_by_qid,
            base=base,
            candidates=candidates,
            hit_k=int(args.hit_k),
        )
        summary["cluster"] = cluster_id
        summary["qids"] = sorted(qids)
        cluster_summaries.append(summary)

    summary_payload = {
        "qid_count": len(feature_maps),
        "feature_node_count": len({feature for features in feature_maps.values() for feature in features}),
        "cluster_count": len(clusters),
        "reported_cluster_count": len(cluster_summaries),
        "args": vars(args),
        "clusters": cluster_summaries,
    }
    (out_dir / "query_semantic_graph.summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    subset_dir = out_dir / "cluster_qids"
    subset_dir.mkdir(exist_ok=True)
    for cluster in cluster_summaries:
        path = subset_dir / f"{cluster['cluster']}.qids.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for qid in cluster["qids"]:
                handle.write(json.dumps({"qid": qid}) + "\n")

    print(f"saved_edges: {out_dir / 'query_semantic_graph.edges.jsonl'}")
    print(f"saved_features: {out_dir / 'query_semantic_graph.features.json'}")
    print(f"saved_qid_clusters: {out_dir / 'query_semantic_graph.qid_clusters.json'}")
    print(f"saved_summary: {out_dir / 'query_semantic_graph.summary.json'}")
    print("qid_count:", summary_payload["qid_count"])
    print("feature_node_count:", summary_payload["feature_node_count"])
    print("cluster_count:", summary_payload["cluster_count"])
    print("reported_cluster_count:", summary_payload["reported_cluster_count"])
    for row in cluster_summaries[: int(args.top_clusters)]:
        print(
            json.dumps(
                {
                    "cluster": row["cluster"],
                    "n": row["n"],
                    "top_types": row["top_types"][:5],
                    "top_domains": row["top_domains"][:5],
                    "candidate_summaries": row["candidate_summaries"],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
