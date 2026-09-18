#!/usr/bin/env python3
"""Bounded graph replay from cached retrieval; NOT an end-to-end benchmark.

No predictions or models are changed. Only a new validation report is written.
Recorded arguments override an explicit, reported legacy-compatibility map.
No unlisted current default is silently used to fill missing saved arguments.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from types import SimpleNamespace

import graph_rerank_page_retrieval_predictions as graph
from audit_racs_runtime_prerequisites import prediction_rows, qid_digest, read_json


# These are hypotheses for options absent from older saved metadata, not a
# claim to have recovered the historical CLI. Replay must test their output.
LEGACY_COMPAT = {
    "neighbor_expansion_window": 0, "expand_neighbors_from_top_dense_pages": 50,
    "expand_neighbors_from_top_sparse_pages": 50, "neighbor_seed_weight": 0.25,
    "doc_doc_page_embedding_dir": "", "doc_doc_embedding_pooling": "page_seed_weighted_mean",
    "doc_doc_embedding_page_top_k": 0, "doc_doc_embedding_min_similarity": 0.0,
    "doc_doc_embedding_cache_docs": 128, "doc_doc_embedding_mutual_top_k": 3,
    "doc_doc_rescue_anchor_top_k": 4, "doc_doc_rescue_rank_min": 5,
    "doc_doc_rescue_rank_max": 20, "doc_doc_confidence_mode": "none",
    "doc_doc_confidence_margin": 0.05, "learned_page_prior_jsonl": "",
    "learned_page_prior_score_field": "learned_score", "learned_page_prior_seed_weight": 0.0,
    "learned_page_prior_min_base_rank": 1, "learned_page_prior_max_base_rank": 1000,
    "learned_page_prior_top_k": 0, "learned_page_prior_normalize": True,
    "ppr_iteration_mode": "fixed", "ppr_graph_size_metric": "nodes",
    "ppr_graph_size_reference": 1000.0, "ppr_graph_size_min_iters": 5,
    "ppr_graph_size_max_iters": 80, "ppr_convergence_tol": 1e-7,
    "ppr_convergence_min_iters": 5,
}
IO_ARGS = {"gold", "question_type", "output_prediction_json", "output_summary_json"}
DISABLED_MODES = ("query_anchor_evidence_mode", "constraint_competition_mode",
                  "heading_breadcrumb_mode", "entity_alias_mode", "position_evidence_mode")


@contextmanager
def argv_for_parser():
    previous = sys.argv
    sys.argv = ["graph", "--dense-prediction-json", "unused", "--sparse-prediction-json",
                "unused", "--output-prediction-json", "unused", "--output-summary-json", "unused"]
    try:
        yield
    finally:
        sys.argv = previous


def settings_from_metadata(metadata):
    with argv_for_parser():
        declared = set(vars(graph.parse_args()))
    missing = declared - set(metadata) - set(LEGACY_COMPAT) - IO_ARGS
    if missing:
        raise ValueError(f"Unrecorded arguments without explicit compatibility values: {sorted(missing)}")
    applied = {k: v for k, v in LEGACY_COMPAT.items() if k not in metadata}
    values = {**applied, **{k: v for k, v in metadata.items() if k in declared - IO_ARGS}}
    args = SimpleNamespace(**values)
    if any(getattr(args, name) != "none" for name in DISABLED_MODES):
        raise ValueError("Replay supports only the audited disabled text/position modes")
    if args.doc_doc_edge_mode not in ("none", "hyperlink_citation"):
        raise ValueError("Unsupported document edge mode")
    if args.expansion_top_pages or args.neighbor_expansion_window:
        raise ValueError("This replay does not load expansion indexes")
    if args.external_page_graph_jsonl or args.learned_page_prior_jsonl or args.doc_doc_page_embedding_dir:
        raise ValueError("Unsupported extra graph/prior/embedding input")
    return args, applied


def select_qids(qids, count):
    if not 1 <= count <= len(qids):
        raise ValueError("Invalid sample count")
    return sorted(qids, key=lambda q: (hashlib.sha256(("racs-graph-replay-v1:" + q).encode()).hexdigest(), q))[:count]


def compare_rows(expected, actual, count):
    def normalize(rows):
        if len(rows) != count:
            raise ValueError(f"Expected {count} pages, got {len(rows)}")
        pairs, scores = [], []
        for row in rows:
            if len(row) < 3 or not str(row[0]) or isinstance(row[1], bool) or int(row[1]) != row[1]:
                raise ValueError("Malformed page row")
            pairs.append((str(row[0]), int(row[1])))
            score = float(row[2])
            if not math.isfinite(score):
                raise ValueError("Nonfinite page score")
            scores.append(score)
        if len(set(pairs)) != count:
            raise ValueError("Duplicate candidate page")
        return pairs, scores
    reference, reference_scores = normalize(expected)
    observed, observed_scores = normalize(actual)
    order_matches = reference == observed
    return {"candidate_sets_match": set(reference) == set(observed),
            "complete_order_matches": order_matches,
            "scores_close": order_matches and all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-8)
                                                      for a, b in zip(reference_scores, observed_scores)),
            "max_aligned_absolute_score_difference": max((abs(a-b) for a, b in zip(reference_scores, observed_scores)), default=0)
                                                       if order_matches else None}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replay(audit_path, sample_size):
    audit = read_json(audit_path)
    if not all(audit["checks"].values()) or len(audit["graphs"]) != 4:
        raise ValueError("Prerequisite audit checks not satisfied")
    root = Path(audit["root"])
    gold_path = Path(audit["gold"]["path"])
    with gold_path.open() as handle:
        gold_qids = [str(json.loads(line)["qid"]) for line in handle if line.strip()]
    if len(set(gold_qids)) != len(gold_qids) or qid_digest(gold_qids) != audit["gold"]["qid_sha256"]:
        raise ValueError("Gold cohort changed since audit")
    qids = select_qids(gold_qids, sample_size)
    fingerprints = {str(audit_path): sha256(audit_path), str(gold_path): sha256(gold_path)}
    cache, catalogs, hyperlinks = {}, {}, {}

    def absolute(path):
        value = Path(path)
        return value if value.is_absolute() else root / value

    def selected_rows(path):
        path = absolute(path)
        if path not in cache:
            all_rows = prediction_rows(read_json(path))
            if qid_digest(all_rows) != audit["gold"]["qid_sha256"]:
                raise ValueError(f"Prediction cohort differs: {path}")
            cache[path] = {q: all_rows[q] for q in qids}
            fingerprints[str(path)] = sha256(path)
        return cache[path]

    results = {}
    for label, item in audit["graphs"].items():
        metadata = item["first_question_metadata"]
        args, applied = settings_from_metadata(metadata)
        if any(getattr(args, key) != 1000 for key in ("dense_top_pages", "sparse_top_pages", "final_top_pages")):
            raise ValueError("Unexpected candidate depth")
        saved = selected_rows(item["path"])
        for qid, row in saved.items():
            current = {k: v for k, v in row["reranker_metadata"].items() if k != "graph"}
            expected = {k: v for k, v in metadata.items() if k != "graph"}
            if current != expected:
                raise ValueError(f"Saved metadata changed since audit: {label}, {qid}")
        dense = selected_rows(args.dense_prediction_json)
        sparse = selected_rows(args.sparse_prediction_json)
        catalogue_path = absolute(args.doc_pages_jsonl)
        if catalogue_path not in catalogs:
            # Audited modes above all disable text/heading/entity features.
            catalogs[catalogue_path] = graph.load_doc_page_catalog(catalogue_path)
            fingerprints[str(catalogue_path)] = sha256(catalogue_path)
        link_graph = None
        if args.pdf_hyperlink_edges_jsonl:
            link_path = absolute(args.pdf_hyperlink_edges_jsonl)
            if link_path not in hyperlinks:
                hyperlinks[link_path] = graph.load_pdf_hyperlink_graph(link_path)
                fingerprints[str(link_path)] = sha256(link_path)
            link_graph = hyperlinks[link_path]
        comparisons = []
        for qid in qids:
            actual, _ = graph.build_qid_graph_ranking(qid=qid, dense_row=dense[qid], sparse_row=sparse[qid],
                args=args, doc_page_catalog=catalogs[catalogue_path], pdf_hyperlink_graph=link_graph,
                gold_row=None)
            comparisons.append({"qid": qid, **compare_rows(saved[qid]["page_retrieval_results"], actual, 1000)})
        counts = {key: sum(row[key] for row in comparisons)
                  for key in ("candidate_sets_match", "complete_order_matches", "scores_close")}
        results[label] = {"questions": len(qids), **counts, "compatibility_values_applied": applied,
                          "comparisons": comparisons}
        print("GRAPH_REPLAY " + json.dumps({"label": label, "questions": len(qids), **counts}), flush=True)
    passed = all(r["complete_order_matches"] == len(qids) and r["scores_close"] == len(qids) for r in results.values())
    return {"status": "sample_graph_replay_validated" if passed else "sample_graph_replay_mismatch",
            "scope": "fixed 16-question default sample of graph stages from cached dense/sparse rankings; not end-to-end timing",
            "selection_rule": "first N SHA256(racs-graph-replay-v1:qid), no outcome filtering",
            "questions": qids, "results": results, "input_sha256": fingerprints,
            "code_sha256": {str(Path(__file__)): sha256(__file__), str(Path(graph.__file__)): sha256(graph.__file__)},
            "python": sys.version, "hostname": platform.node(),
            "not_established": ["full-dataset graph replay", "dense/sparse inference replay", "original invocation",
                                "CAPP feature/scoring replay from regenerated graphs", "end-to-end runtime or memory"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=16)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_json}")
    # Exclusive creation protects existing files even if execution later fails.
    with args.output_json.open("x") as handle:
        try:
            result = replay(args.audit_json, args.sample_size)
        except Exception as exc:
            json.dump({"status": "failed", "error": str(exc), "type": type(exc).__name__}, handle, indent=2)
            raise
        json.dump(result, handle, indent=2, allow_nan=False)
    print("saved_graph_replay_report=" + str(args.output_json), flush=True)
    if result["status"] != "sample_graph_replay_validated":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
