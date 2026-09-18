#!/usr/bin/env python3
"""Read-only inventory for a faithful paired RACS pipeline benchmark.

Writes one new report, never runs models or changes existing artifacts. Saved
timers are inventoried, not treated as comparable end-to-end measurements.
"""

import argparse
from collections import Counter
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics


BASE = "m3docvqa_gpp_hyperlink_node_ablation_exact_maxsim/mmqa_dev_exact_maxsim_gpp_hyperlink_node_no_hyperlink.prediction.json"
LEGACY = "m3docvqa_gpp_hyperlink_node_ablation_mmr_target1"
MODEL_DIR = "m3docvqa_content_aware_exact_maxsim_direct_exactonly_adaptive_norm05"
MODEL_STEM = "mmqa_train_to_dev_content_aware_fixed_alpha_0p40_base_exact_maxsim_gpp_direct_exactonly_adaptive_norm05"
GOLD = "m3docvqa_mmqa_direct_evidence_pseudo_page_labels/mmqa_dev_pseudo_page_labels_direct_exactonly_adaptive_norm05.augmented_gold.jsonl"
GRAPH_PATHS = {"base_exact_maxsim_no_hyperlink": BASE, **{
    variant: f"{LEGACY}/mmqa_dev_gpp_hyperlink_node_{variant}.prediction.json"
    for variant in ("no_hyperlink", "docnode_to_hyperlink_docs", "pagenode_to_hyperlink_pages")}}
COMPACT_KEYS = ("dense_prediction_json", "sparse_prediction_json", "doc_pages_jsonl",
                "splade_index_pt", "pdf_hyperlink_edges_jsonl", "dense_top_pages",
                "sparse_top_pages", "final_top_pages", "final_selection_mode",
                "doc_doc_edge_mode", "doc_doc_edge_weight", "pdf_hyperlink_edge_weight",
                "pdf_hyperlink_target_mode", "restart_prob", "ppr_iters")
PATH_KEYS = ("dense_prediction_json", "sparse_prediction_json", "doc_pages_jsonl",
             "splade_index_pt", "pdf_hyperlink_edges_jsonl", "external_page_graph_jsonl",
             "learned_page_prior_jsonl", "doc_doc_page_embedding_dir")


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def file_info(path, root):
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    info = {"path": str(path), "exists": path.exists()}
    if path.exists():
        info.update(is_file=path.is_file(), bytes=path.stat().st_size)
    return info


def qid_digest(qids):
    return hashlib.sha256(json.dumps(sorted(qids), separators=(",", ":")).encode()).hexdigest()


def prediction_rows(payload):
    if isinstance(payload, dict) and isinstance(payload.get("predictions"), (dict, list)):
        payload = payload["predictions"]
    iterator = payload.items() if isinstance(payload, dict) else enumerate(payload)
    result = {}
    for key, row in iterator:
        if not isinstance(row, dict):
            raise ValueError("Non-object prediction row")
        qid = str(row.get("qid") or key).strip()
        if not qid or qid in result:
            raise ValueError(f"Empty/duplicate question ID: {qid}")
        result[qid] = row
    return result


def timer_inventory(rows):
    values = {}
    for row in rows.values():
        for key, value in row.items():
            if (key.startswith("time_") or key.endswith("_seconds")) and isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                values.setdefault(key, []).append(value)
    return {key: {"count": len(items), "mean": statistics.mean(items),
                  "min": min(items), "max": max(items),
                  "scope": "recorded field; timing boundary/hardware not verified"}
            for key, items in values.items()}


def inspect_graph(path, root, gold_qids):
    result = file_info(path, root)
    if not result.get("is_file"):
        return result
    rows = prediction_rows(read_json(result["path"]))
    if not rows:
        result["error"] = "Empty prediction file"
        return result
    first = next(iter(rows.values())).get("reranker_metadata", {})
    if not isinstance(first, dict):
        raise ValueError(f"Expected graph metadata object: {path}")
    varying = set()
    # Configs need not be inferred from filenames or today's wrapper defaults.
    for row in rows.values():
        metadata = row.get("reranker_metadata", {})
        varying.update(k for k in set(first) | set(metadata) if metadata.get(k) != first.get(k))
    dependencies = {key: file_info(first[key], root) for key in PATH_KEYS if first.get(key)}
    result.update(questions=len(rows), qid_sha256=qid_digest(rows),
                  qids_match_gold=set(rows) == gold_qids,
                  candidate_count_histogram=dict(Counter(len(r.get("page_retrieval_results", [])) for r in rows.values())),
                  first_question_metadata=first,
                  metadata_keys_varying_across_questions=sorted(varying),
                  compact_configuration={k: first.get(k) for k in COMPACT_KEYS},
                  dependencies=dependencies, recorded_timers=timer_inventory(rows))
    return result


def scalar_tree(value, depth=0):
    """Retain configuration/timing, excluding bulky metric arrays and text."""
    if depth > 4:
        return "[nested data omitted]"
    if isinstance(value, dict):
        return {k: scalar_tree(v, depth + 1) for k, v in value.items()
                if k not in {"metrics", "per_qid", "questions", "predictions", "rows", "weights", "mean", "std"}}
    if isinstance(value, list):
        if len(value) <= 40 and all(isinstance(x, (str, int, float, bool, type(None))) for x in value):
            return value
        return {"omitted_list_length": len(value)}
    return value


def inspect_small_json(path, root, byte_limit=4 * 1024 * 1024):
    info = file_info(path, root)
    if not info.get("is_file"):
        return info
    if info["bytes"] > byte_limit:
        info["not_read"] = "size limit; not assumed absent"
    else:
        try:
            info["record"] = scalar_tree(read_json(info["path"]))
        except (ValueError, OSError) as error:
            info["error"] = str(error)
    return info


def tuning_candidates(output, root):
    records = []
    # Only small, relevant saved models; no predictions/logs or broad text search.
    for path in sorted(output.rglob("*.model.json")):
        if "content_aware" not in str(path) or "exact_maxsim" not in str(path):
            continue
        model_info = inspect_small_json(path, root)
        model = model_info.get("record", {})
        settings = model.get("args", {})
        if settings.get("feature_set", "all") != "all":
            continue
        summary_path = path.with_name(path.name.replace(".model.json", ".summary.json"))
        summary = inspect_small_json(summary_path, root)
        s = summary.get("record", {})
        records.append({"model_path": str(path), "feature_count": len(model.get("feature_names", [])),
                        "blend_alpha": settings.get("blend_alpha"),
                        "auto_tune_blend_alpha": settings.get("auto_tune_blend_alpha"),
                        "model_tuning_summary": model.get("tuning_summary"),
                        "train_metadata": model.get("train_metadata"),
                        "summary_path": str(summary_path),
                        "summary_tuning": s.get("tuning_summary"),
                        "inputs": {k: s.get(k) for k in ("train_gold", "eval_gold", "train_base_pred", "eval_base_pred")},
                        "read_errors": [r.get("error", r.get("not_read")) for r in (model_info, summary) if r.get("error") or r.get("not_read")]})
    return records


def audit(root):
    output = root / "output"
    gold_path = output / GOLD
    with gold_path.open() as handle:
        qids = [str(json.loads(line)["qid"]).strip() for line in handle if line.strip()]
    if len(qids) != len(set(qids)) or not all(qids):
        raise ValueError("Gold question IDs are empty or duplicated")
    graphs = {}
    upstream = {}
    for label, relative in GRAPH_PATHS.items():
        graph = inspect_graph(output / relative, root, set(qids))
        graphs[label] = graph
        print("GRAPH_PREREQUISITE " + json.dumps({"label": label,
              "exists": graph["exists"], "questions": graph.get("questions"),
              "qids_match_gold": graph.get("qids_match_gold"),
              "configuration": graph.get("compact_configuration"),
              "missing_recorded_paths": [v["path"] for v in graph.get("dependencies", {}).values() if not v["exists"]]}), flush=True)
        for key in ("dense_prediction_json", "sparse_prediction_json"):
            dependency = graph.get("dependencies", {}).get(key)
            if dependency:
                path = Path(dependency["path"])
                if path not in upstream:
                    summary = path.with_name(path.name.replace(".prediction.json", ".summary.json"))
                    upstream[path] = {"prediction": dependency, "summary": inspect_small_json(summary, root)}
        gc.collect()
    model_path = output / MODEL_DIR / (MODEL_STEM + ".model.json")
    candidates = tuning_candidates(output, root)
    bge = inspect_small_json(output / "racs_bge_reader_top4_15911612/validated_result.json", root)
    graphs_present = all(g.get("exists") for g in graphs.values())
    cohorts_match = all(g.get("qids_match_gold") for g in graphs.values())
    report = {"status": "inventory_only_not_benchmark", "root": str(root),
              "gold": {"path": str(gold_path), "questions": len(qids), "qid_sha256": qid_digest(qids)},
              "graphs": graphs, "upstream": list(upstream.values()),
              "full_model": inspect_small_json(model_path, root),
              "full_summary": inspect_small_json(model_path.with_name(MODEL_STEM + ".summary.json"), root),
              "tuning_candidates": candidates, "bge_validated_result": bge,
              "checks": {"expected_2441_gold": len(qids) == 2441,
                         "four_graph_artifacts_present": graphs_present, "graph_qids_match_gold": cohorts_match},
              "not_established": ["online retrieval/replay equivalence", "end-to-end latency or throughput",
                                  "incremental CPU/GPU memory", "original alpha-selection provenance merely from matching values"]}
    compact_tuning = []
    for candidate in candidates:
        train = candidate.get("train_metadata") or {}
        tuning = candidate.get("summary_tuning") or candidate.get("model_tuning_summary") or train.get("tuning_summary") or {}
        compact_tuning.append({"model": Path(candidate["model_path"]).name,
                               "alpha": candidate["blend_alpha"],
                               "auto_tune": candidate["auto_tune_blend_alpha"],
                               "selected_alpha": tuning.get("selected_blend_alpha"),
                               "fit_qids": train.get("fit_qid_count"),
                               "tune_qids": train.get("tune_qid_count"),
                               "retrained": train.get("retrained_after_tuning"),
                               "eval_gold": candidate["inputs"].get("eval_gold")})
    compact_tuning.sort(key=lambda r: (r["selected_alpha"] is None, r["model"] != MODEL_STEM + ".model.json", r["model"]))
    print("TUNING_CANDIDATES " + json.dumps({"total": len(candidates), "first_20": compact_tuning[:20],
          "full_records_in_report": True}), flush=True)
    for value in upstream.values():
        summary = value["summary"]
        record = summary.get("record", {})
        keys = ("baseline_pred", "qid_jsonl", "gold", "data_name", "split", "embedding_name",
                "index_pt", "model_name_or_path", "encoder_backend", "base_score_source",
                "approx_base_page_token_topk", "from_baseline_top_pages", "top_pages",
                "query_topk_terms", "max_length", "batch_size", "device")
        print("UPSTREAM_SUMMARY " + json.dumps({"prediction": value["prediction"],
              "summary_path": summary["path"], "summary_exists": summary["exists"],
              "configuration": {k: record[k] for k in keys if k in record},
              "error": summary.get("error", summary.get("not_read")),
              "full_record_in_report": True}), flush=True)
    print("BGE_VALIDATED_RESULT " + json.dumps(bge), flush=True)
    print("PREREQUISITE_CHECKS " + json.dumps(report["checks"]), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_json}")
    report = audit(args.repo_root.resolve())
    # Parent must already exist; launcher creates a new isolated job directory.
    with args.output_json.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"saved_runtime_prerequisites={args.output_json}")


if __name__ == "__main__":
    main()
