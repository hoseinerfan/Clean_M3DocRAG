#!/usr/bin/env python3
"""CPU timing of fixed-blend logistic CAPP with cached upstream predictions.

No training, retrieval, graph construction, reader inference, or prediction-file
writing is performed. Feature construction mirrors ca.score_records and is
tested against that implementation. Only a new JSON report is written.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

# Set before importing NumPy. This benchmark deliberately uses one CPU thread.
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
for variable in THREAD_VARS:
    os.environ[variable] = "1"

import numpy as np
import train_content_aware_pseudo_page_reranker as ca


def validate_model(model):
    settings = model.get("args", {})
    for key in ("candidate_top_k", "inference_mode", "blend_alpha"):
        if key not in settings:
            raise ValueError(f"Missing saved setting: {key}; refusing to guess")
    if model.get("model_type", settings.get("model_type", "logistic")) != "logistic":
        raise ValueError("Only logistic CAPP is supported")
    flags = ("query_adaptive_alpha", "learned_query_alpha", "learned_alpha_action",
             "learned_alpha_utility_gate", "base_aware_alpha_utility_gate")
    blocks = (model, settings, model.get("train_metadata", {}))
    if any(block.get("adaptive_alpha_config") or any(block.get(k) for k in flags)
           for block in blocks):
        raise ValueError("Only fixed-alpha CAPP is supported")
    if settings["inference_mode"] != "blend_rerank":
        raise ValueError("Only blend_rerank is supported")
    if int(settings["candidate_top_k"]) < 1 or not 0 <= float(settings["blend_alpha"]) <= 1:
        raise ValueError("Invalid candidate count or blend alpha")
    names = model["feature_names"]
    if not names or len(set(names)) != len(names) or set(names) - set(ca.BASE_FEATURE_NAMES):
        raise ValueError("Expected unique nonvisual feature names")
    for key in ("mean", "std", "weights"):
        values = np.asarray(model[key], dtype=np.float32)
        if values.shape != (len(names),) or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid model {key}")
    if np.any(np.asarray(model["std"]) <= 0) or not math.isfinite(float(model["bias"])):
        raise ValueError("Invalid standard deviations or bias")
    return SimpleNamespace(**settings)


def parse_sources(entries):
    sources = {}
    for entry in entries:
        label, sep, value = entry.partition("=")
        if not sep or not label or not value or label in sources:
            raise ValueError(f"Expected unique LABEL=PATH: {entry}")
        sources[label] = Path(value)
    return sources


def feature_matrix(qid, question, records, pages, sources, names):
    """Keep operations/dtypes in the same order as ca.score_records."""
    normalized = ca.normalize_scores(records)
    doc_rank, page_rank, page_count = ca.doc_rank_maps(records)
    profile = ca.question_profile({"question": question})
    vectors = [ca.feature_vector(
        row, records=records, base_norm_scores=normalized, doc_rank=doc_rank,
        doc_page_rank=page_rank, doc_page_count=page_count, question=profile,
        page_features=pages, source_maps_by_label=sources, qid=qid,
        feature_names=names,
    ) for row in records]
    return np.asarray(vectors, dtype=np.float32)


def timed_query(qid, row, question, pages, sources, model, settings, parameters):
    mean, std, weights, bias = parameters
    start = time.perf_counter()
    records = ca.ranked_page_records(row, settings.candidate_top_k)
    prepared = time.perf_counter()
    X = feature_matrix(qid, question, records, pages, sources, model["feature_names"])
    featured = time.perf_counter()
    probabilities = ca.predict_scorer_proba(ca.standardize_eval(X, mean, std), weights, bias)
    for record, score in zip(records, probabilities):
        record["learned_score"] = float(score)
    scored = time.perf_counter()
    ordered = ca.rerank_records(records, settings)
    uids = [record["uid"] for record in ordered]
    finished = time.perf_counter()
    return uids, {
        "candidate_preparation": prepared - start,
        "feature_extraction": featured - prepared,
        "standardization_and_scoring": scored - featured,
        "blending_and_sorting": finished - scored,
        "total": finished - start,
    }


def summarize(values):
    values = sorted(values)
    total = sum(values)
    return {"n": len(values), "total_seconds": total,
            "mean_seconds": statistics.mean(values),
            "median_seconds": statistics.median(values),
            "p95_seconds": values[math.ceil(0.95 * len(values)) - 1],
            "queries_per_second": len(values) / total if total > 0 else None}


def order_digest(uids):
    return hashlib.sha256(json.dumps(uids, separators=(",", ":")).encode()).hexdigest()


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def peak_rss_mib():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 ** 2 if sys.platform == "darwin" else 1024)


def command_output(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("model-json", "base-pred", "question-jsonl", "page-text-jsonl", "saved-pred", "output-json"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--source", action="append", default=[], required=True)
    parser.add_argument("--expected-qids", type=int, default=2441)
    parser.add_argument("--expected-candidates", type=int, default=1000)
    parser.add_argument("--warmup-passes", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if min(args.expected_qids, args.expected_candidates, args.repeats) < 1 or args.warmup_passes < 1:
        parser.error("Counts, repeats and warmup-passes must be positive")
    if args.output_json.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_json}")
    source_paths = parse_sources(args.source)
    paths = {"model": args.model_json, "base": args.base_pred,
             "questions": args.question_jsonl, "page_text": args.page_text_jsonl,
             "saved_prediction": args.saved_pred,
             **{"source:" + k: v for k, v in source_paths.items()}}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    prep = {}

    def measure(name, operation):
        start = time.perf_counter()
        result = operation()
        prep[name] = time.perf_counter() - start
        print(f"preparation {name}={prep[name]:.3f}s", flush=True)
        return result

    model = measure("model_load", lambda: json.loads(args.model_json.read_text()))
    settings = validate_model(model)
    if settings.candidate_top_k != args.expected_candidates:
        raise ValueError("Saved candidate_top_k differs from expected-candidates")
    base = measure("base_prediction_load", lambda: ca.load_prediction(args.base_pred))
    if len(base) != args.expected_qids:
        raise ValueError(f"Expected {args.expected_qids} questions, got {len(base)}")
    gold = measure("question_file_load", lambda: ca.load_gold(args.question_jsonl))
    questions = {}
    for qid in base:
        question = str(gold.get(qid, {}).get("question", "")).strip()
        if not question:
            raise ValueError(f"Missing question text: {qid}")
        questions[qid] = question
    # Never pass answers, supporting documents, or pseudo-page labels to scoring.
    del gold
    pages = measure("page_text_load_and_tokenization", lambda: ca.load_page_features(args.page_text_jsonl))
    sources = {}
    for label, path in source_paths.items():
        raw_source = measure(f"source_load:{label}", lambda p=path: ca.load_prediction(p))
        if set(base) - set(raw_source):
            raise ValueError(f"Source {label} is missing questions")
        sources[label] = measure(f"source_map_preparation:{label}",
                                lambda: ca.source_maps(raw_source, settings.candidate_top_k))
        del raw_source
    gc.collect()
    inference_inputs_peak = peak_rss_mib()

    # Validation/reference I/O is not part of reported preparation or inference.
    validation_start = time.perf_counter()
    saved = ca.load_prediction(args.saved_pred)
    if set(saved) != set(base):
        raise ValueError("Saved/base question sets differ")
    expected = {}
    for qid, row in base.items():
        records = ca.ranked_page_records(row, settings.candidate_top_k)
        uids = [record["uid"] for record in records]
        saved_uids = ca.ranked_pages(saved[qid])
        if len(uids) != args.expected_candidates or set(uids) != set(saved_uids):
            raise ValueError(f"Candidate count/set mismatch: {qid}")
        if any(uid not in pages for uid in uids):
            raise ValueError(f"Missing candidate page text: {qid}")
        expected[qid] = order_digest(saved_uids)
    del saved
    gc.collect()
    reference_precheck_seconds = time.perf_counter() - validation_start
    mean = np.asarray(model["mean"], dtype=np.float32)
    std = np.asarray(model["std"], dtype=np.float32)
    weights, bias = ca.scorer_from_model_json(model)
    parameters = mean, std, weights, bias
    passes = []
    query_totals = {qid: [] for qid in base}
    for pass_index in range(args.warmup_passes + args.repeats):
        warmup = pass_index < args.warmup_passes
        stage_values = {}
        wall_start = time.perf_counter()
        for index, (qid, row) in enumerate(base.items(), 1):
            uids, timings = timed_query(qid, row, questions[qid], pages, sources,
                                       model, settings, parameters)
            # Validate every warmup and measured pass, outside query timing.
            if order_digest(uids) != expected[qid]:
                raise ValueError(f"Ranking mismatch at {qid}, pass {pass_index + 1}; no successful report written")
            for stage, value in timings.items():
                stage_values.setdefault(stage, []).append(value)
            if not warmup:
                query_totals[qid].append(timings["total"])
            if index % 250 == 0:
                print(f"pass={pass_index + 1} warmup={warmup} questions={index}/{len(base)}", flush=True)
        result = {"warmup": warmup, "stages": {k: summarize(v) for k, v in stage_values.items()},
                  "pass_wall_seconds_including_validation": time.perf_counter() - wall_start,
                  "identical_complete_rankings": len(base)}
        passes.append(result)
        print(f"pass={pass_index + 1} warmup={warmup} total={result['stages']['total']}", flush=True)
    benchmark_peak = peak_rss_mib()
    # Hash after timing so fingerprint reads do not prewarm the measured file loads.
    inputs = {name: fingerprint(path) for name, path in paths.items()}
    numpy_config = io.StringIO()
    with redirect_stdout(numpy_config):
        np.show_config()
    report = {
        "status": "validated", "scope": "cached-input CPU CAPP inference; not end-to-end RAG",
        "excluded_costs": ["dense/sparse retrieval and upstream auxiliary graph generation",
                           "original page text extraction", "reader/model loading and QA",
                           "training", "prediction serialization", "reference validation and fingerprinting"],
        "questions": len(base), "candidate_top_k": settings.candidate_top_k,
        "feature_names": model["feature_names"], "blend_alpha": settings.blend_alpha,
        "source_labels": list(sources), "input_fingerprints": inputs,
        "code_fingerprints": {"benchmark": fingerprint(Path(__file__)), "scorer": fingerprint(Path(ca.__file__))},
        "preparation_seconds": prep, "preparation_total_seconds": sum(prep.values()),
        "preparation_note": "One observed file-load/preparation pass; filesystem cache not controlled. Not a cold-start claim.",
        "reference_precheck_seconds_excluded": reference_precheck_seconds,
        "warmup_passes": args.warmup_passes, "measured_repeats": args.repeats, "passes": passes,
        "per_question_mean_across_repeats": summarize([statistics.mean(v) for v in query_totals.values()]),
        "peak_rss_mib_after_inference_input_preparation": inference_inputs_peak,
        "process_peak_rss_mib_through_benchmark": benchmark_peak,
        "memory_note": "Process high-water marks, not incremental CAPP-only memory. Includes cached inputs, temporary loads and validation reference; allocator retention is possible.",
        "environment": {"hostname": platform.node(), "platform": platform.platform(),
                        "cpu_description": command_output(["lscpu"]) if sys.platform == "linux" else platform.processor(),
                        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                        "python": sys.version, "numpy": np.__version__, "numpy_config": numpy_config.getvalue(),
                        "thread_environment": {key: os.environ[key] for key in THREAD_VARS},
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
                        "git_commit": command_output(["git", "rev-parse", "HEAD"])},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print("VALIDATED_CAPP_RUNTIME " + json.dumps(report["per_question_mean_across_repeats"]))
    print(f"saved_report={args.output_json}")


if __name__ == "__main__":
    main()
