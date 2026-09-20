#!/usr/bin/env python3
"""Test the original Exact MaxSim runner with unforced CPU thread defaults.

Only standard-library imports are allowed before run_original. In particular,
do not import the runtime/diagnostic helpers: their CAPP dependency sets thread
environment variables. This is a diagnostic, not a historical-setting claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")


def write_new(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2)


def validate_inputs(inputs):
    qids = inputs["qids"]
    if len(qids) != 4 or len(set(qids)) != 4:
        raise ValueError("Require the four existing diagnostic questions")
    if any(set(inputs[key]) != set(qids) for key in ("questions", "baseline", "expected")):
        raise ValueError("Question/reference cohort mismatch")
    paths = inputs["paths"]
    for name in ("qids.jsonl", "gold.jsonl"):
        with Path(paths[name]).open() as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if [str(row["qid"]) for row in rows] != qids:
            raise ValueError("Original diagnostic input order changed")
        if name == "gold.jsonl" and any(row["question"] != inputs["questions"][str(row["qid"])] for row in rows):
            raise ValueError("Question text changed")
    baseline = json.loads(Path(paths["baseline.json"]).read_text())
    if set(baseline) != set(qids) or any(baseline[q]["page_retrieval_results"] != inputs["baseline"][q] for q in qids):
        raise ValueError("Diagnostic candidate input changed")
    if json.loads(Path(paths["empty_query.json"]).read_text()) != {} or Path(paths["empty_patch.jsonl"]).read_text().strip():
        raise ValueError("Expected empty auxiliary annotation inputs")
    for key in ("baseline", "expected"):
        for rows in inputs[key].values():
            if len(rows) != 1000 or len({tuple(row[:2]) for row in rows}) != 1000:
                raise ValueError("Expected 1,000 distinct candidate pages")
    c = inputs["config"]
    if c["scoring_query_device"] != "cpu" or c["query_filter"] != "full":
        raise ValueError("Expected CPU/full-query reconstruction")


def original_argv(inputs, output_dir):
    # Kept independent of benchmark imports; test against the existing helper.
    p, c = inputs["paths"], inputs["config"]
    return ["original-exact-native", "--qid-jsonl", p["qids.jsonl"], "--gold", p["gold.jsonl"],
        "--baseline-pred", p["baseline.json"], "--data-name", "m3-docvqa", "--split", "dev",
        "--embedding_name", c["embedding_name"], "--query_token_filter", "full",
        "--retrieval_model_name_or_path", c["backbone"], "--retrieval_adapter_model_name_or_path", c["adapter"],
        "--from-baseline-top-pages", "1000", "--base-score-source", "exact_page_maxsim",
        "--approx-base-page-token-topk", "0", "--weight-base", "1.0", "--weight-visual", "0.0",
        "--weight-non-visual", "0.0", "--weight-balance", "0.0", "--base-only-page-batch-size", "64",
        "--splice-query-token-labels", p["empty_query.json"], "--splice-patch-labels-jsonl", p["empty_patch.jsonl"],
        "--output-jsonl", str(output_dir / "original.jsonl"),
        "--output-summary-json", str(output_dir / "original.summary.json"),
        "--output-prediction-json", str(output_dir / "original.prediction.json")]


def check_native_environment(torch):
    overrides = {key: os.environ[key] for key in THREAD_VARS if key in os.environ}
    if overrides:
        raise ValueError(f"Native-default control has explicit thread overrides: {overrides}")
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    threads = torch.get_num_threads()
    if allocated < 1 or not 1 <= threads <= allocated:
        raise ValueError("Native CPU threads exceed allocation or allocation is missing")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Require exactly one visible CUDA GPU")
    cpuinfo = Path("/proc/cpuinfo")
    cpu_model = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                      if line.startswith("model name")), None) if cpuinfo.is_file() else None
    return {"cpu_threads": threads, "allocated_cpus": allocated,
        "interop_threads": torch.get_num_interop_threads(), "thread_environment": {},
        "torch_version": torch.__version__, "torch_parallel_info": torch.__config__.parallel_info(),
        "cpu_model": cpu_model, "hostname": platform.node(), "platform": platform.platform(),
        "visible_cpu_count": os.cpu_count(), "gpu": torch.cuda.get_device_name(0),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}


def run_original(inputs, output_dir):
    # Do not import any RACS benchmark/diagnostic helper before this run ends.
    import torch
    import run_visual_rerank_batch as runner
    from m3docrag.retrieval import ColPaliRetrievalModel

    env = check_native_environment(torch)
    print("NATIVE_EXACT_ENVIRONMENT " + json.dumps(env), flush=True)
    write_new(output_dir / "environment_before.json", env)
    captured = {}
    encode = ColPaliRetrievalModel.encode_query_with_metadata

    def capture(model, query, to_cpu=False, query_token_filter="full"):
        if len(captured) >= len(inputs["qids"]):
            raise ValueError("Unexpected extra query encoding")
        qid = inputs["qids"][len(captured)]
        if (query != inputs["questions"][qid] or not to_cpu or query_token_filter != "full"
                or str(model.model.device) != "cpu"):
            raise ValueError("Original query encoding inputs/device changed")
        # Keep the four tiny CPU tensors in memory. Serialize only after the
        # original run, so helper imports cannot influence native execution.
        meta = encode(model, query, to_cpu=to_cpu, query_token_filter=query_token_filter)
        captured[qid] = meta
        return meta

    previous = sys.argv
    try:
        sys.argv = original_argv(inputs, output_dir)
        with mock.patch.object(ColPaliRetrievalModel, "encode_query_with_metadata", capture):
            runner.main()
    finally:
        sys.argv = previous
    after = check_native_environment(torch)
    if env != after or list(captured) != inputs["qids"]:
        raise ValueError("Native thread environment or query cohort changed during run")
    return env, captured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        input_bytes = args.inputs.read_bytes()
        inputs = json.loads(input_bytes)
        validate_inputs(inputs)
        env, captured = run_original(inputs, args.output_dir)
        # The native run has FINISHED. Shared helpers may now be imported for
        # serialization/comparison; their environment changes cannot affect it.
        import diagnose_racs_exact_replay as exact
        summary = exact.bench.replay.read_json(args.output_dir / "original.summary.json")
        exact.online.check_exact_configuration(summary)
        if exact.online.scoring_options(summary) != inputs["config"]["exact_options"]:
            raise ValueError("Native runner scoring configuration differs from reference")
        validate_inputs(inputs)
        actual = exact.bench.replay.prediction_rows(exact.bench.replay.read_json(args.output_dir / "original.prediction.json"))
        if set(actual) != set(inputs["qids"]):
            raise ValueError("Native prediction cohort mismatch")
        comparisons = {q: exact.diagnostic.compare_candidate_rows(inputs["expected"][q], actual[q]["page_retrieval_results"])
                       for q in inputs["qids"]}
        queries = {q: exact.save_query(meta, args.output_dir, f"query_native_{q}") for q, meta in captured.items()}
        totals = {"questions": len(comparisons), "candidate_sets_match": sum(r["candidate_sets_match"] for r in comparisons.values()),
            "complete_order_and_score_matches": sum(r["complete_order_matches"] and r["scores_close"] for r in comparisons.values()),
            "top4_order_matches": sum(r["top4_order_matches"] for r in comparisons.values())}
        report = {"status": "diagnostic_complete_not_a_runtime_result", "environment": env,
            "totals": totals, "comparisons": comparisons, "queries": queries,
            "input_path": str(args.inputs), "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
            "code_sha256": {str(p): exact.bench.replay.sha256(p) for p in (
                Path(__file__), ROOT / "scripts/run_visual_rerank_batch.py", ROOT / "scripts/rerank_target_docs_visual_aware.py",
                ROOT / "src/m3docrag/retrieval/colpali.py")},
            "limitations": ["Current unforced CPU defaults, not recovered historical environment",
                "Four fixed diagnostic queries, not full upstream equivalence or publishable runtime",
                "No automatic setting selection, relaxed tolerance, retraining or old-output modification"]}
        write_new(args.output_dir / "diagnostic.json", report)
        print("NATIVE_EXACT_TOTALS " + json.dumps(totals), flush=True)
    except Exception as exc:
        write_new(args.output_dir / "failure.json", {"status": "diagnostic_failed_not_a_runtime_result",
            "type": type(exc).__name__, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
