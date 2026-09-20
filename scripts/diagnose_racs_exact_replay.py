#!/usr/bin/env python3
"""Four-question Exact MaxSim diagnosis. Not a runtime result or auto-fix.

Run the original entry point with eight CPU threads and with one thread,
capture its actual query embeddings, then replay the runtime scoring helper on
those fixed embeddings. Never retrain, modify old artifacts or relax checks.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from unittest import mock

import benchmark_racs_online as online
import diagnose_racs_faiss_replay as diagnostic

bench = online.bench
ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_MODES = ("original_eight_threads", "original_one_thread")
HARNESS_MODES = tuple(f"{mode}_{context}" for mode in ORIGINAL_MODES
                      for context in ("no_grad", "inference_mode")) + ("prior_saved_gpu_query",)


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def prepare(bundle_path, summary_path, prior_dir, output_dir):
    bundle = bench.replay.read_json(bundle_path)
    qids = diagnostic.diagnostic_qids(bundle)
    summary = bench.replay.read_json(summary_path)
    online.check_exact_configuration(summary)
    config = bundle["online"]
    if online.scoring_options(summary) != config["exact_options"]:
        raise ValueError("Saved summary and runtime Exact MaxSim options disagree")
    if summary["embedding_name"] != config["embedding_name"]:
        raise ValueError("Embedding names disagree")
    if config.get("scoring_query_device") != "cpu" or config.get("query_filter") != "full":
        raise ValueError("Diagnostic expects the current CPU/full-query reconstruction")
    exact_path = config["roles"]["exact"]
    expected = {qid: bundle["inputs"][exact_path][qid]["page_retrieval_results"] for qid in qids}
    for rows in expected.values():
        bench.replay.compare_rows(rows, rows, 1000)
    gold, duplicates = {}, []
    with Path(summary["gold"]).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["qid"])
            if qid in qids:
                if qid in gold:
                    duplicates.append(qid)
                gold[qid] = row
    if duplicates or set(gold) != set(qids):
        raise ValueError("Missing/duplicate diagnostic gold questions")
    # A text difference is itself an actionable diagnosis, not permission to
    # silently normalize inputs and claim an equivalent experiment.
    for qid in qids:
        if gold[qid]["question"] != bundle["questions"][qid]:
            raise ValueError(f"Original/runtime question text differs: {qid}")
    prior_report = prior_dir / "diagnostic.json"
    prior = bench.replay.read_json(prior_report)
    if (prior.get("status") != "diagnostic_complete_not_a_runtime_result"
            or prior["questions"] != qids):
        raise ValueError("Prior query diagnostic/cohort mismatch")
    gpu_queries = {}
    for qid in qids:
        matches = [row for row in prior["query_reports"]
                   if row["qid"] == qid and row["encoder_path"] == "direct_gpu"]
        if len(matches) != 1:
            raise ValueError("Missing/duplicate prior GPU query metadata")
        path = prior_dir / f"query_direct_gpu_{qid}.npz"
        gpu_queries[qid] = {"path": str(path.resolve()), "summary": matches[0],
                            "file_sha256": bench.replay.sha256(path)}
    paths = {name: str((output_dir / name).resolve()) for name in
             ("qids.jsonl", "gold.jsonl", "baseline.json", "empty_query.json", "empty_patch.jsonl")}
    write_jsonl(Path(paths["qids.jsonl"]), ({"qid": qid} for qid in qids))
    write_jsonl(Path(paths["gold.jsonl"]), (gold[qid] for qid in qids))
    bench.write_new(Path(paths["baseline.json"]), {qid: {
        "page_retrieval_results": config["baseline_references"][qid]} for qid in qids})
    bench.write_new(Path(paths["empty_query.json"]), {})
    write_jsonl(Path(paths["empty_patch.jsonl"]), [])
    inputs = {"qids": qids, "questions": {qid: gold[qid]["question"] for qid in qids},
              "baseline": {qid: config["baseline_references"][qid] for qid in qids},
              "expected": expected, "config": {k: v for k, v in config.items() if k != "baseline_references"},
              "paths": paths, "gpu_queries": gpu_queries,
              "input_sha256": {str(p.resolve()): bench.replay.sha256(p)
                  for p in (bundle_path, summary_path, prior_report, Path(summary["gold"]))}}
    bench.write_new(output_dir / "inputs.json", inputs)
    return inputs


def original_argv(inputs, output_dir):
    p, c = inputs["paths"], inputs["config"]
    return ["original-exact-diagnostic", "--qid-jsonl", p["qids.jsonl"], "--gold", p["gold.jsonl"],
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


def environment(torch):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA GPU required")
    return {"torch_version": torch.__version__, "cpu_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(), "gpu": torch.cuda.get_device_name(0),
            "hostname": platform.node(), "platform": platform.platform(), "visible_cpu_count": os.cpu_count(),
            "thread_environment": {key: os.environ.get(key) for key in
                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "SLURM_CPUS_PER_TASK")},
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}


def save_query(meta, output_dir, name):
    import numpy as np
    summary = diagnostic.query_summary(meta)
    path = output_dir / f"{name}.npz"
    with path.open("xb") as handle:
        np.savez(handle, embeddings=meta["embeddings"].float().numpy(), token_ids=meta["token_ids"].numpy())
    return {"path": str(path.resolve()), "summary": summary, "file_sha256": bench.replay.sha256(path)}


def load_query(record, torch):
    import numpy as np
    path = Path(record["path"])
    if bench.replay.sha256(path) != record["file_sha256"]:
        raise ValueError("Saved diagnostic query file changed")
    with np.load(path, allow_pickle=False) as data:
        values, ids = data["embeddings"].copy(), data["token_ids"].copy()
    summary = record["summary"]
    if (list(values.shape) != summary["embedding_shape"] or values.dtype != np.float32
            or not np.isfinite(values).all() or ids.tolist() != summary["token_ids"]
            or hashlib.sha256(values.tobytes()).hexdigest() != summary["embedding_float32_sha256"]):
        raise ValueError("Saved query arrays disagree with recorded metadata")
    if len(ids) != len(values) or len(summary["raw_tokens"]) != len(values):
        raise ValueError("Saved query token/embedding dimensions disagree")
    return {"embeddings": torch.from_numpy(values), "token_ids": torch.from_numpy(ids),
            "raw_tokens": summary["raw_tokens"], "kept_token_indices": summary["kept_token_indices"]}


@contextmanager
def argv_scope(values):
    previous = sys.argv
    sys.argv = values
    try:
        yield
    finally:
        sys.argv = previous


def original_worker(inputs, output_dir, threads):
    if threads not in (1, 8):
        raise ValueError("Original-runner diagnostic requires one or eight CPU threads")
    # Imported CAPP benchmark support forces these variables to one. Explicitly
    # restore this diagnostic's declared condition before loading/using torch;
    # never treat the contaminated process default as an independent control.
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(threads)
    import torch
    import run_visual_rerank_batch as runner
    from m3docrag.retrieval import ColPaliRetrievalModel

    torch.set_num_threads(threads)
    env = environment(torch)
    env["requested_cpu_threads"] = threads
    if env["cpu_threads"] != threads:
        raise ValueError("Effective CPU thread count differs from diagnostic condition")
    if hasattr(torch, "__config__"):
        env["torch_parallel_info"] = torch.__config__.parallel_info()
    encode = ColPaliRetrievalModel.encode_query_with_metadata
    captured, models = {}, []

    def capture(model, query, to_cpu=False, query_token_filter="full"):
        if len(captured) >= len(inputs["qids"]):
            raise ValueError("Unexpected extra original-runner query encoding")
        qid = inputs["qids"][len(captured)]
        if query != inputs["questions"][qid] or not to_cpu or query_token_filter != "full":
            raise ValueError("Original runner question/order/encoding settings changed")
        if str(model.model.device) != "cpu":
            raise ValueError("Expected original entry point to encode on CPU")
        meta = encode(model, query, to_cpu=to_cpu, query_token_filter=query_token_filter)
        captured[qid] = {"no_grad": save_query(meta, output_dir, f"query_no_grad_{qid}")}
        models.append(model)
        return meta

    with mock.patch.object(ColPaliRetrievalModel, "encode_query_with_metadata", capture), \
            argv_scope(original_argv(inputs, output_dir)):
        runner.main()
    if list(captured) != inputs["qids"] or len({id(m) for m in models}) != 1:
        raise ValueError("Original runner did not encode the fixed four questions with one model")
    # Run this only AFTER the original entry point has saved its outputs. Extra
    # inference-mode encodings cannot influence that original execution.
    model = models[0]
    for qid in inputs["qids"]:
        with torch.inference_mode():
            meta = encode(model, inputs["questions"][qid], to_cpu=True, query_token_filter="full")
        captured[qid]["inference_mode"] = save_query(meta, output_dir, f"query_inference_mode_{qid}")
    rows = bench.replay.prediction_rows(bench.replay.read_json(output_dir / "original.prediction.json"))
    if set(rows) != set(inputs["qids"]):
        raise ValueError("Original output question set changed")
    comparisons = {qid: diagnostic.compare_candidate_rows(inputs["expected"][qid], rows[qid]["page_retrieval_results"])
                   for qid in inputs["qids"]}
    report = {"status": "original_runner_diagnostic_complete", "environment": env,
              "query_model_device": str(model.model.device), "queries": captured, "comparisons": comparisons}
    bench.write_new(output_dir / "worker.json", report)


def harness_worker(inputs, root, output_dir):
    import torch
    import rerank_target_docs_visual_aware as scoring
    from run_visual_rerank_batch import build_baseline_pool

    torch.set_num_threads(1)
    env = environment(torch)
    original = {mode: bench.replay.read_json(root / mode / "worker.json") for mode in ORIGINAL_MODES}
    documents = list(dict.fromkeys(doc for qid in inputs["qids"]
                                   for doc in build_baseline_pool(inputs["baseline"][qid], 1000)[0]))
    # Only the production dense-scoring method is used. Do not initialize the
    # full online retriever (FAISS, full token table, SPLADE, encoder replicas).
    scorer = object.__new__(online.OnlineRetriever)
    scorer.torch, scorer.config = torch, inputs["config"]
    scorer.embeddings = scoring.load_doc_embeddings_for_doc_ids(documents, inputs["config"]["embedding_name"])
    comparisons = {}
    for mode in HARNESS_MODES:
        comparisons[mode] = {}
        for qid in inputs["qids"]:
            if mode == "prior_saved_gpu_query":
                record = inputs["gpu_queries"][qid]
            else:
                original_mode = next(name for name in ORIGINAL_MODES if mode.startswith(name + "_"))
                context = mode[len(original_mode) + 1:]
                record = original[original_mode]["queries"][qid][context]
            meta = load_query(record, torch)
            with torch.inference_mode():
                rows = scorer.dense_scores(inputs["baseline"][qid], meta, False)
            bench.write_new(output_dir / f"{mode}_{qid}.prediction.json", rows)
            comparisons[mode][qid] = diagnostic.compare_candidate_rows(inputs["expected"][qid], rows)
    bench.write_new(output_dir / "worker.json", {"status": "harness_scoring_diagnostic_complete",
        "environment": env, "loaded_documents": len(documents), "comparisons": comparisons})


def summarize(inputs, original_reports, harness_report):
    for mode, threads in zip(ORIGINAL_MODES, (8, 1)):
        env = original_reports[mode]["environment"]
        if env.get("cpu_threads") != threads or env.get("requested_cpu_threads") != threads:
            raise ValueError("CPU thread controls were not verified as distinct 8/1 conditions")
    cells = {**{mode: report["comparisons"] for mode, report in original_reports.items()},
             **{"harness/" + mode: rows for mode, rows in harness_report["comparisons"].items()}}
    expected_modes = set(ORIGINAL_MODES) | {"harness/" + mode for mode in HARNESS_MODES}
    if set(cells) != expected_modes:
        raise ValueError("Incomplete diagnostic conditions")
    totals = {}
    for mode, rows in cells.items():
        if set(rows) != set(inputs["qids"]):
            raise ValueError("Incomplete diagnostic question set")
        totals[mode] = {"questions": len(rows),
            "candidate_set_matches": sum(r["candidate_sets_match"] for r in rows.values()),
            "complete_order_matches": sum(r["complete_order_matches"] for r in rows.values()),
            "complete_order_and_score_matches": sum(r["complete_order_matches"] and r["scores_close"] for r in rows.values())}
    return {"status": "diagnostic_complete_not_a_runtime_result", "questions": inputs["qids"],
            "totals": totals, "comparisons": cells, "input_sha256": inputs["input_sha256"],
            "environments": {**{k: v["environment"] for k, v in original_reports.items()},
                             "harness": harness_report["environment"]},
            "no_automatic_configuration_selection": True,
            "limitations": ["Four fixed warm-ups, not full validation or proof of historical launch settings",
                "Original entry point is current repository code with metadata capture, not recovered historical source",
                "Eight-thread and one-thread subprocesses are diagnostic conditions, not a thread-count optimization or recovered historical setting",
                "Harness uses recorded query embeddings and cached candidate pages only to isolate scoring, not to measure runtime",
                "Prior GPU query comes from job 15915127, not a fresh query encoding in this job",
                "No FAISS/SPLADE/graph/CAPP/reader run, training, runtime result or old artifact modification"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--exact-summary", type=Path)
    parser.add_argument("--prior-query-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", choices=("original", "harness"))
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--threads", type=int, choices=(1, 8), default=1)
    args = parser.parse_args()
    if args.worker and args.inputs is None:
        parser.error("Worker requires --inputs")
    if not args.worker and any(p is None for p in (args.bundle, args.exact_summary, args.prior_query_dir)):
        parser.error("Require bundle, exact-summary and prior-query-dir")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        if args.worker:
            inputs = bench.replay.read_json(args.inputs)
            if args.worker == "original":
                original_worker(inputs, args.output_dir, args.threads)
            else:
                harness_worker(inputs, args.inputs.parent, args.output_dir)
            return
        inputs = prepare(args.bundle, args.exact_summary, args.prior_query_dir, args.output_dir)
        for name, worker, threads in ((ORIGINAL_MODES[0], "original", 8), (ORIGINAL_MODES[1], "original", 1),
                                      ("harness", "harness", 1)):
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", worker,
                "--threads", str(threads), "--inputs", str((args.output_dir / "inputs.json").resolve()),
                "--output-dir", str((args.output_dir / name).resolve())]
            print("EXACT_DIAGNOSTIC_START " + name, flush=True)
            with (args.output_dir / f"{name}.log").open("x") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = summarize(inputs, {mode: bench.replay.read_json(args.output_dir / mode / "worker.json")
                                   for mode in ORIGINAL_MODES},
                           bench.replay.read_json(args.output_dir / "harness" / "worker.json"))
        report["code_sha256"] = {str(p): bench.replay.sha256(p) for p in
            (Path(__file__), Path(online.__file__), Path(diagnostic.__file__),
             ROOT / "scripts/run_visual_rerank_batch.py", ROOT / "scripts/rerank_target_docs_visual_aware.py",
             ROOT / "src/m3docrag/retrieval/colpali.py")}
        bench.write_new(args.output_dir / "diagnostic.json", report)
        print("EXACT_DIAGNOSTIC_TOTALS " + json.dumps(report["totals"]), flush=True)
        print("saved_diagnostic=" + str(args.output_dir / "diagnostic.json"), flush=True)
    except Exception as exc:
        bench.write_new(args.output_dir / "failure.json", {"status": "diagnostic_failed_not_a_runtime_result",
            "type": type(exc).__name__, "error": str(exc), "worker_logs": "See original_eight_threads.log, original_one_thread.log or harness.log"})
        raise


if __name__ == "__main__":
    main()
