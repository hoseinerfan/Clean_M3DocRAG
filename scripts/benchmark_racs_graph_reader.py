#!/usr/bin/env python3
"""Paired graph-to-answer timing from CACHED dense/sparse retrieval outputs.

This is NOT online query-to-answer timing: dense/sparse query encoding, search,
Exact MaxSim and legacy approximate MaxSim are excluded. CAPP DOES pay for its
three auxiliary graph computations and per-query source-map construction.
No training or original-artifact writes occur. Each condition/pass runs in a
fresh process, sequentially on the same allocated GPU. Reports are exclusive.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_variable] = "1"

import benchmark_capp_runtime as capp
import validate_racs_graph_replay as replay

MAIN = "base_exact_maxsim_no_hyperlink"
AUXILIARIES = {
    "no_hyperlink": "gpp_no_hyperlink",
    "docnode_to_hyperlink_docs": "gpp_doc_hyperlink",
    "pagenode_to_hyperlink_pages": "gpp_page_hyperlink",
}
FULL_STEM = (
    "output/m3docvqa_content_aware_exact_maxsim_direct_exactonly_adaptive_norm05/"
    "mmqa_train_to_dev_content_aware_fixed_alpha_0p40_base_exact_maxsim_gpp_direct_exactonly_adaptive_norm05"
)
SCOPE = "graph-to-answer with cached dense/sparse rankings; NOT online end-to-end"
EXCLUDED = ["dense query encoding and FAISS search", "Exact MaxSim scoring",
            "legacy approximate MaxSim scoring for auxiliary inputs",
            "SPLADE query encoding and search", "offline corpus indexing/text extraction",
            "one-time input/model preparation (reported separately)"]


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def schedule(repeats):
    if repeats < 2 or repeats % 2:
        raise ValueError("Use an even repeat count >= 2 for balanced GPP/CAPP order")
    return [(index + 1, method) for index in range(repeats)
            for method in (("GPP", "CAPP") if index % 2 == 0 else ("CAPP", "GPP"))]


def absolute(root, path):
    path = Path(path)
    return path if path.is_absolute() else root / path


def prepare(audit_path, replay_path, run_dir, count, warmup):
    """Parent-only reference loading; workers never load full prediction files."""
    started = time.perf_counter()
    audit = replay.read_json(audit_path)
    check = replay.read_json(replay_path)
    if not all(audit["checks"].values()) or set(audit["graphs"]) != {MAIN, *AUXILIARIES}:
        raise ValueError("Unexpected prerequisite audit")
    if check.get("status") != "sample_graph_replay_validated" or set(check["results"]) != set(audit["graphs"]):
        raise ValueError("A successful four-graph replay report is required")
    if not all(row["questions"] >= 16 and all(row[key] == row["questions"] for key in
               ("candidate_sets_match", "complete_order_matches", "scores_close"))
               for row in check["results"].values()):
        raise ValueError("Incomplete replay checks")
    root = Path(audit["root"])
    audited_hash = check["input_sha256"].get(str(audit_path))
    if audited_hash != replay.sha256(audit_path):
        raise ValueError("The replay report does not fingerprint this audit")
    fingerprints = {str(audit_path): audited_hash, str(replay_path): replay.sha256(replay_path)}

    def pin(path):
        path = absolute(root, path)
        digest = replay.sha256(path)
        # The earlier replay may have fingerprinted paths relative to its cwd.
        old = {absolute(root, key): value for key, value in check["input_sha256"].items()}
        if path in old and old[path] != digest:
            raise ValueError(f"Input changed since graph replay: {path}")
        fingerprints[str(path)] = digest
        return path

    gold_path = pin(audit["gold"]["path"])
    questions = {}
    with gold_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["qid"])
            if qid in questions or not str(row.get("question", "")).strip():
                raise ValueError("Duplicate qid or absent question text")
            questions[qid] = str(row["question"]).strip()
    if replay.qid_digest(questions) != audit["gold"]["qid_sha256"]:
        raise ValueError("Question cohort differs from audit")
    if count < 1 or warmup < 1:
        raise ValueError("Positive measured and disjoint warm-up counts required")
    chosen = replay.select_qids(list(questions), count + warmup)
    measured, warmed = chosen[:count], chosen[count:]
    cache = {}

    def load_selected(path):
        path = absolute(root, path)
        if path not in cache:
            pin(path)
            rows = replay.prediction_rows(replay.read_json(path))
            if replay.qid_digest(rows) != audit["gold"]["qid_sha256"]:
                raise ValueError(f"Prediction cohort differs: {path}")
            cache[path] = {qid: rows[qid] for qid in chosen}
        return cache[path]

    graphs, inputs = {}, {}
    for label, item in audit["graphs"].items():
        args, compat = replay.settings_from_metadata(item["first_question_metadata"])
        if any(getattr(args, key) != 1000 for key in ("dense_top_pages", "sparse_top_pages", "final_top_pages")):
            raise ValueError("Expected 1000-page graph depth")
        saved = load_selected(item["path"])
        expected = {k: v for k, v in item["first_question_metadata"].items() if k != "graph"}
        for row in saved.values():
            if {k: v for k, v in row["reranker_metadata"].items() if k != "graph"} != expected:
                raise ValueError(f"Graph metadata changed: {label}")
            replay.compare_rows(row["page_retrieval_results"], row["page_retrieval_results"], 1000)
        upstream = {}
        for kind in ("dense", "sparse"):
            path = absolute(root, getattr(args, kind + "_prediction_json"))
            key = str(path)
            upstream[kind] = key
            if key not in inputs:
                # Deliberately discard answers, gold diagnostics and inherited timers.
                inputs[key] = {qid: {"question": row.get("question", ""),
                                    "page_retrieval_results": row["page_retrieval_results"]}
                               for qid, row in load_selected(path).items()}
        args.doc_pages_jsonl = str(pin(args.doc_pages_jsonl))
        if args.pdf_hyperlink_edges_jsonl:
            args.pdf_hyperlink_edges_jsonl = str(pin(args.pdf_hyperlink_edges_jsonl))
        graphs[label] = {"settings": vars(args), "compatibility_values_applied": compat,
                         "inputs": upstream,
                         "references": {qid: saved[qid]["page_retrieval_results"] for qid in chosen}}
    model_path = pin(FULL_STEM + ".model.json")
    model = replay.read_json(model_path)
    settings = capp.validate_model(model)
    if settings.candidate_top_k != 1000 or settings.blend_alpha != 0.4 or len(model["feature_names"]) != 30:
        raise ValueError("Not the expected 30-feature fixed-alpha CAPP model")
    saved_capp = load_selected(FULL_STEM + ".dev.prediction.json")
    references = {}
    for qid in chosen:
        rows = saved_capp[qid]["page_retrieval_results"]
        replay.compare_rows(rows, rows, 1000)
        if {(str(r[0]), int(r[1])) for r in rows} != {
                (str(r[0]), int(r[1])) for r in graphs[MAIN]["references"][qid]}:
            raise ValueError("CAPP candidate set differs from the GPP base")
        references[qid] = capp.order_digest([capp.ca.page_uid(str(r[0]), int(r[1])) for r in rows])
    paths = {}
    for method in ("GPP", "CAPP"):
        labels = [MAIN] + (list(AUXILIARIES) if method == "CAPP" else [])
        needed = {key for label in labels for key in graphs[label]["inputs"].values()}
        bundle = {"scope": SCOPE, "method": method, "measured_qids": measured, "warmup_qids": warmed,
                  "questions": {qid: questions[qid] for qid in chosen},
                  "inputs": {key: inputs[key] for key in needed},
                  "graphs": {label: graphs[label] for label in labels},
                  "model": model if method == "CAPP" else None,
                  "capp_order_digests": references if method == "CAPP" else {},
                  "page_text": graphs[MAIN]["settings"]["doc_pages_jsonl"]}
        path = run_dir / (method.lower() + ".bundle.json")
        write_new(path, bundle)
        paths[method] = path
    return paths, {"scope": SCOPE, "excluded": EXCLUDED, "questions": count,
                   "measured_qids": measured, "warmup_qids": warmed,
                   "selection_rule": "first N SHA256(racs-graph-replay-v1:qid); next W disjoint qids for warm-up",
                   "input_sha256": fingerprints, "bundle_preparation_seconds": time.perf_counter() - started,
                   "parent_peak_rss_mib": capp.peak_rss_mib(),
                   "memory_scope": "fresh condition workers incl. initialization and small validation references; sampled upstream rows, full graph catalog/page text where needed; not production/full-corpus serving memory",
                   "cache_policy": "no document image cache across questions; one render per unique document within each question; OS filesystem cache uncontrolled, condition order balanced",
                   "timer_boundary": "GPU synchronize; main graph (plus auxiliary graphs, source maps and CAPP where needed); PDF rendering/image selection; prompt and Qwen generation; GPU synchronize",
                   "validation_policy": "outside query timer; every regenerated 1000-page graph order+scores and full CAPP order must match; failed workers do not produce a validated aggregate"}


def rank_query(bundle, qid, catalogs, links, pages, model_state):
    stages, regenerated = {}, {}
    for label, entry in bundle["graphs"].items():
        args = SimpleNamespace(**entry["settings"])
        start = time.perf_counter()
        rows, _ = replay.graph.build_qid_graph_ranking(
            qid=qid, dense_row=bundle["inputs"][entry["inputs"]["dense"]][qid],
            sparse_row=bundle["inputs"][entry["inputs"]["sparse"]][qid], args=args,
            doc_page_catalog=catalogs[args.doc_pages_jsonl],
            pdf_hyperlink_graph=links.get(args.pdf_hyperlink_edges_jsonl), gold_row=None)
        stages["graph:" + label] = time.perf_counter() - start
        regenerated[label] = rows
    base = regenerated[MAIN]
    order = [capp.ca.page_uid(str(row[0]), int(row[1])) for row in base]
    if bundle["method"] == "CAPP":
        start = time.perf_counter()
        sources = {source: capp.ca.source_maps({qid: {"page_retrieval_results": regenerated[label]}}, 1000)
                   for label, source in AUXILIARIES.items()}
        stages["source_maps"] = time.perf_counter() - start
        model, settings, parameters = model_state
        order, feature_stages = capp.timed_query(qid, {"page_retrieval_results": base},
            bundle["questions"][qid], pages, sources, model, settings, parameters)
        stages.update({"capp:" + key: value for key, value in feature_stages.items()})
    by_uid = {capp.ca.page_uid(str(row[0]), int(row[1])): row for row in base}
    return [by_uid[uid] for uid in order[:4]], order, regenerated, stages


def validate_rankings(bundle, qid, order, regenerated):
    for label, actual in regenerated.items():
        result = replay.compare_rows(bundle["graphs"][label]["references"][qid], actual, 1000)
        if not result["complete_order_matches"] or not result["scores_close"]:
            raise ValueError(f"Graph replay mismatch: {label} qid={qid} {result}")
    if bundle["method"] == "CAPP" and capp.order_digest(order) != bundle["capp_order_digests"][qid]:
        raise ValueError(f"CAPP complete ranking mismatch: qid={qid}")


def worker(bundle_path, output_path, pass_index):
    # Heavy reader imports are lazy: CPU-only preparation/tests need no GPU stack.
    import torch
    from accelerate import Accelerator
    import run_m3docvqa_external_retrieval_qa as qa

    if output_path.exists():
        raise FileExistsError(output_path)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA GPU is required")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    capp.np.random.seed(0)
    bundle = replay.read_json(bundle_path)
    prepared = time.perf_counter()
    catalogs, links = {}, {}
    for entry in bundle["graphs"].values():
        args = SimpleNamespace(**entry["settings"])
        if args.doc_pages_jsonl not in catalogs:
            catalogs[args.doc_pages_jsonl] = replay.graph.load_doc_page_catalog(Path(args.doc_pages_jsonl))
        # A no-edge GPP worker need not retain the hyperlink graph in memory.
        active = args.doc_doc_edge_mode == "hyperlink_citation" or args.pdf_hyperlink_edge_weight > 0
        if active and args.pdf_hyperlink_edges_jsonl not in links:
            links[args.pdf_hyperlink_edges_jsonl] = replay.graph.load_pdf_hyperlink_graph(Path(args.pdf_hyperlink_edges_jsonl))
    pages, model_state = None, None
    if bundle["method"] == "CAPP":
        pages = capp.ca.load_page_features(Path(bundle["page_text"]))
        model = bundle["model"]
        settings = capp.validate_model(model)
        parameters = tuple(capp.np.asarray(model[key], dtype=capp.np.float32) for key in ("mean", "std", "weights")) + (float(model["bias"]),)
        model_state = model, settings, parameters
        for qid in bundle["questions"]:
            if any(capp.ca.page_uid(str(r[0]), int(r[1])) not in pages for r in bundle["graphs"][MAIN]["references"][qid]):
                raise ValueError("Incomplete candidate page-text coverage")
    rank_preparation_seconds = time.perf_counter() - prepared
    started = time.perf_counter()
    cli = SimpleNamespace(data_name="m3-docvqa", split="dev", bits=16, model_name_or_path="Qwen2-VL-7B-Instruct")
    dataset = qa.M3DocVQADataset(qa.make_dataset_args(cli))
    model_path = qa.resolve_model_path(cli.model_name_or_path)
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    accelerator = Accelerator()
    if accelerator.num_processes != 1:
        raise ValueError("Only single-process inference is supported")
    flash = qa.supports_flash_attention()
    reader = qa.VQAModel(model_name_or_path=model_path, model_type="qwen2", bits=16,
                         use_flash_attn=flash, attn_implementation="flash_attention_2" if flash else "eager")
    reader.model = accelerator.prepare(reader.model)
    torch.cuda.synchronize()
    reader_preparation_seconds = time.perf_counter() - started
    gc.collect()
    initialization_peak = capp.peak_rss_mib()

    def one(qid):
        torch.cuda.synchronize()
        start = time.perf_counter()
        selected, order, regenerated, stages = rank_query(bundle, qid, catalogs, links, pages, model_state)
        ranked = time.perf_counter()
        # No persistent document cache: same declared image-cache policy for both.
        rendered, images = {}, []
        for doc_id, page_idx, *_ in selected:
            if doc_id not in rendered:
                rendered[doc_id] = dataset.get_images_from_doc_id(doc_id)
            if not 0 <= int(page_idx) < len(rendered[doc_id]):
                raise IndexError(f"Unrenderable selected page: {doc_id}, {page_idx}")
            images.append(rendered[doc_id][int(page_idx)])
        imaged = time.perf_counter()
        prompt = qa.short_answer_template.substitute({"question": bundle["questions"][qid]})
        with torch.no_grad():
            answer = reader.generate(images=images, question=prompt)
        torch.cuda.synchronize()
        finished = time.perf_counter()
        stages.update(ranking_total=ranked - start, image_preparation=imaged - ranked,
                      reader_prompt_preprocess_generate=finished - imaged, total=finished - start)
        # Validation and JSON serialization are deliberately outside the timer.
        validate_rankings(bundle, qid, order, regenerated)
        if len(selected) != 4 or len({(r[0], int(r[1])) for r in selected}) != 4:
            raise ValueError("Exactly four unique pages required")
        return {"qid": qid, "seconds": stages, "answer": answer,
                "selected_pages": [[r[0], int(r[1])] for r in selected], "ranking_checks_passed": True}

    for qid in bundle["warmup_qids"]:
        one(qid)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    results = []
    for index, qid in enumerate(bundle["measured_qids"], 1):
        results.append(one(qid))
        if index % 16 == 0 or index == len(bundle["measured_qids"]):
            print(f"GRAPH_READER_PROGRESS pass={pass_index} method={bundle['method']} questions={index}/{len(bundle['measured_qids'])}", flush=True)
    properties = torch.cuda.get_device_properties(0)
    packages = {}
    for name in ("torch", "transformers", "accelerate", "numpy", "qwen-vl-utils", "flash-attn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    config_hashes = {str(p): replay.sha256(p) for p in model_path.glob("*.json")}
    result = {"status": "validated_graph_to_answer_worker", "scope": SCOPE, "excluded": EXCLUDED,
              "method": bundle["method"], "pass": pass_index, "bundle_sha256": replay.sha256(bundle_path),
              "questions": len(results), "warmup_questions": len(bundle["warmup_qids"]), "results": results,
              "summary": capp.summarize([row["seconds"]["total"] for row in results]),
              "preparation_seconds": {"ranking_inputs": rank_preparation_seconds, "reader_and_dataset": reader_preparation_seconds},
              "memory": {"cpu_peak_rss_mib_after_initialization": initialization_peak,
                         "cpu_peak_rss_mib_whole_worker": capp.peak_rss_mib(),
                         "gpu_peak_allocated_mib_measured_phase": torch.cuda.max_memory_allocated() / 2**20,
                         "gpu_peak_reserved_mib_measured_phase": torch.cuda.max_memory_reserved() / 2**20},
              "hardware": {"hostname": platform.node(), "gpu": properties.name, "gpu_total_bytes": properties.total_memory,
                           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                           "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
                           "torch_cpu_threads": torch.get_num_threads(), "cpu_description": capp.command_output(["lscpu"]),
                           "nvidia_smi": capp.command_output(["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"])},
              "reader": {"path": str(model_path.resolve()), "bits": 16, "dtype": str(next(reader.model.parameters()).dtype),
                         "attention": "flash_attention_2" if flash else "eager", "pages": 4, "max_new_tokens": 128,
                         "image_processor": type(reader.processor.image_processor).__name__,
                         "config_sha256": config_hashes, "checkpoint_weight_bytes_hashed": False},
              "python": sys.version, "packages": packages}
    write_new(output_path, result)
    print("GRAPH_READER_WORKER " + json.dumps({"pass": pass_index, "method": bundle["method"], **result["summary"]}), flush=True)


def aggregate(reports, manifest, repeats):
    expected = schedule(repeats)
    if [(r["pass"], r["method"]) for r in reports] != expected:
        raise ValueError("Incomplete or out-of-order condition reports")
    reference_hardware, reference_reader, reference_packages = reports[0]["hardware"], reports[0]["reader"], reports[0]["packages"]
    # lscpu may include current MHz/scaling information; compare stable fields.
    stable_hardware = ("hostname", "gpu", "gpu_total_bytes", "cuda_visible_devices", "slurm_job_id",
                       "slurm_cpus_per_task", "torch_cpu_threads", "nvidia_smi")
    bundle_hashes = {}
    grouped = defaultdict(list)
    for report in reports:
        if report["status"] != "validated_graph_to_answer_worker" or report["scope"] != SCOPE:
            raise ValueError("Unvalidated worker or wrong timing scope")
        if (any(report["hardware"][key] != reference_hardware[key] for key in stable_hardware)
                or report["reader"] != reference_reader or report["packages"] != reference_packages):
            raise ValueError("Hardware, reader or software differs between workers")
        method = report["method"]
        if bundle_hashes.setdefault(method, report["bundle_sha256"]) != report["bundle_sha256"]:
            raise ValueError("Input bundle changed across repeats")
        if [r["qid"] for r in report["results"]] != manifest["measured_qids"]:
            raise ValueError("Worker question cohorts differ")
        if not all(r["ranking_checks_passed"] for r in report["results"]):
            raise ValueError("Ranking check failed")
        for row in report["results"]:
            if not row["seconds"] or any(not math.isfinite(value) or value < 0 for value in row["seconds"].values()) or row["seconds"]["total"] <= 0:
                raise ValueError("Invalid timing value")
        if report["questions"] != manifest["questions"]:
            raise ValueError("Worker question counts differ")
        grouped[report["method"]].append(report)
    methods, paired = {}, {}
    for method, runs in grouped.items():
        stages = runs[0]["results"][0]["seconds"]
        if any(set(row["seconds"]) != set(stages) for run in runs for row in run["results"]):
            raise ValueError("Inconsistent timing stages")
        mean_by_question = {key: [sum(run["results"][index]["seconds"][key] for run in runs) / repeats
                                  for index in range(manifest["questions"])] for key in stages}
        paired[method] = mean_by_question["total"]
        methods[method] = {"per_question_mean_across_repeats": {key: capp.summarize(values) for key, values in mean_by_question.items()},
                           "pass_total_summaries": [r["summary"] for r in runs],
                           "memory_by_fresh_worker": [r["memory"] for r in runs],
                           "preparation_seconds_by_fresh_worker": [r["preparation_seconds"] for r in runs],
                           "qids_with_answer_variation_across_repeats": sum(len({r["results"][i]["answer"] for r in runs}) > 1
                                                                         for i in range(manifest["questions"]))}
    delta = [c - g for c, g in zip(paired["CAPP"], paired["GPP"])]
    return {"status": "validated_cached_retrieval_graph_to_answer", **manifest, "repeats": repeats,
            "hardware": reference_hardware, "reader": reference_reader, "packages": reference_packages,
            "methods": methods, "paired_capp_minus_gpp_seconds": {"mean": sum(delta) / len(delta),
                 "per_question": dict(zip(manifest["measured_qids"], delta))},
            "caveats": ["Not full query-to-answer latency or its incremental overhead; required upstream work is excluded",
                        "Subset timings do not replace full-development QA scores",
                        "Median/p95 summarize per-question means across repeats, not raw request tail latency",
                        "Fresh-worker RSS includes initialization and validation references; not isolated scorer memory",
                        "GPU reserved-memory peaks include allocator cache retained from warm-up",
                        "OS cache and external node load are not controlled; balanced order does not eliminate all variation",
                        "No significance test, concurrency throughput claim, or checkpoint-weight hash validation"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", type=Path)
    parser.add_argument("--replay-json", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--questions", type=int, default=128)
    parser.add_argument("--warmup-questions", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--worker-bundle", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--pass-index", type=int)
    args = parser.parse_args()
    if args.worker_bundle:
        if args.worker_output is None or args.pass_index is None:
            parser.error("Worker requires output and pass index")
        worker(args.worker_bundle, args.worker_output, args.pass_index)
        return
    if args.run_dir is None or args.audit_json is None or args.replay_json is None:
        parser.error("Require run-dir, audit-json and replay-json")
    order = schedule(args.repeats)
    args.run_dir.mkdir()  # No overwrite or automatic resume.
    try:
        bundles, manifest = prepare(args.audit_json, args.replay_json, args.run_dir, args.questions, args.warmup_questions)
        manifest["schedule"] = order
        code_paths = [Path(__file__), Path(capp.__file__), Path(capp.ca.__file__), Path(replay.__file__), Path(replay.graph.__file__),
                      Path("src/m3docrag/vqa/qwen2.py"), Path("scripts/run_m3docvqa_external_retrieval_qa.py")]
        manifest["code_sha256"] = {str(p): replay.sha256(p) for p in code_paths}
        write_new(args.run_dir / "manifest.json", manifest)
        # Release all large preparation objects before the first GPU subprocess.
        gc.collect()
        reports = []
        for index, method in order:
            output = args.run_dir / f"pass{index}_{method.lower()}.json"
            subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--worker-bundle", str(bundles[method]),
                            "--worker-output", str(output), "--pass-index", str(index)], check=True)
            reports.append(replay.read_json(output))
        result = aggregate(reports, manifest, args.repeats)
        write_new(args.run_dir / "runtime.json", result)
        for method in ("GPP", "CAPP"):
            print("GRAPH_TO_ANSWER_RESULT " + json.dumps({"method": method,
                **result["methods"][method]["per_question_mean_across_repeats"]["total"]}), flush=True)
        print("saved_graph_to_answer_report=" + str(args.run_dir / "runtime.json"), flush=True)
    except Exception as exc:
        write_new(args.run_dir / "failure.json", {"status": "failed", "type": type(exc).__name__, "error": str(exc), "scope": SCOPE})
        raise


if __name__ == "__main__":
    main()
