#!/usr/bin/env python3
"""Validate frozen LambdaMART/monoT5 top-four reader runs; never rerank inputs."""
import argparse
import json
import math
import os
from pathlib import Path

from validate_racs_bge_reader import (
    fingerprint, normalize_pages, read_json, require, unique_object,
    validate_outputs, write_new,
)

STEMS = {
    "lambdamart": "mmqa_train_to_dev_lambdamart_base_exact_maxsim_gpp_direct_exactonly_adaptive_norm05",
    "monot5": "mmqa_dev_monot5_base_msmarco_10k_seq2seq_gpp_top1000",
}


def input_paths(repo, method):
    out = repo / "output"
    stem = out / "m3docvqa_standard_reranker_baselines" / STEMS[method]
    paths = {
        "gold": out / "m3docvqa_mmqa_direct_evidence_pseudo_page_labels/mmqa_dev_pseudo_page_labels_direct_exactonly_adaptive_norm05.augmented_gold.jsonl",
        "base": out / "m3docvqa_gpp_hyperlink_node_ablation_exact_maxsim/mmqa_dev_exact_maxsim_gpp_hyperlink_node_no_hyperlink.prediction.json",
        "summary": Path(str(stem) + ".summary.json"),
        "prediction": Path(str(stem) + (".dev.prediction.json" if method == "lambdamart" else ".prediction.json")),
    }
    if method == "lambdamart":
        paths["model"] = Path(str(stem) + ".model.json")
    return paths


def check_config(method, config):
    expected = ({"backend": "lightgbm", "objective": "lambdarank",
                 "inference_mode": "blend_rerank", "blend_alpha": 0.45}
                if method == "lambdamart" else
                {"candidate_top_k": 1000, "rerank_top_k": 1000,
                 "blend_alpha": 1.0, "max_length": 512, "max_page_chars": 4000})
    for key, value in expected.items():
        require(config.get(key) == value, f"Unexpected {method} {key}: {config.get(key)}")
    if method == "monot5":
        require(Path(config.get("model_name_or_path", "")).name == "monot5-base-msmarco-10k",
                "Unexpected monoT5 checkpoint")


def reader_inputs(method, gold, base, prediction, expected_questions=2441,
                  expected_candidates=1000):
    qids = [str(row["qid"]).strip() for row in gold]
    require(len(qids) == len(set(qids)) == expected_questions and all(qids),
            "Gold question count or unique QIDs differ")
    require(set(base) == set(prediction) == set(qids), "Prediction QIDs differ from gold")
    metadata_key = ("graph_aware_ltr_page_reranker" if method == "lambdamart"
                    else "standard_seq2seq_page_reranker")
    inputs = {}
    for qid in qids:
        base_pages = normalize_pages(base[qid], qid, expected_candidates)
        pages = normalize_pages(prediction[qid], qid, expected_candidates)
        require({tuple(p[:2]) for p in base_pages} == {tuple(p[:2]) for p in pages},
                f"Candidate pool differs from GPP: {qid}")
        check_config(method, prediction[qid].get("reranker_metadata", {}).get(metadata_key, {}))
        require(len(pages) >= 4, f"Fewer than four pages: {qid}")
        inputs[qid] = {"qid": qid, "page_retrieval_results": pages[:4]}
    return inputs


def prepare(repo, run, method):
    paths = input_paths(repo, method)
    for path in paths.values():
        require(path.is_file(), f"Missing input: {path}")
    summary = read_json(paths["summary"])
    for field, key in (("eval_gold", "gold"), ("eval_base_pred", "base")) if method == "lambdamart" else (("gold", "gold"), ("base_pred", "base")):
        require(summary.get(field) and Path(summary[field]).resolve() == paths[key].resolve(),
                f"Summary {field} differs from pinned input")
    if method == "lambdamart":
        model = read_json(paths["model"])
        check_config(method, {**model["args"], **model["model_info"]})
        require(model["args"].get("candidate_top_k") == 1000, "Wrong LambdaMART depth")
        require(len(model["feature_names"]) == 40, "Wrong LambdaMART feature count")
    else:
        check_config(method, summary)
        require(summary.get("processed_qid_count") == 2441, "Wrong monoT5 cohort")
    with paths["gold"].open() as handle:
        gold = [json.loads(line, object_pairs_hook=unique_object) for line in handle if line.strip()]
    prediction = read_json(paths["prediction"])
    inputs = reader_inputs(method, gold, read_json(paths["base"]), prediction)
    # Confirm the saved rankings reproduce the thesis retrieval row before QA.
    import train_content_aware_pseudo_page_reranker as ca
    metrics = ca.evaluate_run(label=method, pred=prediction,
                              gold=ca.load_gold(paths["gold"]), recall_ks=[4, 10, 100])
    expected = {"lambdamart": (0.6408, 0.7468, 0.9045),
                "monot5": (0.6609, 0.7879, 0.9278)}[method]
    require(metrics.get("eval_qid_count") == 2188, "Unexpected retrieval cohort")
    for k, value in zip((4, 10, 100), expected):
        require(abs(metrics[f"page@{k}"] - value) <= 0.000051,
                f"Saved {method} page@{k} differs from thesis row")
    paths.update({"reader_input": run / "reader_input.json",
                  "validator_code": Path(__file__),
                  "shared_validator": Path(__file__).with_name("validate_racs_bge_reader.py"),
                  "retrieval_evaluator": repo / "scripts/train_content_aware_pseudo_page_reranker.py",
                  "qa_evaluator": repo / "scripts/evaluate_m3docvqa_qa_runs.py",
                  "reader_code": repo / "scripts/run_m3docvqa_external_retrieval_qa.py",
                  "launcher_code": repo / "examples/sbatch_racs_remaining_reranker_readers.sh"})
    write_new(paths["reader_input"], inputs)
    write_new(run / "run_manifest.json", {
        "method": method, "input_validation": "passed", "questions": 2441,
        "candidate_top_k": 1000, "reader_pages_each": 4,
        "reader_model": "Qwen2-VL-7B-Instruct", "reader_bits": 16,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "retrieval_metrics": metrics,
        "fingerprints": {k: fingerprint(p) for k, p in paths.items()},
    })
    print("RERANKER_READER_INPUT_VALIDATED " + json.dumps({"method": method, "questions": 2441}), flush=True)


def check(run, method):
    manifest = read_json(run / "run_manifest.json")
    require(manifest["method"] == method, "Run method mismatch")
    for key, recorded in manifest["fingerprints"].items():
        require(fingerprint(recorded["path"]) == recorded, f"Input/code changed: {key}")
    result = validate_outputs(read_json(run / "reader_input.json"),
                              read_json(run / "qa.prediction.json"), read_json(run / "qa.eval.json"))
    # Independently recompute answer metrics from the saved answers and gold.
    from evaluate_m3docvqa_qa_runs import summarize
    recomputed = summarize(method, run / "qa.prediction.json",
                           Path(manifest["fingerprints"]["gold"]["path"]), 2441)
    for metric, key in (("list_em", "em"), ("list_f1", "f1")):
        require(math.isclose(result["overall"][metric], recomputed[key], abs_tol=1e-8, rel_tol=0),
                f"Saved {metric} differs from independent evaluation")
    result.update({"method": method, "status": "validated_reader_outputs",
                   "retrieval_metrics": manifest["retrieval_metrics"]})
    write_new(run / "validated_result.json", result)
    print("RERANKER_READER_RESULT " + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "check"))
    parser.add_argument("--method", choices=tuple(STEMS), required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.repo_root.resolve(), args.run_dir.resolve(), args.method)
    else:
        check(args.run_dir.resolve(), args.method)


if __name__ == "__main__":
    main()
