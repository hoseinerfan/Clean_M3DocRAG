#!/usr/bin/env python3
"""Alpha sensitivity: freeze job 15942315's model and change only 0.45 to 0.40."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
from validate_racs_controlled_reader import validate_training
from validate_racs_bge_reader import (
    fingerprint, normalize_pages, read_json, require, validate_outputs, write_new,
)


def reblend(ca, scored, args, alpha):
    local = copy.copy(args)
    local.blend_alpha = alpha
    # rerank_records writes derived scores. Keep the shared scored records intact.
    return ca.rerank_records([dict(row) for row in scored], local)


def verified_pair(ca, scored, args, reference, qid, count=1000):
    replay = reblend(ca, scored, args, 0.45)
    saved_pages = normalize_pages(reference, qid, count)
    replay_pages = normalize_pages({"page_retrieval_results": [r["raw"] for r in replay]}, qid, count)
    require(replay_pages == saved_pages, f"Alpha 0.45 replay differs: {qid}; do not compare different pipelines")
    alpha040 = reblend(ca, scored, args, 0.40)
    require({r["uid"] for r in alpha040} == {r["uid"] for r in scored}, f"Candidate pool changed: {qid}")
    pages040 = normalize_pages({"page_retrieval_results": [r["raw"] for r in alpha040]}, qid, count)
    return saved_pages, pages040


def load_frozen_code(training, manifest):
    path = training / "code/train_content_aware_pseudo_page_reranker.py"
    require(fingerprint(path)["sha256"] == manifest["code_sha256"][path.name], "Training code snapshot changed")
    spec = importlib.util.spec_from_file_location("racs_frozen_ca", path)
    ca = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ca)
    return ca, path


def prepare(repo, run):
    training = repo / "output/racs_controlled_training_15942315"
    result = read_json(training / "result.json")
    model = read_json(training / "final.model.json")
    selection = read_json(training / "selection.json")
    validate_training(result, model, selection)
    require(fingerprint(training / "final.model.json")["sha256"] == result["final_model_sha256"], "Model changed")
    require(fingerprint(training / "manifest.json")["sha256"] == result["manifest_sha256"], "Manifest changed")
    manifest = read_json(training / "manifest.json")
    ca, snapshot = load_frozen_code(training, manifest)
    paths = {}
    for key, recorded in manifest["inputs"].items():
        if key.startswith("dev_"):
            paths[key] = Path(recorded["path"])
            require(fingerprint(paths[key])["sha256"] == recorded["sha256"], f"Input changed: {key}")
    gold = ca.load_gold(paths["dev_gold"])
    base = ca.load_prediction(paths["dev_base"])
    reference = read_json(training / "dev.prediction.json")
    require(len(gold) == 2441 and set(gold) == set(base) == set(reference), "Unexpected dev cohort")
    pages = ca.load_page_features(paths["dev_text"])
    sources = {key.split(":", 1)[1]: ca.source_maps(ca.load_prediction(path), 1000)
               for key, path in paths.items() if key.startswith("dev_source:")}
    require(set(sources) == {"gpp_no_hyperlink", "gpp_doc_hyperlink", "gpp_page_hyperlink"}, "Source set differs")
    args = argparse.Namespace(**model["args"])
    require(args.inference_mode == "blend_rerank" and args.candidate_top_k == 1000, "Unexpected inference configuration")
    mean, std = np.asarray(model["mean"], dtype=np.float32), np.asarray(model["std"], dtype=np.float32)
    weights, bias = ca.scorer_from_model_json(model)
    predictions, inputs = {}, {}
    changed_order = changed_set = 0
    for index, (qid, row) in enumerate(base.items(), 1):
        records = ca.ranked_page_records(row, 1000)
        require(len(records) == 1000 and all(r["uid"] in pages for r in records), f"Missing candidates/page text: {qid}")
        scored = ca.score_records(qid=qid, gold_row={"question": gold[qid]["question"]},
            records=records, page_features=pages, source_maps_by_label=sources,
            mean=mean, std=std, weights=weights, bias=bias, feature_names=model["feature_names"])
        saved_pages, output_pages = verified_pair(ca, scored, args, reference[qid], qid)
        top4 = output_pages[:4]
        predictions[qid] = {"qid": qid, "question": gold[qid]["question"],
                            "page_retrieval_results": output_pages,
                            "reranker_metadata": {"sensitivity_alpha": 0.40, "training_selected_alpha": 0.45,
                                                  "model_sha256": result["final_model_sha256"]}}
        inputs[qid] = {"qid": qid, "page_retrieval_results": top4}
        changed_order += top4 != saved_pages[:4]
        changed_set += {(p[0], p[1]) for p in top4} != {(p[0], p[1]) for p in saved_pages[:4]}
        if index % 250 == 0:
            print(f"ALPHA_REPLAY_AND_REBLEND {index}/2441", flush=True)
    write_new(run / "alpha040.prediction.json", predictions)
    write_new(run / "reader_input.json", inputs)
    metrics = [ca.evaluate_run(label=label, pred=pred, gold=gold, recall_ks=[1, 2, 4, 8, 10, 1000])
               for label, pred in (("same_model_alpha045", reference), ("same_model_alpha040", predictions))]
    report = {"comparison_type": "development_sensitivity_not_hyperparameter_selection", "alpha": 0.40,
              "training_selected_alpha": 0.45, "matched_alpha045_complete_rankings": len(gold),
              "changed_top4_order_qids": changed_order, "changed_top4_set_qids": changed_set,
              "metrics": metrics}
    write_new(run / "retrieval_comparison.json", report)
    paths.update({"model": training / "final.model.json", "training_result": training / "result.json",
                  "selection": training / "selection.json", "training_manifest": training / "manifest.json",
                  "alpha045_rankings": training / "dev.prediction.json", "frozen_training_code": snapshot,
                  "alpha040_rankings": run / "alpha040.prediction.json", "reader_input": run / "reader_input.json",
                  "reader_code": repo / "scripts/run_m3docvqa_external_retrieval_qa.py",
                  "validator": Path(__file__), "shared_validator": Path(__file__).with_name("validate_racs_bge_reader.py"),
                  "training_validator": Path(__file__).with_name("validate_racs_controlled_reader.py")})
    write_new(run / "reader_manifest.json", {**report, "training_job": "15942315", "questions": 2441,
        "reader_model": "Qwen2-VL-7B-Instruct", "bits": 16, "reader_pages_each": 4,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "fingerprints": {key: fingerprint(path) for key, path in paths.items()}})
    print("ALPHA040_INPUT_VALIDATED " + json.dumps(report), flush=True)


def check(run):
    manifest = read_json(run / "reader_manifest.json")
    require(manifest["alpha"] == 0.40 and manifest["training_selected_alpha"] == 0.45, "Wrong experiment")
    for key, record in manifest["fingerprints"].items():
        require(fingerprint(record["path"]) == record, f"Input/code changed during QA: {key}")
    result = validate_outputs(read_json(run / "reader_input.json"),
                             read_json(run / "qa.prediction.json"), read_json(run / "qa.eval.json"))
    result.update({"status": "validated_reader_outputs", "alpha": 0.40, "training_selected_alpha": 0.45,
                   "training_job": "15942315", "comparison_type": manifest["comparison_type"]})
    write_new(run / "validated_result.json", result)
    print("ALPHA040_READER_RESULT " + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "check"))
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    cli = parser.parse_args()
    if cli.mode == "prepare":
        prepare(cli.repo_root.resolve(), cli.run_dir.resolve())
    else:
        check(cli.run_dir.resolve())


if __name__ == "__main__":
    main()
