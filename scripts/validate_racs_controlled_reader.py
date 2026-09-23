#!/usr/bin/env python3
"""Pin the new train-selected CAPP rankings to a separate four-page QA run."""
import argparse
import json
import os
from pathlib import Path

from validate_racs_bge_reader import (
    fingerprint, normalize_pages, read_json, require, validate_outputs, write_new,
)


def validate_training(result, model, selection):
    require(result.get("status") == "validated_training_and_retrieval", "Training/retrieval did not validate")
    require(model.get("train_metadata", {}).get("retrained_after_tuning") is True, "Full refit not confirmed")
    alpha = selection.get("selected_blend_alpha")
    require(alpha == result.get("selected_alpha") == model.get("args", {}).get("blend_alpha") == 0.45,
            "Expected validated job 15942315, alpha 0.45")
    require(selection.get("optimized_metric") == "page@4" and selection.get("tune_eval_qid_count") == 4241,
            "Unexpected validation protocol")
    require(model.get("feature_set") == "all" and len(model.get("feature_names", [])) == 30,
            "Expected full 30-feature model")


def reader_inputs(predictions, qids, expected=2441):
    require(len(qids) == len(set(qids)) == expected and set(predictions) == set(qids), "Question cohort mismatch")
    return {qid: {"qid": qid, "page_retrieval_results": normalize_pages(predictions[qid], qid, 1000)[:4]}
            for qid in qids}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "check"))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    cli = parser.parse_args()
    repo, run = cli.repo_root.resolve(), cli.run_dir.resolve()
    training = repo / "output/racs_controlled_training_15942315"
    if cli.mode == "prepare":
        result = read_json(training / "result.json")
        model = read_json(training / "final.model.json")
        selection = read_json(training / "selection.json")
        validate_training(result, model, selection)
        require(fingerprint(training / "final.model.json")["sha256"] == result["final_model_sha256"], "Model changed")
        require(fingerprint(training / "manifest.json")["sha256"] == result["manifest_sha256"], "Training manifest changed")
        manifest = read_json(training / "manifest.json")
        gold = Path(manifest["inputs"]["dev_gold"]["path"])
        require(fingerprint(gold)["sha256"] == manifest["inputs"]["dev_gold"]["sha256"], "Dev gold changed")
        with gold.open() as handle:
            qids = [str(json.loads(line)["qid"]).strip() for line in handle if line.strip()]
        inputs = reader_inputs(read_json(training / "dev.prediction.json"), qids)
        write_new(run / "reader_input.json", inputs)
        paths = {"gold": gold, "model": training / "final.model.json",
                 "training_result": training / "result.json", "selection": training / "selection.json",
                 "rankings": training / "dev.prediction.json", "reader_input": run / "reader_input.json",
                 "reader_code": repo / "scripts/run_m3docvqa_external_retrieval_qa.py",
                 "validator_code": Path(__file__),
                 "shared_validator": Path(__file__).with_name("validate_racs_bge_reader.py")}
        write_new(run / "reader_manifest.json", {"training_job": "15942315", "alpha": 0.45,
                   "questions": 2441, "reader_pages_each": 4, "model": "Qwen2-VL-7B-Instruct", "bits": 16,
                   "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                   "fingerprints": {key: fingerprint(path) for key, path in paths.items()}})
        print("CONTROLLED_READER_INPUT_VALIDATED questions=2441 pages_each=4 alpha=0.45", flush=True)
    else:
        manifest = read_json(run / "reader_manifest.json")
        for key, recorded in manifest["fingerprints"].items():
            require(fingerprint(recorded["path"]) == recorded, f"Input/code changed: {key}")
        result = validate_outputs(read_json(run / "reader_input.json"),
            read_json(run / "qa.prediction.json"), read_json(run / "qa.eval.json"))
        result.update({"training_job": "15942315", "alpha": 0.45, "status": "validated_reader_outputs"})
        write_new(run / "validated_result.json", result)
        print("CONTROLLED_READER_RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
