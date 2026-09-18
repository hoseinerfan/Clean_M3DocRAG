#!/usr/bin/env python3
"""Validate the retained BGE rankings and their separate RACS reader run.

Standard library only: preflight must run before loading any GPU model.
Original predictions are read-only; all new files use exclusive creation.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle, object_pairs_hook=unique_object)


def write_new(path, payload):
    with Path(path).open("x") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def check_config(config):
    require(Path(config.get("model_name_or_path", "")).name == "bge-reranker-base",
            "Expected bge-reranker-base")
    for key, expected in {"candidate_top_k": 1000, "rerank_top_k": 1000,
                          "blend_alpha": 0.20, "max_length": 512,
                          "max_page_chars": 6000}.items():
        require(config.get(key) == expected, f"Unexpected BGE {key}: {config.get(key)}")


def normalize_pages(row, qid, expected_count):
    require(str(row.get("qid", qid)).strip() == qid, f"qid mismatch: {qid}")
    pages = row.get("page_retrieval_results", [])
    require(len(pages) == expected_count, f"Unexpected candidate count: {qid}")
    normalized = []
    for page in pages:
        require(isinstance(page, list) and len(page) >= 3, f"Malformed page: {qid}")
        doc, index, score = str(page[0]), int(page[1]), float(page[2])
        require(doc.strip() and index >= 0 and index == page[1] and math.isfinite(score),
                f"Invalid page ID, index or score: {qid}")
        normalized.append([doc, index, score])
    require(len({(p[0], p[1]) for p in normalized}) == expected_count,
            f"Duplicate candidate pages: {qid}")
    return normalized


def validate_inputs(gold, base, bge, summary, expected_questions=2441,
                    expected_candidates=1000):
    """Return only the saved top four, never rerank/filter/fill any pages."""
    check_config(summary)
    qids = [str(row["qid"]).strip() for row in gold]
    require(len(qids) == len(set(qids)) == expected_questions and all(qids),
            "Gold question count or unique qids differ")
    require(isinstance(base, dict) and isinstance(bge, dict)
            and set(base) == set(bge) == set(qids), "Prediction qid sets differ from gold")
    require(summary.get("processed_qid_count") == expected_questions,
            "Summary question count differs")
    inputs = {}
    for qid in qids:
        base_pages = normalize_pages(base[qid], qid, expected_candidates)
        bge_pages = normalize_pages(bge[qid], qid, expected_candidates)
        require({(p[0], p[1]) for p in base_pages} == {(p[0], p[1]) for p in bge_pages},
                f"BGE and GPP candidate sets differ: {qid}")
        check_config(bge[qid].get("reranker_metadata", {}).get(
            "standard_cross_encoder_page_reranker", {}))
        require(len(bge_pages) >= 4, f"Fewer than four reader pages: {qid}")
        inputs[qid] = {"qid": qid, "page_retrieval_results": bge_pages[:4]}
    return inputs


def prepare(args):
    with args.gold.open() as handle:
        gold = [json.loads(line, object_pairs_hook=unique_object) for line in handle if line.strip()]
    summary = read_json(args.summary)
    for key, path in (("gold", args.gold), ("base_pred", args.base)):
        require(summary.get(key) and Path(summary[key]).resolve() == path.resolve(),
                f"BGE summary {key} differs from the pinned input")
    inputs = validate_inputs(gold, read_json(args.base), read_json(args.prediction), summary)
    input_path = args.run_dir / "bge_top4.reader_input.json"
    write_new(input_path, inputs)
    repository = Path(__file__).resolve().parents[1]
    paths = {"gold": args.gold, "gpp_base": args.base, "bge_prediction": args.prediction,
             "bge_summary": args.summary, "reader_input": input_path,
             "validator_code": Path(__file__),
             "reader_code": repository / "scripts/run_m3docvqa_external_retrieval_qa.py",
             "launcher_code": repository / "examples/sbatch_racs_bge_reader_top4.sh"}
    manifest = {"input_validation": "passed", "questions": len(inputs),
                "candidate_top_k": 1000, "reader_pages_each": 4,
                "reader_model": "Qwen2-VL-7B-Instruct", "reader_bits": 16,
                "reader_shards": 1, "doc_image_cache_size": 16,
                "bge_blend_alpha": 0.20,
                "fingerprints": {key: fingerprint(path) for key, path in paths.items()},
                "local_model_dir": os.environ.get("LOCAL_MODEL_DIR"),
                "local_data_dir": os.environ.get("LOCAL_DATA_DIR"),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    write_new(args.run_dir / "run_manifest.json", manifest)
    print("BGE_INPUT_VALIDATED " + json.dumps({"questions": len(inputs),
          "same_gpp_candidate_sets": True, "reader_pages_each": 4}), flush=True)


def validate_outputs(inputs, predictions, scores, expected_questions=2441):
    require(len(inputs) == expected_questions and set(inputs) == set(predictions),
            "Reader output does not cover the exact input questions")
    for qid, row in predictions.items():
        pages = normalize_pages(inputs[qid], qid, 4)
        require(row.get("selected_page_retrieval_results") == pages,
                f"Reader did not consume the intended four pages: {qid}")
        require(str(row.get("qid", "")) == qid and isinstance(row.get("pred_answer"), str),
                f"Missing answer or wrong qid: {qid}")
        time_qa = row.get("time_qa")
        require(isinstance(time_qa, (int, float)) and math.isfinite(time_qa) and time_qa >= 0,
                f"Invalid QA time: {qid}")
    overall = scores["overall"]
    for key in ("list_em", "list_f1"):
        require(isinstance(overall.get(key), (int, float)) and math.isfinite(overall[key])
                and 0 <= overall[key] <= 100, f"Invalid evaluation metric: {key}")
    return {"questions": len(predictions), "reader_pages_each": 4, "overall": overall}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "check"))
    parser.add_argument("--run-dir", type=Path, required=True)
    for name in ("gold", "base", "prediction", "summary"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        require(all(getattr(args, name) for name in ("gold", "base", "prediction", "summary")),
                "Prepare requires gold, base, prediction and summary")
        prepare(args)
    else:
        manifest = read_json(args.run_dir / "run_manifest.json")
        for key in ("gold", "reader_input", "reader_code"):
            recorded = manifest["fingerprints"][key]
            require(fingerprint(recorded["path"]) == recorded, f"Input/code changed during QA: {key}")
        stem = args.run_dir / "mmqa_dev_bge_qwen2vl_top4"
        result = validate_outputs(read_json(args.run_dir / "bge_top4.reader_input.json"),
                                  read_json(str(stem) + ".prediction.json"),
                                  read_json(str(stem) + ".eval.json"))
        write_new(args.run_dir / "validated_result.json", result)
        print("BGE_READER_RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
