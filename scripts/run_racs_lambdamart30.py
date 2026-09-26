#!/usr/bin/env python3
"""Prospective LambdaMART baseline using CAPP's exact 30-feature pipeline.

Train-only alpha selection, full-training refit, then untouched development QA.
Historical 40-feature and CAPP artifacts are read-only and never replaced.
"""
import argparse
import copy
import gc
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys

import numpy as np
import run_racs_controlled_training as controlled
import train_content_aware_pseudo_page_reranker as ca
from validate_racs_bge_reader import (
    fingerprint, normalize_pages, read_json, require, validate_outputs, write_new,
)

FEATURES = ca.resolve_feature_names("all")
CAPP_MODEL = (
    "output/m3docvqa_content_aware_exact_maxsim_direct_exactonly_adaptive_norm05/"
    "mmqa_train_to_dev_content_aware_fixed_alpha_0p40_base_exact_maxsim_gpp_"
    "direct_exactonly_adaptive_norm05.model.json"
)
TREE_PARAMS = dict(objective="lambdarank", metric="ndcg", n_estimators=300,
                   learning_rate=0.05, num_leaves=31, min_data_in_leaf=30,
                   subsample=0.9, colsample_bytree=0.9, random_state=13,
                   n_jobs=8, verbosity=-1)


def group_sizes(qids):
    groups, seen, previous = [], set(), None
    for qid in qids:
        if qid != previous:
            require(qid not in seen, "Training rows are not contiguous by question")
            seen.add(qid)
            groups.append(0)
            previous = qid
        groups[-1] += 1
    require(bool(groups), "Empty training data")
    return groups


def feature_rows(qid, question, records, pages, sources):
    """Same feature extraction and full-candidate context as CAPP, no labels."""
    if not records:
        return np.empty((0, len(FEATURES)), dtype=np.float32)
    ranks, within_doc, counts = ca.doc_rank_maps(records)
    normalized = ca.normalize_scores(records)
    profile = ca.question_profile({"question": question})
    return np.asarray([
        ca.feature_vector(r, records=records, base_norm_scores=normalized,
                          doc_rank=ranks, doc_page_rank=within_doc, doc_page_count=counts,
                          question=profile, page_features=pages,
                          source_maps_by_label=sources, qid=qid, feature_names=FEATURES)
        for r in records
    ], dtype=np.float32)


def fit(gold, base, pages, sources, args, output, stem):
    import lightgbm as lgb
    print(f"BUILD_MATRIX {stem} questions={len(gold)}", flush=True)
    X, y, weights, meta, qids = ca.build_matrix(
        gold=gold, base_pred=base, page_features=pages,
        source_maps_by_label=sources, args=args, return_query_ids=True)
    require(X.shape[1] == 30 and np.isfinite(X).all(), "Invalid 30-feature matrix")
    require(set(y.tolist()) == {0, 1}, "Expected binary pseudo-page relevance")
    groups = group_sizes(qids)
    X, mean, std = ca.standardize_train(X)
    model = lgb.LGBMRanker(**TREE_PARAMS)
    print(f"FIT_LAMBDAMART {stem} rows={len(y)} groups={len(groups)}", flush=True)
    model.fit(X, y.astype(int), group=groups, sample_weight=weights,
              feature_name=FEATURES)
    require(model.n_features_in_ == 30, "Fitted model is not 30-dimensional")
    booster_path = output / f"{stem}.booster.txt"
    require(not booster_path.exists(), "Refusing to overwrite booster")
    model.booster_.save_model(str(booster_path))
    write_new(output / f"{stem}.model.json", {
        "feature_names": FEATURES, "feature_set": "all", "mean": mean.tolist(),
        "std": std.tolist(), "model_info": {"backend": "lightgbm", "objective": "lambdarank",
        "params": model.get_params(), "feature_importance": dict(zip(
            FEATURES, model.feature_importances_.tolist()))}, "train_metadata": meta,
        "training_args": vars(args), "booster": fingerprint(booster_path),
        "label_definition": "binary pseudo-page relevance; same sampled rows as controlled CAPP",
    })
    # Evaluate the persisted model, not an unsaved in-memory-only estimator.
    booster = lgb.Booster(model_file=str(booster_path))
    require(booster.feature_name() == FEATURES, "Persisted feature order mismatch")
    np.testing.assert_allclose(booster.predict(X[:100]), model.predict(X[:100]), rtol=0, atol=1e-12)
    return booster, mean, std, meta


def score(qid, row, base, pages, sources, fitted):
    model, mean, std, _ = fitted
    records = ca.ranked_page_records(base[qid], 1000)
    X = feature_rows(qid, row["question"], records, pages, sources)
    values = np.asarray(model.predict(ca.standardize_eval(X, mean, std), num_threads=8), dtype=np.float32)
    require(len(values) == len(records) and np.isfinite(values).all(), "Invalid tree scores")
    for record, value in zip(records, values):
        record["learned_score"] = float(value)
    return records


def select_alpha(gold, base, pages, sources, fitted, args):
    grid = ca.parse_alpha_grid(controlled.GRID)
    hits = {a: 0 for a in grid}
    trial = copy.copy(args)
    for i, (qid, row) in enumerate(gold.items(), 1):
        records = score(qid, row, base, pages, sources, fitted)
        truth = ca.gold_page_uids(row)
        require(bool(truth), "Unlabeled validation question")
        for alpha in grid:
            trial.blend_alpha = alpha
            ranked = ca.rerank_records([dict(r) for r in records], trial)
            hits[alpha] += bool(truth & {r["uid"] for r in ranked[:4]})
        if i % 250 == 0:
            print(f"TRAIN_HOLDOUT {i}/{len(gold)}", flush=True)
    scores = [{"blend_alpha": a, "page@4": hits[a] / len(gold), "hit_count": hits[a]} for a in grid]
    best = max(scores, key=lambda r: (r["page@4"], -r["blend_alpha"]))
    result = {"selected_blend_alpha": best["blend_alpha"], "optimized_metric": "page@4",
              "optimized_metric_value": best["page@4"], "tune_eval_qid_count": len(gold),
              "alpha_scores": scores, "selection_split": "training_holdout_only"}
    controlled.assert_selection(result, args, len(gold))
    return result


def checked_inputs(repo):
    paths = controlled.preflight(repo)
    paths["capp_reference_model"] = repo / CAPP_MODEL
    require(len(FEATURES) == 30 and read_json(paths["capp_reference_model"])["feature_names"] == FEATURES,
            "Feature names/order differ from saved CAPP model")
    return paths


def train(repo, output):
    import lightgbm as lgb
    paths = checked_inputs(repo)
    args = controlled.training_args(paths, output)
    output.mkdir(parents=True, exist_ok=False)
    code = output / "code"
    code.mkdir()
    code_paths = [Path(__file__), Path(ca.__file__), Path(controlled.__file__),
                  Path(__file__).with_name("validate_racs_bge_reader.py")]
    for path in code_paths:
        shutil.copy2(path, code / path.name)
    manifest = {"protocol": "lambdamart30_capp_features_train_only_v1", "feature_names": FEATURES,
                "tree_parameters": TREE_PARAMS, "pair_construction_args": vars(args),
                "selection": {"population": "21206 pseudo-page-labeled training questions",
                              "seed": 13, "holdout_fraction": 0.2, "grid": controlled.GRID,
                              "metric": "page@4", "tie_break": "smallest alpha"},
                "versions": {"python": sys.version, "numpy": np.__version__, "lightgbm": lgb.__version__},
                "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "inputs": {}, "code": {p.name: fingerprint(code / p.name) for p in code_paths}}
    for name, path in paths.items():
        print(f"HASH_INPUT {name}", flush=True)
        manifest["inputs"][name] = fingerprint(path)
    write_new(output / "manifest.json", manifest)
    all_train = controlled.checked_gold(paths["train_gold"])
    require(len(all_train) == 23817, "Unexpected training cohort")
    gold = {q: r for q, r in all_train.items() if ca.gold_page_uids(r)}
    require(len(gold) == 21206, "Unexpected labeled training cohort")
    fit_gold, holdout = ca.split_gold_for_tuning(gold, tune_fraction=0.2, seed=13)
    require(not fit_gold.keys() & holdout.keys(), "Fit/holdout overlap")
    write_new(output / "split.json", {"fit_qids": sorted(fit_gold), "validation_qids": sorted(holdout)})
    base = ca.load_prediction(paths["train_base"])
    pages = ca.load_page_features(paths["train_text"])
    require(controlled.check_base(gold, base, pages, 1000) == 17929, "Unexpected training positives")
    sources = controlled.load_sources(paths, "train", gold, 1000)
    fitted = fit(fit_gold, base, pages, sources, args, output, "fit")
    selection = select_alpha(holdout, base, pages, sources, fitted, args)
    write_new(output / "selection.json", selection)
    print("TRAIN_ONLY_SELECTION " + json.dumps(selection), flush=True)
    args.blend_alpha = selection["selected_blend_alpha"]
    fitted = fit(gold, base, pages, sources, args, output, "final")
    require(fitted[3]["train_qid_with_positive_in_pool"] == 17929, "Unexpected refit cohort")
    train_qids = set(all_train)
    del all_train, gold, fit_gold, holdout, base, pages, sources
    gc.collect()
    print("FINAL_MODEL_SAVED; BEGIN_DEVELOPMENT_EVALUATION", flush=True)
    dev = controlled.checked_gold(paths["dev_gold"])
    require(len(dev) == 2441 and sum(bool(ca.gold_page_uids(r)) for r in dev.values()) == 2188,
            "Unexpected development cohort")
    require(not train_qids & dev.keys(), "Train/development overlap")
    base = ca.load_prediction(paths["dev_base"])
    require(set(base) == set(dev), "Missing/extra base questions")
    pages = ca.load_page_features(paths["dev_text"])
    controlled.check_base(dev, base, pages, 1000)
    sources = controlled.load_sources(paths, "dev", dev, 1000)
    predictions = {}
    for i, (qid, row) in enumerate(dev.items(), 1):
        records = score(qid, {"question": row["question"]}, base, pages, sources, fitted)
        ranked = ca.rerank_records(records, args)
        predictions[qid] = {"qid": qid, "page_retrieval_results": [r["raw"] for r in ranked]}
        if i % 250 == 0:
            print(f"DEVELOPMENT {i}/{len(dev)}", flush=True)
    inputs = reader_inputs(base, predictions)
    write_new(output / "dev.prediction.json", predictions)
    write_new(output / "reader_input.json", inputs)
    metrics = [ca.evaluate_run(label=label, pred=pred, gold=dev, recall_ks=[1, 2, 4, 8, 10, 100, 1000])
               for label, pred in (("GPP", base), ("LambdaMART30", predictions))]
    result = {"status": "validated_training_and_retrieval", "feature_names": FEATURES,
              "selected_alpha": args.blend_alpha, "selection": selection, "metrics": metrics,
              "fingerprints": {name: fingerprint(output / name) for name in
                  ("final.model.json", "final.booster.txt", "manifest.json", "split.json",
                   "selection.json", "dev.prediction.json", "reader_input.json")}}
    write_new(output / "result.json", result)
    print("LAMBDAMART30_RESULT " + json.dumps(result), flush=True)


def reader_inputs(base, predictions, count=2441, candidates=1000):
    require(len(base) == count and set(base) == set(predictions), "Wrong prediction QIDs")
    inputs = {}
    for qid in base:
        before = normalize_pages(base[qid], qid, candidates)
        after = normalize_pages(predictions[qid], qid, candidates)
        require({tuple(p[:2]) for p in before} == {tuple(p[:2]) for p in after}, "Candidate pool changed")
        require(len(after) >= 4, "Reader needs four pages")
        inputs[qid] = {"qid": qid, "page_retrieval_results": after[:4]}
    return inputs


def prepare_reader(training, output):
    result = read_json(training / "result.json")
    require(result["status"] == "validated_training_and_retrieval" and result["feature_names"] == FEATURES,
            "Training has not validated with exactly the CAPP features")
    for recorded in result["fingerprints"].values():
        require(fingerprint(recorded["path"]) == recorded, "Training output changed")
    output.mkdir(parents=True, exist_ok=False)
    gold = Path(read_json(training / "manifest.json")["inputs"]["dev_gold"]["path"])
    inputs = read_json(training / "reader_input.json")
    require(len(inputs) == 2441 and all(len(r["page_retrieval_results"]) == 4 for r in inputs.values()),
            "Invalid four-page reader input")
    shutil.copy2(training / "reader_input.json", output / "reader_input.json")
    paths = {"reader_input": output / "reader_input.json", "gold": gold,
             "training_result": training / "result.json", "workflow_code": Path(__file__),
             "reader_code": Path(__file__).with_name("run_m3docvqa_external_retrieval_qa.py"),
             "qa_evaluator": Path(__file__).with_name("evaluate_m3docvqa_qa_runs.py"),
             "shared_validator": Path(__file__).with_name("validate_racs_bge_reader.py")}
    write_new(output / "run_manifest.json", {"method": "LambdaMART30", "alpha": result["selected_alpha"],
        "training_directory": str(training), "reader_model": "Qwen2-VL-7B-Instruct", "bits": 16,
        "questions": 2441, "reader_pages_each": 4, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "fingerprints": {k: fingerprint(p) for k, p in paths.items()}})


def check_reader(output):
    from evaluate_m3docvqa_qa_runs import summarize
    manifest = read_json(output / "run_manifest.json")
    for recorded in manifest["fingerprints"].values():
        require(fingerprint(recorded["path"]) == recorded, "Reader input/code changed")
    result = validate_outputs(read_json(output / "reader_input.json"),
        read_json(output / "qa.prediction.json"), read_json(output / "qa.eval.json"))
    recomputed = summarize("LambdaMART30", output / "qa.prediction.json",
                           Path(manifest["fingerprints"]["gold"]["path"]), 2441)
    for metric, key in (("list_em", "em"), ("list_f1", "f1")):
        require(math.isclose(result["overall"][metric], recomputed[key], abs_tol=1e-8, rel_tol=0),
                f"Independent QA evaluation disagrees: {metric}")
    result.update(status="validated_reader_outputs", method="LambdaMART30", feature_count=30,
                  alpha=manifest["alpha"], training_directory=manifest["training_directory"])
    write_new(output / "validated_result.json", result)
    print("LAMBDAMART30_READER_RESULT " + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "train", "prepare-reader", "check-reader"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--training-dir", type=Path)
    args = parser.parse_args()
    if args.mode == "preflight":
        checked_inputs(args.repo_root.resolve())
        print("LAMBDAMART30_PREFLIGHT_PASSED", flush=True)
        return
    require(args.run_dir is not None, "--run-dir required")
    output = args.run_dir.resolve()
    if args.mode == "train":
        train(args.repo_root.resolve(), output)
    elif args.mode == "prepare-reader":
        require(args.training_dir is not None, "--training-dir required")
        prepare_reader(args.training_dir.resolve(), output)
    else:
        check_reader(output)


if __name__ == "__main__":
    main()
