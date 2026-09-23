#!/usr/bin/env python3
"""Prospective, train-only alpha selection. Never overwrite paper artifacts.

Uses the adjacent, versioned training module. No development data are supplied
to fitting or alpha selection; development evaluation follows final-model save.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import train_content_aware_pseudo_page_reranker as ca

SEED = 13
GRID = ",".join(f"{i / 20:.2f}" for i in range(21))
EXPECTED_COUNTS = {"train_total": 23817, "train_labeled": 21206,
                   "train_in_pool": 17929, "dev_total": 2441, "dev_labeled": 2188}
SOURCE_VARIANTS = {
    "gpp_no_hyperlink": "no_hyperlink",
    "gpp_doc_hyperlink": "docnode_to_hyperlink_docs",
    "gpp_page_hyperlink": "pagenode_to_hyperlink_pages",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    # Exclusive creation prevents accidental replacement of earlier evidence.
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_paths(repo):
    """Explicit prospective source pairs, NOT recovered historical training provenance."""
    repo = Path(repo).resolve()
    out = repo / "output"
    labels = out / "m3docvqa_mmqa_direct_evidence_pseudo_page_labels"
    paths = {}
    for split, short in (("train", "train"), ("dev", "dev")):
        suffix = "_train" if split == "train" else ""
        paths[f"{split}_gold"] = labels / f"mmqa_{short}_pseudo_page_labels_direct_exactonly_adaptive_norm05.augmented_gold.jsonl"
        paths[f"{split}_base"] = out / f"m3docvqa_gpp_hyperlink_node_ablation_exact_maxsim{suffix}" / f"mmqa_{short}_exact_maxsim_gpp_hyperlink_node_no_hyperlink.prediction.json"
        paths[f"{split}_text"] = repo.parent / "outputs/m3docvqa_page_text" / f"m3docvqa_{short}_page_text.jsonl"
        legacy_suffix = "_train_real" if split == "train" else ""
        folder = out / f"m3docvqa_gpp_hyperlink_node_ablation_mmr_target1{legacy_suffix}"
        for label, variant in SOURCE_VARIANTS.items():
            paths[f"{split}_source:{label}"] = folder / f"mmqa_{short}_gpp_hyperlink_node_{variant}.prediction.json"
    return paths


def preflight(repo):
    paths = input_paths(repo)
    missing = []
    for name, path in paths.items():
        ok = path.is_file() and path.stat().st_size > 0
        print(f"{'FOUND' if ok else 'MISSING'} {name} = {path}", flush=True)
        if not ok:
            missing.append(str(path))
    require(not missing, "Required inputs missing; no training started. Do not substitute source files silently.\n" + "\n".join(missing))
    return paths


def training_args(paths, output):
    argv = ["controlled-training"]
    for option, key in (("train-gold", "train_gold"), ("eval-gold", "dev_gold"),
                        ("train-base-pred", "train_base"), ("eval-base-pred", "dev_base"),
                        ("train-page-text-jsonl", "train_text"), ("eval-page-text-jsonl", "dev_text")):
        argv += [f"--{option}", str(paths[key])]
    argv += ["--output-model-json", str(output / "final.model.json"),
             "--output-prediction-json", str(output / "dev.prediction.json"),
             "--output-summary-json", str(output / "result.json"),
             "--feature-set", "all", "--candidate-top-k", "1000",
             "--negative-sampling-strategy", "rank_stratified",
             "--negatives-per-band", "10", "--max-negatives-per-qid", "64",
             "--model-type", "logistic", "--training-objective", "weighted_bce",
             "--epochs", "80", "--learning-rate", "0.01", "--weight-decay", "0.0001",
             "--batch-size", "65536", "--positive-weight-cap", "20",
             "--seed", str(SEED), "--inference-mode", "blend_rerank",
             "--auto-tune-blend-alpha", "--tune-fraction", "0.20",
             "--tune-blend-alpha-grid", GRID, "--tune-hit-k", "4"]
    for split, flag in (("train", "train-source"), ("dev", "eval-source")):
        for label in SOURCE_VARIANTS:
            argv += [f"--{flag}", f"{label}={paths[f'{split}_source:{label}']}"]
    previous = sys.argv
    try:
        sys.argv = argv
        args = ca.parse_args()
    finally:
        sys.argv = previous
    args.active_feature_names = ca.resolve_feature_names("all")
    require(len(args.active_feature_names) == 30, "Expected the 30 non-visual features")
    return args


def checked_gold(path):
    rows = ca.read_jsonl(path)
    qids = [str(row.get("qid", "")).strip() for row in rows]
    require(all(qids) and len(set(qids)) == len(qids), f"Missing/duplicate question IDs in {path}")
    return dict(zip(qids, rows))


def check_base(gold, base, pages, top_k):
    in_pool = 0
    for qid, row in gold.items():
        records = ca.ranked_page_records(base.get(qid), top_k)
        require(records, f"Missing candidates for {qid}")
        require(str(row.get("question", "")).strip(), f"Missing question text: {qid}")
        missing = [r["uid"] for r in records if r["uid"] not in pages]
        require(not missing, f"Missing page text for {qid}: {missing[:3]}")
        in_pool += bool(ca.gold_page_uids(row) & {r["uid"] for r in records})
    return in_pool


def load_sources(paths, split, qids, top_k):
    maps = {}
    for label in SOURCE_VARIANTS:
        print(f"LOAD_SOURCE {split} {label}", flush=True)
        pred = ca.load_prediction(paths[f"{split}_source:{label}"])
        require(set(qids) <= set(pred), f"Missing {split} questions in source {label}")
        maps[label] = ca.source_maps({qid: pred[qid] for qid in qids}, top_k)
        require(all(maps[label].values()), f"Empty {split} candidate rankings in {label}")
        del pred
        gc.collect()
    return maps


def fit(gold, base, pages, sources, args):
    X, y, row_weights, metadata, qids = ca.build_matrix(
        gold=gold, base_pred=base, page_features=pages,
        source_maps_by_label=sources, args=args, return_query_ids=True)
    require(len(X) > 0 and set(y.tolist()) == {0.0, 1.0}, "Training requires positive and negative examples")
    X_std, mean, std = ca.standardize_train(X)
    weights, bias, history = ca.train_scorer(X_std, y, args, row_weights=row_weights, query_ids=qids)
    require(np.isfinite(mean).all() and np.isfinite(std).all() and np.isfinite(weights).all() and np.isfinite(bias), "Nonfinite fitted parameters")
    return mean, std, weights, bias, history, metadata


def model_payload(fitted, args):
    mean, std, weights, bias, history, metadata = fitted
    return {**ca.scorer_to_json(weights, bias), "mean": mean.tolist(), "std": std.tolist(),
            "feature_names": args.active_feature_names, "feature_set": "all",
            "args": vars(args).copy(), "history": history, "train_metadata": metadata}


def assert_selection(summary, args, tune_count):
    scores = summary["alpha_scores"]
    require([r["blend_alpha"] for r in scores] == ca.parse_alpha_grid(args.tune_blend_alpha_grid), "Alpha grid mismatch")
    require(summary["optimized_metric"] == "page@4", "Unexpected tuning objective")
    require(summary["tune_eval_qid_count"] == tune_count, "Validation questions were silently skipped")
    best = max(scores, key=lambda row: (row["page@4"], -row["blend_alpha"]))
    require(best["blend_alpha"] == summary["selected_blend_alpha"], "Incorrect alpha selection/tie-break")


def run(paths, output):
    args = training_args(paths, output)
    manifest = {
        "protocol": "prospective_training_holdout_v1",
        "not_historical_training_reconstruction": True,
        "selection": {"seed": SEED, "fraction": 0.20, "grid": GRID,
                      "metric": "page@4", "tie_break": "smallest alpha",
                      "split_unit": "question", "split_population": "all pseudo-page-labeled training questions",
                      "no_in_pool_positive": "retained for validation; omitted from fitting by build_matrix"},
        "settings": vars(args).copy(), "inputs": {},
        "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "thread_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    }
    for name, path in paths.items():
        print(f"HASH_INPUT {name}", flush=True)
        manifest["inputs"][name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
    code = output / "code"
    code.mkdir()
    for path in (Path(__file__), Path(ca.__file__)):
        shutil.copy2(path, code / path.name)
    manifest["code_sha256"] = {p.name: sha256(p) for p in code.iterdir()}
    write_json(output / "manifest.json", manifest)

    print("LOAD_TRAIN_INPUTS", flush=True)
    train_all = checked_gold(paths["train_gold"])
    require(len(train_all) == EXPECTED_COUNTS["train_total"], "Unexpected training question count")
    train = {qid: row for qid, row in train_all.items() if ca.gold_page_uids(row)}
    require(len(train) == EXPECTED_COUNTS["train_labeled"], "Unexpected labeled training question count")
    fit_gold, tune_gold = ca.split_gold_for_tuning(train, tune_fraction=0.20, seed=SEED)
    require(not (fit_gold.keys() & tune_gold.keys()), "Fit/validation overlap")
    write_json(output / "split.json", {"fit_qids": sorted(fit_gold), "validation_qids": sorted(tune_gold),
               "unlabeled_excluded_qids": sorted(train_all.keys() - train.keys()), "seed": SEED})
    base = ca.load_prediction(paths["train_base"])
    pages = ca.load_page_features(paths["train_text"])
    require(check_base(train, base, pages, 1000) == EXPECTED_COUNTS["train_in_pool"], "Unexpected number of in-pool positives")
    sources = load_sources(paths, "train", train, 1000)
    print(f"FIT questions={len(fit_gold)}; HOLDOUT questions={len(tune_gold)}", flush=True)
    fitted = fit(fit_gold, base, pages, sources, args)
    write_json(output / "fit.model.json", model_payload(fitted, args))
    mean, std, weights, bias, _, _ = fitted
    tuning = ca.tune_blend_alpha(tune_gold=tune_gold, base_pred=base, page_features=pages,
        source_maps_by_label=sources, mean=mean, std=std, weights=weights, bias=bias, args=args)
    assert_selection(tuning, args, len(tune_gold))
    args.blend_alpha = float(tuning["selected_blend_alpha"])
    write_json(output / "selection.json", tuning)
    print("TRAIN_ONLY_SELECTION " + json.dumps(tuning), flush=True)

    print("REFIT_ALL_ELIGIBLE_TRAINING_QUESTIONS", flush=True)
    fitted = fit(train, base, pages, sources, args)
    model = model_payload(fitted, args)
    model["train_metadata"].update({"retrained_after_tuning": True, "auto_tune_blend_alpha": True,
        "fit_qid_count": len(fit_gold), "tune_qid_count": len(tune_gold),
        "selected_blend_alpha": args.blend_alpha, "tuning_summary": tuning})
    require(model["train_metadata"]["train_qid_with_positive_in_pool"] == EXPECTED_COUNTS["train_in_pool"], "Unexpected refit population")
    write_json(output / "final.model.json", model)
    train_qids = set(train_all)
    del base, pages, sources, train, train_all, fit_gold, tune_gold
    gc.collect()

    print("FINAL_MODEL_SAVED; BEGIN_DEVELOPMENT_EVALUATION", flush=True)
    dev = checked_gold(paths["dev_gold"])
    require(len(dev) == EXPECTED_COUNTS["dev_total"] and sum(bool(ca.gold_page_uids(r)) for r in dev.values()) == EXPECTED_COUNTS["dev_labeled"], "Unexpected dev cohort")
    require(not train_qids.intersection(dev), "Training/development question overlap")
    base = ca.load_prediction(paths["dev_base"])
    require(set(base) == set(dev), "Development base QID set differs")
    pages = ca.load_page_features(paths["dev_text"])
    check_base(dev, base, pages, 1000)
    sources = load_sources(paths, "dev", dev, 1000)
    mean, std, weights, bias, _, _ = fitted
    # Only question text, never development relevance labels/answers, enters inference.
    questions = {qid: {"qid": qid, "question": row["question"]} for qid, row in dev.items()}
    predictions, priors = ca.apply_reranker(gold=questions, base_pred=base, page_features=pages,
        source_maps_by_label=sources, mean=mean, std=std, weights=weights, bias=bias, args=args)
    require(set(predictions) == set(dev), "Missing development predictions")
    for qid in dev:
        before = ca.ranked_page_records(base[qid], 1000)
        after = ca.ranked_page_records(predictions[qid], 1000)
        require({r["uid"] for r in before} == {r["uid"] for r in after}, f"Candidate pool changed: {qid}")
    write_json(output / "dev.prediction.json", predictions)
    del priors
    ks = [1, 2, 4, 5, 8, 10, 20, 100, 1000]
    metrics = [ca.evaluate_run(label=label, pred=pred, gold=dev, recall_ks=ks)
               for label, pred in (("GPP", base), ("CAPP_train_selected", predictions))]
    report = {"status": "validated_training_and_retrieval", "qa_not_yet_run": True,
              "selected_alpha": args.blend_alpha, "selection": tuning,
              "train_metadata": model["train_metadata"], "metrics": metrics,
              "manifest_sha256": sha256(output / "manifest.json"),
              "final_model_sha256": sha256(output / "final.model.json")}
    write_json(output / "result.json", report)
    ca.write_table(output / "retrieval.table.md", metrics, ks)
    print("CONTROLLED_TRAINING_RESULT " + json.dumps({"status": report["status"],
        "selected_alpha": args.blend_alpha, "metrics": metrics, "output": str(output)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    cli = parser.parse_args()
    paths = preflight(cli.repo_root)
    if cli.preflight_only:
        print("INPUT_FILES_PRESENT: CPU job may be submitted; full content checks run on compute node.")
        return
    require(cli.output_dir is not None, "--output-dir is required")
    output = cli.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        run(paths, output)
    except Exception as error:
        write_json(output / "failure.json", {"status": "failed_not_a_result", "error": str(error),
                   "type": type(error).__name__, "elapsed_seconds": time.time() - started})
        raise


if __name__ == "__main__":
    main()
