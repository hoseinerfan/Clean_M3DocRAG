#!/usr/bin/env python3
"""Read-only audit of missing Table 9 QA cells; writes a local evidence report."""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import types

import train_content_aware_pseudo_page_reranker as ca
from validate_racs_bge_reader import unique_object, require, normalize_pages, write_new


def metric_module(repo, remote):
    # Load the unchanged evaluator without importing GPU/dataset packages.
    # word2number is pure Python; permit the installed HPC copy as a fallback.
    try:
        import word2number
    except ModuleNotFoundError:
        root = remote / "env/lib/python3.10/site-packages/word2number"
        spec = importlib.util.spec_from_file_location("word2number", root / "__init__.py",
                                                      submodule_search_locations=[str(root)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    directory = repo / "src/m3docrag/datasets/m3_docvqa"
    package = types.ModuleType("racs_metric_audit")
    package.__path__ = [str(directory)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location("racs_metric_audit.evaluate", directory / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(remote, report_path):
    require(not report_path.exists(), "Refusing to overwrite an existing report")
    repo = Path(__file__).resolve().parents[1]
    metrics = metric_module(repo, remote)
    files = {}

    def read(path, jsonl=False):
        print(f"READ {path.name}", flush=True)
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                f"File changed while reading: {path}")
        files[str(path)] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        if jsonl:
            return [json.loads(line, object_pairs_hook=unique_object) for line in data.splitlines() if line.strip()]
        return json.loads(data, object_pairs_hook=unique_object)

    out = remote / "output"
    rows = read(out / "m3docvqa_mmqa_direct_evidence_pseudo_page_labels/mmqa_dev_pseudo_page_labels_direct_exactonly_adaptive_norm05.augmented_gold.jsonl", True)
    gold = {str(row["qid"]): row for row in rows}
    require(len(rows) == len(gold) == 2441, "Wrong gold cohort")
    answers = {qid: [[str(a["answer"]) for a in row["answers"]]] for qid, row in gold.items()}
    require(sum(bool(ca.gold_page_uids(r)) for r in rows) == 2188, "Wrong labeled cohort")

    def ranking(path):
        data = read(path)
        require(set(data) == set(gold), "Ranking QIDs differ")
        selected, hits = {}, {4: 0, 10: 0}
        for qid, row in data.items():
            pages = normalize_pages(row, qid, 1000)
            selected[qid] = pages[:4]
            truth = ca.gold_page_uids(gold[qid])
            for k in hits:
                hits[k] += bool(truth & {f"{p[0]}_page{p[1]}" for p in pages[:k]})
        return selected, {f"page@{k}": count / 2188 for k, count in hits.items()}

    def qa_check(folder, stem, selected):
        evaluation = read(folder / f"{stem}.eval.json")
        predictions = read(folder / f"{stem}.prediction.json")
        require(set(predictions) == set(gold), "QA QIDs differ")
        qa_answers = {}
        for qid, row in predictions.items():
            require(row.get("qid") == qid, "QA row QID mismatch")
            require(row.get("question") == gold[qid]["question"], "Question text differs")
            actual = normalize_pages({"qid": qid, "page_retrieval_results": row.get("selected_page_retrieval_results", [])}, qid, 4)
            require([p[:2] for p in actual] == [p[:2] for p in selected[qid]],
                    f"Selected ordered pages differ for {qid}")
            require(isinstance(row.get("pred_answer"), str), "Missing answer")
            qa_answers[qid] = row["pred_answer"].strip()
        overall, _ = metrics.evaluate_predictions(qa_answers, answers)
        for metric in ("list_em", "list_f1"):
            require(math.isclose(overall[metric], evaluation["overall"][metric], abs_tol=1e-8, rel_tol=0),
                    f"Saved and recomputed {metric} differ")
        return {"questions": 2441, "ordered_top4_matches": 2441,
                "EM": overall["list_em"], "F1": overall["list_f1"], "matches_saved_eval": True}

    base, base_metrics = ranking(out / "m3docvqa_gpp_hyperlink_node_ablation_exact_maxsim/mmqa_dev_exact_maxsim_gpp_hyperlink_node_no_hyperlink.prediction.json")
    base_qa = qa_check(out / "m3docvqa_final_qa_comparison_exact_maxsim", "mmqa_dev_gpp_no_hyperlink_qwen2vl_top4", base)
    results = {"GPP": {**base_qa, **base_metrics}}
    matrix = out / "m3docvqa_content_aware_exact_maxsim_direct_exactonly_adaptive_norm05_feature_matrix"
    for variant in ("rank_only", "source_only", "content_only"):
        stem = matrix / f"mmqa_train_to_dev_content_aware_feature_{variant}_base_exact_maxsim_gpp"
        model = read(Path(str(stem) + ".model.json"))
        require(model["feature_names"] == ca.resolve_feature_names(variant), "Wrong feature names")
        require(model["args"]["blend_alpha"] == 0.4, "Wrong alpha")
        selected, retrieval = ranking(Path(str(stem) + ".dev.prediction.json"))
        if variant == "rank_only":
            identical = sum([p[:2] for p in selected[qid]] == [p[:2] for p in base[qid]] for qid in gold)
            result = {"questions": 2441, "ordered_top4_matches_gpp": identical,
                      "can_reuse_gpp_qa": identical == 2441, **retrieval}
            if identical == 2441:
                result.update(EM=base_qa["EM"], F1=base_qa["F1"],
                              provenance="GPP QA reused for identical ordered reader pages; not a separate reader run")
        else:
            folder = out / "m3docvqa_final_qa_feature_ablation_direct_exactonly_adaptive_norm05" / variant
            result = {**qa_check(folder, "mmqa_dev_content_aware_qwen2vl_top4", selected), **retrieval}
        results[variant] = result
        print("VALIDATED " + json.dumps({"variant": variant, **result}), flush=True)
        gc.collect()
    report = {"status": "validated_saved_qa_and_rank_identity", "results": results,
              "inputs": files, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "evaluator_sha256": hashlib.sha256((repo / "src/m3docrag/datasets/m3_docvqa/evaluate.py").read_bytes()).hexdigest(),
              "caveat": "Validates saved answers, selected page identities/order and metrics; not a rerun or independent check of historical reader weights."}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_new(report_path, report)
    print(f"SAVED_REPORT {report_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hpc-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    run(args.hpc_root, args.report)
