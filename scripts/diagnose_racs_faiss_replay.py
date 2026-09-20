#!/usr/bin/env python3
"""Four-question FAISS replay diagnosis, never a runtime result or auto-fix.

Reads the failed online bundle and existing assets. Uses the production page
aggregation routine for both supported score sources, on identical ANN hits.
Preserves the index metric, benchmark nprobe=4, model weights and defaults.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

import benchmark_racs_online as online

bench = online.bench
ENCODER_PATHS = ("direct_gpu", "cpu_scoring_encoder", "accelerator_gpu")
SCORE_SOURCES = ("embedding", "faiss_distance")


def diagnostic_qids(bundle):
    qids = bundle.get("warmup_qids", [])
    if len(qids) != 4 or len(set(qids)) != 4:
        raise ValueError("Require exactly four distinct existing warm-up qids")
    if set(qids) & set(bundle.get("measured_qids", [])):
        raise ValueError("Diagnostic warm-ups must not overlap measured qids")
    for qid in qids:
        if qid not in bundle["questions"]:
            raise ValueError(f"Missing question: {qid}")
        rows = bundle["online"]["baseline_references"][qid]
        bench.replay.compare_rows(rows, rows, 1000)
    if bundle["online"]["nprobe"] != 4:
        raise ValueError("Expected the fixed nprobe=4 reconstruction")
    return list(qids)


def compare_candidate_rows(expected, actual):
    """Keep strict production tolerances; report, rather than hide, mismatch."""
    expected_pairs = [tuple(row[:2]) for row in expected]
    actual_pairs = [tuple(row[:2]) for row in actual]
    result = {"expected_pages": len(expected), "actual_pages": len(actual),
              "shared_pages": len(set(expected_pairs) & set(actual_pairs)),
              "top4_order_matches": expected_pairs[:4] == actual_pairs[:4],
              "candidate_sets_match": False, "complete_order_matches": False,
              "scores_close": False}
    if len(expected) == len(actual) == 1000:
        result.update(bench.replay.compare_rows(expected, actual, 1000))
    first = next((i for i, pair in enumerate(zip(expected_pairs, actual_pairs))
                  if pair[0] != pair[1]), None)
    if first is None and len(expected_pairs) != len(actual_pairs):
        first = min(len(expected_pairs), len(actual_pairs))
    result.update(first_order_mismatch_rank=None if first is None else first + 1,
                  expected_top3=expected[:3], actual_top3=actual[:3])
    return result


class FixedSearch:
    """Replay the same fresh search results through both production branches."""
    def __init__(self, query, distances, indices):
        import numpy as np
        self.query = np.array(query, copy=True)
        self.distances, self.indices = distances, indices

    def search(self, query, k):
        import numpy as np
        if not np.array_equal(query, self.query) or k != self.indices.shape[1]:
            raise ValueError("Diagnostic aggregation changed the query or hit budget")
        return self.distances, self.indices


def candidate_modes(retriever, query_meta):
    import numpy as np
    query = query_meta["embeddings"].float().numpy().astype(np.float32)
    distances, indices = retriever.index.search(query, 1000)
    fixed = FixedSearch(query, distances, indices)
    rows = {}
    for source in SCORE_SOURCES:
        rows[source] = retriever.rag._retrieve_pages_from_index_query_meta(
            query_meta, fixed, retriever.token_uids,
            retriever.token_table if source == "embedding" else None, 1000, False)
    return rows


def sampled_index_alignment(index, token_table, token_uids, faiss_module):
    """Inspect 16 evenly spaced IVF lists, at most four vectors per list.

    IVFFlat stores raw float32 vectors. Release both list buffers after access;
    no direct-map creation, index mutation, or full-index reconstruction.
    A passing sample is not proof that every stored vector is aligned.
    """
    import numpy as np
    if type(index).__name__ != "IndexIVFFlat" or int(index.code_size) != int(index.d) * 4:
        raise ValueError("Alignment diagnostic requires raw-float32 IndexIVFFlat")
    if (tuple(token_table.shape) != (int(index.ntotal), int(index.d))
            or len(token_uids) != int(index.ntotal) or int(index.nlist) < 1):
        raise ValueError("Index/token table dimensions disagree")
    lists = sorted(set(int(v) for v in np.linspace(0, int(index.nlist) - 1, 16)))
    samples, empty_lists = [], []
    for list_id in lists:
        size = int(index.invlists.list_size(list_id))
        if not size:
            empty_lists.append(list_id)
            continue
        ids_ptr = index.invlists.get_ids(list_id)
        codes_ptr = None
        try:
            codes_ptr = index.invlists.get_codes(list_id)
            ids = faiss_module.rev_swig_ptr(ids_ptr, size)
            codes = faiss_module.rev_swig_ptr(codes_ptr, size * int(index.code_size))
            vectors = codes.view(np.float32).reshape(size, int(index.d))
            for offset in sorted(set(int(v) for v in np.linspace(0, size - 1, 4))):
                token_id = int(ids[offset])
                if not 0 <= token_id < len(token_table):
                    raise ValueError(f"Stored token id out of range: {token_id}")
                stored, current = vectors[offset], token_table[token_id]
                samples.append({"list_id": list_id, "offset": offset, "token_id": token_id,
                    "mapped_page_uid": token_uids[token_id],
                    "exact_vector_match": bool(np.array_equal(stored, current)),
                    "vector_close": bool(np.allclose(stored, current, rtol=1e-6, atol=1e-8)),
                    "max_absolute_difference": float(np.max(np.abs(stored - current)))})
        finally:
            if codes_ptr is not None:
                index.invlists.release_codes(list_id, codes_ptr)
            index.invlists.release_ids(list_id, ids_ptr)
    return {"scope": "sample only, not exhaustive index verification", "sample_count": len(samples),
            "all_sampled_vectors_close": bool(samples) and all(row["vector_close"] for row in samples),
            "empty_sampled_lists": empty_lists, "samples": samples}


def query_summary(meta, reference=None):
    import numpy as np
    values = meta["embeddings"].float().numpy()
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Invalid query embeddings")
    norms = np.linalg.norm(values, axis=1)
    result = {"embedding_shape": list(values.shape), "embedding_dtype": str(meta["embeddings"].dtype),
              "embedding_float32_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
              "token_ids": meta["token_ids"].tolist(), "raw_tokens": meta["raw_tokens"],
              "kept_token_indices": meta["kept_token_indices"],
              "norm_min": float(norms.min()), "norm_mean": float(norms.mean()),
              "norm_max": float(norms.max())}
    if reference is not None:
        previous = reference["embeddings"].float().numpy()
        same_tokens = result["token_ids"] == reference["token_ids"].tolist()
        result["same_token_ids_as_direct_gpu"] = same_tokens
        if same_tokens and previous.shape == values.shape:
            result["max_absolute_difference_from_direct_gpu"] = float(np.max(np.abs(values - previous)))
            result["values_close_to_direct_gpu"] = bool(np.allclose(values, previous, rtol=1e-6, atol=1e-8))
    return result


def comparison_totals(results, qids):
    totals = {}
    for encoder in ENCODER_PATHS:
        for source in SCORE_SOURCES:
            key = f"{encoder}/{source}"
            cells = [row for row in results if row["encoder_path"] == encoder and row["score_source"] == source]
            if len(cells) != len(qids) or {row["qid"] for row in cells} != set(qids):
                raise ValueError(f"Incomplete or duplicate diagnostic comparisons: {key}")
            totals[key] = {"questions": len(cells),
                "candidate_set_matches": sum(row["comparison"]["candidate_sets_match"] for row in cells),
                "complete_order_matches": sum(row["comparison"]["complete_order_matches"] for row in cells),
                "complete_order_and_score_matches": sum(row["comparison"]["complete_order_matches"]
                    and row["comparison"]["scores_close"] for row in cells)}
    return totals


def run(bundle_path, output_dir):
    import faiss
    import numpy as np
    import torch
    from accelerate import Accelerator
    import run_m3docvqa_external_retrieval_qa as qa

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA GPU required")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    bundle = bench.replay.read_json(bundle_path)
    qids = diagnostic_qids(bundle)
    # Old failed bundles predate this explicit field and used embedding dots.
    # Annotate only the in-memory diagnostic loader input; both score branches
    # are still tested, and no original bundle/default is rewritten.
    bundle["online"].setdefault("faiss_page_score_source", "embedding")
    cli = SimpleNamespace(data_name="m3-docvqa", split="dev", bits=16, model_name_or_path="Qwen2-VL-7B-Instruct")
    retriever = online.OnlineRetriever(bundle, qa.M3DocVQADataset(qa.make_dataset_args(cli)))
    alignment = sampled_index_alignment(retriever.index, retriever.token_table, retriever.token_uids, faiss)
    bench.write_new(output_dir / "index_alignment.json", alignment)
    print("FAISS_INDEX_ALIGNMENT " + json.dumps({k: v for k, v in alignment.items() if k != "samples"}), flush=True)
    reference_queries, results, query_reports = {}, [], []

    for label in ENCODER_PATHS:
        encoder = retriever.scoring_encoder if label == "cpu_scoring_encoder" else retriever.baseline_encoder
        if label == "accelerator_gpu":
            accelerator = Accelerator()
            encoder.model = accelerator.prepare(encoder.model)
            preparation = {"path": "Accelerator().prepare(model)", "mixed_precision": accelerator.mixed_precision,
                           "device": str(accelerator.device), "processes": accelerator.num_processes}
            if accelerator.num_processes != 1 or accelerator.device.type != "cuda":
                raise RuntimeError("Diagnostic expects single-process CUDA Accelerate")
        else:
            preparation = {"path": "model.to(device)", "device": str(encoder.model.device)}
        preparation["parameter_dtype"] = str(next(encoder.model.parameters()).dtype)
        for qid in qids:
            context = torch.no_grad if label == "accelerator_gpu" else torch.inference_mode
            with context():
                meta = encoder.encode_query_with_metadata(bundle["questions"][qid], to_cpu=True, query_token_filter="full")
                candidates = candidate_modes(retriever, meta)
            summary = query_summary(meta, reference_queries.get(qid))
            query_reports.append({"qid": qid, "encoder_path": label, "preparation": preparation, **summary})
            if label == "direct_gpu":
                reference_queries[qid] = meta
            with (output_dir / f"query_{label}_{qid}.npz").open("xb") as handle:
                np.savez(handle, embeddings=meta["embeddings"].float().numpy(), token_ids=meta["token_ids"].numpy())
            for source, actual in candidates.items():
                comparison = compare_candidate_rows(bundle["online"]["baseline_references"][qid], actual)
                row = {"qid": qid, "encoder_path": label, "score_source": source, "comparison": comparison}
                results.append(row)
                bench.write_new(output_dir / f"candidates_{label}_{source}_{qid}.json", actual)
                print("FAISS_DIAGNOSTIC " + json.dumps(row), flush=True)

    totals = comparison_totals(results, qids)
    versions = {}
    for package in ("torch", "numpy", "faiss-cpu", "faiss-gpu", "transformers", "accelerate", "colpali-engine"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    report = {"status": "diagnostic_complete_not_a_runtime_result", "questions": qids,
              "source_bundle": str(bundle_path), "source_bundle_sha256": bench.replay.sha256(bundle_path),
              "query_reports": query_reports, "comparisons": results, "totals": totals,
              "index_alignment": alignment, "assets": retriever.identity, "versions": versions,
              "no_automatic_configuration_selection": True,
              "limitations": ["Four fixed warm-up questions only; no historical launch provenance is inferred",
                  "faiss_distance uses returned values unchanged, even on L2; a match would not make that rule sound similarity scoring",
                  "No runtime/QA result, full corpus mapping certificate, training, or model/index change",
                  "Common loader loads SPLADE but no sparse retrieval, graph, CAPP scoring, or reader is run"]}
    bench.write_new(output_dir / "diagnostic.json", report)
    print("FAISS_DIAGNOSTIC_TOTALS " + json.dumps(totals), flush=True)
    print("saved_diagnostic=" + str(output_dir / "diagnostic.json"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.bundle.is_file():
        raise FileNotFoundError(args.bundle)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        run(args.bundle, args.output_dir)
    except Exception as exc:
        bench.write_new(args.output_dir / "failure.json", {
            "status": "diagnostic_failed_not_a_runtime_result", "type": type(exc).__name__, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
