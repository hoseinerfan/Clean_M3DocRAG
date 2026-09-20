#!/usr/bin/env python3
"""Fail-closed, batch-one online retrieval-to-answer timing; no training.

Saved predictions are validation references ONLY. Actual graph inputs are
generated from the question, resident corpus embeddings/indices and encoders.
Unknown historical launch choices are explicit reconstruction assumptions, not
claimed provenance. A mismatch prevents a publishable timing report.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import benchmark_racs_graph_reader as bench

ONLINE_SCOPE = "online batch-one query-to-answer including dense/SPLADE retrieval, graph ranking and four-page reader"
ONLINE_EXCLUDED = ["offline corpus embedding/index construction and page-text extraction",
                   "one-time resident index/model/input loading (reported separately)",
                   "validation and report serialization"]
FAISS_PAGE_SCORE_SOURCE = "faiss_distance"


class ReplayMismatch(ValueError):
    def __init__(self, details):
        self.details = details
        super().__init__(json.dumps(details))


def require_match(label, qid, expected, actual):
    result = bench.replay.compare_rows(expected, actual, 1000)
    if not (result["complete_order_matches"] and result["scores_close"]):
        first = next((i for i, (a, b) in enumerate(zip(expected, actual))
                      if list(a[:2]) != list(b[:2])), None)
        raise ReplayMismatch({"stage": label, "qid": qid, "comparison": result,
                              "first_order_mismatch_rank": None if first is None else first + 1,
                              "expected_at_mismatch": None if first is None else expected[first],
                              "actual_at_mismatch": None if first is None else actual[first]})


def scoring_options(record, approximate=False):
    """Translate the saved exact run; legacy differences are explicit below."""
    options = {key.replace("approx_base_page_token_", "approx_page_token_"): value
               for key, value in record.items() if key.startswith("approx_base_page_token_")}
    options["coarse_score_dtype"] = options.pop("approx_page_token_coarse_dtype")
    options.update(base_score_source="approx_page_maxsim_topk" if approximate else record["base_score_source"],
                   page_batch_size=0 if approximate else record["base_only_page_batch_size"],
                   report_pruning_diagnostics=False, learned_token_selector_model=None)
    options["approx_page_token_topk"] = 224 if approximate else 0
    return options


def check_exact_configuration(record):
    required = {"base_score_source": "exact_page_maxsim", "from_baseline_top_pages": 1000,
                "query_token_filter": "full", "base_only_page_batch_size": 64,
                "approx_base_page_token_selector": "global_topk",
                "approx_base_page_token_scorer": "query_mean",
                "approx_base_page_token_adaptive_k_mode": "disabled",
                "approx_base_page_token_nonspatial_policy": "keep",
                "approx_base_page_token_coarse_dtype": "fp32",
                "two_stage_exact_top_pages": 0, "two_stage_exact_top_docs": 0,
                "visual_rerank_top_pages": 0, "visual_rerank_top_docs": 0,
                "query_route_config_json": None, "learned_doc_reranker_model": None,
                "learned_token_selector_model": None, "vlm_rerank_top_docs": 0,
                "fixed_weights": {"base": 1.0, "visual": 0.0, "non_visual": 0.0, "balance": 0.0}}
    if any(key not in record or record[key] != value for key, value in required.items()):
        raise ValueError("Unsupported or incomplete Exact MaxSim configuration")


def validate_splade_directories(model_dir, tokenizer_dir=None):
    """Validate explicit local files without importing torch or accessing the Hub."""
    if model_dir is None:
        raise ValueError("Supply --splade-model-dir with the existing local SPLADE checkpoint directory; Hub/cache fallback is disabled")
    model_dir = Path(model_dir).expanduser().resolve()
    tokenizer_dir = Path(tokenizer_dir).expanduser().resolve() if tokenizer_dir else model_dir
    for label, folder in (("model", model_dir), ("tokenizer", tokenizer_dir)):
        if not folder.is_dir():
            raise FileNotFoundError(f"Local SPLADE {label} directory does not exist: {folder}")
    def nonempty(path):
        return path.is_file() and path.stat().st_size > 0
    if not nonempty(model_dir / "config.json"):
        raise FileNotFoundError(f"Local SPLADE model requires config.json: {model_dir}")
    config = bench.replay.read_json(model_dir / "config.json")
    if config.get("model_type") not in (None, "bert"):
        raise ValueError("Expected a BERT-based SPLADE checkpoint, not a replacement architecture")
    weight_files = [p for name in ("model.safetensors", "pytorch_model.bin")
                    if nonempty(p := model_dir / name)]
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        path = model_dir / name
        if not path.is_file():
            continue
        mapping = bench.replay.read_json(path).get("weight_map", {})
        if not mapping or any(not isinstance(value, str) or Path(value).is_absolute()
                              or ".." in Path(value).parts for value in mapping.values()):
            raise ValueError(f"Invalid local weight shard manifest: {path}")
        for shard in sorted(set(mapping.values())):
            if not nonempty(model_dir / shard):
                raise FileNotFoundError(f"Missing local SPLADE weight shard: {model_dir / shard}")
            weight_files.append(model_dir / shard)
    if not weight_files:
        raise FileNotFoundError(f"No local SPLADE model weights found: {model_dir}")
    token_files = [p for name in ("vocab.txt", "tokenizer.json")
                   if nonempty(p := tokenizer_dir / name)]
    if not token_files:
        raise FileNotFoundError(f"Local SPLADE tokenizer requires vocab.txt or tokenizer.json: {tokenizer_dir}")
    if not any(nonempty(tokenizer_dir / name) for name in ("config.json", "tokenizer_config.json")):
        raise FileNotFoundError(f"Local SPLADE tokenizer needs config.json or tokenizer_config.json: {tokenizer_dir}")
    small_files = set(token_files)
    for folder in (model_dir, tokenizer_dir):
        small_files.update(folder.glob("*.json"))
    return {"splade_model_dir": str(model_dir), "splade_tokenizer_dir": str(tokenizer_dir),
            "splade_local_identity": {
                "configuration_and_tokenizer_sha256": {str(p): bench.replay.sha256(p) for p in sorted(small_files)},
                "weight_files": [file_stamp(p) for p in sorted(set(weight_files))],
                "weight_bytes_hashed": False}}


def prepare_online(audit_path, replay_path, run_dir, count, warmup, splade_model_dir=None, splade_tokenizer_dir=None):
    local_splade = validate_splade_directories(splade_model_dir, splade_tokenizer_dir)
    paths, manifest = bench.prepare(audit_path, replay_path, run_dir, count, warmup)
    audit = bench.replay.read_json(audit_path)
    source = {item["prediction"]["path"]: item for item in audit["upstream"]}
    bundle = bench.replay.read_json(paths["CAPP"])
    exact_path = bundle["graphs"][bench.MAIN]["inputs"]["dense"]
    sparse_path = bundle["graphs"][bench.MAIN]["inputs"]["sparse"]
    legacy_paths = {bundle["graphs"][label]["inputs"]["dense"] for label in bench.AUXILIARIES}
    if len(legacy_paths) != 1 or exact_path in legacy_paths:
        raise ValueError("Expected separate exact and shared legacy dense inputs")
    legacy_path = legacy_paths.pop()
    for entry in bundle["graphs"].values():
        if entry["inputs"]["sparse"] != sparse_path:
            raise ValueError("Expected one shared sparse input")
    records = {}
    for label, path in (("exact", exact_path), ("sparse", sparse_path)):
        summary_path = Path(source[path]["summary"]["path"])
        record = bench.replay.read_json(summary_path)
        # The audit elides long lists; compare the scalar configuration below,
        # and fingerprint the complete current summary for this new experiment.
        for key, value in source[path]["summary"]["record"].items():
            if not isinstance(value, (dict, list)) and record.get(key) != value:
                raise ValueError(f"Summary changed since audit: {label}/{key}")
        manifest["input_sha256"][str(summary_path)] = bench.replay.sha256(summary_path)
        records[label] = record
    exact, sparse = records["exact"], records["sparse"]
    check_exact_configuration(exact)
    if (sparse["model_name_or_path"] != "naver/splade-cocondenser-ensembledistil"
            or sparse["top_pages"] != 1000 or sparse["query_topk_terms"] != 32
            or sparse["query_min_weight"] != 0.0):
        raise ValueError("Unexpected saved SPLADE configuration")
    baseline_path = Path(exact["baseline_pred"])
    baseline = bench.replay.prediction_rows(bench.replay.read_json(baseline_path))
    if bench.replay.qid_digest(baseline) != audit["gold"]["qid_sha256"]:
        raise ValueError("FAISS baseline cohort mismatch")
    manifest["input_sha256"][str(baseline_path)] = bench.replay.sha256(baseline_path)
    config = {"roles": {"exact": exact_path, "sparse": sparse_path, "legacy": legacy_path},
              "embedding_name": exact["embedding_name"], "exact_options": scoring_options(exact),
              "legacy_options": scoring_options(exact, approximate=True),
              "splade_index": sparse["index_pt"], "splade_model": sparse["model_name_or_path"],
              "splade_max_length": 64, "nprobe": 4,
              "faiss_page_score_source": FAISS_PAGE_SCORE_SOURCE,
              "faiss_score_source_evidence": {
                  "diagnostic_job": "15915127", "matched_questions": 4,
                  "matched_pages_each": 1000, "maximum_score_difference": 0.0,
                  "scope": "four preselected warm-ups, not a recovered historical command or full validation"},
              "backbone": "colpaligemma-3b-pt-448-base", "adapter": "colpali-v1.2",
              "baseline_query_device": "cuda", "scoring_query_device": "cpu",
              "query_filter": "full", "ignore_pad_scores": False, **local_splade}
    # Index hashing is once, outside the timed workers. Corpus/checkpoint file
    # manifests below are stat/config fingerprints, explicitly not weight hashes.
    manifest["input_sha256"][config["splade_index"]] = bench.replay.sha256(config["splade_index"])
    assumptions = [
        "Original complete upstream launch commands are not recovered; require output equivalence instead of assuming defaults prove history",
        "FAISS IVFFlat nprobe=4, serialized search metric preserved, raw returned distances aggregated by the existing max/sum/descending rule; full query tokens and PAD scores retained",
        "Job 15915127 reproduced all 1000 candidate pages/orders/scores on four warm-ups with GPU queries and faiss_distance, not embedding-dot aggregation; full replay still required",
        "The saved IVF metric is L2: descending distance aggregation is a historical candidate-generation behavior, not a sound similarity rule or the later Exact MaxSim score; no sign/metric conversion is made",
        "Backbone/adapter are explicit repository-wrapper choices; historical full launch command remains unrecovered",
        "Baseline query encoder on GPU, scoring query encoder on CPU, matching the current baseline/visual-rerank entry points; two resident encoder replicas",
        "Exact MaxSim page batch=64; legacy approximate query_mean/global_topk=224, unbatched fp32; inactive options inherited from exact run, diagnostic-only computations omitted",
        "SPLADE transformers backend, max_length=64, one question per query; historical summary does not record max_length or batch size",
        "SPLADE loads only explicitly supplied local model/tokenizer directories; recorded model ID stays separate, with full output replay required",
        "Common FAISS pool and CPU scoring query embedding reused within a question for exact/legacy paths only if both saved outputs reproduce",
        "Full corpus embeddings and indices resident between queries; no result, query embedding, or PDF-image cache across questions",
    ]
    online_paths = {}
    for method, path in paths.items():
        current = bench.replay.read_json(path)
        current.update(scope=ONLINE_SCOPE, online={**config,
            "baseline_references": {qid: baseline[qid]["page_retrieval_results"] for qid in current["questions"]}})
        online_paths[method] = run_dir / f"{method.lower()}.online.bundle.json"
        bench.write_new(online_paths[method], current)
    manifest.update(scope=ONLINE_SCOPE, excluded=ONLINE_EXCLUDED, online_retrieval=True,
                    reconstruction_assumptions=assumptions,
                    memory_scope="fresh worker with full resident dense/sparse corpus indices, two ColPali encoder replicas, SPLADE and Qwen; includes initialization and validation references",
                    timer_boundary="GPU sync; query encoders, FAISS candidate search, exact and required legacy MaxSim, SPLADE search, graph/CAPP, PDF images, Qwen; GPU sync",
                    validation_policy="fresh FAISS, exact, legacy (CAPP only), sparse and graph complete orders/scores plus full CAPP order; no cached fallback; any mismatch blocks aggregate")
    return online_paths, manifest


def file_stamp(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def validate_faiss_page_score_source(source):
    if source not in ("embedding", "faiss_distance"):
        raise ValueError("Explicit faiss_page_score_source must be embedding or faiss_distance; no fallback")
    return source


def configure_faiss_index(index, faiss_module, nprobe, page_score_source):
    """Preserve the serialized metric and record the explicit aggregation path.

    The original builder passes an IP quantizer to IndexIVFFlat but does not
    pass the IVF metric argument (whose default is L2). Quantizer/search/page
    scoring metrics must not be conflated or 'repaired' during reproduction.
    Complete candidate-order and score checks still decide equivalence.
    """
    source = validate_faiss_page_score_source(page_score_source)
    if not hasattr(index, "nprobe"):
        raise ValueError("Expected an IVF FAISS index with nprobe")
    if isinstance(nprobe, bool) or int(nprobe) != nprobe or nprobe < 1:
        raise ValueError("FAISS nprobe must be a positive integer")
    names = {int(faiss_module.METRIC_INNER_PRODUCT): "inner_product",
             int(faiss_module.METRIC_L2): "l2"}
    metric = int(index.metric_type)
    if metric not in names:
        raise ValueError(f"Unsupported saved FAISS search metric: {metric}; expected L2 or inner product, without conversion")
    quantizer = getattr(index, "quantizer", None)
    quantizer_metric = getattr(quantizer, "metric_type", None)
    if quantizer_metric is not None:
        quantizer_metric = int(quantizer_metric)
    saved_nprobe = int(index.nprobe)
    index.nprobe = int(nprobe)  # Same per-run search parameter as the baseline.
    return {"index_class": type(index).__name__, "metric_type": metric,
            "metric_name": names[metric], "quantizer_metric_type": quantizer_metric,
            "quantizer_metric_name": names.get(quantizer_metric, "unknown"),
            "saved_nprobe": saved_nprobe, "nprobe": int(index.nprobe),
            "page_score_source": source,
            "page_score_semantics": "embedding_dot_product" if source == "embedding" else names[metric],
            "page_aggregation": "max_per_page_per_query_token_then_sum_descending",
            "l2_descending_aggregation_caveat": source == "faiss_distance" and names[metric] == "l2"}


class OnlineRetriever:
    def __init__(self, bundle, dataset):
        import faiss
        import numpy as np
        import torch
        from m3docrag.utils.paths import LOCAL_EMBEDDINGS_DIR, LOCAL_MODEL_DIR
        from m3docrag.retrieval.colpali import ColPaliRetrievalModel
        from m3docrag.rag.base import RAGModelBase
        from splade_encoder_backend import SpladeTextEncoder

        self.torch, self.bundle, self.config = torch, bundle, bundle["online"]
        if torch.cuda.get_device_properties(0).total_memory < 40 * 2**30:
            raise RuntimeError("Online benchmark requires a GPU with at least 40 GiB; do not quantize/substitute the fixed reader")
        self.baseline_rows = None
        config = self.config
        page_score_source = validate_faiss_page_score_source(config.get("faiss_page_score_source"))
        # Fail on missing/changed assets or tokenizer load before the FAISS index,
        # full corpus embeddings, ColPali replicas and graph/reader timing work.
        local_splade = validate_splade_directories(config.get("splade_model_dir"), config.get("splade_tokenizer_dir"))
        if local_splade["splade_local_identity"] != config.get("splade_local_identity"):
            raise ValueError("Local SPLADE assets changed after input preparation")
        self.sparse_encoder = SpladeTextEncoder(
            config["splade_model_dir"], "transformers", torch.device("cuda"), config["splade_max_length"],
            tokenizer_name_or_path=config["splade_tokenizer_dir"], local_files_only=True)
        if len(self.sparse_encoder.tokenizer) != int(self.sparse_encoder.model.config.vocab_size):
            raise ValueError("Local SPLADE tokenizer vocabulary size differs from model output vocabulary")
        print("ONLINE_LOCAL_SPLADE " + json.dumps({"recorded_model_id": config["splade_model"],
              "model_dir": config["splade_model_dir"], "tokenizer_dir": config["splade_tokenizer_dir"]}), flush=True)
        faiss.omp_set_num_threads(1)
        embedding_dir = Path(LOCAL_EMBEDDINGS_DIR) / config["embedding_name"]
        index_path = Path(LOCAL_EMBEDDINGS_DIR) / (config["embedding_name"] + "_pageindex_ivfflat") / "index.bin"
        self.index = faiss.read_index(str(index_path))
        faiss_configuration = configure_faiss_index(self.index, faiss, config["nprobe"], page_score_source)
        print("ONLINE_FAISS_INDEX " + json.dumps({"path": str(index_path), **faiss_configuration}), flush=True)
        dataset.args.embedding_name = config["embedding_name"]
        dataset.args.retrieval_model_type = "colpali"
        self.embeddings = dataset.load_all_embeddings()
        if list(self.embeddings) != list(dataset.all_supporting_doc_ids):
            raise ValueError("Corpus document order changed")
        token_count = sum(page.numel() // page.shape[-1] for doc in self.embeddings.values() for page in doc)
        dim = next(iter(self.embeddings.values())).shape[-1]
        if self.index.ntotal != token_count or self.index.d != dim:
            raise ValueError("FAISS index and ordered corpus token table disagree")
        # Same float32 values/order as the baseline's cat(...).float().numpy(),
        # without an additional full-corpus concatenation temporary.
        self.token_table = np.empty((token_count, dim), dtype=np.float32)
        self.token_uids, offset = [], 0
        for doc_id, doc in self.embeddings.items():
            for page_idx, page in enumerate(doc):
                values = page.reshape(-1, dim).float().numpy()
                self.token_table[offset:offset + len(values)] = values
                self.token_uids.extend([f"{doc_id}_page{page_idx}"] * len(values))
                offset += len(values)
        backbone, adapter = (Path(LOCAL_MODEL_DIR) / config[key] for key in ("backbone", "adapter"))
        if not backbone.is_dir() or not adapter.is_dir():
            raise FileNotFoundError(f"Require existing checkpoints: {backbone}, {adapter}")
        self.baseline_encoder = ColPaliRetrievalModel(backbone_name_or_path=backbone, adapter_name_or_path=adapter)
        self.baseline_encoder.model.to(config["baseline_query_device"])
        self.scoring_encoder = ColPaliRetrievalModel(backbone_name_or_path=backbone, adapter_name_or_path=adapter)
        self.scoring_encoder.model.to(config["scoring_query_device"])
        self.rag = RAGModelBase(retrieval_model=self.baseline_encoder)
        payload = torch.load(config["splade_index"], map_location="cpu", weights_only=False)
        if (payload.get("model_name_or_path") != config["splade_model"]
                or payload.get("encoder_backend", "transformers") != "transformers"):
            raise ValueError("SPLADE index/encoder identity mismatch")
        self.sparse_doc_ids = payload["doc_ids"]
        self.sparse_page_indices = payload["page_indices"].to(torch.int64)
        self.page_count = len(payload["page_uids"])
        posting_pages, posting_weights = {}, {}
        for page in range(self.page_count):
            start, end = int(payload["offsets"][page]), int(payload["offsets"][page + 1])
            for term, weight in zip(payload["term_ids"][start:end].tolist(), payload["term_weights"][start:end].tolist()):
                posting_pages.setdefault(int(term), []).append(page)
                posting_weights.setdefault(int(term), []).append(float(weight))
        self.postings = {term: (torch.tensor(ids, dtype=torch.int64), torch.tensor(posting_weights[term], dtype=torch.float32))
                         for term, ids in posting_pages.items()}
        self.identity = {"config": {key: value for key, value in config.items() if key != "baseline_references"},
                         "faiss_index": file_stamp(index_path), "sparse_index": file_stamp(config["splade_index"]),
                         "faiss_configuration": faiss_configuration,
                         "faiss_tokens": token_count, "dense_documents": len(self.embeddings),
                         "sparse_pages": self.page_count,
                         "embedding_files": [file_stamp(embedding_dir / (doc + ".safetensors")) for doc in self.embeddings],
                         "encoder_json_sha256": {str(path): bench.replay.sha256(path)
                             for folder in (backbone, adapter) for path in folder.glob("*.json")},
                         "encoder_weight_files": [file_stamp(path) for folder in (backbone, adapter)
                             for path in sorted(folder.rglob("*")) if path.is_file() and path.suffix in (".safetensors", ".bin")],
                         "splade_model_config": self.sparse_encoder.model.config.to_dict(),
                         "checkpoint_and_embedding_bytes_hashed": False}
        del payload, posting_pages, posting_weights
        gc.collect()

    def dense_scores(self, baseline, query_meta, approximate):
        import rerank_target_docs_visual_aware as scoring
        from run_visual_rerank_batch import build_baseline_pool
        torch = self.torch
        docs, uids, ranks, _, scores = build_baseline_pool(baseline, 1000)
        embeddings = {doc: self.embeddings[doc] for doc in docs}
        specs, meta = scoring.build_page_id_metadata(embeddings, explicit_page_uids=set(uids), nonspatial_token_position="suffix")
        query = query_meta["embeddings"].float().to(device="cuda", dtype=torch.float32)
        tokens = query_meta["raw_tokens"]
        mask = scoring.make_query_score_mask(query_raw_tokens=tokens, ignore_pad_scores_in_final_ranking=False)
        state = scoring.prepare_coarse_query_state(query_emb=query, query_score_mask=mask,
            approx_page_token_scorer="query_mean", query_axis_classes=[], coarse_score_dtype="fp32") if approximate else None
        options = self.config["legacy_options" if approximate else "exact_options"]
        features = scoring.compute_base_only_page_features(page_specs=specs, docid2embs=embeddings,
            query_emb=query, query_score_mask=mask, baseline_page_score_map=scores,
            query_axis_classes=[], query_token_labels=[scoring.clean_token_label(t) for t in tokens],
            page_token_classes_by_uid=None, page_meta_by_uid=meta, prepared_query_state=state, **options)
        _, rows = scoring.build_rankings(features, scoring.WeightConfig(1.0, 0.0, 0.0, 0.0), ranks)
        return [[row["doc_id"], row["page_idx"], row["fused_page_score"]] for row in rows]

    def sparse_search(self, terms):
        torch = self.torch
        scores = torch.zeros(self.page_count, dtype=torch.float32)
        for term, weight in zip(*terms):
            if int(term) in self.postings:
                page_ids, doc_weights = self.postings[int(term)]
                scores.index_add_(0, page_ids, doc_weights * float(weight))
        positive = torch.nonzero(scores > 0, as_tuple=False).squeeze(-1)
        if not positive.numel():
            return []
        values, positions = torch.topk(scores[positive], k=min(1000, positive.numel()))
        return [[str(self.sparse_doc_ids[page]), int(self.sparse_page_indices[page]), float(value)]
                for page, value in zip(positive[positions].tolist(), values.tolist())]

    def retrieve(self, qid):
        from splade_encoder_backend import embedding_rows_to_terms
        source = validate_faiss_page_score_source(self.config.get("faiss_page_score_source"))
        # Explicit reconstruction choice, never chosen from reference rankings
        # at query time. None selects the existing raw-distance branch, without
        # negation or L2-to-IP conversion. Exact MaxSim remains a separate stage.
        candidate_token_table = self.token_table if source == "embedding" else None
        stages, rows = {}, {}
        question = self.bundle["questions"][qid]

        def measured(label, function):
            self.torch.cuda.synchronize()
            started = time.perf_counter()
            result = function()
            self.torch.cuda.synchronize()
            stages["upstream:" + label] = time.perf_counter() - started
            return result

        with self.torch.inference_mode():
            query = measured("faiss_query_encoding", lambda: self.baseline_encoder.encode_query_with_metadata(
                question, to_cpu=True, query_token_filter="full"))
            baseline = measured("faiss_search_and_page_aggregation", lambda: self.rag._retrieve_pages_from_index_query_meta(
                query, self.index, self.token_uids, candidate_token_table, 1000, False))
            self.baseline_rows = baseline  # Validation only; never reused by another query.
            score_query = measured("maxsim_query_encoding", lambda: self.scoring_encoder.encode_query_with_metadata(
                question, to_cpu=True, query_token_filter="full"))
            rows["exact"] = measured("exact_maxsim", lambda: self.dense_scores(baseline, score_query, False))
            if self.bundle["method"] == "CAPP":
                rows["legacy"] = measured("legacy_approximate_maxsim", lambda: self.dense_scores(baseline, score_query, True))
            terms = measured("splade_query_encoding_and_pruning", lambda: embedding_rows_to_terms(
                self.sparse_encoder.encode_queries([question]), topk_terms=32, min_weight=0.0)[0])
            rows["sparse"] = measured("splade_postings_search", lambda: self.sparse_search(terms))
        return {self.config["roles"][role]: {qid: {"question": question, "page_retrieval_results": values}}
                for role, values in rows.items()}, stages

    def validate(self, qid, inputs):
        require_match("faiss_candidate_pool", qid, self.config["baseline_references"][qid], self.baseline_rows)
        for role, path in self.config["roles"].items():
            if role == "legacy" and self.bundle["method"] != "CAPP":
                continue
            require_match(role, qid, self.bundle["inputs"][path][qid]["page_retrieval_results"],
                          inputs[path][qid]["page_retrieval_results"])


def preflight(bundle_path, output_path):
    import torch
    import run_m3docvqa_external_retrieval_qa as qa
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA GPU required")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    bundle = bench.replay.read_json(bundle_path)
    cli = SimpleNamespace(data_name="m3-docvqa", split="dev", bits=16, model_name_or_path="Qwen2-VL-7B-Instruct")
    retriever = OnlineRetriever(bundle, qa.M3DocVQADataset(qa.make_dataset_args(cli)))
    for qid in bundle["warmup_qids"]:
        inputs, _ = retriever.retrieve(qid)
        retriever.validate(qid, inputs)
        print("ONLINE_PREFLIGHT_QID_PASSED " + qid, flush=True)
    bench.write_new(output_path, {"status": "online_upstream_preflight_passed", "qids": bundle["warmup_qids"],
                                  "scope": "upstream only; graph/CAPP checks run in each measured worker"})


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
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--splade-model-dir", type=Path, help="Existing local SPLADE checkpoint directory; no Hub fallback")
    parser.add_argument("--splade-tokenizer-dir", type=Path, help="Existing local tokenizer directory; defaults to the model directory")
    parser.add_argument("--check-local-splade", action="store_true", help="Check local file layout only, without GPU/model loading or writing reports")
    args = parser.parse_args()
    if args.check_local_splade:
        print(json.dumps(validate_splade_directories(args.splade_model_dir, args.splade_tokenizer_dir), indent=2))
        print("LOCAL_SPLADE_FILES_PRESENT_NOT_YET_REPLAY_VALIDATED")
        return
    if args.worker_bundle:
        if args.worker_output is None or (not args.preflight_only and args.pass_index is None):
            parser.error("Worker output and pass index required")
        try:
            if args.preflight_only:
                preflight(args.worker_bundle, args.worker_output)
            else:
                bench.worker(args.worker_bundle, args.worker_output, args.pass_index)
        except Exception as exc:
            bench.write_new(args.worker_output.with_suffix(".failure.json"),
                {"status": "failed_not_a_runtime_result", "error": str(exc),
                 "type": type(exc).__name__, "details": getattr(exc, "details", None)})
            raise
        return
    if args.run_dir is None or args.audit_json is None or args.replay_json is None:
        parser.error("Require run-dir, audit-json and replay-json")
    order = bench.schedule(args.repeats)
    args.run_dir.mkdir()
    try:
        bundles, manifest = prepare_online(args.audit_json, args.replay_json, args.run_dir, args.questions, args.warmup_questions,
                                          args.splade_model_dir, args.splade_tokenizer_dir)
        manifest["schedule"] = order
        code_paths = [Path(__file__), Path(bench.__file__), Path(bench.capp.__file__), Path(bench.capp.ca.__file__),
                      Path(bench.replay.__file__), Path(bench.replay.graph.__file__)]
        code_paths += [Path(p) for p in ("scripts/run_visual_rerank_batch.py", "scripts/rerank_target_docs_visual_aware.py",
            "scripts/splade_encoder_backend.py", "scripts/run_splade_page_retrieval.py",
            "scripts/run_m3docvqa_external_retrieval_qa.py", "src/m3docrag/rag/base.py",
            "src/m3docrag/retrieval/colpali.py", "src/m3docrag/vqa/qwen2.py")]
        manifest["code_sha256"] = {str(p): bench.replay.sha256(p) for p in code_paths}
        bench.write_new(args.run_dir / "manifest.json", manifest)
        gc.collect()
        subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--worker-bundle", str(bundles["CAPP"]),
                        "--worker-output", str(args.run_dir / "preflight.json"), "--preflight-only"], check=True)
        reports = []
        for index, method in order:
            path = args.run_dir / f"pass{index}_{method.lower()}.json"
            subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--worker-bundle", str(bundles[method]),
                            "--worker-output", str(path), "--pass-index", str(index)], check=True)
            reports.append(bench.replay.read_json(path))
        result = bench.aggregate(reports, manifest, args.repeats)
        bench.write_new(args.run_dir / "runtime.json", result)
        for method in ("GPP", "CAPP"):
            print("ONLINE_RUNTIME_RESULT " + json.dumps({"method": method,
                  **result["methods"][method]["per_question_mean_across_repeats"]["total"]}), flush=True)
        print("saved_online_runtime_report=" + str(args.run_dir / "runtime.json"), flush=True)
    except Exception as exc:
        bench.write_new(args.run_dir / "failure.json", {"status": "failed_not_a_runtime_result", "type": type(exc).__name__, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
