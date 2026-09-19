import contextlib
import copy
import inspect
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_racs_online as online
import benchmark_racs_graph_reader as bench
import rerank_target_docs_visual_aware as scoring
import test_benchmark_racs_graph_reader as fixtures


def exact_config():
    return {"base_score_source": "exact_page_maxsim", "from_baseline_top_pages": 1000,
        "query_token_filter": "full", "base_only_page_batch_size": 64,
        "approx_base_page_token_selector": "global_topk", "approx_base_page_token_scorer": "query_mean",
        "approx_base_page_token_adaptive_k_mode": "disabled", "approx_base_page_token_nonspatial_policy": "keep",
        "approx_base_page_token_coarse_dtype": "fp32", "two_stage_exact_top_pages": 0,
        "two_stage_exact_top_docs": 0, "visual_rerank_top_pages": 0, "visual_rerank_top_docs": 0,
        "query_route_config_json": None, "learned_doc_reranker_model": None,
        "learned_token_selector_model": None, "vlm_rerank_top_docs": 0,
        "fixed_weights": {"base": 1.0, "visual": 0.0, "non_visual": 0.0, "balance": 0.0}}


def local_splade_fixture(root):
    folder = root / "splade"
    folder.mkdir()
    (folder / "config.json").write_text('{"model_type":"bert"}')
    (folder / "pytorch_model.bin").write_bytes(b"test-layout-only-not-real-weights")
    (folder / "vocab.txt").write_text("[PAD]\n[UNK]\n")
    return folder


class OnlineBenchmarkTests(unittest.TestCase):
    def test_local_splade_paths_are_explicit_and_same_folder_by_default(self):
        with self.assertRaisesRegex(ValueError, "Hub/cache fallback is disabled"):
            online.validate_splade_directories(None)
        with tempfile.TemporaryDirectory() as directory:
            folder = local_splade_fixture(Path(directory))
            result = online.validate_splade_directories(folder)
            self.assertEqual(result["splade_model_dir"], str(folder.resolve()))
            self.assertEqual(result["splade_tokenizer_dir"], str(folder.resolve()))
            self.assertIn(str((folder / "vocab.txt").resolve()), result["splade_local_identity"]["configuration_and_tokenizer_sha256"])
            self.assertEqual(len(result["splade_local_identity"]["weight_files"]), 1)

    def test_separate_local_tokenizer_files_are_fingerprinted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = local_splade_fixture(root)
            tokenizer = root / "tokenizer"
            tokenizer.mkdir()
            (tokenizer / "tokenizer_config.json").write_text('{"tokenizer_class":"BertTokenizer"}')
            (tokenizer / "tokenizer.json").write_text('{"test":"layout-only"}')
            before = online.validate_splade_directories(folder, tokenizer)
            self.assertEqual(before["splade_tokenizer_dir"], str(tokenizer.resolve()))
            (tokenizer / "tokenizer.json").write_text('{"test":"changed"}')
            self.assertNotEqual(before["splade_local_identity"], online.validate_splade_directories(folder, tokenizer)["splade_local_identity"])

    def test_missing_local_assets_fail_without_hub_fallback(self):
        for missing in ("config.json", "vocab.txt", "pytorch_model.bin"):
            with tempfile.TemporaryDirectory() as directory:
                folder = local_splade_fixture(Path(directory))
                (folder / missing).unlink()
                with self.assertRaises(FileNotFoundError):
                    online.validate_splade_directories(folder)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                online.validate_splade_directories(Path(directory) / "does-not-exist")

    def test_sharded_weights_require_all_local_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = local_splade_fixture(Path(directory))
            (folder / "pytorch_model.bin").unlink()
            (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"first": "part1.safetensors", "second": "part2.safetensors"}}))
            (folder / "part1.safetensors").write_bytes(b"part1")
            with self.assertRaisesRegex(FileNotFoundError, "part2.safetensors"):
                online.validate_splade_directories(folder)
            (folder / "part2.safetensors").write_bytes(b"part2")
            result = online.validate_splade_directories(folder)
            self.assertEqual(len(result["splade_local_identity"]["weight_files"]), 2)

    def test_local_check_does_not_prepare_retrieval_or_write_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = local_splade_fixture(root)
            before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            with mock.patch.object(sys, "argv", ["check", "--check-local-splade", "--splade-model-dir", str(folder)]), \
                 mock.patch.object(online, "prepare_online") as prepare, \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                online.main()
            prepare.assert_not_called()
            self.assertIn("LOCAL_SPLADE_FILES_PRESENT_NOT_YET_REPLAY_VALIDATED", output.getvalue())
            self.assertEqual(before, {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()})

    def test_l2_ivf_with_ip_quantizer_keeps_both_metrics(self):
        faiss = SimpleNamespace(METRIC_INNER_PRODUCT=0, METRIC_L2=1)
        quantizer = SimpleNamespace(metric_type=faiss.METRIC_INNER_PRODUCT)
        index = SimpleNamespace(metric_type=faiss.METRIC_L2, quantizer=quantizer, nprobe=1)
        actual = online.configure_faiss_index(index, faiss, 4)
        self.assertEqual(index.metric_type, faiss.METRIC_L2)
        self.assertEqual(quantizer.metric_type, faiss.METRIC_INNER_PRODUCT)
        self.assertEqual(index.nprobe, 4)
        self.assertEqual(actual["metric_name"], "l2")
        self.assertEqual(actual["quantizer_metric_name"], "inner_product")
        self.assertEqual(actual["saved_nprobe"], 1)
        self.assertEqual(actual["page_score_source"], "embedding_dot_product_not_faiss_distance")

    def test_ip_ivf_remains_supported_without_metric_conversion(self):
        faiss = SimpleNamespace(METRIC_INNER_PRODUCT=0, METRIC_L2=1)
        index = SimpleNamespace(metric_type=faiss.METRIC_INNER_PRODUCT, nprobe=1)
        actual = online.configure_faiss_index(index, faiss, 4)
        self.assertEqual(index.metric_type, faiss.METRIC_INNER_PRODUCT)
        self.assertEqual(actual["metric_name"], "inner_product")
        self.assertIsNone(actual["quantizer_metric_type"])

    def test_unknown_metric_non_ivf_and_invalid_nprobe_still_fail(self):
        faiss = SimpleNamespace(METRIC_INNER_PRODUCT=0, METRIC_L2=1)
        unknown = SimpleNamespace(metric_type=99, nprobe=1)
        with self.assertRaisesRegex(ValueError, "Unsupported saved FAISS search metric: 99"):
            online.configure_faiss_index(unknown, faiss, 4)
        self.assertEqual(unknown.nprobe, 1)
        self.assertEqual(unknown.metric_type, 99)
        with self.assertRaisesRegex(ValueError, "Expected an IVF"):
            online.configure_faiss_index(SimpleNamespace(metric_type=1), faiss, 4)
        for nprobe in (0, -1, 4.5, True):
            index = SimpleNamespace(metric_type=1, nprobe=1)
            with self.assertRaisesRegex(ValueError, "positive integer"):
                online.configure_faiss_index(index, faiss, nprobe)
            self.assertEqual(index.nprobe, 1)

    def test_configuration_rejects_missing_or_different_historical_settings(self):
        online.check_exact_configuration(exact_config())
        for change in ({"query_token_filter": "semantic_only"}, {"fixed_weights": {}}, {"base_only_page_batch_size": 0}):
            with self.assertRaises(ValueError):
                online.check_exact_configuration({**exact_config(), **change})
        record = exact_config()
        del record["query_route_config_json"]
        with self.assertRaises(ValueError):
            online.check_exact_configuration(record)

    def test_scoring_options_use_real_signature_and_distinct_paths(self):
        record = exact_config()
        before = copy.deepcopy(record)
        exact = online.scoring_options(record)
        legacy = online.scoring_options(record, approximate=True)
        self.assertEqual(record, before)
        self.assertEqual(exact["page_batch_size"], 64)
        self.assertEqual(legacy["page_batch_size"], 0)
        self.assertEqual(legacy["approx_page_token_topk"], 224)
        self.assertEqual(legacy["base_score_source"], "approx_page_maxsim_topk")
        signature = inspect.signature(scoring.compute_base_only_page_features)
        self.assertFalse(set(legacy) - set(signature.parameters))

    def test_strict_upstream_validation_checks_order_scores_and_count(self):
        rows = [["doc", i, 1000.0 - i] for i in range(1000)]
        online.require_match("exact", "q", rows, [tuple(r) for r in rows])
        with self.assertRaises(online.ReplayMismatch) as raised:
            online.require_match("legacy", "q", rows, rows[::-1])
        self.assertEqual(raised.exception.details["first_order_mismatch_rank"], 1)
        bad = copy.deepcopy(rows)
        bad[0][2] += 2
        with self.assertRaises(online.ReplayMismatch):
            online.require_match("exact", "q", rows, bad)
        with self.assertRaises(ValueError):
            online.require_match("sparse", "q", rows, rows[:-1])

    def test_graph_consumes_online_rows_and_never_falls_back(self):
        bundle, rows, _ = fixtures.GraphReaderBenchmarkTests().bundle("GPP")
        live = {key: {"q": {"question": "live", "page_retrieval_results": rows[::-1]}}
                for key in ("dense", "sparse")}
        with mock.patch.object(bench.replay.graph, "build_qid_graph_ranking", return_value=(rows, {})) as graph:
            bench.rank_query(bundle, "q", {"pages": None}, {}, None, None, live)
            self.assertIs(graph.call_args.kwargs["dense_row"], live["dense"]["q"])
            self.assertEqual(graph.call_args.kwargs["dense_row"]["question"], "live")
            with self.assertRaises(KeyError):
                bench.rank_query(bundle, "q", {"pages": None}, {}, None, None, {})

    def retriever_stub(self, method):
        obj = object.__new__(online.OnlineRetriever)
        obj.bundle = {"method": method, "questions": {"q": "Question?"}}
        obj.config = {"roles": {"exact": "E", "legacy": "L", "sparse": "S"}}
        obj.torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None), inference_mode=contextlib.nullcontext)
        obj.baseline_encoder = SimpleNamespace(encode_query_with_metadata=mock.Mock(return_value="query-gpu"))
        obj.scoring_encoder = SimpleNamespace(encode_query_with_metadata=mock.Mock(return_value="query-cpu"))
        obj.rag = SimpleNamespace(_retrieve_pages_from_index_query_meta=mock.Mock(return_value=[["fresh", 0, 1.0]]))
        obj.index, obj.token_uids, obj.token_table = object(), [], object()
        obj.sparse_encoder = SimpleNamespace(encode_queries=mock.Mock(return_value="pooled"))
        obj.sparse_search = mock.Mock(return_value=[["sparse", 1, .5]])
        obj.dense_scores = mock.Mock(side_effect=lambda pool, query, approx: [["approx" if approx else "exact", 2, .7]])
        return obj

    def test_online_each_query_recomputes_and_capp_pays_for_legacy(self):
        encoder_module = SimpleNamespace(embedding_rows_to_terms=mock.Mock(return_value=[([1], [2.0])]))
        with mock.patch.dict(sys.modules, {"splade_encoder_backend": encoder_module}):
            for method, expected in (("GPP", {"E", "S"}), ("CAPP", {"E", "L", "S"})):
                obj = self.retriever_stub(method)
                first, stages = obj.retrieve("q")
                second, _ = obj.retrieve("q")
                self.assertEqual(set(first), expected)
                self.assertEqual(first, second)
                self.assertEqual(obj.baseline_encoder.encode_query_with_metadata.call_count, 2)
                self.assertEqual(obj.scoring_encoder.encode_query_with_metadata.call_count, 2)
                self.assertEqual(obj.rag._retrieve_pages_from_index_query_meta.call_count, 2)
                self.assertEqual(obj.dense_scores.call_count, 4 if method == "CAPP" else 2)
                self.assertEqual("upstream:legacy_approximate_maxsim" in stages, method == "CAPP")
                self.assertTrue(all(value >= 0 for value in stages.values()))

    def test_real_scoring_helpers_called_with_compatible_parameters(self):
        # Autospec checks the large scoring API without loading torch/models.
        obj = self.retriever_stub("CAPP")
        del obj.dense_scores  # Restore the actual method.
        record = exact_config()
        expected = inspect.signature(scoring.compute_base_only_page_features).parameters
        options = online.scoring_options(record, approximate=True)
        supplied = {"page_specs", "docid2embs", "query_emb", "query_score_mask", "baseline_page_score_map",
                    "query_axis_classes", "query_token_labels", "page_token_classes_by_uid", "page_meta_by_uid", "prepared_query_state"}
        for key, parameter in expected.items():
            if key not in supplied and parameter.default is inspect.Parameter.empty:
                options.setdefault(key, 1)
        obj.config["legacy_options"] = options
        obj.torch.float32 = "float32"
        obj.embeddings = {"doc": object()}
        tensor = mock.Mock()
        tensor.float.return_value.to.return_value = "gpu-query"
        with mock.patch.object(scoring, "build_page_id_metadata", return_value=([("doc", 0)], {})), \
             mock.patch.object(scoring, "make_query_score_mask", return_value="mask"), \
             mock.patch.object(scoring, "prepare_coarse_query_state", autospec=True, return_value={}) as coarse, \
             mock.patch.object(scoring, "compute_base_only_page_features", autospec=True, return_value=[]) as compute, \
             mock.patch.object(scoring, "build_rankings", return_value=([], [{"doc_id": "doc", "page_idx": 0, "fused_page_score": 1.0}])):
            rows = obj.dense_scores([["doc", 0, 1.0]], {"embeddings": tensor, "raw_tokens": ["token"]}, True)
        self.assertEqual(rows, [["doc", 0, 1.0]])
        self.assertEqual(coarse.call_args.kwargs["approx_page_token_scorer"], "query_mean")
        self.assertEqual(compute.call_args.kwargs["page_batch_size"], 0)

    def test_online_aggregation_requires_upstream_checks_identity_and_partition(self):
        reports, manifest = fixtures.GraphReaderBenchmarkTests().reports()
        manifest.update(online_retrieval=True, scope=online.ONLINE_SCOPE)
        for report in reports:
            report.update(status="validated_online_worker", scope=online.ONLINE_SCOPE, upstream={"same": True})
            for row in report["results"]:
                value = row["seconds"]["total"]
                row.update(upstream_checks_passed=True)
                row["seconds"] = {"total": value, "ranking_total": value / 2,
                    "online_retrieval_total": value / 4, "graph_and_capp_total": value / 4,
                    "image_preparation": value / 4, "reader_prompt_preprocess_generate": value / 4}
        result = bench.aggregate(reports, manifest, 2)
        self.assertEqual(result["status"], "validated_online_query_to_answer")
        for mutation in (lambda r: r[0]["results"][0].update(upstream_checks_passed=False),
                         lambda r: r[1].update(upstream={"different": True}),
                         lambda r: r[0]["results"][0]["seconds"].update(ranking_total=99)):
            changed = copy.deepcopy(reports)
            mutation(changed)
            with self.assertRaises(ValueError):
                bench.aggregate(changed, manifest, 2)

    def test_online_worker_wraps_live_retrieval_in_timer_and_checks_after_generation(self):
        bundle, rows, order = fixtures.GraphReaderBenchmarkTests().bundle("GPP")
        bundle.update(online={}, measured_qids=["q"], warmup_qids=["w"])
        bundle["questions"]["w"] = "Warm?"
        events = []
        live = {"live": True}
        def retrieve(qid):
            events.append("retrieve:" + qid)
            return live, {"upstream:test": .0001}
        upstream = SimpleNamespace(identity={"same": True}, retrieve=retrieve,
            validate=lambda qid, inputs: events.append("validate-upstream:" + qid))
        torch = SimpleNamespace(set_num_threads=lambda _: None, manual_seed=lambda _: None,
            get_num_threads=lambda: 1, no_grad=contextlib.nullcontext,
            cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                synchronize=lambda: events.append("sync"), reset_peak_memory_stats=lambda: None,
                get_device_properties=lambda _: SimpleNamespace(name="fake", total_memory=1),
                max_memory_allocated=lambda: 1, max_memory_reserved=lambda: 2))
        network = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(dtype="bfloat16")]))
        reader = SimpleNamespace(model=network, processor=SimpleNamespace(image_processor=object()),
            generate=lambda **_: events.append("generate") or "answer")
        def rank(*args):
            self.assertIs(args[-1], live)
            events.append("rank")
            return rows[:4], order, {}, {"graph:test": .0001}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, output = root / "bundle.json", root / "worker.json"
            bench.write_new(path, bundle)
            qa = SimpleNamespace(make_dataset_args=lambda cli: cli,
                M3DocVQADataset=lambda _: SimpleNamespace(get_images_from_doc_id=lambda _: list(range(1000))),
                resolve_model_path=lambda _: root, supports_flash_attention=lambda: False,
                VQAModel=lambda **_: reader, short_answer_template=SimpleNamespace(substitute=lambda x: x["question"]))
            modules = {"torch": torch, "run_m3docvqa_external_retrieval_qa": qa,
                       "accelerate": SimpleNamespace(Accelerator=lambda: SimpleNamespace(num_processes=1, prepare=lambda x: x))}
            with mock.patch.dict(sys.modules, modules), \
                 mock.patch.object(online, "OnlineRetriever", return_value=upstream), \
                 mock.patch.object(bench.replay.graph, "load_doc_page_catalog", return_value=None), \
                 mock.patch.object(bench, "rank_query", side_effect=rank), \
                 mock.patch.object(bench, "validate_rankings", side_effect=lambda *a: events.append("validate-graph")), \
                 mock.patch.object(bench.capp, "command_output", return_value="fake"), \
                 contextlib.redirect_stdout(io.StringIO()):
                bench.worker(path, output, 1)
            result = json.loads(output.read_text())
        self.assertEqual(result["status"], "validated_online_worker")
        self.assertEqual(result["scope"], online.ONLINE_SCOPE)
        self.assertEqual(events.count("rank"), 2)
        self.assertLess(events.index("retrieve:q"), events.index("validate-upstream:q"))
        self.assertEqual(events[events.index("validate-upstream:q") - 2:events.index("validate-upstream:q")], ["generate", "sync"])
        row = result["results"][0]
        self.assertTrue(row["upstream_checks_passed"])
        seconds = row["seconds"]
        self.assertAlmostEqual(seconds["ranking_total"], seconds["online_retrieval_total"] + seconds["graph_and_capp_total"])

    def test_prepare_preserves_artifacts_and_sanitizes_baseline_references(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [["doc", i, 1000.0 - i] for i in range(1000)]
            baseline = root / "baseline.json"
            bench.write_new(baseline, {qid: {"page_retrieval_results": rows, "answers": ["secret"]} for qid in ("q", "w")})
            exact = {**exact_config(), "baseline_pred": str(baseline), "embedding_name": "colpali-test"}
            sparse_index = root / "index.pt"
            sparse_index.write_bytes(b"test-only")
            sparse = {"index_pt": str(sparse_index), "model_name_or_path": "naver/splade-cocondenser-ensembledistil",
                      "top_pages": 1000, "query_topk_terms": 32, "query_min_weight": 0.0}
            audit = {"gold": {"qid_sha256": bench.replay.qid_digest(["q", "w"])}, "upstream": []}
            for role, record in (("E", exact), ("S", sparse)):
                path = root / (role + ".summary.json")
                bench.write_new(path, record)
                audit["upstream"].append({"prediction": {"path": role}, "summary": {"path": str(path), "record": record}})
            audit_path = root / "audit.json"
            bench.write_new(audit_path, audit)
            paths = {}
            for method in ("GPP", "CAPP"):
                labels = [bench.MAIN] + (list(bench.AUXILIARIES) if method == "CAPP" else [])
                bundle = {"questions": {"q": "Question?", "w": "Warm?"}, "method": method,
                          "graphs": {label: {"inputs": {"dense": "E" if label == bench.MAIN else "L", "sparse": "S"}} for label in labels}}
                paths[method] = root / (method + ".bundle.json")
                bench.write_new(paths[method], bundle)
            local_model = local_splade_fixture(root)
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            manifest = {"input_sha256": {}}
            with mock.patch.object(bench, "prepare", return_value=(paths, manifest)):
                output, result = online.prepare_online(audit_path, root / "unused", root, 1, 1, local_model)
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            self.assertTrue(result["online_retrieval"])
            for path in output.values():
                self.assertNotIn("secret", path.read_text())
                self.assertNotIn("answers", path.read_text())
                saved = json.loads(path.read_text())
                self.assertEqual(saved["online"]["baseline_references"]["q"], rows)
                self.assertEqual(saved["online"]["splade_model"], "naver/splade-cocondenser-ensembledistil")
                self.assertEqual(saved["online"]["splade_model_dir"], str(local_model.resolve()))


if __name__ == "__main__":
    unittest.main()
