import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_racs_graph_reader as bench
import test_validate_racs_graph_replay as fixtures


class GraphReaderBenchmarkTests(unittest.TestCase):
    def test_balanced_schedule_requires_even_repeats(self):
        self.assertEqual(bench.schedule(2), [(1, "GPP"), (1, "CAPP"), (2, "CAPP"), (2, "GPP")])
        for count in (0, 1, 3):
            with self.assertRaises(ValueError):
                bench.schedule(count)

    def test_exclusive_report_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            bench.write_new(path, {"value": 1})
            with self.assertRaises(FileExistsError):
                bench.write_new(path, {"value": 2})
            self.assertEqual(json.loads(path.read_text()), {"value": 1})

    def bundle(self, method):
        rows = [["doc", page, 1.0 / (page + 1)] for page in range(1000)]
        config = fixtures.metadata()
        config["doc_pages_jsonl"] = "pages"
        entry = {"settings": config, "inputs": {"dense": "dense", "sparse": "sparse"}, "references": {"q": rows}}
        labels = [bench.MAIN] + (list(bench.AUXILIARIES) if method == "CAPP" else [])
        order = [bench.capp.ca.page_uid(row[0], row[1]) for row in rows]
        return {"method": method, "graphs": {label: copy.deepcopy(entry) for label in labels},
                "inputs": {key: {"q": {"question": "Which page?", "page_retrieval_results": rows}}
                           for key in ("dense", "sparse")}, "questions": {"q": "Which page?"},
                "capp_order_digests": {"q": bench.capp.order_digest(order)}}, rows, order

    def test_gpp_generates_one_graph_and_never_calls_capp(self):
        bundle, rows, order = self.bundle("GPP")
        with mock.patch.object(bench.replay.graph, "build_qid_graph_ranking", return_value=(rows, {})) as graph, \
                mock.patch.object(bench.capp, "timed_query") as scorer:
            selected, actual, regenerated, stages = bench.rank_query(bundle, "q", {"pages": None}, {}, None, None)
        self.assertEqual(graph.call_count, 1)
        self.assertIsNone(graph.call_args.kwargs["gold_row"])
        scorer.assert_not_called()
        self.assertEqual(selected, rows[:4])
        self.assertEqual(actual, order)
        bench.validate_rankings(bundle, "q", actual, regenerated)
        self.assertEqual(set(stages), {"graph:" + bench.MAIN})

    def test_capp_pays_for_four_graphs_and_new_source_maps_each_query(self):
        bundle, rows, order = self.bundle("CAPP")
        reversed_order = order[::-1]
        with mock.patch.object(bench.replay.graph, "build_qid_graph_ranking", return_value=(rows, {})) as graph, \
                mock.patch.object(bench.capp, "timed_query", return_value=(reversed_order, {"total": .2})) as scorer:
            selected, actual, regenerated, stages = bench.rank_query(bundle, "q", {"pages": None}, {}, {}, ({}, object(), ()))
        self.assertEqual(graph.call_count, 4)
        self.assertTrue(all(call.kwargs["gold_row"] is None for call in graph.call_args_list))
        self.assertEqual(set(scorer.call_args.args[4]), set(bench.AUXILIARIES.values()))
        self.assertEqual(selected, rows[::-1][:4])
        self.assertIn("source_maps", stages)
        self.assertIn("capp:total", stages)
        with self.assertRaisesRegex(ValueError, "CAPP complete ranking mismatch"):
            bench.validate_rankings(bundle, "q", actual, regenerated)
        with self.assertRaisesRegex(ValueError, "Graph replay mismatch"):
            bench.validate_rankings(bundle, "q", order, {**regenerated, bench.MAIN: rows[::-1]})

    def reports(self):
        manifest = {"measured_qids": ["q1", "q2"], "questions": 2, "scope": bench.SCOPE}
        hardware = {key: "same" for key in ("hostname", "gpu", "gpu_total_bytes", "cuda_visible_devices", "slurm_job_id",
                    "slurm_cpus_per_task", "torch_cpu_threads", "nvidia_smi")}
        reports = []
        for index, method in bench.schedule(2):
            times = ([1, 3] if index == 1 else [3, 5]) if method == "GPP" else ([2, 4] if index == 1 else [4, 6])
            reports.append({"status": "validated_graph_to_answer_worker", "scope": bench.SCOPE,
                "method": method, "pass": index, "questions": 2, "hardware": hardware.copy(),
                "reader": {"path": "same"}, "packages": {}, "bundle_sha256": method,
                "summary": bench.capp.summarize(times), "memory": {}, "preparation_seconds": {},
                "results": [{"qid": qid, "seconds": {"total": value, "part": value / 2},
                             "answer": "same", "ranking_checks_passed": True}
                            for qid, value in zip(manifest["measured_qids"], times)]})
        return reports, manifest

    def test_paired_aggregation_and_variance(self):
        reports, manifest = self.reports()
        reports[2]["results"][0]["answer"] = "changed"
        result = bench.aggregate(reports, manifest, 2)
        self.assertEqual(result["paired_capp_minus_gpp_seconds"]["mean"], 1)
        gpp = result["methods"]["GPP"]["per_question_mean_across_repeats"]["total"]
        self.assertEqual(gpp["mean_seconds"], 3)
        self.assertEqual(gpp["p95_seconds"], 4)
        self.assertEqual(result["methods"]["CAPP"]["qids_with_answer_variation_across_repeats"], 1)
        self.assertIn("NOT online", result["scope"])

    def test_rejects_incomplete_or_incomparable_reports(self):
        reports, manifest = self.reports()
        with self.assertRaises(ValueError):
            bench.aggregate(reports[:-1], manifest, 2)
        for change in (lambda r: r[1]["hardware"].update(gpu="other"),
                       lambda r: r[1]["results"][0].update(qid="other"),
                       lambda r: r[1]["results"][0].update(ranking_checks_passed=False),
                       lambda r: r[2].update(bundle_sha256="changed"),
                       lambda r: r[1]["results"][0]["seconds"].update(total=float("nan"))):
            changed = copy.deepcopy(reports)
            change(changed)
            with self.assertRaises(ValueError):
                bench.aggregate(changed, manifest, 2)

    def test_dynamic_cpu_frequency_description_not_a_hardware_mismatch(self):
        reports, manifest = self.reports()
        reports[0]["hardware"]["cpu_description"] = "current MHz 2000"
        reports[1]["hardware"]["cpu_description"] = "current MHz 2800"
        bench.aggregate(reports, manifest, 2)

    def test_prepare_produces_isolated_sanitized_condition_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit_path, rows = fixtures.ReplayTests().fixture(root)
            audit = json.loads(audit_path.read_text())
            item = next(iter(audit["graphs"].values()))
            audit["graphs"] = {label: item for label in (bench.MAIN, *bench.AUXILIARIES)}
            audit_path.write_text(json.dumps(audit))
            gold_path = Path(audit["gold"]["path"])
            gold_path.write_text(''.join(json.dumps({"qid": qid, "question": "Which page?", "answers": ["hidden"]}) + '\n' for qid in ("q1", "q2")))
            check_path = root / "replay.json"
            check_path.write_text(json.dumps({"status": "sample_graph_replay_validated", "input_sha256": {str(audit_path): bench.replay.sha256(audit_path)},
                "results": {label: {key: 16 for key in ("questions", "candidate_sets_match", "complete_order_matches", "scores_close")}
                            for label in audit["graphs"]}}))
            stem = root / bench.FULL_STEM
            stem.parent.mkdir(parents=True)
            names = bench.capp.ca.BASE_FEATURE_NAMES[:30]
            model = {"feature_names": names, "mean": [0] * 30, "std": [1] * 30, "weights": [0] * 30, "bias": 0,
                     "args": {"candidate_top_k": 1000, "inference_mode": "blend_rerank", "blend_alpha": .4}}
            Path(str(stem) + ".model.json").write_text(json.dumps(model))
            Path(str(stem) + ".dev.prediction.json").write_text(json.dumps({qid: {"page_retrieval_results": rows} for qid in ("q1", "q2")}))
            run_dir = root / "run"
            run_dir.mkdir()
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            paths, manifest = bench.prepare(audit_path, check_path, run_dir, 1, 1)
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            self.assertFalse(set(manifest["measured_qids"]) & set(manifest["warmup_qids"]))
            gpp, capp = (json.loads(paths[m].read_text()) for m in ("GPP", "CAPP"))
            self.assertEqual(set(gpp["graphs"]), {bench.MAIN})
            self.assertIsNone(gpp["model"])
            self.assertEqual(len(capp["graphs"]), 4)
            self.assertNotIn("hidden", paths["CAPP"].read_text())
            self.assertNotIn("answers", paths["CAPP"].read_text())
            # Fingerprint guards reject later modifications to source artifacts.
            check = json.loads(check_path.read_text())
            source_path = root / "upstream.json"
            check["input_sha256"][str(source_path)] = "wrong"
            check_path.write_text(json.dumps(check))
            with self.assertRaisesRegex(ValueError, "Input changed"):
                bench.prepare(audit_path, check_path, root / "unused", 1, 1)

    def test_worker_gpu_boundaries_validation_and_no_cross_question_image_cache(self):
        bundle, rows, order = self.bundle("GPP")
        bundle.update(measured_qids=["q"], warmup_qids=["warm"])
        bundle["questions"]["warm"] = "Warm-up?"
        # This worker test mocks computation; ranking interfaces are tested above.
        events = []
        cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
            synchronize=lambda: events.append("sync"), reset_peak_memory_stats=lambda: events.append("reset_peak"),
            get_device_properties=lambda _: SimpleNamespace(name="fake-gpu", total_memory=1),
            max_memory_allocated=lambda: 1024, max_memory_reserved=lambda: 2048)
        torch = SimpleNamespace(cuda=cuda, set_num_threads=lambda _: None, manual_seed=lambda _: None,
                                no_grad=contextlib.nullcontext, get_num_threads=lambda: 1)
        network = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(dtype="bfloat16")]))
        reader = SimpleNamespace(model=network, processor=SimpleNamespace(image_processor=object()),
                                 generate=lambda **_: events.append("generate") or "answer")
        dataset = SimpleNamespace(get_images_from_doc_id=lambda _: events.append("render") or list(range(1000)))
        accelerator = SimpleNamespace(num_processes=1, prepare=lambda value: value)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_path, output = root / "bundle.json", root / "worker.json"
            bench.write_new(bundle_path, bundle)
            qa = SimpleNamespace(M3DocVQADataset=lambda _: dataset, make_dataset_args=lambda value: value,
                resolve_model_path=lambda _: root, supports_flash_attention=lambda: False,
                VQAModel=lambda **_: reader,
                short_answer_template=SimpleNamespace(substitute=lambda value: value["question"]))
            fake_modules = {"torch": torch, "accelerate": SimpleNamespace(Accelerator=lambda: accelerator),
                            "run_m3docvqa_external_retrieval_qa": qa}
            def fake_rank(*_):
                events.append("rank")
                return rows[:4], order, {bench.MAIN: rows}, {"graph:" + bench.MAIN: .0001}
            def fake_check(*_):
                events.append("validate")
            with mock.patch.dict(sys.modules, fake_modules), \
                    mock.patch.object(bench.replay.graph, "load_doc_page_catalog", return_value=None), \
                    mock.patch.object(bench, "rank_query", side_effect=fake_rank), \
                    mock.patch.object(bench, "validate_rankings", side_effect=fake_check), \
                    mock.patch.object(bench.capp, "command_output", return_value="fake"), \
                    contextlib.redirect_stdout(io.StringIO()):
                bench.worker(bundle_path, output, 1)
            result = json.loads(output.read_text())
            self.assertEqual(result["questions"], 1)
            self.assertEqual(result["warmup_questions"], 1)
            self.assertEqual(events.count("render"), 2)  # One per query, even for same document.
            self.assertEqual(events.count("generate"), 2)
            for index, event in enumerate(events):
                if event == "rank":
                    self.assertEqual(events[index - 1], "sync")
                if event == "validate":
                    self.assertEqual(events[index - 1], "sync")
            self.assertLess(events.index("validate"), events.index("reset_peak"))
            self.assertIn("NOT online", result["scope"])


if __name__ == "__main__":
    unittest.main()
