import contextlib
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import diagnose_racs_exact_replay as exact
import run_visual_rerank_batch as runner
from test_benchmark_racs_online import exact_config
from test_diagnose_racs_faiss_replay import bundle, page_rows, query_meta, Tensor


def fake_torch():
    return SimpleNamespace(__version__="test-not-real-gpu", get_num_threads=lambda: 8,
        get_num_interop_threads=lambda: 8, set_num_threads=mock.Mock(),
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1, get_device_name=lambda n: "fake"),
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
                                 cudnn=SimpleNamespace(allow_tf32=False)),
        inference_mode=contextlib.nullcontext, from_numpy=Tensor)


class ExactDiagnosticTests(unittest.TestCase):
    def fixture(self, root):
        value = bundle()
        summary = {**exact_config(), "embedding_name": "test-embedding", "gold": str(root / "gold.jsonl")}
        value["online"].update(embedding_name=summary["embedding_name"], roles={"exact": "exact"},
            scoring_query_device="cpu", query_filter="full", backbone="backbone", adapter="adapter",
            exact_options=exact.online.scoring_options(summary))
        value["inputs"] = {"exact": {qid: {"page_retrieval_results": page_rows()} for qid in value["warmup_qids"]}}
        exact.write_jsonl(root / "gold.jsonl", ({"qid": qid, "question": value["questions"][qid], "answers": []}
                                               for qid in value["warmup_qids"]))
        source, settings = root / "bundle.json", root / "summary.json"
        source.write_text(json.dumps(value))
        settings.write_text(json.dumps(summary))
        prior = root / "prior"
        prior.mkdir()
        queries = []
        for qid in value["warmup_qids"]:
            record = exact.save_query(query_meta(), prior, f"query_direct_gpu_{qid}")
            queries.append({"qid": qid, "encoder_path": "direct_gpu", **record["summary"]})
        (prior / "diagnostic.json").write_text(json.dumps({"status": "diagnostic_complete_not_a_runtime_result",
            "questions": value["warmup_qids"], "query_reports": queries}))
        output = root / "output"
        output.mkdir()
        return source, settings, prior, output

    def prepared(self, root):
        paths = self.fixture(root)
        return exact.prepare(*paths), paths[-1]

    def test_prepare_preserves_original_inputs_and_fixed_four_question_subset(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = self.fixture(Path(folder))
            before = {p: p.read_bytes() for p in Path(folder).rglob("*") if p.is_file()}
            inputs = exact.prepare(*paths)
            self.assertEqual(inputs["qids"], ["q0", "q1", "q2", "q3"])
            self.assertEqual(set(inputs["expected"]), set(inputs["baseline"]))
            for p, content in before.items():
                self.assertEqual(p.read_bytes(), content)
            gold = [json.loads(line) for line in Path(inputs["paths"]["gold.jsonl"]).read_text().splitlines()]
            self.assertEqual([row["qid"] for row in gold], inputs["qids"])
            self.assertTrue(all("answers" in row for row in gold))
            self.assertNotIn("baseline_references", inputs["config"])

    def test_prepare_rejects_question_normalization_and_duplicate_gold(self):
        for problem in ("text", "duplicate", "missing"):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                paths = self.fixture(root)
                gold = root / "gold.jsonl"
                rows = [json.loads(line) for line in gold.read_text().splitlines()]
                if problem == "text":
                    rows[0]["question"] += " "
                elif problem == "duplicate":
                    rows.append(rows[0])
                else:
                    rows.pop()
                gold.write_text("\n".join(json.dumps(row) for row in rows))
                with self.assertRaises(ValueError):
                    exact.prepare(*paths)

    def test_prepare_rejects_different_scoring_options_and_prior_cohort(self):
        for problem in ("options", "cohort"):
            with tempfile.TemporaryDirectory() as folder:
                paths = self.fixture(Path(folder))
                p = paths[0] if problem == "options" else paths[2] / "diagnostic.json"
                data = json.loads(p.read_text())
                if problem == "options":
                    data["online"]["exact_options"]["page_batch_size"] = 1
                else:
                    data["questions"] = data["questions"][::-1]
                p.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    exact.prepare(*paths)

    def test_original_command_parses_with_the_real_entry_point(self):
        with tempfile.TemporaryDirectory() as folder:
            inputs, output = self.prepared(Path(folder))
            previous = sys.argv
            with exact.argv_scope(exact.original_argv(inputs, output)):
                args = runner.parse_args()
            self.assertIs(sys.argv, previous)
            self.assertEqual(args.base_score_source, "exact_page_maxsim")
            self.assertEqual(args.base_only_page_batch_size, 64)
            self.assertEqual(args.from_baseline_top_pages, 1000)
            self.assertEqual(args.query_token_filter, "full")
            self.assertFalse(args.ignore_pad_scores_in_final_ranking)
            self.assertFalse(args.grid_search)
            self.assertFalse(args.resume_output_jsonl)
            self.assertEqual(args.retrieval_model_name_or_path, inputs["config"]["backbone"])

    def test_saved_query_integrity_and_exact_float32_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            record = exact.save_query(query_meta(), Path(folder), "query")
            loaded = exact.load_query(record, fake_torch())
            np.testing.assert_array_equal(loaded["embeddings"].numpy(), query_meta()["embeddings"].float().numpy())
            bad = copy.deepcopy(record)
            bad["summary"]["token_ids"][0] = 99
            with self.assertRaisesRegex(ValueError, "metadata"):
                exact.load_query(bad, fake_torch())
            bad = copy.deepcopy(record)
            bad["file_sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "file changed"):
                exact.load_query(bad, fake_torch())
            bad = copy.deepcopy(record)
            bad["summary"]["raw_tokens"].pop()
            with self.assertRaisesRegex(ValueError, "dimensions"):
                exact.load_query(bad, fake_torch())

    def test_original_worker_captures_without_changing_original_query_method(self):
        with tempfile.TemporaryDirectory() as folder:
            inputs, root = self.prepared(Path(folder))
            output = root / "original"
            output.mkdir()
            calls = []
            class Encoder:
                model = SimpleNamespace(device="cpu")
                def encode_query_with_metadata(self, query, to_cpu=False, query_token_filter="full"):
                    calls.append(query)
                    if len(calls) > 4:
                        self_check = output / "original.prediction.json"
                        if not self_check.is_file():
                            raise AssertionError("Extra query run preceded original outputs")
                    return query_meta()
            original_method = Encoder.encode_query_with_metadata
            def run_original():
                encoder = Encoder()
                for qid in inputs["qids"]:
                    encoder.encode_query_with_metadata(inputs["questions"][qid], to_cpu=True, query_token_filter="full")
                exact.bench.write_new(output / "original.prediction.json", {
                    qid: {"page_retrieval_results": page_rows()} for qid in inputs["qids"]})
            torch = fake_torch()
            with mock.patch.dict(sys.modules, {"torch": torch,
                    "run_visual_rerank_batch": SimpleNamespace(main=run_original),
                    "m3docrag.retrieval": SimpleNamespace(ColPaliRetrievalModel=Encoder)}):
                exact.original_worker(inputs, output, 1)
            self.assertIs(Encoder.encode_query_with_metadata, original_method)
            torch.set_num_threads.assert_called_once_with(1)
            report = json.loads((output / "worker.json").read_text())
            self.assertEqual(len(calls), 8)
            self.assertEqual(len(list(output.glob("*.npz"))), 8)
            self.assertTrue(all(row["scores_close"] for row in report["comparisons"].values()))

    def test_harness_uses_production_dense_method_without_online_loader(self):
        with tempfile.TemporaryDirectory() as folder:
            inputs, root = self.prepared(Path(folder))
            for mode in exact.ORIGINAL_MODES:
                sub = root / mode
                sub.mkdir()
                queries = {qid: {context: exact.save_query(query_meta(), sub, f"{context}_{qid}")
                    for context in ("no_grad", "inference_mode")} for qid in inputs["qids"]}
                exact.bench.write_new(sub / "worker.json", {"queries": queries})
            output = root / "harness"
            output.mkdir()
            loader = mock.Mock(return_value={"doc": "test-embeddings"})
            modules = {"torch": fake_torch(),
                "rerank_target_docs_visual_aware": SimpleNamespace(load_doc_embeddings_for_doc_ids=loader)}
            with mock.patch.dict(sys.modules, modules), \
                 mock.patch.object(exact.online.OnlineRetriever, "__init__", side_effect=AssertionError("No full loader")), \
                 mock.patch.object(exact.online.OnlineRetriever, "dense_scores", return_value=page_rows()) as score:
                exact.harness_worker(inputs, root, output)
            self.assertEqual(score.call_count, 20)
            self.assertTrue(all(call.args[2] is False for call in score.call_args_list))
            loader.assert_called_once_with(["doc"], "test-embedding")
            report = json.loads((output / "worker.json").read_text())
            self.assertEqual(set(report["comparisons"]), set(exact.HARNESS_MODES))

    def test_summary_requires_all_conditions_and_does_not_select_a_winner(self):
        qids = ["q0", "q1", "q2", "q3"]
        cells = {qid: {"candidate_sets_match": True, "complete_order_matches": True, "scores_close": True} for qid in qids}
        inputs = {"qids": qids, "input_sha256": {}}
        originals = {mode: {"comparisons": copy.deepcopy(cells), "environment": {}} for mode in exact.ORIGINAL_MODES}
        harness = {"comparisons": {mode: copy.deepcopy(cells) for mode in exact.HARNESS_MODES}, "environment": {}}
        harness["comparisons"][exact.HARNESS_MODES[0]]["q0"]["scores_close"] = False
        report = exact.summarize(inputs, originals, harness)
        self.assertEqual(report["status"], "diagnostic_complete_not_a_runtime_result")
        self.assertTrue(report["no_automatic_configuration_selection"])
        self.assertEqual(report["totals"]["harness/" + exact.HARNESS_MODES[0]]["complete_order_and_score_matches"], 3)
        del harness["comparisons"][exact.HARNESS_MODES[-1]]
        with self.assertRaisesRegex(ValueError, "conditions"):
            exact.summarize(inputs, originals, harness)

    def test_cli_refuses_existing_output_and_marks_failures_not_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            argv = ["diag", "--bundle", "b", "--exact-summary", "s", "--prior-query-dir", "q", "--output-dir"]
            with mock.patch.object(sys, "argv", argv + [str(root)]), mock.patch.object(exact, "prepare") as prepare:
                with self.assertRaises(FileExistsError):
                    exact.main()
                prepare.assert_not_called()
            with mock.patch.object(sys, "argv", argv + [str(root / "new")]), \
                 mock.patch.object(exact, "prepare", side_effect=ValueError("fixture")):
                with self.assertRaisesRegex(ValueError, "fixture"):
                    exact.main()
            self.assertEqual(json.loads((root / "new/failure.json").read_text())["status"], "diagnostic_failed_not_a_runtime_result")
            self.assertFalse((root / "new/runtime.json").exists())


if __name__ == "__main__":
    unittest.main()
