import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_capp_runtime as benchmark

ca = benchmark.ca


class BenchmarkCappTests(unittest.TestCase):
    def setUp(self):
        names = ca.BASE_FEATURE_NAMES
        self.model = {
            "feature_names": names, "mean": [0.1] * len(names),
            "std": [0.8] * len(names), "weights": np.linspace(-0.1, 0.2, len(names)).tolist(),
            "bias": -0.1, "model_type": "logistic",
            "args": {"candidate_top_k": 3, "inference_mode": "blend_rerank", "blend_alpha": 0.4},
        }
        self.base = {"q1": {"page_retrieval_results": [["a", 0, 9], ["b", 0, 8], ["a", 1, 7]]},
                     "q2": {"page_retrieval_results": [["a", 1, 5], ["b", 0, 5], ["a", 0, 5]]}}
        self.questions = {"q1": "Which city hosted the 1996 Summer Olympics?", "q2": "Where is Atlanta?"}
        self.pages = {}
        for doc, page, text in [("a", 0, "Atlanta hosted 1996 Summer Olympics"),
                                ("a", 1, "Atlanta is in Georgia"), ("b", 0, "Summer Olympics history")]:
            uid = ca.page_uid(doc, page)
            self.pages[uid] = {"page_uid": uid, "doc_id": doc, "page_idx": page,
                               "norm_text": ca.normalize_text(text), "tokens": set(ca.tokenize(text)),
                               "token_count": len(set(ca.tokenize(text)))}
        self.sources = {"source": ca.source_maps(self.base, 3)}
        self.parameters = (np.asarray(self.model["mean"], dtype=np.float32),
                           np.asarray(self.model["std"], dtype=np.float32),
                           *ca.scorer_from_model_json(self.model))
        self.settings = benchmark.validate_model(self.model)

    def canonical(self, qid):
        mean, std, weights, bias = self.parameters
        records = ca.score_records(
            qid=qid, gold_row={"question": self.questions[qid]},
            records=ca.ranked_page_records(self.base[qid], 3), page_features=self.pages,
            source_maps_by_label=self.sources, mean=mean, std=std, weights=weights,
            bias=bias, feature_names=self.model["feature_names"],
        )
        return records, ca.rerank_records(records, self.settings)

    def test_split_scoring_matches_existing_scorer_and_order(self):
        for qid in self.base:
            canonical, ordered = self.canonical(qid)
            records = ca.ranked_page_records(self.base[qid], 3)
            X = benchmark.feature_matrix(qid, self.questions[qid], records, self.pages,
                                         self.sources, self.model["feature_names"])
            mean, std, weights, bias = self.parameters
            probabilities = ca.predict_scorer_proba(ca.standardize_eval(X, mean, std), weights, bias)
            np.testing.assert_array_equal(probabilities, [r["learned_score"] for r in canonical])
            uids, timing = benchmark.timed_query(qid, self.base[qid], self.questions[qid], self.pages,
                                                self.sources, self.model, self.settings, self.parameters)
            self.assertEqual(uids, [r["uid"] for r in ordered])
            self.assertTrue(all(v >= 0 for v in timing.values()))
            self.assertAlmostEqual(timing["total"], sum(v for k, v in timing.items() if k != "total"))

    def test_model_and_source_guards(self):
        for change in ({"inference_mode": "full_rerank"}, {"learned_query_alpha": True}, {"blend_alpha": 2}):
            model = copy.deepcopy(self.model)
            model["args"].update(change)
            with self.assertRaises(ValueError):
                benchmark.validate_model(model)
        model = copy.deepcopy(self.model)
        del model["args"]["blend_alpha"]
        with self.assertRaises(ValueError):
            benchmark.validate_model(model)
        with self.assertRaises(ValueError):
            benchmark.parse_sources(["same=a", "same=b"])

    def fixture(self, root):
        files = {
            "model-json": root / "model.json", "base-pred": root / "base.json",
            "question-jsonl": root / "questions.jsonl", "page-text-jsonl": root / "pages.jsonl",
            "saved-pred": root / "saved.json", "output-json": root / "new" / "runtime.json",
        }
        files["model-json"].write_text(json.dumps(self.model))
        files["base-pred"].write_text(json.dumps(self.base))
        files["question-jsonl"].write_text("".join(json.dumps({"qid": q, "question": text,
            "answers": ["NEVER_USE_THIS"], "metadata": {"pseudo_gold_pages": ["b_page0"]}}) + "\n"
            for q, text in self.questions.items()))
        files["page-text-jsonl"].write_text("".join(json.dumps({"doc_id": p["doc_id"],
            "page_idx": p["page_idx"], "text": p["norm_text"]}) + "\n" for p in self.pages.values()))
        saved = {q: {"page_retrieval_results": [r["raw"] for r in self.canonical(q)[1]]} for q in self.base}
        files["saved-pred"].write_text(json.dumps(saved))
        argv = [part for key, path in files.items() for part in ("--" + key, str(path))]
        argv += ["--source", f"source={files['base-pred']}", "--expected-qids", "2",
                 "--expected-candidates", "3", "--warmup-passes", "1", "--repeats", "2"]
        return files, argv

    def test_cli_read_only_inputs_and_validated_report(self):
        with tempfile.TemporaryDirectory() as directory:
            files, argv = self.fixture(Path(directory))
            before = {p: p.read_bytes() for k, p in files.items() if k != "output-json"}
            with contextlib.redirect_stdout(io.StringIO()):
                benchmark.main(argv)
            report = json.loads(files["output-json"].read_text())
            self.assertEqual(report["status"], "validated")
            self.assertEqual(report["questions"], 2)
            self.assertEqual([r["warmup"] for r in report["passes"]], [True, False, False])
            self.assertTrue(all(r["identical_complete_rankings"] == 2 for r in report["passes"]))
            self.assertEqual(report["per_question_mean_across_repeats"]["n"], 2)
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            with self.assertRaises(FileExistsError):
                benchmark.main(argv)

    def test_mismatch_aborts_without_success_report(self):
        with tempfile.TemporaryDirectory() as directory:
            files, argv = self.fixture(Path(directory))
            saved = json.loads(files["saved-pred"].read_text())
            saved["q1"]["page_retrieval_results"].reverse()
            files["saved-pred"].write_text(json.dumps(saved))
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "Ranking mismatch"):
                benchmark.main(argv)
            self.assertFalse(files["output-json"].exists())

    def test_missing_page_aborts_without_success_report(self):
        with tempfile.TemporaryDirectory() as directory:
            files, argv = self.fixture(Path(directory))
            files["page-text-jsonl"].write_text("")
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "Missing candidate page text"):
                benchmark.main(argv)
            self.assertFalse(files["output-json"].exists())

    def test_nearest_rank_p95(self):
        result = benchmark.summarize(list(range(1, 21)))
        self.assertEqual(result["p95_seconds"], 19)
        self.assertEqual(result["total_seconds"], 210)

    def test_launcher_reaches_python_without_working_git(self):
        launcher = Path(__file__).resolve().parents[1] / "examples" / "sbatch_racs_capp_runtime.sh"
        for git_available in (False, True):
            with self.subTest(git_available=git_available), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "scripts").mkdir()
                (root / "scripts" / "benchmark_capp_runtime.py").touch()
                bin_dir = root / "env" / "bin"
                bin_dir.mkdir(parents=True)
                python_stub = bin_dir / "python"
                python_stub.write_text('#!/bin/bash\nprintf "BENCHMARK_INVOKED\\n"\n')
                python_stub.chmod(0o755)
                if git_available:
                    git_stub = bin_dir / "git"
                    git_stub.write_text("#!/bin/bash\nexit 128\n")
                    git_stub.chmod(0o755)
                result = subprocess.run(
                    ["/bin/bash", str(launcher)], capture_output=True, text=True,
                    env={**os.environ, "PATH": str(bin_dir), "SLURM_SUBMIT_DIR": str(root),
                         "SLURM_JOB_ID": "test_no_git"},
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("benchmark_git_commit=unavailable", result.stdout)
                self.assertIn("BENCHMARK_INVOKED", result.stdout)

    def test_optional_provenance_command_can_be_missing(self):
        self.assertIsNone(benchmark.command_output(["/nonexistent/racs_test_git", "rev-parse", "HEAD"]))


if __name__ == "__main__":
    unittest.main()
