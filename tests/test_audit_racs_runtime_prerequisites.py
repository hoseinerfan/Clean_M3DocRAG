from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("audit", Path(__file__).resolve().parents[1] / "scripts/audit_racs_runtime_prerequisites.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class PrerequisiteTests(unittest.TestCase):
    def test_duplicate_qids_rejected_and_timers_are_not_end_to_end(self):
        with self.assertRaises(ValueError):
            audit.prediction_rows([{"qid": "same"}, {"qid": "same"}])
        timers = audit.timer_inventory({"q": {"time_qa": 2.5, "question": "text", "time_bad": float("nan")}})
        self.assertEqual(set(timers), {"time_qa"})
        self.assertIn("not verified", timers["time_qa"]["scope"])

    def test_graph_inventory_reports_cohort_config_variation_and_missing_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "graph.prediction.json"
            rows = {q: {"qid": q, "page_retrieval_results": [["doc", 0, 0.5]],
                        "reranker_metadata": {"dense_prediction_json": "missing.json",
                                              "ppr_iters": n}}
                    for q, n in (("q1", 30), ("q2", 31))}
            path.write_text(json.dumps(rows))
            result = audit.inspect_graph(path, root, {"q1", "q2"})
            self.assertTrue(result["qids_match_gold"])
            self.assertEqual(result["metadata_keys_varying_across_questions"], ["ppr_iters"])
            self.assertFalse(result["dependencies"]["dense_prediction_json"]["exists"])
            self.assertEqual(result["candidate_count_histogram"], {1: 2})
            self.assertFalse(audit.inspect_graph(path, root, {"q1"})["qids_match_gold"])

    def test_optional_small_file_missing_and_oversized_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "metadata.json"
            self.assertFalse(audit.inspect_small_json(path, root)["exists"])
            path.write_text('{"alpha": 0.4}')
            self.assertIn("not_read", audit.inspect_small_json(path, root, byte_limit=1))
            self.assertEqual(audit.inspect_small_json(path, root)["record"]["alpha"], 0.4)

    def test_complete_inventory_is_not_benchmark_and_preserves_originals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            gold = output / audit.GOLD
            gold.parent.mkdir(parents=True)
            gold.write_text('{"qid":"q1"}\n')
            original = {}
            for relative in audit.GRAPH_PATHS.values():
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"q1": {"page_retrieval_results": [], "reranker_metadata": {}}}))
                original[path] = path.read_bytes()
            with redirect_stdout(io.StringIO()):
                result = audit.audit(root)
            self.assertEqual(result["status"], "inventory_only_not_benchmark")
            self.assertFalse(result["checks"]["expected_2441_gold"])
            self.assertTrue(result["checks"]["graph_qids_match_gold"])
            self.assertFalse(result["bge_validated_result"]["exists"])
            self.assertEqual(original, {p: p.read_bytes() for p in original})
            existing = root / "report.json"
            existing.write_text("existing")
            with patch("sys.argv", ["audit", "--repo-root", str(root), "--output-json", str(existing)]), self.assertRaises(FileExistsError):
                audit.main()
            self.assertEqual(existing.read_text(), "existing")

    def test_tuning_candidates_require_relevant_full_feature_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "output/m3docvqa_content_aware_exact_maxsim"
            folder.mkdir(parents=True)
            model = folder / "trial.model.json"
            model.write_text(json.dumps({"args": {"feature_set": "all", "blend_alpha": 0.4,
                                                  "auto_tune_blend_alpha": True},
                                         "tuning_summary": {"selected_blend_alpha": 0.4},
                                         "feature_names": ["x"]}))
            model.with_name("trial.summary.json").write_text(json.dumps({"train_gold": "train.jsonl", "eval_gold": "dev.jsonl"}))
            result = audit.tuning_candidates(root / "output", root)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["model_tuning_summary"]["selected_blend_alpha"], 0.4)
            self.assertEqual(result[0]["inputs"]["eval_gold"], "dev.jsonl")
            model.write_text(json.dumps({"args": {"feature_set": "no_source"}}))
            self.assertEqual(audit.tuning_candidates(root / "output", root), [])


if __name__ == "__main__":
    unittest.main()
