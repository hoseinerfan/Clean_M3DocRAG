import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_racs_lambdamart30 as workflow


class LambdaMART30Tests(unittest.TestCase):
    def fixture(self, root):
        gold = {"q1": {"qid": "q1", "question": "Blue lantern festival city?",
                       "metadata": {"gold_page_uids": ["d_page1"]}}}
        base = {"q1": {"qid": "q1", "page_retrieval_results": [
            ["d", 0, 10.0], ["d", 1, 8.0], ["x", 0, 6.0], ["z", 0, 4.0]]}}
        page_path = root / "pages.jsonl"
        page_path.write_text("\n".join(json.dumps(r) for r in [
            {"doc_id": "d", "page_idx": 0, "text": "irrelevant weather"},
            {"doc_id": "d", "page_idx": 1, "text": "Blue lantern festival city Lisbon"},
            {"doc_id": "x", "page_idx": 0, "text": "sports"},
            {"doc_id": "z", "page_idx": 0, "text": "cooking"}]))
        pages = workflow.ca.load_page_features(page_path)
        args = workflow.controlled.training_args(workflow.controlled.input_paths(root), root)
        args.candidate_top_k = 4
        return gold, base, pages, args

    def test_exact_feature_names_not_just_count(self):
        self.assertEqual(workflow.FEATURES, workflow.ca.BASE_FEATURE_NAMES)
        self.assertEqual(len(workflow.FEATURES), 30)
        self.assertFalse(any("visual" in name for name in workflow.FEATURES))

    def test_groups_are_contiguous(self):
        self.assertEqual(workflow.group_sizes(["a", "a", "b"]), [2, 1])
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            workflow.group_sizes(["a", "b", "a"])
        with self.assertRaises(ValueError):
            workflow.group_sizes([])

    def test_features_match_capp_training_and_full_pool_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            gold, base, pages, args = self.fixture(Path(tmp))
            X, y, _, _, qids = workflow.ca.build_matrix(gold=gold, base_pred=base,
                page_features=pages, source_maps_by_label={}, args=args, return_query_ids=True)
            records = workflow.ca.ranked_page_records(base["q1"], 4)
            complete = workflow.feature_rows("q1", gold["q1"]["question"], records, pages, {})
            # Matrix emits positives first; inference retains base order.
            np.testing.assert_array_equal(X[0], complete[1])
            for vector in X:
                self.assertTrue(any(np.array_equal(vector, candidate) for candidate in complete))
            self.assertEqual(X.shape, (4, 30))
            self.assertEqual(y[0], 1)
            self.assertEqual(workflow.group_sizes(qids), [4])
            self.assertEqual(workflow.feature_rows("q", "", [], {}, {}).shape, (0, 30))

    def test_alpha_tie_break_is_smallest_and_no_dev_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            gold, base, pages, args = self.fixture(Path(tmp))
            records = workflow.ca.ranked_page_records(base["q1"], 4)
            for r in records:
                r["learned_score"] = float(r["base_rank"])
            with patch.object(workflow, "score", return_value=records) as scored:
                result = workflow.select_alpha(gold, base, pages, {}, None, args)
            self.assertEqual(result["selected_blend_alpha"], 0.0)
            self.assertEqual(len(result["alpha_scores"]), 21)
            self.assertEqual(result["selection_split"], "training_holdout_only")
            self.assertEqual(scored.call_count, 1)

    def test_reader_preserves_top4_and_rejects_pool_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, base, _, _ = self.fixture(Path(tmp))
            pred = copy.deepcopy(base)
            pred["q1"]["page_retrieval_results"].reverse()
            inputs = workflow.reader_inputs(base, pred, count=1, candidates=4)
            self.assertEqual(inputs["q1"]["page_retrieval_results"], pred["q1"]["page_retrieval_results"])
            pred["q1"]["page_retrieval_results"][0][0] = "outside_pool"
            with self.assertRaisesRegex(ValueError, "Candidate pool changed"):
                workflow.reader_inputs(base, pred, count=1, candidates=4)
            with self.assertRaises(ValueError):
                workflow.reader_inputs(base, {}, count=1, candidates=4)

    def test_model_fit_persistence_and_reload(self):
        try:
            import lightgbm
        except ImportError:
            self.skipTest("LightGBM is not installed in this test environment")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold, base, pages, args = self.fixture(root)
            small_params = {**workflow.TREE_PARAMS, "n_estimators": 3,
                            "min_data_in_leaf": 1, "n_jobs": 1}
            with patch.object(workflow, "TREE_PARAMS", small_params):
                fitted = workflow.fit(gold, base, pages, {}, args, root, "test")
            payload = json.loads((root / "test.model.json").read_text())
            self.assertEqual(payload["feature_names"], workflow.FEATURES)
            self.assertEqual(len(payload["mean"]), 30)
            self.assertEqual(len(payload["model_info"]["feature_importance"]), 30)
            rows = workflow.score("q1", gold["q1"], base, pages, {}, fitted)
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(np.isfinite(r["learned_score"]) for r in rows))


if __name__ == "__main__":
    unittest.main()
