import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import validate_racs_remaining_reranker_readers as run


class RemainingReaderTests(unittest.TestCase):
    def config(self, method):
        if method == "lambdamart":
            return {"backend": "lightgbm", "objective": "lambdarank",
                    "inference_mode": "blend_rerank", "blend_alpha": .45}
        return {"model_name_or_path": "castorini/monot5-base-msmarco-10k",
                "candidate_top_k": 1000, "rerank_top_k": 1000,
                "blend_alpha": 1., "max_length": 512, "max_page_chars": 4000}

    def fixture(self, method):
        pages = [["d", i, float(10-i)] for i in range(5)]
        base = {"q": {"page_retrieval_results": pages}}
        key = "graph_aware_ltr_page_reranker" if method == "lambdamart" else "standard_seq2seq_page_reranker"
        pred = {"q": {"page_retrieval_results": list(reversed(pages)),
                      "reranker_metadata": {key: self.config(method)}}}
        return [{"qid": "q"}], base, pred

    def test_preserves_saved_order_for_both_methods(self):
        for method in run.STEMS:
            gold, base, pred = self.fixture(method)
            original = copy.deepcopy(pred)
            result = run.reader_inputs(method, gold, base, pred, 1, 5)
            self.assertEqual(result["q"]["page_retrieval_results"], pred["q"]["page_retrieval_results"][:4])
            self.assertEqual(pred, original)

    def test_changed_candidate_pool_rejected(self):
        gold, base, pred = self.fixture("monot5")
        pred["q"]["page_retrieval_results"][0] = ["other", 0, 1.]
        with self.assertRaisesRegex(ValueError, "Candidate pool"):
            run.reader_inputs("monot5", gold, base, pred, 1, 5)

    def test_missing_question_rejected(self):
        gold, base, pred = self.fixture("lambdamart")
        with self.assertRaisesRegex(ValueError, "QIDs"):
            run.reader_inputs("lambdamart", gold, base, {}, 1, 5)

    def test_wrong_trial_and_backend_rejected(self):
        for method, key, value in (("monot5", "rerank_top_k", 100),
                                   ("monot5", "blend_alpha", .2),
                                   ("lambdamart", "backend", "sklearn"),
                                   ("lambdamart", "blend_alpha", .4)):
            config = self.config(method)
            config[key] = value
            with self.assertRaises(ValueError):
                run.check_config(method, config)

    def test_duplicate_candidates_rejected(self):
        gold, base, pred = self.fixture("lambdamart")
        pages = pred["q"]["page_retrieval_results"]
        pages[0] = pages[1]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            run.reader_inputs("lambdamart", gold, base, pred, 1, 5)

    def test_paths_select_thesis_top1000_trials(self):
        for method in run.STEMS:
            paths = run.input_paths(Path("/repo"), method)
            self.assertIn("standard_reranker_baselines", str(paths["prediction"]))
            if method == "lambdamart":
                self.assertTrue(str(paths["prediction"]).endswith(".dev.prediction.json"))
            else:
                self.assertTrue(str(paths["prediction"]).endswith("gpp_top1000.prediction.json"))


if __name__ == "__main__":
    unittest.main()
