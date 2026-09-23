import argparse
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_racs_alpha040_reader as job
import train_content_aware_pseudo_page_reranker as ca


class Alpha040Tests(unittest.TestCase):
    def setUp(self):
        self.args = argparse.Namespace(inference_mode="blend_rerank", blend_alpha=0.45)
        self.scored = ca.ranked_page_records({"page_retrieval_results": [["d", i, 10-i] for i in range(6)]}, 6)
        for row, probability in zip(self.scored, [0.4, 0.35, 0.3, 0.25, 0.9, 0.2]):
            row["learned_score"] = probability

    def test_change_only_alpha_without_mutating_shared_data(self):
        before = copy.deepcopy(self.scored)
        before_args = vars(self.args).copy()
        result = job.reblend(ca, self.scored, self.args, 0.40)
        direct_args = copy.copy(self.args)
        direct_args.blend_alpha = 0.40
        direct = ca.rerank_records(copy.deepcopy(before), direct_args)
        self.assertEqual(result, direct)
        self.assertEqual(before, self.scored)
        self.assertEqual(before_args, vars(self.args))

    def test_replay_gate_and_candidate_preservation(self):
        reference = {"page_retrieval_results": [r["raw"] for r in job.reblend(ca, self.scored, self.args, 0.45)]}
        saved, changed = job.verified_pair(ca, self.scored, self.args, reference, "q", count=6)
        self.assertEqual(saved, reference["page_retrieval_results"])
        self.assertEqual({tuple(p[:2]) for p in saved}, {tuple(p[:2]) for p in changed})
        reference["page_retrieval_results"] = list(reversed(reference["page_retrieval_results"]))
        with self.assertRaisesRegex(ValueError, "replay differs"):
            job.verified_pair(ca, self.scored, self.args, reference, "q", count=6)

    def test_raw_optional_fields_do_not_break_replay(self):
        for row in self.scored:
            row["raw"].append("optional metadata")
        reference = {"page_retrieval_results": [r["raw"] for r in job.reblend(ca, self.scored, self.args, 0.45)]}
        saved, changed = job.verified_pair(ca, self.scored, self.args, reference, "q", count=6)
        self.assertTrue(all(len(p) == 3 for p in saved + changed))


if __name__ == "__main__":
    unittest.main()
