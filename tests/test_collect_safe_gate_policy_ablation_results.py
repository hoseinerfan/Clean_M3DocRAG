import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "collect_safe_gate_policy_ablation_results.py"
)
SPEC = importlib.util.spec_from_file_location("collect_safe_gate_policy_ablation_results", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SafeGatePolicyAblationCollectorTests(unittest.TestCase):
    def summary(self, page_hit: int, doc_hit: int, doc_max: int) -> dict:
        return {
            "config": {
                "hit_k": 8,
                "min_page_overlap": 7,
                "min_support_page_votes": 2,
                "promoted_doc_max_base_rank": doc_max,
                "require_promoted_doc_in_base_topk": False,
            },
            "accepted_count": 3,
            "base_page_hit_at_k_count": 100,
            "candidate_page_hit_at_k_count": 102,
            "page_hit_at_k_count": page_hit,
            "recovered": page_hit - 100,
            "lost": 0,
            "net_recovered": page_hit - 100,
            "base_doc_hit_at_k_count": 120,
            "doc_hit_at_k_count": doc_hit,
            "doc_net_recovered": doc_hit - 120,
            "selection_reason_counts": {"page_overlap_below_min": 4},
            "rejected_promotion_reason_counts": {
                "promoted_doc_after_allowed_rank": 2,
                "support_page_votes_below_min": 1,
                "promoted_body_score_not_above_base": 5,
            },
        }

    def test_reports_variant_deltas_against_dataset_control(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control = root / "control.json"
            relaxed = root / "relaxed.json"
            control.write_text(json.dumps(self.summary(101, 120, 8)), encoding="utf-8")
            relaxed.write_text(json.dumps(self.summary(103, 121, 0)), encoding="utf-8")

            rows, hit_k = MODULE.collect_rows(
                [
                    ("MMDocIR", "control", control),
                    ("MMDocIR", "no_doc_rank_cap", relaxed),
                ]
            )
            markdown = MODULE.render_markdown(rows, hit_k)

        self.assertEqual(hit_k, 8)
        self.assertEqual(rows[1]["delta_page_vs_control"], 2)
        self.assertEqual(rows[1]["delta_doc_vs_control"], 1)
        self.assertIn("| MMDocIR | no doc-rank cap |", markdown)
        self.assertIn("| +2 |", markdown)


if __name__ == "__main__":
    unittest.main()
