import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "summarize_safe_gate_case_diagnostics.py"
)
SPEC = importlib.util.spec_from_file_location("summarize_safe_gate_case_diagnostics", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SafeGateCaseDiagnosticsTests(unittest.TestCase):
    def test_builds_aggregate_and_case_sections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path = root / "summary.json"
            cases_path = root / "cases.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "config": {"hit_k": 8},
                        "accepted_count": 2,
                        "base_page_hit_at_k_count": 10,
                        "candidate_page_hit_at_k_count": 11,
                        "page_hit_at_k_count": 10,
                        "recovered": 1,
                        "lost": 1,
                        "net_recovered": 0,
                        "base_doc_hit_at_k_count": 20,
                        "candidate_doc_hit_at_k_count": 20,
                        "doc_hit_at_k_count": 20,
                        "doc_net_recovered": 0,
                        "candidate_recovered": 1,
                        "candidate_lost": 0,
                        "candidate_net_recovered": 1,
                    }
                ),
                encoding="utf-8",
            )
            cases_path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "qid": "q-lost",
                                "question": "lost question",
                                "accepted": True,
                                "movement_vs_base": "lost",
                                "candidate_movement_vs_base": "unchanged",
                                "base_first_gold_page_rank": 8,
                                "candidate_first_gold_page_rank": 9,
                                "output_first_gold_page_rank": 9,
                                "accepted_promoted_pages": [
                                    {
                                        "page_uid": "d_page9",
                                        "candidate_rank": 8,
                                        "base_rank": 9,
                                        "base_doc_rank": 2,
                                        "support_page_vote_count": 2,
                                    }
                                ],
                            },
                            {
                                "qid": "q-recovered",
                                "question": "recovered question",
                                "accepted": True,
                                "movement_vs_base": "recovered",
                                "candidate_movement_vs_base": "recovered",
                                "base_first_gold_page_rank": 9,
                                "candidate_first_gold_page_rank": 8,
                                "output_first_gold_page_rank": 8,
                                "accepted_promoted_pages": [],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            entry = MODULE.build_entry("MMDocIR", "boundary", summary_path, cases_path)
            markdown = MODULE.render_markdown([entry], topn=5)

        self.assertEqual(entry["net"], 0)
        self.assertEqual(entry["accepted_movements"]["lost"], 1)
        self.assertIn("| boundary | 1 | 2 | 10 | 11 | 10 | 1 | 1 | 0 | 0 |", markdown)
        self.assertIn("q-lost", markdown)
        self.assertIn("q-recovered", markdown)


if __name__ == "__main__":
    unittest.main()
