import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "compare_pseudo_page_alpha_rescues.py"
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("compare_pseudo_page_alpha_rescues", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ComparePseudoPageAlphaRescuesTests(unittest.TestCase):
    def write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_reports_unique_rescues_and_covered_page_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold = root / "gold.jsonl"
            base = root / "base.json"
            alpha_a = root / "a.json"
            alpha_b = root / "b.json"
            self.write_jsonl(
                gold,
                [
                    {
                        "qid": "q1",
                        "question": "one",
                        "metadata": {"gold_page_uids": ["d1_page0", "d1_page1"]},
                    },
                    {
                        "qid": "q2",
                        "question": "two",
                        "metadata": {"gold_page_uids": ["d2_page0"]},
                    },
                    {
                        "qid": "q3",
                        "question": "three",
                        "metadata": {"gold_page_uids": ["d3_page0"]},
                    },
                ],
            )
            base.write_text(
                json.dumps(
                    {
                        "q1": {"page_retrieval_results": [["x", 0, 9], ["d1", 0, 1], ["d1", 1, 0]]},
                        "q2": {"page_retrieval_results": [["x", 0, 9], ["d2", 0, 1]]},
                        "q3": {"page_retrieval_results": [["d3", 0, 9], ["x", 0, 1]]},
                    }
                ),
                encoding="utf-8",
            )
            alpha_a.write_text(
                json.dumps(
                    {
                        "q1": {"page_retrieval_results": [["d1", 0, 9], ["x", 0, 1]]},
                        "q2": {"page_retrieval_results": [["x", 0, 9], ["d2", 0, 1]]},
                        "q3": {"page_retrieval_results": [["x", 0, 9], ["d3", 0, 1]]},
                    }
                ),
                encoding="utf-8",
            )
            alpha_b.write_text(
                json.dumps(
                    {
                        "q1": {"page_retrieval_results": [["x", 0, 9], ["d1", 1, 1]]},
                        "q2": {"page_retrieval_results": [["d2", 0, 9], ["x", 0, 1]]},
                        "q3": {"page_retrieval_results": [["d3", 0, 9], ["x", 0, 1]]},
                    }
                ),
                encoding="utf-8",
            )

            old_argv = sys.argv
            try:
                sys.argv = [
                    "compare_pseudo_page_alpha_rescues.py",
                    "--gold",
                    str(gold),
                    "--baseline",
                    str(base),
                    "--run",
                    f"alpha_a={alpha_a}",
                    "--run",
                    f"alpha_b={alpha_b}",
                    "--hit-k",
                    "1",
                ]
                args = MODULE.parse_args()
            finally:
                sys.argv = old_argv

            gold_rows = MODULE.load_gold(Path(args.gold))
            baseline = MODULE.load_prediction(Path(args.baseline))
            runs = [
                MODULE.build_run_cases(
                    label=label,
                    gold=gold_rows,
                    baseline=baseline,
                    candidate=MODULE.load_prediction(path),
                    hit_k=int(args.hit_k),
                )
                for label, path in map(MODULE.parse_labeled_path, args.run)
            ]
            pair = MODULE.pair_summary(runs[0], runs[1])

            self.assertEqual(runs[0]["summary"]["recovered"], 1)
            self.assertEqual(runs[1]["summary"]["recovered"], 1)
            self.assertEqual(pair["recovered_overlap"], 0)
            self.assertEqual(pair["recovered_left_only"], 1)
            self.assertEqual(pair["recovered_right_only"], 1)
            self.assertEqual(pair["covered_pairs_left_only"], 1)
            self.assertEqual(pair["covered_pairs_right_only"], 2)


if __name__ == "__main__":
    unittest.main()
