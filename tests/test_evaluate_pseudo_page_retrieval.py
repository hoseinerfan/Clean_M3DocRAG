import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_pseudo_page_retrieval.py"
SPEC = importlib.util.spec_from_file_location("evaluate_pseudo_page_retrieval", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class EvaluatePseudoPageRetrievalTests(unittest.TestCase):
    def write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_evaluates_page_recall_and_movement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold = root / "gold.jsonl"
            base = root / "base.json"
            cand = root / "cand.json"
            self.write_jsonl(
                gold,
                [
                    {"qid": "q1", "metadata": {"gold_page_uids": ["d1_page1"]}},
                    {"qid": "q2", "metadata": {"gold_page_uids": ["d2_page0"]}},
                    {"qid": "q3", "metadata": {}},
                ],
            )
            base.write_text(
                json.dumps(
                    {
                        "q1": {"page_retrieval_results": [["d1", 0, 3.0], ["d1", 1, 2.0]]},
                        "q2": {"page_retrieval_results": [["d9", 0, 3.0], ["d2", 0, 2.0]]},
                    }
                ),
                encoding="utf-8",
            )
            cand.write_text(
                json.dumps(
                    {
                        "q1": {"page_retrieval_results": [["d1", 1, 3.0], ["d1", 0, 2.0]]},
                        "q2": {"page_retrieval_results": [["d2", 0, 3.0], ["d9", 0, 2.0]]},
                    }
                ),
                encoding="utf-8",
            )

            old_argv = sys.argv
            try:
                sys.argv = [
                    "evaluate_pseudo_page_retrieval.py",
                    "--gold",
                    str(gold),
                    "--run",
                    f"base={base}",
                    "--run",
                    f"cand={cand}",
                    "--recall-k",
                    "1",
                    "2",
                    "--format",
                    "json",
                ]
                rows = MODULE.build_rows(MODULE.parse_args())
            finally:
                sys.argv = old_argv

            by_label = {row["label"]: row for row in rows}
            self.assertEqual(by_label["base"]["n_eval"], 2)
            self.assertEqual(by_label["base"]["skipped_no_page_gold"], 1)
            self.assertEqual(by_label["base"]["page@1"], 0.0)
            self.assertEqual(by_label["base"]["page@2"], 1.0)
            self.assertEqual(by_label["cand"]["page@1"], 1.0)
            self.assertEqual(by_label["cand"]["improved_rank"], 2)
            self.assertEqual(by_label["cand"]["recovered@4"], 0)
            self.assertEqual(by_label["cand"]["lost@4"], 0)


if __name__ == "__main__":
    unittest.main()
