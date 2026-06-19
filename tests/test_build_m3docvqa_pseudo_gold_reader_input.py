import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_m3docvqa_pseudo_gold_reader_input.py"
SPEC = importlib.util.spec_from_file_location("build_m3docvqa_pseudo_gold_reader_input", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BuildM3DocVQAPseudoGoldReaderInputTests(unittest.TestCase):
    def test_visual_proxy_control_replaces_only_proxy_pages(self) -> None:
        rows, missing = MODULE.make_visual_proxy_same_doc_non_gold_rows(
            gold_uids=["d1_page1", "d2_page2"],
            supervision_tiers={"d1_page1": "direct", "d2_page2": "visual_proxy"},
            pages_by_doc={"d1": [0, 1], "d2": [1, 2, 3]},
            top_pages=4,
        )
        self.assertEqual(missing, 0)
        self.assertEqual([(row[0], row[1]) for row in rows], [("d1", 1), ("d2", 1)])

    def test_filters_qids_by_supervision_tier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            augmented = root / "augmented.jsonl"
            original = root / "original.jsonl"
            output = root / "prediction.json"
            filtered = root / "gold.jsonl"
            summary = root / "summary.json"
            rows = [
                {
                    "qid": "q_direct",
                    "question": "direct",
                    "supporting_context": [{"doc_id": "d1"}],
                    "metadata": {
                        "gold_page_uids": ["d1_page0"],
                        "pseudo_gold_qid_supervision_tier": "complete_direct",
                    },
                },
                {
                    "qid": "q_proxy",
                    "question": "proxy",
                    "supporting_context": [{"doc_id": "d2"}],
                    "metadata": {
                        "gold_page_uids": ["d2_page1"],
                        "pseudo_gold_qid_supervision_tier": "complete_hybrid",
                    },
                },
            ]
            payload = "".join(json.dumps(row) + "\n" for row in rows)
            augmented.write_text(payload, encoding="utf-8")
            original.write_text(payload, encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--augmented-gold",
                    str(augmented),
                    "--original-gold",
                    str(original),
                    "--qid-supervision-tier",
                    "complete_direct",
                    "--output-prediction-json",
                    str(output),
                    "--output-filtered-gold",
                    str(filtered),
                    "--output-summary",
                    str(summary),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            prediction = json.loads(output.read_text(encoding="utf-8"))
            stats = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(set(prediction), {"q_direct"})
            self.assertEqual(stats["written_qids"], 1)
            self.assertEqual(stats["skipped_supervision_tier_mismatch"], 1)
            self.assertEqual(stats["qid_supervision_tiers"], ["complete_direct"])


if __name__ == "__main__":
    unittest.main()
