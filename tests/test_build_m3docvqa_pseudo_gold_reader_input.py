import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_m3docvqa_pseudo_gold_reader_input.py"


class BuildM3DocVQAPseudoGoldReaderInputTests(unittest.TestCase):
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
