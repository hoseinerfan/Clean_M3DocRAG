import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import probe_racs_legacy_dense as probe


class ProbeTests(unittest.TestCase):
    def fixture(self, root):
        prediction = root / "legacy.prediction.json"
        rows = {qid: {"qid": qid, "question": "not exported",
                      "reranker_metadata": {"base_score_source": "approx_page_maxsim_topk",
                      "approx_base_page_token_topk": 224,
                      "approx_base_page_token_scorer": "query_mean",
                      "approx_base_page_token_selector": "global_topk",
                      "gold_page_uids": ["not exported"]}} for qid in ("q1", "q2")}
        prediction.write_text(json.dumps(rows))
        audit = root / "audit.json"
        audit.write_text(json.dumps({"gold": {"qid_sha256": probe.qid_digest(rows)},
            "graphs": {"no_hyperlink": {"compact_configuration": {
                "dense_prediction_json": str(prediction)}}}}))
        return prediction, audit, rows

    def test_reads_exact_path_without_writes_or_question_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, audit, _ = self.fixture(root)
            before = {p: p.read_bytes() for p in root.iterdir()}
            result = probe.probe(audit)
            self.assertEqual(result["questions"], 2)
            self.assertEqual(result["configuration_groups"][0]["questions"], 2)
            self.assertEqual(result["missing_core_field_counts"], {})
            self.assertNotIn("not exported", json.dumps(result))
            self.assertEqual(before, {p: p.read_bytes() for p in root.iterdir()})

    def test_variation_and_missing_metadata_not_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            prediction, audit, rows = self.fixture(Path(directory))
            rows["q2"]["reranker_metadata"] = {}
            prediction.write_text(json.dumps(rows))
            result = probe.probe(audit)
            self.assertEqual(len(result["configuration_groups"]), 2)
            self.assertEqual(result["missing_core_field_counts"], {k: 1 for k in probe.CORE_KEYS})

    def test_cohort_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            prediction, audit, rows = self.fixture(Path(directory))
            del rows["q2"]
            prediction.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "question IDs"):
                probe.probe(audit)

    def test_detail_is_scoped_first_row_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, audit, _ = self.fixture(root)
            (root / "legacy.jsonl").write_text('\n' + json.dumps({
                "approx_base_page_token_coarse_dtype": "fp32", "answers": ["not exported"]}) + '\n')
            detail = probe.probe(audit)["possible_detail_jsonl"]
            self.assertEqual(detail["first_row_configuration"], {
                "approx_base_page_token_coarse_dtype": "fp32"})
            self.assertIn("not verified", detail["scope"])


if __name__ == "__main__":
    unittest.main()
