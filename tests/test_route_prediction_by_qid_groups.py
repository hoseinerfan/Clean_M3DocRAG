import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "route_prediction_by_qid_groups.py"
SPEC = importlib.util.spec_from_file_location("route_prediction_by_qid_groups", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RoutePredictionByQidGroupsTests(unittest.TestCase):
    def test_routes_group_qids_and_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            default = root / "default.json"
            specialist = root / "specialist.json"
            qids = root / "single.qids.txt"
            out = root / "routed.json"
            summary = root / "summary.json"

            default.write_text(
                json.dumps(
                    {
                        "q1": {"qid": "q1", "page_retrieval_results": [["default", 0, 1.0]]},
                        "q2": {"qid": "q2", "page_retrieval_results": [["default", 1, 1.0]]},
                    }
                ),
                encoding="utf-8",
            )
            specialist.write_text(
                json.dumps(
                    {
                        "q1": {"qid": "q1", "page_retrieval_results": [["special", 0, 1.0]]},
                    }
                ),
                encoding="utf-8",
            )
            qids.write_text("q1\n", encoding="utf-8")

            old_argv = sys.argv
            try:
                sys.argv = [
                    "route_prediction_by_qid_groups.py",
                    "--prediction",
                    f"default={default}",
                    "--prediction",
                    f"specialist={specialist}",
                    "--default-label",
                    "default",
                    "--route",
                    f"single={qids}:specialist",
                    "--output-prediction-json",
                    str(out),
                    "--output-summary-json",
                    str(summary),
                ]
                MODULE.main()
            finally:
                sys.argv = old_argv

            routed = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(routed["q1"]["page_retrieval_results"][0][0], "special")
            self.assertEqual(routed["q2"]["page_retrieval_results"][0][0], "default")
            self.assertEqual(
                routed["q1"]["reranker_metadata"]["qid_group_route"]["selected_label"],
                "specialist",
            )
            info = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(info["selected_label_counts"], {"default": 1, "specialist": 1})

    def test_rejects_overlapping_groups_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred = root / "pred.json"
            qids_a = root / "a.qids.txt"
            qids_b = root / "b.qids.txt"
            pred.write_text(json.dumps({"q1": {"qid": "q1"}}), encoding="utf-8")
            qids_a.write_text("q1\n", encoding="utf-8")
            qids_b.write_text("q1\n", encoding="utf-8")

            old_argv = sys.argv
            try:
                sys.argv = [
                    "route_prediction_by_qid_groups.py",
                    "--prediction",
                    f"pred={pred}",
                    "--default-label",
                    "pred",
                    "--route",
                    f"a={qids_a}:pred",
                    "--route",
                    f"b={qids_b}:pred",
                    "--output-prediction-json",
                    str(root / "out.json"),
                    "--output-summary-json",
                    str(root / "summary.json"),
                ]
                with self.assertRaises(ValueError):
                    MODULE.main()
            finally:
                sys.argv = old_argv


if __name__ == "__main__":
    unittest.main()
