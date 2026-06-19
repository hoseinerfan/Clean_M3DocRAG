import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "apply_mmqa_image_page_vlm_verification.py"
SPEC = importlib.util.spec_from_file_location("apply_mmqa_image_page_vlm_verification", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ApplyMMQAImagePageVLMVerificationTests(unittest.TestCase):
    def test_verified_only_keeps_direct_and_verified_visual_units(self) -> None:
        row = {
            "qid": "q1",
            "gold_doc_ids": ["d1", "d2"],
            "pseudo_gold_page_uids": ["d1_page0", "d2_page1", "d2_page2"],
            "pseudo_gold_pages": [
                {
                    "page_uid": "d1_page0",
                    "doc_id": "d1",
                    "score": 9.0,
                    "evidence_units": [
                        {"unit_id": "u0", "unit_type": "text_instance", "score": 9.0, "supervision_tier": "direct"}
                    ],
                },
                {
                    "page_uid": "d2_page1",
                    "doc_id": "d2",
                    "score": 7.0,
                    "evidence_units": [
                        {"unit_id": "u1", "unit_type": "image_instance", "score": 7.0, "supervision_tier": "visual_proxy"}
                    ],
                },
                {
                    "page_uid": "d2_page2",
                    "doc_id": "d2",
                    "score": 7.0,
                    "evidence_units": [
                        {"unit_id": "u2", "unit_type": "image_instance", "score": 7.0, "supervision_tier": "visual_proxy"}
                    ],
                },
            ],
        }
        decisions = {
            ("u1", "d2_page1"): {"decision": "verified"},
            ("u2", "d2_page2"): {"decision": "rejected"},
        }
        output, stats = MODULE.transform_label_row(row, decisions, "verified_only")
        self.assertEqual(output["pseudo_gold_page_uids"], ["d1_page0", "d2_page1"])
        self.assertEqual(output["pseudo_gold_pages"][1]["supervision_tier"], "direct_visual_verified")
        self.assertEqual(output["vlm_training_tier"], "partial_vlm_verified_positive_only")
        self.assertEqual(stats["proxy_units_removed"], 1)


if __name__ == "__main__":
    unittest.main()
