import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_mmqa_image_page_vlm_verifier.py"
SPEC = importlib.util.spec_from_file_location("run_mmqa_image_page_vlm_verifier", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MMQAImagePageVLMVerifierTests(unittest.TestCase):
    def test_response_parser_is_conservative(self) -> None:
        self.assertEqual(MODULE.parse_verification_response("yes")[0], "verified")
        self.assertEqual(MODULE.parse_verification_response("No.")[0], "rejected")
        self.assertEqual(MODULE.parse_verification_response("The images may be related")[0], "unclear")

    def test_build_cases_keeps_only_visual_proxy_image_units(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "reference.jpg").write_bytes(b"not-decoded-during-case-build")
            labels = [
                {
                    "qid": "q1",
                    "question_type": "ImageQ",
                    "pseudo_gold_pages": [
                        {
                            "page_uid": "doc_page2",
                            "doc_id": "doc",
                            "page_idx": 2,
                            "score": 7.0,
                            "confidence": "low",
                            "evidence_units": [
                                {
                                    "unit_id": "q1::unit0",
                                    "unit_type": "image_instance",
                                    "doc_id": "image1",
                                    "supervision_tier": "visual_proxy",
                                },
                                {
                                    "unit_id": "q1::unit1",
                                    "unit_type": "text_instance",
                                    "doc_id": "doc",
                                    "supervision_tier": "direct",
                                },
                            ],
                        }
                    ],
                }
            ]
            images = {"image1": {"id": "image1", "title": "Reference", "path": "reference.jpg"}}
            cases = MODULE.build_cases(labels, images, root, set())
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0]["unit_id"], "q1::unit0")
            self.assertEqual(cases[0]["page_uid"], "doc_page2")
            self.assertTrue(Path(cases[0]["reference_image_path"]).is_file())

    def test_page_preflight_falls_back_to_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pdf_dir = Path(tmp)
            (pdf_dir / "doc.pdf").write_bytes(b"preflight-only")
            loader = MODULE.PageImageLoader(page_image_paths={}, pdf_dir=pdf_dir, dpi=144)
            self.assertEqual(loader.preflight("doc_page2"), (True, "pdf"))
            self.assertEqual(loader.preflight("missing_page0"), (False, "missing_pdf"))


if __name__ == "__main__":
    unittest.main()
