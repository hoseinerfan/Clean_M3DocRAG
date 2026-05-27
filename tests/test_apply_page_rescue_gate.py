import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "apply_page_rescue_gate.py"
SPEC = importlib.util.spec_from_file_location("apply_page_rescue_gate", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PageRescueGateDocumentGuardTests(unittest.TestCase):
    def args(self, *, require_topk: bool) -> SimpleNamespace:
        return SimpleNamespace(
            support_page_rank_max=8,
            support_doc_rank_max=0,
            reject_promoted_page_idx=[],
            rescue_rank_min=9,
            rescue_rank_max=9,
            promoted_doc_max_base_rank=8,
            require_promoted_doc_in_base_topk=require_topk,
            min_candidate_score_margin=0.1,
        )

    def check_reason(self, *, require_topk: bool) -> str | None:
        reason, _detail = MODULE.reject_promotion_reason(
            uid="doc-b_page0",
            candidate_rank=8,
            question="question",
            base_top_pages=["doc-a_page0", "doc-a_page1", "doc-a_page2", "doc-a_page3"],
            base_page_rank_by_uid={"doc-b_page0": 9},
            base_doc_rank_by_id={"doc-a": 1, "doc-b": 2},
            candidate_rows=[],
            support_view_rows=[],
            heading_catalog={},
            body_catalog={},
            args=self.args(require_topk=require_topk),
        )
        return reason

    def test_strict_topk_document_membership_is_stricter_than_doc_rank_cap(self) -> None:
        self.assertEqual(self.check_reason(require_topk=True), "promoted_doc_not_in_base_topk")
        self.assertEqual(self.check_reason(require_topk=False), "candidate_score_margin_missing")


if __name__ == "__main__":
    unittest.main()
