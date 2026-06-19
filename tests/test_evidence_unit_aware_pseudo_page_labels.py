import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_mmqa_evidence_unit_aware_pseudo_page_labels.py"
SPEC = importlib.util.spec_from_file_location("build_mmqa_evidence_unit_aware_pseudo_page_labels", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def page(doc_id: str, page_idx: int, text: str) -> dict:
    return {
        "page_uid": f"{doc_id}_page{page_idx}",
        "doc_id": doc_id,
        "page_idx": page_idx,
        "norm_text": MODULE.base.normalize_text(text),
        "tokens": set(MODULE.base.tokenize(text)),
    }


class EvidenceUnitAwarePseudoPageLabelsTests(unittest.TestCase):
    def test_image_title_proxy_is_explicit_and_uses_proxy_threshold(self) -> None:
        unit = MODULE.EvidenceUnit(
            unit_id="q1::unit0",
            unit_type="image_instance",
            doc_id="doc1",
            evidence=(MODULE.base.Evidence(text="Federal government seal", source="image_title", weight=7.0, doc_id="doc1"),),
        )
        matches = MODULE.score_unit_pages(
            unit,
            {"doc1": [page("doc1", 2, "Federal government seal")]},
            min_score=8.0,
            high_confidence_score=14.0,
            min_token_overlap=0.72,
            require_exact_or_high_confidence_fuzzy=True,
            require_direct_evidence_gate=True,
            direct_fuzzy_overlap=0.90,
            image_evidence_mode="proxy",
            image_title_proxy_min_score=7.0,
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].supervision_tier, "visual_proxy")
        self.assertEqual(matches[0].direct_gate, "exact")

    def test_image_proxy_can_be_excluded_without_affecting_other_units(self) -> None:
        image_unit = MODULE.EvidenceUnit(
            unit_id="q1::unit0",
            unit_type="image_instance",
            doc_id="doc1",
            evidence=(MODULE.base.Evidence(text="Federal government seal", source="image_title", weight=7.0, doc_id="doc1"),),
        )
        matches = MODULE.score_unit_pages(
            image_unit,
            {"doc1": [page("doc1", 2, "Federal government seal")]},
            min_score=8.0,
            high_confidence_score=14.0,
            min_token_overlap=0.72,
            require_exact_or_high_confidence_fuzzy=True,
            require_direct_evidence_gate=True,
            direct_fuzzy_overlap=0.90,
            image_evidence_mode="exclude",
        )
        self.assertEqual(matches, [])

    def test_image_proxy_requires_image_title_match_even_without_direct_gate(self) -> None:
        unit = MODULE.EvidenceUnit(
            unit_id="q1::unit0",
            unit_type="image_instance",
            doc_id="doc1",
            evidence=(
                MODULE.base.Evidence(text="Target image", source="image_title", weight=7.0, doc_id="doc1"),
                MODULE.base.Evidence(text="Supporting document", source="supporting_doc_title", weight=9.0, doc_id="doc1"),
            ),
        )
        matches = MODULE.score_unit_pages(
            unit,
            {"doc1": [page("doc1", 0, "Supporting document")]},
            min_score=8.0,
            high_confidence_score=14.0,
            min_token_overlap=0.72,
            require_exact_or_high_confidence_fuzzy=True,
            require_direct_evidence_gate=False,
            direct_fuzzy_overlap=0.90,
            image_evidence_mode="proxy",
            image_title_proxy_min_score=7.0,
        )
        self.assertEqual(matches, [])

    def test_partial_hybrid_is_positive_only_and_preserves_unknown_units(self) -> None:
        direct_unit = MODULE.EvidenceUnit(
            unit_id="q1::unit0",
            unit_type="text_instance",
            doc_id="doc1",
            evidence=(),
        )
        unresolved_unit = MODULE.EvidenceUnit(
            unit_id="q1::unit1",
            unit_type="image_instance",
            doc_id="doc2",
            evidence=(),
        )
        match = MODULE.UnitPageMatch(
            unit=direct_unit,
            page_uid="doc1_page0",
            doc_id="doc1",
            page_idx=0,
            score=9.0,
            direct_gate="exact",
            supervision_tier="direct",
        )
        label = MODULE.PageLabel(
            page_uid="doc1_page0",
            doc_id="doc1",
            page_idx=0,
            score=9.0,
            unit_matches=[match],
        )
        tier = MODULE.qid_supervision_tier(
            selected=[label],
            units=[direct_unit, unresolved_unit],
            mapped_unit_ids={direct_unit.unit_id},
        )
        self.assertEqual(tier, "partial_direct_positive_only")

        augmented = MODULE.augmented_gold_row(
            {"qid": "q1", "metadata": {}},
            [label],
            supervision_tier=tier,
            mapped_unit_ids={direct_unit.unit_id},
            units=[direct_unit, unresolved_unit],
        )
        metadata = augmented["metadata"]
        self.assertEqual(metadata["pseudo_gold_unmapped_evidence_unit_ids"], ["q1::unit1"])
        self.assertEqual(metadata["pseudo_gold_negative_supervision_scope"], "non_support_docs_only")


if __name__ == "__main__":
    unittest.main()
