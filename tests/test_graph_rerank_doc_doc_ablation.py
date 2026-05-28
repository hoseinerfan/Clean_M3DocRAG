import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "graph_rerank_page_retrieval_predictions.py"
)
SPEC = importlib.util.spec_from_file_location("graph_rerank_page_retrieval_predictions", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_args(**overrides):
    defaults = {
        "rrf_k": 10.0,
        "doc_seed_weight": 1.0,
        "doc_seed_mode": "rrf",
        "doc_seed_graph_size_reference": 20.0,
        "doc_seed_graph_size_min_mult": 0.25,
        "doc_seed_graph_size_max_mult": 2.0,
        "doc_doc_top_docs": 20,
        "doc_doc_max_edges_per_doc": 8,
        "doc_doc_min_shared_signals": 1,
        "doc_doc_max_signal_doc_matches": 8,
        "doc_doc_min_semantic_similarity": 0.35,
        "doc_doc_semantic_top_terms": 64,
        "doc_doc_edge_weight": 0.1,
        "heading_breadcrumb_min_token_len": 3,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class GraphRerankDocDocAblationTests(unittest.TestCase):
    def test_avg_page_seed_uses_mean_page_seed_per_doc(self) -> None:
        records = {
            "A_page0": MODULE.PageRecord(doc_id="A", page_idx=0),
            "A_page1": MODULE.PageRecord(doc_id="A", page_idx=1),
            "B_page0": MODULE.PageRecord(doc_id="B", page_idx=0),
        }
        page_seed = {"A_page0": 0.2, "A_page1": 0.4, "B_page0": 0.1}
        args = make_args(doc_seed_mode="avg_page_seed", doc_seed_weight=0.5)

        doc_seed, metadata = MODULE.build_doc_seed(
            records=records,
            page_seed=page_seed,
            dense_doc_ranks={"A": 1, "B": 2},
            sparse_doc_ranks={"A": 2},
            source_weights=MODULE.SourceWeights(1.25, 0.75, {}),
            args=args,
        )

        self.assertAlmostEqual(doc_seed["doc::A"], 0.15)
        self.assertAlmostEqual(doc_seed["doc::B"], 0.05)
        self.assertEqual(metadata["doc_seed_mode"], "avg_page_seed")
        self.assertEqual(metadata["doc_seed_node_count"], 2)

    def test_graph_size_adaptive_scales_doc_rrf_seed(self) -> None:
        records = {
            "A_page0": MODULE.PageRecord(doc_id="A", page_idx=0),
            "B_page0": MODULE.PageRecord(doc_id="B", page_idx=0),
        }
        args = make_args(
            doc_seed_mode="graph_size_adaptive",
            doc_seed_weight=1.0,
            doc_seed_graph_size_reference=1.0,
            doc_seed_graph_size_min_mult=0.0,
            doc_seed_graph_size_max_mult=10.0,
        )

        doc_seed, metadata = MODULE.build_doc_seed(
            records=records,
            page_seed={"A_page0": 0.2, "B_page0": 0.1},
            dense_doc_ranks={"A": 1, "B": 2},
            sparse_doc_ranks={},
            source_weights=MODULE.SourceWeights(1.0, 1.0, {}),
            args=args,
        )

        expected_multiplier = math.sqrt(1.0 / 2.0)
        self.assertAlmostEqual(metadata["doc_seed_graph_size_multiplier"], expected_multiplier)
        self.assertAlmostEqual(doc_seed["doc::A"], expected_multiplier / 11.0)
        self.assertAlmostEqual(doc_seed["doc::B"], expected_multiplier / 12.0)

    def test_dense_sparse_agreement_only_connects_docs_in_both_sources(self) -> None:
        args = make_args()

        pair_scores = MODULE.dense_sparse_agreement_doc_doc_scores(
            selected_doc_ids={"A", "B", "C"},
            dense_doc_ranks={"A": 1, "B": 2, "C": 3},
            sparse_doc_ranks={"A": 1, "B": 3},
            source_weights=MODULE.SourceWeights(1.25, 0.75, {}),
            args=args,
        )

        self.assertEqual(set(pair_scores), {("A", "B")})
        self.assertGreater(pair_scores[("A", "B")], 0.0)

    def test_shared_entity_title_topic_scores_shared_signals(self) -> None:
        records = {
            "A_page0": MODULE.PageRecord(doc_id="A", page_idx=0),
            "B_page0": MODULE.PageRecord(doc_id="B", page_idx=0),
            "C_page0": MODULE.PageRecord(doc_id="C", page_idx=0),
        }
        args = make_args(doc_doc_min_shared_signals=1)

        pair_scores, metadata = MODULE.shared_entity_title_topic_doc_doc_scores(
            selected_doc_ids={"A", "B", "C"},
            records=records,
            page_breadcrumbs={
                "A_page0": ["# Transformer Results"],
                "B_page0": ["# Transformer Results"],
                "C_page0": ["# Different Section"],
            },
            page_entities={
                "A_page0": {"openai": ["OpenAI"]},
                "B_page0": {"openai": ["OpenAI"]},
            },
            args=args,
        )

        self.assertIn(("A", "B"), pair_scores)
        self.assertNotIn(("A", "C"), pair_scores)
        self.assertEqual(metadata["doc_doc_shared_pair_count"], 1)


if __name__ == "__main__":
    unittest.main()
