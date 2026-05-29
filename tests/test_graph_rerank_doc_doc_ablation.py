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
        "doc_seed_page_score_mode": "mean",
        "doc_seed_page_top_k": 3,
        "doc_seed_graph_size_reference": 20.0,
        "doc_seed_graph_size_min_mult": 0.25,
        "doc_seed_graph_size_max_mult": 2.0,
        "doc_doc_top_docs": 20,
        "doc_doc_max_edges_per_doc": 8,
        "doc_doc_min_shared_signals": 1,
        "doc_doc_max_signal_doc_matches": 8,
        "doc_doc_min_semantic_similarity": 0.35,
        "doc_doc_semantic_top_terms": 64,
        "doc_doc_page_embedding_dir": "",
        "doc_doc_embedding_pooling": "page_seed_weighted_mean",
        "doc_doc_embedding_page_top_k": 0,
        "doc_doc_embedding_min_similarity": 0.0,
        "doc_doc_embedding_cache_docs": 128,
        "doc_doc_hyperlink_weight_mode": "log_count",
        "doc_doc_hyperlink_init_mode": "log_count",
        "doc_doc_hyperlink_source_seed_floor": 0.5,
        "doc_doc_hyperlink_source_seed_scale": 0.5,
        "doc_doc_hyperlink_target_support_floor": 0.5,
        "doc_doc_hyperlink_target_support_scale": 0.75,
        "doc_doc_edge_weight": 0.1,
        "ppr_iters": 30,
        "ppr_iteration_mode": "fixed",
        "ppr_graph_size_metric": "nodes",
        "ppr_graph_size_reference": 1000.0,
        "ppr_graph_size_min_iters": 5,
        "ppr_graph_size_max_iters": 80,
        "ppr_convergence_tol": 1e-7,
        "ppr_convergence_min_iters": 5,
        "pdf_hyperlink_edges_jsonl": "links.jsonl",
        "pdf_hyperlink_edge_weight": 0.5,
        "pdf_hyperlink_direction": "source_to_target_doc",
        "pdf_hyperlink_target_mode": "target_doc",
        "pdf_hyperlink_target_pages_per_doc": 1,
        "pdf_hyperlink_target_page_weight_mode": "split",
        "pdf_hyperlink_weight_mode": "uniform",
        "pdf_hyperlink_max_edges_per_source": 0,
        "pdf_hyperlink_source_top_k": 0,
        "pdf_hyperlink_target_doc_top_k": 0,
        "pdf_hyperlink_query_support_weight_mode": "none",
        "heading_breadcrumb_min_token_len": 3,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class FakePageEmbeddingProvider:
    def __init__(self, vectors):
        self.vectors = vectors
        self.error = None

    def page_vector(self, doc_id, page_idx):
        return self.vectors.get((doc_id, page_idx))


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
        self.assertEqual(metadata["doc_seed_page_score_mode"], "mean")
        self.assertEqual(metadata["doc_seed_node_count"], 2)

    def test_page_seed_doc_seed_can_use_max_page_seed_per_doc(self) -> None:
        records = {
            "A_page0": MODULE.PageRecord(doc_id="A", page_idx=0),
            "A_page1": MODULE.PageRecord(doc_id="A", page_idx=1),
            "B_page0": MODULE.PageRecord(doc_id="B", page_idx=0),
        }
        page_seed = {"A_page0": 0.2, "A_page1": 0.4, "B_page0": 0.1}
        args = make_args(
            doc_seed_mode="page_seed",
            doc_seed_page_score_mode="max",
            doc_seed_weight=0.5,
        )

        doc_seed, metadata = MODULE.build_doc_seed(
            records=records,
            page_seed=page_seed,
            dense_doc_ranks={"A": 1, "B": 2},
            sparse_doc_ranks={},
            source_weights=MODULE.SourceWeights(1.0, 1.0, {}),
            args=args,
        )

        self.assertAlmostEqual(doc_seed["doc::A"], 0.2)
        self.assertAlmostEqual(doc_seed["doc::B"], 0.05)
        self.assertEqual(metadata["doc_seed_page_score_mode"], "max")

    def test_page_seed_doc_seed_can_use_topk_mean_page_seed_per_doc(self) -> None:
        records = {
            "A_page0": MODULE.PageRecord(doc_id="A", page_idx=0),
            "A_page1": MODULE.PageRecord(doc_id="A", page_idx=1),
            "A_page2": MODULE.PageRecord(doc_id="A", page_idx=2),
        }
        page_seed = {"A_page0": 0.2, "A_page1": 0.4, "A_page2": 0.8}
        args = make_args(
            doc_seed_mode="page_seed",
            doc_seed_page_score_mode="topk_mean",
            doc_seed_page_top_k=2,
            doc_seed_weight=1.0,
        )

        doc_seed, metadata = MODULE.build_doc_seed(
            records=records,
            page_seed=page_seed,
            dense_doc_ranks={},
            sparse_doc_ranks={},
            source_weights=MODULE.SourceWeights(1.0, 1.0, {}),
            args=args,
        )

        self.assertAlmostEqual(doc_seed["doc::A"], 0.6)
        self.assertEqual(metadata["doc_seed_page_score_mode"], "topk_mean")
        self.assertEqual(metadata["doc_seed_page_top_k"], 2)

    def test_page_seed_doc_seed_can_use_sum_page_seed_per_doc(self) -> None:
        records = {
            "A_page0": MODULE.PageRecord(doc_id="A", page_idx=0),
            "A_page1": MODULE.PageRecord(doc_id="A", page_idx=1),
        }
        page_seed = {"A_page0": 0.2, "A_page1": 0.4}
        args = make_args(
            doc_seed_mode="page_seed",
            doc_seed_page_score_mode="sum",
            doc_seed_weight=0.5,
        )

        doc_seed, metadata = MODULE.build_doc_seed(
            records=records,
            page_seed=page_seed,
            dense_doc_ranks={},
            sparse_doc_ranks={},
            source_weights=MODULE.SourceWeights(1.0, 1.0, {}),
            args=args,
        )

        self.assertAlmostEqual(doc_seed["doc::A"], 0.3)
        self.assertEqual(metadata["doc_seed_page_score_mode"], "sum")

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

    def test_graph_size_adaptive_ppr_iteration_budget_scales_with_nodes(self) -> None:
        graph = {f"n{i}": {} for i in range(25)}
        args = make_args(
            ppr_iters=30,
            ppr_iteration_mode="graph_size",
            ppr_graph_size_metric="nodes",
            ppr_graph_size_reference=100.0,
            ppr_graph_size_min_iters=3,
            ppr_graph_size_max_iters=80,
        )

        iters, metadata = MODULE.effective_ppr_iteration_budget(
            graph=graph,
            seed={"n0": 1.0},
            args=args,
        )

        self.assertEqual(iters, 15)
        self.assertEqual(metadata["ppr_iteration_mode"], "graph_size")
        self.assertEqual(metadata["ppr_graph_size_value"], 25.0)

    def test_ppr_convergence_can_stop_before_max_iterations(self) -> None:
        graph = {
            "A": {"B": 1.0},
            "B": {"A": 1.0},
        }

        _scores, metadata = MODULE.run_ppr(
            graph=graph,
            seed={"A": 1.0},
            restart_prob=0.15,
            iters=50,
            convergence_tol=2.0,
            convergence_min_iters=1,
            return_metadata=True,
        )

        self.assertTrue(metadata["ppr_converged"])
        self.assertLess(metadata["ppr_actual_iters"], 50)

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

    def test_fully_connected_scores_link_every_selected_doc_pair(self) -> None:
        pair_scores = MODULE.fully_connected_doc_doc_scores(
            selected_doc_ids={"A", "B", "C", "D"},
        )

        self.assertEqual(
            set(pair_scores),
            {
                ("A", "B"),
                ("A", "C"),
                ("A", "D"),
                ("B", "C"),
                ("B", "D"),
                ("C", "D"),
            },
        )
        self.assertTrue(all(score == 1.0 for score in pair_scores.values()))

    def test_page_embedding_cosine_links_nearest_document_embeddings(self) -> None:
        records = {
            "A_page0": MODULE.PageRecord(doc_id="A", page_idx=0, dense_rank=1),
            "B_page0": MODULE.PageRecord(doc_id="B", page_idx=0, dense_rank=2),
            "C_page0": MODULE.PageRecord(doc_id="C", page_idx=0, dense_rank=3),
        }
        provider = FakePageEmbeddingProvider(
            {
                ("A", 0): [1.0, 0.0],
                ("B", 0): [0.9, 0.1],
                ("C", 0): [0.0, 1.0],
            }
        )

        pair_scores, metadata = MODULE.page_embedding_cosine_doc_doc_scores(
            selected_doc_ids={"A", "B", "C"},
            records=records,
            page_seed={"A_page0": 1.0, "B_page0": 0.8, "C_page0": 0.7},
            page_embedding_provider=provider,
            args=make_args(
                doc_doc_page_embedding_dir="embeddings",
                doc_doc_embedding_min_similarity=0.8,
                doc_doc_embedding_pooling="mean",
            ),
        )

        self.assertEqual(set(pair_scores), {("A", "B")})
        self.assertGreater(pair_scores[("A", "B")], 0.9)
        self.assertEqual(metadata["doc_doc_embedding_vector_doc_count"], 3)
        self.assertEqual(metadata["doc_doc_embedding_page_vector_count"], 3)
        self.assertEqual(metadata["doc_doc_embedding_pair_count"], 1)

    def test_page_embedding_cosine_can_pool_pages_by_page_seed(self) -> None:
        records = {
            "A_page0": MODULE.PageRecord(doc_id="A", page_idx=0, dense_rank=1),
            "A_page1": MODULE.PageRecord(doc_id="A", page_idx=1, dense_rank=2),
            "B_page0": MODULE.PageRecord(doc_id="B", page_idx=0, dense_rank=3),
            "C_page0": MODULE.PageRecord(doc_id="C", page_idx=0, dense_rank=4),
        }
        provider = FakePageEmbeddingProvider(
            {
                ("A", 0): [1.0, 0.0],
                ("A", 1): [0.0, 1.0],
                ("B", 0): [1.0, 0.0],
                ("C", 0): [0.0, 1.0],
            }
        )

        pair_scores, metadata = MODULE.page_embedding_cosine_doc_doc_scores(
            selected_doc_ids={"A", "B", "C"},
            records=records,
            page_seed={"A_page0": 10.0, "A_page1": 0.1, "B_page0": 1.0, "C_page0": 1.0},
            page_embedding_provider=provider,
            args=make_args(
                doc_doc_page_embedding_dir="embeddings",
                doc_doc_embedding_min_similarity=0.8,
                doc_doc_embedding_pooling="page_seed_weighted_mean",
            ),
        )

        self.assertIn(("A", "B"), pair_scores)
        self.assertNotIn(("A", "C"), pair_scores)
        self.assertEqual(metadata["doc_doc_embedding_page_vector_count"], 4)

    def test_page_embedding_cosine_reports_unavailable_without_embedding_dir(self) -> None:
        pair_scores, metadata = MODULE.page_embedding_cosine_doc_doc_scores(
            selected_doc_ids={"A", "B"},
            records={},
            page_seed={},
            page_embedding_provider=None,
            args=make_args(doc_doc_page_embedding_dir=""),
        )

        self.assertEqual(pair_scores, {})
        self.assertFalse(metadata["doc_doc_embedding_available"])
        self.assertEqual(metadata["doc_doc_embedding_missing_doc_count"], 2)

    def test_hyperlink_citation_target_support_adapts_pair_weights(self) -> None:
        records = {
            "A_page0": MODULE.PageRecord(doc_id="A", page_idx=0),
            "B_page0": MODULE.PageRecord(doc_id="B", page_idx=0),
            "C_page0": MODULE.PageRecord(doc_id="C", page_idx=0),
        }
        graph = MODULE.PdfHyperlinkGraph(
            by_source_page={
                "A_page0": [
                    MODULE.PdfHyperlinkEdge(
                        source_page_uid="A_page0",
                        target_doc_id="B",
                        raw_link_count=1,
                    ),
                    MODULE.PdfHyperlinkEdge(
                        source_page_uid="A_page0",
                        target_doc_id="C",
                        raw_link_count=1,
                    ),
                ]
            },
            edge_count=2,
            source_page_count=1,
            target_doc_count=2,
        )
        args = make_args(
            doc_doc_hyperlink_weight_mode="target_support",
            doc_doc_hyperlink_target_support_floor=0.0,
            doc_doc_hyperlink_target_support_scale=1.0,
        )

        pair_scores, metadata = MODULE.hyperlink_citation_doc_doc_scores(
            selected_doc_ids={"A", "B", "C"},
            records=records,
            page_seed={"A_page0": 1.0},
            dense_doc_ranks={"B": 1, "C": 10},
            sparse_doc_ranks={},
            source_weights=MODULE.SourceWeights(1.0, 1.0, {}),
            pdf_hyperlink_graph=graph,
            args=args,
        )

        self.assertGreater(pair_scores[("A", "B")], pair_scores[("A", "C")])
        self.assertEqual(metadata["doc_doc_hyperlink_weight_mode"], "target_support")
        self.assertEqual(metadata["doc_doc_hyperlink_pair_count"], 2)

    def test_hyperlink_citation_log_count_preserves_raw_link_order(self) -> None:
        records = {
            "A_page0": MODULE.PageRecord(doc_id="A", page_idx=0),
            "B_page0": MODULE.PageRecord(doc_id="B", page_idx=0),
            "C_page0": MODULE.PageRecord(doc_id="C", page_idx=0),
        }
        graph = MODULE.PdfHyperlinkGraph(
            by_source_page={
                "A_page0": [
                    MODULE.PdfHyperlinkEdge(
                        source_page_uid="A_page0",
                        target_doc_id="B",
                        raw_link_count=1,
                    ),
                    MODULE.PdfHyperlinkEdge(
                        source_page_uid="A_page0",
                        target_doc_id="C",
                        raw_link_count=5,
                    ),
                ]
            },
            edge_count=2,
            source_page_count=1,
            target_doc_count=2,
        )

        pair_scores, metadata = MODULE.hyperlink_citation_doc_doc_scores(
            selected_doc_ids={"A", "B", "C"},
            records=records,
            page_seed={"A_page0": 1.0},
            dense_doc_ranks={"B": 1, "C": 10},
            sparse_doc_ranks={},
            source_weights=MODULE.SourceWeights(1.0, 1.0, {}),
            pdf_hyperlink_graph=graph,
            args=make_args(),
        )

        self.assertGreater(pair_scores[("A", "C")], pair_scores[("A", "B")])
        self.assertAlmostEqual(pair_scores[("A", "C")], math.log1p(5))
        self.assertEqual(metadata["doc_doc_hyperlink_init_mode"], "log_count")
        self.assertEqual(metadata["doc_doc_hyperlink_weight_mode"], "log_count")

    def test_hyperlink_citation_uniform_init_ignores_raw_link_count(self) -> None:
        records = {
            "A_page0": MODULE.PageRecord(doc_id="A", page_idx=0),
            "B_page0": MODULE.PageRecord(doc_id="B", page_idx=0),
            "C_page0": MODULE.PageRecord(doc_id="C", page_idx=0),
        }
        graph = MODULE.PdfHyperlinkGraph(
            by_source_page={
                "A_page0": [
                    MODULE.PdfHyperlinkEdge(
                        source_page_uid="A_page0",
                        target_doc_id="B",
                        raw_link_count=1,
                    ),
                    MODULE.PdfHyperlinkEdge(
                        source_page_uid="A_page0",
                        target_doc_id="C",
                        raw_link_count=5,
                    ),
                ]
            },
            edge_count=2,
            source_page_count=1,
            target_doc_count=2,
        )

        pair_scores, metadata = MODULE.hyperlink_citation_doc_doc_scores(
            selected_doc_ids={"A", "B", "C"},
            records=records,
            page_seed={"A_page0": 1.0},
            dense_doc_ranks={"B": 1, "C": 10},
            sparse_doc_ranks={},
            source_weights=MODULE.SourceWeights(1.0, 1.0, {}),
            pdf_hyperlink_graph=graph,
            args=make_args(doc_doc_hyperlink_init_mode="uniform"),
        )

        self.assertAlmostEqual(pair_scores[("A", "B")], pair_scores[("A", "C")])
        self.assertEqual(metadata["doc_doc_hyperlink_init_mode"], "uniform")

    def test_hyperlink_initial_score_modes(self) -> None:
        self.assertAlmostEqual(
            MODULE.hyperlink_citation_initial_score(4, "uniform"),
            1.0,
        )
        self.assertAlmostEqual(
            MODULE.hyperlink_citation_initial_score(4, "sqrt_count"),
            2.0,
        )
        self.assertAlmostEqual(
            MODULE.hyperlink_citation_initial_score(4, "raw_count"),
            4.0,
        )
        self.assertAlmostEqual(
            MODULE.hyperlink_citation_initial_score(4, "log_count"),
            math.log1p(4),
        )

    def test_pdf_hyperlink_edges_default_to_target_doc_node(self) -> None:
        records = {
            "A_page0": MODULE.PageRecord(doc_id="A", page_idx=0, dense_rank=1),
            "B_page0": MODULE.PageRecord(doc_id="B", page_idx=0, dense_rank=2),
        }
        hyperlink_graph = MODULE.PdfHyperlinkGraph(
            by_source_page={
                "A_page0": [
                    MODULE.PdfHyperlinkEdge(
                        source_page_uid="A_page0",
                        target_doc_id="B",
                        raw_link_count=1,
                    )
                ]
            },
            edge_count=1,
            source_page_count=1,
            target_doc_count=1,
        )
        graph: dict[str, dict[str, float]] = {}

        metadata = MODULE.add_pdf_hyperlink_edges(
            graph=graph,
            records=records,
            pdf_hyperlink_graph=hyperlink_graph,
            args=make_args(),
        )

        self.assertAlmostEqual(graph["A_page0"]["doc::B"], 0.5)
        self.assertEqual(metadata["pdf_hyperlink_target_mode"], "target_doc")
        self.assertEqual(metadata["pdf_hyperlink_target_page_count"], 0)
        self.assertEqual(metadata["pdf_hyperlink_target_doc_count"], 1)

    def test_pdf_hyperlink_edges_can_target_best_retrieved_pages(self) -> None:
        records = {
            "A_page0": MODULE.PageRecord(doc_id="A", page_idx=0, dense_rank=1),
            "B_page0": MODULE.PageRecord(doc_id="B", page_idx=0, dense_rank=3),
            "B_page1": MODULE.PageRecord(doc_id="B", page_idx=1, sparse_rank=2),
            "B_page2": MODULE.PageRecord(doc_id="B", page_idx=2, dense_rank=10),
        }
        hyperlink_graph = MODULE.PdfHyperlinkGraph(
            by_source_page={
                "A_page0": [
                    MODULE.PdfHyperlinkEdge(
                        source_page_uid="A_page0",
                        target_doc_id="B",
                        raw_link_count=1,
                    )
                ]
            },
            edge_count=1,
            source_page_count=1,
            target_doc_count=1,
        )
        graph: dict[str, dict[str, float]] = {}

        metadata = MODULE.add_pdf_hyperlink_edges(
            graph=graph,
            records=records,
            pdf_hyperlink_graph=hyperlink_graph,
            args=make_args(
                pdf_hyperlink_target_mode="target_pages",
                pdf_hyperlink_target_pages_per_doc=2,
                pdf_hyperlink_target_page_weight_mode="split",
            ),
        )

        self.assertNotIn("doc::B", graph["A_page0"])
        self.assertAlmostEqual(graph["A_page0"]["B_page1"], 0.25)
        self.assertAlmostEqual(graph["A_page0"]["B_page0"], 0.25)
        self.assertNotIn("B_page2", graph["A_page0"])
        self.assertEqual(metadata["pdf_hyperlink_edge_count_directed"], 2)
        self.assertEqual(metadata["pdf_hyperlink_target_page_count"], 2)
        self.assertEqual(metadata["pdf_hyperlink_target_doc_count"], 1)

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

    def test_max1_per_doc_final_selection_promotes_distinct_docs(self) -> None:
        ranked = [
            (MODULE.PageRecord(doc_id="A", page_idx=0, dense_rank=1), 1.00, 1.0, 0.0, 0.0),
            (MODULE.PageRecord(doc_id="A", page_idx=1, dense_rank=2), 0.99, 0.9, 0.0, 0.0),
            (MODULE.PageRecord(doc_id="B", page_idx=0, dense_rank=3), 0.98, 0.8, 0.0, 0.0),
            (MODULE.PageRecord(doc_id="C", page_idx=0, dense_rank=4), 0.97, 0.7, 0.0, 0.0),
        ]
        args = make_args(
            final_selection_mode="max1_per_doc_then_fill",
            final_selection_top_k=3,
            final_selection_candidate_pool=4,
        )

        selected, metadata = MODULE.apply_final_selection_policy(ranked, args)

        self.assertEqual([item[0].page_uid for item in selected[:3]], ["A_page0", "B_page0", "C_page0"])
        self.assertTrue(metadata["final_selection_reordered"])
        self.assertEqual(metadata["final_selection_selected_doc_count"], 3)


if __name__ == "__main__":
    unittest.main()
