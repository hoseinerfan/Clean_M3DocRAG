import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "train_content_aware_pseudo_page_reranker.py"
)
SPEC = importlib.util.spec_from_file_location("train_content_aware_pseudo_page_reranker", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ContentAwarePseudoPageRerankerTests(unittest.TestCase):
    def write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_feature_set_resolver_covers_full_nonempty_group_matrix(self) -> None:
        expected = {
            "rank_only",
            "source_only",
            "structure_only",
            "content_only",
            "rank_source",
            "rank_structure",
            "rank_content",
            "source_structure",
            "source_content",
            "structure_content",
            "rank_source_structure",
            "no_source",
            "no_structure",
            "source_structure_content",
            "all",
        }
        self.assertTrue(expected.issubset(set(MODULE.FEATURE_SET_NAMES)))
        self.assertEqual(len(expected), 15)

        self.assertEqual(MODULE.resolve_feature_names("rank_only"), MODULE.RANK_FEATURES)
        self.assertEqual(MODULE.resolve_feature_names("source_only"), MODULE.SOURCE_FEATURES)
        self.assertEqual(MODULE.resolve_feature_names("structure_only"), MODULE.STRUCTURE_FEATURES)
        self.assertEqual(MODULE.resolve_feature_names("content_only"), MODULE.CONTENT_FEATURES)
        self.assertCountEqual(
            MODULE.resolve_feature_names("rank_source_structure"),
            MODULE.resolve_feature_names("no_content"),
        )
        self.assertCountEqual(
            MODULE.resolve_feature_names("source_structure_content"),
            [name for name in MODULE.FEATURE_NAMES if name not in set(MODULE.RANK_FEATURES)],
        )

    def test_negative_sampling_policies_share_budget_but_select_different_pages(self) -> None:
        records = [
            {"uid": f"doc{idx}_page0", "doc_id": f"doc{idx}", "base_rank": idx}
            for idx in range(1, 101)
        ]
        positive_uids = {"doc2_page0"}
        base_args = {
            "candidate_top_k": 100,
            "negatives_per_band": 2,
            "max_negatives_per_qid": 5,
            "seed": 13,
        }

        hard = MODULE.pick_negative_indices(
            records,
            positive_uids,
            Namespace(**base_args, negative_sampling_strategy="hard_top"),
            rng=MODULE.random.Random(13),
        )
        uniform = MODULE.pick_negative_indices(
            records,
            positive_uids,
            Namespace(**base_args, negative_sampling_strategy="uniform"),
            rng=MODULE.random.Random(13),
        )
        stratified = MODULE.pick_negative_indices(
            records,
            positive_uids,
            Namespace(**base_args, negative_sampling_strategy="rank_stratified"),
            rng=MODULE.random.Random(13),
        )

        self.assertEqual(hard, [0, 2, 3, 4, 5])
        self.assertEqual(len(uniform), 5)
        self.assertEqual(len(stratified), 5)
        self.assertNotEqual(uniform, hard)
        self.assertNotEqual(stratified, hard)
        self.assertTrue(all(records[idx]["uid"] not in positive_uids for idx in uniform))

    def test_build_matrix_respects_evidence_unit_supervision_tiers(self) -> None:
        gold = {
            "q_complete": {
                "qid": "q_complete",
                "question": "Where is the complete evidence?",
                "supporting_context": [{"doc_id": "docA", "doc_part": "text"}],
                "metadata": {
                    "gold_page_uids": ["docA_page0"],
                    "pseudo_gold_qid_supervision_tier": "complete_direct",
                    "pseudo_gold_negative_supervision_scope": "non_support_docs_only",
                },
            },
            "q_positive_only": {
                "qid": "q_positive_only",
                "question": "Where is the partial evidence?",
                "supporting_context": [{"doc_id": "docB", "doc_part": "text"}],
                "metadata": {
                    "gold_page_uids": ["docB_page0"],
                    "pseudo_gold_qid_supervision_tier": "partial_direct_positive_only",
                    "pseudo_gold_negative_supervision_scope": "non_support_docs_only",
                },
            },
            "q_exclude": {
                "qid": "q_exclude",
                "question": "Where is the excluded evidence?",
                "supporting_context": [{"doc_id": "docC", "doc_part": "text"}],
                "metadata": {
                    "gold_page_uids": ["docC_page0"],
                    "pseudo_gold_qid_supervision_tier": "exclude_unlabeled",
                    "pseudo_gold_negative_supervision_scope": "non_support_docs_only",
                },
            },
        }
        base_pred = {
            "q_complete": {
                "qid": "q_complete",
                "page_retrieval_results": [
                    ["docA", 0, 10.0],
                    ["docA", 1, 9.0],
                    ["docNoise", 0, 8.0],
                ],
            },
            "q_positive_only": {
                "qid": "q_positive_only",
                "page_retrieval_results": [
                    ["docB", 0, 10.0],
                    ["docB", 1, 9.0],
                    ["docOther", 0, 8.0],
                ],
            },
            "q_exclude": {
                "qid": "q_exclude",
                "page_retrieval_results": [
                    ["docC", 0, 10.0],
                    ["docOther", 1, 9.0],
                ],
            },
        }
        args = Namespace(
            candidate_top_k=4,
            negatives_per_band=3,
            max_negatives_per_qid=3,
            seed=13,
            active_feature_names=MODULE.FEATURE_NAMES,
            respect_pseudo_supervision_tiers=True,
        )

        _X, y, row_weights, metadata = MODULE.build_matrix(
            gold=gold,
            base_pred=base_pred,
            page_features={},
            source_maps_by_label={},
            args=args,
        )

        self.assertEqual(int(y.sum()), 2)
        self.assertEqual(metadata["positive_count"], 2)
        self.assertEqual(metadata["negative_count"], 1)
        self.assertEqual(metadata["positive_only_qid_count"], 1)
        self.assertEqual(metadata["skipped_exclude_unlabeled_tier"], 1)
        self.assertEqual(metadata["negative_excluded_support_doc_qids"], 1)
        self.assertEqual(metadata["qid_with_negative_rows"], 1)
        self.assertEqual(metadata["qid_without_negative_rows"], 1)
        self.assertEqual(metadata["supervision_tier_counts"]["complete_direct"], 1)
        self.assertEqual(metadata["supervision_tier_counts"]["partial_direct_positive_only"], 1)
        self.assertEqual(metadata["supervision_tier_counts"]["exclude_unlabeled"], 1)
        self.assertTrue((row_weights == 1.0).all())

    def test_build_matrix_can_use_weighted_eu_v4_supervision(self) -> None:
        gold = {
            "q_complete": {
                "qid": "q_complete",
                "question": "Where is the complete evidence?",
                "supporting_context": [{"doc_id": "docA", "doc_part": "text"}],
                "metadata": {
                    "gold_page_uids": ["docA_page0"],
                    "pseudo_gold_page_supervision_tiers": {"docA_page0": "direct"},
                    "pseudo_gold_qid_supervision_tier": "complete_direct",
                    "pseudo_gold_negative_supervision_scope": "non_support_docs_only",
                },
            },
            "q_partial": {
                "qid": "q_partial",
                "question": "Where is the partial image evidence?",
                "supporting_context": [{"doc_id": "docB", "doc_part": "image"}],
                "metadata": {
                    "gold_page_uids": ["docB_page0"],
                    "pseudo_gold_page_supervision_tiers": {"docB_page0": "visual_proxy"},
                    "pseudo_gold_qid_supervision_tier": "partial_hybrid_positive_only",
                    "pseudo_gold_negative_supervision_scope": "non_support_docs_only",
                },
            },
        }
        base_pred = {
            "q_complete": {
                "qid": "q_complete",
                "page_retrieval_results": [
                    ["docA", 0, 10.0],
                    ["docA", 1, 9.5],
                    ["docA", 3, 9.0],
                    ["docNoise", 0, 8.0],
                ],
            },
            "q_partial": {
                "qid": "q_partial",
                "page_retrieval_results": [
                    ["docB", 0, 10.0],
                    ["docB", 3, 9.0],
                    ["docOther", 0, 8.0],
                ],
            },
        }
        args = Namespace(
            candidate_top_k=4,
            negatives_per_band=3,
            max_negatives_per_qid=5,
            seed=13,
            active_feature_names=MODULE.FEATURE_NAMES,
            respect_pseudo_supervision_tiers=True,
            pseudo_supervision_weighting=True,
            hybrid_positive_weight=0.8,
            visual_proxy_positive_weight=0.6,
            partial_positive_weight=0.5,
            include_safe_support_doc_negatives=True,
            safe_support_doc_negative_weight=0.25,
            max_safe_support_doc_negatives_per_qid=2,
        )

        _X, y, row_weights, metadata = MODULE.build_matrix(
            gold=gold,
            base_pred=base_pred,
            page_features={},
            source_maps_by_label={},
            args=args,
        )

        self.assertEqual(int(y.sum()), 2)
        self.assertEqual(metadata["positive_count"], 2)
        self.assertEqual(metadata["negative_count"], 2)
        self.assertEqual(metadata["safe_support_doc_negative_count"], 1)
        self.assertEqual(metadata["positive_only_qid_count"], 1)
        self.assertTrue(metadata["pseudo_supervision_weighting"])
        self.assertTrue(metadata["include_safe_support_doc_negatives"])
        self.assertIn(1.0, row_weights.tolist())
        self.assertIn(0.5, row_weights.tolist())
        self.assertIn(0.25, row_weights.tolist())

    def test_build_matrix_downweights_context_weak_positive_pages(self) -> None:
        gold = {
            "q_context": {
                "qid": "q_context",
                "question": "Which bridge page should be weakly supervised?",
                "supporting_context": [
                    {"doc_id": "docA", "doc_part": "text"},
                    {"doc_id": "docB", "doc_part": "text"},
                ],
                "metadata": {
                    "gold_page_uids": ["docA_page0", "docB_page2"],
                    "pseudo_gold_qid_supervision_tier": "direct_plus_context_weak",
                    "pseudo_gold_page_supervision_tiers": {
                        "docA_page0": "direct",
                        "docB_page2": "context_weak",
                    },
                },
            }
        }
        base_pred = {
            "q_context": {
                "qid": "q_context",
                "page_retrieval_results": [
                    ["docA", 0, 10.0],
                    ["docB", 2, 9.0],
                    ["docNoise", 0, 8.0],
                ],
            }
        }
        args = Namespace(
            candidate_top_k=3,
            negatives_per_band=2,
            max_negatives_per_qid=2,
            seed=13,
            active_feature_names=MODULE.FEATURE_NAMES,
            respect_pseudo_supervision_tiers=True,
            pseudo_supervision_weighting=True,
            context_weak_positive_weight=0.35,
            hybrid_positive_weight=0.8,
            visual_proxy_positive_weight=0.6,
            partial_positive_weight=0.5,
        )

        _X, y, row_weights, metadata = MODULE.build_matrix(
            gold=gold,
            base_pred=base_pred,
            page_features={},
            source_maps_by_label={},
            args=args,
        )

        positive_weights = sorted(round(float(value), 3) for value in row_weights[y == 1])
        self.assertEqual(positive_weights, [0.35, 1.0])
        self.assertEqual(metadata["positive_count"], 2)
        self.assertEqual(metadata["negative_count"], 1)

    def test_build_matrix_can_downweight_medium_score_strict_labels(self) -> None:
        gold = {
            "q_high": {
                "qid": "q_high",
                "question": "Where is the high confidence evidence?",
                "supporting_context": [{"doc_id": "docA", "doc_part": "text"}],
                "metadata": {
                    "gold_page_uids": ["docA_page0"],
                    "pseudo_gold_page_label_scores": {"docA_page0": 18.0},
                },
            },
            "q_medium": {
                "qid": "q_medium",
                "question": "Where is the medium confidence evidence?",
                "supporting_context": [{"doc_id": "docB", "doc_part": "text"}],
                "metadata": {
                    "gold_page_uids": ["docB_page0"],
                    "pseudo_gold_page_label_scores": {"docB_page0": 10.0},
                },
            },
        }
        base_pred = {
            "q_high": {
                "qid": "q_high",
                "page_retrieval_results": [
                    ["docA", 0, 10.0],
                    ["docNoise", 0, 9.0],
                ],
            },
            "q_medium": {
                "qid": "q_medium",
                "page_retrieval_results": [
                    ["docB", 0, 10.0],
                    ["docNoise", 1, 9.0],
                ],
            },
        }
        args = Namespace(
            candidate_top_k=2,
            negatives_per_band=2,
            max_negatives_per_qid=2,
            seed=13,
            active_feature_names=MODULE.FEATURE_NAMES,
            respect_pseudo_supervision_tiers=False,
            pseudo_supervision_weighting=False,
            strict_label_score_weighting=True,
            strict_high_score_threshold=14.0,
            strict_min_score_threshold=8.0,
            strict_medium_positive_weight=0.7,
            strict_low_positive_weight=0.5,
            strict_missing_score_positive_weight=1.0,
        )

        _X, y, row_weights, metadata = MODULE.build_matrix(
            gold=gold,
            base_pred=base_pred,
            page_features={},
            source_maps_by_label={},
            args=args,
        )

        positive_weights = row_weights[y == 1].tolist()
        self.assertEqual([round(value, 3) for value in sorted(positive_weights)], [0.7, 1.0])
        self.assertTrue(metadata["strict_label_score_weighting"])
        self.assertEqual(metadata["strict_score_band_counts"]["high"], 1)
        self.assertEqual(metadata["strict_score_band_counts"]["medium"], 1)

    def test_mlp_scorer_learns_and_round_trips(self) -> None:
        X = MODULE.np.asarray(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [-1.0, 0.0],
                [-0.9, -0.1],
            ],
            dtype=MODULE.np.float32,
        )
        y = MODULE.np.asarray([1.0, 1.0, 0.0, 0.0], dtype=MODULE.np.float32)
        args = Namespace(
            model_type="mlp",
            mlp_hidden_dim=4,
            seed=7,
            positive_weight_cap=20.0,
            epochs=120,
            learning_rate=0.05,
            weight_decay=0.0,
            batch_size=4,
        )

        weights, bias, history = MODULE.train_scorer(X, y, args)
        probs = MODULE.predict_scorer_proba(X, weights, bias)

        self.assertGreater(float(probs[:2].mean()), 0.8)
        self.assertLess(float(probs[2:].mean()), 0.2)
        self.assertGreater(len(history), 0)

        payload = MODULE.scorer_to_json(weights, bias)
        restored_weights, restored_bias = MODULE.scorer_from_model_json(payload)
        restored_probs = MODULE.predict_scorer_proba(X, restored_weights, restored_bias)
        self.assertTrue(MODULE.np.allclose(probs, restored_probs, atol=1e-6))

    def test_content_reranker_promotes_question_matching_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold = root / "gold.jsonl"
            pred = root / "pred.json"
            pages = root / "pages.jsonl"
            model = root / "model.json"
            out_pred = root / "out.prediction.json"
            summary = root / "summary.json"

            gold_rows = [
                {
                    "qid": "q1",
                    "question": "Which city hosted the Blue Lantern Festival?",
                    "metadata": {"gold_page_uids": ["docA_page1"]},
                    "supporting_context": [{"doc_id": "docA", "doc_part": "text"}],
                },
                {
                    "qid": "q2",
                    "question": "Where was the River Stone award ceremony held?",
                    "metadata": {"gold_page_uids": ["docB_page1"]},
                    "supporting_context": [{"doc_id": "docB", "doc_part": "text"}],
                },
            ]
            prediction = {
                "q1": {
                    "qid": "q1",
                    "page_retrieval_results": [
                        ["docA", 0, 10.0],
                        ["docA", 1, 9.0],
                        ["docX", 0, 8.0],
                    ],
                },
                "q2": {
                    "qid": "q2",
                    "page_retrieval_results": [
                        ["docB", 0, 10.0],
                        ["docB", 1, 9.0],
                        ["docY", 0, 8.0],
                    ],
                },
            }
            page_rows = [
                {"doc_id": "docA", "page_idx": 0, "text": "A generic page about weather."},
                {
                    "doc_id": "docA",
                    "page_idx": 1,
                    "text": "The Blue Lantern Festival was hosted in Lisbon city center.",
                },
                {"doc_id": "docX", "page_idx": 0, "text": "Unrelated sports statistics."},
                {"doc_id": "docB", "page_idx": 0, "text": "A generic page about transport."},
                {
                    "doc_id": "docB",
                    "page_idx": 1,
                    "text": "The River Stone award ceremony was held in Prague.",
                },
                {"doc_id": "docY", "page_idx": 0, "text": "Unrelated cooking notes."},
            ]
            self.write_jsonl(gold, gold_rows)
            pred.write_text(json.dumps(prediction), encoding="utf-8")
            self.write_jsonl(pages, page_rows)

            old_argv = sys.argv
            try:
                sys.argv = [
                    "train_content_aware_pseudo_page_reranker.py",
                    "--train-gold",
                    str(gold),
                    "--eval-gold",
                    str(gold),
                    "--train-base-pred",
                    str(pred),
                    "--eval-base-pred",
                    str(pred),
                    "--train-page-text-jsonl",
                    str(pages),
                    "--eval-page-text-jsonl",
                    str(pages),
                    "--candidate-top-k",
                    "3",
                    "--negatives-per-band",
                    "2",
                    "--max-negatives-per-qid",
                    "2",
                    "--epochs",
                    "40",
                    "--learning-rate",
                    "0.05",
                    "--inference-mode",
                    "full_rerank",
                    "--output-model-json",
                    str(model),
                    "--output-prediction-json",
                    str(out_pred),
                    "--output-summary-json",
                    str(summary),
                ]
                MODULE.main()
            finally:
                sys.argv = old_argv

            output = json.loads(out_pred.read_text(encoding="utf-8"))
            self.assertEqual(output["q1"]["page_retrieval_results"][0][:2], ["docA", 1])
            self.assertEqual(output["q2"]["page_retrieval_results"][0][:2], ["docB", 1])
            summary_payload = json.loads(summary.read_text(encoding="utf-8"))
            metrics = {row["label"]: row for row in summary_payload["metrics"]}
            self.assertEqual(metrics["content_aware_pseudo_page_reranker"]["page@1"], 1.0)

    def test_query_adaptive_alpha_writes_adaptive_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold = root / "gold.jsonl"
            pred = root / "pred.json"
            pages = root / "pages.jsonl"
            model = root / "model.json"
            out_pred = root / "out.prediction.json"
            summary = root / "summary.json"

            gold_rows = []
            prediction = {}
            page_rows = []
            for idx in range(6):
                qid = f"q{idx}"
                doc_id = f"doc{idx}"
                topic = f"signal{idx}"
                gold_rows.append(
                    {
                        "qid": qid,
                        "question": f"Where is the {topic} evidence page?",
                        "metadata": {"gold_page_uids": [f"{doc_id}_page1"]},
                        "supporting_context": [{"doc_id": doc_id, "doc_part": "text"}],
                    }
                )
                prediction[qid] = {
                    "qid": qid,
                    "page_retrieval_results": [
                        [doc_id, 0, 10.0],
                        [doc_id, 1, 9.0],
                        [f"noise{idx}", 0, 8.0],
                    ],
                }
                page_rows.extend(
                    [
                        {"doc_id": doc_id, "page_idx": 0, "text": "A generic unrelated page."},
                        {"doc_id": doc_id, "page_idx": 1, "text": f"The {topic} evidence page says Lisbon."},
                        {"doc_id": f"noise{idx}", "page_idx": 0, "text": "Unrelated notes."},
                    ]
                )

            self.write_jsonl(gold, gold_rows)
            pred.write_text(json.dumps(prediction), encoding="utf-8")
            self.write_jsonl(pages, page_rows)

            old_argv = sys.argv
            try:
                sys.argv = [
                    "train_content_aware_pseudo_page_reranker.py",
                    "--train-gold",
                    str(gold),
                    "--eval-gold",
                    str(gold),
                    "--train-base-pred",
                    str(pred),
                    "--eval-base-pred",
                    str(pred),
                    "--train-page-text-jsonl",
                    str(pages),
                    "--eval-page-text-jsonl",
                    str(pages),
                    "--candidate-top-k",
                    "3",
                    "--negatives-per-band",
                    "2",
                    "--max-negatives-per-qid",
                    "2",
                    "--epochs",
                    "20",
                    "--learning-rate",
                    "0.05",
                    "--inference-mode",
                    "blend_rerank",
                    "--auto-tune-blend-alpha",
                    "--query-adaptive-alpha",
                    "--query-alpha-bins",
                    "2",
                    "--tune-fraction",
                    "0.5",
                    "--tune-hit-k",
                    "1",
                    "--tune-blend-alpha-grid",
                    "0.0,0.5,1.0",
                    "--output-model-json",
                    str(model),
                    "--output-prediction-json",
                    str(out_pred),
                    "--output-summary-json",
                    str(summary),
                ]
                MODULE.main()
            finally:
                sys.argv = old_argv

            model_payload = json.loads(model.read_text(encoding="utf-8"))
            self.assertTrue(model_payload["args"]["query_adaptive_alpha"])
            adaptive = model_payload["train_metadata"]["adaptive_alpha_config"]
            self.assertEqual(adaptive["mode"], "confidence_bins")
            self.assertGreaterEqual(len(adaptive["bin_selected_blend_alpha"]), 1)

            output = json.loads(out_pred.read_text(encoding="utf-8"))
            metadata = output["q0"]["reranker_metadata"]["content_aware_pseudo_page_reranker"]
            self.assertTrue(metadata["query_adaptive_alpha"])
            self.assertIn("query_alpha_confidence", metadata)
            self.assertIn("query_alpha_bin", metadata)

    def test_base_aware_alpha_gate_blocks_lossy_alpha(self) -> None:
        records = [
            {"uid": "docA_page0", "doc_id": "docA", "base_rank": 1, "score": 10.0, "learned_score": 0.1},
            {"uid": "docB_page0", "doc_id": "docB", "base_rank": 2, "score": 9.0, "learned_score": 0.9},
        ]
        feature_len = len(MODULE.ALPHA_UTILITY_FEATURE_NAMES)
        config = {
            "mode": "base_aware_alpha_utility_gate",
            "query_alpha_feature_top_k": 2,
            "alpha_grid": [0.0, 1.0],
            "recovery_weights": [0.0] * feature_len,
            "recovery_bias": 0.1,
            "recovery_feature_mean": [0.0] * feature_len,
            "recovery_feature_std": [1.0] * feature_len,
            "loss_weights": [0.0] * feature_len,
            "loss_bias": 1.0,
            "loss_feature_mean": [0.0] * feature_len,
            "loss_feature_std": [1.0] * feature_len,
            "loss_risk_penalty": 2.0,
            "utility_threshold": 0.0,
        }
        alpha, info = MODULE.predict_base_aware_alpha_utility_gate(
            records=records,
            source_maps_by_label={},
            qid="q1",
            adaptive_config=config,
            fallback_alpha=1.0,
        )
        self.assertEqual(alpha, 0.0)
        self.assertEqual(info["mode"], "base_aware_alpha_utility_gate")
        self.assertEqual(info["selection_reason"], "no_safe_positive_predicted_utility")

    def test_learned_query_alpha_writes_regressor_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold = root / "gold.jsonl"
            pred = root / "pred.json"
            pages = root / "pages.jsonl"
            model = root / "model.json"
            out_pred = root / "out.prediction.json"
            summary = root / "summary.json"

            gold_rows = []
            prediction = {}
            page_rows = []
            for idx in range(8):
                qid = f"q{idx}"
                doc_id = f"doc{idx}"
                topic = f"adaptive{idx}"
                gold_rows.append(
                    {
                        "qid": qid,
                        "question": f"Which page mentions {topic} and the target city?",
                        "metadata": {"gold_page_uids": [f"{doc_id}_page1"]},
                        "supporting_context": [{"doc_id": doc_id, "doc_part": "text"}],
                    }
                )
                prediction[qid] = {
                    "qid": qid,
                    "page_retrieval_results": [
                        [doc_id, 0, 10.0],
                        [doc_id, 1, 9.0],
                        [f"noise{idx}", 0, 8.0],
                    ],
                }
                page_rows.extend(
                    [
                        {"doc_id": doc_id, "page_idx": 0, "text": "A generic unrelated page."},
                        {"doc_id": doc_id, "page_idx": 1, "text": f"The {topic} target city is Lisbon."},
                        {"doc_id": f"noise{idx}", "page_idx": 0, "text": "Unrelated notes."},
                    ]
                )

            self.write_jsonl(gold, gold_rows)
            pred.write_text(json.dumps(prediction), encoding="utf-8")
            self.write_jsonl(pages, page_rows)

            old_argv = sys.argv
            try:
                sys.argv = [
                    "train_content_aware_pseudo_page_reranker.py",
                    "--train-gold",
                    str(gold),
                    "--eval-gold",
                    str(gold),
                    "--train-base-pred",
                    str(pred),
                    "--eval-base-pred",
                    str(pred),
                    "--train-page-text-jsonl",
                    str(pages),
                    "--eval-page-text-jsonl",
                    str(pages),
                    "--candidate-top-k",
                    "3",
                    "--negatives-per-band",
                    "2",
                    "--max-negatives-per-qid",
                    "2",
                    "--epochs",
                    "20",
                    "--learning-rate",
                    "0.05",
                    "--inference-mode",
                    "blend_rerank",
                    "--auto-tune-blend-alpha",
                    "--learned-query-alpha",
                    "--query-alpha-feature-top-k",
                    "3",
                    "--query-alpha-ridge",
                    "0.1",
                    "--tune-fraction",
                    "0.5",
                    "--tune-hit-k",
                    "1",
                    "--tune-blend-alpha-grid",
                    "0.0,0.5,1.0",
                    "--output-model-json",
                    str(model),
                    "--output-prediction-json",
                    str(out_pred),
                    "--output-summary-json",
                    str(summary),
                ]
                MODULE.main()
            finally:
                sys.argv = old_argv

            model_payload = json.loads(model.read_text(encoding="utf-8"))
            self.assertTrue(model_payload["args"]["learned_query_alpha"])
            adaptive = model_payload["train_metadata"]["adaptive_alpha_config"]
            self.assertEqual(adaptive["mode"], "learned_query_regressor")
            self.assertEqual(adaptive["query_alpha_feature_top_k"], 3)
            self.assertIn("predicted_alpha_distribution", adaptive)
            self.assertIn("query_alpha_feature_names", adaptive)

            output = json.loads(out_pred.read_text(encoding="utf-8"))
            metadata = output["q0"]["reranker_metadata"]["content_aware_pseudo_page_reranker"]
            self.assertTrue(metadata["learned_query_alpha"])
            self.assertEqual(metadata["query_alpha_mode"], "learned_query_regressor")
            self.assertIn("raw_alpha", metadata["query_alpha_info"])

    def test_learned_alpha_action_writes_action_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold = root / "gold.jsonl"
            pred = root / "pred.json"
            pages = root / "pages.jsonl"
            model = root / "model.json"
            out_pred = root / "out.prediction.json"
            summary = root / "summary.json"

            gold_rows = []
            prediction = {}
            page_rows = []
            for idx in range(8):
                qid = f"q{idx}"
                doc_id = f"doc{idx}"
                topic = f"action{idx}"
                gold_rows.append(
                    {
                        "qid": qid,
                        "question": f"Which page mentions {topic} and the target city?",
                        "metadata": {"gold_page_uids": [f"{doc_id}_page1"]},
                        "supporting_context": [{"doc_id": doc_id, "doc_part": "text"}],
                    }
                )
                prediction[qid] = {
                    "qid": qid,
                    "page_retrieval_results": [
                        [doc_id, 0, 10.0],
                        [doc_id, 1, 9.0],
                        [f"noise{idx}", 0, 8.0],
                    ],
                }
                page_rows.extend(
                    [
                        {"doc_id": doc_id, "page_idx": 0, "text": "A generic unrelated page."},
                        {"doc_id": doc_id, "page_idx": 1, "text": f"The {topic} target city is Lisbon."},
                        {"doc_id": f"noise{idx}", "page_idx": 0, "text": "Unrelated notes."},
                    ]
                )

            self.write_jsonl(gold, gold_rows)
            pred.write_text(json.dumps(prediction), encoding="utf-8")
            self.write_jsonl(pages, page_rows)

            old_argv = sys.argv
            try:
                sys.argv = [
                    "train_content_aware_pseudo_page_reranker.py",
                    "--train-gold",
                    str(gold),
                    "--eval-gold",
                    str(gold),
                    "--train-base-pred",
                    str(pred),
                    "--eval-base-pred",
                    str(pred),
                    "--train-page-text-jsonl",
                    str(pages),
                    "--eval-page-text-jsonl",
                    str(pages),
                    "--candidate-top-k",
                    "3",
                    "--negatives-per-band",
                    "2",
                    "--max-negatives-per-qid",
                    "2",
                    "--epochs",
                    "20",
                    "--learning-rate",
                    "0.05",
                    "--inference-mode",
                    "blend_rerank",
                    "--auto-tune-blend-alpha",
                    "--learned-alpha-action",
                    "--query-alpha-feature-top-k",
                    "3",
                    "--query-alpha-action-epochs",
                    "20",
                    "--query-alpha-action-learning-rate",
                    "0.05",
                    "--tune-fraction",
                    "0.5",
                    "--tune-hit-k",
                    "1",
                    "--tune-blend-alpha-grid",
                    "0.0,0.5,1.0",
                    "--output-model-json",
                    str(model),
                    "--output-prediction-json",
                    str(out_pred),
                    "--output-summary-json",
                    str(summary),
                ]
                MODULE.main()
            finally:
                sys.argv = old_argv

            model_payload = json.loads(model.read_text(encoding="utf-8"))
            self.assertTrue(model_payload["args"]["learned_alpha_action"])
            adaptive = model_payload["train_metadata"]["adaptive_alpha_config"]
            self.assertEqual(adaptive["mode"], "learned_alpha_action")
            self.assertIn("predicted_action_distribution", adaptive)
            self.assertIn("target_action_accuracy", adaptive)

            output = json.loads(out_pred.read_text(encoding="utf-8"))
            metadata = output["q0"]["reranker_metadata"]["content_aware_pseudo_page_reranker"]
            self.assertTrue(metadata["learned_alpha_action"])
            self.assertEqual(metadata["query_alpha_mode"], "learned_alpha_action")
            self.assertIn("action_probability", metadata["query_alpha_info"])

    def test_learned_alpha_utility_gate_writes_safe_utility_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold = root / "gold.jsonl"
            pred = root / "pred.json"
            pages = root / "pages.jsonl"
            model = root / "model.json"
            out_pred = root / "out.prediction.json"
            summary = root / "summary.json"

            gold_rows = []
            prediction = {}
            page_rows = []
            for idx in range(8):
                qid = f"q{idx}"
                doc_id = f"doc{idx}"
                topic = f"utility{idx}"
                gold_rows.append(
                    {
                        "qid": qid,
                        "question": f"Which page mentions {topic} and the target city?",
                        "metadata": {"gold_page_uids": [f"{doc_id}_page1"]},
                        "supporting_context": [{"doc_id": doc_id, "doc_part": "text"}],
                    }
                )
                prediction[qid] = {
                    "qid": qid,
                    "page_retrieval_results": [
                        [doc_id, 0, 10.0],
                        [doc_id, 1, 9.0],
                        [f"noise{idx}", 0, 8.0],
                    ],
                }
                page_rows.extend(
                    [
                        {"doc_id": doc_id, "page_idx": 0, "text": "A generic unrelated page."},
                        {"doc_id": doc_id, "page_idx": 1, "text": f"The {topic} target city is Lisbon."},
                        {"doc_id": f"noise{idx}", "page_idx": 0, "text": "Unrelated notes."},
                    ]
                )

            self.write_jsonl(gold, gold_rows)
            pred.write_text(json.dumps(prediction), encoding="utf-8")
            self.write_jsonl(pages, page_rows)

            old_argv = sys.argv
            try:
                sys.argv = [
                    "train_content_aware_pseudo_page_reranker.py",
                    "--train-gold",
                    str(gold),
                    "--eval-gold",
                    str(gold),
                    "--train-base-pred",
                    str(pred),
                    "--eval-base-pred",
                    str(pred),
                    "--train-page-text-jsonl",
                    str(pages),
                    "--eval-page-text-jsonl",
                    str(pages),
                    "--candidate-top-k",
                    "3",
                    "--negatives-per-band",
                    "2",
                    "--max-negatives-per-qid",
                    "2",
                    "--epochs",
                    "20",
                    "--learning-rate",
                    "0.05",
                    "--inference-mode",
                    "blend_rerank",
                    "--auto-tune-blend-alpha",
                    "--learned-alpha-utility-gate",
                    "--query-alpha-feature-top-k",
                    "3",
                    "--alpha-utility-ridge",
                    "0.1",
                    "--alpha-utility-threshold-grid",
                    "0.0,0.05",
                    "--tune-fraction",
                    "0.5",
                    "--tune-hit-k",
                    "1",
                    "--tune-blend-alpha-grid",
                    "0.0,0.5,1.0",
                    "--output-model-json",
                    str(model),
                    "--output-prediction-json",
                    str(out_pred),
                    "--output-summary-json",
                    str(summary),
                ]
                MODULE.main()
            finally:
                sys.argv = old_argv

            model_payload = json.loads(model.read_text(encoding="utf-8"))
            self.assertTrue(model_payload["args"]["learned_alpha_utility_gate"])
            adaptive = model_payload["adaptive_alpha_config"]
            self.assertEqual(adaptive["mode"], "learned_alpha_utility_gate")
            self.assertEqual(adaptive["query_alpha_feature_top_k"], 3)
            self.assertIn("selected_threshold_score", adaptive)
            self.assertIn("target_utility_distribution", adaptive)

            output = json.loads(out_pred.read_text(encoding="utf-8"))
            metadata = output["q0"]["reranker_metadata"]["content_aware_pseudo_page_reranker"]
            self.assertTrue(metadata["learned_alpha_utility_gate"])
            self.assertEqual(metadata["query_alpha_mode"], "learned_alpha_utility_gate")
            self.assertIn("selected_predicted_utility", metadata["query_alpha_info"])

    def test_alpha_zero_preserves_base_order(self) -> None:
        records = [
            {"uid": "docA_page0", "doc_id": "docA", "page_idx": 0, "base_rank": 1, "learned_score": 0.0},
            {"uid": "docB_page0", "doc_id": "docB", "page_idx": 0, "base_rank": 2, "learned_score": 1.0},
            {"uid": "docC_page0", "doc_id": "docC", "page_idx": 0, "base_rank": 3, "learned_score": 0.5},
        ]
        args = type("Args", (), {"inference_mode": "blend_rerank", "blend_alpha": 0.0})()

        reranked = MODULE.rerank_records(records, args)

        self.assertEqual([row["uid"] for row in reranked], ["docA_page0", "docB_page0", "docC_page0"])

    def test_alpha_utility_targets_include_gain_neutral_and_loss(self) -> None:
        args = type(
            "Args",
            (),
            {
                "inference_mode": "blend_rerank",
                "blend_alpha": 0.0,
                "query_alpha_feature_top_k": 3,
                "alpha_utility_ridge": 0.1,
                "alpha_utility_threshold_grid": "0.0",
            },
        )()
        tune_cases = [
            {
                "qid": "gain",
                "pages_gold": {"docB_page0"},
                "scored_records": [
                    {"uid": "docA_page0", "doc_id": "docA", "page_idx": 0, "base_rank": 1, "score": 3.0, "learned_score": 0.0},
                    {"uid": "docB_page0", "doc_id": "docB", "page_idx": 0, "base_rank": 2, "score": 2.0, "learned_score": 1.0},
                ],
                "confidence": 0.0,
            },
            {
                "qid": "loss",
                "pages_gold": {"docA_page0"},
                "scored_records": [
                    {"uid": "docA_page0", "doc_id": "docA", "page_idx": 0, "base_rank": 1, "score": 3.0, "learned_score": 0.0},
                    {"uid": "docB_page0", "doc_id": "docB", "page_idx": 0, "base_rank": 2, "score": 2.0, "learned_score": 1.0},
                ],
                "confidence": 1.0,
            },
            {
                "qid": "neutral",
                "pages_gold": {"docA_page0"},
                "scored_records": [
                    {"uid": "docA_page0", "doc_id": "docA", "page_idx": 0, "base_rank": 1, "score": 3.0, "learned_score": 1.0},
                    {"uid": "docB_page0", "doc_id": "docB", "page_idx": 0, "base_rank": 2, "score": 2.0, "learned_score": 0.0},
                ],
                "confidence": 1.0,
            },
        ]

        config = MODULE.fit_learned_alpha_utility_gate_config(
            tune_cases=tune_cases,
            alpha_grid=[0.0, 1.0],
            hit_k=1,
            candidate_args=args,
            source_maps_by_label={},
            args=args,
            global_fallback_alpha=0.0,
        )

        distribution = config["target_utility_distribution"]
        self.assertGreaterEqual(distribution.get("1", 0), 1)
        self.assertGreaterEqual(distribution.get("-1", 0), 1)
        self.assertGreaterEqual(distribution.get("0", 0), 1)

    def test_alpha_utility_gate_chooses_zero_for_non_positive_utility(self) -> None:
        records = [
            {"uid": "docA_page0", "doc_id": "docA", "page_idx": 0, "base_rank": 1, "score": 3.0, "learned_score": 0.0},
            {"uid": "docB_page0", "doc_id": "docB", "page_idx": 0, "base_rank": 2, "score": 2.0, "learned_score": 1.0},
        ]
        feature_len = len(MODULE.ALPHA_UTILITY_FEATURE_NAMES)
        config = {
            "mode": "learned_alpha_utility_gate",
            "query_alpha_feature_top_k": 2,
            "alpha_grid": [0.0, 0.5, 1.0],
            "weights": [0.0] * feature_len,
            "bias": -0.1,
            "feature_mean": [0.0] * feature_len,
            "feature_std": [1.0] * feature_len,
            "utility_threshold": 0.0,
        }

        alpha, info = MODULE.predict_learned_alpha_utility_gate(
            records=records,
            source_maps_by_label={},
            qid="q",
            adaptive_config=config,
            fallback_alpha=0.5,
        )

        self.assertEqual(alpha, 0.0)
        self.assertEqual(info["selection_reason"], "no_positive_predicted_utility")

    def test_doc_head_blend_preserves_doc_order_but_swaps_best_page_head(self) -> None:
        records = [
            {"uid": "docA_page0", "doc_id": "docA", "page_idx": 0, "base_rank": 1, "learned_score": 0.1},
            {"uid": "docA_page1", "doc_id": "docA", "page_idx": 1, "base_rank": 2, "learned_score": 0.9},
            {"uid": "docB_page0", "doc_id": "docB", "page_idx": 0, "base_rank": 3, "learned_score": 0.7},
            {"uid": "docC_page0", "doc_id": "docC", "page_idx": 0, "base_rank": 4, "learned_score": 0.6},
        ]
        args = type("Args", (), {"inference_mode": "doc_head_blend", "blend_alpha": 1.0})()

        reranked = MODULE.rerank_records(records, args)

        self.assertEqual(
            [row["uid"] for row in reranked],
            ["docA_page1", "docB_page0", "docC_page0", "docA_page0"],
        )

    def test_doc_slot_blend_only_reorders_same_doc_early_slots(self) -> None:
        records = [
            {"uid": "docA_page0", "doc_id": "docA", "page_idx": 0, "base_rank": 1, "learned_score": 0.1},
            {"uid": "docA_page1", "doc_id": "docA", "page_idx": 1, "base_rank": 2, "learned_score": 0.9},
            {"uid": "docB_page0", "doc_id": "docB", "page_idx": 0, "base_rank": 3, "learned_score": 0.7},
            {"uid": "docA_page2", "doc_id": "docA", "page_idx": 2, "base_rank": 5, "learned_score": 1.0},
        ]
        args = type(
            "Args",
            (),
            {"inference_mode": "doc_slot_blend", "blend_alpha": 1.0, "promotion_rank_max": 3},
        )()

        reranked = MODULE.rerank_records(records, args)

        self.assertEqual(
            [row["uid"] for row in reranked],
            ["docA_page1", "docA_page0", "docB_page0", "docA_page2"],
        )

    def test_parse_alpha_grid_dedupes_and_rejects_invalid_values(self) -> None:
        self.assertEqual(MODULE.parse_alpha_grid("0.1, 0.20,0.1"), [0.1, 0.2])
        with self.assertRaises(ValueError):
            MODULE.parse_alpha_grid("0.1,1.5")

    def test_split_gold_for_tuning_is_deterministic_and_disjoint(self) -> None:
        gold = {f"q{i}": {"qid": f"q{i}"} for i in range(10)}

        fit_a, tune_a = MODULE.split_gold_for_tuning(gold, tune_fraction=0.2, seed=13)
        fit_b, tune_b = MODULE.split_gold_for_tuning(gold, tune_fraction=0.2, seed=13)

        self.assertEqual(tune_a, tune_b)
        self.assertEqual(fit_a, fit_b)
        self.assertEqual(len(tune_a), 2)
        self.assertFalse(set(fit_a) & set(tune_a))

    def test_pairwise_ranknet_learns_question_local_ordering(self) -> None:
        X = MODULE.np.asarray(
            [[1.0, 0.0], [-1.0, 0.0], [0.8, 0.2], [-0.8, -0.2]],
            dtype=MODULE.np.float32,
        )
        y = MODULE.np.asarray([1.0, 0.0, 1.0, 0.0], dtype=MODULE.np.float32)
        query_ids = MODULE.np.asarray(["q1", "q1", "q2", "q2"], dtype=object)
        args = Namespace(
            model_type="logistic",
            training_objective="pairwise_ranknet",
            seed=7,
            epochs=80,
            learning_rate=0.05,
            weight_decay=0.0,
            batch_size=4,
        )

        weights, bias, history = MODULE.train_scorer(
            X,
            y,
            args,
            query_ids=query_ids,
        )
        scores = MODULE.predict_scorer_proba(X, weights, bias)
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[2], scores[3])
        self.assertEqual(history[-1]["pair_count"], 2.0)


if __name__ == "__main__":
    unittest.main()
