import importlib.util
import json
import sys
import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
