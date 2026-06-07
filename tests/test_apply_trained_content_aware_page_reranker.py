import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

SCRIPT_PATH = SCRIPTS_DIR / "apply_trained_content_aware_page_reranker.py"
SPEC = importlib.util.spec_from_file_location("apply_trained_content_aware_page_reranker", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ApplyTrainedContentAwarePageRerankerTests(unittest.TestCase):
    def write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_applies_saved_model_to_new_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model.json"
            pred_path = root / "pred.json"
            pages_path = root / "pages.jsonl"
            gold_path = root / "gold.jsonl"
            out_pred = root / "out.prediction.json"
            out_summary = root / "summary.json"

            feature_names = MODULE.ca.FEATURE_NAMES
            weights = [0.0] * len(feature_names)
            weights[feature_names.index("question_token_recall")] = 6.0
            weights[feature_names.index("base_rank_recip")] = -0.2
            model_path.write_text(
                json.dumps(
                    {
                        "feature_names": feature_names,
                        "weights": weights,
                        "bias": 0.0,
                        "mean": [0.0] * len(feature_names),
                        "std": [1.0] * len(feature_names),
                        "args": {
                            "candidate_top_k": 3,
                            "inference_mode": "blend_rerank",
                            "blend_alpha": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            pred_path.write_text(
                json.dumps(
                    {
                        "q1": {
                            "qid": "q1",
                            "page_retrieval_results": [
                                ["docA", 0, 10.0],
                                ["docB", 0, 9.0],
                                ["docC", 0, 8.0],
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.write_jsonl(
                pages_path,
                [
                    {"doc_id": "docA", "page_idx": 0, "text": "generic unrelated text"},
                    {"doc_id": "docB", "page_idx": 0, "text": "blue lantern festival lisbon evidence"},
                    {"doc_id": "docC", "page_idx": 0, "text": "other notes"},
                ],
            )
            self.write_jsonl(
                gold_path,
                [
                    {
                        "qid": "q1",
                        "question": "blue lantern festival lisbon",
                        "supporting_context": [{"doc_id": "docB", "page_idx": 0}],
                    }
                ],
            )

            old_argv = sys.argv
            try:
                sys.argv = [
                    "apply_trained_content_aware_page_reranker.py",
                    "--model-json",
                    str(model_path),
                    "--base-pred",
                    str(pred_path),
                    "--page-text-jsonl",
                    str(pages_path),
                    "--gold",
                    str(gold_path),
                    "--output-prediction-json",
                    str(out_pred),
                    "--output-summary-json",
                    str(out_summary),
                ]
                MODULE.main()
            finally:
                sys.argv = old_argv

            output = json.loads(out_pred.read_text(encoding="utf-8"))
            self.assertEqual(output["q1"]["page_retrieval_results"][0][:2], ["docB", 0])
            summary = json.loads(out_summary.read_text(encoding="utf-8"))
            self.assertEqual(summary["metrics"][1]["page@1"], 1.0)

    def test_applies_saved_alpha_utility_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model.json"
            pred_path = root / "pred.json"
            pages_path = root / "pages.jsonl"
            gold_path = root / "gold.jsonl"
            out_pred = root / "out.prediction.json"
            out_summary = root / "summary.json"

            feature_names = MODULE.ca.FEATURE_NAMES
            weights = [0.0] * len(feature_names)
            weights[feature_names.index("question_token_recall")] = 6.0
            utility_feature_len = len(MODULE.ca.ALPHA_UTILITY_FEATURE_NAMES)
            model_path.write_text(
                json.dumps(
                    {
                        "feature_names": feature_names,
                        "weights": weights,
                        "bias": 0.0,
                        "mean": [0.0] * len(feature_names),
                        "std": [1.0] * len(feature_names),
                        "adaptive_alpha_config": {
                            "mode": "learned_alpha_utility_gate",
                            "query_alpha_feature_top_k": 2,
                            "alpha_grid": [0.0, 1.0],
                            "weights": [0.0] * utility_feature_len,
                            "bias": -0.1,
                            "feature_mean": [0.0] * utility_feature_len,
                            "feature_std": [1.0] * utility_feature_len,
                            "utility_threshold": 0.0,
                        },
                        "args": {
                            "candidate_top_k": 2,
                            "inference_mode": "blend_rerank",
                            "blend_alpha": 1.0,
                            "learned_alpha_utility_gate": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            pred_path.write_text(
                json.dumps(
                    {
                        "q1": {
                            "qid": "q1",
                            "page_retrieval_results": [
                                ["docA", 0, 10.0],
                                ["docB", 0, 9.0],
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.write_jsonl(
                pages_path,
                [
                    {"doc_id": "docA", "page_idx": 0, "text": "generic unrelated text"},
                    {"doc_id": "docB", "page_idx": 0, "text": "blue lantern festival lisbon evidence"},
                ],
            )
            self.write_jsonl(
                gold_path,
                [
                    {
                        "qid": "q1",
                        "question": "blue lantern festival lisbon",
                        "supporting_context": [{"doc_id": "docA", "page_idx": 0}],
                    }
                ],
            )

            old_argv = sys.argv
            try:
                sys.argv = [
                    "apply_trained_content_aware_page_reranker.py",
                    "--model-json",
                    str(model_path),
                    "--base-pred",
                    str(pred_path),
                    "--page-text-jsonl",
                    str(pages_path),
                    "--gold",
                    str(gold_path),
                    "--output-prediction-json",
                    str(out_pred),
                    "--output-summary-json",
                    str(out_summary),
                ]
                MODULE.main()
            finally:
                sys.argv = old_argv

            output = json.loads(out_pred.read_text(encoding="utf-8"))
            self.assertEqual(output["q1"]["page_retrieval_results"][0][:2], ["docA", 0])
            metadata = output["q1"]["trained_content_aware_transfer_metadata"]
            self.assertEqual(metadata["query_alpha_mode"], "learned_alpha_utility_gate")
            self.assertEqual(metadata["blend_alpha"], 0.0)


if __name__ == "__main__":
    unittest.main()
