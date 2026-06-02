import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

SCRIPT_PATH = SCRIPTS_DIR / "train_m3docvqa_ltr_page_reranker.py"
SPEC = importlib.util.spec_from_file_location("train_m3docvqa_ltr_page_reranker", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class M3DocVQALtrPageRerankerTests(unittest.TestCase):
    def write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_pick_ltr_indices_keeps_pseudo_page_and_negatives(self) -> None:
        records = [
            {"uid": "d1_page0", "doc_id": "d1", "base_rank": 1},
            {"uid": "d1_page1", "doc_id": "d1", "base_rank": 2},
            {"uid": "d2_page0", "doc_id": "d2", "base_rank": 3},
            {"uid": "d3_page0", "doc_id": "d3", "base_rank": 21},
        ]
        labels = [0, 2, 0, 0]
        args = type(
            "Args",
            (),
            {
                "pseudo_page_label": 2,
                "max_same_doc_pages_per_qid": 2,
                "seed": 13,
                "candidate_top_k": 1000,
                "negatives_per_band": 1,
                "max_negatives_per_qid": 3,
            },
        )()

        indices = MODULE.pick_ltr_indices(records, labels, args)

        self.assertIn(1, indices)
        self.assertGreaterEqual(len(indices), 2)

    def test_sklearn_ltr_promotes_question_matching_page(self) -> None:
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
                    "supporting_context": [{"doc_id": "docA"}],
                },
                {
                    "qid": "q2",
                    "question": "Where was the River Stone award ceremony held?",
                    "metadata": {"gold_page_uids": ["docB_page1"]},
                    "supporting_context": [{"doc_id": "docB"}],
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
                {"doc_id": "docA", "page_idx": 1, "text": "The Blue Lantern Festival was hosted in Lisbon."},
                {"doc_id": "docX", "page_idx": 0, "text": "Unrelated sports statistics."},
                {"doc_id": "docB", "page_idx": 0, "text": "A generic page about transport."},
                {"doc_id": "docB", "page_idx": 1, "text": "The River Stone award ceremony was held in Prague."},
                {"doc_id": "docY", "page_idx": 0, "text": "Unrelated cooking notes."},
            ]
            self.write_jsonl(gold, gold_rows)
            pred.write_text(json.dumps(prediction), encoding="utf-8")
            self.write_jsonl(pages, page_rows)

            old_argv = sys.argv
            try:
                sys.argv = [
                    "train_m3docvqa_ltr_page_reranker.py",
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
                    "--backend",
                    "sklearn",
                    "--candidate-top-k",
                    "3",
                    "--negatives-per-band",
                    "2",
                    "--max-negatives-per-qid",
                    "2",
                    "--max-same-doc-pages-per-qid",
                    "2",
                    "--n-estimators",
                    "30",
                    "--learning-rate",
                    "0.1",
                    "--num-leaves",
                    "8",
                    "--min-data-in-leaf",
                    "1",
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
            self.assertEqual(summary_payload["model_info"]["backend"], "sklearn")


if __name__ == "__main__":
    unittest.main()
