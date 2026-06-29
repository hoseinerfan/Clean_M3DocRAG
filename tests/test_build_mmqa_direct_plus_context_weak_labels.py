import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_mmqa_direct_plus_context_weak_labels.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_mmqa_direct_plus_context_weak_labels", SCRIPT_PATH
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BuildMMQADirectPlusContextWeakLabelsTests(unittest.TestCase):
    def write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_adds_one_ranked_bridge_page_without_labeling_unlabeled_qids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold = root / "gold.jsonl"
            direct_labels = root / "direct_labels.jsonl"
            direct_augmented = root / "direct_augmented.jsonl"
            base_pred = root / "base.prediction.json"
            tables = root / "tables.jsonl"
            out_labels = root / "out_labels.jsonl"
            out_augmented = root / "out_augmented.jsonl"
            summary = root / "summary.json"

            self.write_jsonl(
                gold,
                [
                    {
                        "qid": "q1",
                        "question": "Which page has the answer?",
                        "answers": [
                            {
                                "answer": "Lisbon",
                                "text_instances": [{"doc_id": "docA", "text": "Lisbon"}],
                                "table_indices": [],
                                "image_instances": [],
                            }
                        ],
                        "supporting_context": [
                            {"doc_id": "docA", "doc_part": "text"},
                            {"doc_id": "docB", "doc_part": "text"},
                        ],
                        "metadata": {"type": "Compose(TextQ,TextQ)"},
                    },
                    {
                        "qid": "q2",
                        "question": "This question has no direct label.",
                        "answers": [
                            {
                                "answer": "Prague",
                                "text_instances": [{"doc_id": "docC", "text": "Prague"}],
                                "table_indices": [],
                                "image_instances": [],
                            }
                        ],
                        "supporting_context": [
                            {"doc_id": "docC", "doc_part": "text"},
                            {"doc_id": "docD", "doc_part": "text"},
                        ],
                        "metadata": {"type": "Compose(TextQ,TextQ)"},
                    },
                ],
            )
            self.write_jsonl(
                direct_labels,
                [
                    {
                        "qid": "q1",
                        "pseudo_gold_page_uids": ["docA_page0"],
                        "pseudo_gold_pages": [
                            {
                                "page_uid": "docA_page0",
                                "doc_id": "docA",
                                "page_idx": 0,
                                "score": 1.0,
                                "supervision_tier": "direct",
                            }
                        ],
                    }
                ],
            )
            self.write_jsonl(
                direct_augmented,
                [
                    {
                        "qid": "q1",
                        "question": "Which page has the answer?",
                        "supporting_context": [
                            {"doc_id": "docA", "doc_part": "text"},
                            {"doc_id": "docB", "doc_part": "text"},
                        ],
                        "metadata": {
                            "gold_page_uids": ["docA_page0"],
                            "pseudo_gold_page_uids": ["docA_page0"],
                            "pseudo_gold_page_supervision_tiers": {"docA_page0": "direct"},
                        },
                    }
                ],
            )
            base_pred.write_text(
                json.dumps(
                    {
                        "q1": {
                            "qid": "q1",
                            "page_retrieval_results": [
                                ["docB", 2, 9.0],
                                ["docB", 3, 8.5],
                                ["docA", 0, 8.0],
                            ],
                        },
                        "q2": {
                            "qid": "q2",
                            "page_retrieval_results": [["docD", 1, 7.0]],
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.write_jsonl(tables, [])

            old_argv = sys.argv
            try:
                sys.argv = [
                    "build_mmqa_direct_plus_context_weak_labels.py",
                    "--gold",
                    str(gold),
                    "--direct-labels-jsonl",
                    str(direct_labels),
                    "--direct-augmented-gold-jsonl",
                    str(direct_augmented),
                    "--base-prediction-json",
                    str(base_pred),
                    "--mmqa-tables-jsonl",
                    str(tables),
                    "--output-labels-jsonl",
                    str(out_labels),
                    "--output-augmented-gold-jsonl",
                    str(out_augmented),
                    "--output-summary-json",
                    str(summary),
                    "--context-score",
                    "0.25",
                ]
                MODULE.main()
            finally:
                sys.argv = old_argv

            labels = {
                row["qid"]: row
                for row in (json.loads(line) for line in out_labels.read_text(encoding="utf-8").splitlines())
            }
            self.assertEqual(labels["q1"]["pseudo_gold_page_uids"], ["docA_page0", "docB_page2"])
            self.assertEqual(labels["q1"]["context_weak_page_uids"], ["docB_page2"])
            weak_page = labels["q1"]["pseudo_gold_pages"][1]
            self.assertEqual(weak_page["supervision_tier"], "context_weak")
            self.assertEqual(weak_page["base_rank"], 1)
            self.assertEqual(labels["q2"].get("pseudo_gold_page_uids"), None)

            augmented = {
                row["qid"]: row
                for row in (
                    json.loads(line) for line in out_augmented.read_text(encoding="utf-8").splitlines()
                )
            }
            meta = augmented["q1"]["metadata"]
            self.assertEqual(meta["gold_page_uids"], ["docA_page0", "docB_page2"])
            self.assertEqual(meta["pseudo_gold_context_weak_page_uids"], ["docB_page2"])
            self.assertEqual(meta["pseudo_gold_page_supervision_tiers"]["docB_page2"], "context_weak")

            payload = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(payload["counts"]["bridge_context_doc_refs"], 1)
            self.assertEqual(payload["counts"]["context_augmented_qids"], 1)
            self.assertEqual(payload["counts"]["context_weak_page_count"], 1)
            self.assertEqual(payload["counts"]["unlabeled_qids_skipped_for_context"], 1)


if __name__ == "__main__":
    unittest.main()
