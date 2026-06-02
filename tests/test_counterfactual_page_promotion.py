import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

SCRIPT_PATH = SCRIPTS_DIR / "train_counterfactual_page_promotion.py"
SPEC = importlib.util.spec_from_file_location("train_counterfactual_page_promotion", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CounterfactualPagePromotionTests(unittest.TestCase):
    def write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def args(self):
        return type(
            "Args",
            (),
            {
                "candidate_top_k": 10,
                "repair_hit_k": 5,
                "insert_rank": 5,
                "promotion_rank_min": 6,
                "promotion_rank_max": 7,
                "max_promotions_per_qid": 1,
                "negatives_per_band": 4,
                "max_negatives_per_qid": 8,
                "already_hit_negatives_per_qid": 4,
                "seed": 13,
            },
        )()

    def test_insert_promotions_preserves_top_four_for_rank_five_repair(self) -> None:
        records = [
            {"uid": f"d{i}_page0", "base_rank": i, "raw": [f"d{i}", 0, 10 - i]}
            for i in range(1, 8)
        ]
        promoted = [records[5]]

        reranked = MODULE.insert_promotions(records, promoted, insert_rank=5)

        self.assertEqual([row["uid"] for row in reranked[:6]], [
            "d1_page0",
            "d2_page0",
            "d3_page0",
            "d4_page0",
            "d6_page0",
            "d5_page0",
        ])

    def test_counterfactual_labels_only_mark_repairing_candidates_positive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold_path = root / "gold.jsonl"
            pages_path = root / "pages.jsonl"
            pred = {
                "q_miss": {
                    "qid": "q_miss",
                    "page_retrieval_results": [
                        ["a", 0, 10.0],
                        ["b", 0, 9.0],
                        ["c", 0, 8.0],
                        ["d", 0, 7.0],
                        ["e", 0, 6.0],
                        ["gold", 0, 5.0],
                        ["neg", 0, 4.0],
                    ],
                },
                "q_hit": {
                    "qid": "q_hit",
                    "page_retrieval_results": [
                        ["gold2", 0, 10.0],
                        ["b2", 0, 9.0],
                        ["c2", 0, 8.0],
                        ["d2", 0, 7.0],
                        ["e2", 0, 6.0],
                        ["gold2", 1, 5.0],
                        ["neg2", 0, 4.0],
                    ],
                },
            }
            gold_rows = [
                {
                    "qid": "q_miss",
                    "question": "Where is the blue festival described?",
                    "metadata": {"gold_page_uids": ["gold_page0"]},
                },
                {
                    "qid": "q_hit",
                    "question": "Where is the green ceremony described?",
                    "metadata": {"gold_page_uids": ["gold2_page0", "gold2_page1"]},
                },
            ]
            page_rows = []
            for row in pred.values():
                for doc_id, page_idx, _score in row["page_retrieval_results"]:
                    page_rows.append(
                        {
                            "doc_id": doc_id,
                            "page_idx": page_idx,
                            "text": f"{doc_id} page {page_idx} evidence text",
                        }
                    )
            self.write_jsonl(gold_path, gold_rows)
            self.write_jsonl(pages_path, page_rows)
            gold = MODULE.ca.load_gold(gold_path)
            page_features = MODULE.ca.load_page_features(pages_path)

            X, y, meta = MODULE.build_counterfactual_matrix(
                gold=gold,
                base_pred=pred,
                page_features=page_features,
                source_maps_by_label={},
                args=self.args(),
            )

            self.assertEqual(X.shape[1], len(MODULE.FEATURE_NAMES))
            self.assertEqual(int(y.sum()), 1)
            self.assertGreater(int(meta["negative_count"]), 0)
            self.assertEqual(int(meta["repairable_miss_qid_count"]), 1)
            self.assertEqual(int(meta["base_hit_qid_count"]), 1)


if __name__ == "__main__":
    unittest.main()
