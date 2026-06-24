from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "summarize_m3docvqa_training_objective_ablation.py"
)
SPEC = importlib.util.spec_from_file_location("summary_module", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class TrainingObjectiveSummaryTests(unittest.TestCase):
    def test_build_summary_groups_runs_by_objective(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            model_a = tmp_dir / "weighted.json"
            model_b = tmp_dir / "pairwise.json"
            model_payload = {
                "train_metadata": {
                    "tuning_summary": {
                        "selected_blend_alpha": 0.4,
                        "optimized_metric_value": 0.55,
                        "tune_eval_qid_count": 10,
                    }
                }
            }
            model_a.write_text(json.dumps(model_payload), encoding="utf-8")
            model_b.write_text(json.dumps(model_payload), encoding="utf-8")
            rows = [
                {"label": "GPP", "n_eval": 2, "page@4": 0.6},
                {
                    "label": "weighted_bce_seed_13",
                    "n_eval": 2,
                    "page_mrr": 0.5,
                    "doc_mrr": 0.8,
                    "page@4": 0.7,
                    "doc@4": 0.9,
                    "page@10": 0.8,
                    "page@100": 1.0,
                    "recovered@4": 3,
                    "lost@4": 1,
                    "net@4": 2,
                },
                {
                    "label": "pairwise_ranknet_seed_13",
                    "n_eval": 2,
                    "page_mrr": 0.6,
                    "doc_mrr": 0.8,
                    "page@4": 0.75,
                    "doc@4": 0.9,
                    "page@10": 0.85,
                    "page@100": 1.0,
                    "recovered@4": 4,
                    "lost@4": 1,
                    "net@4": 3,
                },
            ]
            summary = MODULE.build_summary(
                rows,
                [
                    ("weighted_bce", 13, model_a),
                    ("pairwise_ranknet", 13, model_b),
                ],
            )

        self.assertEqual(len(summary["runs"]), 2)
        self.assertEqual(
            [row["objective"] for row in summary["aggregate"]],
            ["pairwise_ranknet", "weighted_bce"],
        )
        rendered = MODULE.render_markdown(summary)
        self.assertIn("pairwise_ranknet", rendered)
        self.assertIn("weighted_bce", rendered)


if __name__ == "__main__":
    unittest.main()
