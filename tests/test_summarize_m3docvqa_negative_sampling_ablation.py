import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_m3docvqa_negative_sampling_ablation import build_summary, render_markdown


class NegativeSamplingSummaryTest(unittest.TestCase):
    def test_aggregates_by_strategy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_rows = [{"label": "GPP", "page@4": 0.69}]
            model_specs = []
            for strategy, base_score in [("hard_top", 0.75), ("uniform", 0.76)]:
                for seed in (13, 42):
                    model_path = root / f"{strategy}_{seed}.json"
                    model_path.write_text(
                        json.dumps(
                            {
                                "train_metadata": {
                                    "tuning_summary": {
                                        "selected_blend_alpha": 0.4,
                                        "optimized_metric_value": base_score,
                                        "tune_eval_qid_count": 100,
                                    }
                                }
                            }
                        ),
                        encoding="utf-8",
                    )
                    model_specs.append((strategy, seed, model_path))
                    eval_rows.append(
                        {
                            "label": f"{strategy}_seed_{seed}",
                            "n_eval": 200,
                            "page_mrr": 0.56,
                            "doc_mrr": 0.88,
                            "page@4": base_score + seed / 100000,
                            "doc@4": 0.95,
                            "page@10": 0.87,
                            "page@100": 0.96,
                            "recovered@4": 10,
                            "lost@4": 2,
                            "net@4": 8,
                        }
                    )

            summary = build_summary(eval_rows, model_specs)
            self.assertEqual(len(summary["aggregate"]), 2)
            uniform = next(row for row in summary["aggregate"] if row["strategy"] == "uniform")
            self.assertAlmostEqual(uniform["heldout_train_page@4"]["mean"], 0.76)
            self.assertIn("uniform", render_markdown(summary))


if __name__ == "__main__":
    unittest.main()
