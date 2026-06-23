import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_m3docvqa_capp_seed_stability import build_summary, render_markdown


class SeedStabilitySummaryTest(unittest.TestCase):
    def test_builds_seed_and_aggregate_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_specs = []
            eval_rows = [{"label": "GPP", "page@4": 0.69}]
            for seed, page4 in [(13, 0.77), (42, 0.76), (73, 0.78)]:
                model_path = root / f"seed_{seed}.json"
                model_path.write_text(
                    json.dumps(
                        {
                            "train_metadata": {
                                "tuning_summary": {
                                    "selected_blend_alpha": 0.4,
                                    "optimized_metric_value": 0.58,
                                    "tune_eval_qid_count": 100,
                                }
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                model_specs.append((seed, model_path))
                eval_rows.append(
                    {
                        "label": f"seed_{seed}",
                        "n_eval": 200,
                        "page_mrr": 0.56,
                        "doc_mrr": 0.88,
                        "page@4": page4,
                        "doc@4": 0.95,
                        "page@10": 0.87,
                        "page@100": 0.96,
                        "recovered@4": 10,
                        "lost@4": 2,
                        "net@4": 8,
                    }
                )

            summary = build_summary(eval_rows, model_specs)
            self.assertEqual(summary["aggregate"]["seed_count"], 3)
            self.assertAlmostEqual(summary["aggregate"]["page@4"]["mean"], 0.77)
            self.assertAlmostEqual(summary["aggregate"]["page@4"]["sample_std"], 0.01)
            text = render_markdown(summary)
            self.assertIn("| 13 | 0.40 | 0.5800 |", text)
            self.assertIn("| page@4 | 0.7700 | 0.0100 |", text)


if __name__ == "__main__":
    unittest.main()
