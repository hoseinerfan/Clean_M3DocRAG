import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_racs_controlled_training as run


class ControlledTrainingTests(unittest.TestCase):
    def fixture(self, root):
        paths = run.input_paths(root / "custom/Clean_M3DocRAG")
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        for split, count in (("train", 21), ("dev", 3)):
            rows = []
            pred = {}
            for i in range(count):
                qid = f"{split}{i}"
                positive = "d_page5" if i % 2 else "d_page4"
                if split == "train" and i == 19:
                    positive = "not_in_pool_page0"
                rows.append({"qid": qid, "question": "Where is Atlanta located?",
                             "gold_page_uids": [] if i == count - 1 else [positive]})
                pred[qid] = {"question": "Where is Atlanta located?",
                             "page_retrieval_results": [["d", j, 10 - j] for j in range(6)]}
            paths[f"{split}_gold"].write_text("".join(json.dumps(r) + "\n" for r in rows))
            paths[f"{split}_base"].write_text(json.dumps(pred))
            pages = [{"doc_id": "d", "page_idx": j,
                      "text": "Atlanta is in Georgia" if j > 3 else "Historical dates"} for j in range(6)]
            paths[f"{split}_text"].write_text("".join(json.dumps(r) + "\n" for r in pages))
            for label in run.SOURCE_VARIANTS:
                paths[f"{split}_source:{label}"].write_text(json.dumps(pred))
        return paths

    def test_prospective_fit_select_refit_and_dev_label_isolation(self):
        counts = {"train_total": 21, "train_labeled": 20, "train_in_pool": 19,
                  "dev_total": 3, "dev_labeled": 2}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.fixture(root)
            outputs = []
            real_load = run.checked_gold
            for index in range(2):
                out = root / f"result{index}"
                out.mkdir()
                def guarded_load(path):
                    if path == paths["dev_gold"]:
                        self.assertTrue((out / "final.model.json").is_file())
                        self.assertTrue((out / "selection.json").is_file())
                    return real_load(path)
                if index:
                    data = [json.loads(s) for s in paths["dev_gold"].read_text().splitlines()]
                    for row in data[:2]:
                        row["gold_page_uids"] = ["d_page0"]
                    paths["dev_gold"].write_text("".join(json.dumps(r) + "\n" for r in data))
                with patch.dict(run.EXPECTED_COUNTS, counts), patch.object(run, "checked_gold", guarded_load), contextlib.redirect_stdout(io.StringIO()):
                    run.run(paths, out)
                report = json.loads((out / "result.json").read_text())
                model = json.loads((out / "final.model.json").read_text())
                split = json.loads((out / "split.json").read_text())
                self.assertEqual(report["status"], "validated_training_and_retrieval")
                self.assertTrue(report["qa_not_yet_run"])
                self.assertTrue(model["train_metadata"]["retrained_after_tuning"])
                self.assertEqual(len(split["fit_qids"]), 16)
                self.assertEqual(len(split["validation_qids"]), 4)
                self.assertFalse(set(split["fit_qids"]) & set(split["validation_qids"]))
                self.assertEqual(model["train_metadata"]["train_qid_with_positive_in_pool"], 19)
                self.assertEqual(len(model["feature_names"]), 30)
                self.assertEqual(len(report["selection"]["alpha_scores"]), 21)
                outputs.append((model, report, json.loads((out / "dev.prediction.json").read_text())))
            # Changing development labels cannot affect fit, alpha, refit, or ranking.
            for key in ("weights", "bias", "mean", "std"):
                np.testing.assert_array_equal(outputs[0][0][key], outputs[1][0][key])
            self.assertEqual(outputs[0][1]["selection"], outputs[1][1]["selection"])
            self.assertEqual(outputs[0][2], outputs[1][2])

    def test_missing_inputs_stop_preflight(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "Required inputs missing"):
                run.preflight(Path(tmp))

    def test_duplicate_gold_and_output_overwrite_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gold.jsonl"
            path.write_text('{"qid":"q"}\n{"qid":"q"}\n')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                run.checked_gold(path)
            with self.assertRaises(FileExistsError):
                run.write_json(path, {})

    def test_missing_auxiliary_questions_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            paths = self.fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError, "Missing train questions"):
                run.load_sources(paths, "train", ["absent"], 1000)

    def test_ties_choose_smallest_alpha(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = run.input_paths(Path(tmp))
            args = run.training_args(paths, Path(tmp))
            scores = [{"blend_alpha": v, "page@4": 0.5} for v in run.ca.parse_alpha_grid(run.GRID)]
            summary = {"alpha_scores": scores, "optimized_metric": "page@4",
                       "tune_eval_qid_count": 4, "selected_blend_alpha": 0.0}
            run.assert_selection(summary, args, 4)
            summary["selected_blend_alpha"] = 0.4
            with self.assertRaisesRegex(ValueError, "tie-break"):
                run.assert_selection(summary, args, 4)


if __name__ == "__main__":
    unittest.main()
