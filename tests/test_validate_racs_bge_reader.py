import copy
from argparse import Namespace
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bge_validator", ROOT / "scripts/validate_racs_bge_reader.py")
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


class BGEReaderValidationTests(unittest.TestCase):
    def setUp(self):
        self.config = {"model_name_or_path": "/models/bge-reranker-base", "candidate_top_k": 1000,
                       "rerank_top_k": 1000, "blend_alpha": 0.2, "max_length": 512,
                       "max_page_chars": 6000, "processed_qid_count": 2}
        self.gold = [{"qid": "q0"}, {"qid": "q1"}]
        self.base = {r["qid"]: {"qid": r["qid"], "page_retrieval_results":
                     [["doc", p, float(8 - p)] for p in range(8)]} for r in self.gold}
        self.bge = copy.deepcopy(self.base)
        for row in self.bge.values():
            row["page_retrieval_results"].reverse()
            row["reranker_metadata"] = {"standard_cross_encoder_page_reranker": dict(self.config)}

    def validate(self):
        return validator.validate_inputs(self.gold, self.base, self.bge, self.config, 2, 8)

    def test_preserves_saved_order_and_scores_without_filling(self):
        inputs = self.validate()
        self.assertEqual(inputs["q0"]["page_retrieval_results"], self.bge["q0"]["page_retrieval_results"][:4])
        self.assertEqual(set(inputs["q0"]), {"qid", "page_retrieval_results"})

    def test_rejects_wrong_summary_or_per_question_config(self):
        for key, value in (("blend_alpha", 1.0), ("rerank_top_k", 100),
                           ("candidate_top_k", 100), ("model_name_or_path", "other-model")):
            for config in (self.config, self.bge["q0"]["reranker_metadata"]["standard_cross_encoder_page_reranker"]):
                with self.subTest(key=key, config=config):
                    old = config[key]
                    config[key] = value
                    with self.assertRaises(ValueError):
                        self.validate()
                    config[key] = old

    def test_rejects_cohort_and_candidate_changes(self):
        original = copy.deepcopy((self.gold, self.base, self.bge))
        mutations = [lambda: self.gold.append(self.gold[0]),
                     lambda: self.bge.pop("q0"),
                     lambda: self.bge["q0"]["page_retrieval_results"].pop(),
                     lambda: self.bge["q0"]["page_retrieval_results"][0].__setitem__(0, "new-doc"),
                     lambda: self.bge["q0"]["page_retrieval_results"].__setitem__(0, ["doc", 6, 1.0]),
                     lambda: self.bge["q0"]["page_retrieval_results"][0].__setitem__(2, float("nan"))]
        for mutate in mutations:
            self.gold, self.base, self.bge = copy.deepcopy(original)
            mutate()
            with self.assertRaises(ValueError):
                self.validate()

    def test_output_validation_rejects_missing_answers_or_wrong_pages(self):
        inputs = self.validate()
        predictions = {q: {"qid": q, "pred_answer": "answer", "time_qa": 1.0,
                           "selected_page_retrieval_results": r["page_retrieval_results"]}
                       for q, r in inputs.items()}
        scores = {"overall": {"list_em": 30.0, "list_f1": 40.0}}
        self.assertEqual(validator.validate_outputs(inputs, predictions, scores, 2)["questions"], 2)
        for field, value in (("selected_page_retrieval_results", []), ("pred_answer", None),
                             ("qid", "other"), ("time_qa", float("inf"))):
            changed = copy.deepcopy(predictions)
            changed["q0"][field] = value
            with self.assertRaises(ValueError):
                validator.validate_outputs(inputs, changed, scores, 2)
        with self.assertRaises(ValueError):
            validator.validate_outputs(inputs, {"q0": predictions["q0"]}, scores, 2)
        with self.assertRaises(ValueError):
            validator.validate_outputs(inputs, predictions, {"overall": {"list_em": 30, "list_f1": float("nan")}}, 2)

    def test_duplicate_keys_and_exclusive_output(self):
        with self.assertRaises(ValueError):
            json.loads('{"q0": {}, "q0": {}}', object_pairs_hook=validator.unique_object)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            validator.write_new(path, {"ok": True})
            before = validator.fingerprint(path)
            with self.assertRaises(FileExistsError):
                validator.write_new(path, {"ok": False})
            self.assertEqual(before, validator.fingerprint(path))

    def test_prepare_and_check_write_isolated_validated_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "new_run"
            run_dir.mkdir()
            args = Namespace(run_dir=run_dir, gold=root / "gold.jsonl", base=root / "base.json",
                             prediction=root / "bge.json", summary=root / "summary.json")
            args.gold.write_text("\n".join(json.dumps(row) for row in self.gold) + "\n")
            validator.write_new(args.base, self.base)
            validator.write_new(args.prediction, self.bge)
            validator.write_new(args.summary, {**self.config, "gold": str(args.gold),
                                               "base_pred": str(args.base)})
            originals = [validator.fingerprint(p) for p in (args.gold, args.base, args.prediction, args.summary)]
            check_inputs = validator.validate_inputs
            with patch.object(validator, "validate_inputs", side_effect=lambda *a: check_inputs(*a, 2, 8)), redirect_stdout(io.StringIO()):
                validator.prepare(args)
            inputs = validator.read_json(run_dir / "bge_top4.reader_input.json")
            manifest = validator.read_json(run_dir / "run_manifest.json")
            self.assertEqual(manifest["questions"], 2)
            self.assertEqual(manifest["reader_model"], "Qwen2-VL-7B-Instruct")
            predictions = {q: {"qid": q, "pred_answer": "answer", "time_qa": 1.0,
                               "selected_page_retrieval_results": r["page_retrieval_results"]}
                           for q, r in inputs.items()}
            stem = run_dir / "mmqa_dev_bge_qwen2vl_top4"
            validator.write_new(str(stem) + ".prediction.json", predictions)
            validator.write_new(str(stem) + ".eval.json", {"overall": {"list_em": 30, "list_f1": 40}})
            check_outputs = validator.validate_outputs
            argv = ["validator", "check", "--run-dir", str(run_dir)]
            with patch("sys.argv", argv), patch.object(validator, "validate_outputs", side_effect=lambda *a: check_outputs(*a, 2)), redirect_stdout(io.StringIO()):
                validator.main()
            self.assertEqual(validator.read_json(run_dir / "validated_result.json")["questions"], 2)
            self.assertEqual(originals, [validator.fingerprint(p) for p in (args.gold, args.base, args.prediction, args.summary)])
            args.gold.write_text(args.gold.read_text() + "\n")
            with patch("sys.argv", argv), self.assertRaisesRegex(ValueError, "changed during QA"):
                validator.main()


if __name__ == "__main__":
    unittest.main()
