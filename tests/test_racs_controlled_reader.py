import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import validate_racs_controlled_reader as reader


class ControlledReaderTests(unittest.TestCase):
    def test_training_record_and_alpha_are_pinned(self):
        result = {"status": "validated_training_and_retrieval", "selected_alpha": 0.45}
        model = {"train_metadata": {"retrained_after_tuning": True}, "args": {"blend_alpha": 0.45},
                 "feature_set": "all", "feature_names": list(range(30))}
        selection = {"selected_blend_alpha": 0.45, "optimized_metric": "page@4", "tune_eval_qid_count": 4241}
        reader.validate_training(result, model, selection)
        for change in ({"blend_alpha": 0.4}, {"blend_alpha": 0.5}):
            altered = copy.deepcopy(model)
            altered["args"].update(change)
            with self.assertRaises(ValueError):
                reader.validate_training(result, altered, selection)

    def test_top4_order_preserved_and_question_mismatch_rejected(self):
        pages = [["d", i, 1000-i] for i in range(1000)]
        pred = {"q": {"page_retrieval_results": pages}}
        inputs = reader.reader_inputs(pred, ["q"], expected=1)
        self.assertEqual(inputs["q"]["page_retrieval_results"], pages[:4])
        with self.assertRaises(ValueError):
            reader.reader_inputs(pred, ["other"], expected=1)

    def test_reader_output_must_use_exact_pages(self):
        pages = [["d", i, 10-i] for i in range(4)]
        inputs = {"q": {"page_retrieval_results": pages}}
        output = {"q": {"qid": "q", "pred_answer": "answer", "time_qa": 1.0,
                         "selected_page_retrieval_results": pages}}
        scores = {"overall": {"list_em": 50, "list_f1": 60}}
        reader.validate_outputs(inputs, output, scores, expected_questions=1)
        output["q"]["selected_page_retrieval_results"] = list(reversed(pages))
        with self.assertRaises(ValueError):
            reader.validate_outputs(inputs, output, scores, expected_questions=1)


if __name__ == "__main__":
    unittest.main()
