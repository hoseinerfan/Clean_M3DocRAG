import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import validate_racs_graph_replay as replay


def metadata():
    with replay.argv_for_parser():
        result = vars(replay.graph.parse_args())
    return {k: v for k, v in result.items() if k not in replay.IO_ARGS}


class ReplayTests(unittest.TestCase):
    def test_explicit_compatibility_and_saved_values_take_precedence(self):
        saved = metadata()
        saved["dense_weight"] = 1.25
        del saved["ppr_iteration_mode"]
        args, applied = replay.settings_from_metadata(saved)
        self.assertEqual(args.dense_weight, 1.25)
        self.assertEqual(applied, {"ppr_iteration_mode": "fixed"})
        del saved["dense_weight"]
        with self.assertRaisesRegex(ValueError, "without explicit"):
            replay.settings_from_metadata(saved)

    def test_unsupported_extra_work_is_not_silently_omitted(self):
        for key, value in (("neighbor_expansion_window", 1), ("learned_page_prior_jsonl", "prior"),
                           ("query_anchor_evidence_mode", "enabled")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                replay.settings_from_metadata({**metadata(), key: value})

    def test_sample_is_deterministic_and_order_independent(self):
        qids = [f"q{i}" for i in range(30)]
        self.assertEqual(replay.select_qids(qids, 16), replay.select_qids(list(reversed(qids)), 16))
        self.assertEqual(len(set(replay.select_qids(qids, 16))), 16)
        with self.assertRaises(ValueError):
            replay.select_qids(qids, 31)

    def test_rank_score_and_candidate_checks(self):
        rows = [["a", 0, 1.0], ["b", 0, 0.5]]
        self.assertTrue(replay.compare_rows(rows, rows, 2)["scores_close"])
        self.assertFalse(replay.compare_rows(rows, rows[::-1], 2)["complete_order_matches"])
        self.assertFalse(replay.compare_rows(rows, [["a", 0, 2.0], rows[1]], 2)["scores_close"])
        for changed in ([rows[0], rows[0]], [["a", 0, float("nan")], rows[1]], rows[:1]):
            with self.assertRaises(ValueError):
                replay.compare_rows(rows, changed, 2)

    def test_real_graph_function_accepts_materialized_settings(self):
        saved = metadata()
        saved.update(dense_top_pages=3, sparse_top_pages=3, final_top_pages=3,
                     dense_weight=1.25, sparse_weight=0.75)
        args, _ = replay.settings_from_metadata(saved)
        source = {"question": "Which city?", "page_retrieval_results": [["a", 0, 3], ["b", 0, 2], ["a", 1, 1]]}
        rows, _ = replay.graph.build_qid_graph_ranking(qid="q", dense_row=source, sparse_row=source, args=args)
        self.assertTrue(replay.compare_rows(rows, rows, 3)["complete_order_matches"])

    def fixture(self, root):
        rows = [["doc", i, 1 / (i + 1)] for i in range(1000)]
        predictions = {q: {"qid": q, "question": "Question?", "page_retrieval_results": rows} for q in ("q1", "q2")}
        upstream = root / "upstream.json"
        upstream.write_text(json.dumps(predictions))
        pages = root / "pages.jsonl"
        pages.write_text(json.dumps({"doc_id": "doc", "page_idx": 999}) + "\n")
        gold = root / "gold.jsonl"
        gold.write_text(''.join(json.dumps({"qid": q, "answers": ["never pass to ranking"]}) + '\n' for q in predictions))
        config = metadata()
        config.update(dense_prediction_json=str(upstream), sparse_prediction_json=str(upstream),
                      doc_pages_jsonl=str(pages))
        saved = root / "saved.json"
        saved.write_text(json.dumps({q: {**row, "reranker_metadata": config} for q, row in predictions.items()}))
        audit = root / "audit.json"
        audit.write_text(json.dumps({"root": str(root), "gold": {"path": str(gold), "qid_sha256": replay.qid_digest(predictions)},
            "checks": {"expected_2441_gold": True, "four_graph_artifacts_present": True, "graph_qids_match_gold": True},
            "graphs": {label: {"path": str(saved), "first_question_metadata": config} for label in ("main", "aux1", "aux2", "aux3")}}))
        return audit, rows

    def test_replay_handoff_read_only_inputs_and_no_gold_to_ranking(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit, rows = self.fixture(root)
            before = {p: p.read_bytes() for p in root.iterdir()}
            with mock.patch.object(replay.graph, "build_qid_graph_ranking", return_value=(rows, {})) as ranker, contextlib.redirect_stdout(io.StringIO()):
                result = replay.replay(audit, 2)
            self.assertEqual(result["status"], "sample_graph_replay_validated")
            self.assertEqual(ranker.call_count, 8)
            self.assertTrue(all(call.kwargs["gold_row"] is None for call in ranker.call_args_list))
            self.assertEqual(before, {p: p.read_bytes() for p in root.iterdir()})

    def test_mismatch_not_marked_validated_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit, rows = self.fixture(root)
            output = root / "report.json"
            argv = ["replay", "--audit-json", str(audit), "--sample-size", "2", "--output-json", str(output)]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(replay.graph, "build_qid_graph_ranking", return_value=(rows[::-1], {})), contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as error:
                replay.main()
            self.assertEqual(error.exception.code, 2)
            before = output.read_bytes()
            self.assertEqual(json.loads(before)["status"], "sample_graph_replay_mismatch")
            with mock.patch.object(sys, "argv", argv), self.assertRaises(FileExistsError):
                replay.main()
            self.assertEqual(before, output.read_bytes())


if __name__ == "__main__":
    unittest.main()
