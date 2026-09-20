import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import diagnose_racs_faiss_replay as diagnostic


def page_rows():
    return [["doc", i, 1.0 / (i + 1)] for i in range(1000)]


def bundle():
    qids = [f"q{i}" for i in range(4)]
    return {"warmup_qids": qids, "measured_qids": ["measured"],
            "questions": {qid: "Which page?" for qid in qids},
            "online": {"nprobe": 4, "baseline_references": {qid: page_rows() for qid in qids}}}


class Tensor:
    """Small array-backed stand-in; unit tests do not claim GPU equivalence."""
    def __init__(self, values):
        self.values = np.asarray(values)
        self.dtype = str(self.values.dtype)

    def float(self):
        return Tensor(self.values.astype(np.float32))

    def numpy(self):
        return self.values

    def tolist(self):
        return self.values.tolist()


def query_meta():
    return {"embeddings": Tensor([[1., 0.], [0., 1.]]), "token_ids": Tensor([1, 2]),
            "raw_tokens": ["a", "b"], "kept_token_indices": [0, 1]}


class InvertedLists:
    def __init__(self):
        self.ids = [np.array([0, 2], dtype=np.int64), np.array([1, 3], dtype=np.int64)]
        self.vectors = [np.array([[1, 0], [3, 0]], dtype=np.float32),
                        np.array([[2, 0], [4, 0]], dtype=np.float32)]
        self.released = []

    def list_size(self, i):
        return len(self.ids[i])

    def get_ids(self, i):
        return self.ids[i]

    def get_codes(self, i):
        return self.vectors[i].view(np.uint8).reshape(-1)

    def release_ids(self, i, pointer):
        self.released.append(("ids", i))

    def release_codes(self, i, pointer):
        self.released.append(("codes", i))


class IndexIVFFlat:
    nlist, d, ntotal, code_size = 2, 2, 4, 8

    def __init__(self):
        self.invlists = InvertedLists()


class DiagnosticTests(unittest.TestCase):
    def test_fixed_four_warmups_only(self):
        value = bundle()
        self.assertEqual(diagnostic.diagnostic_qids(value), [f"q{i}" for i in range(4)])
        for qids in (["q0"], ["q0"] * 4):
            bad = copy.deepcopy(value)
            bad["warmup_qids"] = qids
            with self.assertRaises(ValueError):
                diagnostic.diagnostic_qids(bad)
        value["measured_qids"] = ["q0"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            diagnostic.diagnostic_qids(value)

    def test_missing_reference_and_wrong_budget_fail(self):
        value = bundle()
        value["online"]["nprobe"] = 1
        with self.assertRaisesRegex(ValueError, "nprobe"):
            diagnostic.diagnostic_qids(value)
        value["online"]["nprobe"] = 4
        value["online"]["baseline_references"]["q0"].pop()
        with self.assertRaisesRegex(ValueError, "1000"):
            diagnostic.diagnostic_qids(value)

    def test_strict_order_score_and_candidate_differences(self):
        rows = page_rows()
        self.assertTrue(diagnostic.compare_candidate_rows(rows, rows)["scores_close"])
        changed = copy.deepcopy(rows)
        changed[0][2] += .01
        result = diagnostic.compare_candidate_rows(rows, changed)
        self.assertTrue(result["complete_order_matches"])
        self.assertFalse(result["scores_close"])
        changed[0], changed[1] = changed[1], changed[0]
        result = diagnostic.compare_candidate_rows(rows, changed)
        self.assertEqual(result["shared_pages"], 1000)
        self.assertEqual(result["first_order_mismatch_rank"], 1)
        self.assertFalse(result["complete_order_matches"])
        changed[1] = ["other", 1, .1]
        self.assertEqual(diagnostic.compare_candidate_rows(rows, changed)["shared_pages"], 999)
        result = diagnostic.compare_candidate_rows(rows, rows[:-1])
        self.assertFalse(result["scores_close"])
        self.assertEqual(result["first_order_mismatch_rank"], 1000)

    def test_nonfinite_scores_are_not_accepted(self):
        rows = page_rows()
        bad = copy.deepcopy(rows)
        bad[0][2] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            diagnostic.compare_candidate_rows(rows, bad)

    def test_both_modes_use_one_search_and_production_aggregation(self):
        meta = query_meta()
        query = meta["embeddings"].float().numpy()
        distances = np.ones((2, 1000), dtype=np.float32)
        indices = np.tile(np.arange(1000), (2, 1))
        table = np.ones((1000, 2), dtype=np.float32)
        index = SimpleNamespace(search=mock.Mock(return_value=(distances, indices)))
        seen = []

        def aggregate(q, fixed, uids, values, k, ignore_pad):
            self.assertIs(q, meta)
            self.assertEqual(k, 1000)
            self.assertFalse(ignore_pad)
            actual_distances, actual_indices = fixed.search(query, k)
            self.assertIs(actual_distances, distances)
            self.assertIs(actual_indices, indices)
            seen.append(values)
            return page_rows()

        retriever = SimpleNamespace(index=index, token_table=table, token_uids=["doc_page0"] * 1000,
            rag=SimpleNamespace(_retrieve_pages_from_index_query_meta=aggregate))
        result = diagnostic.candidate_modes(retriever, meta)
        self.assertEqual(index.search.call_count, 1)
        self.assertEqual(set(result), set(diagnostic.SCORE_SOURCES))
        self.assertIs(seen[0], table)
        self.assertIsNone(seen[1])

    def test_fixed_search_rejects_changed_query_or_k(self):
        query = np.ones((1, 2), dtype=np.float32)
        fixed = diagnostic.FixedSearch(query, np.ones((1, 2)), np.ones((1, 2)))
        for values, k in ((query * 2, 2), (query, 1)):
            with self.assertRaises(ValueError):
                fixed.search(values, k)

    def alignment(self, index):
        return diagnostic.sampled_index_alignment(index, np.array([[1, 0], [2, 0], [3, 0], [4, 0]], dtype=np.float32),
            [f"doc_page{i}" for i in range(4)], SimpleNamespace(rev_swig_ptr=lambda pointer, size: pointer))

    def test_sampled_alignment_and_release(self):
        index = IndexIVFFlat()
        result = self.alignment(index)
        self.assertEqual(result["sample_count"], 4)
        self.assertTrue(result["all_sampled_vectors_close"])
        self.assertEqual(len(index.invlists.released), 4)
        index.invlists.ids[0] = np.array([2, 0], dtype=np.int64)
        self.assertFalse(self.alignment(index)["all_sampled_vectors_close"])

    def test_bad_token_id_releases_buffers(self):
        index = IndexIVFFlat()
        index.invlists.ids[0][0] = 999
        with self.assertRaisesRegex(ValueError, "out of range"):
            self.alignment(index)
        self.assertEqual(index.invlists.released, [("codes", 0), ("ids", 0)])

    def test_query_summary_detects_changed_embeddings_and_tokens(self):
        direct = query_meta()
        changed = query_meta()
        changed["embeddings"] = Tensor([[.5, 0.], [0., 1.]])
        result = diagnostic.query_summary(changed, direct)
        self.assertTrue(result["same_token_ids_as_direct_gpu"])
        self.assertFalse(result["values_close_to_direct_gpu"])
        self.assertEqual(result["max_absolute_difference_from_direct_gpu"], .5)
        changed["token_ids"] = Tensor([5, 2])
        result = diagnostic.query_summary(changed, direct)
        self.assertFalse(result["same_token_ids_as_direct_gpu"])
        self.assertNotIn("values_close_to_direct_gpu", result)

    def test_full_orchestration_is_diagnostic_only_and_preserves_inputs(self):
        model = SimpleNamespace(device="cuda", parameters=lambda: iter([SimpleNamespace(dtype="bfloat16")]))
        encoder = SimpleNamespace(model=model, encode_query_with_metadata=lambda *a, **kw: query_meta())
        retriever = SimpleNamespace(index=None, token_table=None, token_uids=None, identity={"fixture": True},
                                    baseline_encoder=encoder, scoring_encoder=encoder)
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
            set_num_threads=lambda n: None, manual_seed=lambda n: None,
            inference_mode=contextlib.nullcontext, no_grad=contextlib.nullcontext)
        accelerator = SimpleNamespace(prepare=lambda m: m, mixed_precision="no", num_processes=1,
                                      device=SimpleNamespace(type="cuda"))
        modules = {"torch": torch, "faiss": SimpleNamespace(),
                   "accelerate": SimpleNamespace(Accelerator=lambda: accelerator),
                   "run_m3docvqa_external_retrieval_qa": SimpleNamespace(make_dataset_args=lambda x: x, M3DocVQADataset=lambda x: x)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bundle.json"
            source.write_text(json.dumps(bundle()))
            original = source.read_bytes()
            output = root / "diagnostic"
            output.mkdir()
            with mock.patch.dict(sys.modules, modules), mock.patch.object(diagnostic.online, "OnlineRetriever", return_value=retriever) as loader, \
                 mock.patch.object(diagnostic, "sampled_index_alignment", return_value={"samples": [], "fixture": True}), \
                 mock.patch.object(diagnostic, "candidate_modes", return_value={k: page_rows() for k in diagnostic.SCORE_SOURCES}), \
                 contextlib.redirect_stdout(io.StringIO()):
                diagnostic.run(source, output)
            self.assertEqual(loader.call_args.args[0]["online"]["faiss_page_score_source"], "embedding")
            report = json.loads((output / "diagnostic.json").read_text())
            self.assertEqual(report["status"], "diagnostic_complete_not_a_runtime_result")
            self.assertTrue(report["no_automatic_configuration_selection"])
            self.assertEqual(len(report["comparisons"]), 24)
            self.assertEqual(len(list(output.glob("query_*.npz"))), 12)
            self.assertEqual(source.read_bytes(), original)
            self.assertFalse((output / "runtime.json").exists())
            results = report["comparisons"]
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                diagnostic.comparison_totals(results[:-1], report["questions"])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                diagnostic.comparison_totals(results + [results[0]], report["questions"])

    def test_cli_refuses_existing_directory_before_running(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bundle.json"
            source.write_text("{}")
            with mock.patch.object(sys, "argv", ["diagnostic", "--bundle", str(source), "--output-dir", str(root)]), \
                 mock.patch.object(diagnostic, "run") as run:
                with self.assertRaises(FileExistsError):
                    diagnostic.main()
                run.assert_not_called()

    def test_cli_failure_is_not_a_runtime_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bundle.json"
            source.write_text("{}")
            output = root / "new"
            with mock.patch.object(sys, "argv", ["diagnostic", "--bundle", str(source), "--output-dir", str(output)]), \
                 mock.patch.object(diagnostic, "run", side_effect=ValueError("test failure")):
                with self.assertRaises(ValueError):
                    diagnostic.main()
            self.assertEqual(json.loads((output / "failure.json").read_text())["status"], "diagnostic_failed_not_a_runtime_result")
            self.assertFalse((output / "runtime.json").exists())


if __name__ == "__main__":
    unittest.main()
