import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import diagnose_racs_exact_native as native
import test_diagnose_racs_exact_replay as prior_tests
import diagnose_racs_exact_replay as exact


class NativeDiagnosticTests(unittest.TestCase):
    def test_module_import_has_no_thread_side_effect_or_heavy_import(self):
        # Fresh interpreter: other tests import the single-thread benchmark.
        code = "import os, sys; before=dict(os.environ); import diagnose_racs_exact_native; assert dict(os.environ)==before; assert 'torch' not in sys.modules; assert 'benchmark_capp_runtime' not in sys.modules"
        env = {**os.environ, "PYTHONPATH": str(native.ROOT / "scripts")}
        subprocess.run([sys.executable, "-B", "-c", code], env=env, check=True)

    def test_argv_is_identical_to_checked_original_runner_arguments(self):
        with tempfile.TemporaryDirectory() as folder:
            inputs, output = prior_tests.ExactDiagnosticTests().prepared(Path(folder))
            self.assertEqual(native.original_argv(inputs, output)[1:], exact.original_argv(inputs, output)[1:])
            native.validate_inputs(inputs)

    def test_input_text_order_and_candidates_are_checked_before_run(self):
        for issue in ("text", "order", "candidates", "duplicate", "annotations"):
            with tempfile.TemporaryDirectory() as folder:
                inputs, _ = prior_tests.ExactDiagnosticTests().prepared(Path(folder))
                if issue == "text":
                    inputs['questions']['q0'] += ' '
                elif issue == "order":
                    inputs['qids'].reverse()
                elif issue == "candidates":
                    inputs['baseline']['q0'][0][2] += 1
                elif issue == "duplicate":
                    inputs['qids'][1] = inputs['qids'][0]
                else:
                    Path(inputs['paths']['empty_query.json']).write_text('{"unexpected": true}')
                with self.assertRaises(ValueError):
                    native.validate_inputs(inputs)

    def torch(self):
        torch = prior_tests.fake_torch()
        torch.get_num_threads = lambda: 24
        torch.__config__ = SimpleNamespace(parallel_info=lambda: "test-parallel-info")
        return torch

    def test_native_thread_count_is_observed_not_set(self):
        torch = self.torch()
        with mock.patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "48"}, clear=True):
            env = native.check_native_environment(torch)
        self.assertEqual(env['cpu_threads'], 24)
        self.assertEqual(env['thread_environment'], {})
        torch.set_num_threads.assert_not_called()

    def test_environment_guard_refuses_overrides_and_oversubscription(self):
        for variables in ({}, {"SLURM_CPUS_PER_TASK": "8"},
                          {"SLURM_CPUS_PER_TASK": "48", "OMP_NUM_THREADS": "1"}):
            with mock.patch.dict(os.environ, variables, clear=True), self.assertRaises(ValueError):
                native.check_native_environment(self.torch())

    def test_original_capture_does_not_import_or_call_benchmark_helpers(self):
        # AST guard on the run boundary supplements fresh-interpreter import QA.
        tree = ast.parse(Path(native.__file__).read_text())
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run_original')
        imports = [alias.name for n in ast.walk(run) if isinstance(n, ast.Import) for alias in n.names]
        self.assertEqual(imports, ['torch', 'run_visual_rerank_batch'])
        self.assertNotIn('set_num_threads', ast.unparse(run))
        self.assertNotIn('exact.', ast.unparse(run))

    def test_native_runner_checks_inputs_preserves_method_and_restores_argv(self):
        with tempfile.TemporaryDirectory() as folder:
            inputs, output = prior_tests.ExactDiagnosticTests().prepared(Path(folder))
            class Encoder:
                model = SimpleNamespace(device='cpu')
                def encode_query_with_metadata(self, query, to_cpu=False, query_token_filter='full'):
                    return {'query': query}
            original = Encoder.encode_query_with_metadata
            def main():
                model = Encoder()
                for qid in inputs['qids']:
                    model.encode_query_with_metadata(inputs['questions'][qid], to_cpu=True)
            argv = sys.argv
            torch = self.torch()
            with mock.patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "48"}, clear=True), \
                 mock.patch.dict(sys.modules, {'torch': torch, 'run_visual_rerank_batch': SimpleNamespace(main=main),
                    'm3docrag.retrieval': SimpleNamespace(ColPaliRetrievalModel=Encoder)}):
                env, queries = native.run_original(inputs, output)
            self.assertIs(sys.argv, argv)
            self.assertIs(Encoder.encode_query_with_metadata, original)
            self.assertEqual(list(queries), inputs['qids'])
            self.assertEqual(env['cpu_threads'], 24)
            torch.set_num_threads.assert_not_called()

    def test_existing_output_is_refused_without_modification(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(sys, 'argv', ['native', '--inputs', 'unused', '--output-dir', folder]):
                with self.assertRaises(FileExistsError):
                    native.main()
            self.assertEqual(list(Path(folder).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
