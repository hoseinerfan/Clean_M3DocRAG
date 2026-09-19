"""Constructor/loader contracts without requiring torch or downloading weights."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock


def backend_without_torch():
    name = "_racs_splade_loading_test"
    path = Path(__file__).resolve().parents[1] / "scripts/splade_encoder_backend.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {name: module, "torch": SimpleNamespace()}):
        spec.loader.exec_module(module)
    return module


class LocalSpladeLoadingTests(unittest.TestCase):
    def loaders(self):
        return SimpleNamespace(AutoTokenizer=mock.Mock(), AutoModelForMaskedLM=mock.Mock())

    def test_local_only_is_passed_to_both_loaders(self):
        module, transformers = backend_without_torch(), self.loaders()
        with mock.patch.dict(sys.modules, {"transformers": transformers}):
            module.SpladeTextEncoder("/existing/model", "transformers", "cpu", 64,
                tokenizer_name_or_path="/existing/tokenizer", local_files_only=True)
        transformers.AutoTokenizer.from_pretrained.assert_called_once_with("/existing/tokenizer", local_files_only=True)
        transformers.AutoModelForMaskedLM.from_pretrained.assert_called_once_with("/existing/model", local_files_only=True)
        transformers.AutoModelForMaskedLM.from_pretrained.return_value.to.assert_called_once_with("cpu")

    def test_shared_directory_loads_without_hub_identifier(self):
        module, transformers = backend_without_torch(), self.loaders()
        with mock.patch.dict(sys.modules, {"transformers": transformers}):
            module.SpladeTextEncoder("/existing/splade", "transformers", "cpu", 64, local_files_only=True)
        transformers.AutoTokenizer.from_pretrained.assert_called_once_with("/existing/splade", local_files_only=True)
        transformers.AutoModelForMaskedLM.from_pretrained.assert_called_once_with("/existing/splade", local_files_only=True)

    def test_existing_calls_keep_their_original_loading_behavior(self):
        module, transformers = backend_without_torch(), self.loaders()
        with mock.patch.dict(sys.modules, {"transformers": transformers}):
            module.SpladeTextEncoder("naver/original", "transformers", "cuda", 64)
        transformers.AutoTokenizer.from_pretrained.assert_called_once_with("naver/original")
        transformers.AutoModelForMaskedLM.from_pretrained.assert_called_once_with("naver/original")

    def test_local_failure_is_not_retried_with_a_remote_name(self):
        module, transformers = backend_without_torch(), self.loaders()
        transformers.AutoTokenizer.from_pretrained.side_effect = OSError("incomplete local files")
        with mock.patch.dict(sys.modules, {"transformers": transformers}):
            with self.assertRaisesRegex(OSError, "incomplete local files"):
                module.SpladeTextEncoder("/existing/model", "transformers", "cpu", 64, local_files_only=True)
        self.assertEqual(transformers.AutoTokenizer.from_pretrained.call_count, 1)
        transformers.AutoModelForMaskedLM.from_pretrained.assert_not_called()


if __name__ == "__main__":
    unittest.main()
