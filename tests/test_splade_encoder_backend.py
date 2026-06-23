import importlib.util
import sys
import unittest
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "splade_encoder_backend.py"
SPEC = importlib.util.spec_from_file_location("splade_encoder_backend", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SpladeEncoderBackendTests(unittest.TestCase):
    def test_backend_auto_selects_v3_sparse_encoder(self) -> None:
        self.assertEqual(
            MODULE.resolve_encoder_backend("naver/splade-v3", "auto"),
            "sentence-transformers",
        )
        self.assertEqual(
            MODULE.resolve_encoder_backend(
                "naver/splade-cocondenser-ensembledistil", "auto"
            ),
            "transformers",
        )

    def test_dense_rows_are_pruned_and_sorted(self) -> None:
        embeddings = torch.tensor([[0.0, 0.4, 0.9, 0.2], [0.7, 0.0, 0.8, 0.0]])
        rows = MODULE.embedding_rows_to_terms(
            embeddings,
            topk_terms=2,
            min_weight=0.0,
        )
        self.assertEqual(rows[0][0], [2, 1])
        self.assertEqual(rows[1][0], [2, 0])

    def test_sparse_rows_are_pruned_and_sorted(self) -> None:
        indices = torch.tensor([[0, 0, 0, 1, 1], [1, 2, 3, 0, 2]])
        values = torch.tensor([0.4, 0.9, 0.2, 0.7, 0.8])
        embeddings = torch.sparse_coo_tensor(indices, values, size=(2, 4))
        rows = MODULE.embedding_rows_to_terms(
            embeddings,
            topk_terms=2,
            min_weight=0.0,
        )
        self.assertEqual(rows[0][0], [2, 1])
        self.assertEqual(rows[1][0], [2, 0])


if __name__ == "__main__":
    unittest.main()
