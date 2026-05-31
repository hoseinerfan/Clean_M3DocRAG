import unittest
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from m3docrag.rag.base import RAGModelBase


class FakeRetrievalModel:
    def __init__(self, query_emb):
        self.model = torch.nn.Identity()
        self.query_emb = query_emb

    def encode_query_with_metadata(self, **_kwargs):
        return {
            "embeddings": self.query_emb,
            "raw_tokens": ["query"],
        }


class FakeIndex:
    def __init__(self, distances, indices):
        self.distances = np.asarray(distances, dtype=np.float32)
        self.indices = np.asarray(indices, dtype=np.int64)

    def search(self, query_emb, k):
        assert query_emb.shape[0] == self.distances.shape[0]
        assert k == self.distances.shape[1]
        return self.distances, self.indices


class FaissDistanceRetrievalTests(unittest.TestCase):
    def test_retrieval_can_score_from_faiss_distances_without_token_matrix(self):
        rag = RAGModelBase(retrieval_model=FakeRetrievalModel(torch.ones(1, 2)))
        index = FakeIndex(distances=[[0.1, 0.9, 0.4]], indices=[[0, 1, 2]])
        token2pageuid = ["docA_page0", "docB_page2", "docA_page0"]

        results = rag.retrieve_pages_from_docs(
            query="question",
            docid2embs={},
            index=index,
            token2pageuid=token2pageuid,
            all_token_embeddings=None,
            n_return_pages=3,
        )

        self.assertEqual(results[0], ("docB", 2, 0.9))
        self.assertEqual(results[1], ("docA", 0, 0.4))

    def test_embedding_scores_preserve_original_max_per_page_behavior(self):
        rag = RAGModelBase(retrieval_model=FakeRetrievalModel(torch.tensor([[1.0, 0.0]])))
        index = FakeIndex(distances=[[0.0, 0.0, 0.0]], indices=[[0, 1, 2]])
        token2pageuid = ["docA_page0", "docB_page2", "docA_page0"]
        all_token_embeddings = np.asarray(
            [
                [0.1, 0.0],
                [0.9, 0.0],
                [0.4, 0.0],
            ],
            dtype=np.float32,
        )

        results = rag.retrieve_pages_from_docs(
            query="question",
            docid2embs={},
            index=index,
            token2pageuid=token2pageuid,
            all_token_embeddings=all_token_embeddings,
            n_return_pages=3,
        )

        self.assertEqual(results[0], ("docB", 2, 0.9))
        self.assertEqual(results[1], ("docA", 0, 0.4))


if __name__ == "__main__":
    unittest.main()
