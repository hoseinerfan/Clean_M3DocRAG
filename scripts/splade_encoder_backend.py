from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


EncoderBackend = Literal["transformers", "sentence-transformers"]


def resolve_encoder_backend(model_name_or_path: str, requested: str) -> EncoderBackend:
    if requested not in {"auto", "transformers", "sentence-transformers"}:
        raise ValueError(f"Unsupported SPLADE encoder backend: {requested}")
    if requested != "auto":
        return requested  # type: ignore[return-value]
    if "splade-v3" in model_name_or_path.lower():
        return "sentence-transformers"
    return "transformers"


def _prune_items(
    term_ids: torch.Tensor,
    term_weights: torch.Tensor,
    *,
    topk_terms: int,
    min_weight: float,
) -> tuple[list[int], list[float]]:
    keep = term_weights > float(min_weight)
    term_ids = term_ids[keep]
    term_weights = term_weights[keep]
    if term_ids.numel() == 0:
        return [], []
    if topk_terms > 0 and term_ids.numel() > topk_terms:
        top_values, top_indices = torch.topk(term_weights, k=topk_terms)
        term_ids = term_ids[top_indices]
        term_weights = top_values
    order = torch.argsort(term_weights, descending=True)
    term_ids = term_ids[order]
    term_weights = term_weights[order]
    return term_ids.tolist(), [float(value) for value in term_weights.tolist()]


def embedding_rows_to_terms(
    embeddings: torch.Tensor,
    *,
    topk_terms: int,
    min_weight: float,
) -> list[tuple[list[int], list[float]]]:
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.as_tensor(embeddings)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D SPLADE embedding tensor, got {embeddings.shape}")

    row_count, vocab_size = embeddings.shape
    if embeddings.layout == torch.strided:
        term_ids = torch.arange(vocab_size, dtype=torch.int64)
        return [
            _prune_items(
                term_ids,
                embeddings[row].detach().to(torch.float32).cpu(),
                topk_terms=topk_terms,
                min_weight=min_weight,
            )
            for row in range(row_count)
        ]

    coalesced = embeddings.to_sparse_coo().coalesce().cpu()
    indices = coalesced.indices()
    values = coalesced.values().to(torch.float32)
    rows: list[tuple[list[int], list[float]]] = []
    for row in range(row_count):
        mask = indices[0] == row
        rows.append(
            _prune_items(
                indices[1, mask].to(torch.int64),
                values[mask],
                topk_terms=topk_terms,
                min_weight=min_weight,
            )
        )
    return rows


@dataclass
class SpladeTextEncoder:
    model_name_or_path: str
    backend: EncoderBackend
    device: torch.device
    max_length: int

    def __post_init__(self) -> None:
        self.tokenizer = None
        self.model = None
        self.sparse_encoder = None
        if self.backend == "sentence-transformers":
            try:
                from sentence_transformers import SparseEncoder
            except (ImportError, AttributeError) as exc:
                raise RuntimeError(
                    "SPLADE-v3 requires sentence-transformers>=5 with SparseEncoder support. "
                    "Install it in the active environment before running this backend."
                ) from exc
            self.sparse_encoder = SparseEncoder(
                self.model_name_or_path,
                device=str(self.device),
            )
            self.sparse_encoder.max_seq_length = int(self.max_length)
            self.tokenizer = self.sparse_encoder.tokenizer
            return

        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name_or_path)
        self.model.eval()
        self.model.to(self.device)

    @property
    def vocab_size(self) -> int:
        return int(getattr(self.tokenizer, "vocab_size", 0) or 0)

    def _encode_transformers(self, texts: list[str]) -> torch.Tensor:
        assert self.model is not None and self.tokenizer is not None
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=int(self.max_length),
            return_tensors="pt",
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        outputs = self.model(**batch)
        values = torch.log1p(torch.relu(outputs.logits))
        values = values * batch["attention_mask"].unsqueeze(-1)
        pooled = values.max(dim=1).values
        special_token_ids = list(getattr(self.tokenizer, "all_special_ids", []) or [])
        if special_token_ids:
            pooled[:, special_token_ids] = 0.0
        return pooled.cpu()

    def encode_documents(self, texts: list[str]) -> torch.Tensor:
        if self.backend == "transformers":
            return self._encode_transformers(texts)
        assert self.sparse_encoder is not None
        return self.sparse_encoder.encode_document(
            texts,
            batch_size=len(texts),
            show_progress_bar=False,
            convert_to_tensor=True,
            save_to_cpu=True,
        )

    def encode_queries(self, texts: list[str]) -> torch.Tensor:
        if self.backend == "transformers":
            return self._encode_transformers(texts)
        assert self.sparse_encoder is not None
        return self.sparse_encoder.encode_query(
            texts,
            batch_size=len(texts),
            show_progress_bar=False,
            convert_to_tensor=True,
            save_to_cpu=True,
        )
