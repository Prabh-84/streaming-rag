"""Embedding interface (app/core/embeddings.py). The real model is exercised only in the opt-in
integration test (tests/integration/test_embedding_model.py)."""

from __future__ import annotations

import pytest

from app.core.embeddings import (
    EMBEDDING_DIM,
    Embedder,
    EmbeddingError,
    SentenceTransformerEmbedder,
    embedding_dimension,
)
from tests.fakes import FakeEmbedder, hash_vector


class _StubModel:
    """Stands in for a loaded SentenceTransformer so no weights are downloaded."""

    def __init__(self, dimension: int = EMBEDDING_DIM) -> None:
        self.dimension = dimension
        self.encode_calls = 0

    def encode(self, texts, **kwargs):
        self.encode_calls += 1
        assert kwargs["normalize_embeddings"] is True
        if isinstance(texts, str):
            return [0.5] * self.dimension
        return [[0.5] * self.dimension for _ in texts]


def test_embedders_satisfy_protocol():
    assert isinstance(FakeEmbedder(), Embedder)
    assert isinstance(SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5"), Embedder)
    assert embedding_dimension() == EMBEDDING_DIM == 384


def test_sentence_transformer_embedder_is_lazy():
    embedder = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")
    assert embedder._model is None  # constructing never downloads or loads weights


def test_embed_is_cached_per_text():
    embedder = SentenceTransformerEmbedder("stub")
    embedder._model = _StubModel()
    first = embedder.embed("venue capacity")
    second = embedder.embed("venue capacity")
    embedder.embed("different text")
    assert first == second
    assert len(first) == EMBEDDING_DIM
    assert embedder._model.encode_calls == 2


def test_embed_batch_returns_one_vector_per_text():
    embedder = SentenceTransformerEmbedder("stub")
    embedder._model = _StubModel()
    vectors = embedder.embed_batch(["a", "b", "c"])
    assert len(vectors) == 3 and all(len(v) == EMBEDDING_DIM for v in vectors)
    assert embedder.embed_batch([]) == []


def test_wrong_dimension_output_raises():
    embedder = SentenceTransformerEmbedder("stub")
    embedder._model = _StubModel(dimension=10)
    with pytest.raises(EmbeddingError, match="384-d"):
        embedder.embed("x")


def test_hash_vector_is_deterministic_normalized_and_overlap_sensitive():
    a, b, c = (
        hash_vector("cancellation policy"),
        hash_vector("cancellation fee"),
        hash_vector("zzz"),
    )
    assert a == hash_vector("cancellation policy")
    assert abs(sum(v * v for v in a) - 1.0) < 1e-9

    def cos(x, y):
        return sum(i * j for i, j in zip(x, y, strict=True))

    assert cos(a, b) > cos(a, c)
