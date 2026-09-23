"""Dense index (app/retrieval/dense.py) against Qdrant local in-memory mode."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from qdrant_client import models

from app.core.embeddings import EMBEDDING_DIM
from app.retrieval.dense import (
    COLLECTION_NAME,
    CorpusIsolationError,
    DenseIndex,
    DenseIndexError,
    QueryEmbeddingError,
)
from tests.fakes import FakeEmbedder, hash_vector, make_chunks

ALPHA_TEXTS = [
    "Orion Hall seats 120 guests in theatre layout.",
    "Cancellation policy: free cancellation up to 14 days before the event.",
    "In-house vegetarian catering is available.",
]
BETA_TEXTS = [
    "International travel requires director approval.",
    "Cancellation policy: report a cancelled trip within 2 days.",
]


async def indexed(client, embedder=None, settings=None) -> DenseIndex:
    dense = DenseIndex(client, embedder or FakeEmbedder(), settings)
    await dense.ensure_collection()
    for corpus_id, texts in (("alpha", ALPHA_TEXTS), ("beta", BETA_TEXTS)):
        chunks = make_chunks(corpus_id, texts)
        await dense.upsert_chunks(corpus_id, chunks, [hash_vector(c.text) for c in chunks])
    return dense


async def test_ensure_collection_creates_spec_config_and_is_idempotent(qdrant_memory):
    dense = DenseIndex(qdrant_memory, FakeEmbedder())
    await dense.ensure_collection()
    await dense.ensure_collection()
    info = await qdrant_memory.get_collection(COLLECTION_NAME)
    assert info.config.params.vectors.size == EMBEDDING_DIM
    assert info.config.params.vectors.distance == models.Distance.COSINE


async def test_ensure_collection_rejects_mismatched_existing_collection(qdrant_memory):
    await qdrant_memory.create_collection(
        COLLECTION_NAME, vectors_config=models.VectorParams(size=10, distance=models.Distance.DOT)
    )
    with pytest.raises(DenseIndexError, match="expected 384-d/Cosine"):
        await DenseIndex(qdrant_memory, FakeEmbedder()).ensure_collection()


def test_embedder_with_wrong_dimension_is_rejected(qdrant_memory):
    with pytest.raises(DenseIndexError, match="384-d"):
        DenseIndex(qdrant_memory, FakeEmbedder(dimension=128))


async def test_dense_search(qdrant_memory):
    dense = await indexed(qdrant_memory)
    hits = await dense.search("alpha", "cancellation policy days", top_k=3)

    assert len(hits) == 3
    assert "Cancellation policy" in hits[0].text
    assert [h.rank for h in hits] == [1, 2, 3]
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    top = hits[0]
    assert top.corpus_id == "alpha" and top.doc_id == "doc" and top.section == "§1"
    assert top.chunk_id == make_chunks("alpha", ALPHA_TEXTS)[1].chunk_id


async def test_dense_search_is_scoped_to_corpus(qdrant_memory):
    dense = await indexed(qdrant_memory)
    alpha_ids = {c.chunk_id for c in make_chunks("alpha", ALPHA_TEXTS)}
    beta_ids = {c.chunk_id for c in make_chunks("beta", BETA_TEXTS)}

    alpha_hits = await dense.search("alpha", "cancellation policy", top_k=10)
    beta_hits = await dense.search("beta", "cancellation policy", top_k=10)
    assert {h.chunk_id for h in alpha_hits} == alpha_ids
    assert {h.chunk_id for h in beta_hits} == beta_ids
    assert {h.corpus_id for h in alpha_hits} == {"alpha"}
    assert {h.corpus_id for h in beta_hits} == {"beta"}


async def test_dense_search_empty_query_and_unknown_corpus_return_nothing(qdrant_memory):
    dense = await indexed(qdrant_memory)
    assert await dense.search("alpha", "   ", top_k=5) == []
    assert await dense.search("alpha", "catering", top_k=0) == []
    assert await dense.search("gamma", "catering", top_k=5) == []


async def test_count_and_delete_stale_are_corpus_scoped(qdrant_memory):
    dense = await indexed(qdrant_memory)
    assert await dense.count("alpha") == 3 and await dense.count("beta") == 2

    keep = {make_chunks("alpha", ALPHA_TEXTS)[0].chunk_id}
    assert await dense.delete_stale("alpha", keep) == 2
    assert await dense.point_ids("alpha") == keep
    assert await dense.count("beta") == 2  # the other corpus is untouched


async def test_count_is_zero_when_collection_missing(qdrant_memory):
    assert await DenseIndex(qdrant_memory, FakeEmbedder()).count("alpha") == 0


async def test_upsert_validates_inputs(qdrant_memory):
    dense = DenseIndex(qdrant_memory, FakeEmbedder())
    await dense.ensure_collection()
    chunks = make_chunks("alpha", ALPHA_TEXTS[:1])
    with pytest.raises(DenseIndexError, match="1 chunks but 0 vectors"):
        await dense.upsert_chunks("alpha", chunks, [])
    with pytest.raises(DenseIndexError, match="expected 384"):
        await dense.upsert_chunks("alpha", chunks, [[0.1] * 5])


async def test_query_embedding_retries_once_then_succeeds(qdrant_memory):
    embedder = FakeEmbedder(fail_first=1)
    dense = await indexed(qdrant_memory, embedder)
    hits = await dense.search("alpha", "catering", top_k=1)
    assert hits and embedder.embed_calls == 2


async def test_query_embedding_gives_up_after_one_retry(qdrant_memory):
    embedder = FakeEmbedder(fail_first=2)
    dense = await indexed(qdrant_memory, embedder)
    with pytest.raises(QueryEmbeddingError, match="2 attempts"):
        await dense.search("alpha", "catering", top_k=1)
    assert embedder.embed_calls == 2


async def test_search_timeout_raises(qdrant_memory, monkeypatch):
    dense = await indexed(qdrant_memory)

    async def slow_query(*args, **kwargs):
        await asyncio.sleep(1)

    monkeypatch.setattr(qdrant_memory, "query_points", slow_query)
    with pytest.raises(TimeoutError):
        await dense.search("alpha", "catering", top_k=1, timeout_ms=50)


async def test_isolation_guard_rejects_foreign_payload(qdrant_memory, monkeypatch):
    """If the payload filter were ever bypassed, the post-query check must fail loudly."""
    dense = await indexed(qdrant_memory)
    foreign = SimpleNamespace(
        id="00000000-0000-5000-8000-000000000000",
        score=0.9,
        payload={
            "corpus_id": "beta",
            "doc_id": "d",
            "section": "§1",
            "text": "t",
            "chunk_index": 0,
            "token_count": 1,
        },
    )

    async def leaky_query(*args, **kwargs):
        return SimpleNamespace(points=[foreign])

    monkeypatch.setattr(qdrant_memory, "query_points", leaky_query)
    with pytest.raises(CorpusIsolationError):
        await dense.search("alpha", "catering", top_k=1)
