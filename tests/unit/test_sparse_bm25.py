"""BM25 sparse index (app/retrieval/sparse_bm25.py)."""

from __future__ import annotations

import asyncio
import pickle

import pytest

from app.retrieval.sparse_bm25 import (
    BM25Index,
    SparseIndexError,
    SparseIndexNotFoundError,
    SparseIndexRegistry,
    bm25_index_path,
    tokenize,
)
from tests.fakes import make_chunks, make_settings

ALPHA_TEXTS = [
    "Orion Hall seats 120 guests in theatre layout.",
    "Cancellation policy: free cancellation up to 14 days before the event.",
    "In-house vegetarian catering is available.",
    "Parking is two blocks away from Lumen Pavilion.",
    "A signed contract and a deposit confirm any booking.",
]
BETA_TEXTS = [
    "International travel requires director approval.",
    "Cancellation policy: report a cancelled trip within 2 days.",
    "Domestic meal allowance is 40 USD per day.",
    "Receipts must be submitted within 30 days.",
    "A booking after travel starts needs written justification.",
]


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("Cancellation-Policy: 14 DAYS!") == ["cancellation", "policy", "14", "days"]
    assert tokenize("?!") == []


def test_bm25_search():
    index = BM25Index.build("alpha", make_chunks("alpha", ALPHA_TEXTS))
    hits = index.search("cancellation policy", top_k=3)

    assert hits[0].text == ALPHA_TEXTS[1]
    assert hits[0].corpus_id == "alpha" and hits[0].rank == 1
    assert all(h.score > 0 for h in hits)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_bm25_returns_nothing_without_term_overlap_or_tokens():
    index = BM25Index.build("alpha", make_chunks("alpha", ALPHA_TEXTS))
    assert index.search("zebra quantum", top_k=5) == []
    assert index.search("?!", top_k=5) == []
    assert index.search("catering", top_k=0) == []


def test_bm25_top_k_is_respected():
    index = BM25Index.build("alpha", make_chunks("alpha", ALPHA_TEXTS))
    assert len(index.search("hall cancellation catering parking deposit", top_k=2)) == 2


def test_build_rejects_empty_input():
    with pytest.raises(SparseIndexError, match="no chunks"):
        BM25Index.build("alpha", [])
    with pytest.raises(SparseIndexError, match="no indexable tokens"):
        BM25Index.build("alpha", make_chunks("alpha", ["?!"]))


def test_save_load_roundtrip_is_deterministic(tmp_path):
    chunks = make_chunks("alpha", ALPHA_TEXTS)
    path_a, path_b = tmp_path / "a.pkl", tmp_path / "b.pkl"
    BM25Index.build("alpha", chunks, "fp1").save(path_a)
    BM25Index.build("alpha", chunks, "fp1").save(path_b)
    assert path_a.read_bytes() == path_b.read_bytes()

    loaded = BM25Index.load(path_a, "alpha")
    assert loaded.fingerprint == "fp1" and loaded.size == len(ALPHA_TEXTS)
    original = BM25Index.build("alpha", chunks).search("deposit booking", 3)
    assert loaded.search("deposit booking", 3) == original


def test_load_rejects_index_built_for_another_corpus(tmp_path):
    """A beta index renamed to alpha's filename must never be served for alpha."""
    path = bm25_index_path("alpha", tmp_path)
    BM25Index.build("beta", make_chunks("beta", BETA_TEXTS)).save(path)
    with pytest.raises(SparseIndexError, match="belongs to corpus 'beta'"):
        BM25Index.load(path, "alpha")


def test_load_rejects_missing_corrupt_and_foreign_files(tmp_path):
    with pytest.raises(SparseIndexNotFoundError):
        BM25Index.load(tmp_path / "missing.pkl", "alpha")

    garbage = tmp_path / "garbage.pkl"
    garbage.write_bytes(b"not a pickle")
    with pytest.raises(SparseIndexError, match="unreadable"):
        BM25Index.load(garbage, "alpha")

    wrong_format = tmp_path / "wrong.pkl"
    wrong_format.write_bytes(pickle.dumps({"format_version": 999, "corpus_id": "alpha"}))
    with pytest.raises(SparseIndexError, match="unsupported format"):
        BM25Index.load(wrong_format, "alpha")


def _registry(tmp_path) -> SparseIndexRegistry:
    for corpus_id, texts in (("alpha", ALPHA_TEXTS), ("beta", BETA_TEXTS)):
        BM25Index.build(corpus_id, make_chunks(corpus_id, texts)).save(
            bm25_index_path(corpus_id, tmp_path)
        )
    return SparseIndexRegistry(make_settings(tmp_path, tmp_path))


async def test_registry_keeps_per_corpus_indexes_isolated(tmp_path):
    registry = _registry(tmp_path)
    alpha_ids = {c.chunk_id for c in make_chunks("alpha", ALPHA_TEXTS)}
    beta_ids = {c.chunk_id for c in make_chunks("beta", BETA_TEXTS)}

    alpha_hits = await registry.search("alpha", "cancellation policy booking", top_k=20)
    beta_hits = await registry.search("beta", "cancellation policy booking", top_k=20)

    assert alpha_hits and beta_hits  # the shared terms hit in both corpora...
    assert {h.chunk_id for h in alpha_hits} <= alpha_ids  # ...but never across them
    assert {h.chunk_id for h in beta_hits} <= beta_ids
    assert await registry.search("beta", "Orion Hall vegetarian", top_k=20) == []


async def test_registry_missing_corpus_raises_not_found(tmp_path):
    registry = _registry(tmp_path)
    assert registry.is_available("alpha") and not registry.is_available("gamma")
    with pytest.raises(SparseIndexNotFoundError):
        await registry.search("gamma", "anything", top_k=5)


async def test_registry_caches_until_invalidated(tmp_path):
    registry = _registry(tmp_path)
    first = registry.get("alpha")
    assert registry.get("alpha") is first
    registry.invalidate("alpha")
    assert registry.get("alpha") is not first


async def test_registry_search_timeout(tmp_path, monkeypatch):
    registry = _registry(tmp_path)
    index = registry.get("alpha")

    def slow_search(text, top_k):
        import time

        time.sleep(0.5)
        return []

    monkeypatch.setattr(index, "search", slow_search)
    with pytest.raises(asyncio.TimeoutError):
        await registry.search("alpha", "catering", top_k=5, timeout_ms=50)
