"""REQ-CORPUS-02: corpus A retrieval never returns corpus B evidence (docs/EVALUATION.md §3).

Both fixture corpora share one Qdrant collection (as in production) and share vocabulary
("booking", "policy", "cancellation"), so isolation is enforced by corpus_id scoping, not by the
corpora happening to use different words.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.events import RetrievalTrigger
from app.models.retrieval_event import RetrievalRequest
from app.retrieval.dense import DenseIndex
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.sparse_bm25 import SparseIndexRegistry, bm25_index_path
from scripts.ingest_corpus import ingest_corpus, manifest_path
from tests.fakes import FakeEmbedder

SHARED_QUERY = "cancellation policy booking"


def chunk_ids_of(corpus_id: str, processed_dir: str) -> set[str]:
    lines = (Path(processed_dir) / f"chunks__{corpus_id}.jsonl").read_text(encoding="utf-8")
    return {json.loads(line)["chunk_id"] for line in lines.splitlines()}


@pytest.fixture
async def two_corpora(test_settings, qdrant_memory):
    embedder = FakeEmbedder()
    for corpus_id in ("alpha", "beta"):
        await ingest_corpus(
            corpus_id, settings=test_settings, embedder=embedder, qdrant_client=qdrant_memory
        )
    dense = DenseIndex(qdrant_memory, embedder, test_settings)
    retriever = HybridRetriever(dense, SparseIndexRegistry(test_settings), test_settings)
    return retriever, dense, test_settings


def req(corpus_id: str, text: str = SHARED_QUERY) -> RetrievalRequest:
    return RetrievalRequest(
        session_id=f"sess_{corpus_id}",
        trace_id="trc",
        sub_query_id="sq",
        text=text,
        corpus_id=corpus_id,
        trigger=RetrievalTrigger.FINAL,
        t_offset_ms=0,
    )


async def test_corpus_isolation(two_corpora):
    retriever, _, settings = two_corpora
    alpha_ids = chunk_ids_of("alpha", settings.processed_dir)
    beta_ids = chunk_ids_of("beta", settings.processed_dir)
    assert alpha_ids.isdisjoint(beta_ids)

    alpha = await retriever.retrieve(req("alpha"))
    beta = await retriever.retrieve(req("beta"))

    for result, own, other, corpus_id in (
        (alpha, alpha_ids, beta_ids, "alpha"),
        (beta, beta_ids, alpha_ids, "beta"),
    ):
        assert result.dense and result.sparse, f"shared query should hit {corpus_id} in both modes"
        hit_ids = {h.chunk_id for h in result.dense + result.sparse}
        assert hit_ids <= own
        assert hit_ids.isdisjoint(other)
        assert {h.corpus_id for h in result.dense + result.sparse} == {corpus_id}


async def test_corpus_specific_terms_do_not_leak(two_corpora):
    retriever, _, settings = two_corpora
    alpha_ids = chunk_ids_of("alpha", settings.processed_dir)

    # "Orion Hall" exists only in alpha; querying beta must not surface any alpha chunk.
    result = await retriever.retrieve(req("beta", "Orion Hall vegetarian catering seats"))
    assert result.sparse == []  # no beta chunk shares these terms
    assert {h.chunk_id for h in result.dense}.isdisjoint(alpha_ids)


async def test_multiple_corpus_indexes_are_independent(two_corpora, qdrant_memory):
    _, dense, settings = two_corpora
    processed = settings.processed_dir
    assert bm25_index_path("alpha", processed).is_file()
    assert bm25_index_path("beta", processed).is_file()
    assert (
        manifest_path("alpha", processed).is_file() and manifest_path("beta", processed).is_file()
    )
    assert await dense.count("alpha") == 9 and await dense.count("beta") == 7

    # Force-reingesting one corpus (which deletes its stale points) never touches the other.
    await ingest_corpus(
        "alpha", settings=settings, embedder=FakeEmbedder(), qdrant_client=qdrant_memory, force=True
    )
    assert await dense.count("beta") == 7
    assert await dense.point_ids("beta") == chunk_ids_of("beta", processed)


async def test_unknown_corpus_returns_no_evidence(two_corpora):
    retriever, _, _ = two_corpora
    result = await retriever.retrieve(req("gamma"))
    assert result.dense == [] and result.sparse == []
    assert result.sparse_error is not None  # no BM25 index: corpus was never ingested
