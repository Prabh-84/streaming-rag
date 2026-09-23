"""Dense indexing/retrieval against a real Qdrant server (the docker-compose `qdrant` service).

Skipped when no server answers at QDRANT_TEST_URL (default http://localhost:6333). Start one with
`docker compose -f docker/docker-compose.yml up -d qdrant`. Uses a throwaway collection so it
never touches the real `corpus_chunks` data.
"""

from __future__ import annotations

import os
import urllib.request
import uuid

import pytest
from qdrant_client import AsyncQdrantClient

from app.retrieval.dense import DenseIndex
from scripts.ingest_corpus import ingest_corpus
from tests.fakes import FakeEmbedder

QDRANT_TEST_URL = os.environ.get("QDRANT_TEST_URL", "http://localhost:6333")


def _server_up() -> bool:
    try:
        with urllib.request.urlopen(f"{QDRANT_TEST_URL}/readyz", timeout=2):
            return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _server_up(), reason=f"no Qdrant server at {QDRANT_TEST_URL}"),
]


@pytest.fixture
async def server_client():
    client = AsyncQdrantClient(url=QDRANT_TEST_URL)
    collection = f"corpus_chunks_test_{uuid.uuid4().hex[:12]}"
    yield client, collection
    await client.delete_collection(collection)
    await client.close()


async def test_ingest_and_search_on_real_server_is_corpus_scoped(server_client, test_settings):
    client, collection = server_client
    embedder = FakeEmbedder()
    for corpus_id in ("alpha", "beta"):
        report = await ingest_corpus(
            corpus_id,
            settings=test_settings,
            embedder=embedder,
            qdrant_client=client,
            collection_name=collection,
        )
        assert report.status == "ingested"

    dense = DenseIndex(client, embedder, test_settings, collection_name=collection)
    assert await dense.count("alpha") == 9 and await dense.count("beta") == 7

    alpha_hits = await dense.search("alpha", "cancellation policy booking", top_k=20)
    beta_hits = await dense.search("beta", "cancellation policy booking", top_k=20)
    assert {h.corpus_id for h in alpha_hits} == {"alpha"} and len(alpha_hits) == 9
    assert {h.corpus_id for h in beta_hits} == {"beta"} and len(beta_hits) == 7

    # Re-run is a no-op against the real server too.
    again = await ingest_corpus(
        "alpha",
        settings=test_settings,
        embedder=embedder,
        qdrant_client=client,
        collection_name=collection,
    )
    assert again.status == "unchanged"


async def test_real_server_payload_indexes_created(server_client, test_settings):
    client, collection = server_client
    await DenseIndex(
        client, FakeEmbedder(), test_settings, collection_name=collection
    ).ensure_collection()
    info = await client.get_collection(collection)
    assert {"chunk_id", "doc_id", "corpus_id", "section", "text"} <= set(info.payload_schema)
