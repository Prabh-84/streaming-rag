"""The real BAAI/bge-small-en-v1.5 model. Opt-in (downloads ~130MB on first run):

RUN_MODEL_TESTS=1 pytest tests/integration/test_embedding_model.py
"""

from __future__ import annotations

import math
import os

import pytest

from app.core.embeddings import EMBEDDING_DIM, SentenceTransformerEmbedder
from app.retrieval.dense import DenseIndex
from scripts.ingest_corpus import ingest_corpus

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1 to load the real model"
)


@pytest.fixture(scope="module")
def bge() -> SentenceTransformerEmbedder:
    embedder = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")
    embedder.warmup()
    return embedder


def test_real_model_produces_normalized_384d_deterministic_vectors(bge):
    vector = bge.embed("What is the cancellation policy for Orion Hall?")
    assert len(vector) == EMBEDDING_DIM
    assert math.isclose(sum(v * v for v in vector), 1.0, rel_tol=1e-4)
    batch = bge.embed_batch(["What is the cancellation policy for Orion Hall?"])
    assert all(math.isclose(a, b, abs_tol=1e-5) for a, b in zip(batch[0], vector, strict=True))


async def test_real_model_dense_retrieval_ranks_the_relevant_chunk_first(
    bge, test_settings, qdrant_memory
):
    await ingest_corpus("alpha", settings=test_settings, embedder=bge, qdrant_client=qdrant_memory)
    dense = DenseIndex(qdrant_memory, bge, test_settings)
    hits = await dense.search("alpha", "How late can I cancel a booking at Orion Hall?", top_k=3)
    assert hits[0].doc_id == "orion_hall" and "Cancellation Policy" in hits[0].text
