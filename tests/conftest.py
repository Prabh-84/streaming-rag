"""Shared fixtures. Builders and fakes live in tests/fakes.py."""

from __future__ import annotations

from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient

from app.core.config import Settings
from tests.fakes import FIXTURE_CORPORA, FakeEmbedder, make_settings


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    return make_settings(FIXTURE_CORPORA, tmp_path / "processed")


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
async def qdrant_memory():
    """Real Qdrant semantics (filters, cosine search, scroll/delete) in local in-memory mode."""
    client = AsyncQdrantClient(location=":memory:")
    yield client
    await client.close()
