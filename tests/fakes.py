"""Test doubles and builders. Never used by application code.

Test corpora live under tests/fixtures/corpora/ — synthetic, and entirely separate from the
production corpus location (data/corpus/).
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import time
from collections.abc import Sequence
from pathlib import Path

from app.core.config import Settings
from app.core.embeddings import EMBEDDING_DIM
from app.core.events import EventType, TelemetryEvent
from app.models.chunk import Chunk
from app.retrieval.sparse_bm25 import tokenize

FIXTURE_CORPORA = Path(__file__).parent / "fixtures" / "corpora"


def make_settings(corpus_root: Path, processed_dir: Path, **overrides: object) -> Settings:
    return Settings(
        _env_file=None,
        CORPUS_ROOT=str(corpus_root),
        PROCESSED_DIR=str(processed_dir),
        **overrides,
    )


def make_chunks(corpus_id: str, texts: list[str], doc_id: str = "doc") -> list[Chunk]:
    return [
        Chunk(
            chunk_id=Chunk.make_chunk_id(corpus_id, doc_id, f"§{i}", i, text),
            doc_id=doc_id,
            section=f"§{i}",
            text=text,
            token_count=len(text.split()),
            chunk_index=i,
            bm25_tokens=tokenize(text),
        )
        for i, text in enumerate(texts)
    ]


def hash_vector(text: str, dimension: int = EMBEDDING_DIM) -> tuple[float, ...]:
    """Deterministic bag-of-words vector: each token increments a sha256-chosen dimension, then
    L2-normalize. Texts sharing words get positive cosine similarity, so dense ranking in tests
    is meaningful without loading a real model."""
    vector = [0.0] * dimension
    for token in tokenize(text):
        vector[int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % dimension] += 1.0
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        vector[0] = 1.0
        return tuple(vector)
    return tuple(v / norm for v in vector)


class FakeEmbedder:
    def __init__(
        self,
        model_name: str = "fake-hash-embedder-v1",
        dimension: int = EMBEDDING_DIM,
        *,
        fail_first: int = 0,
        delay_s: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.embed_calls = 0
        self.batch_calls = 0
        self._failures_remaining = fail_first
        self._delay_s = delay_s

    def embed(self, text: str) -> tuple[float, ...]:
        self.embed_calls += 1
        if self._failures_remaining > 0:
            self._failures_remaining -= 1
            raise RuntimeError("injected embedding failure")
        if self._delay_s:
            time.sleep(self._delay_s)
        return hash_vector(text, self.dimension)

    def embed_batch(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        self.batch_calls += 1
        return [hash_vector(t, self.dimension) for t in texts]

    def warmup(self) -> None:
        if self._delay_s:
            time.sleep(self._delay_s)


class FakeDecomposer:
    """Test double for app.decomposition.multi_intent.DecompositionLLM. Never touches the
    network; controller/decomposition unit tests must not depend on a real Anthropic API key."""

    def __init__(
        self,
        sub_queries: list[dict[str, str]] | None = None,
        *,
        raise_error: bool = False,
        delay_s: float = 0.0,
    ) -> None:
        self.calls: list[str] = []
        self._sub_queries = sub_queries
        self._raise_error = raise_error
        self._delay_s = delay_s

    async def decompose(self, transcript: str, *, timeout_ms: int) -> dict[str, object] | None:
        self.calls.append(transcript)
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        if self._raise_error:
            raise RuntimeError("injected decomposition failure")
        if self._sub_queries is None:
            return None
        return {"sub_queries": self._sub_queries}


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[TelemetryEvent] = []

    def __call__(self, event: TelemetryEvent) -> None:
        self.events.append(event)

    def of_type(self, event_type: EventType) -> list[TelemetryEvent]:
        return [e for e in self.events if e.event_type == event_type]
