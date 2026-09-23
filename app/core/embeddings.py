"""Embedding model wrapper: BAAI/bge-small-en-v1.5, local CPU inference (PRD_TRD.md §8).

The model is loaded lazily on first use (not at import or construction time), so importing this
module never triggers a model download. Callers on the latency-critical path should call
`warmup()` at startup so the first real query doesn't pay the load cost.

Caching: an in-process LRU cache keyed by text (original TRD §9.5) avoids re-embedding an
identical string. Corpus embeddings are computed once at ingestion via `embed_batch` (uncached)
and persisted in Qdrant — never recomputed at query time.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Protocol, runtime_checkable

from app.core.config import get_settings

EMBEDDING_DIM = 384  # bge-small-en-v1.5 vector size (PRD_TRD.md §7.3, §7.13)


class EmbeddingError(RuntimeError):
    """The embedding model is misconfigured or returned an unexpected vector."""


@runtime_checkable
class Embedder(Protocol):
    model_name: str
    dimension: int

    def embed(self, text: str) -> tuple[float, ...]: ...

    def embed_batch(self, texts: Sequence[str]) -> list[tuple[float, ...]]: ...

    def warmup(self) -> None: ...


def _to_vector(values: Any, dimension: int) -> tuple[float, ...]:
    vector = tuple(float(x) for x in values)
    if len(vector) != dimension:
        raise EmbeddingError(f"expected a {dimension}-d embedding, got {len(vector)}")
    return vector


class SentenceTransformerEmbedder:
    """Local sentence-transformers embedder. Vectors are L2-normalized (cosine-ready)."""

    def __init__(
        self,
        model_name: str,
        *,
        dimension: int = EMBEDDING_DIM,
        batch_size: int = 32,
        cache_size: int = 4096,
    ) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self._batch_size = batch_size
        self._model: Any = None
        self._lock = threading.Lock()
        self._embed_cached = lru_cache(maxsize=cache_size)(self._embed_uncached)

    def _get_model(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    model = SentenceTransformer(self.model_name, device="cpu")
                    actual = model.get_sentence_embedding_dimension()
                    if actual != self.dimension:
                        raise EmbeddingError(
                            f"{self.model_name} produces {actual}-d vectors; the corpus schema "
                            f"requires {self.dimension}-d (PRD_TRD.md §7.13)"
                        )
                    self._model = model
        return self._model

    def warmup(self) -> None:
        self._get_model()

    def embed(self, text: str) -> tuple[float, ...]:
        return self._embed_cached(text)

    def _embed_uncached(self, text: str) -> tuple[float, ...]:
        vector = self._get_model().encode(text, normalize_embeddings=True, show_progress_bar=False)
        return _to_vector(vector, self.dimension)

    def embed_batch(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        if not texts:
            return []
        vectors = self._get_model().encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [_to_vector(v, self.dimension) for v in vectors]


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformerEmbedder:
    """Process-wide embedder built from settings (one model instance per process)."""
    return SentenceTransformerEmbedder(get_settings().embedding_model)


def embed(text: str) -> tuple[float, ...]:
    """Embed a single string with the process-wide corpus embedding model."""
    return get_embedder().embed(text)


def embedding_dimension() -> int:
    return EMBEDDING_DIM
