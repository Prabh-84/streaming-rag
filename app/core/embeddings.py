"""Embedding model wrapper: BAAI/bge-small-en-v1.5, local CPU inference (PRD_TRD.md §8).

The model is loaded lazily on first use (not at import time) so importing this module — e.g. from
config/model validation tests in Phase 1 — never triggers a model download or CPU load. Business
logic that actually calls embed() lands in later phases; this module only provides the wrapper.

Caching: an in-process LRU cache keyed by text (functools.lru_cache-style, per the original TRD
§9.5 caching strategy) avoids re-embedding an identical string within a session's lifetime.
"""

from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings

_EMBEDDING_DIM = 384  # bge-small-en-v1.5 vector size (PRD_TRD.md §7.3, §7.13)

_model = None  # lazily-initialized sentence_transformers.SentenceTransformer singleton


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        settings = get_settings()
        _model = SentenceTransformer(settings.embedding_model)
    return _model


@lru_cache(maxsize=4096)
def embed(text: str) -> tuple[float, ...]:
    """Embed a single string with the corpus embedding model.

    Returns a tuple (hashable, cache-friendly) rather than a list; callers needing a mutable
    vector should call list(embed(text)).
    """
    model = _get_model()
    vector = model.encode(text, normalize_embeddings=True)
    return tuple(float(x) for x in vector)


def embedding_dimension() -> int:
    return _EMBEDDING_DIM
