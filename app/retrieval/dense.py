"""Qdrant dense index wrapper (PRD_TRD.md §7.13, REQ-EVID-01, REQ-CORPUS-02).

Collection `corpus_chunks`: 384-d cosine vectors, one point per chunk, point id = chunk_id.
Every query carries a `corpus_id` payload filter, and every returned hit is re-checked against
the requested corpus_id — the filter is the isolation mechanism, the re-check makes a filter bug
fail loudly instead of silently leaking another corpus's evidence.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from qdrant_client import AsyncQdrantClient, models

from app.core.config import Settings, get_settings
from app.core.embeddings import EMBEDDING_DIM, Embedder
from app.core.slots import validate_corpus_id
from app.models.chunk import Chunk, RetrievedChunk

COLLECTION_NAME = "corpus_chunks"
_UPSERT_BATCH_SIZE = 128
_SCROLL_PAGE_SIZE = 1024
_PAYLOAD_INDEXES: tuple[tuple[str, models.PayloadSchemaType], ...] = (
    ("chunk_id", models.PayloadSchemaType.KEYWORD),
    ("doc_id", models.PayloadSchemaType.KEYWORD),
    ("corpus_id", models.PayloadSchemaType.KEYWORD),
    ("section", models.PayloadSchemaType.KEYWORD),
    ("text", models.PayloadSchemaType.TEXT),
)


class DenseIndexError(RuntimeError):
    """The Qdrant collection is misconfigured or an index operation received bad input."""


class CorpusIsolationError(RuntimeError):
    """A retrieval result belonged to a corpus other than the one requested (SECURITY class)."""


class QueryEmbeddingError(RuntimeError):
    """The query could not be embedded within the embedding call policy."""


def corpus_filter(corpus_id: str) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="corpus_id", match=models.MatchValue(value=validate_corpus_id(corpus_id))
            )
        ]
    )


def create_qdrant_client(url: str) -> AsyncQdrantClient:
    return AsyncQdrantClient(url=url)


class DenseIndex:
    def __init__(
        self,
        client: AsyncQdrantClient,
        embedder: Embedder,
        settings: Settings | None = None,
        *,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        if embedder.dimension != EMBEDDING_DIM:
            raise DenseIndexError(
                f"embedder {embedder.model_name!r} is {embedder.dimension}-d; the corpus schema "
                f"requires {EMBEDDING_DIM}-d vectors"
            )
        self._client = client
        self._embedder = embedder
        self._settings = settings or get_settings()
        self.collection_name = collection_name

    async def ensure_collection(self) -> None:
        """Create the collection with the spec'd config, or verify an existing one matches."""
        if await self._client.collection_exists(self.collection_name):
            info = await self._client.get_collection(self.collection_name)
            params = info.config.params.vectors
            if not isinstance(params, models.VectorParams):
                raise DenseIndexError(
                    f"collection {self.collection_name!r} uses named vectors; expected one "
                    f"unnamed {EMBEDDING_DIM}-d cosine vector"
                )
            if params.size != EMBEDDING_DIM or params.distance != models.Distance.COSINE:
                raise DenseIndexError(
                    f"collection {self.collection_name!r} is {params.size}-d/{params.distance}; "
                    f"expected {EMBEDDING_DIM}-d/Cosine. Recreate it or change collection_name."
                )
            return

        await self._client.create_collection(
            self.collection_name,
            vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE),
        )
        for field_name, schema in _PAYLOAD_INDEXES:
            await self._client.create_payload_index(
                self.collection_name, field_name=field_name, field_schema=schema
            )

    async def upsert_chunks(
        self, corpus_id: str, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]
    ) -> None:
        corpus_id = validate_corpus_id(corpus_id)
        if len(chunks) != len(vectors):
            raise DenseIndexError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        points = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            if len(vector) != EMBEDDING_DIM:
                raise DenseIndexError(
                    f"vector for chunk {chunk.chunk_id} is {len(vector)}-d, "
                    f"expected {EMBEDDING_DIM}"
                )
            points.append(
                models.PointStruct(
                    id=chunk.chunk_id,
                    vector=list(vector),
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "corpus_id": corpus_id,
                        "section": chunk.section,
                        "text": chunk.text,
                        "chunk_index": chunk.chunk_index,
                        "token_count": chunk.token_count,
                    },
                )
            )
        for start in range(0, len(points), _UPSERT_BATCH_SIZE):
            await self._client.upsert(
                self.collection_name, points=points[start : start + _UPSERT_BATCH_SIZE], wait=True
            )

    async def point_ids(self, corpus_id: str) -> set[str]:
        """All point ids stored for one corpus."""
        ids: set[str] = set()
        offset = None
        while True:
            records, offset = await self._client.scroll(
                self.collection_name,
                scroll_filter=corpus_filter(corpus_id),
                limit=_SCROLL_PAGE_SIZE,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            ids.update(str(r.id) for r in records)
            if offset is None:
                return ids

    async def delete_stale(self, corpus_id: str, keep_ids: set[str]) -> int:
        """Delete this corpus's points that are not in keep_ids. Never touches other corpora,
        because candidates come only from this corpus's filtered scroll."""
        stale = sorted((await self.point_ids(corpus_id)) - keep_ids)
        for start in range(0, len(stale), _UPSERT_BATCH_SIZE):
            await self._client.delete(
                self.collection_name,
                points_selector=models.PointIdsList(
                    points=stale[start : start + _UPSERT_BATCH_SIZE]
                ),
                wait=True,
            )
        return len(stale)

    async def count(self, corpus_id: str) -> int:
        if not await self._client.collection_exists(self.collection_name):
            return 0
        result = await self._client.count(
            self.collection_name, count_filter=corpus_filter(corpus_id), exact=True
        )
        return result.count

    async def warmup(self) -> None:
        """Load the embedding model outside any timeout (the first load takes seconds)."""
        await asyncio.to_thread(self._embedder.warmup)

    async def _embed_query(self, text: str) -> tuple[float, ...]:
        """Original TRD §9.4 embedding policy: EMBEDDING_TIMEOUT_MS per attempt, 1 retry after
        EMBEDDING_RETRY_BACKOFF_MS."""
        timeout = self._settings.embedding_timeout_ms / 1000
        last_error: BaseException | None = None
        for attempt in range(2):
            if attempt:
                await asyncio.sleep(self._settings.embedding_retry_backoff_ms / 1000)
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(self._embedder.embed, text), timeout
                )
            except Exception as exc:
                last_error = exc
        raise QueryEmbeddingError(f"query embedding failed after 2 attempts: {last_error!r}") from (
            last_error
        )

    async def search(
        self, corpus_id: str, text: str, top_k: int, *, timeout_ms: int | None = None
    ) -> list[RetrievedChunk]:
        corpus_id = validate_corpus_id(corpus_id)
        if not text.strip() or top_k <= 0:
            return []
        vector = await self._embed_query(text)
        return await self.search_vector(corpus_id, vector, top_k, timeout_ms=timeout_ms)

    async def search_vector(
        self,
        corpus_id: str,
        vector: Sequence[float],
        top_k: int,
        *,
        timeout_ms: int | None = None,
    ) -> list[RetrievedChunk]:
        """Nearest chunks within one corpus. The Qdrant call is bounded by RETRIEVAL_TIMEOUT_MS
        (original TRD §9.4: no retry — a timed-out search contributes no evidence)."""
        corpus_id = validate_corpus_id(corpus_id)
        ms = self._settings.retrieval_timeout_ms if timeout_ms is None else timeout_ms
        response = await asyncio.wait_for(
            self._client.query_points(
                self.collection_name,
                query=list(vector),
                query_filter=corpus_filter(corpus_id),
                limit=top_k,
                with_payload=True,
            ),
            ms / 1000,
        )
        hits: list[RetrievedChunk] = []
        for rank, point in enumerate(response.points, start=1):
            payload = point.payload or {}
            if payload.get("corpus_id") != corpus_id:
                raise CorpusIsolationError(
                    f"dense search for corpus {corpus_id!r} returned point {point.id} from corpus "
                    f"{payload.get('corpus_id')!r}"
                )
            hits.append(
                RetrievedChunk(
                    chunk_id=str(point.id),
                    doc_id=payload["doc_id"],
                    corpus_id=corpus_id,
                    section=payload["section"],
                    text=payload["text"],
                    chunk_index=payload["chunk_index"],
                    token_count=payload["token_count"],
                    score=float(point.score),
                    rank=rank,
                )
            )
        return hits
