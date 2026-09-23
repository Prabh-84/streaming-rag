"""BM25 sparse index: build, persist, load, search (PRD_TRD.md §7.13, REQ-EVID-01, REQ-CORPUS-02).

One index per corpus at `<PROCESSED_DIR>/bm25__<corpus_id>.pkl`. The pickle records the corpus_id
it was built for, and loading verifies it against the requested corpus_id, so a renamed or
swapped file can never serve another corpus's chunks.

Only load pickles produced by scripts/ingest_corpus.py — unpickling runs arbitrary code, so these
files are trusted local build artifacts, never user input.
"""

from __future__ import annotations

import asyncio
import os
import pickle
import re
import tempfile
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from app.core.config import Settings, get_settings
from app.core.slots import validate_corpus_id
from app.models.chunk import Chunk, RetrievedChunk

INDEX_FORMAT_VERSION = 1
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_RECORD_FIELDS = ("chunk_id", "doc_id", "section", "text", "chunk_index", "token_count")


class SparseIndexError(RuntimeError):
    """The BM25 index is missing, corrupt, or belongs to a different corpus."""


class SparseIndexNotFoundError(SparseIndexError):
    """No BM25 index has been built for this corpus_id (corpus not ingested)."""


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens. Shared by index build and query time so both sides match."""
    return _TOKEN_RE.findall(text.lower())


def bm25_index_path(corpus_id: str, processed_dir: str | Path) -> Path:
    return Path(processed_dir) / f"bm25__{validate_corpus_id(corpus_id)}.pkl"


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write via temp file + rename so a crash never leaves a half-written artifact behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class BM25Index:
    """BM25Okapi over one corpus's chunks, plus the chunk metadata needed to return hits."""

    def __init__(
        self, corpus_id: str, fingerprint: str, records: list[dict[str, Any]], bm25: BM25Okapi
    ) -> None:
        self.corpus_id = corpus_id
        self.fingerprint = fingerprint
        self._records = records
        self._bm25 = bm25

    @property
    def size(self) -> int:
        return len(self._records)

    def chunk_ids(self) -> list[str]:
        return [r["chunk_id"] for r in self._records]

    @classmethod
    def build(cls, corpus_id: str, chunks: Sequence[Chunk], fingerprint: str = "") -> BM25Index:
        validate_corpus_id(corpus_id)
        if not chunks:
            raise SparseIndexError(f"cannot build a BM25 index for corpus {corpus_id!r}: no chunks")
        corpus_tokens = [c.bm25_tokens or tokenize(c.text) for c in chunks]
        empty = [c.chunk_id for c, toks in zip(chunks, corpus_tokens, strict=True) if not toks]
        if empty:
            raise SparseIndexError(f"chunks with no indexable tokens: {empty}")
        records = [{f: getattr(c, f) for f in _RECORD_FIELDS} for c in chunks]
        return cls(corpus_id, fingerprint, records, BM25Okapi(corpus_tokens))

    def save(self, path: Path) -> None:
        payload = {
            "format_version": INDEX_FORMAT_VERSION,
            "corpus_id": self.corpus_id,
            "fingerprint": self.fingerprint,
            "records": self._records,
            "bm25": self._bm25,
        }
        atomic_write_bytes(path, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))

    @classmethod
    def load(cls, path: Path, expected_corpus_id: str) -> BM25Index:
        validate_corpus_id(expected_corpus_id)
        if not path.is_file():
            raise SparseIndexNotFoundError(
                f"no BM25 index for corpus {expected_corpus_id!r} at {path}; run ingestion first"
            )
        try:
            with path.open("rb") as f:
                payload = pickle.load(f)
        except Exception as exc:
            raise SparseIndexError(f"BM25 index at {path} is unreadable: {exc}") from exc

        if not isinstance(payload, dict) or payload.get("format_version") != INDEX_FORMAT_VERSION:
            raise SparseIndexError(f"BM25 index at {path} has an unsupported format")
        if payload.get("corpus_id") != expected_corpus_id:
            raise SparseIndexError(
                f"BM25 index at {path} belongs to corpus {payload.get('corpus_id')!r}, "
                f"not {expected_corpus_id!r}"
            )
        records, bm25 = payload.get("records"), payload.get("bm25")
        if not isinstance(records, list) or not isinstance(bm25, BM25Okapi):
            raise SparseIndexError(f"BM25 index at {path} is corrupt")
        if len(records) != bm25.corpus_size:
            raise SparseIndexError(f"BM25 index at {path} has mismatched records and index size")
        return cls(expected_corpus_id, payload.get("fingerprint", ""), records, bm25)

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Top-k chunks by BM25 score. Zero-score chunks (no term overlap) are not returned; ties
        break by index order so results are deterministic."""
        tokens = tokenize(query)
        if not tokens or top_k <= 0:
            return []
        scores = self._bm25.get_scores(tokens)
        order = np.argsort(-scores, kind="stable")
        hits: list[RetrievedChunk] = []
        for i in order:
            score = float(scores[i])
            if score <= 0 or len(hits) >= top_k:
                break
            record = self._records[int(i)]
            hits.append(
                RetrievedChunk(corpus_id=self.corpus_id, score=score, rank=len(hits) + 1, **record)
            )
        return hits


class SparseIndexRegistry:
    """Per-corpus BM25 indexes, loaded lazily and cached. Lookups are keyed strictly by corpus_id,
    so a query for one corpus can only ever reach that corpus's index file."""

    def __init__(
        self, settings: Settings | None = None, *, processed_dir: str | Path | None = None
    ):
        self._settings = settings or get_settings()
        self._processed_dir = Path(processed_dir or self._settings.processed_dir)
        self._indexes: dict[str, BM25Index] = {}
        self._lock = threading.Lock()

    def get(self, corpus_id: str) -> BM25Index:
        corpus_id = validate_corpus_id(corpus_id)
        with self._lock:
            index = self._indexes.get(corpus_id)
            if index is None:
                index = BM25Index.load(bm25_index_path(corpus_id, self._processed_dir), corpus_id)
                self._indexes[corpus_id] = index
            return index

    def is_available(self, corpus_id: str) -> bool:
        return bm25_index_path(corpus_id, self._processed_dir).is_file()

    def invalidate(self, corpus_id: str | None = None) -> None:
        """Drop cached indexes (all, or one corpus) so the next lookup reloads from disk."""
        with self._lock:
            if corpus_id is None:
                self._indexes.clear()
            else:
                self._indexes.pop(corpus_id, None)

    async def search(
        self, corpus_id: str, text: str, top_k: int, *, timeout_ms: int | None = None
    ) -> list[RetrievedChunk]:
        """BM25 search bounded by RETRIEVAL_TIMEOUT_MS. First-time index load happens outside the
        timeout (one-off cost, like model warmup); only the search itself is time-bounded."""
        index = await asyncio.to_thread(self.get, corpus_id)
        ms = self._settings.retrieval_timeout_ms if timeout_ms is None else timeout_ms
        return await asyncio.wait_for(asyncio.to_thread(index.search, text, top_k), ms / 1000)
