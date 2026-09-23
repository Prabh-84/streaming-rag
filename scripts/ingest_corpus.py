"""Corpus ingestion: discover -> load -> validate slots.yaml -> chunk -> embed -> index.

Writes, per corpus:
  - Qdrant points in `corpus_chunks` (payload includes corpus_id)       REQ-CORPUS-02
  - <PROCESSED_DIR>/bm25__<corpus_id>.pkl       BM25 index               REQ-CORPUS-02
  - <PROCESSED_DIR>/chunks__<corpus_id>.jsonl   audit copy of the chunks
  - <PROCESSED_DIR>/manifest__<corpus_id>.json  fingerprint; written last as the commit marker

Idempotent: chunk ids are deterministic (PRD_TRD.md §7.2), so the corpus fingerprint changes only
when content, chunking parameters, or the embedding model change. An unchanged corpus is a no-op
(no embedding, no writes); pass --force to rebuild anyway, which reproduces identical output.

Usage:
    python scripts/ingest_corpus.py                       # DEFAULT_CORPUS_ID
    python scripts/ingest_corpus.py --corpus-id venues
    python scripts/ingest_corpus.py --all                 # every corpus dir under CORPUS_ROOT
    python scripts/ingest_corpus.py --corpus-id venues --force

Corpus layout: <CORPUS_ROOT>/<corpus_id>/slots.yaml plus .md/.markdown/.txt documents
(subdirectories allowed). Markdown-style '#' headings delimit sections (§1, §2, ...); text before
the first heading is §0.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

if __package__ in (None, ""):  # run as `python scripts/ingest_corpus.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog  # noqa: E402
from qdrant_client import AsyncQdrantClient  # noqa: E402

from app.core.config import Settings, get_settings  # noqa: E402
from app.core.embeddings import Embedder, SentenceTransformerEmbedder  # noqa: E402
from app.core.slots import (  # noqa: E402
    SLOTS_FILENAME,
    InvalidCorpusIdError,
    SlotSchemaError,
    load_slot_schema,
    validate_corpus_id,
)
from app.models.chunk import Chunk  # noqa: E402
from app.models.document import Document  # noqa: E402
from app.retrieval.dense import COLLECTION_NAME, DenseIndex, create_qdrant_client  # noqa: E402
from app.retrieval.sparse_bm25 import (  # noqa: E402
    BM25Index,
    SparseIndexError,
    atomic_write_bytes,
    bm25_index_path,
    tokenize,
)

log = structlog.get_logger()

SUPPORTED_EXTENSIONS = (".md", ".markdown", ".txt")
MANIFEST_FORMAT_VERSION = 1

_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^ {0,3}(```|~~~)")
_COUNT_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n[ \t]*\n")


class IngestionError(RuntimeError):
    """The corpus is missing or malformed; ingestion refuses to proceed."""


@dataclass(frozen=True)
class Section:
    label: str  # "§0" for text before the first heading, then "§1", "§2", ... per heading
    title: str
    body: str


@dataclass(frozen=True)
class LoadedDocument:
    document: Document
    sections: list[Section]


@dataclass
class IngestionReport:
    corpus_id: str
    status: str  # "ingested" | "unchanged"
    document_count: int
    chunk_count: int
    fingerprint: str
    skipped_files: list[str] = field(default_factory=list)
    stale_points_deleted: int = 0
    duration_ms: int = 0


# ---------------------------------------------------------------------------------------------
# Discovery and loading
# ---------------------------------------------------------------------------------------------


def count_tokens(text: str) -> int:
    """Approximate token count (words + punctuation marks). Deterministic and model-free; used for
    chunk sizing and Chunk.token_count (the evidence token budget in Phase 5)."""
    return len(_COUNT_TOKEN_RE.findall(text))


def discover_corpora(corpus_root: str | Path) -> list[str]:
    root = Path(corpus_root)
    if not root.is_dir():
        return []
    return sorted(
        p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))
    )


def discover_documents(corpus_dir: Path) -> tuple[list[Path], list[Path]]:
    """Supported documents and skipped (unsupported) files, both sorted by relative path.
    Hidden files/dirs and the corpus's own slots.yaml are ignored entirely."""
    documents: list[Path] = []
    skipped: list[Path] = []
    for path in sorted(corpus_dir.rglob("*"), key=lambda p: p.relative_to(corpus_dir).as_posix()):
        rel = path.relative_to(corpus_dir)
        if path.is_dir() or any(part.startswith(".") for part in rel.parts):
            continue
        if rel.as_posix() == SLOTS_FILENAME:
            continue
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.append(path)
        else:
            skipped.append(path)
    return documents, skipped


def split_sections(text: str) -> list[Section]:
    """Split on markdown '#' headings (ignoring '#' lines inside fenced code blocks). Sections with
    no body are dropped but still consume a number, so labels are stable for a given document."""
    sections: list[Section] = []
    label_no, title, body_lines = 0, "", []
    in_fence = False

    def flush() -> None:
        body = "\n".join(body_lines).strip()
        if body:
            sections.append(Section(label=f"§{label_no}", title=title, body=body))

    for line in text.split("\n"):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        heading = None if in_fence else _HEADING_RE.match(line)
        if heading and heading.group(2):
            flush()
            label_no += 1
            title, body_lines = heading.group(2).strip(), []
        else:
            body_lines.append(line)
    flush()
    return sections


def _document_title(text: str, fallback: str) -> str:
    first_heading = None
    in_fence = False
    for line in text.split("\n"):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        heading = None if in_fence else _HEADING_RE.match(line)
        if heading and heading.group(2):
            if len(heading.group(1)) == 1:
                return heading.group(2).strip()
            first_heading = first_heading or heading.group(2).strip()
    return first_heading or fallback


def load_document(
    path: Path, corpus_dir: Path, corpus_id: str, ingested_at: datetime
) -> LoadedDocument | None:
    """Load one document. Returns None for a document with no content (skipped, not an error);
    raises IngestionError for undecodable input."""
    rel = PurePosixPath(path.relative_to(corpus_dir).as_posix())
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise IngestionError(f"{corpus_id}/{rel} is not valid UTF-8 text: {exc}") from exc
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    sections = split_sections(text)
    if not sections:
        return None
    document = Document(
        doc_id=Document.make_doc_id(rel),
        title=_document_title(text, fallback=rel.stem),
        source_path=f"{corpus_id}/{rel.as_posix()}",
        corpus_id=corpus_id,
        ingested_at=ingested_at,
        content_hash=hashlib.sha256(raw).hexdigest(),
        section_count=len(sections),
    )
    return LoadedDocument(document=document, sections=sections)


# ---------------------------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------------------------


def _windows(paragraph: str, budget: int, overlap: int) -> list[str]:
    """Split one over-long paragraph into overlapping token windows, preserving original text."""
    spans = list(_COUNT_TOKEN_RE.finditer(paragraph))
    step = max(budget - overlap, 1)
    pieces = []
    for start in range(0, len(spans), step):
        end = min(start + budget, len(spans))
        pieces.append(paragraph[spans[start].start() : spans[end - 1].end()])
        if end == len(spans):
            break
    return pieces


def _section_pieces(body: str, budget: int, overlap: int) -> list[str]:
    """Greedily pack whole paragraphs up to the token budget; window-split any single paragraph
    that exceeds it on its own."""
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for paragraph in (p.strip() for p in _PARAGRAPH_SPLIT_RE.split(body)):
        if not paragraph:
            continue
        tokens = count_tokens(paragraph)
        if tokens > budget:
            if current:
                pieces.append("\n\n".join(current))
                current, current_tokens = [], 0
            pieces.extend(_windows(paragraph, budget, overlap))
        elif current_tokens + tokens > budget:
            pieces.append("\n\n".join(current))
            current, current_tokens = [paragraph], tokens
        else:
            current.append(paragraph)
            current_tokens += tokens
    if current:
        pieces.append("\n\n".join(current))
    return pieces


def chunk_document(
    loaded: LoadedDocument, corpus_id: str, max_tokens: int, overlap_tokens: int
) -> list[Chunk]:
    """Chunks never cross a section boundary. Each chunk repeats its section title as the first
    line, so a mid-section chunk stays self-describing for both dense and sparse retrieval."""
    doc_id = loaded.document.doc_id
    chunks: list[Chunk] = []
    for section in loaded.sections:
        title_tokens = count_tokens(section.title)
        budget = max(max_tokens - title_tokens, 1)
        overlap = min(overlap_tokens, budget - 1)
        for piece in _section_pieces(section.body, budget, overlap):
            if not tokenize(piece):  # punctuation-only fragment
                continue
            text = f"{section.title}\n\n{piece}" if section.title else piece
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=Chunk.make_chunk_id(corpus_id, doc_id, section.label, index, text),
                    doc_id=doc_id,
                    section=section.label,
                    text=text,
                    token_count=count_tokens(text),
                    chunk_index=index,
                    bm25_tokens=tokenize(text),
                )
            )
    return chunks


def compute_fingerprint(chunks: list[Chunk], embedding_model: str, dimension: int) -> str:
    """Chunk ids already hash corpus_id, doc_id, section, position and text, so the ordered id
    list plus the embedding model identifies the full chunk + embedding set."""
    payload = json.dumps(
        {
            "format_version": MANIFEST_FORMAT_VERSION,
            "embedding_model": embedding_model,
            "dimension": dimension,
            "chunk_ids": [c.chunk_id for c in chunks],
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def manifest_path(corpus_id: str, processed_dir: str | Path) -> Path:
    return Path(processed_dir) / f"manifest__{validate_corpus_id(corpus_id)}.json"


def chunks_path(corpus_id: str, processed_dir: str | Path) -> Path:
    return Path(processed_dir) / f"chunks__{validate_corpus_id(corpus_id)}.jsonl"


# ---------------------------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------------------------


def _prepare(
    corpus_id: str, settings: Settings, ingested_at: datetime
) -> tuple[list[LoadedDocument], list[Chunk], list[str]]:
    corpus_dir = Path(settings.corpus_root) / corpus_id
    if not corpus_dir.is_dir():
        raise IngestionError(
            f"corpus directory not found for corpus_id={corpus_id!r}: {corpus_dir}"
        )

    load_slot_schema(corpus_id, settings.corpus_root)  # REQ-CORPUS-01: fail fast on bad slots

    doc_paths, unsupported = discover_documents(corpus_dir)
    skipped = [p.relative_to(corpus_dir).as_posix() for p in unsupported]
    if not doc_paths:
        raise IngestionError(
            f"corpus {corpus_id!r} has no supported documents ({', '.join(SUPPORTED_EXTENSIONS)})"
        )

    loaded: list[LoadedDocument] = []
    seen_doc_ids: dict[str, str] = {}
    for path in doc_paths:
        rel = path.relative_to(corpus_dir).as_posix()
        doc = load_document(path, corpus_dir, corpus_id, ingested_at)
        if doc is None:
            skipped.append(rel)
            log.warning("document_skipped_empty", corpus_id=corpus_id, path=rel)
            continue
        other = seen_doc_ids.get(doc.document.doc_id)
        if other is not None:
            raise IngestionError(
                f"documents {other!r} and {rel!r} in corpus {corpus_id!r} map to the same doc_id "
                f"{doc.document.doc_id!r}; rename one of them"
            )
        seen_doc_ids[doc.document.doc_id] = rel
        loaded.append(doc)

    chunks: list[Chunk] = []
    for doc in loaded:
        chunks.extend(
            chunk_document(doc, corpus_id, settings.chunk_max_tokens, settings.chunk_overlap_tokens)
        )
    if not chunks:
        raise IngestionError(f"corpus {corpus_id!r} produced no chunks")
    return loaded, chunks, sorted(skipped)


async def _is_up_to_date(
    corpus_id: str, fingerprint: str, chunk_count: int, settings: Settings, dense: DenseIndex
) -> bool:
    manifest_file = manifest_path(corpus_id, settings.processed_dir)
    if not manifest_file.is_file():
        return False
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if manifest.get("fingerprint") != fingerprint or manifest.get("chunk_count") != chunk_count:
        return False
    try:
        index = BM25Index.load(bm25_index_path(corpus_id, settings.processed_dir), corpus_id)
    except SparseIndexError:
        return False
    if index.fingerprint != fingerprint or index.size != chunk_count:
        return False
    return await dense.count(corpus_id) == chunk_count


async def ingest_corpus(
    corpus_id: str,
    *,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    qdrant_client: AsyncQdrantClient | None = None,
    collection_name: str = COLLECTION_NAME,
    force: bool = False,
) -> IngestionReport:
    """Ingest one corpus. Raises IngestionError / SlotSchemaError / InvalidCorpusIdError on bad
    input; nothing is written to Qdrant or disk unless the whole corpus loads and chunks cleanly."""
    started = time.perf_counter()
    settings = settings or get_settings()
    corpus_id = validate_corpus_id(corpus_id)
    embedder = embedder or SentenceTransformerEmbedder(settings.embedding_model)
    log.info("ingestion_started", corpus_id=corpus_id, corpus_root=settings.corpus_root)

    loaded, chunks, skipped = _prepare(corpus_id, settings, datetime.now(UTC))
    fingerprint = compute_fingerprint(chunks, embedder.model_name, embedder.dimension)
    log.info(
        "corpus_chunked",
        corpus_id=corpus_id,
        documents=len(loaded),
        chunks=len(chunks),
        skipped=skipped,
        fingerprint=fingerprint,
    )

    owns_client = qdrant_client is None
    client = qdrant_client or create_qdrant_client(settings.qdrant_url)
    try:
        dense = DenseIndex(client, embedder, settings, collection_name=collection_name)

        def report(status: str, stale: int = 0) -> IngestionReport:
            return IngestionReport(
                corpus_id=corpus_id,
                status=status,
                document_count=len(loaded),
                chunk_count=len(chunks),
                fingerprint=fingerprint,
                skipped_files=skipped,
                stale_points_deleted=stale,
                duration_ms=round((time.perf_counter() - started) * 1000),
            )

        if not force and await _is_up_to_date(corpus_id, fingerprint, len(chunks), settings, dense):
            result = report("unchanged")
            log.info("ingestion_unchanged", corpus_id=corpus_id, chunks=len(chunks))
            return result

        vectors = await asyncio.to_thread(embedder.embed_batch, [c.text for c in chunks])
        log.info("embeddings_generated", corpus_id=corpus_id, count=len(vectors))

        await dense.ensure_collection()
        await dense.upsert_chunks(corpus_id, chunks, vectors)
        stale = await dense.delete_stale(corpus_id, {c.chunk_id for c in chunks})
        log.info("qdrant_indexed", corpus_id=corpus_id, upserted=len(chunks), stale_deleted=stale)

        processed = Path(settings.processed_dir)
        BM25Index.build(corpus_id, chunks, fingerprint).save(bm25_index_path(corpus_id, processed))
        atomic_write_bytes(
            chunks_path(corpus_id, processed),
            "".join(
                json.dumps({"corpus_id": corpus_id, **c.model_dump()}, ensure_ascii=False) + "\n"
                for c in chunks
            ).encode("utf-8"),
        )
        manifest = {
            "format_version": MANIFEST_FORMAT_VERSION,
            "corpus_id": corpus_id,
            "fingerprint": fingerprint,
            "embedding_model": embedder.model_name,
            "dimension": embedder.dimension,
            "chunk_count": len(chunks),
            "document_count": len(loaded),
            "chunk_max_tokens": settings.chunk_max_tokens,
            "chunk_overlap_tokens": settings.chunk_overlap_tokens,
            "documents": [d.document.model_dump(mode="json") for d in loaded],
        }
        atomic_write_bytes(
            manifest_path(corpus_id, processed),
            json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"),
        )
        result = report("ingested", stale)
        log.info("ingestion_completed", corpus_id=corpus_id, **_summary(result))
        return result
    finally:
        if owns_client:
            await client.close()


def _summary(r: IngestionReport) -> dict[str, object]:
    return {
        "status": r.status,
        "documents": r.document_count,
        "chunks": r.chunk_count,
        "stale_points_deleted": r.stale_points_deleted,
        "duration_ms": r.duration_ms,
    }


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


async def _run_cli(corpus_ids: list[str], settings: Settings, force: bool) -> int:
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    client = create_qdrant_client(settings.qdrant_url)
    failures = 0
    try:
        for corpus_id in corpus_ids:
            try:
                await ingest_corpus(
                    corpus_id,
                    settings=settings,
                    embedder=embedder,
                    qdrant_client=client,
                    force=force,
                )
            except (IngestionError, SlotSchemaError, InvalidCorpusIdError) as exc:
                failures += 1
                log.error("ingestion_failed", corpus_id=corpus_id, error=str(exc))
            except Exception as exc:  # backend failures: Qdrant unreachable, model load, ...
                failures += 1
                log.error(
                    "ingestion_failed",
                    corpus_id=corpus_id,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
    finally:
        await client.close()
    return 1 if failures else 0


def _configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest a corpus into Qdrant + BM25.")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--corpus-id", help="corpus directory name under CORPUS_ROOT")
    target.add_argument("--all", action="store_true", help="ingest every corpus under CORPUS_ROOT")
    parser.add_argument(
        "--force", action="store_true", help="rebuild even if the corpus is unchanged"
    )
    args = parser.parse_args(argv)
    _configure_logging()

    settings = get_settings()
    if args.all:
        corpus_ids = discover_corpora(settings.corpus_root)
        if not corpus_ids:
            log.error("no_corpora_found", corpus_root=settings.corpus_root)
            return 1
    else:
        corpus_ids = [args.corpus_id or settings.default_corpus_id]
    return asyncio.run(_run_cli(corpus_ids, settings, args.force))


if __name__ == "__main__":
    sys.exit(main())
