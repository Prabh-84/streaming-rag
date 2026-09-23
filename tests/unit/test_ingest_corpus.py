"""Ingestion pipeline (scripts/ingest_corpus.py) end to end, with a fake embedder and Qdrant
local in-memory mode."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from app.core.config import get_settings
from app.core.slots import InvalidCorpusIdError, SlotSchemaError
from app.retrieval.dense import DenseIndex
from app.retrieval.sparse_bm25 import BM25Index, bm25_index_path
from scripts import ingest_corpus as ingest
from scripts.ingest_corpus import IngestionError, chunks_path, ingest_corpus, manifest_path
from tests.fakes import FIXTURE_CORPORA, FakeEmbedder, make_settings

SLOTS = 'slots:\n  - name: venue\n    patterns: ["Hall"]\n'


def make_corpus(root: Path, corpus_id: str, files: dict[str, str | bytes]) -> Path:
    corpus_dir = root / corpus_id
    for name, content in files.items():
        path = corpus_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
    return root


async def test_ingest_writes_qdrant_bm25_chunks_and_manifest(test_settings, qdrant_memory):
    embedder = FakeEmbedder()
    report = await ingest_corpus(
        "alpha", settings=test_settings, embedder=embedder, qdrant_client=qdrant_memory
    )
    processed = Path(test_settings.processed_dir)

    assert report.status == "ingested"
    assert (report.document_count, report.chunk_count) == (3, 9)
    assert report.skipped_files == []
    assert await DenseIndex(qdrant_memory, embedder, test_settings).count("alpha") == 9

    bm25 = BM25Index.load(bm25_index_path("alpha", processed), "alpha")
    assert bm25.size == 9 and bm25.fingerprint == report.fingerprint

    lines = chunks_path("alpha", processed).read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert len(records) == 9 and {r["corpus_id"] for r in records} == {"alpha"}
    assert {r["doc_id"] for r in records} == {
        "orion_hall",
        "lumen_pavilion",
        "policies/booking_rules",
    }

    manifest = json.loads(manifest_path("alpha", processed).read_text(encoding="utf-8"))
    assert manifest["fingerprint"] == report.fingerprint
    assert manifest["chunk_count"] == 9 and manifest["embedding_model"] == embedder.model_name
    assert [d["doc_id"] for d in manifest["documents"]] == [
        "lumen_pavilion",
        "orion_hall",
        "policies/booking_rules",
    ]
    assert {r["chunk_id"] for r in records} == set(bm25.chunk_ids())


async def test_ingest_idempotent(test_settings, qdrant_memory):
    """Definition of Done: an unchanged corpus re-ingests as a no-op."""
    embedder = FakeEmbedder()
    first = await ingest_corpus(
        "alpha", settings=test_settings, embedder=embedder, qdrant_client=qdrant_memory
    )
    bm25_file = bm25_index_path("alpha", test_settings.processed_dir)
    mtime = bm25_file.stat().st_mtime_ns

    second = await ingest_corpus(
        "alpha", settings=test_settings, embedder=embedder, qdrant_client=qdrant_memory
    )
    assert second.status == "unchanged"
    assert second.fingerprint == first.fingerprint
    assert embedder.batch_calls == 1  # nothing re-embedded
    assert bm25_file.stat().st_mtime_ns == mtime  # nothing rewritten


async def test_forced_reingest_reproduces_identical_output(test_settings, qdrant_memory):
    embedder = FakeEmbedder()
    dense = DenseIndex(qdrant_memory, embedder, test_settings)
    processed = Path(test_settings.processed_dir)

    first = await ingest_corpus(
        "alpha", settings=test_settings, embedder=embedder, qdrant_client=qdrant_memory
    )
    ids_before = await dense.point_ids("alpha")
    bm25_before = bm25_index_path("alpha", processed).read_bytes()
    chunks_before = chunks_path("alpha", processed).read_bytes()

    again = await ingest_corpus(
        "alpha", settings=test_settings, embedder=embedder, qdrant_client=qdrant_memory, force=True
    )
    assert again.status == "ingested" and again.fingerprint == first.fingerprint
    assert await dense.point_ids("alpha") == ids_before
    assert bm25_index_path("alpha", processed).read_bytes() == bm25_before
    assert chunks_path("alpha", processed).read_bytes() == chunks_before


async def test_content_change_reindexes_and_removes_stale_points(tmp_path, qdrant_memory):
    root = tmp_path / "corpora"
    shutil.copytree(FIXTURE_CORPORA / "alpha", root / "alpha")
    settings = make_settings(root, tmp_path / "processed")
    embedder = FakeEmbedder()
    dense = DenseIndex(qdrant_memory, embedder, settings)

    first = await ingest_corpus(
        "alpha", settings=settings, embedder=embedder, qdrant_client=qdrant_memory
    )
    (root / "alpha" / "lumen_pavilion.md").unlink()
    second = await ingest_corpus(
        "alpha", settings=settings, embedder=embedder, qdrant_client=qdrant_memory
    )

    assert second.status == "ingested" and second.fingerprint != first.fingerprint
    assert second.chunk_count == 5 and second.stale_points_deleted == 4
    assert await dense.count("alpha") == 5
    hits = await dense.search("alpha", "Lumen Pavilion rooftop parking", top_k=10)
    assert all(h.doc_id != "lumen_pavilion" for h in hits)


async def test_chunking_parameter_change_invalidates_fingerprint(tmp_path, qdrant_memory):
    embedder = FakeEmbedder()
    default = make_settings(FIXTURE_CORPORA, tmp_path / "p")
    small = make_settings(
        FIXTURE_CORPORA, tmp_path / "p", CHUNK_MAX_TOKENS=12, CHUNK_OVERLAP_TOKENS=2
    )
    first = await ingest_corpus(
        "alpha", settings=default, embedder=embedder, qdrant_client=qdrant_memory
    )
    second = await ingest_corpus(
        "alpha", settings=small, embedder=embedder, qdrant_client=qdrant_memory
    )
    assert second.status == "ingested" and second.chunk_count > first.chunk_count


async def test_ingest_rejects_missing_slots(tmp_path, qdrant_memory):
    root = make_corpus(tmp_path, "c1", {"doc.md": "# Doc\n\nbody text"})
    settings = make_settings(root, tmp_path / "processed")
    with pytest.raises(SlotSchemaError, match="slots.yaml not found"):
        await ingest_corpus(
            "c1", settings=settings, embedder=FakeEmbedder(), qdrant_client=qdrant_memory
        )
    assert not (tmp_path / "processed").exists()  # nothing written on failure


async def test_ingest_rejects_invalid_slots(tmp_path, qdrant_memory):
    root = make_corpus(tmp_path, "c1", {"slots.yaml": "slots: []", "doc.md": "text"})
    settings = make_settings(root, tmp_path / "processed")
    with pytest.raises(SlotSchemaError):
        await ingest_corpus(
            "c1", settings=settings, embedder=FakeEmbedder(), qdrant_client=qdrant_memory
        )


async def test_ingest_missing_corpus_directory(test_settings, qdrant_memory):
    with pytest.raises(IngestionError, match="corpus directory not found"):
        await ingest_corpus(
            "nope", settings=test_settings, embedder=FakeEmbedder(), qdrant_client=qdrant_memory
        )


async def test_ingest_invalid_corpus_id(test_settings, qdrant_memory):
    with pytest.raises(InvalidCorpusIdError):
        await ingest_corpus(
            "../alpha", settings=test_settings, embedder=FakeEmbedder(), qdrant_client=qdrant_memory
        )


@pytest.mark.parametrize(
    ("files", "fragment"),
    [
        ({"slots.yaml": SLOTS, "notes.pdf": b"%PDF"}, "no supported documents"),
        ({"slots.yaml": SLOTS, "empty.md": "# Title only\n"}, "produced no chunks"),
        ({"slots.yaml": SLOTS, "bad.txt": "caf\xe9".encode("latin-1")}, "not valid UTF-8"),
        ({"slots.yaml": SLOTS, "a.md": "alpha text", "a.txt": "other"}, "same doc_id"),
    ],
)
async def test_malformed_corpus_fails_clearly(tmp_path, qdrant_memory, files, fragment):
    root = make_corpus(tmp_path, "c1", files)
    settings = make_settings(root, tmp_path / "processed")
    with pytest.raises(IngestionError, match=fragment):
        await ingest_corpus(
            "c1", settings=settings, embedder=FakeEmbedder(), qdrant_client=qdrant_memory
        )


async def test_unsupported_and_empty_files_are_skipped_and_reported(tmp_path, qdrant_memory):
    root = make_corpus(
        tmp_path,
        "c1",
        {
            "slots.yaml": SLOTS,
            "doc.md": "# Doc\n\nreal body text",
            "img.png": b"\x89PNG",
            "e.md": "",
        },
    )
    settings = make_settings(root, tmp_path / "processed")
    report = await ingest_corpus(
        "c1", settings=settings, embedder=FakeEmbedder(), qdrant_client=qdrant_memory
    )
    assert report.document_count == 1 and report.chunk_count == 1
    assert report.skipped_files == ["e.md", "img.png"]


@pytest.fixture
def cli_env(monkeypatch, tmp_path):
    """Point the CLI's get_settings() at a temp corpus root; restore the singleton afterwards."""

    def configure(corpus_root: Path) -> None:
        monkeypatch.setenv("CORPUS_ROOT", str(corpus_root))
        monkeypatch.setenv("PROCESSED_DIR", str(tmp_path / "processed"))
        get_settings.cache_clear()

    yield configure
    get_settings.cache_clear()


def test_cli_missing_corpus_exits_nonzero(cli_env, tmp_path):
    cli_env(tmp_path / "empty_root")
    assert ingest.main(["--corpus-id", "nope"]) == 1


def test_cli_all_with_no_corpora_exits_nonzero(cli_env, tmp_path):
    cli_env(tmp_path / "empty_root")
    assert ingest.main(["--all"]) == 1


def test_cli_rejects_corpus_id_and_all_together():
    with pytest.raises(SystemExit) as exc:
        ingest.main(["--corpus-id", "alpha", "--all"])
    assert exc.value.code == 2
