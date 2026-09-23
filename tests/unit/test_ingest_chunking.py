"""Discovery, loading, section splitting, chunking, stable ids (scripts/ingest_corpus.py)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts.ingest_corpus import (
    IngestionError,
    chunk_document,
    count_tokens,
    discover_corpora,
    discover_documents,
    load_document,
    split_sections,
)
from tests.fakes import FIXTURE_CORPORA

NOW = datetime(2026, 9, 23, tzinfo=UTC)


def _load(path: Path, corpus_dir: Path, corpus_id: str = "c1"):
    loaded = load_document(path, corpus_dir, corpus_id, NOW)
    assert loaded is not None
    return loaded


def test_discover_documents_filters_supported_and_skips_slots_and_hidden(tmp_path):
    (tmp_path / "slots.yaml").write_text("slots: []", encoding="utf-8")
    (tmp_path / "a.md").write_text("# A\n\nbody", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("body", encoding="utf-8")
    (tmp_path / "c.MARKDOWN").write_text("body", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / ".hidden.md").write_text("secret", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "x.md").write_text("vcs", encoding="utf-8")

    docs, skipped = discover_documents(tmp_path)
    assert [p.relative_to(tmp_path).as_posix() for p in docs] == ["a.md", "c.MARKDOWN", "sub/b.txt"]
    assert [p.name for p in skipped] == ["image.png"]


def test_discover_corpora_lists_only_directories(tmp_path):
    (tmp_path / "beta").mkdir()
    (tmp_path / "alpha").mkdir()
    (tmp_path / ".cache").mkdir()
    (tmp_path / "README.md").write_text("docs", encoding="utf-8")
    assert discover_corpora(tmp_path) == ["alpha", "beta"]
    assert discover_corpora(tmp_path / "missing") == []


def test_split_sections_labels_titles_and_preamble():
    text = (
        "Intro line before any heading.\n\n"
        "# Title\n\n"  # heading with no body: dropped, but consumes §1
        "## Capacity\n\nSeats 120.\n\n"
        "```\n# not a heading, inside a code fence\n```\n\n"
        "## Empty\n\n"
        "### Catering ###\n\nVegetarian menu."
    )
    sections = split_sections(text)
    assert [(s.label, s.title) for s in sections] == [
        ("§0", ""),
        ("§2", "Capacity"),
        ("§4", "Catering"),
    ]
    assert "# not a heading" in sections[1].body
    assert sections[2].body == "Vegetarian menu."


def test_load_document_metadata():
    corpus_dir = FIXTURE_CORPORA / "alpha"
    loaded = _load(corpus_dir / "orion_hall.md", corpus_dir, "alpha")
    doc = loaded.document
    assert doc.doc_id == "orion_hall"
    assert doc.title == "Orion Hall"
    assert doc.corpus_id == "alpha"
    assert doc.source_path == "alpha/orion_hall.md"
    assert doc.section_count == 4
    assert len(doc.content_hash) == 64
    assert doc.ingested_at == NOW


def test_doc_id_is_relative_path_and_title_falls_back_to_stem(tmp_path):
    (tmp_path / "sub dir").mkdir()
    path = tmp_path / "sub dir" / "my notes.txt"
    path.write_text("plain text without headings", encoding="utf-8")
    loaded = _load(path, tmp_path)
    assert loaded.document.doc_id == "sub_dir/my_notes"
    assert loaded.document.title == "my notes"
    assert [s.label for s in loaded.sections] == ["§0"]


def test_empty_document_is_skipped(tmp_path):
    path = tmp_path / "empty.md"
    path.write_text("# Only A Heading\n\n   \n", encoding="utf-8")
    assert load_document(path, tmp_path, "c1", NOW) is None


def test_non_utf8_document_fails_clearly(tmp_path):
    path = tmp_path / "latin1.txt"
    path.write_bytes("caf\xe9 menu".encode("latin-1"))
    with pytest.raises(IngestionError, match="not valid UTF-8"):
        load_document(path, tmp_path, "c1", NOW)


def test_crlf_and_bom_are_normalized(tmp_path):
    path = tmp_path / "win.md"
    path.write_bytes("\ufeff# Win\r\n\r\nLine one.\r\n".encode())
    loaded = _load(path, tmp_path)
    assert loaded.document.title == "Win"
    assert loaded.sections[0].body == "Line one."


def test_count_tokens():
    assert count_tokens("") == 0
    assert count_tokens("Seats 120 guests.") == 4
    assert count_tokens("a,b") == 3


def test_chunks_carry_section_title_metadata_and_bm25_tokens():
    corpus_dir = FIXTURE_CORPORA / "alpha"
    chunks = chunk_document(
        _load(corpus_dir / "orion_hall.md", corpus_dir, "alpha"), "alpha", 200, 40
    )
    assert [c.section for c in chunks] == ["§1", "§2", "§3", "§4"]
    assert [c.chunk_index for c in chunks] == [0, 1, 2, 3]
    cancellation = chunks[2]
    assert cancellation.text.startswith("Cancellation Policy\n\n")
    assert "14 days" in cancellation.text
    assert cancellation.doc_id == "orion_hall"
    assert cancellation.token_count == count_tokens(cancellation.text)
    assert "cancellation" in cancellation.bm25_tokens


def test_long_paragraph_is_window_split_within_budget_with_overlap(tmp_path):
    words = " ".join(f"w{i}" for i in range(120))
    path = tmp_path / "long.md"
    path.write_text(f"## Long\n\n{words}", encoding="utf-8")
    chunks = chunk_document(_load(path, tmp_path), "c1", max_tokens=31, overlap_tokens=10)

    assert len(chunks) > 1
    assert all(c.token_count <= 31 for c in chunks)
    bodies = [c.text.split("\n\n", 1)[1].split() for c in chunks]
    for previous, current in zip(bodies, bodies[1:], strict=False):
        assert previous[-10:] == current[:10]  # 10-token overlap between consecutive windows
    assert bodies[0][0] == "w0" and bodies[-1][-1] == "w119"  # nothing lost


def test_short_paragraphs_are_packed_without_crossing_sections(tmp_path):
    path = tmp_path / "packed.md"
    path.write_text("## A\n\none two.\n\nthree four.\n\n## B\n\nfive six.", encoding="utf-8")
    chunks = chunk_document(_load(path, tmp_path), "c1", 200, 40)
    assert [(c.section, c.text) for c in chunks] == [
        ("§1", "A\n\none two.\n\nthree four."),
        ("§2", "B\n\nfive six."),
    ]


def test_chunk_ids_are_stable_unique_and_content_addressed(tmp_path):
    corpus_dir = FIXTURE_CORPORA / "alpha"
    loaded = _load(corpus_dir / "orion_hall.md", corpus_dir, "alpha")
    first = chunk_document(loaded, "alpha", 200, 40)
    second = chunk_document(loaded, "alpha", 200, 40)
    ids = [c.chunk_id for c in first]

    assert ids == [c.chunk_id for c in second]  # deterministic across runs
    assert len(set(ids)) == len(ids)
    assert all(uuid.UUID(i).version == 5 for i in ids)  # valid Qdrant point ids
    other_corpus = [c.chunk_id for c in chunk_document(loaded, "beta", 200, 40)]
    assert set(ids).isdisjoint(other_corpus)  # same text in another corpus never collides

    changed = tmp_path / "orion_hall.md"
    changed.write_text(
        (corpus_dir / "orion_hall.md").read_text(encoding="utf-8").replace("120", "150"),
        encoding="utf-8",
    )
    new_ids = [
        c.chunk_id for c in chunk_document(_load(changed, tmp_path, "alpha"), "alpha", 200, 40)
    ]
    assert new_ids[0] == ids[0] and new_ids[1] != ids[1]  # only the edited chunk changes id
