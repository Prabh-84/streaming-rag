"""Evidence Fusion (pseudocode 12.D; REQ-EVID-02, REQ-EVID-04) against the alpha fixture corpus,
whose slots.yaml declares `venue` as the entity key and `capacity` as a numeric attribute
specifically so a genuine (entity, attribute) contradiction is testable (REQ-EVAL-01: no
benchmark-specific text, only synthetic fixture content).
"""

from __future__ import annotations

import pytest

from app.controller.entity_extraction import get_corpus_matcher
from app.core.slots import get_slot_schema
from app.models.chunk import RetrievedChunk
from app.models.evidence import Evidence
from app.models.retrieval_event import RetrievalResult
from app.retrieval.fusion import dedup, flag_contradictions, rrf_fuse
from tests.fakes import FIXTURE_CORPORA, FakeEmbedder


@pytest.fixture(autouse=True)
def _clear_schema_caches():
    get_slot_schema.cache_clear()
    get_corpus_matcher.cache_clear()
    yield
    get_slot_schema.cache_clear()
    get_corpus_matcher.cache_clear()


@pytest.fixture(autouse=True)
def _use_fixture_corpus_root(monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "corpus_root", str(FIXTURE_CORPORA))


def _chunk(
    chunk_id: str, text: str, score: float, rank: int, corpus_id: str = "alpha"
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="doc",
        corpus_id=corpus_id,
        section="§1",
        text=text,
        chunk_index=0,
        token_count=len(text.split()),
        score=score,
        rank=rank,
    )


# --- A. RRF scoring/order ---------------------------------------------------------------------


def test_rrf_fusion_combines_dense_and_sparse_ranks():
    chunk_a = _chunk("00000000-0000-0000-0000-00000000000a", "chunk a text", 0.9, 1)
    chunk_b = _chunk("00000000-0000-0000-0000-00000000000b", "chunk b text", 0.8, 2)
    chunk_c = _chunk("00000000-0000-0000-0000-00000000000c", "chunk c text", 0.7, 2)

    result = RetrievalResult(
        sub_query_id="sq1",
        corpus_id="alpha",
        dense=[chunk_a, chunk_b],  # ranks 0, 1
        sparse=[chunk_b, chunk_c],  # ranks 0, 1
    )
    evidence, chunks_by_id = rrf_fuse([result], rrf_k=60)

    by_chunk = {e.chunk_id: e.fusion_score for e in evidence}
    assert by_chunk[chunk_a.chunk_id] == pytest.approx(1 / 61)
    assert by_chunk[chunk_b.chunk_id] == pytest.approx(1 / 62 + 1 / 61)
    assert by_chunk[chunk_c.chunk_id] == pytest.approx(1 / 62)
    # chunk_b appears in both lists, so it must outrank both single-list chunks.
    ranked = sorted(evidence, key=lambda e: -e.fusion_score)
    assert [e.chunk_id for e in ranked] == [chunk_b.chunk_id, chunk_a.chunk_id, chunk_c.chunk_id]
    assert set(chunks_by_id) == {chunk_a.chunk_id, chunk_b.chunk_id, chunk_c.chunk_id}
    assert all(e.sub_query_id == "sq1" for e in evidence)


def test_rrf_fusion_tags_each_row_with_its_sub_query_id():
    chunk_a = _chunk("00000000-0000-0000-0000-00000000000a", "about a", 0.9, 1)
    chunk_b = _chunk("00000000-0000-0000-0000-00000000000b", "about b", 0.9, 1)
    results = [
        RetrievalResult(sub_query_id="sq1", corpus_id="alpha", dense=[chunk_a], sparse=[]),
        RetrievalResult(sub_query_id="sq2", corpus_id="alpha", dense=[chunk_b], sparse=[]),
    ]
    evidence, _ = rrf_fuse(results, rrf_k=60)
    assert {e.sub_query_id for e in evidence} == {"sq1", "sq2"}


# --- B. Deduplication --------------------------------------------------------------------------


def test_dedup_collapses_same_chunk_id_across_sub_queries():
    """REQ-EVID-02: the same physical chunk relevant to two different sub-intents is one piece of
    evidence, not two — the "globally across sub-queries" half of fusion."""
    shared_id = "00000000-0000-0000-0000-00000000000a"
    chunk = _chunk(shared_id, "shared chunk text", 0.9, 1)
    results = [
        RetrievalResult(sub_query_id="sq1", corpus_id="alpha", dense=[chunk], sparse=[]),
        RetrievalResult(
            sub_query_id="sq2", corpus_id="alpha", dense=[chunk, chunk], sparse=[]
        ),  # higher fusion_score for sq2 (appears "twice")
    ]
    evidence, chunks_by_id = rrf_fuse(results, rrf_k=60)
    deduped = dedup(evidence, chunks_by_id, threshold=0.95, embedder=FakeEmbedder())

    survivors = [e for e in deduped if e.duplicate_of is None]
    duplicates = [e for e in deduped if e.duplicate_of is not None]
    assert len(survivors) == 1
    assert len(duplicates) == 1
    assert survivors[0].sub_query_id == "sq2"  # the higher-fusion_score instance
    assert duplicates[0].duplicate_of == survivors[0].evidence_id


def test_dedup_collapses_near_duplicate_text_above_threshold():
    """Two different chunk_ids with identical text are near-duplicates under the embedder's
    cosine similarity (DEDUP_THRESHOLD), even though chunk_id itself doesn't match."""
    chunk_1 = _chunk("00000000-0000-0000-0000-000000000001", "identical wording here", 0.9, 1)
    chunk_2 = _chunk("00000000-0000-0000-0000-000000000002", "identical wording here", 0.5, 2)
    result = RetrievalResult(
        sub_query_id="sq1", corpus_id="alpha", dense=[chunk_1, chunk_2], sparse=[]
    )
    evidence, chunks_by_id = rrf_fuse([result], rrf_k=60)
    deduped = dedup(evidence, chunks_by_id, threshold=0.95, embedder=FakeEmbedder())

    survivors = [e for e in deduped if e.duplicate_of is None]
    assert len(survivors) == 1
    assert survivors[0].chunk_id == chunk_1.chunk_id  # higher fusion_score (rank 0 in dense)


def test_dedup_keeps_genuinely_different_chunks():
    chunk_1 = _chunk("00000000-0000-0000-0000-000000000001", "cancellation policy details", 0.9, 1)
    chunk_2 = _chunk("00000000-0000-0000-0000-000000000002", "catering menu options", 0.8, 2)
    result = RetrievalResult(
        sub_query_id="sq1", corpus_id="alpha", dense=[chunk_1, chunk_2], sparse=[]
    )
    evidence, chunks_by_id = rrf_fuse([result], rrf_k=60)
    deduped = dedup(evidence, chunks_by_id, threshold=0.95, embedder=FakeEmbedder())
    assert all(e.duplicate_of is None for e in deduped)


# --- C. Contradiction flagging -------------------------------------------------------------------


def test_contradiction_pair_flagged_for_same_entity_conflicting_numeric_value():
    chunk_a = _chunk(
        "00000000-0000-0000-0000-00000000000a",
        "Orion Hall can comfortably seat up to 200 guests for a formal dinner.",
        0.9,
        1,
    )
    chunk_b = _chunk(
        "00000000-0000-0000-0000-00000000000b",
        "Orion Hall's ballroom accommodates a maximum of 150 guests.",
        0.8,
        2,
    )
    result = RetrievalResult(
        sub_query_id="sq1", corpus_id="alpha", dense=[chunk_a, chunk_b], sparse=[]
    )
    evidence, chunks_by_id = rrf_fuse([result], rrf_k=60)
    flagged = flag_contradictions(evidence, chunks_by_id, "alpha")

    pair_ids = {e.chunk_id: e.contradiction_pair_id for e in flagged}
    assert pair_ids[chunk_a.chunk_id] is not None
    assert pair_ids[chunk_a.chunk_id] == pair_ids[chunk_b.chunk_id]


def test_no_contradiction_across_different_entities():
    chunk_a = _chunk(
        "00000000-0000-0000-0000-00000000000a", "Orion Hall seats up to 200 guests.", 0.9, 1
    )
    chunk_b = _chunk(
        "00000000-0000-0000-0000-00000000000b", "Lumen Pavilion seats up to 150 guests.", 0.8, 2
    )
    result = RetrievalResult(
        sub_query_id="sq1", corpus_id="alpha", dense=[chunk_a, chunk_b], sparse=[]
    )
    evidence, chunks_by_id = rrf_fuse([result], rrf_k=60)
    flagged = flag_contradictions(evidence, chunks_by_id, "alpha")
    assert all(e.contradiction_pair_id is None for e in flagged)


def test_no_contradiction_when_values_agree():
    chunk_a = _chunk(
        "00000000-0000-0000-0000-00000000000a", "Orion Hall seats up to 200 guests.", 0.9, 1
    )
    chunk_b = _chunk(
        "00000000-0000-0000-0000-00000000000b",
        "Orion Hall can host 200 guests comfortably.",
        0.8,
        2,
    )
    result = RetrievalResult(
        sub_query_id="sq1", corpus_id="alpha", dense=[chunk_a, chunk_b], sparse=[]
    )
    evidence, chunks_by_id = rrf_fuse([result], rrf_k=60)
    flagged = flag_contradictions(evidence, chunks_by_id, "alpha")
    assert all(e.contradiction_pair_id is None for e in flagged)


def test_no_contradiction_detection_without_entity_key_slot():
    """beta's slots.yaml declares no is_entity_key slot - a safe default, not a regression."""
    chunk_a = _chunk(
        "00000000-0000-0000-0000-00000000000a", "domestic trips use USD", 0.9, 1, corpus_id="beta"
    )
    chunk_b = _chunk(
        "00000000-0000-0000-0000-00000000000b",
        "international trips use EUR",
        0.8,
        2,
        corpus_id="beta",
    )
    result = RetrievalResult(
        sub_query_id="sq1", corpus_id="beta", dense=[chunk_a, chunk_b], sparse=[]
    )
    evidence, chunks_by_id = rrf_fuse([result], rrf_k=60)
    flagged = flag_contradictions(evidence, chunks_by_id, "beta")
    assert all(e.contradiction_pair_id is None for e in flagged)


def test_contradiction_skips_duplicate_rows():
    """A row already marked duplicate_of must never participate in contradiction comparison -
    it's about to be excluded from reranking anyway."""
    chunk_a = _chunk(
        "00000000-0000-0000-0000-00000000000a", "Orion Hall seats up to 200 guests.", 0.9, 1
    )
    chunk_b = _chunk(
        "00000000-0000-0000-0000-00000000000b", "Orion Hall's ballroom fits 150 guests.", 0.8, 2
    )
    evidence = [
        Evidence(sub_query_id="sq1", chunk_id=chunk_a.chunk_id, fusion_score=0.9),
        Evidence(
            sub_query_id="sq2",
            chunk_id=chunk_b.chunk_id,
            fusion_score=0.8,
            duplicate_of="some-other-id",
        ),
    ]
    chunks_by_id = {chunk_a.chunk_id: chunk_a, chunk_b.chunk_id: chunk_b}
    flagged = flag_contradictions(evidence, chunks_by_id, "alpha")
    assert all(e.contradiction_pair_id is None for e in flagged)
