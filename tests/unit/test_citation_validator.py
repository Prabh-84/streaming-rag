"""Citation / Grounding Validator (pseudocode 12.F; REQ-GROUND-01/02/03).

Section labels here (`"§2"`, `"§3"`, ...) match the real shape scripts/ingest_corpus.py produces
(`split_sections`'s `f"§{label_no}"`) — never a human-readable heading like "Cancellation Policy",
which is `Section.title` and is baked into the chunk's own text, not carried as structured
metadata. Using the real shape here is what caught this module's original bug: a citation tag
format that quietly required section names to *not* already start with "§".
"""

from __future__ import annotations

from app.grounding.citation_validator import (
    Accept,
    Regenerate,
    Uncertain,
    build_evidence_lookup,
    contradiction_note,
    extract_citation_tags,
    finalize_sentence,
    format_citation_tag,
    lexical_overlap,
    stamp_citation_ids,
    uncertainty_text,
    validate_sentence,
)
from app.models.chunk import RetrievedChunk
from app.models.evidence import Evidence

ENTAILMENT_MIN = 0.30


def _chunk(chunk_id: str, doc_id: str, section: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        corpus_id="alpha",
        section=section,
        text=text,
        chunk_index=0,
        token_count=len(text.split()),
        score=0.9,
        rank=1,
    )


def _evidence(sub_query_id: str, chunk_id: str, **overrides) -> Evidence:
    defaults = {"fusion_score": 0.9, "rerank_score": 0.8}
    defaults.update(overrides)
    return Evidence(sub_query_id=sub_query_id, chunk_id=chunk_id, **defaults)


CHUNK_CANCEL = _chunk("c1", "orion_hall", "§3", "Orion Hall requires 14 days notice for a refund.")
CHUNK_CATERING = _chunk(
    "c2", "orion_hall", "§4", "Orion Hall offers vegetarian and gluten-free catering."
)
CANCEL_TAG = format_citation_tag("orion_hall", "§3")


def _lookup(*evidence_and_chunks):
    evidence = [ev for ev, _ in evidence_and_chunks]
    chunks_by_id = {c.chunk_id: c for _, c in evidence_and_chunks}
    return build_evidence_lookup(evidence, chunks_by_id)


# --- format_citation_tag / extract_citation_tags ------------------------------------------------


def test_format_citation_tag_does_not_double_the_section_marker():
    """section is already self-prefixed ("§3") - the rendered tag must not add a second "§"."""
    assert format_citation_tag("orion_hall", "§3") == "[orion_hall §3]"


def test_extract_citation_tags_parses_doc_id_and_section():
    tags = extract_citation_tags(f"Orion Hall needs 14 days notice {CANCEL_TAG}.")
    assert tags == ["orion_hall::§3"]


def test_extract_citation_tags_multiple():
    text = "See [a §1] and also [b §2]."
    assert extract_citation_tags(text) == ["a::§1", "b::§2"]


def test_extract_citation_tags_none():
    assert extract_citation_tags("No citation here.") == []


# --- build_evidence_lookup ----------------------------------------------------------------------


def test_build_evidence_lookup_keys_by_doc_id_and_section():
    lookup = _lookup((_evidence("sq1", "c1"), CHUNK_CANCEL))
    assert set(lookup) == {"orion_hall::§3"}
    assert lookup["orion_hall::§3"].chunk_id == "c1"


def test_build_evidence_lookup_excludes_duplicates():
    ev = _evidence("sq1", "c1", duplicate_of="some-other-evidence-id")
    lookup = _lookup((ev, CHUNK_CANCEL))
    assert lookup == {}


def test_build_evidence_lookup_prefers_higher_fusion_score_on_collision():
    """Two evidence rows resolving to the same (doc_id, section) - the higher-scored one wins the
    citation tag slot deterministically."""
    chunk_dup = _chunk("c1b", "orion_hall", "§3", "Slightly different wording.")
    low = _evidence("sq1", "c1", fusion_score=0.5)
    high = _evidence("sq1", "c1b", fusion_score=0.9)
    lookup = _lookup((low, CHUNK_CANCEL), (high, chunk_dup))
    assert lookup["orion_hall::§3"].chunk_id == "c1b"


# --- lexical_overlap --------------------------------------------------------------------------


def test_lexical_overlap_high_for_restated_content():
    score = lexical_overlap(
        "Orion Hall requires 14 days notice.", "Orion Hall requires 14 days notice for a refund."
    )
    assert score > 0.8


def test_lexical_overlap_low_for_unrelated_text():
    score = lexical_overlap("The weather is nice today.", CHUNK_CANCEL.text)
    assert score < 0.3


# --- validate_sentence / finalize_sentence ------------------------------------------------------


def test_valid_citation_is_accepted():
    lookup = _lookup((_evidence("sq1", "c1"), CHUNK_CANCEL))
    sentence = f"Orion Hall requires 14 days notice for a refund {CANCEL_TAG}."
    result = validate_sentence(sentence, lookup, entailment_min=ENTAILMENT_MIN)
    assert isinstance(result, Accept)
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == "c1"
    assert result.citations[0].doc_id == "orion_hall"


def test_fabricated_citation_id_is_rejected():
    lookup = _lookup((_evidence("sq1", "c1"), CHUNK_CANCEL))
    sentence = "Orion Hall has a strict policy [made_up_doc §99]."
    result = validate_sentence(sentence, lookup, entailment_min=ENTAILMENT_MIN)
    assert isinstance(result, Regenerate)
    assert result.reason == "fabricated_id"


def test_missing_citation_is_rejected():
    lookup = _lookup((_evidence("sq1", "c1"), CHUNK_CANCEL))
    result = validate_sentence(
        "Orion Hall requires 14 days notice.", lookup, entailment_min=ENTAILMENT_MIN
    )
    assert isinstance(result, Regenerate)
    assert result.reason == "missing_citation"


def test_low_entailment_is_rejected():
    lookup = _lookup((_evidence("sq1", "c1"), CHUNK_CANCEL))
    sentence = f"The moon is made of cheese {CANCEL_TAG}."
    result = validate_sentence(sentence, lookup, entailment_min=ENTAILMENT_MIN)
    assert isinstance(result, Regenerate)
    assert result.reason == "low_entailment"


def test_finalize_sentence_accept_passes_through():
    lookup = _lookup((_evidence("sq1", "c1"), CHUNK_CANCEL))
    sentence = f"Orion Hall requires 14 days notice for a refund {CANCEL_TAG}."
    result = finalize_sentence(sentence, lookup, entailment_min=ENTAILMENT_MIN)
    assert isinstance(result, Accept)


def test_finalize_sentence_failure_becomes_uncertain():
    lookup = _lookup((_evidence("sq1", "c1"), CHUNK_CANCEL))
    result = finalize_sentence("No citation at all.", lookup, entailment_min=ENTAILMENT_MIN)
    assert isinstance(result, Uncertain)
    assert result.reason == "missing_citation"


# --- stamp_citation_ids / uncertainty_text ------------------------------------------------------


def test_stamp_citation_ids_sets_answer_version_id():
    lookup = _lookup((_evidence("sq1", "c1"), CHUNK_CANCEL))
    sentence = f"Orion Hall requires 14 days notice for a refund {CANCEL_TAG}."
    result = validate_sentence(sentence, lookup, entailment_min=ENTAILMENT_MIN)
    stamped = stamp_citation_ids(result.citations, "av-123")
    assert all(c.answer_version_id == "av-123" for c in stamped)


def test_uncertainty_text_never_states_an_unsupported_fact():
    text = uncertainty_text("fabricated_id")
    assert "fabricated_id" in text
    assert "[" in text and "]" in text  # visibly marked, not blended into normal prose


# --- contradiction_note (REQ-GROUND-03) ----------------------------------------------------------


def test_contradiction_note_names_both_sides():
    chunk_high = _chunk("c1", "orion_hall", "§2", "Orion Hall seats up to 200 guests.")
    chunk_low = _chunk("c3", "lumen_pavilion", "§2", "Lumen Pavilion seats up to 150 guests.")
    ev_high = _evidence("sq1", "c1", contradiction_pair_id="pair-1")
    ev_low = _evidence("sq1", "c3", contradiction_pair_id="pair-1")
    lookup = _lookup((ev_high, chunk_high), (ev_low, chunk_low))
    note = contradiction_note(lookup, "pair-1")
    assert "[orion_hall §2]" in note
    assert "[lumen_pavilion §2]" in note
