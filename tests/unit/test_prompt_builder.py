"""Prompt construction for the Streaming Answer Generator (pseudocode 12.I)."""

from __future__ import annotations

from datetime import UTC, datetime

from app.generation.prompt_builder import build_prompt
from app.grounding.citation_validator import build_evidence_lookup
from app.models.answer_version import AnswerVersion
from app.models.chunk import RetrievedChunk
from app.models.evidence import Evidence
from app.models.sub_query import SubQuery


def _sub_query(text: str) -> SubQuery:
    return SubQuery(
        session_id="sess_1",
        text=text,
        intent_label="single",
        parent_transcript_offset=0,
        created_at=datetime.now(UTC),
    )


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


def test_prompt_lists_sub_queries_and_evidence_with_citation_tags():
    sub_queries = [_sub_query("What is the cancellation policy for Orion Hall?")]
    chunk = _chunk("c1", "orion_hall", "§3", "Requires 14 days notice.")
    evidence = [Evidence(sub_query_id="sq1", chunk_id="c1", fusion_score=0.9)]
    lookup = build_evidence_lookup(evidence, {"c1": chunk})

    prompt = build_prompt(sub_queries, lookup)

    assert "What is the cancellation policy for Orion Hall?" in prompt
    assert "[orion_hall §3]: Requires 14 days notice." in prompt


def test_prompt_includes_contradiction_note_when_flagged():
    chunk_a = _chunk("c1", "orion_hall", "§2", "Seats 200 guests.")
    chunk_b = _chunk("c2", "lumen_pavilion", "§2", "Seats 150 guests.")
    evidence = [
        Evidence(sub_query_id="sq1", chunk_id="c1", fusion_score=0.9, contradiction_pair_id="p1"),
        Evidence(sub_query_id="sq1", chunk_id="c2", fusion_score=0.8, contradiction_pair_id="p1"),
    ]
    lookup = build_evidence_lookup(evidence, {"c1": chunk_a, "c2": chunk_b})

    prompt = build_prompt([_sub_query("What is the capacity?")], lookup)

    assert "disagree" in prompt.lower()
    assert "[orion_hall §2]" in prompt
    assert "[lumen_pavilion §2]" in prompt


def test_prompt_omits_contradiction_note_when_not_flagged():
    chunk = _chunk("c1", "orion_hall", "§2", "Seats 200 guests.")
    evidence = [Evidence(sub_query_id="sq1", chunk_id="c1", fusion_score=0.9)]
    lookup = build_evidence_lookup(evidence, {"c1": chunk})
    prompt = build_prompt([_sub_query("What is the capacity?")], lookup)
    assert "disagree" not in prompt.lower()


def test_prompt_includes_prior_version_for_refinement_turns():
    chunk = _chunk("c1", "orion_hall", "§2", "Seats 200 guests.")
    evidence = [Evidence(sub_query_id="sq1", chunk_id="c1", fusion_score=0.9)]
    lookup = build_evidence_lookup(evidence, {"c1": chunk})
    prior = AnswerVersion(
        session_id="sess_1",
        version_no=1,
        text="Orion Hall seats 200 guests.",
        created_at=datetime.now(UTC),
    )

    prompt = build_prompt([_sub_query("What about capacity?")], lookup, prior_version=prior)

    assert "Orion Hall seats 200 guests." in prompt
    assert "version 1" in prompt


def test_prompt_deterministic_for_same_inputs():
    chunk = _chunk("c1", "orion_hall", "§2", "Seats 200 guests.")
    evidence = [Evidence(sub_query_id="sq1", chunk_id="c1", fusion_score=0.9)]
    lookup = build_evidence_lookup(evidence, {"c1": chunk})
    sub_queries = [_sub_query("What is the capacity?")]
    assert build_prompt(sub_queries, lookup) == build_prompt(sub_queries, lookup)
