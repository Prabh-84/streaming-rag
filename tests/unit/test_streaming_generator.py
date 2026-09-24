"""Streaming Answer Generator orchestration (pseudocode 12.I; REQ-GROUND-01/02/03) — uses
FakeGenerationLLM throughout so these are fast, deterministic, offline tests of the grounding
wiring, never dependent on a real Anthropic/Gemini call.

Section labels here (`"§2"`, `"§3"`, ...) match the real shape scripts/ingest_corpus.py produces,
never a human-readable heading — see test_citation_validator.py's module docstring for why that
distinction matters (it's what caught a real double-"§" formatting bug during development).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.core.events import EventType
from app.generation.streaming_generator import stream_answer
from app.grounding.citation_validator import format_citation_tag
from app.models.answer_version import AnswerVersion
from app.models.chunk import RetrievedChunk
from app.models.evidence import Evidence
from app.models.retrieval_event import RetrievalResult
from app.models.sub_query import SubQuery
from app.session.session_store import SessionStore
from tests.fakes import FIXTURE_CORPORA, FakeGenerationLLM, RecordingSink, make_settings


@pytest.fixture
def settings(tmp_path):
    return make_settings(FIXTURE_CORPORA, tmp_path, MIN_RELEVANCE=0.35, ENTAILMENT_MIN=0.30)


@pytest.fixture
async def session(settings):
    store = SessionStore(settings)
    return await store.create("alpha")


def _chunk(
    chunk_id: str, doc_id: str, section: str, text: str, corpus_id: str = "alpha"
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        corpus_id=corpus_id,
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


def _sub_query(text: str, session_id: str = "sess_1", intent_label: str = "single") -> SubQuery:
    return SubQuery(
        session_id=session_id,
        text=text,
        intent_label=intent_label,
        parent_transcript_offset=0,
        created_at=datetime.now(UTC),
    )


def _retrieval_result(
    sub_query_id: str, chunks: list[RetrievedChunk], corpus_id: str = "alpha"
) -> RetrievalResult:
    return RetrievalResult(sub_query_id=sub_query_id, corpus_id=corpus_id, dense=chunks, sparse=[])


CHUNK_CANCEL = _chunk(
    "c-cancel", "orion_hall", "§3", "Orion Hall requires 14 days notice for a refund."
)
CHUNK_CATERING = _chunk(
    "c-catering",
    "orion_hall",
    "§4",
    "Orion Hall offers vegetarian and gluten-free catering options.",
)
CANCEL_TAG = format_citation_tag("orion_hall", "§3")
CATERING_TAG = format_citation_tag("orion_hall", "§4")


# --- 1. Citations map to actual retrieved evidence -----------------------------------------------


async def test_citations_map_to_actual_retrieved_evidence(session, settings):
    sub_queries = [_sub_query("What is the cancellation policy for Orion Hall?")]
    evidence = [_evidence("sq1", CHUNK_CANCEL.chunk_id)]
    retrievals = [_retrieval_result("sq1", [CHUNK_CANCEL])]
    sink = RecordingSink()
    llm = FakeGenerationLLM(
        chunks=[f"Orion Hall requires 14 days notice for a refund {CANCEL_TAG}."]
    )

    answer = await stream_answer(
        session,
        sub_queries,
        evidence,
        retrievals,
        trace_id="t1",
        llm=llm,
        settings=settings,
        sink=sink,
    )

    assert answer is not None
    citation_events = sink.of_type(EventType.CITATION_CREATED)
    assert len(citation_events) == 1
    assert citation_events[0].payload["chunk_id"] == CHUNK_CANCEL.chunk_id
    assert citation_events[0].payload["doc_id"] == "orion_hall"
    assert citation_events[0].payload["section"] == "§3"
    assert len(answer.citations) == 1


# --- 2. Every factual answer claim has supporting evidence ---------------------------------------


async def test_sentence_without_citation_becomes_uncertainty_not_a_bare_claim(session, settings):
    """A sentence the model produced with no citation tag must never reach the answer as an
    unsupported fact - REQ-GROUND-01's mandatory-citation requirement."""
    sub_queries = [_sub_query("What is the cancellation policy for Orion Hall?")]
    evidence = [_evidence("sq1", CHUNK_CANCEL.chunk_id)]
    retrievals = [_retrieval_result("sq1", [CHUNK_CANCEL])]
    sink = RecordingSink()
    llm = FakeGenerationLLM(
        chunks=["Orion Hall requires 14 days notice."],  # no citation tag anywhere
        complete_response="Still no citation here either.",  # regeneration also fails to cite
    )

    answer = await stream_answer(
        session,
        sub_queries,
        evidence,
        retrievals,
        trace_id="t1",
        llm=llm,
        settings=settings,
        sink=sink,
    )

    assert answer is not None
    assert "Unable to verify" in answer.text
    assert answer.citations == []
    uncertainty_events = sink.of_type(EventType.UNCERTAINTY)
    assert len(uncertainty_events) == 1
    assert uncertainty_events[0].payload["reason"] == "missing_citation"
    assert len(llm.complete_calls) == 1  # exactly one regeneration attempt (pseudocode 12.F)


# --- 3. Insufficient evidence produces appropriate uncertainty (REQ-GROUND-02) --------------------


async def test_no_evidence_at_all_produces_uncertainty_and_never_calls_the_llm(session, settings):
    sub_queries = [_sub_query("What is the cancellation policy for Orion Hall?")]
    sink = RecordingSink()
    llm = FakeGenerationLLM(chunks=["should never be used"])

    answer = await stream_answer(
        session, sub_queries, [], [], trace_id="t1", llm=llm, settings=settings, sink=sink
    )

    assert answer is None
    assert llm.stream_calls == []  # never even attempted generation
    uncertainty_events = sink.of_type(EventType.UNCERTAINTY)
    assert len(uncertainty_events) == 1
    assert uncertainty_events[0].payload["reason"] == "insufficient_evidence"
    assert uncertainty_events[0].payload["clarifying_question"]


async def test_low_relevance_evidence_produces_uncertainty(session, settings):
    """Fused evidence exists but every rerank_score is below MIN_RELEVANCE - REQ-GROUND-02's exact
    trigger condition."""
    sub_queries = [_sub_query("What is the cancellation policy?")]
    evidence = [_evidence("sq1", CHUNK_CANCEL.chunk_id, rerank_score=0.1)]
    retrievals = [_retrieval_result("sq1", [CHUNK_CANCEL])]
    sink = RecordingSink()
    llm = FakeGenerationLLM(chunks=["should never be used"])

    answer = await stream_answer(
        session,
        sub_queries,
        evidence,
        retrievals,
        trace_id="t1",
        llm=llm,
        settings=settings,
        sink=sink,
    )

    assert answer is None
    assert llm.stream_calls == []
    assert sink.of_type(EventType.UNCERTAINTY)[0].payload["reason"] == "insufficient_evidence"


# --- 4. Fabricated chunk/doc IDs are rejected (REQ-GROUND-01) -------------------------------------


async def test_fabricated_citation_is_never_accepted(session, settings):
    sub_queries = [_sub_query("What is the cancellation policy for Orion Hall?")]
    evidence = [_evidence("sq1", CHUNK_CANCEL.chunk_id)]
    retrievals = [_retrieval_result("sq1", [CHUNK_CANCEL])]
    sink = RecordingSink()
    llm = FakeGenerationLLM(
        chunks=["Orion Hall has a lenient policy [made_up_doc §99]."],
        complete_response="Still fabricated [also_made_up §98].",
    )

    answer = await stream_answer(
        session,
        sub_queries,
        evidence,
        retrievals,
        trace_id="t1",
        llm=llm,
        settings=settings,
        sink=sink,
    )

    assert answer is not None
    assert answer.citations == []  # the fabricated tag never became a real citation
    citation_events = sink.of_type(EventType.CITATION_CREATED)
    assert all(e.payload["doc_id"] != "made_up_doc" for e in citation_events)
    assert all(e.payload["doc_id"] != "also_made_up" for e in citation_events)
    assert sink.of_type(EventType.UNCERTAINTY)[0].payload["reason"] == "fabricated_id"
    assert "Unable to verify" in answer.text


async def test_regeneration_recovers_a_fixable_sentence(session, settings):
    """The one-retry regeneration loop (pseudocode 12.F) succeeds when the corrected sentence
    cites a real tag - proving retry isn't just a formality that always fails."""
    sub_queries = [_sub_query("What is the cancellation policy for Orion Hall?")]
    evidence = [_evidence("sq1", CHUNK_CANCEL.chunk_id)]
    retrievals = [_retrieval_result("sq1", [CHUNK_CANCEL])]
    sink = RecordingSink()
    llm = FakeGenerationLLM(
        chunks=["Orion Hall has a lenient policy [made_up_doc §97]."],
        complete_response=f"Orion Hall requires 14 days notice for a refund {CANCEL_TAG}.",
    )

    answer = await stream_answer(
        session,
        sub_queries,
        evidence,
        retrievals,
        trace_id="t1",
        llm=llm,
        settings=settings,
        sink=sink,
    )

    assert len(llm.complete_calls) == 1
    assert len(answer.citations) == 1
    assert sink.of_type(EventType.CITATION_CREATED)[0].payload["chunk_id"] == CHUNK_CANCEL.chunk_id
    assert sink.of_type(EventType.UNCERTAINTY) == []  # recovered - no uncertainty needed


# --- 5. Contradiction pairs are surfaced (REQ-GROUND-03) ------------------------------------------


async def test_contradiction_pair_sentence_with_both_citations_is_accepted(session, settings):
    chunk_high = _chunk("c-high", "orion_hall", "§2", "Orion Hall seats up to 200 guests.")
    chunk_low = _chunk("c-low", "lumen_pavilion", "§2", "Lumen Pavilion seats up to 150 guests.")
    evidence = [
        _evidence("sq1", chunk_high.chunk_id, contradiction_pair_id="pair-1"),
        _evidence("sq1", chunk_low.chunk_id, contradiction_pair_id="pair-1"),
    ]
    retrievals = [_retrieval_result("sq1", [chunk_high, chunk_low])]
    sink = RecordingSink()
    high_tag = format_citation_tag("orion_hall", "§2")
    low_tag = format_citation_tag("lumen_pavilion", "§2")
    sentence = (
        f"Sources disagree on capacity: Orion Hall seats up to 200 guests {high_tag}, "
        f"while Lumen Pavilion seats up to 150 guests {low_tag}."
    )
    llm = FakeGenerationLLM(chunks=[sentence])

    answer = await stream_answer(
        session,
        sub_queries=[_sub_query("What is the capacity?")],
        evidence=evidence,
        retrievals=retrievals,
        trace_id="t1",
        llm=llm,
        settings=settings,
        sink=sink,
    )

    assert answer is not None
    assert len(answer.citations) == 2
    cited_doc_ids = {e.payload["doc_id"] for e in sink.of_type(EventType.CITATION_CREATED)}
    assert cited_doc_ids == {"orion_hall", "lumen_pavilion"}


# --- 6. Multi-intent answers retain evidence per sub-intent (REQ-EVID / REQ-INTENT interaction) ---


async def test_multi_intent_answer_retains_a_citation_per_sub_intent(session, settings):
    sub_queries = [
        _sub_query("What is the cancellation policy for Orion Hall?"),
        _sub_query("What are the catering options for Orion Hall?"),
    ]
    evidence = [
        _evidence(sub_queries[0].sub_query_id, CHUNK_CANCEL.chunk_id),
        _evidence(sub_queries[1].sub_query_id, CHUNK_CATERING.chunk_id),
    ]
    retrievals = [
        _retrieval_result(sub_queries[0].sub_query_id, [CHUNK_CANCEL]),
        _retrieval_result(sub_queries[1].sub_query_id, [CHUNK_CATERING]),
    ]
    sink = RecordingSink()
    llm = FakeGenerationLLM(
        chunks=[
            f"Orion Hall requires 14 days notice for a refund {CANCEL_TAG}. ",
            f"Orion Hall offers vegetarian and gluten-free catering options {CATERING_TAG}.",
        ]
    )

    answer = await stream_answer(
        session,
        sub_queries,
        evidence,
        retrievals,
        trace_id="t1",
        llm=llm,
        settings=settings,
        sink=sink,
    )

    assert len(answer.citations) == 2
    cited_sections = {e.payload["section"] for e in sink.of_type(EventType.CITATION_CREATED)}
    assert cited_sections == {"§3", "§4"}


# --- 7. Session refinement preserves grounding (Phase 6 interaction) -----------------------------


async def test_second_turn_supersedes_first_and_still_enforces_grounding(session, settings):
    """A refinement turn (an existing answer_versions entry) must still reject fabricated
    citations and correctly link version_no/supersedes - grounding rules don't relax after
    turn 1."""
    sub_queries = [_sub_query("What is the cancellation policy for Orion Hall?")]
    evidence = [_evidence("sq1", CHUNK_CANCEL.chunk_id)]
    retrievals = [_retrieval_result("sq1", [CHUNK_CANCEL])]
    sink = RecordingSink()
    first_llm = FakeGenerationLLM(
        chunks=[f"Orion Hall requires 14 days notice for a refund {CANCEL_TAG}."]
    )
    first = await stream_answer(
        session,
        sub_queries,
        evidence,
        retrievals,
        trace_id="t1",
        llm=first_llm,
        settings=settings,
        sink=sink,
    )
    assert first.version_no == 1
    assert first.supersedes is None
    assert session.answer_versions == [first]

    second_sub_queries = [_sub_query("What about catering?")]
    second_evidence = [_evidence("sq2", CHUNK_CATERING.chunk_id)]
    second_retrievals = [_retrieval_result("sq2", [CHUNK_CATERING])]
    second_llm = FakeGenerationLLM(
        chunks=["Fabricated claim [nowhere §96]."],
        complete_response="Still fabricated [nowhere §96].",
    )
    second = await stream_answer(
        session,
        second_sub_queries,
        second_evidence,
        second_retrievals,
        trace_id="t2",
        llm=second_llm,
        settings=settings,
        sink=sink,
    )

    assert second.version_no == 2
    assert second.supersedes == 1
    assert second.citations == []  # grounding still rejects fabrication on turn 2
    assert session.answer_versions == [first, second]


# --- 8. Corpus isolation is preserved ------------------------------------------------------------


async def test_generation_never_mixes_evidence_across_corpora(settings):
    """Two independent sessions on different corpora, each producing a grounded answer - session
    A's citations must only ever reference alpha chunks, session B's only ever beta chunks."""
    store = SessionStore(settings)
    session_alpha = await store.create("alpha")
    session_beta = await store.create("beta")

    chunk_alpha = _chunk(
        "c-alpha", "orion_hall", "§3", "Orion Hall requires 14 days notice.", corpus_id="alpha"
    )
    chunk_beta = _chunk(
        "c-beta",
        "travel_policy",
        "§2",
        "International trips are reimbursed in EUR.",
        corpus_id="beta",
    )
    alpha_tag = format_citation_tag("orion_hall", "§3")
    beta_tag = format_citation_tag("travel_policy", "§2")

    sink = RecordingSink()
    answer_alpha = await stream_answer(
        session_alpha,
        [_sub_query("What is the cancellation policy?")],
        [_evidence("sqA", chunk_alpha.chunk_id)],
        [_retrieval_result("sqA", [chunk_alpha], corpus_id="alpha")],
        trace_id="tA",
        llm=FakeGenerationLLM(chunks=[f"Orion Hall requires 14 days notice {alpha_tag}."]),
        settings=settings,
        sink=sink,
    )
    answer_beta = await stream_answer(
        session_beta,
        [_sub_query("What currency are international trips reimbursed in?")],
        [_evidence("sqB", chunk_beta.chunk_id)],
        [_retrieval_result("sqB", [chunk_beta], corpus_id="beta")],
        trace_id="tB",
        llm=FakeGenerationLLM(chunks=[f"International trips are reimbursed in EUR {beta_tag}."]),
        settings=settings,
        sink=sink,
    )

    assert answer_alpha.citations and answer_beta.citations
    alpha_citation_doc_ids = {
        e.payload["doc_id"]
        for e in sink.of_type(EventType.CITATION_CREATED)
        if e.session_id == session_alpha.session_id
    }
    beta_citation_doc_ids = {
        e.payload["doc_id"]
        for e in sink.of_type(EventType.CITATION_CREATED)
        if e.session_id == session_beta.session_id
    }
    assert alpha_citation_doc_ids == {"orion_hall"}
    assert beta_citation_doc_ids == {"travel_policy"}


# --- 9. First-turn behavior remains correct -------------------------------------------------------


async def test_first_turn_produces_version_one_with_no_supersedes(session, settings):
    sub_queries = [_sub_query("What is the cancellation policy for Orion Hall?")]
    evidence = [_evidence("sq1", CHUNK_CANCEL.chunk_id)]
    retrievals = [_retrieval_result("sq1", [CHUNK_CANCEL])]
    llm = FakeGenerationLLM(
        chunks=[f"Orion Hall requires 14 days notice for a refund {CANCEL_TAG}."]
    )

    answer = await stream_answer(
        session, sub_queries, evidence, retrievals, trace_id="t1", llm=llm, settings=settings
    )

    assert answer.version_no == 1
    assert answer.supersedes is None
    assert isinstance(answer, AnswerVersion)


async def test_no_sub_queries_produces_no_answer(session, settings):
    llm = FakeGenerationLLM(chunks=["should never be used"])
    answer = await stream_answer(session, [], [], [], trace_id="t1", llm=llm, settings=settings)
    assert answer is None
    assert llm.stream_calls == []
