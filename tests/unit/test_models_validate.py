"""Phase 1 acceptance test (PRD_TRD.md §10, Phase 1 row)."""

from datetime import UTC, datetime

import pytest
from app.core.embeddings import embedding_dimension
from app.core.events import EventType, RetrievalTrigger, TelemetryEvent
from app.models.answer_version import AnswerVersion
from app.models.chunk import Chunk
from app.models.citation import Citation
from app.models.document import Document
from app.models.embedding import Embedding
from app.models.evaluation_result import EvaluationResult
from app.models.evidence import Evidence
from app.models.retrieval_event import RetrievalEvent, RetrievalMode
from app.models.session_state import Claim, SessionState
from app.models.sub_query import SubQuery
from pydantic import ValidationError


def test_document_requires_corpus_id():
    doc = Document(
        title="Venue Policies",
        source_path="data/corpus/default/venues.md",
        corpus_id="default",
        ingested_at=datetime.now(UTC),
        content_hash="abc123",
        section_count=4,
    )
    assert doc.doc_id
    assert doc.corpus_id == "default"

    with pytest.raises(ValidationError):
        Document(
            title="Missing corpus_id",
            source_path="x",
            ingested_at=datetime.now(UTC),
            content_hash="abc123",
            section_count=1,
        )


def test_chunk_defaults_and_required_fields():
    chunk = Chunk(
        doc_id="doc_1",
        section="§2",
        text="Cancellation within 48h is free.",
        token_count=8,
        chunk_index=1,
    )
    assert chunk.chunk_id
    assert chunk.bm25_tokens == []


def test_embedding_dimension_enforced():
    vector = [0.0] * embedding_dimension()
    emb = Embedding(
        chunk_id="chunk_1",
        vector=vector,
        model_name="BAAI/bge-small-en-v1.5",
        created_at=datetime.now(UTC),
    )
    assert len(emb.vector) == embedding_dimension()

    with pytest.raises(ValidationError):
        Embedding(
            chunk_id="chunk_1", vector=[0.0, 0.0], model_name="x", created_at=datetime.now(UTC)
        )


def test_retrieval_event_trigger_tagging():
    """REQ-OBS-05: trigger enum must accept 'multi_intent' (Theme 4 Guide contradiction fix)."""
    event = RetrievalEvent(
        session_id="sess_1",
        sub_query_id="sq_1",
        mode=RetrievalMode.DENSE,
        result_chunk_ids=["c1", "c2"],
        scores=[0.9, 0.7],
        latency_ms=120,
        t_offset_ms=1600,
        trigger=RetrievalTrigger.MULTI_INTENT,
    )
    assert event.trigger == "multi_intent"


def test_sub_query_and_evidence_and_citation_roundtrip():
    sq = SubQuery(
        session_id="sess_1",
        text="venue capacity",
        intent_label="capacity",
        parent_transcript_offset=800,
        created_at=datetime.now(UTC),
    )
    ev = Evidence(sub_query_id=sq.sub_query_id, chunk_id="chunk_1", fusion_score=0.42)
    assert ev.rerank_score is None
    citation = Citation(
        answer_version_id="av_1",
        chunk_id="chunk_1",
        doc_id="doc_1",
        section="§2",
        claim_text="Cancellation within 48h is free.",
        entailment_score=0.81,
    )
    assert citation.citation_id


def test_session_state_claim_status_and_topic_embedding_validation():
    claim = Claim(claim_id="claim_1", text="Capacity is 30.", chunk_id="chunk_1", origin_version=1)
    session = SessionState(
        session_id="sess_1",
        corpus_id="default",
        claims=[claim],
        created_at=datetime.now(UTC),
        last_active_at=datetime.now(UTC),
    )
    assert session.status == "active"
    assert session.claims[0].status == "active"

    with pytest.raises(ValidationError):
        SessionState(
            session_id="sess_1",
            corpus_id="default",
            topic_embedding=[0.1, 0.2],  # wrong dimension
            created_at=datetime.now(UTC),
            last_active_at=datetime.now(UTC),
        )


def test_session_state_requires_corpus_id():
    """REQ-CORPUS-02 / PRD_TRD.md §7.8: corpus_id is mandatory, not defaulted — a session with no
    corpus binding cannot exist, since later retrieval calls have no other way to learn it."""
    with pytest.raises(ValidationError):
        SessionState(
            session_id="sess_1", created_at=datetime.now(UTC), last_active_at=datetime.now(UTC)
        )


def test_session_state_corpus_id_scopes_retrieval_and_preserves_isolation():
    """REQ-CORPUS-02: corpus_id persists on SessionState so a retrieval call issued on a later
    turn (a separate TRANSCRIPT_CHUNK, potentially many turns after session creation) can still
    scope itself without the client resending it. Also proves session/corpus isolation: two
    sessions on different corpora never share state or resolve to the same index."""
    session_a = SessionState(
        session_id="sess_a",
        corpus_id="venues_corpus",
        created_at=datetime.now(UTC),
        last_active_at=datetime.now(UTC),
    )
    session_b = SessionState(
        session_id="sess_b",
        corpus_id="hr_policy_corpus",
        created_at=datetime.now(UTC),
        last_active_at=datetime.now(UTC),
    )

    # Simulates the scoping decision REQ-CORPUS-02 requires of the Hybrid Retriever (Phase 5):
    # every retrieval call resolves its index purely from session.corpus_id.
    def bm25_index_path_for(session: SessionState) -> str:
        return f"data/processed/bm25__{session.corpus_id}.pkl"

    assert bm25_index_path_for(session_a) == "data/processed/bm25__venues_corpus.pkl"
    assert bm25_index_path_for(session_b) == "data/processed/bm25__hr_policy_corpus.pkl"
    assert bm25_index_path_for(session_a) != bm25_index_path_for(session_b)

    # Session/corpus isolation: mutating one session's state never touches the other's.
    session_a.entities["location"] = "Pune"
    assert session_b.entities == {}
    assert session_b.corpus_id != session_a.corpus_id


def test_answer_version_supersedes_is_optional():
    av = AnswerVersion(
        session_id="sess_1", version_no=1, text="Answer text.", created_at=datetime.now(UTC)
    )
    assert av.supersedes is None


def test_evaluation_result_defaults_to_queued():
    result = EvaluationResult(
        test_set="benchmarks/streaming_suite_v1", corpus_id="default", started_at=datetime.now(UTC)
    )
    assert result.status == "queued"
    assert result.completed_at is None


def test_telemetry_event_envelope_is_frozen():
    event = TelemetryEvent(
        session_id="sess_1", event_type=EventType.SESSION_RESYNC, trace_id="trc_1"
    )
    assert event.event_id
    with pytest.raises(ValidationError):
        event.session_id = "sess_2"
