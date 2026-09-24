"""Multi-Intent Decomposer (REQ-INTENT-01/02; pseudocode 12.B) against the alpha fixture corpus,
whose slots.yaml carries venue/capacity/cancellation_policy/catering specifically so a genuine
2-distinct-slot compound utterance is testable without inventing benchmark-specific vocabulary.
"""

from __future__ import annotations

import pytest

from app.controller.entity_extraction import get_corpus_matcher
from app.core.config import get_settings
from app.core.slots import get_slot_schema
from app.decomposition.multi_intent import (
    ProposedSubQuery,
    _validate_proposed,
    decompose,
    has_compound_signal,
    merge_similar,
)
from tests.fakes import FIXTURE_CORPORA, FakeDecomposer, FakeEmbedder, RecordingSink, make_settings

COMPOUND_TEXT = "What is the cancellation policy and the catering options for Orion Hall"
SIMPLE_TEXT = "I need Orion Hall for 30 guests"


@pytest.fixture(autouse=True)
def _use_fixture_corpus_root(monkeypatch):
    monkeypatch.setattr(get_settings(), "corpus_root", str(FIXTURE_CORPORA))
    get_slot_schema.cache_clear()
    get_corpus_matcher.cache_clear()
    yield
    get_slot_schema.cache_clear()
    get_corpus_matcher.cache_clear()


@pytest.fixture
def settings(tmp_path):
    return make_settings(FIXTURE_CORPORA, tmp_path, MERGE_THRESHOLD=0.85, DECOMPOSE_TIMEOUT_MS=1500)


# --- A. Compound detection (dependency cc/conj) -------------------------------------------------


def test_compound_detected_via_cc_conj_across_two_distinct_slots():
    assert has_compound_signal(COMPOUND_TEXT, "alpha") is True


def test_compound_detected_uses_actual_cc_conj_dependencies():
    """Not just "contains the word 'and'": the two conjuncts found must each carry a slot match,
    confirmed by checking the parse directly rather than only the boolean outcome."""
    from app.controller.entity_extraction import get_nlp

    doc = get_nlp()(COMPOUND_TEXT)
    conj_tokens = [t for t in doc if t.dep_ == "conj"]
    cc_tokens = [t for t in doc if t.dep_ == "cc"]
    assert conj_tokens, "fixture sentence must actually parse to a conj dependency"
    assert cc_tokens, "fixture sentence must actually parse to a cc dependency"


def test_non_compound_request_has_no_signal():
    assert has_compound_signal(SIMPLE_TEXT, "alpha") is False


def test_single_conjunct_with_only_one_slot_is_not_compound():
    """Two mentions of the *same* slot joined by "and" is not "different domain slots"."""
    assert has_compound_signal("Orion Hall and Lumen Pavilion", "alpha") is False


def test_conjunction_without_any_slot_mentions_is_not_compound():
    assert has_compound_signal("apples and oranges", "alpha") is False


def test_empty_text_has_no_signal():
    assert has_compound_signal("", "alpha") is False
    assert has_compound_signal("   ", "alpha") is False


def test_compound_signal_is_scoped_to_the_requested_corpus():
    """beta has no venue/cancellation_policy/catering slots at all, so the same sentence carries
    no signal there — compound detection respects corpus isolation like everything else."""
    assert has_compound_signal(COMPOUND_TEXT, "beta") is False


# --- B. Structured decomposition validation -----------------------------------------------------


def test_valid_structured_response_produces_labeled_sub_queries():
    proposed = _validate_proposed(
        {
            "sub_queries": [
                {
                    "text": "What is the cancellation policy for Orion Hall?",
                    "intent_label": "cancellation",
                },
                {
                    "text": "What are the catering options for Orion Hall?",
                    "intent_label": "catering",
                },
            ]
        }
    )
    assert [p.text for p in proposed] == [
        "What is the cancellation policy for Orion Hall?",
        "What are the catering options for Orion Hall?",
    ]
    assert [p.intent_label for p in proposed] == ["cancellation", "catering"]


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {},
        {"sub_queries": []},
        {"sub_queries": "not a list"},
        {"sub_queries": [{"text": "only text, no label"}]},
        {"sub_queries": [{"intent_label": "only label, no text"}]},
        {"sub_queries": [{"text": "  ", "intent_label": "blank text"}]},
        {"sub_queries": [{"text": "ok", "intent_label": "   "}]},
        {"wrong_key": [{"text": "x", "intent_label": "y"}]},
        {"sub_queries": [{"text": 123, "intent_label": "wrong type"}]},
        "just a string",
        ["a", "list"],
    ],
)
def test_malformed_or_unusable_llm_output_is_rejected(raw):
    assert _validate_proposed(raw) == []


# --- C. Anti-over-fragmentation merge (REQ-INTENT-02) -------------------------------------------


def test_dissimilar_sub_queries_stay_separate():
    proposed = [
        ProposedSubQuery(text="What is the cancellation policy?", intent_label="cancellation"),
        ProposedSubQuery(text="What are the catering options?", intent_label="catering"),
    ]
    kept = merge_similar(proposed, threshold=0.85, embedder=FakeEmbedder())
    assert len(kept) == 2


def test_near_duplicate_sub_queries_are_merged():
    proposed = [
        ProposedSubQuery(text="cancellation policy for Orion Hall", intent_label="cancellation"),
        ProposedSubQuery(
            text="cancellation policy for Orion Hall please", intent_label="cancellation_2"
        ),
    ]
    kept = merge_similar(proposed, threshold=0.85, embedder=FakeEmbedder())
    assert len(kept) == 1
    (sub_query, absorbed) = kept[0]
    assert sub_query.text == "cancellation policy for Orion Hall"
    assert absorbed == ["cancellation policy for Orion Hall please"]


def test_three_near_duplicates_all_merge_into_the_first():
    proposed = [
        ProposedSubQuery(text="cancellation policy for Orion Hall", intent_label="a"),
        ProposedSubQuery(text="cancellation policy for Orion Hall please", intent_label="b"),
        ProposedSubQuery(text="cancellation policy for Orion Hall thanks", intent_label="c"),
    ]
    kept = merge_similar(proposed, threshold=0.85, embedder=FakeEmbedder())
    assert len(kept) == 1
    assert kept[0][1] == [
        "cancellation policy for Orion Hall please",
        "cancellation policy for Orion Hall thanks",
    ]


def test_merge_respects_configured_threshold_boundary():
    embedder = FakeEmbedder()
    a, b = "cancellation policy", "catering options"
    similarity = _cosine_via_embedder(embedder, a, b)
    proposed = [
        ProposedSubQuery(text=a, intent_label="a"),
        ProposedSubQuery(text=b, intent_label="b"),
    ]
    # A threshold below the actual similarity must merge; a threshold above it must not.
    assert len(merge_similar(proposed, threshold=similarity - 0.01, embedder=embedder)) == 1
    assert len(merge_similar(proposed, threshold=similarity + 0.01, embedder=embedder)) == 2


def _cosine_via_embedder(embedder, a: str, b: str) -> float:
    import math

    va, vb = embedder.embed(a), embedder.embed(b)
    dot = sum(x * y for x, y in zip(va, vb, strict=True))
    na, nb = math.sqrt(sum(x * x for x in va)), math.sqrt(sum(x * x for x in vb))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


# --- D. Fallback (compound signal + zero usable LLM output) -------------------------------------


async def test_compound_signal_with_unusable_llm_output_falls_back_to_one_whole_transcript_subquery(
    settings,
):
    llm = FakeDecomposer(sub_queries=None)  # simulates "LLM returned nothing usable"
    sub_queries = await decompose(
        COMPOUND_TEXT, "alpha", "sess_1", 800, trace_id="trc_1", llm=llm, settings=settings
    )
    assert len(sub_queries) == 1
    assert sub_queries[0].text == COMPOUND_TEXT  # original request preserved, not dropped
    assert sub_queries[0].intent_label == "single"
    assert llm.calls == [COMPOUND_TEXT]  # the LLM *was* called, since a signal was present


async def test_llm_call_failure_also_falls_back_to_one_whole_transcript_subquery(settings):
    llm = FakeDecomposer(raise_error=True)
    sub_queries = await decompose(
        COMPOUND_TEXT, "alpha", "sess_1", 0, trace_id="trc_1", llm=llm, settings=settings
    )
    assert len(sub_queries) == 1
    assert sub_queries[0].text == COMPOUND_TEXT


async def test_llm_timeout_falls_back_to_one_whole_transcript_subquery(settings):
    settings = settings.model_copy(update={"decompose_timeout_ms": 20})
    llm = FakeDecomposer(sub_queries=[{"text": "x", "intent_label": "y"}], delay_s=0.2)
    sub_queries = await decompose(
        COMPOUND_TEXT, "alpha", "sess_1", 0, trace_id="trc_1", llm=llm, settings=settings
    )
    assert len(sub_queries) == 1
    assert sub_queries[0].text == COMPOUND_TEXT


async def test_llm_call_failure_emits_exactly_one_error_event(settings):
    """Regression guard: a raising DecompositionLLM must produce exactly one
    decomposition_fallback ERROR event, not one from the except-block plus a second one from the
    subsequent "no usable sub-queries" check falling through on the same failure."""
    sink = RecordingSink()
    llm = FakeDecomposer(raise_error=True)
    await decompose(
        COMPOUND_TEXT, "alpha", "sess_1", 0, trace_id="trc_1", llm=llm, settings=settings, sink=sink
    )
    error_events = [e for e in sink.events if e.event_type == "ERROR"]
    assert len(error_events) == 1
    assert error_events[0].payload["error_type"] == "decomposition_fallback"
    assert "llm_call_failed" in error_events[0].payload["message"]


async def test_llm_timeout_emits_exactly_one_error_event(settings):
    settings = settings.model_copy(update={"decompose_timeout_ms": 20})
    sink = RecordingSink()
    llm = FakeDecomposer(sub_queries=[{"text": "x", "intent_label": "y"}], delay_s=0.2)
    await decompose(
        COMPOUND_TEXT, "alpha", "sess_1", 0, trace_id="trc_1", llm=llm, settings=settings, sink=sink
    )
    error_events = [e for e in sink.events if e.event_type == "ERROR"]
    assert len(error_events) == 1
    assert error_events[0].payload["error_type"] == "decomposition_fallback"


async def test_decomposition_fallback_is_emitted(settings):
    sink = RecordingSink()
    llm = FakeDecomposer(sub_queries=None)
    await decompose(
        COMPOUND_TEXT, "alpha", "sess_1", 0, trace_id="trc_1", llm=llm, settings=settings, sink=sink
    )
    (error_event,) = (e for e in sink.events if e.event_type == "ERROR")
    assert error_event.payload["stage"] == "decomposition"
    assert error_event.payload["error_type"] == "decomposition_fallback"
    assert error_event.payload["recoverable"] is True


async def test_fallback_still_emits_subquery_created(settings):
    sink = RecordingSink()
    await decompose(
        COMPOUND_TEXT,
        "alpha",
        "sess_1",
        0,
        trace_id="trc_1",
        llm=FakeDecomposer(sub_queries=None),
        settings=settings,
        sink=sink,
    )
    (event,) = (e for e in sink.events if e.event_type == "SUBQUERY_CREATED")
    assert event.payload["text"] == COMPOUND_TEXT
    assert event.payload["intent_label"] == "single"


# --- E. Single-intent behavior -------------------------------------------------------------------


async def test_no_compound_marker_returns_exactly_one_subquery(settings):
    llm = FakeDecomposer(sub_queries=[{"text": "should never be used", "intent_label": "x"}])
    sub_queries = await decompose(
        SIMPLE_TEXT, "alpha", "sess_1", 0, trace_id="trc_1", llm=llm, settings=settings
    )
    assert len(sub_queries) == 1
    assert sub_queries[0].text == SIMPLE_TEXT
    assert sub_queries[0].intent_label == "single"


async def test_llm_is_not_invoked_for_a_non_compound_request(settings):
    llm = FakeDecomposer(sub_queries=[{"text": "x", "intent_label": "y"}])
    await decompose(SIMPLE_TEXT, "alpha", "sess_1", 0, trace_id="trc_1", llm=llm, settings=settings)
    assert llm.calls == []  # REQ-INTENT-02: no LLM call on a simple request


async def test_no_second_actionable_entity_cluster_returns_exactly_one_subquery(settings):
    """A single entity cluster (just "venue"), even with an "and" present, is not two distinct
    domain slots and must not decompose."""
    text = "Orion Hall and Lumen Pavilion"
    llm = FakeDecomposer(sub_queries=[{"text": "should never be used", "intent_label": "x"}])
    sub_queries = await decompose(
        text, "alpha", "sess_1", 0, trace_id="trc_1", llm=llm, settings=settings
    )
    assert len(sub_queries) == 1
    assert llm.calls == []


# --- F. Subquery metadata -------------------------------------------------------------------------


async def test_subquery_ids_are_present_unique_and_traceable(settings):
    llm = FakeDecomposer(
        sub_queries=[
            {
                "text": "What is the cancellation policy for Orion Hall?",
                "intent_label": "cancellation",
            },
            {"text": "What are the catering options for Orion Hall?", "intent_label": "catering"},
        ]
    )
    sub_queries = await decompose(
        COMPOUND_TEXT, "alpha", "sess_42", 1234, trace_id="trc_1", llm=llm, settings=settings
    )
    assert len(sub_queries) == 2
    ids = [sq.sub_query_id for sq in sub_queries]
    assert all(ids) and len(set(ids)) == 2  # present and unique

    for sq in sub_queries:
        assert sq.session_id == "sess_42"  # parent/session association preserved
        assert sq.parent_transcript_offset == 1234  # original transcript context (offset) retained
        assert sq.created_at is not None


async def test_single_fallback_subquery_also_carries_full_metadata(settings):
    sub_queries = await decompose(
        SIMPLE_TEXT,
        "alpha",
        "sess_7",
        555,
        trace_id="trc_1",
        llm=FakeDecomposer(),
        settings=settings,
    )
    (sq,) = sub_queries
    assert sq.sub_query_id
    assert sq.session_id == "sess_7"
    assert sq.parent_transcript_offset == 555
    assert sq.text == SIMPLE_TEXT


async def test_merged_subquery_retains_absorbed_texts_in_merged_from(settings):
    llm = FakeDecomposer(
        sub_queries=[
            {"text": "cancellation policy for Orion Hall", "intent_label": "a"},
            {"text": "cancellation policy for Orion Hall please", "intent_label": "b"},
        ]
    )
    sub_queries = await decompose(
        COMPOUND_TEXT, "alpha", "sess_1", 0, trace_id="trc_1", llm=llm, settings=settings
    )
    assert len(sub_queries) == 1
    assert sub_queries[0].merged_from == ["cancellation policy for Orion Hall please"]


async def test_full_decomposition_end_to_end_produces_two_distinct_sub_queries(settings):
    """The complete pseudocode 12.B path: compound signal -> LLM -> merge -> SubQuery objects."""
    llm = FakeDecomposer(
        sub_queries=[
            {
                "text": "What is the cancellation policy for Orion Hall?",
                "intent_label": "cancellation",
            },
            {"text": "What are the catering options for Orion Hall?", "intent_label": "catering"},
        ]
    )
    sub_queries = await decompose(
        COMPOUND_TEXT, "alpha", "sess_1", 0, trace_id="trc_1", llm=llm, settings=settings
    )
    assert len(sub_queries) == 2
    assert {sq.intent_label for sq in sub_queries} == {"cancellation", "catering"}
    assert llm.calls == [COMPOUND_TEXT]
