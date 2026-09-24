"""Session Delta Refinement classification (pseudocode 12.G; REQ-SESS-01)."""

from __future__ import annotations

from app.session.delta_engine import (
    build_delta_query,
    classify_segment,
    has_contrastive_marker,
)
from tests.fakes import FakeEmbedder, make_settings

FIXTURE_CORPORA = None  # not needed - these are pure functions, no corpus I/O


def _settings(**overrides):
    from pathlib import Path

    return make_settings(Path("."), Path("."), REFINEMENT_THRESHOLD=0.75, **overrides)


# --- has_contrastive_marker ---------------------------------------------------------------------


def test_contrastive_markers_detected():
    for text in [
        "actually, can you check the capacity instead",
        "no wait, I meant Lumen Pavilion",
        "scratch that, let's go with 50 guests",
        "on second thought, what about catering",
    ]:
        assert has_contrastive_marker(text), text


def test_no_contrastive_marker_in_plain_text():
    assert not has_contrastive_marker("what is the cancellation policy for Orion Hall")


# --- classify_segment ---------------------------------------------------------------------------


def test_high_similarity_is_refinement():
    embedder = FakeEmbedder()
    settings = _settings()
    topic = embedder.embed("Orion Hall cancellation policy for 30 guests")
    result = classify_segment(
        "Orion Hall cancellation policy for 30 guests", topic, embedder, settings
    )
    assert result == "refinement"  # identical text -> cosine 1.0


def test_low_similarity_is_new_topic():
    embedder = FakeEmbedder()
    settings = _settings()
    topic = embedder.embed("Orion Hall cancellation policy for thirty guests at a formal dinner")
    result = classify_segment(
        "completely unrelated words about something else entirely", topic, embedder, settings
    )
    assert result == "new_topic"


def test_contrastive_marker_forces_refinement_even_at_low_similarity():
    """A contrastive marker overrides similarity (pseudocode 12.G: `sim >= THRESHOLD or
    contrastive`) - the whole point is catching corrections that don't restate prior wording."""
    embedder = FakeEmbedder()
    settings = _settings()
    topic = embedder.embed("Orion Hall cancellation policy for thirty guests")
    result = classify_segment("actually make that fifty guests instead", topic, embedder, settings)
    assert result == "refinement"


def test_mid_range_similarity_is_ambiguous():
    """REQ-SESS-01 failure clause: the 0.4..REFINEMENT_THRESHOLD gap must never guess."""
    embedder = FakeEmbedder()
    settings = _settings()
    topic = embedder.embed("Orion Hall cancellation policy for thirty guests at a formal dinner")
    # Partial word overlap with the topic, but well short of it and not a contrastive marker.
    text = "guests dinner formal arrangements"
    result = classify_segment(text, topic, embedder, settings)
    sim_direct = _cosine(embedder.embed(text), topic)
    assert 0.4 <= sim_direct < settings.refinement_threshold, sim_direct  # sanity-check the fixture
    assert result == "ambiguous"


def _cosine(a, b):
    import math

    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


# --- build_delta_query ----------------------------------------------------------------------------


def test_build_delta_query_is_deterministic_and_sorted():
    delta = {"capacity": "50", "venue": "Lumen Pavilion"}
    assert build_delta_query(delta) == "capacity: 50; venue: Lumen Pavilion"
    # Same dict, different insertion order -> identical output (determinism).
    delta_reordered = {"venue": "Lumen Pavilion", "capacity": "50"}
    assert build_delta_query(delta) == build_delta_query(delta_reordered)


def test_build_delta_query_empty_dict():
    assert build_delta_query({}) == ""
