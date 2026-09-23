"""Presentation-only suppression (REQ-SUPPRESS-01/02) against the alpha fixture corpus."""

from __future__ import annotations

import pytest

from app.controller.entity_extraction import get_corpus_matcher
from app.controller.suppression import check_suppression, is_presentation_only
from app.core.config import get_settings
from app.core.slots import get_slot_schema
from tests.fakes import FIXTURE_CORPORA


@pytest.fixture(autouse=True)
def _use_fixture_corpus_root(monkeypatch):
    monkeypatch.setattr(get_settings(), "corpus_root", str(FIXTURE_CORPORA))
    get_slot_schema.cache_clear()
    get_corpus_matcher.cache_clear()
    yield
    get_slot_schema.cache_clear()
    get_corpus_matcher.cache_clear()


@pytest.mark.parametrize(
    "text",
    [
        "can you repeat that",
        "please reformat this as bullets",
        "shorten your answer",
        "can you bulletize this",
        "summarize what you said",
        "rephrase that for me",
        "please translate this to Spanish",
        "REPEAT THAT",
    ],
)
def test_is_presentation_only_matches_expected_phrasings(text):
    assert is_presentation_only(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "what is the cancellation policy for Orion Hall",
        "I need a venue for 30 guests",
        "how many people can Lumen Pavilion hold",
    ],
)
def test_is_presentation_only_does_not_match_factual_requests(text):
    assert is_presentation_only(text) is False


def test_check_suppression_fires_when_presentation_only_and_no_new_entity():
    result = check_suppression(
        "please repeat that in two bullets",
        entities={"venue": "Orion Hall"},
        corpus_id="alpha",
        has_prior_answer=True,
    )
    assert result is not None
    assert result.retrieval_required is False
    assert result.reason == "presentation_restructure"


def test_check_suppression_does_not_fire_without_a_pattern_match():
    result = check_suppression(
        "what is the cancellation policy",
        entities={},
        corpus_id="alpha",
        has_prior_answer=True,
    )
    assert result is None


def test_check_suppression_does_not_fire_when_a_new_fact_is_also_requested():
    """12.H: "asks for a new fact too - do not suppress" — a new entity delta wins over the
    pattern match, e.g. "summarize AND tell me about Lumen Pavilion"."""
    result = check_suppression(
        "summarize that and tell me about Lumen Pavilion",
        entities={"venue": "Orion Hall"},
        corpus_id="alpha",
        has_prior_answer=True,
    )
    assert result is None


def test_first_turn_suppression_guard():
    """REQ-SUPPRESS-02: with no prior answer, a presentation-only pattern must never suppress —
    there is nothing yet to reformat."""
    result = check_suppression(
        "please repeat that in two bullets",
        entities={},
        corpus_id="alpha",
        has_prior_answer=False,
    )
    assert result is None
