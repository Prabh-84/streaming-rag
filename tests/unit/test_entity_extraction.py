"""Deterministic entity extraction (REQ-NLP-01) against the synthetic fixture corpora."""

from __future__ import annotations

import pytest

from app.controller.entity_extraction import entity_diff, extract_entities, get_corpus_matcher
from app.core.config import get_settings
from app.core.slots import get_slot_schema
from tests.fakes import FIXTURE_CORPORA


@pytest.fixture(autouse=True)
def _use_fixture_corpus_root(monkeypatch):
    """get_slot_schema()/get_corpus_matcher() read the global Settings singleton (they're
    process-wide caches, not per-call-configurable), so point it at the synthetic test corpora."""
    monkeypatch.setattr(get_settings(), "corpus_root", str(FIXTURE_CORPORA))
    get_slot_schema.cache_clear()
    get_corpus_matcher.cache_clear()
    yield
    get_slot_schema.cache_clear()
    get_corpus_matcher.cache_clear()


def test_entity_extraction_deterministic():
    """REQ-NLP-01 test name. Same input, same corpus -> same output, every time."""
    text = "I need Orion Hall for 30 guests please"
    first = extract_entities(text, "alpha")
    second = extract_entities(text, "alpha")
    assert first == second == {"venue": "Orion Hall", "capacity": "30"}


def test_categorical_slot_matches_case_insensitively_and_returns_canonical_value():
    assert extract_entities("book me orion hall", "alpha") == {"venue": "Orion Hall"}
    assert extract_entities("LUMEN PAVILION please", "alpha") == {"venue": "Lumen Pavilion"}


def test_numeric_slot_finds_the_adjacent_number_before_the_anchor_word():
    assert extract_entities("a venue for 45 people", "alpha") == {"capacity": "45"}
    assert extract_entities("we need 120 guests total", "alpha") == {"capacity": "120"}


def test_numeric_anchor_without_a_nearby_number_is_not_extracted():
    assert extract_entities("how many guests can you fit", "alpha") == {}


def test_no_match_returns_empty_dict():
    assert extract_entities("what is the weather like today", "alpha") == {}


def test_empty_and_whitespace_text_returns_empty_dict():
    assert extract_entities("", "alpha") == {}
    assert extract_entities("   ", "alpha") == {}


def test_extraction_is_scoped_to_the_requested_corpus():
    """Corpus isolation at the NLP layer: beta's slots (trip_type/currency) never fire on alpha's
    vocabulary, and vice versa — each corpus's Matcher only knows its own slots.yaml."""
    assert extract_entities("Orion Hall for 30 guests", "beta") == {}
    assert extract_entities("an international trip in USD", "alpha") == {}
    assert extract_entities("an international trip in USD", "beta") == {
        "trip_type": "international",
        "currency": "USD",
    }


def test_later_mention_of_the_same_slot_wins():
    text = "for 30 guests, actually make it 45 guests"
    assert extract_entities(text, "alpha") == {"capacity": "45"}


def test_entity_diff_reports_only_new_or_changed_slots():
    assert entity_diff({"venue": "Orion Hall"}, {}) == {"venue": "Orion Hall"}
    assert entity_diff({"venue": "Orion Hall"}, {"venue": "Orion Hall"}) == {}
    assert entity_diff({"venue": "Orion Hall", "capacity": "30"}, {"venue": "Orion Hall"}) == {
        "capacity": "30"
    }
    assert entity_diff({}, {"venue": "Orion Hall"}) == {}  # a slot disappearing is not a delta
