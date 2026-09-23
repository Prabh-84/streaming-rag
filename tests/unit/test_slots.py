"""Tests for REQ-CORPUS-01 (per-corpus slots.yaml schema)."""

import pytest
from app.core.slots import SlotSchemaError, load_slot_schema


def test_load_slot_schema_valid(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpus" / "test_corpus"
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "slots.yaml").write_text(
        """
slots:
  - name: location
    patterns: ["in {LOCATION}", "at {LOCATION}"]
    value_type: text
  - name: capacity
    patterns: ["for {NUM} people", "{NUM} attendees"]
    value_type: numeric
""",
        encoding="utf-8",
    )
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "corpus_root", str(tmp_path / "corpus"))

    schema = load_slot_schema("test_corpus")
    assert schema.corpus_id == "test_corpus"
    assert schema.slot_names() == ["location", "capacity"]


def test_missing_slots_yaml_raises(tmp_path, monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "corpus_root", str(tmp_path / "corpus"))
    with pytest.raises(SlotSchemaError):
        load_slot_schema("nonexistent_corpus")


def test_slot_without_patterns_is_rejected(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpus" / "bad_corpus"
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "slots.yaml").write_text(
        "slots:\n  - name: location\n    patterns: []\n", encoding="utf-8"
    )

    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "corpus_root", str(tmp_path / "corpus"))
    with pytest.raises(SlotSchemaError):
        load_slot_schema("bad_corpus")
