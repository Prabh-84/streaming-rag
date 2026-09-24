"""Tests for REQ-CORPUS-01 (per-corpus slots.yaml schema)."""

import re
from pathlib import Path

import pytest

from app.core.slots import (
    InvalidCorpusIdError,
    SlotSchemaError,
    load_slot_schema,
    validate_corpus_id,
)
from tests.fakes import FIXTURE_CORPORA


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


# --- Phase 2: full slots.yaml validation (REQ-CORPUS-01) -------------------------------------

VALID_SLOT = '  - name: venue\n    patterns: ["Hall"]\n'


def _write(tmp_path: Path, content: str | bytes, corpus_id: str = "c1") -> Path:
    corpus_dir = tmp_path / corpus_id
    corpus_dir.mkdir(parents=True, exist_ok=True)
    path = corpus_dir / "slots.yaml"
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return tmp_path


def test_slots_schema_loads():
    schema = load_slot_schema("alpha", FIXTURE_CORPORA)
    assert schema.corpus_id == "alpha"
    # capacity/cancellation_policy/catering (Phase 4) support compound-signal test fixtures.
    assert schema.slot_names() == ["venue", "capacity", "cancellation_policy", "catering"]
    assert schema.slots[1].value_type == "numeric"


@pytest.mark.parametrize(
    ("content", "fragment"),
    [
        ("slots: [\n  - name: venue", "not valid YAML"),
        ("", "must be a mapping"),
        ("- just\n- a list\n", "must be a mapping"),
        ("description: no slots here\n", "must be a mapping"),
        ("slots: []\n", "failed validation"),
        ("slots:\n" + VALID_SLOT + "version: 2\n", "unknown top-level keys"),
        ("corpus_id: other\nslots:\n" + VALID_SLOT, "declares corpus_id='other'"),
        ("slots:\n" + VALID_SLOT + VALID_SLOT, "duplicate slot names"),
        ("slots:\n  - name: venue\n", "failed validation"),  # patterns missing
        ('slots:\n  - patterns: ["x"]\n', "failed validation"),  # name missing
        ('slots:\n  - name: venue\n    patterns: ["  "]\n', "failed validation"),
        ('slots:\n  - name: venue\n    patterns: ["a", "a"]\n', "failed validation"),
        ('slots:\n  - name: Venue Name\n    patterns: ["x"]\n', "failed validation"),
        ('slots:\n  - name: venue\n    patterns: ["x"]\n    value_type: money\n', "failed"),
        ('slots:\n  - name: venue\n    patterns: ["x"]\n    regex: true\n', "failed validation"),
    ],
)
def test_malformed_slots_yaml_is_rejected(tmp_path, content, fragment):
    root = _write(tmp_path, content)
    with pytest.raises(SlotSchemaError, match=re.escape(fragment)):
        load_slot_schema("c1", root)


def test_non_utf8_slots_yaml_is_rejected(tmp_path):
    root = _write(tmp_path, b"slots:\n  - name: caf\xe9\n")
    with pytest.raises(SlotSchemaError, match="not valid UTF-8"):
        load_slot_schema("c1", root)


@pytest.mark.parametrize("bad_id", ["../etc", "a/b", "", ".hidden", "x" * 65, "a b", None])
def test_invalid_corpus_id_is_rejected(bad_id):
    with pytest.raises(InvalidCorpusIdError):
        validate_corpus_id(bad_id)
    with pytest.raises(SlotSchemaError):
        load_slot_schema(bad_id, FIXTURE_CORPORA)


def test_valid_corpus_ids_are_accepted():
    for good in ["default", "alpha", "hr-policy_v2", "A1"]:
        assert validate_corpus_id(good) == good
