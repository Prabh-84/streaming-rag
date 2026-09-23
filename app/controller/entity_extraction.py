"""Deterministic, rule-based entity/slot extraction (REQ-NLP-01).

Uses spaCy (`en_core_web_sm`) for tokenization/POS only; slot-filling is a `PhraseMatcher` (part
of spaCy's Matcher family) compiled from the active corpus's `SlotSchema` (REQ-CORPUS-01) —
never an opaque statistical NER model, so extraction stays auditable and corpus-agnostic.

Only the entity-extraction half of REQ-NLP-01 lives here. Compound-signal detection (`cc`/`conj`
dependency-tree walking) is the Multi-Intent Decomposer's concern (REQ-INTENT-01, Phase 4) and is
out of scope for the Retrieval Controller built in this phase.

Two slot-value strategies, chosen by `Slot.value_type`:
- `categorical` / `text` / `date`: literal phrase match (case-insensitive); the value is the
  slot's own canonical pattern string, e.g. patterns=["Orion Hall"] always yields "Orion Hall"
  regardless of the input's casing.
- `numeric`: patterns are anchor/unit words (e.g. "guests", "people"); the value is the nearest
  numeric token adjacent to an anchor match (checked before, then after, within 2 tokens) — this
  matches natural phrasing like "30 people" or "capacity of 30".
"""

from __future__ import annotations

from functools import lru_cache

from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc

from app.core.config import get_settings
from app.core.slots import Slot, SlotSchema, get_slot_schema

_NUMERIC_LOOKAROUND = 2  # tokens to scan on each side of an anchor match for a number


@lru_cache(maxsize=1)
def get_nlp() -> Language:
    """Lazy singleton. Constructing this module never loads the model; only calling this does —
    call it (or warmup()) at startup so the first real transcript chunk isn't the one that pays
    the load cost (mirrors app.core.embeddings' embedder pattern)."""
    import spacy

    return spacy.load(get_settings().spacy_model)


def warmup() -> None:
    get_nlp()


class CorpusMatcher:
    """A compiled PhraseMatcher for one corpus's SlotSchema, plus the metadata to interpret hits.

    Each pattern gets its own match_id (not one id per slot with many patterns) specifically so a
    hit can be traced back to which literal pattern string matched — that string is the canonical
    value returned for categorical/text/date slots, regardless of the input's casing.
    """

    def __init__(self, nlp: Language, schema: SlotSchema) -> None:
        self.schema = schema
        self._matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        self._meta: dict[int, tuple[Slot, str]] = {}
        for slot in schema.slots:
            for i, pattern in enumerate(slot.patterns):
                key = f"slot::{slot.name}::{i}"
                self._meta[nlp.vocab.strings[key]] = (slot, pattern)
                self._matcher.add(key, [nlp.make_doc(pattern)])

    def extract(self, doc: Doc) -> dict[str, str]:
        entities: dict[str, str] = {}
        for match_id, start, end in self._matcher(doc):
            slot, pattern = self._meta[match_id]
            if slot.value_type == "numeric":
                value = self._numeric_value(doc, start, end)
                if value is not None:
                    entities[slot.name] = value  # last occurrence in the buffer wins
            else:
                entities[slot.name] = pattern  # canonical: the pattern, not the matched casing
        return entities

    @staticmethod
    def _numeric_value(doc: Doc, start: int, end: int) -> str | None:
        before = doc[max(0, start - _NUMERIC_LOOKAROUND) : start]
        after = doc[end : min(len(doc), end + _NUMERIC_LOOKAROUND)]
        for token in list(reversed(before)) + list(after):
            if token.like_num:
                return token.text.replace(",", "")
        return None


@lru_cache(maxsize=32)
def get_corpus_matcher(corpus_id: str) -> CorpusMatcher:
    """Cached per corpus_id. Call get_corpus_matcher.cache_clear() in tests that mutate a
    corpus's slots.yaml and need the matcher rebuilt."""
    return CorpusMatcher(get_nlp(), get_slot_schema(corpus_id))


def extract_entities(text: str, corpus_id: str) -> dict[str, str]:
    """Extract slot->value pairs for one corpus from arbitrary transcript text (typically the
    full session buffer, re-extracted each chunk — cheap for short transcripts, and means a slot
    mentioned earlier stays known as the buffer grows, per REQ-NLP-01)."""
    if not text.strip():
        return {}
    doc = get_nlp()(text)
    return get_corpus_matcher(corpus_id).extract(doc)


def entity_diff(new: dict[str, str], old: dict[str, str]) -> dict[str, str]:
    """Slots in `new` that are absent from `old` or whose value changed. Truthy iff there is an
    actionable delta (REQ-CTRL-01/02's "actionable entity delta")."""
    return {k: v for k, v in new.items() if old.get(k) != v}
