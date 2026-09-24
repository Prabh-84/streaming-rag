"""Session Delta Refinement — classification half (pseudocode 12.G; REQ-SESS-01).

Once a session has an established topic (`session.has_retrieved_for_topic` is True and
`session.topic_embedding` is set), a later chunk that carries a genuine entity delta is
classified as one of:

- `refinement` — the same topic, a new constraint (topic-similarity >= REFINEMENT_THRESHOLD or a
  contrastive marker like "actually"/"instead"). Only the delta is retrieved for (REQ-SESS-01's
  "only the delta entity/constraint builds new sub-queries") — see
  `retrieval_controller._refine`, which reuses the *existing* decomposition/retrieval/fusion/
  reranking machinery (Phase 2/4/5) for the delta text, never a second implementation of it. This
  module only decides whether to and what to retrieve for, never how.
- `new_topic` — similarity < 0.4 (a spec-mandated literal, pseudocode 12.G; no REQ-ID promotes it
  to a named setting the way REFINEMENT_THRESHOLD=0.75 is). Falls through to the existing
  stability-gated WAIT/RETRIEVE pipeline unchanged, exactly as if this were a fresh session.
- `ambiguous` — the 0.4..REFINEMENT_THRESHOLD gap. REQ-SESS-01's failure clause: never guess: the
  controller returns NO_RETRIEVAL{reason="ambiguous_refinement"} rather than picking a side.

Claim-lifecycle bookkeeping (REQ-SESS-02: contradiction-checking prior claims against the delta,
marking them superseded, carrying non-contradicted ones forward) operates on claims *extracted
from a synthesized answer* (Grounding Validator, REQ-GROUND-01) — Synthesis and Grounding are
both later phases. This phase's "prior evidence" is `session.last_evidence` (Phase 5's per-turn
output set), reused as-is when a turn has no new information; there is no claim ledger yet.
"""

from __future__ import annotations

import math
import re
from typing import Literal

from app.core.config import Settings
from app.core.embeddings import Embedder

Classification = Literal["new_topic", "refinement", "ambiguous"]

_NEW_TOPIC_THRESHOLD = 0.4  # pseudocode 12.G's own literal, not a named Settings field

_CONTRASTIVE_MARKERS = re.compile(
    r"\b(actually|instead|correction|scratch that|change that|rather than|"
    r"on second thought|i meant|no wait|sorry,? i mean)\b",
    re.IGNORECASE,
)


def has_contrastive_marker(text: str) -> bool:
    """Deterministic, rule-based (REQ-NLP-01's spirit: no opaque model for a signal this cheap)."""
    return bool(_CONTRASTIVE_MARKERS.search(text))


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return 0.0 if norm_a == 0 or norm_b == 0 else dot / (norm_a * norm_b)


def classify_segment(
    text: str,
    topic_embedding: tuple[float, ...],
    embedder: Embedder,
    settings: Settings,
) -> Classification:
    """Pseudocode 12.G's classification half. Only called once a topic is established — the
    caller (retrieval_controller.on_chunk) is responsible for the "no prior topic yet" first-turn
    guard, exactly as check_suppression's REQ-SUPPRESS-02 guard works."""
    sim = _cosine(embedder.embed(text), topic_embedding)
    if sim >= settings.refinement_threshold or has_contrastive_marker(text):
        return "refinement"
    if sim < _NEW_TOPIC_THRESHOLD:
        return "new_topic"
    return "ambiguous"


def build_delta_query(delta_entities: dict[str, str]) -> str:
    """Deterministic, LLM-free sub-query text for the delta (pseudocode 12.G's
    build_delta_query) — the new/changed slot values only, in slot-name order so the same delta
    always produces byte-identical text (REQ-EVAL's determinism concerns)."""
    return "; ".join(f"{slot}: {value}" for slot, value in sorted(delta_entities.items()))
