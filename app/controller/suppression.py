"""Presentation-only query suppression (REQ-SUPPRESS-01, REQ-SUPPRESS-02; pseudocode 12.H).

Checked before the Controller's stability/entity RETRIEVE path (original TRD §8.6), so a
presentation-only follow-up never reaches retrieval at all.

Scope note: REQ-SUPPRESS-01's pseudocode calls `llm_confirm_suppression` when a match is
ambiguous. This phase excludes LLM calls from the deterministic controller path, so `is_ambiguous`
always returns False here — an unambiguous pattern match suppresses, anything else falls through
to normal WAIT/RETRIEVE/NO_RETRIEVAL logic (the safe default: never suppress on uncertain
grounds). LLM-based confirmation is deferred to whichever later phase first wires in an LLM.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from app.controller.entity_extraction import entity_diff, extract_entities

_SUPPRESSION_PATTERN = re.compile(
    r"\b(reformat|shorten|expand|bulleti[sz]e|bullet(?:s|ed|ing)?|translat\w*|repeat(?:s|ed|ing)?"
    r"|rephrase\w*|summari[sz]\w*)\b",
    re.IGNORECASE,
)


class SuppressionResult(BaseModel):
    retrieval_required: bool
    reason: str


def is_presentation_only(text: str) -> bool:
    return bool(_SUPPRESSION_PATTERN.search(text))


def _is_ambiguous(text: str) -> bool:  # noqa: ARG001 - placeholder pending LLM confirmation
    return False


def check_suppression(
    buffer_text: str,
    entities: dict[str, str],
    corpus_id: str,
    *,
    has_prior_answer: bool,
) -> SuppressionResult | None:
    """Returns None (not suppressed) if the transcript should proceed to normal controller logic,
    or a SuppressionResult routing to presentation-only reformatting of the existing answer.

    First-turn guard (REQ-SUPPRESS-02): with no prior answer, there is nothing to reformat, so a
    pattern match here is not a valid suppression case regardless of wording.
    """
    if not has_prior_answer:
        return None
    if not is_presentation_only(buffer_text):
        return None

    new_entities = extract_entities(buffer_text, corpus_id)
    if entity_diff(new_entities, entities):
        return None  # asks for a new fact too - do not suppress (original TRD 12.H)
    if _is_ambiguous(buffer_text):
        return None
    return SuppressionResult(retrieval_required=False, reason="presentation_restructure")
