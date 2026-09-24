"""Prompt construction for the Streaming Answer Generator (pseudocode 12.I; original TRD §8.4).

Every chunk in the evidence block is labeled with the exact `[doc_id section]` tag the model must
reuse verbatim to cite it — "using only IDs present in the supplied evidence block (never
free-form)" (original TRD §8.4). The model physically cannot fabricate a citation to content it
was never shown; `citation_validator.validate_sentence` then checks it never invents a tag either.
`section` is already a self-contained label (e.g. `"§3"`, see citation_validator's module
docstring) — the tag is `doc_id` + a space + that label, never a second literal "§".
"""

from __future__ import annotations

from app.grounding.citation_validator import EvidenceChunk, contradiction_note, format_citation_tag
from app.models.answer_version import AnswerVersion
from app.models.sub_query import SubQuery

_SYSTEM_PROMPT = (
    "You answer questions using only the evidence block provided below. Every factual sentence "
    "must end with a citation tag in the exact form [doc_id section], copied verbatim from the "
    "evidence block — never invent a tag, and never cite a tag that isn't listed. If the evidence "
    "does not answer a question, say so plainly rather than guessing. When two sources disagree, "
    "state both values explicitly with their own citations rather than picking one."
)


def build_prompt(
    sub_queries: list[SubQuery],
    evidence_by_tag: dict[str, EvidenceChunk],
    *,
    prior_version: AnswerVersion | None = None,
) -> str:
    """Deterministic prompt text: the questions being answered, the evidence block (one entry per
    citable tag), and — when REQ-EVID-04 flagged a contradiction — an explicit instruction to
    surface both sides (REQ-GROUND-03)."""
    lines = ["Questions to answer:"]
    for sq in sub_queries:
        lines.append(f"- {sq.text}")

    lines.append("")
    lines.append("Evidence block (cite using the exact tag shown):")
    for tag in sorted(evidence_by_tag):
        ec = evidence_by_tag[tag]
        lines.append(f"{format_citation_tag(ec.doc_id, ec.section)}: {ec.text}")

    pair_ids = {
        ec.contradiction_pair_id for ec in evidence_by_tag.values() if ec.contradiction_pair_id
    }
    if pair_ids:
        lines.append("")
        for pair_id in sorted(pair_ids):
            lines.append(contradiction_note(evidence_by_tag, pair_id))

    if prior_version is not None:
        lines.append("")
        lines.append(f"Prior answer (version {prior_version.version_no}): {prior_version.text}")
        lines.append("This turn refines that answer with new information — update it accordingly.")

    return "\n".join(lines)
