"""Citation / Grounding Validator (pseudocode 12.F; REQ-GROUND-01/02/03).

Purely deterministic — no LLM call is required to validate a sentence, only to *regenerate* one
that failed validation (REQ-GROUND-01's "forces regeneration"). This is what makes fabrication
structurally impossible rather than merely discouraged: a citation tag naming a `[doc_id section]`
this turn's evidence set never produced is a lookup miss, not a judgment call.

`Chunk.section` (scripts/ingest_corpus.py's `split_sections`) is already a self-contained,
already-"§"-prefixed label such as `"§3"` — never a human-readable heading (that's
`Section.title`, baked into the chunk's own text instead, never carried as structured metadata).
So a citation tag is `[doc_id section]` with a single space, e.g. `"[orion_hall §3]"` — reusing
`section` verbatim, never prepending a second literal "§" (original TRD §8.4's own example,
`[Doc_12 §2]`, is exactly this: doc_id, a space, then the self-prefixed section token).

- `extract_citation_tags` parses the model's own inline `[doc_id section]` tags.
- `validate_sentence` (REQ-GROUND-01): a cited tag that doesn't resolve to a chunk in the
  evidence set is fabrication -> `Regenerate("fabricated_id")`; a factual-looking sentence with no
  citation at all -> `Regenerate("missing_citation")`; a citation whose chunk text doesn't
  actually support the sentence (lexical-overlap entailment, the spec's own cost-cutting fallback
  for the cross-encoder entailment check) -> `Regenerate("low_entailment")`.
- `finalize_sentence` caps regeneration at one retry (pseudocode 12.F: `attempt >= 1`) — a
  sentence that still fails becomes `Uncertain`, never emitted as an unsupported claim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.models.chunk import RetrievedChunk
from app.models.citation import Citation
from app.models.evidence import Evidence
from app.retrieval.sparse_bm25 import tokenize

_CITATION_TAG = re.compile(r"\[\s*(\S+)\s+(\S+)\s*\]")
_TAG_SEP = "::"  # joins doc_id/section into one lookup key; neither ever contains "::"


def format_citation_tag(doc_id: str, section: str) -> str:
    """The one place the `[doc_id section]` tag text is rendered — used by prompt_builder to
    write the evidence block and mirrored in reverse by extract_citation_tags, so the two can
    never drift out of sync the way a hand-duplicated format string once did."""
    return f"[{doc_id} {section}]"


@dataclass(frozen=True)
class EvidenceChunk:
    """The lookup unit validate_sentence needs per evidence row: enough of its Evidence row and
    RetrievedChunk to check citation membership and score entailment."""

    evidence_id: str
    chunk_id: str
    doc_id: str
    section: str
    text: str
    contradiction_pair_id: str | None


@dataclass(frozen=True)
class Accept:
    text: str
    citations: list[Citation]


@dataclass(frozen=True)
class Regenerate:
    reason: str  # "fabricated_id" | "missing_citation" | "low_entailment"


@dataclass(frozen=True)
class Uncertain:
    reason: str


def build_evidence_lookup(
    evidence: list[Evidence], chunks_by_id: dict[str, RetrievedChunk]
) -> dict[str, EvidenceChunk]:
    """Keyed by (doc_id, section) — the citation tag's own addressing scheme — not chunk_id, since
    the model only ever sees `[doc_id section]`, never a raw chunk_id (original TRD §8.4: "using
    only IDs present in the supplied evidence block", and the block is written in that format).
    If two evidence chunks share a (doc_id, section) pair, the higher-fusion_score one wins the
    slot deterministically."""
    by_tag: dict[tuple[str, str], EvidenceChunk] = {}
    for e in evidence:
        if e.duplicate_of is not None:
            continue
        chunk = chunks_by_id.get(e.chunk_id)
        if chunk is None:
            continue
        key = (chunk.doc_id, chunk.section)
        existing = by_tag.get(key)
        if existing is not None:
            existing_score = next(
                (ev.fusion_score for ev in evidence if ev.evidence_id == existing.evidence_id), 0.0
            )
            if e.fusion_score <= existing_score:
                continue
        by_tag[key] = EvidenceChunk(
            evidence_id=e.evidence_id,
            chunk_id=e.chunk_id,
            doc_id=chunk.doc_id,
            section=chunk.section,
            text=chunk.text,
            contradiction_pair_id=e.contradiction_pair_id,
        )
    return {f"{doc_id}{_TAG_SEP}{section}": ec for (doc_id, section), ec in by_tag.items()}


def extract_citation_tags(sentence: str) -> list[str]:
    """Returns each cited tag as "doc_id::section" (matching build_evidence_lookup's key format),
    in order of appearance."""
    return [
        f"{doc_id.strip()}{_TAG_SEP}{section.strip()}"
        for doc_id, section in _CITATION_TAG.findall(sentence)
    ]


def lexical_overlap(sentence: str, chunk_text: str) -> float:
    """Deterministic entailment fallback (original TRD §8.4 explicitly sanctions this: "cross-
    encoder entailment score... can fall back to lexical overlap to cut cost"). Token-overlap
    ratio against the sentence's own vocabulary, so a sentence that only restates chunk content
    scores high regardless of chunk length."""
    sentence_tokens = set(tokenize(sentence))
    chunk_tokens = set(tokenize(chunk_text))
    if not sentence_tokens or not chunk_tokens:
        return 0.0
    return len(sentence_tokens & chunk_tokens) / len(sentence_tokens)


def validate_sentence(
    sentence: str,
    evidence_by_tag: dict[str, EvidenceChunk],
    *,
    entailment_min: float,
) -> Accept | Regenerate:
    """Pseudocode 12.F `validate_sentence`, adapted to this project's types."""
    tags = extract_citation_tags(sentence)
    if any(tag not in evidence_by_tag for tag in tags):
        return Regenerate("fabricated_id")
    if not tags:
        return Regenerate("missing_citation")

    resolved = [evidence_by_tag[tag] for tag in tags]
    entailment = max(lexical_overlap(sentence, ec.text) for ec in resolved)
    if entailment < entailment_min:
        return Regenerate("low_entailment")

    citations = [
        Citation(
            answer_version_id="",  # filled in by the caller once the AnswerVersion id is known
            chunk_id=ec.chunk_id,
            doc_id=ec.doc_id,
            section=ec.section,
            claim_text=sentence,
            entailment_score=lexical_overlap(sentence, ec.text),
        )
        for ec in resolved
    ]
    return Accept(text=sentence, citations=citations)


def finalize_sentence(
    sentence: str,
    evidence_by_tag: dict[str, EvidenceChunk],
    *,
    entailment_min: float,
) -> Accept | Uncertain:
    """Pseudocode 12.F `finalize_sentence`, minus the LLM regeneration call itself (the caller —
    streaming_generator.py — owns that, since it alone holds the GenerationLLM); this function is
    the pure post-attempt outcome mapping. One retry only (REQ-GROUND-01's failure clause: "if
    still failing, sentence replaced with an uncertainty statement")."""
    result = validate_sentence(sentence, evidence_by_tag, entailment_min=entailment_min)
    if isinstance(result, Accept):
        return result
    return Uncertain(reason=result.reason)


def uncertainty_text(reason: str) -> str:
    """A plain, honest stand-in for a sentence that couldn't be grounded — never fabricated
    content, never silently dropped (REQ-GROUND-01's failure clause)."""
    return f"[Unable to verify this claim from the retrieved evidence: {reason}.]"


def stamp_citation_ids(citations: list[Citation], answer_version_id: str) -> list[Citation]:
    """Citation.answer_version_id is only knowable once the turn's AnswerVersion id is minted;
    validate_sentence runs per-sentence, mid-stream, before that id exists."""
    return [c.model_copy(update={"answer_version_id": answer_version_id}) for c in citations]


def contradiction_note(evidence_by_tag: dict[str, EvidenceChunk], pair_id: str) -> str:
    """REQ-GROUND-03: a deterministic, always-emitted discrepancy note naming both cited tags in a
    contradiction pair, appended to the prompt context so the model is instructed to surface both
    sides rather than silently pick one (see prompt_builder.build_prompt)."""
    tags = sorted(
        format_citation_tag(ec.doc_id, ec.section)
        for ec in evidence_by_tag.values()
        if ec.contradiction_pair_id == pair_id
    )
    return f"Note: sources {' and '.join(tags)} disagree — state both values explicitly."
