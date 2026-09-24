"""Multi-Intent Decomposer (REQ-INTENT-01/02; pseudocode 12.B).

Decomposition only runs after the Controller has already decided RETRIEVE — it is a downstream
refinement, not a second retrieval trigger (original TRD §6.2). It first checks a cheap,
deterministic syntactic signal (a coordinating conjunction joining two noun phrases bound to
different domain slots — REQ-NLP-01's compound-signal half, deferred from Phase 3). Only when that
signal is present does it call the configured LLM for structured decomposition; a simple request
never pays for an LLM call.

Every proposed sub-query is embedded and pairwise-compared against the ones already kept — any
pair whose cosine exceeds MERGE_THRESHOLD is merged into one, preventing pitfall 5
(over-fragmentation). If the LLM call fails, times out, or returns nothing usable, the whole
transcript is used as a single fallback sub-query rather than dropping the turn.

The LLM behind structured decomposition is swappable via LLM_PROVIDER (anthropic | gemini,
default anthropic) — both implementations satisfy the same DecompositionLLM protocol and must
produce the identical {"sub_queries": [{"text", "intent_label"}, ...]} shape before it ever
reaches _validate_proposed(); everything downstream of that call (validation, merge, fallback,
retrieval) is provider-agnostic.
"""

from __future__ import annotations

import asyncio
import math
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any, Protocol, runtime_checkable

import structlog
from pydantic import BaseModel, ValidationError, field_validator

from app.controller.entity_extraction import get_corpus_matcher, get_nlp
from app.core.config import Settings, get_settings
from app.core.embeddings import Embedder, get_embedder
from app.core.events import EventSink, EventType, TelemetryEvent
from app.models.sub_query import SubQuery

log = structlog.get_logger()

_SINGLE_INTENT_LABEL = "single"

DECOMPOSE_TOOL_NAME = "submit_sub_queries"

_DECOMPOSE_SYSTEM_PROMPT = (
    "You split a single user request into independent, self-contained sub-questions. Each "
    "sub-query must be answerable on its own, using only the wording already present in the "
    "request. Do not invent facts, do not answer the request, and do not add information that "
    "was not said. If the request only has one real question, return exactly one sub-query "
    "containing the whole request."
)

_DECOMPOSE_TOOL_SCHEMA: dict[str, Any] = {
    "name": DECOMPOSE_TOOL_NAME,
    "description": "Submit the decomposed sub-queries for a user request.",
    "input_schema": {
        "type": "object",
        "properties": {
            "sub_queries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "One independent, self-contained sub-question.",
                        },
                        "intent_label": {
                            "type": "string",
                            "description": "A short (1-3 word) label for this sub-question's "
                            "topic.",
                        },
                    },
                    "required": ["text", "intent_label"],
                },
            }
        },
        "required": ["sub_queries"],
    },
}


# --------------------------------------------------------------------------------------------
# Compound-signal detection (REQ-NLP-01, REQ-INTENT-01/02) - deterministic, no LLM.
# --------------------------------------------------------------------------------------------


def _subtree_bounds(token: Any) -> tuple[int, int]:
    subtree = list(token.subtree)
    return min(t.i for t in subtree), max(t.i for t in subtree) + 1  # token index [start, end)


def has_compound_signal(text: str, corpus_id: str) -> bool:
    """True iff a coordinating conjunction (`cc`) joins two conjuncts (`conj`) that are each
    bound to a *different* domain slot from the active corpus's SlotSchema — the frozen spec's
    exact signal (original TRD §6.2: "coordinating conjunctions joining two distinct noun phrases
    bound to different domain slots"). Requires >=2 slot mentions total; a single mention can
    never span two different slots. A single request with no such structure is never decomposed,
    so it never pays for an LLM call (REQ-INTENT-02)."""
    if not text.strip():
        return False

    doc = get_nlp()(text)
    hits = get_corpus_matcher(corpus_id).raw_matches(doc)
    if len(hits) < 2:
        return False

    for head in doc:
        if not any(child.dep_ == "cc" for child in head.children):
            continue
        conjuncts = [head, *(c for c in head.children if c.dep_ == "conj")]
        if len(conjuncts) < 2:
            continue

        slot_sets = []
        for member in conjuncts:
            start, end = _subtree_bounds(member)
            slot_sets.append({slot for slot, s, e in hits if s >= start and e <= end})

        distinct_slots = {s for slots in slot_sets for s in slots}
        members_with_a_slot = sum(1 for slots in slot_sets if slots)
        if len(distinct_slots) >= 2 and members_with_a_slot >= 2:
            return True
    return False


# --------------------------------------------------------------------------------------------
# LLM structured decomposition (REQ-INTENT-01) - only reached when has_compound_signal is True.
# --------------------------------------------------------------------------------------------


class ProposedSubQuery(BaseModel):
    text: str
    intent_label: str

    @field_validator("text", "intent_label")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


class ProposedSubQueryList(BaseModel):
    sub_queries: list[ProposedSubQuery]


@runtime_checkable
class DecompositionLLM(Protocol):
    async def decompose(self, transcript: str, *, timeout_ms: int) -> Any: ...


class AnthropicDecomposer:
    """Wraps the configured Claude model behind DecompositionLLM. Lazily constructs the SDK
    client on first use (mirrors app.core.embeddings/get_nlp's lazy-singleton pattern), so
    importing this module or starting the app never requires a real API key — only a genuine
    compound request does."""

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        return self._client

    async def decompose(self, transcript: str, *, timeout_ms: int) -> Any:
        # Transport-level timeout as a second line of defense; decompose() (the orchestrator
        # below) is the authoritative enforcement point and does not trust any DecompositionLLM
        # implementation, including this one, to honor timeout_ms correctly on its own.
        client = self._get_client()
        response = await client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=_DECOMPOSE_SYSTEM_PROMPT,
            tools=[_DECOMPOSE_TOOL_SCHEMA],
            tool_choice={"type": "tool", "name": DECOMPOSE_TOOL_NAME},
            messages=[{"role": "user", "content": transcript}],
            timeout=timeout_ms / 1000,
        )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == DECOMPOSE_TOOL_NAME:
                return block.input
        return None


class GeminiDecomposer:
    """Wraps the configured Gemini model behind DecompositionLLM, using Gemini's native
    response_schema JSON mode (not tool-use, since Gemini's structured-output guarantee is
    stronger there) to produce the same {"sub_queries": [...]} shape the Anthropic path emits.
    Lazily constructs the SDK client on first use, same lazy-singleton pattern as
    AnthropicDecomposer above."""

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    async def decompose(self, transcript: str, *, timeout_ms: int) -> Any:
        import json

        from google.genai import types

        client = self._get_client()
        response = await client.aio.models.generate_content(
            model=self._model,
            contents=transcript,
            config=types.GenerateContentConfig(
                system_instruction=_DECOMPOSE_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=_DECOMPOSE_TOOL_SCHEMA["input_schema"],
                # Transport-level timeout as a second line of defense, same reasoning as
                # AnthropicDecomposer.decompose(); decompose() below is the authoritative
                # enforcement point. Gemini's API hard-rejects any request deadline under 10s
                # (400 INVALID_ARGUMENT) before attempting generation at all, so this floor is
                # required for the SDK call to even be well-formed — it does not change the
                # project's own timeout budget, which asyncio.wait_for() in decompose() still
                # enforces and can still cancel a slow call well before Gemini's floor elapses.
                http_options=types.HttpOptions(timeout=max(timeout_ms, 10_000)),
            ),
        )
        if not response.text:
            return None
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            return None


_PROVIDERS = {"anthropic", "gemini"}


@lru_cache(maxsize=1)
def get_decomposer() -> DecompositionLLM:
    settings = get_settings()
    if settings.llm_provider == "gemini":
        return GeminiDecomposer(settings.gemini_api_key, settings.gemini_model)
    if settings.llm_provider == "anthropic":
        return AnthropicDecomposer(settings.anthropic_api_key, settings.llm_model)
    raise ValueError(
        f"Unknown LLM_PROVIDER {settings.llm_provider!r}; expected one of {sorted(_PROVIDERS)}"
    )


def _validate_proposed(raw: Any) -> list[ProposedSubQuery]:
    """Strict rejection of malformed/unusable LLM output (task requirement): anything that isn't
    a well-formed {"sub_queries": [{"text", "intent_label"}, ...]} with at least one non-blank
    entry is treated as zero usable sub-queries, triggering the fallback path — never allowed to
    flow free-form text into the retrieval pipeline."""
    if not isinstance(raw, dict):
        return []
    try:
        parsed = ProposedSubQueryList.model_validate(raw)
    except ValidationError:
        return []
    return parsed.sub_queries


# --------------------------------------------------------------------------------------------
# Anti-over-fragmentation merge (REQ-INTENT-02) - pure function, no I/O.
# --------------------------------------------------------------------------------------------


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return 0.0 if norm_a == 0 or norm_b == 0 else dot / (norm_a * norm_b)


def merge_similar(
    proposed: list[ProposedSubQuery], threshold: float, embedder: Embedder
) -> list[tuple[ProposedSubQuery, list[str]]]:
    """Pairwise-compare each proposed sub-query's embedding against the ones already kept; a
    cosine above `threshold` merges it into the existing one instead of keeping a near-duplicate
    (pseudocode 12.B). Returns (kept, [texts absorbed into it]) pairs, in first-seen order."""
    kept: list[tuple[ProposedSubQuery, list[str]]] = []
    embeddings: list[tuple[float, ...]] = []
    for sq in proposed:
        sq_embedding = embedder.embed(sq.text)
        dup_index = next(
            (
                i
                for i, kept_embedding in enumerate(embeddings)
                if _cosine(kept_embedding, sq_embedding) > threshold
            ),
            None,
        )
        if dup_index is None:
            kept.append((sq, []))
            embeddings.append(sq_embedding)
        else:
            kept[dup_index][1].append(sq.text)
            log.info(
                "over_fragmentation_prevented", kept=kept[dup_index][0].text, merged_away=sq.text
            )
    return kept


# --------------------------------------------------------------------------------------------
# Orchestration (pseudocode 12.B) - what the Retrieval Controller calls on RETRIEVE.
# --------------------------------------------------------------------------------------------


def _fallback(session_id: str, transcript: str, offset_ms: int) -> list[SubQuery]:
    return [
        SubQuery(
            session_id=session_id,
            text=transcript,
            intent_label=_SINGLE_INTENT_LABEL,
            parent_transcript_offset=offset_ms,
            created_at=datetime.now(UTC),
        )
    ]


def _emit_subquery_created(
    sub_query: SubQuery, session_id: str, trace_id: str, sink: EventSink | None
) -> None:
    if sink is None:
        return
    sink(
        TelemetryEvent(
            session_id=session_id,
            trace_id=trace_id,
            event_type=EventType.SUBQUERY_CREATED,
            payload={
                "sub_query_id": sub_query.sub_query_id,
                "text": sub_query.text,
                "intent_label": sub_query.intent_label,
            },
        )
    )


def _emit_fallback(session_id: str, trace_id: str, sink: EventSink | None, reason: str) -> None:
    log.info("decomposition_fallback", session_id=session_id, reason=reason)
    if sink is None:
        return
    sink(
        TelemetryEvent(
            session_id=session_id,
            trace_id=trace_id,
            event_type=EventType.ERROR,
            payload={
                "stage": "decomposition",
                "error_type": "decomposition_fallback",
                "message": reason,
                "recoverable": True,
            },
        )
    )


async def decompose(
    transcript: str,
    corpus_id: str,
    session_id: str,
    offset_ms: int,
    *,
    trace_id: str,
    llm: DecompositionLLM | None = None,
    embedder: Embedder | None = None,
    settings: Settings | None = None,
    sink: EventSink | None = None,
) -> list[SubQuery]:
    """Pseudocode 12.B, adapted to this project's types. Always returns >=1 SubQuery; the caller
    (Retrieval Controller) never needs its own fallback path."""
    settings = settings or get_settings()

    if not has_compound_signal(transcript, corpus_id):
        sub_queries = _fallback(session_id, transcript, offset_ms)
    else:
        llm = llm or get_decomposer()
        fallback_reason: str | None = None
        proposed: list[ProposedSubQuery] = []
        try:
            # Authoritative timeout enforcement: wraps the call regardless of whether the
            # DecompositionLLM implementation honors timeout_ms itself (original TRD §6.3 -
            # "Decomposer LLM call fails/times out" is one failure mode, not two).
            raw = await asyncio.wait_for(
                llm.decompose(transcript, timeout_ms=settings.decompose_timeout_ms),
                settings.decompose_timeout_ms / 1000,
            )
            proposed = _validate_proposed(raw)
            if not proposed:
                fallback_reason = "llm_returned_no_usable_sub_queries"
        except Exception as exc:  # call failed, timed out, or was cancelled by the line above
            fallback_reason = f"llm_call_failed: {exc}"

        if fallback_reason is not None:
            # Exactly one fallback signal per failed decomposition attempt, whichever of the two
            # failure modes above produced it (a call that raises must not ALSO fall through to
            # the "no usable sub-queries" branch and double-emit).
            _emit_fallback(session_id, trace_id, sink, fallback_reason)
            sub_queries = _fallback(session_id, transcript, offset_ms)
        else:
            embedder = embedder or get_embedder()
            merged = await asyncio.to_thread(
                merge_similar, proposed, settings.merge_threshold, embedder
            )
            sub_queries = [
                SubQuery(
                    session_id=session_id,
                    text=sq.text,
                    intent_label=sq.intent_label,
                    parent_transcript_offset=offset_ms,
                    created_at=datetime.now(UTC),
                    merged_from=absorbed,
                )
                for sq, absorbed in merged
            ]

    for sub_query in sub_queries:
        _emit_subquery_created(sub_query, session_id, trace_id, sink)
    return sub_queries
