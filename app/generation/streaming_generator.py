"""Streaming Answer Generator (pseudocode 12.I; REQ-GROUND-01/02/03).

Orchestrates one turn's generation: build_prompt -> stream tokens from the configured LLM ->
split into sentences -> validate each against the evidence set (citation_validator, no LLM
required) -> emit ANSWER_DELTA per sentence, regenerating a failing sentence once (pseudocode
12.F's `attempt >= 1` cap) before falling back to an uncertainty statement -> emit
ANSWER_VERSION_CREATED once the whole answer is complete.

The generation LLM is swappable via the same LLM_PROVIDER setting Phase 4's decomposer uses
(anthropic | gemini) - a *different* interface (free-text streaming, not structured tool-use/JSON
mode), but the same lazy-singleton-per-provider pattern and the same "never trust the provider's
own timeout" orchestrator-level enforcement (app.decomposition.multi_intent.decompose).
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any, Protocol, runtime_checkable

import structlog

from app.core.config import Settings, get_settings
from app.core.events import EventSink, EventType, TelemetryEvent
from app.generation.prompt_builder import build_prompt
from app.grounding.citation_validator import (
    Accept,
    EvidenceChunk,
    Uncertain,
    build_evidence_lookup,
    finalize_sentence,
    stamp_citation_ids,
    uncertainty_text,
)
from app.models import new_id
from app.models.answer_version import AnswerVersion
from app.models.citation import Citation
from app.models.evidence import Evidence
from app.models.retrieval_event import RetrievalResult
from app.models.sub_query import SubQuery
from app.session.session_store import SessionRecord

log = structlog.get_logger()

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_REGENERATE_MAX_ATTEMPTS = 1  # pseudocode 12.F: attempt >= 1 stops retrying

_SYSTEM_PROMPT = (
    "You answer questions using only the evidence block provided. Every factual sentence must "
    "end with a citation tag in the exact form [doc_id section], copied verbatim from the "
    "evidence block. Never invent a tag and never cite one that isn't listed. If the evidence "
    "does not answer a question, say so plainly. When sources disagree, state both values with "
    "their own citations."
)

_REGEN_SYSTEM_PROMPT = (
    "Rewrite the given sentence so it ends with a citation tag [doc_id section] chosen only from "
    "the tags listed, copied verbatim. Keep the factual content the same. Respond with only the "
    "rewritten sentence."
)


@runtime_checkable
class GenerationLLM(Protocol):
    def stream(self, system: str, prompt: str) -> AsyncIterator[str]: ...

    async def complete(self, system: str, prompt: str) -> str: ...


class AnthropicGenerator:
    """Wraps the configured Claude model behind GenerationLLM. Lazily constructs the SDK client
    on first use, same pattern as app.decomposition.multi_intent.AnthropicDecomposer."""

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        return self._client

    async def stream(self, system: str, prompt: str) -> AsyncIterator[str]:
        client = self._get_client()
        async with client.messages.stream(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def complete(self, system: str, prompt: str) -> str:
        client = self._get_client()
        response = await client.messages.create(
            model=self._model,
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )


class GeminiGenerator:
    """Wraps the configured Gemini model behind GenerationLLM, same lazy-client pattern as
    app.decomposition.multi_intent.GeminiDecomposer."""

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    async def stream(self, system: str, prompt: str) -> AsyncIterator[str]:
        from google.genai import types

        client = self._get_client()
        response = await client.aio.models.generate_content_stream(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system),
        )
        async for chunk in response:
            if chunk.text:
                yield chunk.text

    async def complete(self, system: str, prompt: str) -> str:
        from google.genai import types

        client = self._get_client()
        response = await client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system),
        )
        return response.text or ""


_PROVIDERS = {"anthropic", "gemini"}


@lru_cache(maxsize=1)
def get_generator() -> GenerationLLM:
    settings = get_settings()
    if settings.llm_provider == "gemini":
        return GeminiGenerator(settings.gemini_api_key, settings.gemini_model)
    if settings.llm_provider == "anthropic":
        return AnthropicGenerator(settings.anthropic_api_key, settings.llm_model)
    raise ValueError(
        f"Unknown LLM_PROVIDER {settings.llm_provider!r}; expected one of {sorted(_PROVIDERS)}"
    )


def _split_sentence(buffer: str) -> tuple[str, str] | None:
    match = _SENTENCE_BOUNDARY.search(buffer)
    if match is None:
        return None
    return buffer[: match.start() + 1].strip(), buffer[match.end() :]


async def _stream_with_timeout(
    llm: GenerationLLM, system: str, prompt: str, timeout_ms: int
) -> AsyncIterator[str]:
    """Authoritative timeout enforcement over the *whole* stream, the same principle as
    decompose()'s asyncio.wait_for — no provider implementation is trusted to self-enforce a
    deadline. A stream that stalls mid-way raises TimeoutError, caught by the caller."""

    async def _drain(queue: asyncio.Queue[str | None]) -> None:
        try:
            async for token in llm.stream(system, prompt):
                await queue.put(token)
        finally:
            await queue.put(None)

    queue: asyncio.Queue[str | None] = asyncio.Queue()
    producer = asyncio.create_task(_drain(queue))
    try:
        while True:
            token = await asyncio.wait_for(queue.get(), timeout_ms / 1000)
            if token is None:
                break
            yield token
    finally:
        producer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await producer


async def stream_answer(
    session: SessionRecord,
    sub_queries: list[SubQuery],
    evidence: list[Evidence],
    retrievals: list[RetrievalResult],
    *,
    trace_id: str,
    llm: GenerationLLM | None = None,
    settings: Settings | None = None,
    sink: EventSink | None = None,
) -> AnswerVersion | None:
    """Returns the newly committed AnswerVersion, or None if nothing was generated (no
    sub-queries, or REQ-GROUND-02 insufficient evidence for every sub-intent)."""
    settings = settings or get_settings()
    if not sub_queries:
        return None

    chunks_by_id = {c.chunk_id: c for r in retrievals for c in (*r.dense, *r.sparse)}
    evidence_by_tag = build_evidence_lookup(evidence, chunks_by_id)
    active_evidence = [e for e in evidence if e.duplicate_of is None]
    top_score = max((e.rerank_score or 0.0) for e in active_evidence) if active_evidence else 0.0

    if not active_evidence or top_score < settings.min_relevance:
        # REQ-GROUND-02: nothing usable to ground an answer in - say so, never guess.
        for sub_query in sub_queries:
            _emit_uncertainty(
                session,
                trace_id,
                sink,
                sub_query.sub_query_id,
                "insufficient_evidence",
                f"Could you provide more detail about {sub_query.text}?",
            )
        return None

    llm = llm or get_generator()
    prior_version = session.answer_versions[-1] if session.answer_versions else None
    prompt = build_prompt(sub_queries, evidence_by_tag, prior_version=prior_version)
    answer_version_id = new_id()
    version_no = (prior_version.version_no + 1) if prior_version else 1

    sentences: list[str] = []
    citations: list[Citation] = []
    buffer = ""
    try:
        async for token in _stream_with_timeout(
            llm, _SYSTEM_PROMPT, prompt, settings.generation_timeout_ms
        ):
            buffer += token
            while True:
                split = _split_sentence(buffer)
                if split is None:
                    break
                sentence, buffer = split
                text, sentence_citations = await _finalize_and_emit(
                    session,
                    trace_id,
                    sink,
                    sentence,
                    evidence_by_tag,
                    sub_queries,
                    answer_version_id,
                    version_no,
                    settings,
                    llm,
                    is_final=False,
                )
                sentences.append(text)
                citations.extend(sentence_citations)
    except Exception as exc:
        log.info("generation_failed", session_id=session.session_id, reason=str(exc))
        _emit_error(session, trace_id, sink, "GenerationFailed", str(exc))

    if buffer.strip():
        text, sentence_citations = await _finalize_and_emit(
            session,
            trace_id,
            sink,
            buffer.strip(),
            evidence_by_tag,
            sub_queries,
            answer_version_id,
            version_no,
            settings,
            llm,
            is_final=True,
        )
        sentences.append(text)
        citations.extend(sentence_citations)

    if not sentences:
        return None

    answer_version = AnswerVersion(
        answer_version_id=answer_version_id,
        session_id=session.session_id,
        version_no=version_no,
        text=" ".join(sentences),
        citations=[c.citation_id for c in citations],
        supersedes=prior_version.version_no if prior_version else None,
        new_claim_ids=[c.citation_id for c in citations],
        created_at=datetime.now(UTC),
    )
    session.answer_versions.append(answer_version)
    _emit_answer_version_created(session, trace_id, sink, answer_version)
    return answer_version


async def _finalize_and_emit(
    session: SessionRecord,
    trace_id: str,
    sink: EventSink | None,
    sentence: str,
    evidence_by_tag: dict[str, EvidenceChunk],
    sub_queries: list[SubQuery],
    answer_version_id: str,
    version_no: int,
    settings: Settings,
    llm: GenerationLLM,
    *,
    is_final: bool,
) -> tuple[str, list[Citation]]:
    result = finalize_sentence(sentence, evidence_by_tag, entailment_min=settings.entailment_min)

    attempt = 0
    while isinstance(result, Uncertain) and attempt < _REGENERATE_MAX_ATTEMPTS:
        attempt += 1
        try:
            regen_prompt = (
                f"Available tags: {', '.join(sorted(evidence_by_tag))}\nSentence: {sentence}"
            )
            regenerated = await asyncio.wait_for(
                llm.complete(_REGEN_SYSTEM_PROMPT, regen_prompt),
                settings.generation_timeout_ms / 1000,
            )
        except Exception as exc:
            log.info("regeneration_failed", session_id=session.session_id, reason=str(exc))
            break
        sentence = regenerated.strip()
        result = finalize_sentence(
            sentence, evidence_by_tag, entailment_min=settings.entailment_min
        )

    if isinstance(result, Accept):
        citations = stamp_citation_ids(result.citations, answer_version_id)
        text = result.text
        for citation in citations:
            _emit_citation_created(session, trace_id, sink, citation)
    else:
        sub_query_id = sub_queries[0].sub_query_id if sub_queries else ""
        _emit_uncertainty(session, trace_id, sink, sub_query_id, result.reason, None)
        text = uncertainty_text(result.reason)
        citations = []

    _emit_answer_delta(session, trace_id, sink, version_no, text, is_final)
    return text, citations


def _emit_answer_delta(
    session: SessionRecord,
    trace_id: str,
    sink: EventSink | None,
    version_no: int,
    text_delta: str,
    is_final_sentence: bool,
) -> None:
    if sink is None:
        return
    sink(
        TelemetryEvent(
            session_id=session.session_id,
            trace_id=trace_id,
            event_type=EventType.ANSWER_DELTA,
            payload={
                "version_no": version_no,
                "text_delta": text_delta,
                "is_final_sentence": is_final_sentence,
            },
        )
    )


def _emit_answer_version_created(
    session: SessionRecord, trace_id: str, sink: EventSink | None, answer_version: AnswerVersion
) -> None:
    if sink is None:
        return
    sink(
        TelemetryEvent(
            session_id=session.session_id,
            trace_id=trace_id,
            event_type=EventType.ANSWER_VERSION_CREATED,
            payload={
                "version_no": answer_version.version_no,
                "text": answer_version.text,
                "citations": list(answer_version.citations),
                "supersedes": answer_version.supersedes,
            },
        )
    )


def _emit_citation_created(
    session: SessionRecord, trace_id: str, sink: EventSink | None, citation: Citation
) -> None:
    if sink is None:
        return
    sink(
        TelemetryEvent(
            session_id=session.session_id,
            trace_id=trace_id,
            event_type=EventType.CITATION_CREATED,
            payload={
                "chunk_id": citation.chunk_id,
                "doc_id": citation.doc_id,
                "section": citation.section,
                "claim_text": citation.claim_text,
            },
        )
    )


def _emit_uncertainty(
    session: SessionRecord,
    trace_id: str,
    sink: EventSink | None,
    sub_query_id: str,
    reason: str,
    clarifying_question: str | None,
) -> None:
    if sink is None:
        return
    sink(
        TelemetryEvent(
            session_id=session.session_id,
            trace_id=trace_id,
            event_type=EventType.UNCERTAINTY,
            payload={
                "sub_query_id": sub_query_id,
                "reason": reason,
                "clarifying_question": clarifying_question,
            },
        )
    )


def _emit_error(
    session: SessionRecord, trace_id: str, sink: EventSink | None, error_type: str, message: str
) -> None:
    if sink is None:
        return
    sink(
        TelemetryEvent(
            session_id=session.session_id,
            trace_id=trace_id,
            event_type=EventType.ERROR,
            payload={
                "stage": "generation",
                "error_type": error_type,
                "message": message,
                "recoverable": True,
            },
        )
    )
