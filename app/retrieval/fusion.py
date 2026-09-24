"""Evidence Fusion (pseudocode 12.D; REQ-EVID-02, REQ-EVID-04).

Two pure-function stages, run after all of one utterance's sub-queries have been retrieved
(REQ-EVID-01, hybrid.py) and before reranking (REQ-EVID-03, reranking/cross_encoder.py):

1. `rrf_fuse` — Reciprocal Rank Fusion combines each sub-query's own dense + sparse rank lists
   into one fused score per chunk (`rrf_k`=60), exactly as pseudocode 12.D's `fuse()` does. This
   produces one `Evidence` row per (sub_query_id, chunk_id) pair with `fusion_score` set and
   everything else still unset.
2. `dedup` — collapses duplicate chunks *globally, across all sub-queries* (REQ-EVID-02's "RRF...
   per sub-query and globally across sub-queries"): two rows are duplicates if they share the same
   `chunk_id` (exact) or their chunks' text cosine similarity exceeds `DEDUP_THRESHOLD` (near-
   duplicate content under different chunk_ids, e.g. overlapping ingestion windows). The
   higher-fusion_score instance is kept; every other instance in its group gets `duplicate_of` set
   to the kept instance's `evidence_id` and is excluded from reranking, but the row itself is kept
   in the returned list for telemetry/audit (REQ-EVID-02's "retained... for telemetry").
3. `flag_contradictions` — REQ-EVID-04. Only meaningful for numeric slots (a categorical slot's
   extracted value is always its own canonical pattern string, never the chunk's actual free-text
   content, so two chunks about the same categorical topic never "disagree" under that
   extraction). Two *deduped, non-duplicate* evidence chunks are flagged as a `contradiction_pair`
   when the corpus's SlotSchema declares an entity-key slot (e.g. `venue`), both chunks name the
   *same* entity, and they carry *different* values for some numeric slot (e.g. `capacity`). A
   corpus with no entity-key slot simply has no contradiction detection — a safe default.
"""

from __future__ import annotations

import math
from collections import defaultdict

from app.controller.entity_extraction import extract_entities
from app.core.embeddings import Embedder
from app.core.slots import get_slot_schema
from app.models import new_id
from app.models.chunk import RetrievedChunk
from app.models.evidence import Evidence
from app.models.retrieval_event import RetrievalResult


def rrf_fuse(
    retrieval_results: list[RetrievalResult], rrf_k: int
) -> tuple[list[Evidence], dict[str, RetrievedChunk]]:
    """Pseudocode 12.D `fuse()`'s RRF half. One Evidence row per (sub_query_id, chunk_id) pair
    that appeared in that sub-query's dense and/or sparse results, `fusion_score` = the sum of
    1/(rrf_k + rank + 1) over every list (dense, sparse) the chunk appeared in for that sub-query.
    Not yet deduped across sub-queries — see `dedup`."""
    chunks_by_id: dict[str, RetrievedChunk] = {}
    evidence: list[Evidence] = []
    for result in retrieval_results:
        scores: dict[str, float] = defaultdict(float)
        for rank, chunk in enumerate(result.dense):
            scores[chunk.chunk_id] += 1.0 / (rrf_k + rank + 1)
            chunks_by_id[chunk.chunk_id] = chunk
        for rank, chunk in enumerate(result.sparse):
            scores[chunk.chunk_id] += 1.0 / (rrf_k + rank + 1)
            chunks_by_id[chunk.chunk_id] = chunk
        for chunk_id, score in scores.items():
            evidence.append(
                Evidence(sub_query_id=result.sub_query_id, chunk_id=chunk_id, fusion_score=score)
            )
    return evidence, chunks_by_id


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return 0.0 if norm_a == 0 or norm_b == 0 else dot / (norm_a * norm_b)


def dedup(
    evidence: list[Evidence],
    chunks_by_id: dict[str, RetrievedChunk],
    threshold: float,
    embedder: Embedder,
) -> list[Evidence]:
    """REQ-EVID-02. Mutates and returns `evidence`: every row's `duplicate_of` is set except the
    one kept per duplicate group. Grouping is global (every sub-query's rows are compared against
    every other's), which is what makes this the "globally across sub-queries" half of fusion."""
    order = sorted(range(len(evidence)), key=lambda i: -evidence[i].fusion_score)
    kept_indices: list[int] = []
    kept_embeddings: list[tuple[float, ...] | None] = []

    for i in order:
        chunk = chunks_by_id[evidence[i].chunk_id]
        dup_of: int | None = None
        for j_pos, j in enumerate(kept_indices):
            kept_chunk = chunks_by_id[evidence[j].chunk_id]
            if chunk.chunk_id == kept_chunk.chunk_id:
                dup_of = j
                break
            if kept_embeddings[j_pos] is None:
                kept_embeddings[j_pos] = embedder.embed(kept_chunk.text)
            if _cosine(embedder.embed(chunk.text), kept_embeddings[j_pos]) > threshold:  # type: ignore[arg-type]
                dup_of = j
                break
        if dup_of is None:
            kept_indices.append(i)
            kept_embeddings.append(None)
        else:
            evidence[i].duplicate_of = evidence[dup_of].evidence_id

    return evidence


def flag_contradictions(
    evidence: list[Evidence], chunks_by_id: dict[str, RetrievedChunk], corpus_id: str
) -> list[Evidence]:
    """REQ-EVID-04. Mutates and returns `evidence`: sets `contradiction_pair_id` (a shared id) on
    every pair of non-duplicate rows that name the same entity but disagree on a numeric slot."""
    schema = get_slot_schema(corpus_id)
    entity_slot = schema.entity_key_slot()
    if entity_slot is None:
        return evidence

    numeric_slots = {s.name for s in schema.slots if s.value_type == "numeric"}
    active = [e for e in evidence if e.duplicate_of is None]
    extracted = {
        e.evidence_id: extract_entities(chunks_by_id[e.chunk_id].text, corpus_id) for e in active
    }

    for i, a in enumerate(active):
        entities_a = extracted[a.evidence_id]
        entity_value_a = entities_a.get(entity_slot)
        if entity_value_a is None:
            continue
        for b in active[i + 1 :]:
            entities_b = extracted[b.evidence_id]
            if entities_b.get(entity_slot) != entity_value_a:
                continue  # not about the same entity - not comparable
            for slot_name in numeric_slots:
                value_a = entities_a.get(slot_name)
                value_b = entities_b.get(slot_name)
                if value_a is None or value_b is None or value_a == value_b:
                    continue
                pair_id = a.contradiction_pair_id or b.contradiction_pair_id or new_id()
                a.contradiction_pair_id = pair_id
                b.contradiction_pair_id = pair_id

    return evidence
