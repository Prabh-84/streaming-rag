"""Reranking with global evidence cap (pseudocode 12.E; REQ-EVID-03, resolution #6).

Three stages, run per RETRIEVE turn after fusion (fusion.py):

1. `rerank_subquery` — cross-encoder scores (sub_query.text, chunk.text) pairs for one sub-query's
   deduped candidates (top `RERANK_CANDIDATES` by fusion_score, matching pseudocode 12.E's
   `candidates[:RERANK_CANDIDATES]` slice), keeps the top `FINAL_K` scoring >= `MIN_RELEVANCE`.
2. `cap_global_evidence` — merges every sub-query's kept list; if the combined count exceeds
   `MAX_TOTAL_EVIDENCE`, truncates by global rerank score while guaranteeing every sub-query that
   had >=1 kept chunk retains at least one (protects Gate G3/G4 from one dominant sub-query
   starving another's citations).
3. `truncate_to_token_budget` — final prefix truncation (highest rerank score first) so the
   returned set never exceeds `EVIDENCE_TOKEN_BUDGET` tokens.

The cross-encoder model is loaded lazily (mirrors app.core.embeddings' pattern): importing this
module or starting the app never triggers a model download, only the first real RETRIEVE turn
that reaches reranking does.
"""

from __future__ import annotations

import threading
from functools import lru_cache
from typing import Any, Protocol, runtime_checkable

from app.core.config import get_settings
from app.models.chunk import RetrievedChunk
from app.models.evidence import Evidence


@runtime_checkable
class Reranker(Protocol):
    def score(self, pairs: list[tuple[str, str]]) -> list[float]: ...


class CrossEncoderReranker:
    """Wraps cross-encoder/ms-marco-MiniLM-L-6-v2 behind `Reranker`. Scores are sigmoid-activated
    so they land in (0, 1) — the only scale on which a fixed `MIN_RELEVANCE`=0.35 threshold is
    meaningful; the model's raw (pre-activation) output is an unbounded ranking logit."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: Any = None
        self._lock = threading.Lock()

    def _get_model(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder
                    from torch.nn import Sigmoid

                    self._model = CrossEncoder(
                        self._model_name, device="cpu", default_activation_function=Sigmoid()
                    )
        return self._model

    def warmup(self) -> None:
        self._get_model()

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        scores = self._get_model().predict(list(pairs), show_progress_bar=False)
        return [float(s) for s in scores]


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker(get_settings().reranker_model)


def rerank_subquery(
    sub_query_text: str,
    candidates: list[Evidence],
    chunks_by_id: dict[str, RetrievedChunk],
    reranker: Reranker,
    *,
    min_relevance: float,
    final_k: int,
    rerank_candidates: int,
) -> list[Evidence]:
    """Pseudocode 12.E `rerank()`. `candidates` should already be deduped (fusion.dedup) — this
    does not itself filter out duplicate_of rows, since a sub-query's candidate list is expected
    to be pre-filtered by the caller (the orchestrator in retrieval_controller.py)."""
    pool = sorted(candidates, key=lambda e: -e.fusion_score)[:rerank_candidates]
    if not pool:
        return []
    pairs = [(sub_query_text, chunks_by_id[e.chunk_id].text) for e in pool]
    scores = reranker.score(pairs)
    for e, s in zip(pool, scores, strict=True):
        e.rerank_score = s
    ranked = sorted(pool, key=lambda e: -(e.rerank_score or 0.0))
    return [e for e in ranked if (e.rerank_score or 0.0) >= min_relevance][:final_k]


def cap_global_evidence(
    per_subquery_kept: dict[str, list[Evidence]], max_total: int
) -> list[Evidence]:
    """REQ-EVID-03 resolution #6. Every non-empty sub-query's single best-scoring chunk is
    guaranteed a slot (the floor); the remaining budget is filled by global rerank score across
    everyone's leftovers. If the floor alone exceeds max_total (more contributing sub-queries than
    the cap), the floor itself is truncated by score — the guarantee cannot survive a cap smaller
    than the number of sub-intents needing it."""
    all_kept = [e for lst in per_subquery_kept.values() for e in lst]
    if len(all_kept) <= max_total:
        return sorted(all_kept, key=lambda e: -(e.rerank_score or 0.0))

    floor = [
        max(lst, key=lambda e: e.rerank_score or 0.0) for lst in per_subquery_kept.values() if lst
    ]
    floor.sort(key=lambda e: -(e.rerank_score or 0.0))
    if len(floor) >= max_total:
        return floor[:max_total]

    floor_ids = {e.evidence_id for e in floor}
    rest = sorted(
        (e for e in all_kept if e.evidence_id not in floor_ids),
        key=lambda e: -(e.rerank_score or 0.0),
    )[: max_total - len(floor)]
    return sorted(floor + rest, key=lambda e: -(e.rerank_score or 0.0))


def truncate_to_token_budget(
    evidence: list[Evidence], chunks_by_id: dict[str, RetrievedChunk], budget: int
) -> list[Evidence]:
    """Prefix truncation on `evidence`'s given order (expected: already rank-sorted by the
    caller) — keeps the highest-ranked items and stops at the first one that would push the
    cumulative chunk token count over `budget`."""
    kept: list[Evidence] = []
    total = 0
    for e in evidence:
        tokens = chunks_by_id[e.chunk_id].token_count
        if total + tokens > budget:
            break
        kept.append(e)
        total += tokens
    return kept
