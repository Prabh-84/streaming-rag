"""Reranking with global evidence cap (pseudocode 12.E; REQ-EVID-03) using FakeReranker — the
real cross-encoder/ms-marco-MiniLM-L-6-v2 model is exercised separately in
test_reranking_real_model.py (gated behind RUN_MODEL_TESTS=1, same convention as
test_embedding_model.py).
"""

from __future__ import annotations

from app.models.chunk import RetrievedChunk
from app.models.evidence import Evidence
from app.reranking.cross_encoder import (
    cap_global_evidence,
    rerank_subquery,
    truncate_to_token_budget,
)
from tests.fakes import FakeReranker


def _chunk(chunk_id: str, text: str, token_count: int | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="doc",
        corpus_id="alpha",
        section="§1",
        text=text,
        chunk_index=0,
        token_count=token_count if token_count is not None else len(text.split()),
        score=1.0,
        rank=1,
    )


def _evidence(sub_query_id: str, chunk_id: str, fusion_score: float = 0.5) -> Evidence:
    return Evidence(sub_query_id=sub_query_id, chunk_id=chunk_id, fusion_score=fusion_score)


# --- D. Reranker selection / order --------------------------------------------------------------


def test_rerank_order_follows_reranker_score_not_fusion_score():
    chunks = {
        "c1": _chunk("c1", "low relevance text"),
        "c2": _chunk("c2", "high relevance text"),
    }
    # c1 has the higher fusion_score, but the reranker prefers c2.
    candidates = [
        _evidence("sq1", "c1", fusion_score=0.9),
        _evidence("sq1", "c2", fusion_score=0.1),
    ]
    reranker = FakeReranker(score_fn=lambda q, t: 0.9 if "high" in t else 0.2)

    kept = rerank_subquery(
        "query text",
        candidates,
        chunks,
        reranker,
        min_relevance=0.0,
        final_k=6,
        rerank_candidates=30,
    )
    assert [e.chunk_id for e in kept] == ["c2", "c1"]
    assert kept[0].rerank_score == 0.9
    assert kept[1].rerank_score == 0.2


def test_rerank_candidates_cap_limits_how_many_get_scored():
    """Only the top RERANK_CANDIDATES (by fusion_score) are ever sent to the reranker (pseudocode
    12.E's candidates[:RERANK_CANDIDATES] slice) - protects the reranker from an unbounded pool."""
    chunks = {f"c{i}": _chunk(f"c{i}", f"text number {i}") for i in range(10)}
    candidates = [_evidence("sq1", f"c{i}", fusion_score=float(i)) for i in range(10)]
    reranker = FakeReranker(score_fn=lambda q, t: 1.0)

    rerank_subquery(
        "q", candidates, chunks, reranker, min_relevance=0.0, final_k=6, rerank_candidates=3
    )
    (call,) = reranker.calls
    assert len(call) == 3  # only the top-3-by-fusion_score were scored
    scored_texts = {text for _, text in call}
    assert scored_texts == {"text number 9", "text number 8", "text number 7"}


# --- E. FINAL_K -----------------------------------------------------------------------------------


def test_final_k_limits_kept_chunks_per_subquery():
    chunks = {f"c{i}": _chunk(f"c{i}", f"text {i}") for i in range(10)}
    candidates = [_evidence("sq1", f"c{i}", fusion_score=1.0) for i in range(10)]
    reranker = FakeReranker(score_fn=lambda q, t: 0.9)  # all above MIN_RELEVANCE

    kept = rerank_subquery(
        "q", candidates, chunks, reranker, min_relevance=0.35, final_k=6, rerank_candidates=30
    )
    assert len(kept) == 6


# --- F. Relevance threshold -----------------------------------------------------------------------


def test_min_relevance_threshold_drops_low_scoring_chunks():
    chunks = {
        "above": _chunk("above", "above threshold"),
        "below": _chunk("below", "below threshold"),
    }
    candidates = [_evidence("sq1", "above"), _evidence("sq1", "below")]
    reranker = FakeReranker(score_fn=lambda q, t: 0.5 if "above" in t else 0.2)

    kept = rerank_subquery(
        "q", candidates, chunks, reranker, min_relevance=0.35, final_k=6, rerank_candidates=30
    )
    assert [e.chunk_id for e in kept] == ["above"]


def test_zero_chunks_above_min_relevance_is_a_valid_empty_result():
    """REQ-EVID-03 failure behavior: a sub-query with 0 chunks above MIN_RELEVANCE returns an
    empty kept list (flagged low_evidence downstream), not an error."""
    chunks = {"c1": _chunk("c1", "irrelevant text")}
    candidates = [_evidence("sq1", "c1")]
    reranker = FakeReranker(score_fn=lambda q, t: 0.1)

    kept = rerank_subquery(
        "q", candidates, chunks, reranker, min_relevance=0.35, final_k=6, rerank_candidates=30
    )
    assert kept == []


def test_rerank_subquery_with_no_candidates_never_calls_reranker():
    reranker = FakeReranker()
    kept = rerank_subquery(
        "q", [], {}, reranker, min_relevance=0.35, final_k=6, rerank_candidates=30
    )
    assert kept == []
    assert reranker.calls == []


# --- G. Global max 15 / H. Minimum evidence per sub-intent ----------------------------------------


def _kept(sub_query_id: str, chunk_ids: list[str], scores: list[float]) -> list[Evidence]:
    items = []
    for chunk_id, score in zip(chunk_ids, scores, strict=True):
        e = Evidence(sub_query_id=sub_query_id, chunk_id=chunk_id, fusion_score=score)
        e.rerank_score = score
        items.append(e)
    return items


def test_global_evidence_cap_truncates_to_max_total():
    per_subquery_kept = {
        "sq1": _kept("sq1", [f"a{i}" for i in range(8)], [0.9 - i * 0.01 for i in range(8)]),
        "sq2": _kept("sq2", [f"b{i}" for i in range(8)], [0.8 - i * 0.01 for i in range(8)]),
        "sq3": _kept("sq3", [f"c{i}" for i in range(8)], [0.7 - i * 0.01 for i in range(8)]),
    }
    capped = cap_global_evidence(per_subquery_kept, max_total=15)
    assert len(capped) == 15
    # highest global scores kept: sq1 dominates since its scores are all highest.
    assert capped[0].chunk_id == "a0"


def test_evidence_cap_preserves_min_one_per_subintent():
    """A sub-query whose single chunk scores far below everyone else's must still survive the cap
    (protects Gate G3/G4 from one dominant sub-query starving another's citations)."""
    per_subquery_kept = {
        "weak_sq": _kept("weak_sq", ["low"], [0.01]),
        "strong_sq": _kept(
            "strong_sq", [f"hi{i}" for i in range(14)], [0.99 - i * 0.001 for i in range(14)]
        ),
    }
    capped = cap_global_evidence(per_subquery_kept, max_total=15)
    assert len(capped) == 15
    assert "low" in {e.chunk_id for e in capped}


def test_evidence_cap_under_budget_returns_everything_sorted():
    per_subquery_kept = {
        "sq1": _kept("sq1", ["a", "b"], [0.5, 0.9]),
    }
    capped = cap_global_evidence(per_subquery_kept, max_total=15)
    assert [e.chunk_id for e in capped] == ["b", "a"]


def test_evidence_cap_excludes_empty_subquery_from_floor():
    """A sub-query with zero kept chunks (low_evidence) contributes nothing - "excluded from the
    floor guarantee (nothing to guarantee)" per REQ-EVID-03's failure clause."""
    per_subquery_kept = {
        "empty_sq": [],
        "sq1": _kept("sq1", ["a"], [0.9]),
    }
    capped = cap_global_evidence(per_subquery_kept, max_total=15)
    assert [e.chunk_id for e in capped] == ["a"]


def test_evidence_cap_floor_larger_than_max_total_truncates_floor_by_score():
    per_subquery_kept = {f"sq{i}": _kept(f"sq{i}", [f"c{i}"], [float(i)]) for i in range(20)}
    capped = cap_global_evidence(per_subquery_kept, max_total=5)
    assert len(capped) == 5
    assert [e.chunk_id for e in capped] == ["c19", "c18", "c17", "c16", "c15"]


# --- I. Token budget --------------------------------------------------------------------------


def test_truncate_to_token_budget_stops_before_exceeding():
    chunks = {
        "a": _chunk("a", "x", token_count=1000),
        "b": _chunk("b", "x", token_count=1000),
        "c": _chunk("c", "x", token_count=1000),
        "d": _chunk("d", "x", token_count=1000),
    }
    evidence = [_evidence("sq1", cid) for cid in ["a", "b", "c", "d"]]
    kept = truncate_to_token_budget(evidence, chunks, budget=3000)
    assert [e.chunk_id for e in kept] == ["a", "b", "c"]  # 3000 tokens exactly; "d" would exceed


def test_truncate_to_token_budget_keeps_all_when_under_budget():
    chunks = {"a": _chunk("a", "x", token_count=100), "b": _chunk("b", "x", token_count=100)}
    evidence = [_evidence("sq1", "a"), _evidence("sq1", "b")]
    kept = truncate_to_token_budget(evidence, chunks, budget=3000)
    assert len(kept) == 2
