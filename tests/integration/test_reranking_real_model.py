"""The real cross-encoder/ms-marco-MiniLM-L-6-v2 model. Opt-in (downloads on first run):

RUN_MODEL_TESTS=1 pytest tests/integration/test_reranking_real_model.py
"""

from __future__ import annotations

import os

import pytest

from app.reranking.cross_encoder import CrossEncoderReranker

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1 to load the real model"
)


@pytest.fixture(scope="module")
def cross_encoder() -> CrossEncoderReranker:
    reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker.warmup()
    return reranker


def test_real_model_scores_are_sigmoid_activated_and_rank_relevant_text_first(cross_encoder):
    query = "What is the cancellation policy for Orion Hall?"
    pairs = [
        (query, "Orion Hall requires 14 days notice for a full refund on cancellations."),
        (query, "The kitchen offers vegetarian and gluten-free catering options."),
    ]
    scores = cross_encoder.score(pairs)
    assert len(scores) == 2
    assert all(0.0 <= s <= 1.0 for s in scores)  # sigmoid-activated, not a raw unbounded logit
    assert scores[0] > 0.35 > scores[1]  # relevant clears MIN_RELEVANCE, irrelevant doesn't
