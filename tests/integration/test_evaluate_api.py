"""POST /evaluate, GET /evaluate/{run_id} (docs/API.md §7-8; REQ-EVAL-01; REQ-SEC-04).

Exercises the endpoint's own contract (auth, queue/poll shape, 409 on a duplicate in-flight run)
against a stubbed benchmark run - the actual benchmarks/streaming_suite_v1 replay is exercised
separately by benchmarks/ (harness) and scripts/run_benchmark.py, never here: this file must not
know the suite's name or any BENCH-nn identifier (REQ-EVAL-01/HC-2).
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient
from qdrant_client import AsyncQdrantClient

from app.controller.entity_extraction import get_corpus_matcher
from app.core.config import get_settings
from app.core.slots import get_slot_schema
from scripts.ingest_corpus import ingest_corpus
from tests.fakes import FIXTURE_CORPORA, FakeEmbedder

API_KEY = "change_me_local_dev"
EVAL_KEY = "change_me_eval_test"


@pytest.fixture
async def app_client(tmp_path, monkeypatch):
    monkeypatch.setenv("CORPUS_ROOT", str(FIXTURE_CORPORA))
    monkeypatch.setenv("PROCESSED_DIR", str(tmp_path))
    monkeypatch.setenv("API_KEY", API_KEY)
    monkeypatch.setenv("EVAL_KEY", EVAL_KEY)
    get_settings.cache_clear()
    get_slot_schema.cache_clear()
    get_corpus_matcher.cache_clear()

    settings = get_settings()
    embedder = FakeEmbedder()
    qdrant_client = AsyncQdrantClient(location=":memory:")
    await ingest_corpus("alpha", settings=settings, embedder=embedder, qdrant_client=qdrant_client)

    import app.main as main_module

    monkeypatch.setattr(main_module, "create_qdrant_client", lambda url: qdrant_client)
    monkeypatch.setattr(main_module, "get_embedder", lambda: embedder)

    app = main_module.create_app()
    with TestClient(app) as client:
        yield client

    get_settings.cache_clear()
    get_slot_schema.cache_clear()
    get_corpus_matcher.cache_clear()


def auth_headers(*, eval_key: str | None = EVAL_KEY) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {API_KEY}"}
    if eval_key is not None:
        headers["X-Eval-Key"] = eval_key
    return headers


class _FakeReport:
    def __init__(self) -> None:
        self.gates = {"G2": {"value": 0.9, "target": 0.8, "pass": True}}
        self.metrics = {"early_retrieval_rate": 0.9}


def _patch_harness(monkeypatch, *, delay_s: float = 0.0, raises: bool = False):
    import benchmarks.harness as harness

    def _fake_run_benchmark(*, use_fakes: bool = False):  # noqa: ARG001
        if raises:
            raise RuntimeError("injected benchmark failure")
        if delay_s:
            time.sleep(delay_s)
        return _FakeReport()

    monkeypatch.setattr(harness, "run_benchmark", _fake_run_benchmark)


def test_evaluate_requires_api_key(app_client, monkeypatch):
    _patch_harness(monkeypatch)
    resp = app_client.post(
        "/evaluate",
        json={"test_set": "benchmarks/streaming_suite_v1", "corpus_id": "alpha"},
        headers={"X-Eval-Key": EVAL_KEY},
    )
    assert resp.status_code == 401


def test_evaluate_requires_eval_key(app_client, monkeypatch):
    _patch_harness(monkeypatch)
    resp = app_client.post(
        "/evaluate",
        json={"test_set": "benchmarks/streaming_suite_v1", "corpus_id": "alpha"},
        headers=auth_headers(eval_key=None),
    )
    assert resp.status_code == 403


def test_evaluate_rejects_wrong_eval_key(app_client, monkeypatch):
    _patch_harness(monkeypatch)
    resp = app_client.post(
        "/evaluate",
        json={"test_set": "benchmarks/streaming_suite_v1", "corpus_id": "alpha"},
        headers=auth_headers(eval_key="not_the_real_key"),
    )
    assert resp.status_code == 403


def test_evaluate_queues_and_reports_completion(app_client, monkeypatch):
    _patch_harness(monkeypatch, delay_s=0.2)
    resp = app_client.post(
        "/evaluate",
        json={"test_set": "benchmarks/streaming_suite_v1", "corpus_id": "alpha"},
        headers=auth_headers(),
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["status"] == "queued"
    run_id = body["run_id"]

    # Poll GET /evaluate/{run_id} until it completes (docs/API.md §8: "not polled synchronously").
    for _ in range(50):
        poll = app_client.get(f"/evaluate/{run_id}", headers=auth_headers())
        assert poll.status_code == 200
        if poll.json()["status"] == "complete":
            break
        time.sleep(0.05)
    else:
        pytest.fail("evaluation run never completed")

    result = poll.json()
    assert result["gates"]["G2"]["pass"] is True
    assert result["metrics"]["early_retrieval_rate"] == 0.9


def test_evaluate_unknown_run_id_returns_404(app_client, monkeypatch):
    _patch_harness(monkeypatch)
    resp = app_client.get("/evaluate/does_not_exist", headers=auth_headers())
    assert resp.status_code == 404


def test_evaluate_second_run_for_same_test_set_returns_409_while_in_flight(app_client, monkeypatch):
    _patch_harness(monkeypatch, delay_s=2.0)
    payload = {"test_set": "benchmarks/streaming_suite_v1", "corpus_id": "alpha"}
    first = app_client.post("/evaluate", json=payload, headers=auth_headers())
    assert first.status_code == 202

    second = app_client.post("/evaluate", json=payload, headers=auth_headers())
    assert second.status_code == 409


def test_evaluate_failure_is_reported_not_raised(app_client, monkeypatch):
    _patch_harness(monkeypatch, raises=True)
    resp = app_client.post(
        "/evaluate",
        json={"test_set": "benchmarks/streaming_suite_v1", "corpus_id": "alpha"},
        headers=auth_headers(),
    )
    run_id = resp.json()["run_id"]

    for _ in range(50):
        poll = app_client.get(f"/evaluate/{run_id}", headers=auth_headers())
        if poll.json()["status"] in ("complete", "failed"):
            break
        time.sleep(0.05)
    else:
        pytest.fail("evaluation run never settled")

    assert poll.json()["status"] == "failed"
