"""REQ-DEPLOY-01: /health is always 200; /ready gates on Qdrant connectivity, ingestion having
completed, and the embedding model being warmed."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from tests.fakes import FakeEmbedder


def _wait_until_embedder_warm(client, attempts: int = 50):
    """The warmup task runs in the background (never blocks /health); poll briefly rather than
    assume it has already completed by the first request."""
    resp = client.get("/ready")
    for _ in range(attempts):
        if resp.json().get("embedder"):
            return resp
        time.sleep(0.01)
        resp = client.get("/ready")
    return resp


def test_health_always_200(monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "get_embedder", FakeEmbedder)
    app = main_module.create_app()
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


def test_ready_gates_on_qdrant_connectivity(monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "_check_qdrant", lambda url: False)
    monkeypatch.setattr(main_module, "get_embedder", FakeEmbedder)
    app = main_module.create_app()
    with TestClient(app) as client:
        resp = client.get("/ready")
        assert resp.status_code == 503
        assert resp.json()["qdrant"] is False


def test_ready_ok_when_qdrant_ingestion_and_embedder_ready(monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "_check_qdrant", lambda url: True)
    monkeypatch.setattr(main_module, "get_embedder", FakeEmbedder)
    app = main_module.create_app()
    with TestClient(app) as client:
        resp = _wait_until_embedder_warm(client)
        assert resp.status_code == 200
        assert resp.json() == {
            "status": "ready",
            "qdrant": True,
            "ingestion": True,
            "embedder": True,
        }


def test_ready_stays_not_ready_while_embedder_is_not_yet_warm(monkeypatch):
    """The embedder gate must actually gate — a slow warmup must not report ready early."""
    import app.main as main_module

    monkeypatch.setattr(main_module, "_check_qdrant", lambda url: True)
    monkeypatch.setattr(main_module, "get_embedder", lambda: FakeEmbedder(delay_s=0.2))
    app = main_module.create_app()
    with TestClient(app) as client:
        resp = client.get("/ready")
        assert resp.status_code == 503
        assert resp.json() == {
            "status": "not_ready",
            "qdrant": True,
            "ingestion": True,
            "embedder": False,
        }


def test_ready_stays_not_ready_when_warmup_fails(monkeypatch):
    import app.main as main_module

    class BrokenEmbedder:
        def warmup(self) -> None:
            raise RuntimeError("model load failed")

    monkeypatch.setattr(main_module, "_check_qdrant", lambda url: True)
    monkeypatch.setattr(main_module, "get_embedder", BrokenEmbedder)
    app = main_module.create_app()
    with TestClient(app) as client:
        time.sleep(0.05)  # let the background warmup task run and fail
        resp = client.get("/ready")
        assert resp.status_code == 503
        assert resp.json()["embedder"] is False
