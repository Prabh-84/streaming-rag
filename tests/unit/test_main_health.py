"""REQ-DEPLOY-01: /health is always 200; /ready gates on Qdrant connectivity, ingestion having
completed, and the embedding + spaCy models being warmed."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from tests.fakes import FakeEmbedder


async def _fake_ingest_corpus(corpus_id, *, settings, embedder, qdrant_client) -> None:
    return None


def _patch_fast_ingestion(monkeypatch, main_module) -> None:
    """Startup ingestion (Phase 10) defaults to instant/no-op: one fake corpus_id "discovered",
    ingesting it does nothing real. Without this, ingestion_complete would stay False forever in
    these tests (CORPUS_ROOT here has no real corpus directories), and /ready would never turn
    ready - a real Qdrant/embedder/nlp concern is not what these tests are checking."""
    monkeypatch.setattr(main_module, "discover_corpora", lambda corpus_root: ["fake"])
    monkeypatch.setattr(main_module, "ingest_corpus", _fake_ingest_corpus)


def _patch_fast_warmup(monkeypatch, main_module) -> None:
    """All three startup background tasks default to instant/no-op so tests don't pay a real
    model-load or ingestion cost."""
    monkeypatch.setattr(main_module, "get_embedder", FakeEmbedder)
    monkeypatch.setattr(main_module.entity_extraction, "warmup", lambda: None)
    _patch_fast_ingestion(monkeypatch, main_module)


def _wait_until_ready(client, attempts: int = 50):
    """Warmup tasks run in the background (never blocking /health); poll briefly rather than
    assume they have already completed by the first request."""
    resp = client.get("/ready")
    for _ in range(attempts):
        if resp.json().get("status") == "ready":
            return resp
        time.sleep(0.01)
        resp = client.get("/ready")
    return resp


def test_health_always_200(monkeypatch):
    import app.main as main_module

    _patch_fast_warmup(monkeypatch, main_module)
    app = main_module.create_app()
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


def test_ready_gates_on_qdrant_connectivity(monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "_check_qdrant", lambda url: False)
    _patch_fast_warmup(monkeypatch, main_module)
    app = main_module.create_app()
    with TestClient(app) as client:
        resp = client.get("/ready")
        assert resp.status_code == 503
        assert resp.json()["qdrant"] is False


def test_ready_ok_when_everything_is_warm(monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "_check_qdrant", lambda url: True)
    _patch_fast_warmup(monkeypatch, main_module)
    app = main_module.create_app()
    with TestClient(app) as client:
        resp = _wait_until_ready(client)
        assert resp.status_code == 200
        assert resp.json() == {
            "status": "ready",
            "qdrant": True,
            "ingestion": True,
            "embedder": True,
            "nlp": True,
        }


def test_ready_stays_not_ready_while_embedder_is_not_yet_warm(monkeypatch):
    """The embedder gate must actually gate — a slow warmup must not report ready early."""
    import app.main as main_module

    monkeypatch.setattr(main_module, "_check_qdrant", lambda url: True)
    monkeypatch.setattr(main_module, "get_embedder", lambda: FakeEmbedder(delay_s=0.2))
    monkeypatch.setattr(main_module.entity_extraction, "warmup", lambda: None)
    _patch_fast_ingestion(monkeypatch, main_module)
    app = main_module.create_app()
    with TestClient(app) as client:
        resp = client.get("/ready")
        assert resp.status_code == 503
        assert resp.json() == {
            "status": "not_ready",
            "qdrant": True,
            "ingestion": True,
            "embedder": False,
            "nlp": True,
        }


def test_ready_stays_not_ready_while_nlp_is_not_yet_warm(monkeypatch):
    import app.main as main_module

    def slow_warmup() -> None:
        time.sleep(0.2)

    monkeypatch.setattr(main_module, "_check_qdrant", lambda url: True)
    monkeypatch.setattr(main_module, "get_embedder", FakeEmbedder)
    monkeypatch.setattr(main_module.entity_extraction, "warmup", slow_warmup)
    _patch_fast_ingestion(monkeypatch, main_module)
    app = main_module.create_app()
    with TestClient(app) as client:
        resp = client.get("/ready")
        assert resp.status_code == 503
        assert resp.json()["nlp"] is False


def test_ready_stays_not_ready_when_warmup_fails(monkeypatch):
    import app.main as main_module

    class BrokenEmbedder:
        model_name = "broken-embedder"
        dimension = (
            384  # must look structurally valid: DenseIndex is also built from get_embedder()
        )

        def warmup(self) -> None:
            raise RuntimeError("model load failed")

    monkeypatch.setattr(main_module, "_check_qdrant", lambda url: True)
    monkeypatch.setattr(main_module, "get_embedder", BrokenEmbedder)
    monkeypatch.setattr(main_module.entity_extraction, "warmup", lambda: None)
    app = main_module.create_app()
    with TestClient(app) as client:
        time.sleep(0.05)  # let the background warmup task run and fail
        resp = client.get("/ready")
        assert resp.status_code == 503
        assert resp.json()["embedder"] is False


def test_ready_stays_not_ready_when_no_corpus_is_mounted(monkeypatch):
    """REQ-DEPLOY-01 / PRD_TRD.md §11 risk register: "corpus not mounted" must surface as /ready
    never turning healthy, not as a hard startup crash or a silently-empty index."""
    import app.main as main_module

    monkeypatch.setattr(main_module, "_check_qdrant", lambda url: True)
    monkeypatch.setattr(main_module, "get_embedder", FakeEmbedder)
    monkeypatch.setattr(main_module.entity_extraction, "warmup", lambda: None)
    monkeypatch.setattr(main_module, "discover_corpora", lambda corpus_root: [])
    app = main_module.create_app()
    with TestClient(app) as client:
        time.sleep(0.05)  # let the background ingestion task run (and find nothing to ingest)
        resp = client.get("/ready")
        assert resp.status_code == 503
        assert resp.json()["ingestion"] is False


def test_ready_stays_not_ready_when_a_discovered_corpus_fails_to_ingest(monkeypatch):
    """A corpus that IS mounted but fails validation/ingestion must not be reported as complete
    just because some other corpus (or none) would have succeeded."""
    import app.main as main_module

    async def _broken_ingest(corpus_id, *, settings, embedder, qdrant_client):
        raise RuntimeError("injected ingestion failure")

    monkeypatch.setattr(main_module, "_check_qdrant", lambda url: True)
    monkeypatch.setattr(main_module, "get_embedder", FakeEmbedder)
    monkeypatch.setattr(main_module.entity_extraction, "warmup", lambda: None)
    monkeypatch.setattr(main_module, "discover_corpora", lambda corpus_root: ["broken"])
    monkeypatch.setattr(main_module, "ingest_corpus", _broken_ingest)
    app = main_module.create_app()
    with TestClient(app) as client:
        time.sleep(0.05)  # let the background ingestion task run (and fail)
        resp = client.get("/ready")
        assert resp.status_code == 503
        assert resp.json()["ingestion"] is False
