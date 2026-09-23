"""REQ-DEPLOY-01: /health is always 200; /ready gates on live Qdrant connectivity."""

from app.main import create_app
from fastapi.testclient import TestClient


def test_health_always_200():
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


def test_ready_gates_on_qdrant_connectivity(monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "_check_qdrant", lambda url: False)
    app = main_module.create_app()
    with TestClient(app) as client:
        resp = client.get("/ready")
        assert resp.status_code == 503
        assert resp.json()["qdrant"] is False


def test_ready_ok_when_qdrant_and_ingestion_ready(monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "_check_qdrant", lambda url: True)
    app = main_module.create_app()
    with TestClient(app) as client:
        resp = client.get("/ready")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ready", "qdrant": True, "ingestion": True}
