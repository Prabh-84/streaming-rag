"""Phase 1 acceptance test (PRD_TRD.md §10, Phase 1 row)."""

import os

from app.core.config import Settings, get_settings


def test_config_loads_with_defaults(monkeypatch):
    monkeypatch.delenv("STABILITY_THRESHOLD", raising=False)
    settings = Settings(_env_file=None)
    assert settings.stability_threshold == 0.90
    assert settings.max_wait_chunks == 6
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.max_total_evidence == 15


def test_config_reads_env_override(monkeypatch):
    monkeypatch.setenv("STABILITY_THRESHOLD", "0.75")
    monkeypatch.setenv("API_KEY", "test_key_123")
    settings = Settings(_env_file=None)
    assert settings.stability_threshold == 0.75
    assert settings.api_key == "test_key_123"


def test_get_settings_is_cached():
    a = get_settings()
    b = get_settings()
    assert a is b


def test_env_example_keys_match_settings_fields():
    """Every key in .env.example must correspond to a Settings field alias, and vice versa,
    so no tunable can silently drift out of sync (PRD_TRD.md §9.8 design principle)."""
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env_example_path = os.path.join(here, ".env.example")
    with open(env_example_path, encoding="utf-8") as f:
        env_keys = {
            line.split("=", 1)[0].strip() for line in f if line.strip() and not line.startswith("#")
        }

    settings_aliases = {
        field.alias or name.upper() for name, field in Settings.model_fields.items()
    }
    assert env_keys == settings_aliases
