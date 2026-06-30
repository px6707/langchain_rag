"""Tests for LangSmith / APP_ENV configuration defaults."""

import os

import pytest


def test_langsmith_project_defaults_to_rag_env(monkeypatch):
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING_ENABLED", raising=False)
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test_key")

    from importlib import reload

    import app.config as config_module

    reload(config_module)
    assert config_module.settings.app_env == "prod"
    assert config_module.settings.langsmith_project == "rag-prod"
    assert config_module.settings.langsmith_tracing_enabled is True


def test_langsmith_tracing_disabled_without_api_key(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING_ENABLED", raising=False)
    monkeypatch.setenv("APP_ENV", "dev")

    from importlib import reload

    import app.config as config_module

    reload(config_module)
    assert config_module.settings.langsmith_tracing_enabled is False


def test_langsmith_tracing_force_disabled(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test_key")
    monkeypatch.setenv("LANGSMITH_TRACING_ENABLED", "false")
    monkeypatch.setenv("APP_ENV", "dev")

    from importlib import reload

    import app.config as config_module

    reload(config_module)
    assert config_module.settings.langsmith_tracing_enabled is False
