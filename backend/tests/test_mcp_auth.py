import base64
import os

import pytest

from app.mcp.auth import resolve_remote_auth


def test_no_auth_passthrough():
    config = {"transport": "http", "url": "https://example.com/mcp"}
    resolved = resolve_remote_auth("weather", config)
    assert resolved == config
    assert "auth" not in resolved


def test_bearer_auth(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MCP_WEATHER_TOKEN", "secret-token")
    config = {
        "transport": "http",
        "url": "https://example.com/mcp",
        "auth": {"type": "bearer", "token_env": "MCP_WEATHER_TOKEN"},
    }
    resolved = resolve_remote_auth("weather", config)
    assert resolved is not None
    assert resolved["headers"] == {"Authorization": "Bearer secret-token"}
    assert "auth" not in resolved


def test_api_key_auth(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MCP_PARTNER_API_KEY", "key-123")
    config = {
        "transport": "http",
        "url": "https://partner.example.com/mcp",
        "auth": {
            "type": "api_key",
            "header": "X-API-Key",
            "token_env": "MCP_PARTNER_API_KEY",
        },
    }
    resolved = resolve_remote_auth("partner", config)
    assert resolved is not None
    assert resolved["headers"] == {"X-API-Key": "key-123"}


def test_basic_auth(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MCP_LEGACY_USER", "alice")
    monkeypatch.setenv("MCP_LEGACY_PASS", "secret")
    config = {
        "transport": "sse",
        "url": "https://legacy.example.com/sse",
        "auth": {
            "type": "basic",
            "user_env": "MCP_LEGACY_USER",
            "pass_env": "MCP_LEGACY_PASS",
        },
    }
    resolved = resolve_remote_auth("legacy", config)
    assert resolved is not None
    expected = base64.b64encode(b"alice:secret").decode()
    assert resolved["headers"] == {"Authorization": f"Basic {expected}"}


def test_missing_env_returns_none(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MCP_WEATHER_TOKEN", raising=False)
    config = {
        "transport": "http",
        "url": "https://example.com/mcp",
        "auth": {"type": "bearer", "token_env": "MCP_WEATHER_TOKEN"},
    }
    assert resolve_remote_auth("weather", config) is None


def test_merge_headers_auth_overrides(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MCP_WEATHER_TOKEN", "new-token")
    config = {
        "transport": "http",
        "url": "https://example.com/mcp",
        "headers": {"Authorization": "Bearer old", "X-Custom": "keep"},
        "auth": {"type": "bearer", "token_env": "MCP_WEATHER_TOKEN"},
    }
    resolved = resolve_remote_auth("weather", config)
    assert resolved is not None
    assert resolved["headers"] == {
        "Authorization": "Bearer new-token",
        "X-Custom": "keep",
    }


def test_stdio_ignores_auth():
    config = {
        "transport": "stdio",
        "command": "python",
        "args": ["./server.py"],
        "auth": {"type": "bearer", "token_env": "MCP_TOKEN"},
    }
    resolved = resolve_remote_auth("math", config)
    assert resolved is not None
    assert "auth" not in resolved
    assert "headers" not in resolved
