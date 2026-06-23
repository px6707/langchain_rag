import base64
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_REMOTE_TRANSPORTS = frozenset({"http", "streamable_http", "streamable-http", "sse"})
_UNSUPPORTED_AUTH_TRANSPORTS = frozenset({"stdio", "websocket"})


def _read_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _build_auth_headers(auth: dict[str, Any]) -> dict[str, str] | None:
    auth_type = auth.get("type")
    if not isinstance(auth_type, str):
        logger.error("MCP auth config missing or invalid 'type'")
        return None

    if auth_type == "bearer":
        token_env = auth.get("token_env")
        if not isinstance(token_env, str) or not token_env.strip():
            logger.error("MCP bearer auth requires non-empty 'token_env'")
            return None
        token = _read_env(token_env)
        if token is None:
            logger.error("MCP bearer auth env var %r is missing or empty", token_env)
            return None
        return {"Authorization": f"Bearer {token}"}

    if auth_type == "api_key":
        header = auth.get("header")
        token_env = auth.get("token_env")
        if not isinstance(header, str) or not header.strip():
            logger.error("MCP api_key auth requires non-empty 'header'")
            return None
        if not isinstance(token_env, str) or not token_env.strip():
            logger.error("MCP api_key auth requires non-empty 'token_env'")
            return None
        token = _read_env(token_env)
        if token is None:
            logger.error("MCP api_key auth env var %r is missing or empty", token_env)
            return None
        return {header: token}

    if auth_type == "basic":
        user_env = auth.get("user_env")
        pass_env = auth.get("pass_env")
        if not isinstance(user_env, str) or not user_env.strip():
            logger.error("MCP basic auth requires non-empty 'user_env'")
            return None
        if not isinstance(pass_env, str) or not pass_env.strip():
            logger.error("MCP basic auth requires non-empty 'pass_env'")
            return None
        username = _read_env(user_env)
        password = _read_env(pass_env)
        if username is None:
            logger.error("MCP basic auth env var %r is missing or empty", user_env)
            return None
        if password is None:
            logger.error("MCP basic auth env var %r is missing or empty", pass_env)
            return None
        encoded = base64.b64encode(f"{username}:{password}".encode()).decode()
        return {"Authorization": f"Basic {encoded}"}

    logger.error("MCP auth unsupported type: %r", auth_type)
    return None


def resolve_remote_auth(server_name: str, config: dict[str, Any]) -> dict[str, Any] | None:
    resolved = dict(config)
    auth = resolved.pop("auth", None)
    if auth is None:
        return resolved
    if not isinstance(auth, dict):
        logger.error("MCP server %r: 'auth' must be an object", server_name)
        return None

    transport = resolved.get("transport", "stdio")
    if transport in _UNSUPPORTED_AUTH_TRANSPORTS:
        logger.warning(
            "MCP server %r: 'auth' is ignored for transport %r",
            server_name,
            transport,
        )
        return resolved
    if transport not in _REMOTE_TRANSPORTS:
        logger.warning(
            "MCP server %r: 'auth' is ignored for unknown transport %r",
            server_name,
            transport,
        )
        return resolved

    auth_headers = _build_auth_headers(auth)
    if auth_headers is None:
        logger.error("MCP server %r: failed to resolve auth headers", server_name)
        return None

    existing_headers = resolved.get("headers")
    if existing_headers is None:
        resolved["headers"] = auth_headers
        return resolved

    if not isinstance(existing_headers, dict):
        logger.error("MCP server %r: 'headers' must be an object when using 'auth'", server_name)
        return None

    merged = dict(existing_headers)
    for key in auth_headers:
        if key in merged:
            logger.warning(
                "MCP server %r: auth header %r overrides existing value",
                server_name,
                key,
            )
    merged.update(auth_headers)
    resolved["headers"] = merged
    return resolved
