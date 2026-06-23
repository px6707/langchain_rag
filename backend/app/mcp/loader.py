import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, cast

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import Connection

from app.mcp.auth import resolve_remote_auth
from app.config import settings

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent

_client: MultiServerMCPClient | None = None
_mcp_tools: list[BaseTool] = []
_server_names: list[str] = []


def _resolve_config_path() -> Path:
    configured = Path(settings.mcp_servers_file)
    if configured.is_absolute():
        return configured
    return (BACKEND_ROOT / configured).resolve()


def _resolve_stdio_paths(server_config: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(server_config)
    args = resolved.get("args")
    if not isinstance(args, list):
        return resolved

    new_args: list[Any] = []
    for arg in args:
        if isinstance(arg, str) and (arg.startswith("./") or arg.endswith(".py")):
            path = Path(arg)
            if not path.is_absolute():
                path = (BACKEND_ROOT / path).resolve()
            new_args.append(str(path))
        else:
            new_args.append(arg)
    resolved["args"] = new_args

    command = resolved.get("command")
    if command in {"python", "python3"}:
        resolved["command"] = sys.executable

    return resolved


def _as_connection(config: dict[str, Any]) -> Connection:
    # JSON config is validated at runtime by langchain_mcp_adapters when connecting.
    return cast(Connection, cast(object, config))


def _load_server_connections() -> dict[str, Connection]:
    config_path = _resolve_config_path()
    if not config_path.is_file():
        logger.info("MCP config not found at %s, skipping MCP tools", config_path)
        return {}

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        return {}

    connections: dict[str, Connection] = {}
    for name, server_config in raw.items():
        if not isinstance(server_config, dict):
            logger.warning("Skipping MCP server %s: config must be an object", name)
            continue
        transport = server_config.get("transport", "stdio")
        if transport == "stdio":
            connections[name] = _as_connection(_resolve_stdio_paths(server_config))
        else:
            resolved = resolve_remote_auth(name, dict(server_config))
            if resolved is None:
                continue
            connections[name] = _as_connection(resolved)
    return connections


def _allowed_tool_names() -> set[str] | None:
    if not settings.mcp_tool_allowlist.strip():
        return None
    return {name.strip() for name in settings.mcp_tool_allowlist.split(",") if name.strip()}


def _filter_tools(tools: list[BaseTool]) -> list[BaseTool]:
    allowlist = _allowed_tool_names()
    if allowlist is None:
        return tools
    return [tool for tool in tools if tool.name in allowlist]


async def _load_tools_from_client(client: MultiServerMCPClient) -> list[BaseTool]:
    tools: list[BaseTool] = []
    for server_name in client.connections:
        try:
            server_tools = await client.get_tools(server_name=server_name)
            tools.extend(server_tools)
            logger.info("Loaded %d MCP tools from server '%s'", len(server_tools), server_name)
        except Exception:
            logger.warning("Failed to load MCP tools from server '%s'", server_name, exc_info=True)
    return _filter_tools(tools)


async def init_mcp() -> list[BaseTool]:
    global _client, _mcp_tools, _server_names

    _client = None
    _mcp_tools = []
    _server_names = []

    if not settings.mcp_enabled:
        logger.info("MCP disabled via settings")
        return []

    connections = _load_server_connections()
    if not connections:
        return []

    client = MultiServerMCPClient(connections, tool_name_prefix=True)
    tools = await _load_tools_from_client(client)

    _client = client
    _mcp_tools = tools
    _server_names = list(connections.keys())
    return tools


async def close_mcp() -> None:
    global _client, _mcp_tools, _server_names
    _client = None
    _mcp_tools = []
    _server_names = []


async def reload_mcp() -> list[BaseTool]:
    await close_mcp()
    return await init_mcp()


def get_mcp_tools() -> list[BaseTool]:
    return list(_mcp_tools)


def get_mcp_status() -> dict[str, Any]:
    return {
        "mcp_enabled": settings.mcp_enabled,
        "mcp_servers_loaded": len(_server_names),
        "mcp_server_names": list(_server_names),
        "mcp_tools_count": len(_mcp_tools),
        "mcp_tool_names": [tool.name for tool in _mcp_tools],
    }


def init_mcp_sync() -> list[BaseTool]:
    return asyncio.run(init_mcp())


def reload_mcp_sync() -> list[BaseTool]:
    return asyncio.run(reload_mcp())
