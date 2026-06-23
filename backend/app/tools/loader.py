import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path

from langchain_core.tools import BaseTool

from app.mcp.loader import get_mcp_tools

TOOLS_DIR = Path(__file__).parent
EXCLUDED_FILES = {"__init__.py", "loader.py"}

_loaded_tools: list[BaseTool] = []


def _load_module_from_path(path: Path):
    module_name = f"app.tools.{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _collect_tools_from_module(module) -> list[BaseTool]:
    tools: list[BaseTool] = []

    explicit = getattr(module, "TOOLS", None)
    if explicit:
        for item in explicit:
            if isinstance(item, BaseTool):
                tools.append(item)

    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        obj = getattr(module, attr_name)
        if isinstance(obj, BaseTool):
            tools.append(obj)
        elif callable(obj) and getattr(obj, "name", None) and hasattr(obj, "invoke"):
            try:
                if isinstance(obj, BaseTool):
                    tools.append(obj)
            except Exception:
                pass

    seen: set[str] = set()
    unique: list[BaseTool] = []
    for tool in tools:
        if tool.name not in seen:
            seen.add(tool.name)
            unique.append(tool)
    return unique


def _load_local_tools() -> list[BaseTool]:
    tools: list[BaseTool] = []

    for path in sorted(TOOLS_DIR.glob("*.py")):
        if path.name in EXCLUDED_FILES:
            continue
        module = _load_module_from_path(path)
        tools.extend(_collect_tools_from_module(module))

    return tools


def _merge_tools(local_tools: list[BaseTool], mcp_tools: list[BaseTool]) -> list[BaseTool]:
    merged: list[BaseTool] = []
    seen: set[str] = set()

    for tool in local_tools:
        if tool.name not in seen:
            seen.add(tool.name)
            merged.append(tool)

    for tool in mcp_tools:
        if tool.name not in seen:
            seen.add(tool.name)
            merged.append(tool)

    return merged


def load_all_tools() -> list[BaseTool]:
    global _loaded_tools
    _loaded_tools = _merge_tools(_load_local_tools(), get_mcp_tools())
    return _loaded_tools


def get_loaded_tools() -> Sequence[BaseTool]:
    if not _loaded_tools:
        return load_all_tools()
    return _loaded_tools


def reload_tools() -> list[BaseTool]:
    for path in TOOLS_DIR.glob("*.py"):
        if path.name in EXCLUDED_FILES:
            continue
        module_name = f"app.tools.{path.stem}"
        sys.modules.pop(module_name, None)
    return load_all_tools()
