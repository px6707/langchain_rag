import logging
import os
from pathlib import Path
from typing import Any

from openviking.client import LocalClient
from openviking.server.identity import RequestContext, Role
from openviking_cli.session.user_id import UserIdentifier

from app.config import settings

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent

_local_client: LocalClient | None = None
_init_error: str | None = None
_healthy: bool = False


def _resolve_path(path: str) -> str:
    configured = Path(path)
    if configured.is_absolute():
        return str(configured)
    return str((BACKEND_ROOT / configured).resolve())


def request_context(user_id: str) -> RequestContext:
    return RequestContext(
        user=UserIdentifier(settings.openviking_account_id, user_id),
        role=Role(Role.USER),
    )


def is_openviking_available() -> bool:
    return settings.openviking_enabled and _local_client is not None and _init_error is None


def get_local_client() -> LocalClient | None:
    return _local_client


async def init_openviking() -> None:
    global _local_client, _init_error, _healthy

    if not settings.openviking_enabled:
        logger.info("OpenViking disabled (openviking_enabled=false)")
        return

    if settings.openviking_config_file:
        config_path = Path(settings.openviking_config_file).expanduser()
        os.environ["OPENVIKING_CONFIG_FILE"] = str(config_path.resolve())

    try:
        _local_client = LocalClient(path=_resolve_path(settings.openviking_path))
        await _local_client.initialize()
        _healthy = _local_client.is_healthy()
        _init_error = None
        logger.info("OpenViking initialized at %s", _resolve_path(settings.openviking_path))
    except Exception as exc:
        _local_client = None
        _healthy = False
        _init_error = str(exc)
        logger.exception("OpenViking initialization failed: %s", exc)


async def close_openviking() -> None:
    global _local_client, _init_error, _healthy

    if _local_client is not None:
        try:
            await _local_client.close()
        except Exception:
            logger.exception("Error closing OpenViking client")
    _local_client = None
    _init_error = None
    _healthy = False


def get_openviking_status() -> dict[str, Any]:
    return {
        "openviking_enabled": settings.openviking_enabled,
        "openviking_available": is_openviking_available(),
        "openviking_healthy": _healthy if is_openviking_available() else False,
        "openviking_path": _resolve_path(settings.openviking_path),
        "openviking_error": _init_error,
    }
