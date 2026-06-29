import asyncio
from typing import Any

from elasticsearch import AsyncElasticsearch
from sqlalchemy import text

from app.config import settings
from app.database import engine

_PROBE_TIMEOUT_SEC = 2.0


async def check_postgres() -> tuple[bool, str | None]:
    try:
        async with asyncio.timeout(_PROBE_TIMEOUT_SEC):
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        return True, None
    except Exception as exc:
        return False, str(exc)


async def check_elasticsearch() -> tuple[bool, str | None]:
    client = AsyncElasticsearch(settings.es_url, request_timeout=_PROBE_TIMEOUT_SEC)
    try:
        async with asyncio.timeout(_PROBE_TIMEOUT_SEC):
            healthy = await client.ping()
        if healthy:
            return True, None
        return False, "Elasticsearch ping returned false"
    except Exception as exc:
        return False, str(exc)
    finally:
        await client.close()


async def build_health_payload(
    *,
    extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int]:
    postgres_healthy, postgres_error = await check_postgres()
    elasticsearch_healthy, elasticsearch_error = await check_elasticsearch()

    core_healthy = postgres_healthy and elasticsearch_healthy
    body: dict[str, Any] = {
        "status": "ok" if core_healthy else "unhealthy",
        "postgres_healthy": postgres_healthy,
        "postgres_error": postgres_error,
        "elasticsearch_healthy": elasticsearch_healthy,
        "elasticsearch_error": elasticsearch_error,
    }
    if extra:
        body.update(extra)

    status_code = 200 if core_healthy else 503
    return body, status_code
