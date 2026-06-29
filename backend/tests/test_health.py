import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from app.health import build_health_payload, check_elasticsearch, check_postgres


@asynccontextmanager
async def _mock_connect():
    conn = AsyncMock()
    conn.execute = AsyncMock()
    yield conn


def test_check_postgres_healthy():
    mock_engine = MagicMock()
    mock_engine.connect.return_value = _mock_connect()
    with patch("app.health.engine", mock_engine):
        healthy, error = asyncio.run(check_postgres())
    assert healthy is True
    assert error is None


def test_check_postgres_unhealthy():
    mock_engine = MagicMock()
    mock_engine.connect.side_effect = ConnectionError("db down")
    with patch("app.health.engine", mock_engine):
        healthy, error = asyncio.run(check_postgres())
    assert healthy is False
    assert error == "db down"


def test_check_elasticsearch_healthy():
    mock_client = AsyncMock()
    mock_client.ping = AsyncMock(return_value=True)
    mock_client.close = AsyncMock()

    with patch("app.health.AsyncElasticsearch", return_value=mock_client):
        healthy, error = asyncio.run(check_elasticsearch())

    assert healthy is True
    assert error is None
    mock_client.close.assert_awaited_once()


def test_check_elasticsearch_ping_false():
    mock_client = AsyncMock()
    mock_client.ping = AsyncMock(return_value=False)
    mock_client.close = AsyncMock()

    with patch("app.health.AsyncElasticsearch", return_value=mock_client):
        healthy, error = asyncio.run(check_elasticsearch())

    assert healthy is False
    assert error == "Elasticsearch ping returned false"
    mock_client.close.assert_awaited_once()


def test_check_elasticsearch_exception():
    mock_client = AsyncMock()
    mock_client.ping = AsyncMock(side_effect=ConnectionError("es down"))
    mock_client.close = AsyncMock()

    with patch("app.health.AsyncElasticsearch", return_value=mock_client):
        healthy, error = asyncio.run(check_elasticsearch())

    assert healthy is False
    assert error == "es down"
    mock_client.close.assert_awaited_once()


def test_build_health_payload_ok():
    with (
        patch("app.health.check_postgres", return_value=(True, None)),
        patch("app.health.check_elasticsearch", return_value=(True, None)),
    ):
        body, status_code = asyncio.run(
            build_health_payload(extra={"langsmith_tracing": False})
        )

    assert status_code == 200
    assert body["status"] == "ok"
    assert body["postgres_healthy"] is True
    assert body["elasticsearch_healthy"] is True
    assert body["langsmith_tracing"] is False


def test_build_health_payload_postgres_unhealthy():
    with (
        patch("app.health.check_postgres", return_value=(False, "db down")),
        patch("app.health.check_elasticsearch", return_value=(True, None)),
    ):
        body, status_code = asyncio.run(build_health_payload())

    assert status_code == 503
    assert body["status"] == "unhealthy"
    assert body["postgres_healthy"] is False
    assert body["postgres_error"] == "db down"
    assert body["elasticsearch_healthy"] is True


def test_build_health_payload_elasticsearch_unhealthy():
    with (
        patch("app.health.check_postgres", return_value=(True, None)),
        patch("app.health.check_elasticsearch", return_value=(False, "es down")),
    ):
        body, status_code = asyncio.run(build_health_payload())

    assert status_code == 503
    assert body["status"] == "unhealthy"
    assert body["postgres_healthy"] is True
    assert body["elasticsearch_healthy"] is False
    assert body["elasticsearch_error"] == "es down"
