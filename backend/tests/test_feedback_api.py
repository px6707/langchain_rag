"""Tests for LangSmith feedback API."""

from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.auth.dependencies import get_current_user
from app.main import app
from app.models import User

RUN_ID = "00000000-0000-0000-0000-000000000001"


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def authed_client():
    user = User(
        id=uuid4(),
        username="testuser",
        is_admin=False,
        is_active=True,
        password_hash="x",
    )

    async def override_user():
        return user

    app.dependency_overrides[get_current_user] = override_user
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_feedback_requires_auth(client):
    response = client.post(
        "/api/chat/feedback",
        json={"run_id": RUN_ID, "kind": "thumbs_down", "reason": "other"},
    )
    assert response.status_code == 401


@patch("app.routers.feedback.is_langsmith_enabled", return_value=False)
def test_feedback_503_when_tracing_disabled(mock_enabled, authed_client):
    response = authed_client.post(
        "/api/chat/feedback",
        json={"run_id": RUN_ID, "kind": "thumbs_down", "reason": "other"},
    )
    assert response.status_code == 503


@patch("app.routers.feedback.submit_user_feedback")
@patch("app.routers.feedback.is_langsmith_enabled", return_value=True)
def test_feedback_thumbs_up(mock_enabled, mock_submit, authed_client):
    response = authed_client.post(
        "/api/chat/feedback",
        json={
            "run_id": RUN_ID,
            "trace_id": RUN_ID,
            "kind": "thumbs_up",
            "session_id": "test-session",
        },
    )
    assert response.status_code == 200
    mock_submit.assert_called_once()
    kwargs = mock_submit.call_args.kwargs
    assert kwargs["key"] == "user_thumbs_up"
    assert kwargs["score"] == 1.0
    assert kwargs["extra"]["user_id"]


@patch("app.routers.feedback.submit_user_feedback")
@patch("app.routers.feedback.is_langsmith_enabled", return_value=True)
def test_feedback_thumbs_down_with_reason(mock_enabled, mock_submit, authed_client):
    response = authed_client.post(
        "/api/chat/feedback",
        json={
            "run_id": RUN_ID,
            "kind": "thumbs_down",
            "reason": "hallucination",
            "comment": "编造了合同条款",
        },
    )
    assert response.status_code == 200
    kwargs = mock_submit.call_args.kwargs
    assert kwargs["key"] == "user_thumbs_down"
    assert kwargs["score"] == 0.0
    assert kwargs["extra"]["reason"] == "hallucination"


def test_feedback_thumbs_down_missing_reason_422(authed_client):
    response = authed_client.post(
        "/api/chat/feedback",
        json={"run_id": RUN_ID, "kind": "thumbs_down"},
    )
    assert response.status_code == 422
