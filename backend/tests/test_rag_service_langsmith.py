"""Tests for RAGService LangSmith RunnableConfig."""

from unittest.mock import patch

from app.services.rag_service import RAGService


@patch("app.services.rag_service.is_langsmith_enabled", return_value=True)
@patch("app.services.rag_service.settings")
def test_agent_config_includes_user_and_admin(mock_settings, _mock_ls):
    mock_settings.app_env = "dev"
    service = RAGService()
    config = service._agent_config(
        "session-1",
        "user-42",
        run_name="rag_chat",
        is_admin=True,
    )
    assert config["metadata"]["user_id"] == "user-42"
    assert config["metadata"]["is_admin"] is True
    assert config["metadata"]["env"] == "dev"
    assert "env:dev" in config["tags"]
    assert "admin" in config["tags"]


@patch("app.services.rag_service.is_langsmith_enabled", return_value=False)
def test_agent_config_minimal_when_tracing_off(_mock_ls):
    service = RAGService()
    config = service._agent_config("session-1", "user-42", is_admin=False)
    assert config == {"configurable": {"thread_id": "session-1"}}
