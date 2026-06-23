import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.middleware.openviking import (
    OpenVikingMemoryMiddleware,
    reset_ov_request_context,
    set_ov_request_context,
)
from app.openviking.memory_service import format_memory_context
from app.openviking.session_service import maybe_commit, sync_turn_to_openviking


def test_format_memory_context_groups_preferences():
    items = [
        (
            "memory",
            {
                "uri": "viking://user/alice/memories/preferences/coding_style.md",
                "abstract": "偏好 Python 类型注解",
                "score": 0.91,
            },
        ),
        (
            "memory",
            {
                "uri": "viking://user/alice/memories/entities/project-x.md",
                "abstract": "正在做 Project X",
                "score": 0.82,
            },
        ),
    ]
    text = format_memory_context(items)
    assert text is not None
    assert "用户偏好" in text
    assert "偏好 Python 类型注解" in text
    assert "相关实体" in text
    assert "Project X" in text


def test_maybe_commit_skips_below_threshold():
    mock_session = AsyncMock()
    mock_client = MagicMock()
    mock_client.service.sessions.get = AsyncMock(return_value=mock_session)

    with (
        patch("app.openviking.session_service.is_openviking_available", return_value=True),
        patch("app.openviking.session_service.get_local_client", return_value=mock_client),
        patch("app.openviking.session_service.settings.openviking_commit_every_messages", 20),
    ):
        asyncio.run(maybe_commit("user-1", "sess-1", 5))

    mock_client.service.sessions.get.assert_not_called()
    mock_session.commit_async.assert_not_called()


def test_maybe_commit_triggers_at_threshold():
    mock_session = AsyncMock()
    mock_client = MagicMock()
    mock_client.service.sessions.get = AsyncMock(return_value=mock_session)

    with (
        patch("app.openviking.session_service.is_openviking_available", return_value=True),
        patch("app.openviking.session_service.get_local_client", return_value=mock_client),
        patch("app.openviking.session_service.settings.openviking_commit_every_messages", 10),
    ):
        asyncio.run(maybe_commit("user-1", "sess-1", 10))

    mock_client.service.sessions.get.assert_awaited_once()
    mock_session.commit_async.assert_awaited_once()


def test_sync_turn_noop_when_disabled():
    append_mock = AsyncMock()
    with (
        patch("app.openviking.session_service.is_openviking_available", return_value=False),
        patch("app.openviking.session_service.append_message", append_mock),
    ):
        asyncio.run(
            sync_turn_to_openviking(
                "user-1",
                "sess-1",
                user_message="hello",
                assistant_message="hi",
            )
        )
    append_mock.assert_not_called()


def test_openviking_middleware_injects_memory_context():
    middleware = OpenVikingMemoryMiddleware()
    base_system = SystemMessage(content="base prompt")
    request = SimpleNamespace(
        messages=[HumanMessage(content="我喜欢简洁回答")],
        system_message=base_system,
    )
    request.override = lambda **kwargs: SimpleNamespace(
        messages=request.messages,
        system_message=kwargs.get("system_message", request.system_message),
    )

    captured: dict[str, SystemMessage | None] = {"system": None}

    async def handler(req):
        captured["system"] = req.system_message
        return SimpleNamespace()

    token = set_ov_request_context("user-abc", "sess-1")
    try:
        with patch(
            "app.agent.middleware.openviking.find_user_memories",
            new=AsyncMock(return_value="### 用户偏好\n- 喜欢简洁"),
        ):
            asyncio.run(middleware.awrap_model_call(request, handler))
    finally:
        reset_ov_request_context(token)

    assert captured["system"] is not None
    content = captured["system"].content
    assert isinstance(content, str)
    assert "用户长期记忆" in content
    assert "喜欢简洁" in content
