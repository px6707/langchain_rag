import logging

from openviking.message.part import TextPart

from app.config import settings
from app.openviking.client import get_local_client, is_openviking_available, request_context

logger = logging.getLogger(__name__)


async def ensure_session(user_id: str, session_id: str) -> None:
    if not is_openviking_available():
        return

    client = get_local_client()
    if client is None:
        return

    ctx = request_context(user_id)
    try:
        await client.service.initialize_user_directories(ctx)
        await client.service.sessions.get(session_id, ctx, auto_create=True)
    except Exception:
        logger.exception(
            "OpenViking ensure_session failed user_id=%s session_id=%s",
            user_id,
            session_id,
        )


async def append_message(user_id: str, session_id: str, role: str, content: str) -> int | None:
    if not is_openviking_available() or not content.strip():
        return None

    client = get_local_client()
    if client is None:
        return None

    ctx = request_context(user_id)
    try:
        await client.service.initialize_user_directories(ctx)
        session = await client.service.sessions.get(session_id, ctx, auto_create=True)
        session.add_message(role, [TextPart(text=content)])
        return len(session.messages)
    except Exception:
        logger.exception(
            "OpenViking append_message failed user_id=%s session_id=%s role=%s",
            user_id,
            session_id,
            role,
        )
        return None


async def maybe_commit(user_id: str, session_id: str, message_count: int | None) -> None:
    if not is_openviking_available() or message_count is None:
        return
    if message_count < settings.openviking_commit_every_messages:
        return

    client = get_local_client()
    if client is None:
        return

    ctx = request_context(user_id)
    try:
        session = await client.service.sessions.get(session_id, ctx, auto_create=False)
        await session.commit_async()
        logger.info(
            "OpenViking session committed user_id=%s session_id=%s messages=%s",
            user_id,
            session_id,
            message_count,
        )
    except Exception:
        logger.exception(
            "OpenViking commit failed user_id=%s session_id=%s",
            user_id,
            session_id,
        )


async def sync_turn_to_openviking(
    user_id: str,
    session_id: str,
    *,
    user_message: str | None = None,
    assistant_message: str | None = None,
) -> None:
    """Write one chat turn to OpenViking and commit when threshold is reached."""
    if not is_openviking_available():
        return

    message_count: int | None = None
    if user_message:
        message_count = await append_message(user_id, session_id, "user", user_message)
    if assistant_message:
        message_count = await append_message(user_id, session_id, "assistant", assistant_message)
    if message_count is not None:
        await maybe_commit(user_id, session_id, message_count)
