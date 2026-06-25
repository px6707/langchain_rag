from contextlib import AsyncExitStack

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.config import settings

_checkpointer: BaseCheckpointSaver | None = None
_exit_stack: AsyncExitStack | None = None


async def init_checkpointer() -> BaseCheckpointSaver:
    global _checkpointer, _exit_stack
    if _checkpointer is not None:
        return _checkpointer

    _exit_stack = AsyncExitStack()
    saver = await _exit_stack.enter_async_context(
        AsyncPostgresSaver.from_conn_string(settings.checkpointer_database_url)
    )
    await saver.setup()
    _checkpointer = saver
    return _checkpointer


async def close_checkpointer() -> None:
    global _checkpointer, _exit_stack
    if _exit_stack is not None:
        await _exit_stack.aclose()
    _checkpointer = None
    _exit_stack = None


def get_checkpointer() -> BaseCheckpointSaver:
    if _checkpointer is None:
        raise RuntimeError("Checkpointer not initialized")
    return _checkpointer
