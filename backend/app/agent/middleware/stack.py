from typing import Any

from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
    PIIMiddleware,
    SummarizationMiddleware,
    TodoListMiddleware,
    ToolRetryMiddleware,
)
from langchain.agents.middleware.types import AgentMiddleware
from langchain_openai import ChatOpenAI

from app.agent.middleware.openviking import OpenVikingMemoryMiddleware
from app.agent.middleware.retrieval import RetrievalMiddleware
from app.agent.middleware.skills import SkillsMiddleware
from app.config import settings

TODO_SYSTEM_PROMPT_ZH = """## write_todos 任务规划

你可用 `write_todos` 工具管理复杂、多步骤任务。请遵守以下规则：

**何时使用：**
- 任务需要 3 个及以上明确步骤
- 用户一次提出多个子任务
- 需要先规划再逐步执行、且计划可能随进展调整

**何时不要使用：**
- 简单问答或 1-2 步即可完成的请求（例如查时间、基于文档直接回答）
- 纯信息查询，无需拆解步骤

**如何使用：**
- 开始执行某步前，将其标为 `in_progress`
- 完成后立即标为 `completed`，不要批量延迟更新
- 除非全部完成，否则至少保留一项 `in_progress`
- 每轮模型调用最多调用一次 `write_todos`（不要并行多次调用）

**完成时：**
- `write_todos` 只记录进度，不是给用户的最终答案
- 最后一轮 `write_todos` 之后，必须在后续消息中用中文给出用户要的实质结果"""


HITL_TOOL_CONFIGS: dict[str, InterruptOnConfig] = {
    "get_current_time": InterruptOnConfig(allowed_decisions=["approve", "reject"]),
    "send_email": InterruptOnConfig(
        allowed_decisions=["approve", "edit", "reject"],
        description="请确认邮件内容与发件账号后再发送",
    ),
    "run_skill_script": InterruptOnConfig(
        allowed_decisions=["approve", "reject"],
        description="请确认 skill 脚本路径与参数后再执行",
    ),
}


def _build_hitl_interrupt_on() -> dict[str, bool | InterruptOnConfig]:
    interrupt_on: dict[str, bool | InterruptOnConfig] = {}
    for name in settings.hitl_tools.split(","):
        tool_name = name.strip()
        if not tool_name:
            continue
        interrupt_on[tool_name] = HITL_TOOL_CONFIGS.get(
            tool_name,
            InterruptOnConfig(allowed_decisions=["approve", "reject"]),
        )
    return interrupt_on


def build_middleware_stack(llm: ChatOpenAI) -> list[AgentMiddleware[Any, Any, Any]]:
    stack: list[AgentMiddleware[Any, Any, Any]] = []

    if settings.pii_enabled:
        for pii_type in settings.pii_types.split(","):
            name = pii_type.strip()
            if not name:
                continue
            stack.append(
                PIIMiddleware(
                    name,  # type: ignore[arg-type]
                    strategy=settings.pii_strategy,  # type: ignore[arg-type]
                    apply_to_input=True,
                    apply_to_output=True,
                )
            )

    stack.append(
        SummarizationMiddleware(
            llm,
            trigger=("messages", settings.summarization_trigger_messages),
            keep=("messages", settings.summarization_keep_messages),
        )
    )

    if settings.skills_enabled:
        stack.append(SkillsMiddleware())

    if settings.openviking_enabled:
        stack.append(OpenVikingMemoryMiddleware())

    stack.append(RetrievalMiddleware())

    if settings.todo_list_enabled:
        stack.append(TodoListMiddleware(system_prompt=TODO_SYSTEM_PROMPT_ZH))

    if settings.hitl_enabled:
        interrupt_on = _build_hitl_interrupt_on()
        if interrupt_on:
            stack.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    stack.append(
        ToolRetryMiddleware(
            max_retries=settings.tool_retry_max_retries,
            initial_delay=settings.tool_retry_initial_delay,
            on_failure="continue",
        )
    )

    return stack
