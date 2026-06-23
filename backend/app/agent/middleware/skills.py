from typing import Any

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)

from app.config import settings
from app.skills.loader import get_loaded_skills, get_skill_content

SKILLS_SYSTEM_PROMPT_ZH = """## Skills 使用说明

你可以通过 `load_skill` 工具按需加载领域技能指令。启动时你只能看到各 skill 的名称与描述；当用户任务匹配某个 skill 时，请先调用 `load_skill` 再执行。

规则：
- 仅在任务与 skill 描述相关时加载，不要预加载所有 skill
- 加载后遵循 skill 中的步骤与约束
- 若不确定该用哪个 skill，可先询问用户或选择最相关的一个"""

SKILL_SCRIPT_PROMPT_ZH = """
### Skill 资源与脚本（Level 3）

加载 skill 后，可按需使用以下工具访问 supporting files：
- `list_skill_files(skill_name)`：列出 scripts/、references/、assets/ 下的文件
- `read_skill_file(skill_name, file_path, offset=0, limit=200)`：分页读取文件内容
- `run_skill_script(skill_name, script_path, script_args=[])`：执行 scripts/ 下的 .py 或 .sh 脚本（需用户批准）

仅在 skill 指令明确要求运行脚本时调用 `run_skill_script`；先 `list_skill_files` 或 `read_skill_file` 确认路径。"""


def _build_skills_catalog() -> str:
    skills = get_loaded_skills()
    if not skills:
        return "当前未配置可用 skills。"

    lines = [f"- **{skill.name}**: {skill.description}" for skill in skills]
    return "\n".join(lines)


@tool
def load_skill(skill_name: str) -> str:
    """加载指定 skill 的完整指令内容。

    当用户任务与某个 skill 的描述匹配时使用。返回该 skill 的详细工作流与约束。
    """
    content = get_skill_content(skill_name)
    if content is None:
        available = ", ".join(skill.name for skill in get_loaded_skills()) or "无"
        return f"未找到 skill '{skill_name}'。可用 skills: {available}"
    return f"已加载 skill: {skill_name}\n\n{content}"


class SkillsMiddleware(AgentMiddleware[AgentState[Any], None, Any]):
    tools = [load_skill]

    async def awrap_model_call(
        self,
        request: ModelRequest[None],
        handler,
    ) -> ModelResponse:
        catalog = _build_skills_catalog()
        script_addendum = SKILL_SCRIPT_PROMPT_ZH if settings.skill_script_enabled else ""
        addendum = (
            f"\n\n{SKILLS_SYSTEM_PROMPT_ZH}{script_addendum}\n\n"
            f"### Available Skills\n\n{catalog}\n"
        )

        if request.system_message is not None:
            base = request.system_message.content
            if isinstance(base, list):
                base = str(base)
            new_system = SystemMessage(content=f"{base}{addendum}")
        else:
            new_system = SystemMessage(content=addendum.strip())

        request = request.override(system_message=new_system)
        return await handler(request)
