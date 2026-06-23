from app.config import settings
from app.skills.loader import list_skill_files as _list_skill_files
from app.skills.paths import SkillPathError, resolve_read_path, resolve_script_path
from app.skills.runner import SkillScriptRunner

_runner = SkillScriptRunner()

if settings.skill_script_enabled:
    from langchain_core.tools import tool

    @tool
    def run_skill_script(
        skill_name: str,
        script_path: str,
        script_args: list[str] | None = None,
    ) -> str:
        """在指定 skill 的 scripts/ 目录下执行脚本（Python 或 Bash）。

        仅当 skill 指令要求运行 scripts/ 下脚本时使用。script_args 为传递给脚本的字符串列表，不使用 shell。
        """
        try:
            script = resolve_script_path(skill_name, script_path)
            result = _runner.run(script, script_args or [])
        except SkillPathError as exc:
            return f"Error: {exc}"
        except ValueError as exc:
            return f"Error: {exc}"
        except OSError as exc:
            return f"Error: 执行失败: {exc}"

        output = result.combined_output
        header = f"exit_code={result.exit_code}\n"
        if output:
            return header + output
        return header + "(无输出)"

    @tool
    def read_skill_file(
        skill_name: str,
        file_path: str,
        offset: int = 0,
        limit: int = 200,
    ) -> str:
        """分页读取 skill 目录内的文件（scripts/、references/、assets/ 或 SKILL.md）。

        offset 为起始行号（0-indexed），limit 为最多返回行数。
        """
        if offset < 0:
            return "Error: offset 不能为负数"
        if limit < 1 or limit > 2000:
            return "Error: limit 必须在 1..2000 之间"

        try:
            path = resolve_read_path(skill_name, file_path)
            text = path.read_text(encoding="utf-8", errors="replace")
        except SkillPathError as exc:
            return f"Error: {exc}"
        except OSError as exc:
            return f"Error: 读取失败: {exc}"

        lines = text.splitlines()
        total = len(lines)
        selected = lines[offset : offset + limit]
        body = "\n".join(selected)
        header = f"# {file_path} (lines {offset}-{min(offset + limit, total)} of {total})\n\n"
        if not selected and offset >= total:
            return f"Error: offset {offset} 超出文件行数 ({total})"
        return header + body

    @tool
    def list_skill_files(skill_name: str) -> str:
        """列出指定 skill 目录下的资源文件（scripts/、references/、assets/）。"""
        from app.skills.loader import get_loaded_skills

        names = {s.name for s in get_loaded_skills()}
        if skill_name not in names:
            available = ", ".join(sorted(names)) or "无"
            return f"Error: 未找到 skill '{skill_name}'。可用: {available}"

        files = _list_skill_files(skill_name)
        if not files:
            return f"skill '{skill_name}' 下暂无资源文件。"
        return "\n".join(files)

    TOOLS = [run_skill_script, read_skill_file, list_skill_files]
else:
    TOOLS: list = []
