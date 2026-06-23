import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from app.config import settings

TRUNCATION_NOTICE = (
    "\n[Output was truncated due to size limits. "
    "Consider reducing script output or increasing skill_script_max_output_bytes.]"
)


@dataclass(frozen=True)
class ExecuteResult:
    exit_code: int
    stdout: str
    stderr: str
    truncated: bool

    @property
    def combined_output(self) -> str:
        parts: list[str] = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            if parts:
                parts.append("\n--- stderr ---\n")
            parts.append(self.stderr)
        output = "".join(parts)
        if self.truncated:
            output += TRUNCATION_NOTICE
        return output


def _minimal_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for key in ("PATH", "LANG", "LC_ALL", "HOME", "TMPDIR", "TEMP", "TMP"):
        value = os.environ.get(key)
        if value:
            env[key] = value
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _build_argv(script: Path) -> list[str]:
    suffix = script.suffix.lower()
    if suffix == ".py":
        return [sys.executable, str(script)]
    if suffix == ".sh":
        return ["/bin/bash", str(script)]
    raise ValueError(f"不支持的脚本类型: {suffix}")


def _truncate_output(text: str, max_bytes: int) -> tuple[str, bool]:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text, False
    truncated = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return truncated, True


class SkillScriptRunner:
    def run(self, script: Path, args: list[str], *, timeout: int | None = None) -> ExecuteResult:
        effective_timeout = timeout if timeout is not None else settings.skill_script_timeout_seconds
        max_timeout = settings.skill_script_max_timeout_seconds
        if effective_timeout <= 0 or effective_timeout > max_timeout:
            raise ValueError(f"timeout 必须在 1..{max_timeout} 秒之间")

        argv = _build_argv(script) + list(args)
        try:
            completed = subprocess.run(
                argv,
                cwd=str(script.parent),
                env=_minimal_env(),
                shell=False,
                capture_output=True,
                timeout=effective_timeout,
                text=True,
                errors="replace",
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            max_bytes = settings.skill_script_max_output_bytes
            stdout, trunc_out = _truncate_output(stdout, max_bytes)
            stderr, trunc_err = _truncate_output(stderr, max_bytes // 2 if stderr else max_bytes)
            return ExecuteResult(
                exit_code=-1,
                stdout=stdout,
                stderr=f"命令超时 ({effective_timeout}s)\n{stderr}".strip(),
                truncated=trunc_out or trunc_err,
            )

        max_bytes = settings.skill_script_max_output_bytes
        stdout, trunc_out = _truncate_output(completed.stdout or "", max_bytes)
        stderr, trunc_err = _truncate_output(completed.stderr or "", max_bytes // 2 if completed.stderr else max_bytes)

        return ExecuteResult(
            exit_code=completed.returncode,
            stdout=stdout,
            stderr=stderr,
            truncated=trunc_out or trunc_err,
        )
