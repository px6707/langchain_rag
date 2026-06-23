import os
from pathlib import Path

from app.config import settings
from app.skills.loader import get_loaded_skills

READ_PREFIXES = ("scripts", "references", "assets")
EXECUTE_PREFIXES = ("scripts",)
READ_EXTRA_FILES = frozenset({"SKILL.md"})


class SkillPathError(ValueError):
    pass


def _get_skill_root(skill_name: str) -> Path:
    for skill in get_loaded_skills():
        if skill.name == skill_name:
            return skill.path.resolve()
    raise SkillPathError(f"未找到 skill '{skill_name}'")


def _normalize_relative_path(relative_path: str) -> str:
    raw = relative_path.strip().replace("\\", "/")
    if not raw or "\0" in raw:
        raise SkillPathError("无效的文件路径")
    if raw.startswith("~") or os.path.isabs(raw):
        raise SkillPathError("不允许绝对路径或 ~ 路径")
    if ".." in raw.split("/"):
        raise SkillPathError("不允许路径穿越 (..)")
    return raw.lstrip("/")


def _check_symlink_safe(path: Path, skill_root: Path) -> None:
    current = path
    while True:
        if current.is_symlink():
            resolved = current.resolve()
            if not resolved.is_relative_to(skill_root):
                raise SkillPathError(f"符号链接指向 skill 目录外: {path}")
        if current == skill_root:
            break
        parent = current.parent
        if parent == current:
            break
        current = parent


def resolve_skill_file(
    skill_name: str,
    relative_path: str,
    *,
    allowed_prefixes: tuple[str, ...],
    allowed_extensions: frozenset[str] | None = None,
    allow_skill_md: bool = False,
) -> Path:
    skill_root = _get_skill_root(skill_name)
    normalized = _normalize_relative_path(relative_path)

    if allow_skill_md and normalized == "SKILL.md":
        target = (skill_root / "SKILL.md").resolve()
    else:
        if not any(
            normalized == prefix or normalized.startswith(f"{prefix}/")
            for prefix in allowed_prefixes
        ):
            allowed = ", ".join(allowed_prefixes)
            raise SkillPathError(f"路径必须在以下前缀下: {allowed}")

        target = (skill_root / normalized).resolve()

    if not target.is_relative_to(skill_root):
        raise SkillPathError("路径解析后超出 skill 目录")

    _check_symlink_safe(target, skill_root)

    if not target.exists():
        raise SkillPathError(f"文件不存在: {relative_path}")

    if not target.is_file():
        raise SkillPathError(f"不是普通文件: {relative_path}")

    if allowed_extensions is not None:
        suffix = target.suffix.lower()
        if suffix not in allowed_extensions:
            allowed = ", ".join(sorted(allowed_extensions))
            raise SkillPathError(f"不允许的文件扩展名，仅支持: {allowed}")

    max_bytes = settings.skill_script_max_file_bytes
    size = target.stat().st_size
    if size > max_bytes:
        raise SkillPathError(f"文件过大 ({size} bytes)，上限 {max_bytes} bytes")

    return target


def get_allowed_extensions() -> frozenset[str]:
    raw = settings.skill_script_allowed_extensions
    return frozenset(ext.strip().lower() for ext in raw.split(",") if ext.strip())


def resolve_script_path(skill_name: str, script_path: str) -> Path:
    return resolve_skill_file(
        skill_name,
        script_path,
        allowed_prefixes=EXECUTE_PREFIXES,
        allowed_extensions=get_allowed_extensions(),
    )


def resolve_read_path(skill_name: str, file_path: str) -> Path:
    normalized = _normalize_relative_path(file_path)
    if normalized == "SKILL.md":
        return resolve_skill_file(
            skill_name,
            file_path,
            allowed_prefixes=READ_PREFIXES,
            allow_skill_md=True,
        )
    return resolve_skill_file(
        skill_name,
        file_path,
        allowed_prefixes=READ_PREFIXES,
    )
