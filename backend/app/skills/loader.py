import re
from pathlib import Path

from app.config import settings
from app.skills.types import SkillMeta

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
SKILL_FILE_NAME = "SKILL.md"
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)
FIELD_RE = re.compile(r"^(\w+):\s*(.+)$", re.MULTILINE)

_loaded_skills: dict[str, SkillMeta] = {}
_skill_bodies: dict[str, str] = {}


def _resolve_skills_dir() -> Path:
    configured = Path(settings.skills_dir)
    if configured.is_absolute():
        return configured
    return (BACKEND_ROOT / configured).resolve()


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    match = FRONTMATTER_RE.match(text.strip())
    if not match:
        return {}, text.strip()

    frontmatter_block, body = match.group(1), match.group(2).strip()
    fields: dict[str, str] = {}
    for key, value in FIELD_RE.findall(frontmatter_block):
        fields[key] = value.strip().strip('"').strip("'")
    return fields, body


def _parse_skill_file(path: Path) -> tuple[SkillMeta, str] | None:
    text = path.read_text(encoding="utf-8")
    fields, body = _parse_frontmatter(text)
    name = fields.get("name") or path.parent.name
    description = fields.get("description", "")
    if not description:
        return None
    meta = SkillMeta(name=name, description=description, path=path.parent)
    content = f"# {name}\n\n{body}" if body else f"# {name}"
    return meta, content


def _allowed_skill_names() -> set[str] | None:
    if not settings.skills_allowlist.strip():
        return None
    return {name.strip() for name in settings.skills_allowlist.split(",") if name.strip()}


def load_all_skills() -> list[SkillMeta]:
    global _loaded_skills, _skill_bodies
    skills_dir = _resolve_skills_dir()
    allowlist = _allowed_skill_names()
    loaded: dict[str, SkillMeta] = {}
    bodies: dict[str, str] = {}

    if skills_dir.is_dir():
        for skill_path in sorted(skills_dir.glob(f"*/{SKILL_FILE_NAME}")):
            parsed = _parse_skill_file(skill_path)
            if parsed is None:
                continue
            meta, content = parsed
            if allowlist is not None and meta.name not in allowlist:
                continue
            loaded[meta.name] = meta
            bodies[meta.name] = content

    _loaded_skills = loaded
    _skill_bodies = bodies
    return list(loaded.values())


def get_loaded_skills() -> list[SkillMeta]:
    if not _loaded_skills:
        return load_all_skills()
    return list(_loaded_skills.values())


def get_skill_content(name: str) -> str | None:
    if not _skill_bodies:
        load_all_skills()
    return _skill_bodies.get(name)


def list_skill_files(skill_name: str) -> list[str]:
    if not _loaded_skills:
        load_all_skills()
    meta = _loaded_skills.get(skill_name)
    if meta is None:
        return []

    files: list[str] = []
    for subdir in ("scripts", "references", "assets"):
        resource_dir = meta.path / subdir
        if not resource_dir.is_dir():
            continue
        for path in sorted(resource_dir.rglob("*")):
            if path.is_file():
                files.append(str(path.relative_to(meta.path)))
    return files


def reload_skills() -> list[SkillMeta]:
    _loaded_skills.clear()
    _skill_bodies.clear()
    return load_all_skills()
