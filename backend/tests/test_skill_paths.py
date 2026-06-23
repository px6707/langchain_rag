import pytest

from app.skills.loader import load_all_skills, reload_skills
from app.skills.paths import SkillPathError, resolve_read_path, resolve_script_path


@pytest.fixture(autouse=True)
def _load_skills():
    reload_skills()


def test_resolve_script_path_success():
    path = resolve_script_path("example-rag-assistant", "scripts/echo_config.py")
    assert path.name == "echo_config.py"
    assert path.is_file()


def test_resolve_script_rejects_traversal():
    with pytest.raises(SkillPathError, match="路径穿越"):
        resolve_script_path("example-rag-assistant", "scripts/../SKILL.md")


def test_resolve_script_rejects_non_scripts_prefix():
    with pytest.raises(SkillPathError, match="前缀"):
        resolve_script_path("example-rag-assistant", "references/foo.md")


def test_resolve_script_rejects_unknown_skill():
    with pytest.raises(SkillPathError, match="未找到 skill"):
        resolve_script_path("nonexistent-skill", "scripts/foo.py")


def test_resolve_read_path_skill_md():
    path = resolve_read_path("example-rag-assistant", "SKILL.md")
    assert path.name == "SKILL.md"


def test_resolve_read_path_references(tmp_path, monkeypatch):
    skill_dir = tmp_path / "test-skill"
    scripts = skill_dir / "scripts"
    refs = skill_dir / "references"
    scripts.mkdir(parents=True)
    refs.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: test-skill\ndescription: test skill for paths\n---\n\n# test\n",
        encoding="utf-8",
    )
    (refs / "note.md").write_text("# note", encoding="utf-8")

    monkeypatch.setattr("app.config.settings.skills_dir", str(tmp_path))
    monkeypatch.setattr("app.skills.loader.settings.skills_dir", str(tmp_path))
    reload_skills()

    loaded = [s.name for s in __import__("app.skills.loader", fromlist=["get_loaded_skills"]).get_loaded_skills()]
    assert "test-skill" in loaded

    path = resolve_read_path("test-skill", "references/note.md")
    assert path.read_text(encoding="utf-8") == "# note"
