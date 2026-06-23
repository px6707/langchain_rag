import pytest

from app.skills.loader import reload_skills


@pytest.fixture(autouse=True)
def _load_skills():
    reload_skills()


def test_list_skill_files_tool():
    from app.tools import skill_scripts

    if not skill_scripts.TOOLS:
        pytest.skip("skill_script_enabled=false")

    result = skill_scripts.list_skill_files.invoke({"skill_name": "example-rag-assistant"})
    assert "scripts/echo_config.py" in result


def test_read_skill_file_tool():
    from app.tools import skill_scripts

    if not skill_scripts.TOOLS:
        pytest.skip("skill_script_enabled=false")

    result = skill_scripts.read_skill_file.invoke(
        {"skill_name": "example-rag-assistant", "file_path": "SKILL.md", "offset": 0, "limit": 5}
    )
    assert "SKILL.md" in result
    assert not result.startswith("Error:")


def test_run_skill_script_tool():
    from app.tools import skill_scripts

    if not skill_scripts.TOOLS:
        pytest.skip("skill_script_enabled=false")

    result = skill_scripts.run_skill_script.invoke(
        {
            "skill_name": "example-rag-assistant",
            "script_path": "scripts/echo_config.py",
            "script_args": [],
        }
    )
    assert result.startswith("exit_code=0")
    assert "retrieval_k=4" in result


def test_run_skill_script_rejects_bad_path():
    from app.tools import skill_scripts

    if not skill_scripts.TOOLS:
        pytest.skip("skill_script_enabled=false")

    result = skill_scripts.run_skill_script.invoke(
        {
            "skill_name": "example-rag-assistant",
            "script_path": "../SKILL.md",
            "script_args": [],
        }
    )
    assert result.startswith("Error:")
