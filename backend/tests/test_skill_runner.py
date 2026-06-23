import pytest

from app.config import settings
from app.skills.loader import reload_skills
from app.skills.paths import resolve_script_path
from app.skills.runner import SkillScriptRunner, _truncate_output


@pytest.fixture(autouse=True)
def _load_skills():
    reload_skills()


def test_run_echo_config_script():
    script = resolve_script_path("example-rag-assistant", "scripts/echo_config.py")
    result = SkillScriptRunner().run(script, [])
    assert result.exit_code == 0
    assert "retrieval_k=4" in result.stdout
    assert not result.truncated


def test_run_with_args_json():
    script = resolve_script_path("example-rag-assistant", "scripts/echo_config.py")
    result = SkillScriptRunner().run(script, ["--json"])
    assert result.exit_code == 0
    assert '"retrieval_k": 4' in result.stdout


def test_truncate_output():
    text = "a" * 1000
    truncated, was_truncated = _truncate_output(text, 100)
    assert was_truncated
    assert len(truncated.encode("utf-8")) <= 100


def test_run_invalid_timeout():
    script = resolve_script_path("example-rag-assistant", "scripts/echo_config.py")
    runner = SkillScriptRunner()
    with pytest.raises(ValueError, match="timeout"):
        runner.run(script, [], timeout=settings.skill_script_max_timeout_seconds + 1)
