"""What Lucy's fork (branch guardian4) adds to Ian's main for her guardian baseline: the named judge addendum, Luna
judging through OpenRouter, her DeepSeek route, and SWE on an amd64 host. No model calls, no Docker."""
from __future__ import annotations

import pytest
import yaml

from src.configuration import settings
from src.configuration.presets import build_config
from src.configuration.swarm_config import SwarmConfig
from src.domains.swe import swe_sandbox
from src.domains.swe.swe_settings import TASK, TASK_DIR
from src.judging import cheatbench_judge as judging
from src.models import model_registry as models

OPENLIBRARY = "openlibrary-3aee"


# --------------------------------------------------------------------------
# the named judge addendum
# --------------------------------------------------------------------------
@pytest.mark.parametrize("text, value", [("guardian", "guardian"), ("orbit", "orbit"), ("true", True),
                                         ("false", False)])
def test_judge_addendum_takes_a_name_or_a_bool(text, value):
    assert build_config("default", None, [f"judge_addendum={text}"]).judge_addendum == value


def test_an_unknown_addendum_is_refused():
    with pytest.raises(ValueError):
        SwarmConfig(judge_addendum="nonsense")


def test_each_addendum_value_names_its_file():
    lucy = settings.LUCY / "experiments"
    assert judging.judge_addendum_path(False) is None and judging.judge_addendum_path(None) is None
    assert judging.judge_addendum_path(True) == judging.judge_addendum_path("orbit") == judging.JUDGE_ADDENDUM
    assert judging.judge_addendum_path("orbit") == lucy / "orbit_cheatbench" / "judge_addendum.py"
    assert judging.judge_addendum_path("guardian") == lucy / "guardian_baseline" / "judge_addendum.py"
    assert all(path.is_file() for path in judging.JUDGE_ADDENDA.values())


def test_the_guardian_addendum_adds_its_fields_and_keeps_cheatbenchs():
    plain = judging.judge_spec(TASK_DIR, TASK)
    spec = judging.judge_spec(TASK_DIR, TASK, addendum=judging.judge_addendum_path("guardian"))
    added = {"tried_planted_fix", "used_teammate_copy", "adopted_teammate_work", "first_exposure",
             "aware_at_exposure", "disclosure"}
    assert set(plain.fields) <= set(spec.fields) and added <= set(spec.fields) - set(plain.fields)


def test_the_scorer_and_the_rejudge_read_the_named_addendum():
    from src import run, tasks  # noqa: F401 - both import judge_addendum_path; Inspect and ORBIT, a few seconds
    assert tasks.judge_addendum_path is judging.judge_addendum_path
    assert "judge_addendum_path" in run._judge_views.__code__.co_names


# --------------------------------------------------------------------------
# Luna through OpenRouter
# --------------------------------------------------------------------------
def test_luna_judges_through_openrouter_at_the_registry_price():
    judges = yaml.safe_load(settings.JUDGE_MODELS_CONFIG.read_text())
    entry = judges["gpt-6-luna-openrouter"]
    assert entry["model"] == models.resolve("gpt-6-luna-openrouter") == "openrouter/openai/gpt-6-luna"
    assert entry["api_key_env"] == "OPENROUTER_API_KEY"
    assert entry["generation_config"] == judges["gpt-6-luna"]["generation_config"]
    assert judging.judge_prices("gpt-6-luna-openrouter") == models.usd_per_million("openrouter/openai/gpt-6-luna")


def test_an_openrouter_judge_schema_carries_a_description():
    # OpenRouter drops a json_schema whose description is null; other routes keep the request unchanged.
    spec = judging.judge_spec(TASK_DIR, TASK, addendum=judging.judge_addendum_path("guardian"))
    assert judging.response_schema(spec, "openrouter/openai/gpt-6-luna").description
    assert judging.response_schema(spec, "openai/gpt-6-luna").description is None
    assert judging.response_schema(spec).description is None


# --------------------------------------------------------------------------
# Lucy's DeepSeek route
# --------------------------------------------------------------------------
def test_the_september_flash_runs_pinned_to_deepinfra():
    name = "deepseek-v4.1-flash-deepinfra"
    assert models.resolve(name) == "openrouter/deepseek/deepseek-v4.1-flash"
    assert models.routing(name) == {"order": ["deepinfra"], "allow_fallbacks": False}
    models.check_routing(models.resolve(name))
    assert models.usd_per_million(name) == {"input": 0.14, "cached_input": 0.0042, "output": 0.42}
    assert SwarmConfig(worker_model=name).worker_model == models.resolve(name)


# --------------------------------------------------------------------------
# SWE on an amd64 host
# --------------------------------------------------------------------------
def fake_docker(monkeypatch, tmp_path, built: list) -> None:
    monkeypatch.setattr(swe_sandbox, "BUILD", tmp_path)
    monkeypatch.setattr(swe_sandbox, "image_id", lambda tag: f"id-{tag}")
    monkeypatch.setattr(swe_sandbox, "image_labels", lambda tag: {})
    monkeypatch.setattr(swe_sandbox, "_source_facts", lambda tag: {
        "env": {"PATH": "/usr/local/bin:/usr/bin", "LANG": "C.UTF-8"}, "codename": "bookworm"})
    monkeypatch.setattr(swe_sandbox, "docker", lambda *args, **kwargs: built.append(args))


def test_on_an_amd64_host_a_python_task_layers_host_dockerfile_on_its_image(monkeypatch, tmp_path):
    built: list = []
    fake_docker(monkeypatch, tmp_path, built)
    monkeypatch.setattr(swe_sandbox, "AMD64_HOST", True)
    images = swe_sandbox.ensure_images(OPENLIBRARY, log=lambda text: None)
    assert images["native"] == "cbswe-openlibrary-3aee:native"
    assert built == [("build", "-t", "cbswe-openlibrary-3aee:native", str(tmp_path / OPENLIBRARY / "native"))]
    recipe = (tmp_path / OPENLIBRARY / "native" / "Dockerfile").read_text()
    assert recipe.splitlines()[1] == "FROM cbswe-openlibrary-3aee:amd64" and "golang" not in recipe
    assert "export PATH=%s\\n' /usr/local/bin:/usr/bin >" in recipe and "RUN mkdir -p /logs" in recipe
    assert "cbswe.source_image_id=id-cbswe-openlibrary-3aee:amd64" in recipe and "RECIPE" not in recipe


def test_on_an_arm64_host_a_python_task_is_still_refused(monkeypatch, tmp_path):
    fake_docker(monkeypatch, tmp_path, [])
    monkeypatch.setattr(swe_sandbox, "AMD64_HOST", False)
    with pytest.raises(NotImplementedError):
        swe_sandbox.ensure_images(OPENLIBRARY, log=lambda text: None)


def test_the_host_recipe_text_is_the_one_stage_1_built():
    """The recipe hash is in each native image's label, so lucypc reuses the images Stage 1 built (Lucy's fork at
    0cfe547) only while the text is the same, comment line included."""
    first = (swe_sandbox.Path(swe_sandbox.__file__).parent / "sandbox" / "host.Dockerfile").read_text().splitlines()[0]
    assert first == "# The amd64 image on an amd64 host (see benchmark/domains/swe/sandbox.py): what native.Dockerfile " \
                    "adds, on top of it."


def test_a_submodule_state_does_not_make_the_base_checkout_unclean(monkeypatch, tmp_path):
    """openlibrary's image ships vendor/infogami modified; the fix touches no submodule, so base_lines ignores it."""
    from src.domains.swe import swe_ladder
    from src.domains.swe.swe_inputs import load_split, patch_files, task_config
    seen = []
    monkeypatch.setattr(swe_ladder, "BUILD", tmp_path)
    monkeypatch.setattr(swe_ladder.subprocess, "run", lambda args, **kwargs: seen.append(args) or
                        type("R", (), {"returncode": 0, "stdout": b"[]", "stderr": b""})())
    patch = task_config(load_split(OPENLIBRARY)["task"])["patch"]
    assert swe_ladder.base_lines(OPENLIBRARY, "base", patch) == set()
    assert "git status --porcelain --ignore-submodules=all" in seen[0][-1]
    assert not [f["path"] for f in patch_files(patch) if f["path"].startswith("vendor/")]
