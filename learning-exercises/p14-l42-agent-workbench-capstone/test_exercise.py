"""Тесты к уроку «Капстоун: переносимый пакет воркбенча». Правь exercise.py."""

import pytest

from exercise import (
    CI_WORKFLOW,
    FANOUT_TARGETS,
    LOCK_FILE,
    PACK_ROOT,
    REQUIRED_PACK_FILES,
    SHIP_STAGES,
    STATE_FILES,
    assemble_pack,
    classify_bump,
    classify_pack_candidates,
    fanout_targets,
    install_pack,
    lint_pack,
    ship_pack,
    uninstall_pack,
)


def _parts(drop=()):
    """Полный набор частей пакета; VERSION генерирует сборка, его тут нет."""
    return {
        rel: "содержимое " + rel
        for rel in REQUIRED_PACK_FILES
        if rel != "VERSION" and rel not in drop
    }


def _installed(version="1.0.0", repo=None):
    return install_pack(repo or {}, assemble_pack(_parts(), version), version)


# ------------------------------------------------- classify_pack_candidates
def test_project_tasks_are_excluded_with_a_reason():
    out = classify_pack_candidates([{"path": "tasks/T-1.md", "kind": "project_task"}])
    assert out["included"] == []
    assert out["excluded"][0]["path"] == "tasks/T-1.md"
    assert out["excluded"][0]["reason"] != ""


def test_schemas_scripts_and_docs_stay_in():
    out = classify_pack_candidates(
        [
            {"path": "schemas/agent_state.schema.json", "kind": "schema"},
            {"path": "scripts/verify_agent.py", "kind": "script"},
            {"path": "docs/agent-rules.md", "kind": "doc"},
        ]
    )
    assert len(out["included"]) == 3 and out["excluded"] == []


def test_vendor_sdk_and_onboarding_prose_are_both_out():
    out = classify_pack_candidates(
        [
            {"path": "sdk/openai_client.py", "kind": "vendor_sdk"},
            {"path": "welcome.md", "kind": "onboarding_prose"},
        ]
    )
    assert [e["path"] for e in out["excluded"]] == ["sdk/openai_client.py", "welcome.md"]


def test_classification_is_sorted_and_order_independent():
    a = {"path": "b.md", "kind": "doc"}
    b = {"path": "a.md", "kind": "doc"}
    assert classify_pack_candidates([a, b]) == classify_pack_candidates([b, a])
    assert classify_pack_candidates([a, b])["included"] == ["a.md", "b.md"]


# -------------------------------------------------------------- assemble_pack
def test_assembled_pack_carries_the_generated_version_file():
    pack = assemble_pack(_parts(), "1.2.3")
    assert pack[PACK_ROOT + "/VERSION"] == "1.2.3\n"


def test_assemble_refuses_a_pack_missing_a_required_script():
    with pytest.raises(ValueError) as err:
        assemble_pack(_parts(drop=("scripts/verify_agent.py",)), "1.0.0")
    assert "scripts/verify_agent.py" in str(err.value)


def test_assemble_roots_every_path_under_the_pack_directory():
    pack = assemble_pack(_parts(), "1.0.0")
    assert all(path.startswith(PACK_ROOT + "/") for path in pack)


def test_assemble_ignores_a_version_file_supplied_in_parts():
    """VERSION — свойство сборки, а не файл, который кто-то принёс с диска."""
    parts = dict(_parts(), VERSION="9.9.9\n")
    assert assemble_pack(parts, "1.0.0")[PACK_ROOT + "/VERSION"] == "1.0.0\n"


# ------------------------------------------------------------- classify_bump
def test_major_bump_requires_a_state_migration():
    assert classify_bump("1.2.3", "2.0.0") == {"kind": "major", "action": "migrate state"}


def test_minor_bump_only_requires_a_checker_run():
    assert classify_bump("1.2.3", "1.3.0")["action"] == "re-run checker"


def test_patch_and_same_require_nothing():
    assert classify_bump("1.2.3", "1.2.4")["kind"] == "patch"
    assert classify_bump("1.2.3", "1.2.3") == {"kind": "same", "action": "nothing"}
    assert classify_bump("1.2.3", "1.2.4")["action"] == "nothing"


def test_version_is_compared_numerically_not_as_a_string():
    """Как строки "1.10.0" < "1.9.0" — и откат прошёл бы за апгрейд."""
    assert classify_bump("1.9.0", "1.10.0")["kind"] == "minor"


def test_downgrade_is_refused():
    with pytest.raises(ValueError):
        classify_bump("2.0.0", "1.9.9")


def test_malformed_version_is_refused():
    with pytest.raises(ValueError):
        classify_bump("1.0", "1.0.1")


# ------------------------------------------------------------ fanout_targets
def test_every_link_points_at_the_single_source():
    links = fanout_targets(assemble_pack(_parts(), "1.0.0"))
    assert len(links) == len(FANOUT_TARGETS)
    assert {link["source"] for link in links} == {PACK_ROOT + "/AGENTS.md"}


def test_no_link_points_at_itself():
    links = fanout_targets(assemble_pack(_parts(), "1.0.0"))
    assert all(link["link"] != link["source"] for link in links)


def test_pack_without_agents_md_has_no_links():
    assert fanout_targets({}) == []


# --------------------------------------------------------------- install_pack
def test_install_writes_the_lock_file():
    assert _installed("1.4.0")[LOCK_FILE] == "1.4.0\n"


def test_install_refuses_to_overwrite_without_force():
    installed = _installed()
    with pytest.raises(ValueError):
        install_pack(installed, assemble_pack(_parts(), "1.0.0"), "1.0.0")


def test_forced_reinstall_is_idempotent():
    installed = _installed()
    again = install_pack(installed, assemble_pack(_parts(), "1.0.0"), "1.0.0", force=True)
    assert again == installed


def test_install_does_not_mutate_the_target_repo():
    repo = {"app.py": "print(1)"}
    install_pack(repo, assemble_pack(_parts(), "1.0.0"), "1.0.0")
    assert repo == {"app.py": "print(1)"}


def test_ci_is_wired_only_when_the_repo_already_has_workflows():
    pack = assemble_pack(_parts(), "1.0.0")
    assert CI_WORKFLOW not in install_pack({}, pack, "1.0.0")
    with_ci = install_pack({".github/workflows/ci.yml": "on: push"}, pack, "1.0.0")
    assert CI_WORKFLOW in with_ci


# ------------------------------------------------------------------ lint_pack
def test_freshly_installed_pack_lints_clean():
    assert lint_pack(_installed("1.0.0"), "1.0.0") == []


def test_stale_lock_is_reported():
    repo = dict(_installed("2.0.0"))
    repo[LOCK_FILE] = "1.0.0\n"
    assert any("1.0.0" in problem for problem in lint_pack(repo, "2.0.0"))


def test_missing_required_file_is_reported_by_name():
    repo = dict(_installed("1.0.0"))
    del repo[PACK_ROOT + "/schemas/task_board.schema.json"]
    assert "нет файла schemas/task_board.schema.json" in lint_pack(repo, "1.0.0")


def test_empty_repo_reports_every_missing_piece_including_the_lock():
    problems = lint_pack({}, "1.0.0")
    # каждый обязательный файл плюс несовпавший VERSION плюс отсутствующий замок
    assert len(problems) == len(REQUIRED_PACK_FILES) + 2
    assert any(LOCK_FILE in problem for problem in problems)


# -------------------------------------------------------------- uninstall_pack
def test_uninstall_keeps_user_state_and_outputs():
    repo = dict(_installed())
    repo["agent_state.json"] = "{}"
    repo["outputs/handoff/s1/handoff.json"] = "{}"
    left = uninstall_pack(repo)
    assert left == {"agent_state.json": "{}", "outputs/handoff/s1/handoff.json": "{}"}


def test_uninstall_removes_pack_files_lock_and_symlinks():
    left = uninstall_pack(_installed())
    assert left == {}


def test_uninstall_refuses_on_uncommitted_state():
    """Состояние принадлежит пользователю: пакет его не выносит молча."""
    with pytest.raises(ValueError):
        uninstall_pack(_installed(), dirty_state_files=[STATE_FILES[0]])


def test_keep_agents_md_leaves_the_router_in_place():
    left = uninstall_pack(_installed(), keep_agents_md=True)
    assert list(left) == [PACK_ROOT + "/AGENTS.md"]


def test_uninstall_does_not_mutate_the_repo():
    repo = _installed()
    before = len(repo)
    uninstall_pack(repo)
    assert len(repo) == before


# ------------------------------------------------------------------ ship_pack
def test_happy_path_installs_and_reports_every_stage_ok():
    result = ship_pack(_parts(), {".github/workflows/ci.yml": "on: push"}, "1.0.0")
    assert result["ok"] is True
    assert all(stage["ok"] for stage in result["stages"])
    assert result["repo"][LOCK_FILE] == "1.0.0\n"


def test_stage_order_follows_ship_stages():
    result = ship_pack(_parts(), {}, "1.0.0")
    assert [s["name"] for s in result["stages"]] == [name for name, _ in SHIP_STAGES]


def test_pipeline_fails_entirely_when_a_required_stage_fails():
    result = ship_pack(_parts(drop=("bin/install.sh",)), {}, "1.0.0")
    assert result["ok"] is False


def test_stages_after_a_required_failure_are_skipped():
    result = ship_pack(_parts(drop=("bin/install.sh",)), {}, "1.0.0")
    stages = {s["name"]: s for s in result["stages"]}
    assert stages["assemble"]["ok"] is False
    assert stages["install"]["detail"] == "skipped"
    assert stages["lint"]["detail"] == "skipped"


def test_failed_required_stage_leaves_the_repo_untouched():
    """Половина установленного пакета хуже, чем ни одного."""
    repo = {"app.py": "print(1)"}
    result = ship_pack(_parts(drop=("docs/reviewer-rubric.md",)), repo, "1.0.0")
    assert result["repo"] == repo


def test_optional_stage_failure_does_not_fail_the_pipeline():
    result = ship_pack(_parts(), {}, "1.0.0")
    stages = {s["name"]: s for s in result["stages"]}
    assert stages["ci_wiring"]["ok"] is False
    assert stages["ci_wiring"]["required"] is False
    assert result["ok"] is True
    assert result["repo"][LOCK_FILE] == "1.0.0\n"


def test_force_does_not_smuggle_a_pack_past_a_stale_lock():
    """Замок целевого репозитория проверяется ДО записи, force его не отменяет."""
    old = _installed("1.0.0")
    result = ship_pack(_parts(), old, "2.0.0", force=True)
    stages = {s["name"]: s for s in result["stages"]}
    assert stages["lint"]["ok"] is False
    assert result["ok"] is False
    assert result["repo"] == old
