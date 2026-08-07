"""Тесты к уроку «Воркбенч на реальном репозитории». Правь exercise.py."""

import pytest

from exercise import (
    OUTCOME_KEYS,
    adapt_scope_contract,
    classify_touched_files,
    compare_pipelines,
    false_negative_reason,
    handoff_quality,
    measure_run,
    render_before_after,
    simulate_test_run,
)

SAMPLE_TREE = ["app.py", "test_app.py", "README.md", "scripts/release.sh"]

FULL_PACKET = {
    "next_action": "разобрать предупреждение про таймаут",
    "changed_files": ["app.py", "test_app.py"],
    "verdict_pointer": {"verification": "v.json", "review": "r.json"},
}


def _prompt_only_run():
    return {
        "touched": ["app.py", "README.md"],
        "repo_after": {"app.py": "if len(pw) < 8: raise Invalid"},
        "checks": [],
        "acceptance_test": "t_short_pw",
        "commands": ["cat README.md"],
        "handoff": None,
        "reviewer_scores": {"scope": 0, "tests": 0},
    }


def _workbench_run():
    return {
        "touched": ["app.py", "test_app.py"],
        "repo_after": {
            "app.py": "if len(pw) < 8: raise Invalid(422)",
            "test_app.py": "assert resp.status == 422",
        },
        "checks": [
            {"name": "t_short_pw", "file": "app.py", "requires": ["len(pw) < 8"]},
            {"name": "t_envelope", "file": "test_app.py", "requires": ["422"]},
        ],
        "acceptance_test": "t_short_pw",
        "commands": ["python3 -m pytest -q"],
        "handoff": FULL_PACKET,
        "reviewer_scores": {"scope": 2, "tests": 2},
    }


# ------------------------------------------------------ adapt_scope_contract
def test_contract_allows_the_directories_that_actually_hold_python():
    contract = adapt_scope_contract(["app.py", "src/core.py", "docs/guide.md"])
    assert contract["allowed_globs"] == ["*.py", "src/*.py"]


def test_forbidden_glob_with_no_match_in_the_tree_is_dropped():
    """Запрет на несуществующее ничего не защищает — только шумит в контракте."""
    contract = adapt_scope_contract(["app.py"])
    assert contract["forbidden_globs"] == []


def test_protected_globs_survive_even_without_a_match():
    contract = adapt_scope_contract(["app.py"], protected=["secrets/*"])
    assert contract["forbidden_globs"] == ["secrets/*"]


def test_acceptance_command_comes_from_repo_evidence():
    assert adapt_scope_contract(SAMPLE_TREE)["acceptance_command"] == "python3 -m pytest -q"
    js = adapt_scope_contract(["package.json", "index.js"])
    assert js["acceptance_command"] == "npm test"


def test_repo_without_test_evidence_has_no_acceptance_command():
    assert adapt_scope_contract(["app.py", "README.md"])["acceptance_command"] is None


# --------------------------------------------------- classify_touched_files
def test_forbidden_zone_beats_the_allowed_glob():
    """"*.py" по fnmatch накрывает и scripts/release.py — спасает только запрет."""
    contract = {"allowed_globs": ["*.py"], "forbidden_globs": ["scripts/*"]}
    out = classify_touched_files(["app.py", "scripts/release.py"], contract)
    assert out == {"in_scope": ["app.py"], "outside_scope": ["scripts/release.py"]}


def test_file_matching_nothing_is_outside_scope():
    contract = {"allowed_globs": ["*.py"], "forbidden_globs": []}
    assert classify_touched_files(["docs/x.md"], contract)["outside_scope"] == ["docs/x.md"]


def test_classification_is_sorted_and_order_independent():
    contract = {"allowed_globs": ["*.py"], "forbidden_globs": []}
    forward = classify_touched_files(["b.py", "a.py"], contract)
    backward = classify_touched_files(["a.py", "b.py"], contract)
    assert forward == backward == {"in_scope": ["a.py", "b.py"], "outside_scope": []}


def test_empty_touch_list_is_clean():
    contract = adapt_scope_contract(SAMPLE_TREE)
    assert classify_touched_files([], contract) == {"in_scope": [], "outside_scope": []}


# ---------------------------------------------------------- simulate_test_run
def test_check_passes_when_the_marker_is_in_the_file():
    repo = {"app.py": "if len(pw) < 8: raise Invalid"}
    checks = [{"name": "t_short_pw", "file": "app.py", "requires": ["len(pw) < 8"]}]
    assert simulate_test_run(repo, checks) == {
        "ran": True,
        "passed": ["t_short_pw"],
        "failed": [],
        "exit_code": 0,
    }


def test_missing_file_fails_the_check():
    checks = [{"name": "t_envelope", "file": "test_app.py", "requires": ["422"]}]
    result = simulate_test_run({}, checks)
    assert result["failed"] == ["t_envelope"] and result["exit_code"] == 1


def test_zero_checks_is_not_a_success():
    """«Тесты прошли» без прогона — то самое непроверяемое утверждение."""
    result = simulate_test_run({"app.py": "всё готово"}, [])
    assert result["ran"] is False
    assert result["exit_code"] == 1


def test_exit_code_is_nonzero_when_any_check_fails():
    repo = {"app.py": "pass"}
    checks = [
        {"name": "ok", "file": "app.py", "requires": ["pass"]},
        {"name": "bad", "file": "app.py", "requires": ["422"]},
    ]
    result = simulate_test_run(repo, checks)
    assert result["passed"] == ["ok"] and result["exit_code"] == 1


# ------------------------------------------------------------ handoff_quality
def test_missing_handoff_scores_zero():
    assert handoff_quality(None) == 0


def test_full_packet_scores_three():
    assert handoff_quality(FULL_PACKET) == 3


def test_packet_without_next_action_loses_a_point():
    weak = dict(FULL_PACKET, next_action="   ")
    assert handoff_quality(weak) == 2


# ----------------------------------------------------------------- measure_run
def test_measure_returns_all_five_outcomes():
    contract = adapt_scope_contract(SAMPLE_TREE)
    outcomes = measure_run(_workbench_run(), contract)
    assert set(outcomes) == set(OUTCOME_KEYS)


def test_measure_counts_files_outside_scope():
    contract = adapt_scope_contract(SAMPLE_TREE)
    assert measure_run(_prompt_only_run(), contract)["files_outside_scope"] == 1


def test_acceptance_is_not_met_when_the_proving_test_never_ran():
    """Правильный код в файле не доказывает цель, если тест не запускался."""
    contract = adapt_scope_contract(SAMPLE_TREE)
    assert measure_run(_prompt_only_run(), contract)["acceptance_met"] is False
    assert measure_run(_workbench_run(), contract)["acceptance_met"] is True


def test_tests_actually_run_requires_the_acceptance_command_in_commands():
    contract = adapt_scope_contract(SAMPLE_TREE)
    run = _workbench_run()
    run["commands"] = ["python3 -c 'print(1)'"]
    assert measure_run(run, contract)["tests_actually_run"] is False


def test_reviewer_total_sums_the_rubric():
    contract = adapt_scope_contract(SAMPLE_TREE)
    assert measure_run(_workbench_run(), contract)["reviewer_total"] == 4


# ------------------------------------------------------------ compare_pipelines
def test_fewer_files_outside_scope_wins():
    rows = compare_pipelines({"files_outside_scope": 3}, {"files_outside_scope": 0})
    row = next(r for r in rows if r["outcome"] == "files_outside_scope")
    assert row["winner"] == "candidate"


def test_equal_outcomes_are_a_tie_not_a_coin_flip():
    rows = compare_pipelines({"reviewer_total": 4}, {"reviewer_total": 4})
    row = next(r for r in rows if r["outcome"] == "reviewer_total")
    assert row["winner"] == "tie"


def test_comparison_rows_follow_the_documented_outcome_order():
    rows = compare_pipelines({}, {})
    assert [r["outcome"] for r in rows] == list(OUTCOME_KEYS)


def test_booleans_compare_as_numbers():
    rows = compare_pipelines({"acceptance_met": False}, {"acceptance_met": True})
    row = next(r for r in rows if r["outcome"] == "acceptance_met")
    assert row["winner"] == "candidate"


# --------------------------------------------------------- render_before_after
def test_report_has_a_row_per_outcome():
    rows = compare_pipelines({}, {})
    text = render_before_after(rows)
    for key in OUTCOME_KEYS:
        assert ("| %s |" % key) in text


def test_report_counts_workbench_wins():
    contract = adapt_scope_contract(SAMPLE_TREE)
    rows = compare_pipelines(
        measure_run(_prompt_only_run(), contract),
        measure_run(_workbench_run(), contract),
    )
    assert "выиграл воркбенч в 5 из 5 исходов" in render_before_after(rows)


def test_report_is_stable_across_calls():
    rows = compare_pipelines({"reviewer_total": 1}, {"reviewer_total": 2})
    assert render_before_after(rows) == render_before_after(rows)


def test_workbench_run_wins_every_outcome_on_the_sample_app():
    """Сквозная проверка: тот же таск, два конвейера, числа спорят сами."""
    contract = adapt_scope_contract(SAMPLE_TREE)
    rows = compare_pipelines(
        measure_run(_prompt_only_run(), contract),
        measure_run(_workbench_run(), contract),
    )
    assert {r["winner"] for r in rows} == {"candidate"}


# ------------------------------------------------------- false_negative_reason
def test_single_step_formatter_is_a_false_negative():
    assert false_negative_reason({"kind": "formatter", "steps": 1}) != ""


def test_multi_step_task_of_a_fast_kind_is_not_a_false_negative():
    assert false_negative_reason({"kind": "formatter", "steps": 4}) == ""


def test_forbidden_zone_cancels_the_fast_path():
    task = {"kind": "one_line_lint", "steps": 1, "touches_forbidden": True}
    assert false_negative_reason(task) == ""


def test_ordinary_feature_task_is_not_a_false_negative():
    assert false_negative_reason({"kind": "feature", "steps": 1}) == ""
