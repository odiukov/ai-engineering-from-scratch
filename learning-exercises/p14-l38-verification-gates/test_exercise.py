"""Тесты к уроку «Гейты верификации». Правь exercise.py."""

import pytest

from exercise import (
    COVERAGE_FLOOR,
    GATE_ORDER,
    REPORT_DIR,
    apply_override,
    finding,
    gate_coverage,
    gate_feedback,
    gate_rules,
    gate_scope,
    run_gates,
    verification_report,
)

ACCEPT = "pytest -q tests/test_signup.py"


def artifacts(**over):
    """Чистый набор артефактов; любой ключ подменяется именованным аргументом."""
    base = {
        "scope": {
            "acceptance": [ACCEPT],
            "allowed_files": ["app/*.py", "tests/*.py"],
            "forbidden_files": ["scripts/*", ".github/*"],
        },
        "feedback": [{"command": ACCEPT, "exit_code": 0}],
        "diff": {"touched_files": ["app/signup.py", "tests/test_signup.py"]},
        "rules": [{"id": "no-todo", "severity": "block", "passed": True}],
        "coverage": {"measured": 84.0, "previous": 84.0},
    }
    base.update(over)
    return base


def codes(findings):
    return [f["code"] for f in findings]


# ------------------------------------------------------------------ finding
def test_finding_carries_code_severity_and_source():
    f = finding("NULL_EXIT", "block", "нет exit_code", "feedback")
    assert (f["code"], f["severity"], f["source"]) == ("NULL_EXIT", "block", "feedback")


def test_a_fresh_finding_is_not_overridden_yet():
    assert finding("X", "warn", "m", "scope")["overridden"] is False


def test_a_misspelled_severity_is_rejected():
    """"blok" молча не заблокирует ничего — поэтому ValueError, а не тихий пропуск."""
    with pytest.raises(ValueError):
        finding("X", "blok", "m", "scope")


# ------------------------------------------------------------- gate_feedback
def test_a_ran_and_green_acceptance_command_produces_nothing():
    assert gate_feedback(artifacts()) == []


def test_acceptance_command_that_never_ran_is_blocked():
    found = gate_feedback(artifacts(feedback=[{"command": "ruff check", "exit_code": 0}]))
    assert codes(found) == ["ACCEPTANCE_NOT_RUN"]
    assert found[0]["severity"] == "block"


def test_acceptance_command_with_a_nonzero_exit_is_blocked():
    found = gate_feedback(artifacts(feedback=[{"command": ACCEPT, "exit_code": 1}]))
    assert codes(found) == ["ACCEPTANCE_FAILED"]


def test_a_null_exit_anywhere_in_the_log_is_blocked():
    """Сорванный прогон любой команды означает, что журналу нельзя верить целиком."""
    found = gate_feedback(
        artifacts(
            feedback=[{"command": ACCEPT, "exit_code": 0}, {"command": "ruff", "exit_code": None}]
        )
    )
    assert codes(found) == ["NULL_EXIT"]


def test_the_last_run_of_a_command_is_the_one_that_counts():
    found = gate_feedback(
        artifacts(
            feedback=[{"command": ACCEPT, "exit_code": 1}, {"command": ACCEPT, "exit_code": 0}]
        )
    )
    assert found == []


# ---------------------------------------------------------------- gate_scope
def test_edits_inside_the_contract_produce_nothing():
    assert gate_scope(artifacts()) == []


def test_a_write_into_a_forbidden_zone_blocks():
    found = gate_scope(artifacts(diff={"touched_files": ["scripts/release.sh"]}))
    assert codes(found) == ["FORBIDDEN_WRITE"]
    assert found[0]["severity"] == "block"


def test_a_write_outside_the_contract_only_warns():
    found = gate_scope(artifacts(diff={"touched_files": ["README.md"]}))
    assert codes(found) == ["OFF_SCOPE_WRITE"]
    assert found[0]["severity"] == "warn"


def test_forbidden_beats_allowed_when_both_patterns_match():
    """Иначе достаточно расширить allowed_files, чтобы обойти запрет."""
    art = artifacts(
        scope={
            "acceptance": [],
            "allowed_files": ["scripts/*"],
            "forbidden_files": ["scripts/release.sh"],
        },
        diff={"touched_files": ["scripts/release.sh"]},
    )
    assert codes(gate_scope(art)) == ["FORBIDDEN_WRITE"]


# ---------------------------------------------------------------- gate_rules
def test_passing_rules_produce_nothing():
    assert gate_rules(artifacts()) == []


def test_a_failed_rule_keeps_its_own_severity():
    art = artifacts(rules=[{"id": "line-length", "severity": "warn", "passed": False}])
    found = gate_rules(art)
    assert found[0]["severity"] == "warn"
    assert "line-length" in found[0]["message"]


def test_the_failed_rule_id_reaches_the_report():
    art = artifacts(rules=[{"id": "no-todo", "severity": "block", "passed": False}])
    assert "no-todo" in gate_rules(art)[0]["message"]


def test_every_failed_rule_gets_its_own_finding():
    art = artifacts(
        rules=[
            {"id": "a", "severity": "block", "passed": False},
            {"id": "b", "severity": "warn", "passed": False},
            {"id": "c", "severity": "block", "passed": True},
        ]
    )
    assert len(gate_rules(art)) == 2


# ------------------------------------------------------------- gate_coverage
def test_coverage_above_the_floor_produces_nothing():
    assert gate_coverage(artifacts()) == []


def test_coverage_below_the_floor_blocks():
    art = artifacts(coverage={"measured": COVERAGE_FLOOR - 1, "previous": COVERAGE_FLOOR - 1})
    assert codes(gate_coverage(art)) == ["COVERAGE_BELOW_FLOOR"]


def test_a_deleted_test_shows_up_as_a_regression_even_above_the_floor():
    """95 -> 81 остаётся выше пола: без второго порога это прошло бы зелёным."""
    art = artifacts(coverage={"measured": 81.0, "previous": 95.0})
    assert codes(gate_coverage(art)) == ["COVERAGE_REGRESSION"]


def test_a_missing_coverage_report_only_warns():
    found = gate_coverage(artifacts(coverage=None))
    assert codes(found) == ["COVERAGE_MISSING"]
    assert found[0]["severity"] == "warn"


def test_the_floor_can_come_from_the_artifact_itself():
    art = artifacts(coverage={"measured": 50.0, "previous": 50.0, "floor": 40.0})
    assert gate_coverage(art) == []


# ------------------------------------------------------------------ run_gates
def test_clean_artifacts_run_every_gate_and_pass():
    result = run_gates(artifacts())
    assert result["passed"] is True
    assert result["ran"] == list(GATE_ORDER)
    assert result["skipped"] == []


def test_a_blocking_gate_stops_the_gates_after_it():
    """Главное свойство: упавший гейт не пропускает дальше."""
    result = run_gates(artifacts(feedback=[]))
    assert result["passed"] is False
    assert result["ran"] == ["feedback"]
    assert result["skipped"] == list(GATE_ORDER[1:])


def test_the_gates_after_a_block_are_really_not_called():
    calls = []

    def spy(art):
        calls.append("spy")
        return []

    def always_blocks(art):
        return [finding("BOOM", "block", "m", "spy")]

    run_gates(artifacts(), gates=[("boom", always_blocks), ("spy", spy)])
    assert calls == []


def test_a_warning_alone_does_not_stop_the_run():
    result = run_gates(artifacts(diff={"touched_files": ["README.md"]}))
    assert result["passed"] is True
    assert codes(result["findings"]) == ["OFF_SCOPE_WRITE"]
    assert result["skipped"] == []


def test_strict_mode_turns_the_same_warning_into_a_block():
    result = run_gates(artifacts(diff={"touched_files": ["README.md"]}), strict=True)
    assert result["passed"] is False
    assert result["findings"][0]["severity"] == "block"
    assert result["findings"][0]["promoted_from"] == "warn"


def test_strict_mode_short_circuits_earlier_than_the_normal_run():
    art = artifacts(diff={"touched_files": ["README.md"]})
    assert run_gates(art, strict=True)["skipped"] == list(GATE_ORDER[2:])
    assert run_gates(art)["skipped"] == []


def test_gate_order_is_respected():
    order = []
    gates = [(name, lambda art, n=name: order.append(n) or []) for name in ("c", "a", "b")]
    run_gates(artifacts(), gates=gates)
    assert order == ["c", "a", "b"]


# --------------------------------------------------------- verification_report
def test_the_report_lands_on_one_path_per_task():
    report = verification_report("T-17", artifacts())
    assert report["path"] == f"{REPORT_DIR}/T-17.json"


def test_a_clean_task_passes_and_records_the_clock_it_was_given():
    report = verification_report("T-1", artifacts(), now=1234)
    assert report["passed"] is True
    assert report["generated_at"] == 1234


def test_a_report_without_a_task_id_is_refused():
    with pytest.raises(ValueError):
        verification_report("", artifacts())


def test_a_fresh_report_carries_no_overrides():
    assert verification_report("T-1", artifacts())["overrides"] == []


# ------------------------------------------------------------- apply_override
def blocked_report():
    return verification_report("T-9", artifacts(feedback=[]), now=10)


def test_a_signed_override_flips_the_verdict():
    report, row = apply_override(
        blocked_report(), "ACCEPTANCE_NOT_RUN", "команда переехала в CI", "u-42", "abc123", 11
    )
    assert report["passed"] is True
    assert row["overridden_by"] == "u-42"
    assert row["commit"] == "abc123"


def test_an_override_without_a_reason_is_refused():
    with pytest.raises(ValueError):
        apply_override(blocked_report(), "ACCEPTANCE_NOT_RUN", "", "u-42", "abc123", 11)


def test_an_agent_cannot_sign_its_own_pass():
    """Без этой проверки гейт становится театром."""
    with pytest.raises(ValueError):
        apply_override(
            blocked_report(), "ACCEPTANCE_NOT_RUN", "мне так удобнее", "agent:builder", "abc", 11
        )


def test_a_warning_has_nothing_to_override():
    report = verification_report("T-2", artifacts(diff={"touched_files": ["README.md"]}))
    with pytest.raises(KeyError):
        apply_override(report, "OFF_SCOPE_WRITE", "ок", "u-42", "abc123", 11)


def test_the_original_report_is_left_intact():
    original = blocked_report()
    apply_override(original, "ACCEPTANCE_NOT_RUN", "ок", "u-42", "abc123", 11)
    assert original["passed"] is False
    assert original["findings"][0]["overridden"] is False
    assert original["overrides"] == []


def test_one_override_does_not_clear_a_second_unrelated_block():
    art = artifacts(feedback=[], rules=[{"id": "no-todo", "severity": "block", "passed": False}])
    report = verification_report("T-3", art)
    # feedback падает первым, до правил дело не доходит — сначала чиним его
    assert report["gates_skipped"]
    fixed = verification_report("T-3", artifacts(rules=art["rules"]))
    after, _ = apply_override(fixed, "RULE_FAILED", "долг оформлен", "u-42", "abc", 12)
    assert after["passed"] is True
