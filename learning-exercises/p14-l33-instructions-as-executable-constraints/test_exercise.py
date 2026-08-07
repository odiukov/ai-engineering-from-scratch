"""Тесты к уроку «Инструкции агента как исполняемые ограничения».

Правь exercise.py.
"""

import pytest

from exercise import (
    CATEGORIES,
    CONFIDENCE_THRESHOLD,
    check_rules,
    compile_rule,
    expired_rules,
    is_operational,
    parse_rules,
    rules_lock,
    severity_verdict,
)

RULES_MD = """# Agent Rules

## startup/state-file-read
- category: startup
- check: must_read_state
- severity: block
Агент читает agent_state.json до первого вызова инструмента.

## forbidden/no-release-script-edits
- category: forbidden
- check: no_edits_to
- arg: scripts/*.sh
- severity: block
- expires_at: 2026-12-01
Не редактируй релизные скрипты вне утверждённой релизной задачи.

## done/tests-pass
- category: definition_of_done
- check: tests_exit_zero
Задача закрыта, только когда её приёмочная команда вышла с нулём.

## uncertainty/be-careful
- category: uncertainty
Будь аккуратен и думай дважды.
"""


def clean_trace():
    return {
        "read_state": True,
        "edited_files": ["app.py", "test_app.py"],
        "tests_exit_code": 0,
        "confidence": 0.9,
        "asked_for_help": False,
        "added_dependencies": [],
        "approvals": [],
    }


# -------------------------------------------------------------- parse_rules
def test_every_rule_block_becomes_a_rule():
    assert [r["slug"] for r in parse_rules(RULES_MD)] == [
        "startup/state-file-read",
        "forbidden/no-release-script-edits",
        "done/tests-pass",
        "uncertainty/be-careful",
    ]


def test_rule_fields_are_parsed():
    rule = parse_rules(RULES_MD)[1]
    assert (rule["category"], rule["check"], rule["arg"]) == (
        "forbidden",
        "no_edits_to",
        "scripts/*.sh",
    )
    assert rule["expires_at"] == "2026-12-01"


def test_severity_defaults_to_warn_when_not_written():
    assert parse_rules(RULES_MD)[2]["severity"] == "warn"


def test_description_survives_parsing():
    assert "релизные скрипты" in parse_rules(RULES_MD)[1]["description"]


def test_category_outside_the_five_is_refused():
    """Правило, не влезшее в пять категорий, хочет быть двумя правилами."""
    with pytest.raises(ValueError):
        parse_rules("## x/y\n- category: vibes\n- check: tests_exit_zero\nтекст\n")


def test_unknown_severity_is_refused():
    with pytest.raises(ValueError):
        parse_rules("## x/y\n- category: startup\n- severity: critical\nтекст\n")


# ------------------------------------------------------------ is_operational
def test_a_rule_with_a_known_check_is_operational():
    assert is_operational(parse_rules(RULES_MD)[0]) is True


def test_a_rule_without_a_check_is_aspirational():
    """«Будь аккуратен» — пожелание: проверить его нечем."""
    assert is_operational(parse_rules(RULES_MD)[3]) is False


def test_an_unknown_check_name_is_not_operational():
    assert is_operational({"check": "be_extra_careful"}) is False


# ------------------------------------------------------------- compile_rule
def test_compiled_rule_rejects_a_violating_diff_and_passes_a_clean_one():
    """Ядро урока: инструкция превращается в предикат над diff."""
    predicate = compile_rule({"check": "no_edits_to", "arg": "scripts/*.sh"})
    assert predicate({"edited_files": ["app.py", "scripts/release.sh"]}) is False
    assert predicate({"edited_files": ["app.py", "test_app.py"]}) is True


def test_forbidden_glob_survives_a_file_move():
    """Glob, а не точный путь: файл переехал, правило продолжает ловить."""
    predicate = compile_rule({"check": "no_edits_to", "arg": "scripts/*"})
    assert predicate({"edited_files": ["scripts/ci_release.sh"]}) is False


def test_startup_rule_reads_the_state_flag():
    predicate = compile_rule({"check": "must_read_state"})
    assert predicate({"read_state": True}) is True
    assert predicate({"read_state": False}) is False


def test_tests_never_run_is_not_the_same_as_tests_green():
    """None означает «не запускали» и правило это не пропускает."""
    predicate = compile_rule({"check": "tests_exit_zero"})
    assert predicate({"tests_exit_code": None}) is False
    assert predicate({"tests_exit_code": 0}) is True


def test_low_confidence_is_forgiven_only_if_the_agent_asked():
    predicate = compile_rule({"check": "ask_when_unsure"})
    low = CONFIDENCE_THRESHOLD - 0.3
    assert predicate({"confidence": low, "asked_for_help": False}) is False
    assert predicate({"confidence": low, "asked_for_help": True}) is True


def test_new_dependency_needs_an_explicit_approval():
    predicate = compile_rule({"check": "approve_new_dependency"})
    assert predicate({"added_dependencies": ["fastapi"], "approvals": []}) is False
    assert predicate({"added_dependencies": ["fastapi"], "approvals": ["fastapi"]}) is True


def test_an_uncompilable_rule_raises_instead_of_always_passing():
    """Правило, которое всегда проходит, хуже отсутствующего."""
    with pytest.raises(ValueError):
        compile_rule({"check": None})
    with pytest.raises(ValueError):
        compile_rule({"check": "no_edits_to", "arg": None})


# -------------------------------------------------------------- check_rules
def test_a_clean_run_passes_every_operational_rule():
    results = check_rules(parse_rules(RULES_MD), clean_trace())
    assert [r["status"] for r in results] == ["pass", "pass", "pass", "unchecked"]


def test_a_violating_run_fails_exactly_the_rules_it_broke():
    trace = clean_trace()
    trace["edited_files"] = ["app.py", "scripts/release.sh"]
    trace["read_state"] = False
    results = check_rules(parse_rules(RULES_MD), trace)
    assert [r["status"] for r in results] == ["fail", "fail", "pass", "unchecked"]


def test_unchecked_is_not_the_same_as_pass():
    """Отчёт обязан отличать «проверили» от «проверить нечем»."""
    results = check_rules(parse_rules(RULES_MD), clean_trace())
    assert results[3]["status"] == "unchecked"


def test_results_keep_the_rule_order_and_severity():
    results = check_rules(parse_rules(RULES_MD), clean_trace())
    assert results[0]["slug"] == "startup/state-file-read"
    assert results[0]["severity"] == "block"


# --------------------------------------------------------- severity_verdict
def test_a_failing_block_rule_stops_the_run():
    trace = clean_trace()
    trace["read_state"] = False
    assert severity_verdict(check_rules(parse_rules(RULES_MD), trace)) == "block"


def test_a_failing_warn_rule_only_warns():
    trace = clean_trace()
    trace["tests_exit_code"] = 1
    assert severity_verdict(check_rules(parse_rules(RULES_MD), trace)) == "warn"


def test_info_failures_do_not_gate():
    assert severity_verdict([{"severity": "info", "status": "fail"}]) == "pass"


def test_empty_report_passes():
    assert severity_verdict([]) == "pass"


def test_block_wins_over_warn():
    results = [
        {"severity": "warn", "status": "fail"},
        {"severity": "block", "status": "fail"},
    ]
    assert severity_verdict(results) == "block"


# ----------------------------------------------------------- expired_rules
def test_rule_past_its_date_is_expired():
    assert expired_rules(parse_rules(RULES_MD), "2027-01-01") == [
        "forbidden/no-release-script-edits"
    ]


def test_rule_before_its_date_is_alive():
    assert expired_rules(parse_rules(RULES_MD), "2026-01-01") == []


def test_rule_without_a_date_never_expires():
    """Именно так набор правил дорастает до восьмидесяти нерабочих штук."""
    assert expired_rules(parse_rules(RULES_MD), "2099-01-01") == [
        "forbidden/no-release-script-edits"
    ]


def test_expiry_uses_the_now_argument_not_the_wall_clock():
    rules = [{"slug": "a", "expires_at": "2026-05-05"}]
    assert expired_rules(rules, "2026-05-04") == []
    assert expired_rules(rules, "2026-05-06") == ["a"]


# --------------------------------------------------------------- rules_lock
def test_lock_keeps_only_operational_rules():
    assert [r["slug"] for r in rules_lock(parse_rules(RULES_MD))] == [
        "startup/state-file-read",
        "forbidden/no-release-script-edits",
        "done/tests-pass",
    ]


def test_lock_is_stable_when_the_markdown_is_reordered():
    """Markdown — источник, lock — кэш: перестановка блоков не должна шуметь."""
    blocks = RULES_MD.split("\n## ")
    reordered = blocks[0] + "\n## " + "\n## ".join(reversed(blocks[1:]))
    assert rules_lock(parse_rules(RULES_MD)) == rules_lock(parse_rules(reordered))


def test_lock_is_sorted_by_category_then_slug():
    rules = [
        {"slug": "z", "category": "startup", "check": "must_read_state", "severity": "warn"},
        {"slug": "a", "category": "startup", "check": "must_read_state", "severity": "warn"},
        {"slug": "m", "category": CATEGORIES[-1], "check": "tests_exit_zero", "severity": "warn"},
    ]
    assert [r["slug"] for r in rules_lock(rules)] == ["a", "z", "m"]


def test_lock_carries_the_argument_the_hot_path_needs():
    entry = rules_lock(parse_rules(RULES_MD))[1]
    assert entry["arg"] == "scripts/*.sh"
    assert entry["severity"] == "block"
