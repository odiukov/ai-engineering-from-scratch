"""Тесты к уроку «Режимы разрешений: allow/ask/deny». Правь exercise.py."""

import pytest

from exercise import (
    classifier_verdict,
    decide,
    mode_decision,
    parse_rule,
    risk_class,
    rule_matches,
    rule_specificity,
    strictest,
)


def act(tool, text=""):
    return {"tool": tool, "input": text}


def rule(pattern, decision):
    return {"pattern": pattern, "decision": decision}


# -------------------------------------------------------------- risk_class
def test_risk_class_of_read_only_tools():
    assert [risk_class(act(t)) for t in ("Read", "Glob", "Grep")] == ["read"] * 3


def test_risk_class_of_file_writing_tools():
    assert [risk_class(act(t)) for t in ("Edit", "Write")] == ["edit", "edit"]


def test_risk_class_separates_shell_from_network():
    assert (risk_class(act("Bash")), risk_class(act("WebFetch"))) == ("exec", "network")


def test_risk_class_of_an_unknown_tool_is_not_read():
    """Новый MCP-сервер, забытый в таблице, не должен получить права чтения."""
    assert risk_class(act("SomeMCPTool")) == "exec"


# -------------------------------------------------------------- parse_rule
def test_parse_rule_of_a_bare_tool_has_no_spec():
    assert parse_rule("Read") == ("Read", None)


def test_parse_rule_splits_tool_and_spec():
    assert parse_rule("Bash(git push:*)") == ("Bash", "git push:*")


def test_parse_rule_distinguishes_empty_spec_from_missing_spec():
    """"Bash()" и "Bash" разбираются по-разному и матчатся по-разному."""
    assert parse_rule("Bash()") == ("Bash", "")
    assert parse_rule("Bash") == ("Bash", None)


# ------------------------------------------------------------ rule_matches
def test_rule_matches_bare_tool_covers_any_input():
    assert rule_matches("Read", act("Read", "anything at all")) is True


def test_rule_matches_requires_the_same_tool():
    assert rule_matches("Read", act("Write", "a.py")) is False


def test_rule_matches_prefix_rule_only_on_the_prefix():
    assert rule_matches("Bash(git push:*)", act("Bash", "git push origin")) is True
    assert rule_matches("Bash(git push:*)", act("Bash", "git status")) is False


def test_rule_matches_exact_rule_needs_the_whole_input():
    """"Bash(ls)" — это ровно "ls", а не "ls -la"."""
    assert rule_matches("Bash(ls)", act("Bash", "ls")) is True
    assert rule_matches("Bash(ls)", act("Bash", "ls -la")) is False


def test_rule_matches_is_case_sensitive_on_the_tool_name():
    assert rule_matches("bash(*)", act("Bash", "ls")) is False


# ------------------------------------------------------- rule_specificity
def test_rule_specificity_of_a_wildcard_equals_a_bare_tool():
    assert rule_specificity("Bash(*)") == rule_specificity("Bash")


def test_rule_specificity_grows_with_the_prefix_length():
    assert rule_specificity("Bash(git log:*)") > rule_specificity("Bash(git:*)")


def test_rule_specificity_of_an_exact_rule_beats_any_prefix():
    """Точное правило конкретнее длинного префикса, каким бы длинным он ни был."""
    long_prefix = "Bash(" + "x" * 200 + ":*)"
    assert rule_specificity("Bash(ls)") > rule_specificity(long_prefix)


def test_rule_specificity_of_a_wildcard_is_the_floor():
    assert rule_specificity("Read") < rule_specificity("Read(a.py)")


# --------------------------------------------------------------- strictest
def test_strictest_lets_one_deny_beat_many_allows():
    """Это не голосование большинством."""
    assert strictest(["allow", "allow", "allow", "deny"]) == "deny"


def test_strictest_prefers_ask_over_allow():
    assert strictest(["allow", "ask"]) == "ask"


def test_strictest_does_not_depend_on_the_order():
    assert strictest(["deny", "allow", "ask"]) == strictest(["allow", "ask", "deny"])


def test_strictest_of_nothing_is_an_error_not_an_allow():
    """Тихое «раз правил нет, значит можно» — это дыра в правах."""
    with pytest.raises(ValueError):
        strictest([])


# ----------------------------------------------------------- mode_decision
def test_mode_decision_accept_edits_auto_approves_writes():
    assert mode_decision("acceptEdits", act("Write", "src/a.py")) == "allow"


def test_mode_decision_accept_edits_still_asks_before_shell():
    """Файлы — да, оболочка — нет. Ровно этим acceptEdits и отличается."""
    assert mode_decision("acceptEdits", act("Bash", "pytest -q")) == "ask"


def test_mode_decision_plan_asks_even_before_reading():
    assert mode_decision("plan", act("Read", "a.py")) == "ask"


def test_mode_decision_dont_ask_denies_what_no_rule_pre_approved():
    assert mode_decision("dontAsk", act("Read", "a.py")) == "deny"


def test_mode_decision_rejects_an_unknown_mode_instead_of_guessing():
    with pytest.raises(ValueError):
        mode_decision("acceptEdit", act("Write", "a.py"))


# ------------------------------------------------------ classifier_verdict
def test_classifier_verdict_approves_a_tool_inside_the_declared_task():
    assert classifier_verdict(act("Read", "a.py"), ("Read", "Edit")) == "allow"


def test_classifier_verdict_escalates_a_tool_outside_the_task_to_a_human():
    """Классификатор возвращает решение человеку, а не блокирует насмерть."""
    assert classifier_verdict(act("Bash", "ls"), ("Read", "Edit")) == "ask"


def test_classifier_verdict_with_an_empty_scope_escalates_everything():
    assert classifier_verdict(act("Read", "a.py"), ()) == "ask"


# ------------------------------------------------------------------ decide
def test_decide_falls_back_to_the_mode_when_no_rule_matches():
    assert decide(act("Bash", "rm -rf build"), [], "bypassPermissions") == "allow"


def test_decide_lets_deny_beat_allow_at_equal_specificity():
    rules = [rule("Bash(git push:*)", "allow"), rule("Bash(git push:*)", "deny")]
    assert decide(act("Bash", "git push origin"), rules, "default") == "deny"


def test_decide_lets_a_more_specific_allow_beat_a_general_deny():
    """Конкретность разбирается ДО строгости — иначе узкие исключения невозможны."""
    rules = [rule("Bash(*)", "deny"), rule("Bash(git status:*)", "allow")]
    assert decide(act("Bash", "git status"), rules, "default") == "allow"


def test_decide_lets_a_more_specific_deny_beat_a_general_allow():
    rules = [rule("Bash(*)", "allow"), rule("Bash(git push:*)", "deny")]
    assert decide(act("Bash", "git push origin main"), rules, "default") == "deny"


def test_decide_does_not_depend_on_the_order_of_rules_in_the_config():
    rules = [
        rule("Bash(*)", "deny"),
        rule("Bash(git:*)", "ask"),
        rule("Bash(git status)", "allow"),
    ]
    action = act("Bash", "git status")
    assert decide(action, rules, "default") == decide(
        action, list(reversed(rules)), "default"
    )


def test_decide_in_auto_mode_escalates_a_tool_outside_the_declared_scope():
    assert decide(act("Bash", "ls"), [], "auto", allowed_tools=("Read",)) == "ask"
    assert decide(act("Read", "a.py"), [], "auto", allowed_tools=("Read",)) == "allow"


def test_decide_approves_every_step_of_an_exfiltration_chain():
    """Каждое действие по отдельности законно; опасна композиция.

    Именно этот разрыв Anthropic и называет причиной, по которой
    классификатор — слой, а не решение.
    """
    rules = [
        rule("Read(*)", "allow"),
        rule("Write(*)", "allow"),
        rule("Bash(git push:*)", "allow"),
    ]
    chain = [
        act("Read", "~/.aws/credentials"),
        act("Write", "/tmp/notes.txt"),
        act("Bash", "git push origin main"),
    ]
    assert [decide(a, rules, "default") for a in chain] == ["allow"] * 3
