"""Тесты к уроку «Failure modes: почему агенты ломаются». Правь exercise.py."""

import pytest

from exercise import (
    FAILURE_MODES,
    cascade_radius,
    context_violations,
    first_repeat_index,
    mode_distribution,
    scope_creep_targets,
    success_hallucination,
    tag_trace,
    tool_problems,
)

SEARCH = {"tool": "search", "args": {"query": "invoice 4711"}}
READ = {"tool": "read_file", "args": {"path": "README.md"}}


# ----------------------------------------------------------- tool_problems
def test_clean_trace_has_no_tool_problems():
    assert tool_problems([SEARCH, READ]) == {"unknown": [], "bad_args": []}


def test_unknown_tool_is_reported_by_name():
    assert tool_problems([{"tool": "magic_scan", "args": {}}])["unknown"] == ["magic_scan"]


def test_missing_required_argument_is_tool_misuse():
    assert tool_problems([{"tool": "write_file", "args": {"path": "a"}}])["bad_args"] == [0]


def test_extra_argument_is_tool_misuse():
    """Лишний ключ — почти всегда перепутанные схемы двух инструментов."""
    steps = [{"tool": "read_file", "args": {"path": "a", "content": "b"}}]
    assert tool_problems(steps)["bad_args"] == [0]


def test_invented_tool_is_not_also_an_argument_problem():
    """Схемы у выдуманного инструмента нет — ругаться на аргументы нечем."""
    problems = tool_problems([{"tool": "magic_scan", "args": {"anything": 1}}])
    assert problems["bad_args"] == []


# ------------------------------------------------------- first_repeat_index
def test_three_identical_calls_in_a_row_are_a_loop():
    assert first_repeat_index([SEARCH, SEARCH, SEARCH]) == 2


def test_two_identical_calls_are_not_a_loop_yet():
    assert first_repeat_index([SEARCH, SEARCH]) is None


def test_identical_calls_separated_by_other_work_are_not_a_loop():
    """Главное свойство: зацикливание — это повтор ПОДРЯД, а не вообще."""
    assert first_repeat_index([SEARCH, READ, SEARCH, READ, SEARCH]) is None


def test_same_tool_with_different_arguments_is_not_a_repeat():
    steps = [
        {"tool": "search", "args": {"query": "a"}},
        {"tool": "search", "args": {"query": "b"}},
        {"tool": "search", "args": {"query": "c"}},
    ]
    assert first_repeat_index(steps) is None


def test_repeat_limit_below_two_is_value_error():
    with pytest.raises(ValueError):
        first_repeat_index([SEARCH, SEARCH], limit=1)


# ------------------------------------------------------------ cascade_radius
def test_radius_counts_steps_after_the_first_error():
    steps = [dict(SEARCH, status="error"), READ, {"tool": "list_dir", "args": {"path": "."}}]
    assert cascade_radius(steps) == 2


def test_clean_trace_has_zero_radius():
    assert cascade_radius([SEARCH, READ]) == 0


def test_error_on_the_last_step_has_zero_radius():
    assert cascade_radius([SEARCH, dict(READ, status="error")]) == 0


def test_radius_is_measured_from_the_first_error_not_the_last():
    """От последней ошибки радиус всегда маленький — самый дорогой каскад пропадёт."""
    steps = [dict(SEARCH, status="error"), READ, dict(READ, status="error"), READ, SEARCH]
    assert cascade_radius(steps) == 4


# -------------------------------------------------------- context_violations
def test_forbidden_tool_is_a_violation():
    steps = [SEARCH, {"tool": "send_email", "args": {"to": "b@x", "body": "hi"}}]
    assert context_violations(steps, {"forbidden_tools": ("send_email",)}) == [1]


def test_forbidden_path_prefix_is_a_violation():
    steps = [READ, {"tool": "write_file", "args": {"path": "src/a.py", "content": ""}}]
    assert context_violations(steps, {"forbidden_paths": ("src/",)}) == [1]


def test_without_constraints_nothing_is_a_violation():
    steps = [{"tool": "write_file", "args": {"path": "src/a.py", "content": ""}}]
    assert context_violations(steps, {}) == []


def test_a_query_that_merely_looks_like_a_path_is_not_a_violation():
    """Искать «src/» можно — обращения к src/ при этом не происходит."""
    steps = [{"tool": "search", "args": {"query": "src/secret"}}]
    assert context_violations(steps, {"forbidden_paths": ("src/",)}) == []


def test_late_violation_keeps_its_index():
    """Позиция и есть сигнал: тридцать шагов помнил ограничение, на тридцать первом забыл."""
    steps = [READ] * 30 + [{"tool": "write_file", "args": {"path": "src/a.py", "content": ""}}]
    assert context_violations(steps, {"forbidden_paths": ("src/",)}) == [30]


# ----------------------------------------------------- scope_creep_targets
def test_writing_outside_the_request_is_scope_creep():
    steps = [
        {"tool": "write_file", "args": {"path": "README.md", "content": "x"}},
        {"tool": "write_file", "args": {"path": "src/a.py", "content": "x"}},
    ]
    assert scope_creep_targets(steps, ("README.md",)) == ["src/a.py"]


def test_writing_the_requested_target_is_not_creep():
    steps = [{"tool": "write_file", "args": {"path": "README.md", "content": "x"}}]
    assert scope_creep_targets(steps, ("README.md",)) == []


def test_reading_is_never_scope_creep():
    """Осмотреться агент имеет право — расширение задачи начинается с записи."""
    assert scope_creep_targets([{"tool": "read_file", "args": {"path": "secret"}}], ()) == []


def test_extra_recipient_is_scope_creep():
    steps = [
        {"tool": "send_email", "args": {"to": "boss@x", "body": "report"}},
        {"tool": "send_email", "args": {"to": "all@x", "body": "report"}},
    ]
    assert scope_creep_targets(steps, ("boss@x",)) == ["all@x"]


# --------------------------------------------------- success_hallucination
def test_claimed_success_without_state_change_is_a_hallucination():
    trace = {
        "steps": [{"tool": "write_file", "args": {"path": "a", "content": "b"},
                   "status": "error"}],
        "claims_success": True,
        "state_changed": False,
    }
    assert success_hallucination(trace) is True


def test_read_only_trace_claiming_success_is_fine():
    """Чтение и не должно ничего менять — это не фальшивый успех."""
    trace = {"steps": [SEARCH, READ], "claims_success": True, "state_changed": False}
    assert success_hallucination(trace) is False


def test_real_state_change_is_not_a_hallucination():
    trace = {
        "steps": [{"tool": "write_file", "args": {"path": "a", "content": "b"}}],
        "claims_success": True,
        "state_changed": True,
    }
    assert success_hallucination(trace) is False


def test_honest_failure_report_is_not_a_hallucination():
    trace = {
        "steps": [{"tool": "write_file", "args": {"path": "a", "content": "b"},
                   "status": "error"}],
        "claims_success": False,
        "state_changed": False,
    }
    assert success_hallucination(trace) is False


# ---------------------------------------------------------------- tag_trace
def test_clean_trace_has_no_labels():
    assert tag_trace({"steps": [SEARCH, READ]}) == []


def test_one_trace_can_carry_several_labels():
    trace = {
        "steps": [
            dict(SEARCH, status="error"),
            {"tool": "magic_scan", "args": {}},
            {"tool": "write_file", "args": {"path": "src/a.py", "content": "y"}},
        ],
        "constraints": {"forbidden_paths": ("src/",)},
        "allowed_targets": ("README.md",),
        "claims_success": True,
        "state_changed": False,
    }
    assert tag_trace(trace) == [
        "cascading_error",
        "context_loss",
        "hallucinated_action",
        "scope_creep",
        "success_hallucination",
    ]


def test_single_retry_after_an_error_is_not_a_cascade():
    """Один шаг после ошибки — это обычный retry, а не каскад."""
    trace = {"steps": [dict(SEARCH, status="error"), SEARCH]}
    assert tag_trace(trace) == []


def test_every_label_belongs_to_the_known_mode_list():
    trace = {"steps": [{"tool": "magic_scan", "args": {}},
                       {"tool": "read_file", "args": {"file": "a"}}]}
    labels = tag_trace(trace)
    assert labels == ["hallucinated_action", "tool_misuse"]
    assert set(labels) <= set(FAILURE_MODES)


def test_repeat_limit_is_configurable_per_trace():
    trace = {"steps": [SEARCH, SEARCH], "repeat_limit": 2}
    assert tag_trace(trace) == ["repeat_loop"]


# --------------------------------------------------------- mode_distribution
def test_distribution_counts_traces_not_individual_hits():
    """Пять зацикливаний в одном трейсе — это всё равно один пойманный трейс."""
    trace = {"steps": [SEARCH] * 5}
    assert mode_distribution([trace]) == {"repeat_loop": 1}


def test_modes_that_never_fired_are_absent():
    assert mode_distribution([{"steps": [SEARCH, READ]}]) == {}


def test_distribution_aggregates_over_traces():
    traces = [
        {"steps": [{"tool": "magic_scan", "args": {}}]},
        {"steps": [{"tool": "magic_scan", "args": {}}, SEARCH]},
        {"steps": [{"tool": "read_file", "args": {"file": "a"}}]},
    ]
    assert mode_distribution(traces) == {"hallucinated_action": 2, "tool_misuse": 1}


def test_distribution_keys_are_sorted():
    traces = [{"steps": [{"tool": "read_file", "args": {"file": "a"}},
                         {"tool": "magic_scan", "args": {}}]}]
    assert list(mode_distribution(traces)) == sorted(mode_distribution(traces))
