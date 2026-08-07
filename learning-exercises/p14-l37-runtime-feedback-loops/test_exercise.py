"""Тесты к уроку «Петли обратной связи во время выполнения». Правь exercise.py."""

import pytest

from exercise import (
    HEAD_LINES,
    MAX_TURNS,
    TAIL_LINES,
    deterministic_tail,
    loop_can_advance,
    make_record,
    redact,
    retry_chain,
    rotate,
    run_feedback_loop,
)


def numbered(n):
    """Вывод из n пронумерованных строк — удобно проверять, что именно уцелело."""
    return "\n".join(f"line-{i}" for i in range(n))


def make_clock(start=1000, step=10):
    """Часы-счётчик: время идёт параметром, никакого time.time()."""
    box = {"now": start}

    def clock():
        value = box["now"]
        box["now"] += step
        return value

    return clock


def ok_record(command_id, parent=None, exit_code=0):
    return make_record(
        ["pytest"], exit_code, "1 passed", "", "жду зелёный", 0, 1, command_id,
        parent_command_id=parent,
    )


# --------------------------------------------------------- deterministic_tail
def test_short_output_survives_untouched():
    text = numbered(HEAD_LINES + TAIL_LINES)
    assert deterministic_tail(text) == (text, 0)


def test_long_output_keeps_the_head_and_the_tail():
    kept, dropped = deterministic_tail(numbered(20), head=2, tail=3)
    assert dropped == 15
    assert kept.splitlines() == [
        "line-0", "line-1", "...truncated 15 lines...", "line-17", "line-18", "line-19",
    ]


def test_truncation_is_deterministic_across_calls():
    """Одинаковый вывод обязан давать одинаковую запись, иначе две попытки не сравнить."""
    text = numbered(200)
    assert deterministic_tail(text) == deterministic_tail(text)


def test_the_last_line_is_never_dropped():
    """Итоговая ошибка живёт в конце — усечение обязано её сохранить."""
    kept, _ = deterministic_tail(numbered(500) + "\nFAILED test_signup")
    assert kept.splitlines()[-1] == "FAILED test_signup"


def test_empty_output_drops_nothing():
    assert deterministic_tail("") == ("", 0)


# ------------------------------------------------------------------- redact
def test_bearer_token_never_reaches_the_record():
    clean, n = redact("Authorization: Bearer sk-live-abc.123")
    assert "sk-live-abc.123" not in clean
    assert n == 1


def test_key_value_secrets_are_stripped_but_the_key_name_stays():
    clean, n = redact("password=hunter2")
    assert clean == "password=[REDACTED]"
    assert n == 1


def test_clean_output_is_returned_byte_for_byte():
    assert redact("1 passed in 0.02s") == ("1 passed in 0.02s", 0)


def test_every_secret_in_a_multiline_log_is_counted():
    log = "api_key=abc\nAKIA0123456789ABCDEF\nxoxb-1-2-abcdef\nall good"
    clean, n = redact(log)
    assert n == 3
    assert "abc\n" not in clean and "0123456789ABCDEF" not in clean


# --------------------------------------------------------------- make_record
def test_command_is_stored_as_argv_not_as_a_shell_string():
    record = make_record(["pytest", "-q"], 0, "", "", "", 0, 1, "a-0")
    assert record["command"] == ("pytest", "-q")


def test_record_redacts_before_it_truncates():
    """Ловушка порядка: усечь сначала — и секрет уцелеет в оставленном хвосте."""
    stdout = numbered(50) + "\nBearer sk-secret-tail"
    record = make_record(["run"], 0, stdout, "", "", 0, 1, "a-0")
    assert "sk-secret-tail" not in record["stdout_tail"]
    assert record["redacted"] == 1
    assert record["dropped_stdout_lines"] > 0


def test_stdout_and_stderr_are_truncated_independently():
    record = make_record(["run"], 1, numbered(50), "boom", "", 0, 1, "a-0")
    assert record["dropped_stdout_lines"] > 0
    assert record["dropped_stderr_lines"] == 0
    assert record["stderr_tail"] == "boom"


def test_empty_command_is_rejected():
    with pytest.raises(ValueError):
        make_record([], 0, "", "", "", 0, 1, "a-0")


def test_negative_duration_is_rejected():
    with pytest.raises(ValueError):
        make_record(["pytest"], 0, "", "", "", 0, -1, "a-0")


def test_record_carries_every_field_the_next_turn_reads():
    record = make_record(["pytest"], 3, "out", "err", "жду падение", 777, 42, "a-1", "a-0")
    assert record["exit_code"] == 3
    assert record["started_at"] == 777
    assert record["duration_ms"] == 42
    assert record["agent_note"] == "жду падение"
    assert record["parent_command_id"] == "a-0"


# ---------------------------------------------------------- loop_can_advance
def test_failure_is_a_signal_and_does_not_block_the_loop():
    assert loop_can_advance(ok_record("a-0", exit_code=17)) is True


def test_success_does_not_block_the_loop():
    assert loop_can_advance(ok_record("a-0")) is True


def test_missing_exit_code_blocks_the_loop():
    record = make_record(["pytest"], None, "", "", "", 0, 1, "a-0", error="timeout")
    assert loop_can_advance(record) is False


def test_error_blocks_even_when_an_exit_code_is_present():
    record = make_record(["pytest"], 0, "", "", "", 0, 1, "a-0", error="runner crashed")
    assert loop_can_advance(record) is False


# --------------------------------------------------------------- retry_chain
def test_chain_runs_from_the_first_attempt_to_the_asked_one():
    records = [ok_record("a-0"), ok_record("a-1", "a-0"), ok_record("a-2", "a-1")]
    assert [r["command_id"] for r in retry_chain(records, "a-2")] == ["a-0", "a-1", "a-2"]


def test_a_root_attempt_is_a_chain_of_one():
    records = [ok_record("a-0"), ok_record("b-0")]
    assert retry_chain(records, "b-0") == [records[1]]


def test_chain_ignores_attempts_from_other_branches():
    records = [ok_record("a-0"), ok_record("a-1", "a-0"), ok_record("z-0")]
    assert [r["command_id"] for r in retry_chain(records, "a-1")] == ["a-0", "a-1"]


def test_unknown_command_id_is_reported_not_guessed():
    with pytest.raises(KeyError):
        retry_chain([ok_record("a-0")], "nope")


def test_cycle_in_parent_links_is_rejected():
    records = [ok_record("a-0", "a-1"), ok_record("a-1", "a-0")]
    with pytest.raises(ValueError):
        retry_chain(records, "a-1")


# -------------------------------------------------------------------- rotate
def test_append_below_the_limit_only_grows_the_current_file():
    assert rotate({0: 100}, 50, limit=1000) == {0: 150}


def test_overflow_pushes_the_current_file_one_generation_down():
    assert rotate({0: 990}, 50, limit=1000) == {0: 50, 1: 990}


def test_the_oldest_generation_is_dropped_not_kept_forever():
    store = {0: 999, 1: 1, 2: 2}
    assert rotate(store, 10, limit=1000, max_rotations=2) == {0: 10, 1: 999, 2: 1}


def test_a_record_larger_than_the_limit_is_still_written():
    """Потерять обратную связь хуже, чем разово превысить порог."""
    assert rotate({0: 0}, 5000, limit=1000)[0] == 5000


def test_rotate_does_not_mutate_the_store_it_was_given():
    store = {0: 990}
    rotate(store, 50, limit=1000)
    assert store == {0: 990}


def test_negative_incoming_is_rejected():
    with pytest.raises(ValueError):
        rotate({0: 0}, -1)


# ---------------------------------------------------------- run_feedback_loop
def green_runner(turn, state):
    return {"command": ["pytest"], "exit_code": 0, "stdout": "1 passed", "duration_ms": 5}


def red_runner(turn, state):
    return {"command": ["pytest"], "exit_code": 1, "stderr": "1 failed", "duration_ms": 5}


def flaky_runner(turn, state):
    """Красный на первой попытке, зелёный после того, как state починили."""
    if state == "fixed":
        return {"command": ["pytest"], "exit_code": 0, "stdout": "1 passed", "duration_ms": 5}
    return {"command": ["pytest"], "exit_code": 1, "stderr": "1 failed", "duration_ms": 5}


def test_green_run_stops_the_loop_after_one_attempt():
    result = run_feedback_loop(green_runner, lambda r, s: "fixed", "жду зелёный", make_clock())
    assert result["status"] == "passed"
    assert result["turns"] == 1


def test_a_fixed_signal_makes_the_second_attempt_pass():
    result = run_feedback_loop(flaky_runner, lambda r, s: "fixed", "чиню", make_clock())
    assert result["status"] == "passed"
    assert result["turns"] == 2
    assert [r["exit_code"] for r in result["records"]] == [1, 0]


def test_an_unfixable_signal_stops_the_loop_instead_of_spinning():
    """Главное свойство: fixer сдался — петля останавливается на первой же попытке."""
    result = run_feedback_loop(red_runner, lambda r, s: None, "чиню", make_clock())
    assert result["status"] == "stuck"
    assert result["turns"] == 1


def test_a_loop_that_never_goes_green_terminates_at_the_turn_budget():
    result = run_feedback_loop(red_runner, lambda r, s: "again", "чиню", make_clock())
    assert result["status"] == "exhausted"
    assert result["turns"] == MAX_TURNS


def test_a_missing_exit_code_blocks_the_loop_and_never_reads_as_success():
    def broken_runner(turn, state):
        return {"command": ["pytest"], "exit_code": None, "error": "timeout", "duration_ms": 0}

    result = run_feedback_loop(broken_runner, lambda r, s: "fixed", "чиню", make_clock())
    assert result["status"] == "blocked"
    assert result["turns"] == 1


def test_the_loop_leaves_a_readable_retry_chain():
    result = run_feedback_loop(red_runner, lambda r, s: "again", "чиню", make_clock())
    last = result["records"][-1]["command_id"]
    assert retry_chain(result["records"], last) == result["records"]


def test_every_attempt_gets_its_own_timestamp_from_the_clock():
    result = run_feedback_loop(red_runner, lambda r, s: "again", "чиню", make_clock(500, 100))
    expected = [500 + 100 * i for i in range(MAX_TURNS)]
    assert [r["started_at"] for r in result["records"]] == expected
