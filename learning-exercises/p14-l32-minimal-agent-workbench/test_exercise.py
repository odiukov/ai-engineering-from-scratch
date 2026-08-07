"""Тесты к уроку «Минимальный воркбенч агента». Правь exercise.py."""

import pytest

from exercise import (
    BOARD_FILE,
    ROUTER_FILE,
    STATE_FILE,
    board_summary,
    lint_router,
    next_task,
    pull_task,
    router_links,
    run_session,
    run_turn,
)

GOOD_ROUTER = """# AGENTS.md

Читай перед работой:

1. `agent_state.json` — где остановилась прошлая сессия.
2. `task_board.json` — что в работе и что дальше.
3. `docs/agent-rules.md` — правила, загружаются по требованию.

Verification command: python3 -m pytest -x
"""

GOOD_FS = {
    ROUTER_FILE: GOOD_ROUTER,
    STATE_FILE: "{}",
    BOARD_FILE: "[]",
    "docs/agent-rules.md": "# rules",
}


def fresh_state():
    return {"active_task_id": None, "touched_files": [], "next_action": ""}


def fresh_board():
    return [
        {"id": "T-001", "goal": "валидация /signup", "status": "todo", "priority": 1},
        {"id": "T-002", "goal": "описать контракт", "status": "todo", "priority": 5},
    ]


# ------------------------------------------------------------ router_links
def test_router_links_finds_the_three_files():
    assert router_links(GOOD_ROUTER) == [
        STATE_FILE,
        BOARD_FILE,
        "docs/agent-rules.md",
    ]


def test_expressions_in_backticks_are_not_links():
    """`state.active_task_id` — выражение, а не файл."""
    assert router_links("проверь `state.active_task_id`") == []


def test_commands_with_spaces_are_not_links():
    assert router_links("запусти `python3 -m pytest -x`") == []


def test_router_links_deduplicate_but_keep_order():
    text = "`b.json` потом `a.md` потом снова `b.json`"
    assert router_links(text) == ["b.json", "a.md"]


# ------------------------------------------------------------- lint_router
def test_a_healthy_router_has_no_problems():
    assert lint_router(GOOD_ROUTER, GOOD_FS) == []


def test_broken_link_is_reported_with_its_path():
    """Ссылка на несуществующий файл хуже отсутствующего правила."""
    fs = dict(GOOD_FS)
    del fs["docs/agent-rules.md"]
    assert "broken_link:docs/agent-rules.md" in lint_router(GOOD_ROUTER, fs)


def test_router_without_state_and_board_links_is_incomplete():
    problems = lint_router("# AGENTS.md\n\nVerification command: pytest\n", GOOD_FS)
    assert "missing_state_link" in problems
    assert "missing_board_link" in problems


def test_router_without_verification_command_is_flagged():
    text = GOOD_ROUTER.replace("Verification command: python3 -m pytest -x", "тесты есть")
    assert "no_verification" in lint_router(text, GOOD_FS)


def test_long_router_is_flagged_because_nobody_reads_it():
    text = GOOD_ROUTER + "\n" * 60
    assert "too_long" in lint_router(text, GOOD_FS)


def test_lint_output_is_stable_across_calls():
    """Одинаковый вход — одинаковый вывод, иначе отчёт шумит в diff."""
    fs = {ROUTER_FILE: GOOD_ROUTER}
    assert lint_router(GOOD_ROUTER, fs) == lint_router(GOOD_ROUTER, fs)


# --------------------------------------------------------------- next_task
def test_next_task_prefers_the_highest_priority():
    assert next_task(fresh_board())["id"] == "T-002"


def test_ties_are_broken_by_position_on_the_board():
    board = [
        {"id": "T-001", "status": "todo", "priority": 3},
        {"id": "T-002", "status": "todo", "priority": 3},
    ]
    assert next_task(board)["id"] == "T-001"


def test_missing_priority_counts_as_zero():
    board = [
        {"id": "T-001", "status": "todo"},
        {"id": "T-002", "status": "todo", "priority": 1},
    ]
    assert next_task(board)["id"] == "T-002"


def test_only_todo_tasks_are_pullable():
    board = [
        {"id": "T-001", "status": "in_progress", "priority": 9},
        {"id": "T-002", "status": "done", "priority": 9},
    ]
    assert next_task(board) is None


def test_next_task_does_not_change_the_board():
    board = fresh_board()
    next_task(board)
    assert [t["status"] for t in board] == ["todo", "todo"]


# --------------------------------------------------------------- pull_task
def test_pulling_moves_the_task_to_in_progress():
    state, board = pull_task(fresh_state(), fresh_board())
    assert state["active_task_id"] == "T-002"
    assert board[1]["status"] == "in_progress"


def test_pull_does_not_mutate_the_inputs():
    """Читаем старый файл, пишем новый — как атомарная запись на диск."""
    state, board = fresh_state(), fresh_board()
    pull_task(state, board)
    assert state["active_task_id"] is None
    assert [t["status"] for t in board] == ["todo", "todo"]


def test_a_busy_agent_does_not_pull_a_second_task():
    """Два незакрытых задания одновременно и есть потеря фокуса."""
    state = fresh_state()
    state["active_task_id"] = "T-001"
    new_state, new_board = pull_task(state, fresh_board())
    assert new_state["active_task_id"] == "T-001"
    assert [t["status"] for t in new_board] == ["todo", "todo"]


def test_empty_board_leaves_the_agent_idle():
    state, board = pull_task(fresh_state(), [])
    assert state["active_task_id"] is None
    assert "idle" in state["next_action"]


# ---------------------------------------------------------------- run_turn
def test_first_turn_pulls_work_from_the_board():
    state, _ = run_turn(fresh_state(), fresh_board(), ["app.py"])
    assert state["active_task_id"] == "T-002"


def test_a_turn_touches_exactly_one_file():
    state = fresh_state()
    state["active_task_id"] = "T-001"
    board = [{"id": "T-001", "goal": "g", "status": "in_progress"}]
    state, _ = run_turn(state, board, ["app.py", "test_app.py"])
    assert state["touched_files"] == ["app.py"]


def test_task_closes_once_every_allowed_file_is_touched():
    state = fresh_state()
    state["active_task_id"] = "T-001"
    state["touched_files"] = ["app.py"]
    board = [{"id": "T-001", "goal": "g", "status": "in_progress"}]
    state, board = run_turn(state, board, ["app.py"])
    assert board[0]["status"] == "done"
    assert state["active_task_id"] is None
    assert state["touched_files"] == []


def test_active_task_missing_from_the_board_resets_the_agent():
    """Задачу удалили руками — не притворяемся, что работаем над ней."""
    state = fresh_state()
    state["active_task_id"] = "T-404"
    state, board = run_turn(state, [], ["app.py"])
    assert state["active_task_id"] is None


def test_run_turn_does_not_mutate_the_inputs():
    state, board = fresh_state(), fresh_board()
    run_turn(state, board, ["app.py"])
    assert state == fresh_state()
    assert board == fresh_board()


# ------------------------------------------------------------- run_session
def test_zero_turns_change_nothing():
    state, board = run_session(fresh_state(), fresh_board(), ["app.py"], 0)
    assert (state, board) == (fresh_state(), fresh_board())


def test_a_session_resumes_exactly_where_the_previous_one_stopped():
    """Шесть ходов подряд == две сессии по три: состояние живёт в файле."""
    files = ["app.py", "test_app.py"]
    one_go = run_session(fresh_state(), fresh_board(), files, 6)
    first, board = run_session(fresh_state(), fresh_board(), files, 3)
    resumed = run_session(first, board, files, 3)
    assert resumed == one_go


def test_a_long_enough_session_empties_the_board():
    files = ["app.py"]
    _, board = run_session(fresh_state(), fresh_board(), files, 10)
    assert board_summary(board)["done"] == 2


def test_tasks_are_finished_in_priority_order():
    files = ["app.py"]
    _, board = run_session(fresh_state(), fresh_board(), files, 3)
    done = [t["id"] for t in board if t["status"] == "done"]
    assert done == ["T-002"]


# ----------------------------------------------------------- board_summary
def test_empty_board_summary_still_has_every_status():
    assert board_summary([]) == {
        "todo": 0,
        "in_progress": 0,
        "done": 0,
        "blocked": 0,
    }


def test_board_summary_counts_each_status():
    board = [
        {"id": "T-001", "status": "todo"},
        {"id": "T-002", "status": "done"},
        {"id": "T-003", "status": "done"},
    ]
    summary = board_summary(board)
    assert (summary["todo"], summary["done"], summary["blocked"]) == (1, 2, 0)


def test_invented_status_is_refused():
    """«Почти done» означает, что кто-то придумал статус в обход схемы."""
    with pytest.raises(ValueError):
        board_summary([{"id": "T-001", "status": "almost_done"}])
