"""Тесты к уроку «Ландшафт кодовых агентов». Правь exercise.py."""

import pytest

from exercise import (
    blast_radius,
    compare_axes,
    pass_rate,
    rank_agents,
    rank_changes,
    scaffold_delta,
    score_excluding_easy,
    simulate_scaffold,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def task(lines, solved):
    return {"id": f"t{lines}-{solved}", "lines": lines, "solved": solved}


# --------------------------------------------------------------- pass_rate
def test_pass_rate_counts_solved_over_total():
    assert pass_rate([task(1, True), task(1, False)]) == APPROX(0.5)


def test_pass_rate_of_an_empty_suite_is_zero_not_a_crash():
    assert pass_rate([]) == APPROX(0.0)


def test_pass_rate_of_all_solved_is_one():
    assert pass_rate([task(3, True) for _ in range(7)]) == APPROX(1.0)


def test_pass_rate_ignores_how_big_the_change_was():
    """Сам по себе pass_rate слеп к сложности задачи — в этом и проблема."""
    easy = [task(1, True), task(1, True), task(40, False)]
    assert pass_rate(easy) == APPROX(2 / 3)


# ------------------------------------------------------ score_excluding_easy
def test_score_excluding_easy_drops_the_one_line_tail():
    tasks = [task(1, True), task(1, True), task(40, False)]
    assert score_excluding_easy(tasks, 10) == APPROX(0.0)


def test_score_excluding_easy_is_lower_when_the_tail_carried_the_score():
    """Ровно эффект SWE-bench Verified против SWE-bench Pro."""
    tasks = [task(1, True)] * 8 + [task(30, True), task(30, False)]
    assert score_excluding_easy(tasks, 10) < pass_rate(tasks)


def test_score_excluding_easy_keeps_tasks_exactly_at_the_boundary():
    assert score_excluding_easy([task(10, True), task(9, False)], 10) == APPROX(1.0)


def test_score_excluding_easy_with_nothing_left_is_zero():
    assert score_excluding_easy([task(1, True), task(2, True)], 10) == APPROX(0.0)


def test_score_excluding_easy_with_zero_threshold_equals_pass_rate():
    tasks = [task(1, True), task(30, False), task(12, True)]
    assert score_excluding_easy(tasks, 0) == APPROX(pass_rate(tasks))


# -------------------------------------------------------------- rank_agents
def test_rank_agents_sorts_from_best_to_worst():
    assert rank_agents({"aider": 0.5, "cline": 0.6, "swe-agent": 0.43}) == [
        ("cline", 0.6),
        ("aider", 0.5),
        ("swe-agent", 0.43),
    ]


def test_rank_agents_breaks_ties_alphabetically_and_deterministically():
    """Без tie-break «поднялся на позицию» ничего не значит."""
    first = rank_agents({"b": 0.5, "a": 0.5, "c": 0.5})
    second = rank_agents({"c": 0.5, "a": 0.5, "b": 0.5})
    assert first == second == [("a", 0.5), ("b", 0.5), ("c", 0.5)]


def test_rank_agents_of_an_empty_board_is_empty():
    assert rank_agents({}) == []


def test_rank_agents_keeps_the_scores_alongside_the_names():
    assert rank_agents({"solo": 0.42})[0][1] == APPROX(0.42)


# ------------------------------------------------------------ rank_changes
def test_rank_changes_reports_positive_for_a_climb():
    before = rank_agents({"a": 1.0, "b": 0.0})
    after = rank_agents({"a": 0.0, "b": 1.0})
    assert rank_changes(before, after) == {"a": -1, "b": 1}


def test_rank_changes_is_all_zeros_when_nothing_moved():
    board = rank_agents({"a": 1.0, "b": 0.5, "c": 0.1})
    assert set(rank_changes(board, board).values()) == {0}


def test_rank_changes_skips_agents_missing_from_one_side():
    before = rank_agents({"a": 1.0, "gone": 0.5})
    after = rank_agents({"a": 1.0, "new": 0.5})
    assert rank_changes(before, after) == {"a": 0}


def test_rank_changes_shows_the_leaderboard_shuffle_after_dropping_easy_tasks():
    """Исключили лёгкий хвост — и лидер поменялся."""
    tail_lover = [task(1, True)] * 8 + [task(30, False)] * 2
    hard_worker = [task(1, False)] * 8 + [task(30, True)] * 2
    before = rank_agents(
        {"tail": pass_rate(tail_lover), "hard": pass_rate(hard_worker)}
    )
    after = rank_agents(
        {
            "tail": score_excluding_easy(tail_lover, 10),
            "hard": score_excluding_easy(hard_worker, 10),
        }
    )
    assert rank_changes(before, after)["hard"] == 1


# ---------------------------------------------------------- scaffold_delta
def test_scaffold_delta_is_the_headline_number_of_the_lesson():
    assert scaffold_delta(43.2, 59.8) == pytest.approx(16.6, abs=1e-9)


def test_scaffold_delta_is_signed_when_the_scaffold_hurts():
    assert scaffold_delta(59.8, 43.2) == pytest.approx(-16.6, abs=1e-9)


def test_scaffold_delta_is_points_not_a_ratio():
    """40 -> 80 это +40 пунктов, а не +100 процентов. Разные утверждения."""
    assert scaffold_delta(40.0, 80.0) == APPROX(40.0)


# -------------------------------------------------------- simulate_scaffold
def test_simulate_scaffold_json_tool_calls_take_one_turn_per_file():
    assert simulate_scaffold(3, 1) == {"turns": 4, "blast": 1}


def test_simulate_scaffold_codeact_compresses_edits_into_fewer_turns():
    assert simulate_scaffold(3, 3)["turns"] < simulate_scaffold(3, 1)["turns"]


def test_simulate_scaffold_pays_for_speed_with_blast_radius():
    """Меньше ходов — больше файлов под одним непроверенным действием."""
    json_like = simulate_scaffold(6, 1)
    codeact = simulate_scaffold(6, 6)
    assert codeact["turns"] < json_like["turns"]
    assert codeact["blast"] > json_like["blast"]


def test_simulate_scaffold_always_spends_a_final_done_turn():
    assert simulate_scaffold(0, 5) == {"turns": 1, "blast": 0}


def test_simulate_scaffold_blast_is_observed_not_advertised():
    """Ёмкость 10 на двух багах трогает два файла — в отчёте о риске это важно."""
    assert simulate_scaffold(2, 10)["blast"] == 2


def test_simulate_scaffold_rejects_a_nonpositive_capacity():
    with pytest.raises(ValueError):
        simulate_scaffold(3, 0)


# ------------------------------------------------------------ blast_radius
def test_blast_radius_takes_the_worst_single_action():
    actions = [{"files": ["a.py"]}, {"files": ["a.py", "b.py"]}, {"files": []}]
    assert blast_radius(actions) == 2


def test_blast_radius_is_a_max_not_a_sum():
    """Аудит спрашивает «что успеет одно действие», а не «сколько за сессию»."""
    actions = [{"files": ["a.py"]} for _ in range(50)]
    assert blast_radius(actions) == 1


def test_blast_radius_of_an_empty_trace_is_zero():
    assert blast_radius([]) == 0


# ------------------------------------------------------------- compare_axes
def test_compare_axes_names_the_winner_per_axis():
    a = {"retrieval": 3, "verifier": 1}
    b = {"retrieval": 1, "verifier": 2}
    assert compare_axes(a, b) == {"retrieval": "a", "verifier": "b"}


def test_compare_axes_reports_a_tie():
    assert compare_axes({"isolation": 2}, {"isolation": 2}) == {"isolation": "tie"}


def test_compare_axes_calls_an_unmeasured_axis_unknown_not_a_loss():
    """Неизмеренная ось — не нулевая ось. Подставлять ноль значит выдумывать."""
    assert compare_axes({"retrieval": 1}, {"verifier": 1}) == {
        "retrieval": "unknown",
        "verifier": "unknown",
    }


def test_compare_axes_returns_axes_in_alphabetical_order():
    a = {"verifier": 1, "isolation": 1, "retrieval": 1}
    b = {"verifier": 0, "isolation": 0, "retrieval": 0}
    assert list(compare_axes(a, b)) == ["isolation", "retrieval", "verifier"]


def test_compare_axes_is_antisymmetric():
    a = {"retrieval": 3, "verifier": 1, "isolation": 2}
    b = {"retrieval": 1, "verifier": 2, "isolation": 2}
    flip = {"a": "b", "b": "a", "tie": "tie", "unknown": "unknown"}
    forward = compare_axes(a, b)
    assert compare_axes(b, a) == {k: flip[v] for k, v in forward.items()}
