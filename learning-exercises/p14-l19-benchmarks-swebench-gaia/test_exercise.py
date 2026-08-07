"""Тесты к уроку «Бенчмарки: SWE-bench, GAIA, AgentBench». Правь exercise.py."""

import random

import pytest

from exercise import (
    FAILED,
    PASSED,
    clean_resolve_rate,
    contaminated_ids,
    gaia_level,
    is_resolved,
    parse_patch,
    pass_at_k,
    resolve_rate,
    solution_leakage,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

PATCH = (
    "--- a/calc.py\n"
    "+++ b/calc.py\n"
    "@@ -1,3 +1,3 @@\n"
    " def double(value):\n"
    "-    return value\n"
    "+    return value * 2\n"
)


# ------------------------------------------------------------- parse_patch
def test_parse_patch_splits_added_and_removed():
    assert parse_patch(PATCH) == {
        "calc.py": {"added": ["    return value * 2"],
                    "removed": ["    return value"]}
    }


def test_parse_patch_does_not_read_the_header_as_a_changed_line():
    """"+++ b/calc.py" начинается с "+", но это путь, а не добавленная строка."""
    added = parse_patch(PATCH)["calc.py"]["added"]
    assert all("calc.py" not in line for line in added)


def test_parse_patch_ignores_context_lines():
    """Строка " def double(value):" идёт без изменений и не попадает никуда."""
    changes = parse_patch(PATCH)["calc.py"]
    assert "def double(value):" not in changes["added"]
    assert "def double(value):" not in changes["removed"]


def test_parse_patch_handles_several_files():
    multi = PATCH + "+++ b/util.py\n@@ -0,0 +1 @@\n+import math\n"
    assert sorted(parse_patch(multi)) == ["calc.py", "util.py"]


def test_parse_patch_of_empty_diff_is_empty():
    assert parse_patch("") == {}


# -------------------------------------------------------------- is_resolved
def test_is_resolved_needs_both_gates_green():
    results = {"bug_fixed": PASSED, "old_feature": PASSED}
    assert is_resolved(results, ["bug_fixed"], ["old_feature"]) is True


def test_is_resolved_rejects_a_patch_that_does_not_fix_the_bug():
    results = {"bug_fixed": FAILED, "old_feature": PASSED}
    assert is_resolved(results, ["bug_fixed"], ["old_feature"]) is False


def test_is_resolved_rejects_a_patch_that_breaks_a_passing_test():
    """PASS_TO_PASS — второе горло: починил баг, сломал соседа — не засчитано."""
    results = {"bug_fixed": PASSED, "old_feature": FAILED}
    assert is_resolved(results, ["bug_fixed"], ["old_feature"]) is False


def test_is_resolved_treats_a_missing_test_as_failed():
    """Тест, который не запустился, — это не "прошёл"."""
    assert is_resolved({"bug_fixed": PASSED}, ["bug_fixed"], ["never_ran"]) is False


def test_is_resolved_refuses_a_task_without_fail_to_pass():
    with pytest.raises(ValueError):
        is_resolved({"a": PASSED}, [], ["a"])


# ------------------------------------------------------------ resolve_rate
def test_resolve_rate_counts_the_solved_share():
    assert resolve_rate([("t1", True), ("t2", False)]) == APPROX(0.5)


def test_resolve_rate_of_no_tasks_is_zero():
    assert resolve_rate([]) == APPROX(0.0)


def test_resolve_rate_does_not_depend_on_task_order():
    rng = random.Random(17)
    tasks = [(f"t{i}", i % 3 == 0) for i in range(30)]
    shuffled = list(tasks)
    rng.shuffle(shuffled)
    assert resolve_rate(shuffled) == APPROX(resolve_rate(tasks))


def test_resolve_rate_refuses_a_duplicated_task_id():
    with pytest.raises(ValueError):
        resolve_rate([("t1", True), ("t1", False)])


# --------------------------------------------------------- solution_leakage
def test_solution_leakage_spots_the_fix_pasted_into_the_issue():
    issue = "Doubling is broken. The fix is:     return value * 2"
    assert solution_leakage(issue, PATCH) == APPROX(1.0)


def test_solution_leakage_is_zero_for_an_honest_issue():
    issue = "double() returns the input unchanged for every argument"
    assert solution_leakage(issue, PATCH) == APPROX(0.0)


def test_solution_leakage_ignores_trivial_short_lines():
    """Строка "}" совпадёт с чем угодно, поэтому в знаменатель не идёт."""
    noise = "+++ b/x.c\n+}\n+  {\n"
    assert solution_leakage("the issue text mentions } and {", noise) == APPROX(0.0)


def test_solution_leakage_of_an_empty_patch_is_zero_not_a_crash():
    assert solution_leakage("anything at all", "") == APPROX(0.0)


# --------------------------------------------------------- contaminated_ids
def test_contaminated_ids_flags_the_leaked_task():
    tasks = [("t1", "fix: return value * 2", PATCH),
             ("t2", "double() is broken", PATCH)]
    assert contaminated_ids(tasks) == {"t1"}


def test_contaminated_ids_includes_the_task_exactly_at_the_threshold():
    """Сравнение >=, а не >: при пороге 1.0 полностью списанный патч — грязный."""
    tasks = [("t1", "fix: return value * 2", PATCH)]
    assert contaminated_ids(tasks, 1.0) == {"t1"}


def test_contaminated_ids_of_no_tasks_is_empty():
    assert contaminated_ids([]) == set()


# ------------------------------------------------------- clean_resolve_rate
def test_clean_resolve_rate_drops_the_dirty_task_from_the_denominator():
    """Грязная решённая задача уходит и из числителя, и из знаменателя."""
    report = clean_resolve_rate([("t1", True), ("t2", True), ("t3", False)], {"t2"})
    assert report["evaluated"] == 2
    assert report["rate"] == APPROX(0.5)


def test_clean_resolve_rate_reports_how_many_were_excluded():
    """Сколько выкинули — часть отчёта: 50% по 500 задачам и по 12 не равны."""
    report = clean_resolve_rate([("t1", True), ("t2", True)], {"t2"})
    assert report["excluded"] == 1


def test_clean_resolve_rate_ignores_contaminated_ids_that_are_not_in_the_run():
    report = clean_resolve_rate([("t1", True)], {"t9"})
    assert (report["excluded"], report["evaluated"]) == (0, 1)


def test_clean_resolve_rate_of_a_fully_contaminated_run_is_zero():
    report = clean_resolve_rate([("t1", True)], {"t1"})
    assert (report["rate"], report["evaluated"]) == (APPROX(0.0), 0)


# ---------------------------------------------------------------- pass_at_k
def test_pass_at_k_with_one_attempt_is_the_sample_share():
    assert pass_at_k(10, 1, 1) == APPROX(0.1)


def test_pass_at_k_reaches_one_when_all_attempts_are_drawn():
    assert pass_at_k(10, 1, 10) == APPROX(1.0)


def test_pass_at_k_is_zero_when_nothing_was_solved():
    assert pass_at_k(10, 0, 5) == APPROX(0.0)


def test_pass_at_k_never_decreases_as_k_grows():
    """Определяющее свойство метрики: больше попыток — не хуже результат."""
    values = [pass_at_k(20, 3, k) for k in range(1, 21)]
    assert all(a <= b + 1e-12 for a, b in zip(values, values[1:]))


def test_pass_at_k_refuses_k_larger_than_n():
    with pytest.raises(ValueError):
        pass_at_k(5, 2, 6)


# -------------------------------------------------------------- gaia_level
def test_gaia_level_of_a_one_hop_question_is_one():
    assert gaia_level("What is the capital of France?") == 1


def test_gaia_level_of_a_two_tool_question_is_two():
    assert gaia_level("Search the paper and extract the first author.") == 2


def test_gaia_level_of_a_long_multimodal_chain_is_three():
    q = ("Visit the arXiv listing, find the chart in the pdf, then extract "
         "the audio caption and finally search for the video mirror.")
    assert gaia_level(q) == 3


def test_gaia_level_never_drops_when_more_hops_are_added():
    """Монотонность: добавили шаг — сложность не уменьшилась."""
    base = "Find the author."
    longer = base + " Then find the video. Next extract the chart."
    assert gaia_level(longer) >= gaia_level(base)


def test_gaia_level_does_not_match_step_words_inside_longer_words():
    """"and" сидит внутри "Andrew" — как подстрока это ложный шаг."""
    assert gaia_level("Who is Andrew?") == gaia_level("Who is Bob?")
