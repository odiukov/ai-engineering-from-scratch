"""Тесты к уроку «Reflexion: обучение словами вместо градиентов». Правь exercise.py."""

from exercise import (
    actor,
    add_reflection,
    binary_evaluator,
    expire_reflections,
    heuristic_evaluator,
    memory_prompt,
    reflect,
    run_reflexion,
)

TARGET = 20


# ------------------------------------------------------------ binary_evaluator
def test_binary_evaluator_marks_an_exact_hit_as_success():
    assert binary_evaluator([9, 9, 2], TARGET) == (True, 0)


def test_binary_evaluator_signs_the_delta_negative_when_undershooting():
    ok, delta = binary_evaluator([1, 1, 1], TARGET)
    assert ok is False
    assert delta == -17


def test_binary_evaluator_signs_the_delta_positive_when_overshooting():
    assert binary_evaluator([9, 9, 9], TARGET) == (False, 7)


# --------------------------------------------------------- heuristic_evaluator
def test_heuristic_evaluator_stays_silent_on_a_healthy_trajectory():
    assert heuristic_evaluator(["look", "take", "put"]) == []


def test_heuristic_evaluator_detects_a_stuck_loop():
    flags = heuristic_evaluator(["look", "look"])
    assert len(flags) == 1
    assert flags[0].startswith("stuck:")


def test_heuristic_evaluator_detects_an_inefficient_trajectory():
    flags = heuristic_evaluator(["a", "b", "c", "d", "e", "f"], max_steps=5)
    assert any(f.startswith("inefficient:") for f in flags)


def test_heuristic_evaluator_needs_no_ground_truth_to_fire():
    """Эвристике не нужен ни целевой ответ, ни оценка модели — только траектория."""
    assert heuristic_evaluator(["a", "a", "b", "c", "d", "e", "f"], max_steps=5) != []


# ------------------------------------------------------------------- reflect
def test_reflect_says_how_much_was_missing():
    assert reflect([1, 1, 1], -17) == "сумма 3 меньше цели на 17: бери числа крупнее"


def test_reflect_says_how_much_was_over():
    assert reflect([9, 9, 9], 7) == "сумма 27 больше цели на 7: бери числа мельче"


def test_reflect_on_success_does_not_invent_a_problem():
    assert reflect([9, 9, 2], 0) == "получилось"


def test_reflect_carries_the_number_the_next_attempt_needs():
    """Рефлексия без конкретики бесполезна — в тексте обязан быть размер промаха."""
    assert "17" in reflect([1, 1, 1], -17)


# --------------------------------------------------------------- add_reflection
def test_add_reflection_appends_to_the_end():
    assert add_reflection([{"trial": 1}], {"trial": 2}) == [{"trial": 1}, {"trial": 2}]


def test_add_reflection_does_not_mutate_the_input_memory():
    memory = [{"trial": 1}]
    add_reflection(memory, {"trial": 2})
    assert memory == [{"trial": 1}]


def test_add_reflection_evicts_the_oldest_entry_when_full():
    memory = [{"trial": i} for i in range(1, 7)]
    updated = add_reflection(memory, {"trial": 7}, max_len=6)
    assert len(updated) == 6
    assert updated[0] == {"trial": 2}
    assert updated[-1] == {"trial": 7}


def test_add_reflection_never_exceeds_the_limit_however_many_are_added():
    memory = []
    for i in range(50):
        memory = add_reflection(memory, {"trial": i}, max_len=6)
    assert len(memory) == 6


# ------------------------------------------------------------ expire_reflections
def test_expire_reflections_drops_entries_older_than_the_ttl():
    assert expire_reflections([{"trial": 1}, {"trial": 9}], now=10, ttl=3) == [{"trial": 9}]


def test_expire_reflections_keeps_an_entry_exactly_at_the_ttl_boundary():
    assert expire_reflections([{"trial": 7}], now=10, ttl=3) == [{"trial": 7}]


def test_expire_reflections_depends_on_the_now_argument_not_on_the_wall_clock():
    """Одна и та же память при разном now даёт разный результат — время параметр."""
    memory = [{"trial": 5}]
    assert expire_reflections(memory, now=6, ttl=3) == memory
    assert expire_reflections(memory, now=20, ttl=3) == []


def test_expire_reflections_leaves_a_fresh_buffer_alone():
    memory = [{"trial": 8}, {"trial": 9}, {"trial": 10}]
    assert expire_reflections(memory, now=10, ttl=5) == memory


# --------------------------------------------------------------- memory_prompt
def test_memory_prompt_marks_an_empty_buffer_explicitly():
    assert memory_prompt([]) == "(нет прошлых рефлексий)"


def test_memory_prompt_lists_one_line_per_reflection():
    prompt = memory_prompt([{"trial": 1, "text": "недобрал"}, {"trial": 2, "text": "перебрал"}])
    assert prompt.splitlines() == ["- попытка 1: недобрал", "- попытка 2: перебрал"]


def test_memory_prompt_keeps_the_trial_number_visible():
    assert "3" in memory_prompt([{"trial": 3, "text": "мимо"}])


# --------------------------------------------------------------------- actor
def test_actor_without_memory_gives_the_naive_guess():
    assert actor([]) == [1, 1, 1]


def test_actor_uses_the_last_reflection_to_correct_the_sum():
    assert sum(actor([{"attempt": [1, 1, 1], "delta": -17}])) == TARGET


def test_actor_stays_inside_the_allowed_range():
    attempt = actor([{"attempt": [1, 1, 1], "delta": -100}])
    assert all(1 <= x <= 9 for x in attempt)


def test_actor_moves_down_when_the_reflection_says_it_overshot():
    assert sum(actor([{"attempt": [9, 9, 9], "delta": 7}])) == TARGET


def test_actor_is_deterministic_for_the_same_memory():
    memory = [{"attempt": [1, 1, 1], "delta": -17}]
    assert actor(memory) == actor(memory)


# -------------------------------------------------------------- run_reflexion
def test_run_reflexion_converges_when_memory_is_on():
    trials = run_reflexion(TARGET, max_trials=4, use_memory=True)
    assert trials[-1]["success"] is True


def test_run_reflexion_stops_as_soon_as_it_succeeds():
    trials = run_reflexion(TARGET, max_trials=8, use_memory=True)
    assert len(trials) == 2
    assert all(not t["success"] for t in trials[:-1])


def test_run_reflexion_without_memory_never_adapts():
    """Базовая линия из урока: без рефлексии в промпте актёр повторяет себя."""
    trials = run_reflexion(TARGET, max_trials=4, use_memory=False)
    assert len(trials) == 4
    assert all(t["attempt"] == trials[0]["attempt"] for t in trials)
    assert not any(t["success"] for t in trials)


def test_run_reflexion_second_attempt_is_closer_than_the_first():
    """Смысл verbal RL: одна словесная рефлексия уже уменьшает промах."""
    trials = run_reflexion(TARGET, max_trials=4, use_memory=True)
    assert abs(trials[1]["delta"]) < abs(trials[0]["delta"])


def test_run_reflexion_records_a_reflection_for_every_trial():
    trials = run_reflexion(TARGET, max_trials=4, use_memory=False)
    assert all(t["reflection"] for t in trials)
