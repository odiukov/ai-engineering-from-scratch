"""Тесты к уроку «Hogwild! Inference: воркеры на общем кеше». Правь exercise.py."""

import pytest

from exercise import (
    amdahl_time,
    best_worker_count,
    hogwild_speedup,
    hogwild_time,
    next_category,
    run_hogwild,
    useful_work,
    visible_counts,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

QUOTA = 10  # столько токенов закрывают одну подцель


# --------------------------------------------------------------- amdahl_time
def test_amdahl_splits_only_the_parallel_part():
    assert amdahl_time(10000, 0.7, 4) == APPROX(4750.0)


def test_one_worker_takes_the_serial_time():
    assert amdahl_time(10000, 0.7, 1) == APPROX(10000.0)


def test_a_fully_parallel_task_divides_cleanly():
    assert amdahl_time(10000, 1.0, 4) == APPROX(2500.0)


def test_the_serial_part_is_a_floor_no_worker_count_can_break():
    """Сколько воркеров ни дай, ниже t * (1 - p) не упасть."""
    assert amdahl_time(10000, 0.7, 1000) > 3000


# -------------------------------------------------------------- hogwild_time
def test_hogwild_adds_coordination_on_top_of_amdahl():
    assert hogwild_time(10000, 0.7, 4, 200) == APPROX(5550.0)


def test_free_coordination_is_plain_amdahl():
    assert hogwild_time(10000, 0.7, 4, 0) == APPROX(amdahl_time(10000, 0.7, 4))


def test_coordination_cost_grows_with_the_worker_count():
    cheap = hogwild_time(10000, 0.7, 4, 10)
    dear = hogwild_time(10000, 0.7, 4, 500)
    assert dear - cheap == APPROX(4 * (500 - 10))


# ------------------------------------------------------------ hogwild_speedup
def test_long_reasoning_task_is_a_win():
    """Пример из урока: 10k токенов, p=0.7, c=200, четыре воркера."""
    assert hogwild_speedup(10000, 0.7, 4, 200) == pytest.approx(1.80, abs=0.01)


def test_short_chat_task_is_a_loss():
    """Меньше единицы — координация съела больше, чем дала параллельность."""
    assert hogwild_speedup(1000, 0.3, 4, 200) < 1.0


def test_speedup_rises_when_the_task_parallelizes_better():
    assert hogwild_speedup(10000, 0.9, 4, 200) > hogwild_speedup(10000, 0.5, 4, 200)


# --------------------------------------------------------- best_worker_count
def test_optimum_sits_inside_the_range_not_at_its_edge():
    """Выигрыш падает как 1/n, а расход растёт как n — оптимум всегда внутри."""
    assert best_worker_count(10000, 0.7, 200, 16) == 6


def test_expensive_coordination_leaves_a_single_worker():
    assert best_worker_count(1000, 0.3, 500, 8) == 1


def test_best_worker_count_never_exceeds_the_limit():
    assert best_worker_count(1_000_000, 0.99, 1, 8) <= 8


# ------------------------------------------------------------ visible_counts
def test_visible_counts_tallies_by_category():
    assert visible_counts([(0, 0, 1, 0), (0, 1, 1, 1)], 3) == [0, 2, 0]


def test_lag_hides_the_freshest_writes():
    assert visible_counts([(0, 0, 1, 0), (0, 1, 1, 1)], 3, 1) == [0, 1, 0]


def test_lag_longer_than_the_cache_shows_nothing():
    """Срез с конца тут даёт пустоту, а не хвост — на этом легко ошибиться."""
    assert visible_counts([(0, 0, 1, 0)], 3, 9) == [0, 0, 0]


# ------------------------------------------------------------- next_category
def test_empty_cache_sends_everyone_to_the_first_subgoal():
    assert next_category([0, 0, 0], 1.0) == 0


def test_a_busy_subgoal_pushes_the_worker_elsewhere():
    assert next_category([2, 0, 0], 1.0) == 1


def test_without_coordination_the_worker_ignores_the_cache():
    assert next_category([2, 0, 0], 0.0) == 0
    assert next_category([99, 0, 0], 0.0) == 0


def test_the_worker_returns_to_the_first_subgoal_once_load_is_even():
    assert next_category([1, 1, 1], 1.0) == 0


# --------------------------------------------------------------- run_hogwild
def test_every_worker_writes_once_per_round():
    assert len(run_hogwild(2, 20, 4)) == 40


def test_a_single_worker_walks_the_subgoals_in_turn():
    assert run_hogwild(1, 2, 2) == [(0, 0, 0, 0), (1, 0, 1, 0)]


def test_coordinating_workers_spread_across_all_subgoals():
    cache = run_hogwild(2, 20, 4)
    touched = {category for _, _, category, _ in cache}
    assert touched == {0, 1, 2, 3}


def test_workers_without_coordination_pile_onto_one_subgoal():
    cache = run_hogwild(2, 20, 4, coordination_weight=0.0)
    assert {category for _, _, category, _ in cache} == {0}


# ---------------------------------------------------------------- useful_work
def test_two_workers_writing_the_same_token_did_the_work_once():
    assert useful_work([(0, 0, 0, 0), (0, 1, 0, 0)], QUOTA) == 1


def test_two_workers_on_different_subgoals_did_it_twice():
    assert useful_work([(0, 0, 0, 0), (0, 1, 1, 0)], QUOTA) == 2


def test_tokens_beyond_the_quota_are_wasted():
    assert useful_work([(0, 0, 0, 5)], 3) == 0


def test_shared_cache_doubles_the_work_of_two_workers():
    """Главное утверждение урока при мгновенной видимости кеша."""
    one = useful_work(run_hogwild(1, 20, 4), QUOTA)
    two = useful_work(run_hogwild(2, 20, 4), QUOTA)
    assert two == 2 * one


def test_without_coordination_the_second_worker_adds_nothing():
    """Одинаковые модели без чтения кеша делают одну и ту же работу."""
    one = useful_work(run_hogwild(1, 20, 4, coordination_weight=0.0), QUOTA)
    two = useful_work(run_hogwild(2, 20, 4, coordination_weight=0.0), QUOTA)
    assert two == one


def test_stale_cache_breaks_the_coordination():
    """Асинхронность важнее намерений: воркер не видит — воркер дублирует."""
    fresh = useful_work(run_hogwild(2, 20, 4, 1.0, lag=0), QUOTA)
    stale = useful_work(run_hogwild(2, 20, 4, 1.0, lag=1), QUOTA)
    assert stale < fresh
