"""Тесты к уроку «Расписания learning rate и разогрев». Правь exercise.py."""

import pytest

from exercise import (
    constant_schedule,
    cosine_schedule,
    descend,
    linear_warmup,
    lr_curve,
    one_cycle_schedule,
    peak_step,
    step_decay_schedule,
    warmup_cosine_schedule,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


# ------------------------------------------------------- constant_schedule
def test_constant_schedule_ignores_the_step():
    assert constant_schedule(0, lr=0.05) == APPROX(0.05)
    assert constant_schedule(9999, lr=0.05) == APPROX(0.05)


def test_constant_curve_is_flat():
    assert lr_curve(lambda s: constant_schedule(s, lr=0.3), 4) == APPROX([0.3] * 4)


# ----------------------------------------------------- step_decay_schedule
def test_step_decay_holds_the_rate_inside_a_step():
    assert step_decay_schedule(0, lr=0.1, step_size=100) == APPROX(0.1)
    assert step_decay_schedule(99, lr=0.1, step_size=100) == APPROX(0.1)


def test_step_decay_drops_exactly_at_the_boundary():
    """Целочисленное деление делает график лестницей, а не наклоном."""
    assert step_decay_schedule(100, lr=0.1, step_size=100) == pytest.approx(0.01)


def test_step_decay_compounds_over_several_drops():
    assert step_decay_schedule(250, lr=0.1, step_size=100) == pytest.approx(1e-3)


def test_step_decay_with_gamma_one_never_decays():
    assert step_decay_schedule(500, lr=0.1, step_size=100, gamma=1.0) == APPROX(0.1)


# --------------------------------------------------------- cosine_schedule
def test_cosine_starts_at_the_peak():
    assert cosine_schedule(0, lr=0.1, total_steps=100) == APPROX(0.1)


def test_cosine_is_exactly_half_way_in_the_middle():
    assert cosine_schedule(50, lr=0.1, total_steps=100) == pytest.approx(0.05)


def test_cosine_ends_at_the_floor():
    assert cosine_schedule(100, lr=0.1, total_steps=100, lr_min=0.001) == APPROX(0.001)


def test_cosine_never_climbs_back_after_the_end():
    """Голый косинус за пределами total_steps поехал бы вверх — отсеки шаг."""
    assert cosine_schedule(500, lr=0.1, total_steps=100) == APPROX(0.0)


def test_cosine_decreases_monotonically():
    curve = lr_curve(lambda s: cosine_schedule(s, lr=0.1, total_steps=60), 60)
    assert all(a >= b for a, b in zip(curve, curve[1:]))


# ------------------------------------------------------------ linear_warmup
def test_linear_warmup_starts_at_zero():
    assert linear_warmup(0, lr=0.1, warmup_steps=10) == APPROX(0.0)


def test_linear_warmup_is_half_way_at_half_time():
    assert linear_warmup(5, lr=0.1, warmup_steps=10) == APPROX(0.05)


def test_linear_warmup_reaches_the_target_and_stays():
    assert linear_warmup(10, lr=0.1, warmup_steps=10) == APPROX(0.1)
    assert linear_warmup(50, lr=0.1, warmup_steps=10) == APPROX(0.1)


def test_linear_warmup_without_warmup_does_not_divide_by_zero():
    assert linear_warmup(0, lr=0.1, warmup_steps=0) == APPROX(0.1)


# -------------------------------------------------- warmup_cosine_schedule
def test_warmup_cosine_starts_at_zero():
    assert warmup_cosine_schedule(0, lr=0.1, total_steps=100, warmup_steps=10) == APPROX(0.0)


def test_warmup_cosine_peaks_exactly_at_the_end_of_warmup():
    """Если косинус не перезапустить с нуля, пик уедет и в графике будет излом."""
    curve = lr_curve(
        lambda s: warmup_cosine_schedule(s, lr=0.1, total_steps=100, warmup_steps=10), 100
    )
    assert peak_step(curve) == 10
    assert curve[10] == APPROX(0.1)


def test_warmup_cosine_ends_at_the_floor():
    value = warmup_cosine_schedule(100, lr=0.1, total_steps=100, warmup_steps=10, lr_min=0.002)
    assert value == APPROX(0.002)


def test_warmup_cosine_never_exceeds_the_peak():
    curve = lr_curve(
        lambda s: warmup_cosine_schedule(s, lr=0.1, total_steps=200, warmup_steps=20), 200
    )
    assert max(curve) <= 0.1 + 1e-12


# ------------------------------------------------------ one_cycle_schedule
def test_one_cycle_starts_at_a_twenty_fifth_of_the_peak():
    assert one_cycle_schedule(0, lr=0.1, total_steps=100) == APPROX(0.004)


def test_one_cycle_peaks_in_the_middle():
    curve = lr_curve(lambda s: one_cycle_schedule(s, lr=0.1, total_steps=100), 100)
    assert peak_step(curve) == 50
    assert curve[50] == APPROX(0.1)


def test_one_cycle_goes_up_then_down():
    curve = lr_curve(lambda s: one_cycle_schedule(s, lr=0.1, total_steps=100), 100)
    assert all(a < b for a, b in zip(curve[:50], curve[1:50]))
    assert all(a > b for a, b in zip(curve[50:], curve[51:]))


def test_one_cycle_ends_near_zero():
    assert one_cycle_schedule(99, lr=0.1, total_steps=100) < 0.1 / 25


# ------------------------------------------------------------- peak_step
def test_peak_step_finds_the_maximum():
    assert peak_step([0.0, 0.5, 0.2]) == 1


def test_peak_step_prefers_the_earliest_of_equal_maxima():
    assert peak_step([0.1, 0.1, 0.1]) == 0


# --------------------------------------------------------------- lr_curve
def test_lr_curve_length_matches_total_steps():
    assert len(lr_curve(lambda s: 0.1, 7)) == 7


def test_lr_curve_starts_from_step_zero():
    assert lr_curve(lambda s: float(s), 3) == APPROX([0.0, 1.0, 2.0])


# ----------------------------------------------------------------- descend
def test_descend_returns_one_more_point_than_steps():
    assert len(descend(lambda s: 0.1, 1.0, 5)) == 6


def test_descend_shrinks_geometrically_with_a_small_rate():
    assert descend(lambda s: 0.1, 1.0, 2) == APPROX([1.0, 0.8, 0.64])


def test_zero_learning_rate_never_moves():
    assert descend(lambda s: 0.0, 5.0, 3) == APPROX([5.0] * 4)


def test_rate_above_half_makes_the_point_oscillate():
    """Множитель (1 - 2*lr) стал отрицательным — точка прыгает через минимум."""
    path = descend(lambda s: 0.6, 1.0, 4)[1:]
    assert all(a * b < 0 for a, b in zip(path, path[1:]))


def test_rate_above_one_diverges_on_a_perfect_parabola():
    """Даже на x^2 слишком большой lr разносит оптимизацию в бесконечность."""
    assert abs(descend(lambda s: 1.2, 1.0, 100)[-1]) > 1e10


def test_cosine_decay_rescues_a_too_large_peak_rate():
    """Тот же пиковый lr=1.2, но косинус успевает опустить его ниже единицы."""
    path = descend(lambda s: cosine_schedule(s, lr=1.2, total_steps=100), 1.0, 100)
    assert abs(path[-1]) < 1e-6


def test_warmup_keeps_the_early_steps_from_overshooting():
    """С разогревом траектория вообще не отходит от старта дальше, чем началась."""
    path = descend(
        lambda s: warmup_cosine_schedule(s, lr=1.2, total_steps=100, warmup_steps=20),
        1.0,
        100,
    )
    naked = descend(lambda s: cosine_schedule(s, lr=1.2, total_steps=100), 1.0, 100)
    assert max(abs(v) for v in path) <= 1.0
    assert max(abs(v) for v in naked) > 100
