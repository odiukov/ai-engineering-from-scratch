"""Тесты к уроку «Рекурсивное самоулучшение: гонка capability и alignment».

Правь exercise.py.
"""

import random

import pytest

from exercise import (
    audit_cycles,
    crossing_cycle,
    crossing_share,
    next_cycle,
    noisy_race,
    race,
    self_improve,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


# ------------------------------------------------------------- next_cycle
def test_next_cycle_multiplies_both_metrics():
    assert next_cycle(1.0, 1.0, 1.15, 1.08) == APPROX((1.15, 1.08))


def test_next_cycle_with_unit_rates_changes_nothing():
    assert next_cycle(2.0, 3.0, 1.0, 1.0) == APPROX((2.0, 3.0))


def test_next_cycle_with_equal_rates_keeps_the_ratio():
    """Равные темпы не создают разрыва: отношение C/A сохраняется."""
    c, a = next_cycle(4.0, 2.0, 1.3, 1.3)
    assert c / a == APPROX(2.0)


def test_next_cycle_rate_below_one_shrinks_the_metric():
    c, _a = next_cycle(10.0, 10.0, 0.5, 1.0)
    assert c == APPROX(5.0)


# ------------------------------------------------------------------- race
def test_race_length_includes_cycle_zero():
    assert len(race(7, 1.1, 1.05)) == 8


def test_race_starts_with_zero_gap():
    first = race(5, 1.2, 1.1)[0]
    assert first == (0, APPROX(1.0), APPROX(1.0), APPROX(0.0))


def test_race_gap_stays_zero_when_rates_are_equal():
    """Смысловое свойство: alignment успевает — разрыв не появляется вообще."""
    gaps = [gap for _cyc, _c, _a, gap in race(20, 1.1, 1.1)]
    assert all(g == APPROX(0.0) for g in gaps)


def test_race_gap_grows_monotonically_when_capability_leads():
    gaps = [gap for _cyc, _c, _a, gap in race(15, 1.15, 1.08)]
    assert all(b > a for a, b in zip(gaps, gaps[1:]))


def test_race_matches_next_cycle_applied_twice():
    _cyc, c, a, _gap = race(2, 1.2, 1.1)[2]
    expected = next_cycle(*next_cycle(1.0, 1.0, 1.2, 1.1), 1.2, 1.1)
    assert (c, a) == APPROX(expected)


def test_race_honours_custom_start_values():
    _cyc, c, a, gap = race(0, 1.2, 1.1, start_c=3.0, start_a=1.0)[0]
    assert (c, a, gap) == APPROX((3.0, 1.0, 2.0))


# --------------------------------------------------------- crossing_cycle
def test_crossing_cycle_finds_the_first_crossing():
    assert crossing_cycle(race(30, 1.15, 1.08), 1.5) == 9


def test_crossing_cycle_returns_minus_one_when_never_crossed():
    assert crossing_cycle(race(30, 1.1, 1.1), 0.5) == -1


def test_crossing_cycle_is_inclusive_at_the_threshold():
    """Ровно на пороге — уже пересечение, а не «ещё можно»."""
    traj = [(0, 1.0, 1.0, 0.0), (1, 2.0, 1.0, 1.0)]
    assert crossing_cycle(traj, 1.0) == 1


def test_crossing_cycle_zero_means_unsafe_before_the_first_cycle():
    traj = race(5, 1.1, 1.1, start_c=3.0, start_a=1.0)
    assert crossing_cycle(traj, 1.5) == 0


# ------------------------------------------------------------- noisy_race
def test_noisy_race_with_zero_noise_equals_the_clean_race():
    noisy = noisy_race(10, 1.12, 1.07, 0.0, 0.0, random.Random(0))
    assert [c for _cyc, c, _a, _g in noisy] == APPROX(
        [c for _cyc, c, _a, _g in race(10, 1.12, 1.07)]
    )


def test_noisy_race_is_reproducible_for_the_same_seed():
    a = noisy_race(20, 1.1, 1.05, 0.05, 0.05, random.Random(7))
    b = noisy_race(20, 1.1, 1.05, 0.05, 0.05, random.Random(7))
    assert a == b


def test_noisy_race_differs_between_seeds():
    a = noisy_race(20, 1.1, 1.05, 0.05, 0.05, random.Random(1))
    b = noisy_race(20, 1.1, 1.05, 0.05, 0.05, random.Random(2))
    assert a != b


def test_noisy_race_floor_stops_a_bad_draw_from_collapsing_the_system():
    """Темп 0.0 с нулевым шумом всё равно поднимается до floor=0.9."""
    traj = noisy_race(1, 0.0, 0.0, 0.0, 0.0, random.Random(0), floor=0.9)
    _cyc, c, a, _gap = traj[1]
    assert (c, a) == APPROX((0.9, 0.9))


def test_noisy_race_does_not_touch_the_global_random():
    """Функция обязана брать числа из переданного rng, а не из random.*"""
    random.seed(123)
    before = random.random()
    random.seed(123)
    noisy_race(50, 1.1, 1.05, 0.05, 0.05, random.Random(9))
    assert random.random() == APPROX(before)


# ------------------------------------------------------------ crossing_share
def test_crossing_share_is_a_probability():
    share = crossing_share(50, 20, 1.12, 1.08, 0.02, 0.02, 1.0, random.Random(3))
    assert 0.0 <= share <= 1.0


def test_crossing_share_is_higher_when_capability_leads_harder():
    fast = crossing_share(100, 25, 1.20, 1.05, 0.02, 0.02, 2.0, random.Random(4))
    slow = crossing_share(100, 25, 1.10, 1.09, 0.02, 0.02, 2.0, random.Random(4))
    assert fast > slow


def test_crossing_share_is_zero_for_an_unreachable_threshold():
    assert crossing_share(20, 10, 1.05, 1.05, 0.01, 0.01, 1e9, random.Random(5)) == 0.0


def test_crossing_share_with_no_trials_is_zero():
    assert crossing_share(0, 10, 1.5, 1.0, 0.1, 0.1, 0.5, random.Random(6)) == 0.0


def test_crossing_share_is_reproducible_for_the_same_seed():
    args = (60, 20, 1.13, 1.08, 0.03, 0.03, 1.2)
    assert crossing_share(*args, random.Random(8)) == crossing_share(
        *args, random.Random(8)
    )


# ----------------------------------------------------------- self_improve
def test_self_improve_stops_at_the_ceiling_while_the_metric_still_grows():
    """Главное свойство урока: потолок сильнее «стало же лучше»."""
    out = self_improve(lambda x: x + 1, float, 0, 3)
    assert out["reason"] == "ceiling"
    assert out["cycles"] == 3
    assert out["system"] == 3


def test_self_improve_stops_early_when_the_proposal_brings_no_gain():
    out = self_improve(lambda x: x, float, 5, 100)
    assert out["reason"] == "no_gain"
    assert out["cycles"] == 0


def test_self_improve_rejects_a_proposal_that_makes_things_worse():
    out = self_improve(lambda x: x - 1, float, 5, 10)
    assert (out["system"], out["reason"]) == (5, "no_gain")


def test_self_improve_history_starts_with_the_starting_score():
    out = self_improve(lambda x: x + 2, float, 1, 4)
    assert out["history"][0] == APPROX(1.0)
    assert len(out["history"]) == out["cycles"] + 1


def test_self_improve_history_is_strictly_increasing():
    out = self_improve(lambda x: x + 1, float, 0, 6)
    h = out["history"]
    assert all(b > a for a, b in zip(h, h[1:]))


def test_self_improve_with_zero_ceiling_runs_nothing():
    calls = []

    def propose(x):
        calls.append(x)
        return x + 1

    out = self_improve(propose, float, 10, 0)
    assert (out["cycles"], out["reason"], calls) == (0, "ceiling", [])


def test_self_improve_min_gain_filters_out_marginal_improvements():
    """Прирост 0.01 при min_gain=0.5 — это шум, а не улучшение."""
    out = self_improve(lambda x: x + 0.01, float, 0.0, 50, min_gain=0.5)
    assert (out["cycles"], out["reason"]) == (0, "no_gain")


# ----------------------------------------------------------- audit_cycles
def test_audit_cycles_lists_every_checkpoint():
    assert audit_cycles(10, 3) == [3, 6, 9]


def test_audit_cycles_without_a_human_is_empty():
    assert audit_cycles(10, 0) == []


def test_audit_cycles_with_a_period_longer_than_the_run_is_empty():
    assert audit_cycles(5, 12) == []


def test_audit_cycles_with_period_one_checks_every_cycle():
    assert audit_cycles(4, 1) == [1, 2, 3, 4]
