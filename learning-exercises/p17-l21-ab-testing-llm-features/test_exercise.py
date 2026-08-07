"""Тесты к уроку «A/B-тесты LLM-фич». Правь exercise.py."""

import random
import statistics

import pytest

from exercise import (
    NONDETERMINISM_BUFFER,
    benjamini_hochberg,
    normal_cdf,
    proportion_test,
    run_experiment,
    sample_size,
    srm_check,
    wilson_interval,
    z_quantile,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)
ROUGH = lambda x: pytest.approx(x, abs=1e-4)


# -------------------------------------------------------------- normal_cdf
def test_normal_cdf_is_a_half_at_zero():
    assert normal_cdf(0.0) == APPROX(0.5)


def test_normal_cdf_matches_the_textbook_1_96():
    assert normal_cdf(1.96) == ROUGH(0.975002)


def test_normal_cdf_is_symmetric():
    assert normal_cdf(1.3) + normal_cdf(-1.3) == APPROX(1.0)


# -------------------------------------------------------------- z_quantile
def test_z_quantile_inverts_normal_cdf():
    for z in (-2.5, -0.7, 0.0, 0.7, 2.5):
        assert z_quantile(normal_cdf(z)) == ROUGH(z)


def test_z_quantile_gives_the_famous_1_96():
    assert z_quantile(0.975) == ROUGH(1.959964)


def test_z_quantile_of_a_certainty_is_an_error():
    """Квантиль единицы бесконечен — вернуть «правдоподобное 40» нельзя."""
    with pytest.raises(ValueError):
        z_quantile(1.0)


# ------------------------------------------------------------- sample_size
def test_sample_size_for_the_lesson_example():
    """База 3%, ожидаемый лифт 5%, мощность 80% — цифра из урока."""
    assert sample_size(0.03, 0.05, buffer=1.0) == 207938


def test_nondeterminism_buffer_inflates_the_sample():
    plain = sample_size(0.03, 0.05, buffer=1.0)
    padded = sample_size(0.03, 0.05)
    assert padded > plain
    assert padded == pytest.approx(plain * NONDETERMINISM_BUFFER, rel=1e-4)


def test_halving_the_effect_roughly_quadruples_the_sample():
    """Знаменатель квадратичный — вот цена мелких лифтов."""
    big = sample_size(0.03, 0.20, buffer=1.0)
    small = sample_size(0.03, 0.10, buffer=1.0)
    assert small / big == pytest.approx(4.0, rel=0.05)


def test_more_power_costs_more_observations():
    assert sample_size(0.03, 0.10, power=0.95) > sample_size(0.03, 0.10, power=0.80)


def test_stricter_alpha_costs_more_observations():
    assert sample_size(0.03, 0.10, alpha=0.01) > sample_size(0.03, 0.10, alpha=0.05)


def test_zero_lift_needs_infinite_data_and_is_refused():
    with pytest.raises(ValueError):
        sample_size(0.03, 0.0)


# ---------------------------------------------------------- wilson_interval
def test_wilson_interval_on_45_of_50():
    lo, hi = wilson_interval(45, 50)
    assert (lo, hi) == (ROUGH(0.786398), ROUGH(0.956524))


def test_wilson_stays_inside_zero_one_on_a_perfect_score():
    """Наивная формула на 10 из 10 даёт (1.0, 1.0) — «уверены на 100%»."""
    lo, hi = wilson_interval(10, 10)
    assert hi == APPROX(1.0)
    assert lo == ROUGH(0.722467)


def test_wilson_stays_inside_zero_one_on_a_zero_score():
    lo, hi = wilson_interval(0, 10)
    assert lo == APPROX(0.0)
    assert hi == ROUGH(0.277533)


def test_wilson_interval_is_not_symmetric_around_the_observed_rate():
    """Центр смещён к 0.5 — это и есть поправка Уилсона, а не ошибка."""
    lo, hi = wilson_interval(45, 50)
    p = 45 / 50
    assert (p - lo) > (hi - p)


def test_more_data_narrows_the_interval():
    narrow = wilson_interval(900, 1000)
    wide = wilson_interval(9, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_impossible_counts_are_refused():
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


# --------------------------------------------------------- proportion_test
def test_clear_lift_is_significant():
    result = proportion_test(300, 10000, 360, 10000)
    assert result["effect"] == ROUGH(0.006)
    assert result["z"] == ROUGH(2.375013)
    assert result["p_value"] == ROUGH(0.017548)
    assert result["significant"] is True


def test_tiny_difference_on_the_same_sample_is_noise():
    assert proportion_test(300, 10000, 305, 10000)["significant"] is False


def test_identical_rates_give_zero_effect_and_p_value_one():
    result = proportion_test(300, 10000, 300, 10000)
    assert result["effect"] == APPROX(0.0)
    assert result["p_value"] == ROUGH(1.0)


def test_same_rates_but_more_data_turn_noise_into_a_result():
    """Значимость — это про эффект И объём, а не про эффект."""
    assert proportion_test(30, 1000, 36, 1000)["significant"] is False
    assert proportion_test(3000, 100000, 3600, 100000)["significant"] is True


def test_direction_of_the_effect_flips_the_z_but_not_the_p_value():
    forward = proportion_test(300, 10000, 360, 10000)
    backward = proportion_test(360, 10000, 300, 10000)
    assert forward["z"] == ROUGH(-backward["z"])
    assert forward["p_value"] == ROUGH(backward["p_value"])


def test_two_empty_arms_are_not_a_discovery():
    result = proportion_test(0, 100, 0, 100)
    assert result["z"] == APPROX(0.0)
    assert result["significant"] is False


# ---------------------------------------------------------------- srm_check
def test_a_healthy_split_shows_no_srm():
    assert srm_check(5000, 5000)["srm"] is False


def test_the_same_lopsided_ratio_is_noise_on_a_small_sample():
    """47/53 на сотне — обычная выборочная дрожь."""
    assert srm_check(47, 53)["srm"] is False


def test_the_same_ratio_is_a_broken_assignment_on_a_large_sample():
    """47/53 на десяти тысячах случайно не выпадает — механизм сломан."""
    result = srm_check(4700, 5300)
    assert result["srm"] is True
    assert result["observed_share"] == ROUGH(0.47)


def test_srm_respects_a_deliberately_uneven_design():
    """90/10 — не поломка, если так и задумано."""
    assert srm_check(9000, 1000, expected_share=0.9)["srm"] is False
    assert srm_check(9000, 1000, expected_share=0.5)["srm"] is True


def test_srm_alpha_is_strict_on_purpose():
    """При alpha=0.05 половина дашборда была бы красной без всякой поломки."""
    borderline = srm_check(5100, 4900)
    assert borderline["p_value"] < 0.05
    assert borderline["srm"] is False


# ------------------------------------------------------- benjamini_hochberg
def test_bh_rejects_only_the_clear_winner():
    assert benjamini_hochberg([0.001, 0.04, 0.9]) == (True, False, False)


def test_bh_rejects_a_consistent_family():
    assert benjamini_hochberg([0.001, 0.02, 0.03]) == (True, True, True)


def test_bh_looks_for_the_largest_passing_rank_not_the_first_failing_one():
    """Наивный обход остановился бы на ранге 1 и потерял два открытия."""
    assert benjamini_hochberg([0.001, 0.04, 0.045]) == (True, True, True)


def test_bh_keeps_the_input_order():
    assert benjamini_hochberg([0.9, 0.001, 0.04]) == (False, True, False)


def test_bh_of_nothing_is_nothing():
    assert benjamini_hochberg([]) == ()


def test_bh_is_less_conservative_than_bonferroni():
    """Двадцать тестов: Бонферрони режет мощность, BH — мягче."""
    p_values = [0.001, 0.002, 0.003, 0.004] + [0.5] * 16
    bh = benjamini_hochberg(p_values)
    bonferroni = tuple(p < 0.05 / len(p_values) for p in p_values)
    assert sum(bh) > sum(bonferroni)


# ----------------------------------------------------------- run_experiment
def test_experiment_splits_the_traffic_roughly_in_half():
    result = run_experiment(0.10, 0.10, 4000, random.Random(0))
    assert result["n_a"] + result["n_b"] == 4000
    assert srm_check(result["n_a"], result["n_b"])["srm"] is False


def test_experiment_is_reproducible_for_the_same_seed():
    a = run_experiment(0.10, 0.12, 2000, random.Random(3))
    b = run_experiment(0.10, 0.12, 2000, random.Random(3))
    assert a == b


def test_fixed_horizon_finds_nothing_where_there_is_nothing():
    result = run_experiment(0.10, 0.10, 4000, random.Random(0))
    assert result["stopped_at"] is None
    assert result["significant"] is False


def test_peeking_declares_a_winner_on_data_with_no_effect():
    """Те же данные, тот же seed. Разница только в том, что подглядывали."""
    honest = run_experiment(0.10, 0.10, 4000, random.Random(0))
    peeker = run_experiment(0.10, 0.10, 4000, random.Random(0), peek_every=200)
    assert honest["significant"] is False
    assert peeker["significant"] is True
    assert peeker["stopped_at"] == 1000


def test_peeking_blows_past_the_declared_false_positive_rate():
    """alpha=0.05 обещает 5% ложных срабатываний. Двадцать взглядов — в разы больше."""
    seeds = range(120)
    honest = [run_experiment(0.10, 0.10, 3000, random.Random(s)) for s in seeds]
    peeker = [run_experiment(0.10, 0.10, 3000, random.Random(s), peek_every=150) for s in seeds]
    honest_rate = sum(r["significant"] for r in honest) / len(honest)
    peeker_rate = sum(r["significant"] for r in peeker) / len(peeker)
    assert honest_rate < 0.10
    assert peeker_rate > 0.20


def test_a_test_stopped_on_significance_overstates_the_effect():
    """Главная беда peeking — не ложные победы, а завышенный эффект.

    Настоящий эффект 0.02. Фиксированный горизонт даёт около него.
    Остановка «как только загорелось зелёным» отбирает ровно те прогоны,
    где шум был в нужную сторону, и средняя оценка уезжает почти вдвое.
    Дальше эта цифра идёт в квартальный отчёт.
    """
    true_effect = 0.02
    seeds = range(150)
    honest = [run_experiment(0.10, 0.12, 6000, random.Random(s)) for s in seeds]
    peeker = [run_experiment(0.10, 0.12, 6000, random.Random(s), peek_every=200) for s in seeds]
    stopped = [r for r in peeker if r["stopped_at"] is not None]

    honest_mean = statistics.mean(r["effect"] for r in honest)
    stopped_mean = statistics.mean(r["effect"] for r in stopped)

    assert len(stopped) > 50
    assert honest_mean == pytest.approx(true_effect, abs=0.004)
    assert stopped_mean > honest_mean * 1.4


def test_min_per_arm_blocks_peeking_on_a_handful_of_observations():
    """На десятке наблюдений нормальное приближение не работает вовсе."""
    result = run_experiment(0.10, 0.10, 60, random.Random(0), peek_every=10, min_per_arm=50)
    assert result["looks"] == 0
