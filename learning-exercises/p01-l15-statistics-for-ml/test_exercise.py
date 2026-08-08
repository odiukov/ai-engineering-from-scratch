"""Тесты к уроку «Статистика для машинного обучения». Правь exercise.py."""

import math

import pytest

from exercise import (
    bootstrap_ci,
    cohens_d,
    mean,
    pearson,
    percentile,
    spearman,
    variance,
    welch_t,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


# -------------------------------------------------------------------- mean
def test_mean_of_a_short_list():
    assert mean([1, 2, 3, 4]) == APPROX(2.5)


def test_mean_of_a_constant_list_is_that_constant():
    assert mean([7, 7, 7]) == APPROX(7.0)


def test_a_single_outlier_drags_the_mean_far_from_the_median():
    """Ровно тот пример из урока: среднее 202, медиана 3."""
    values = [1, 2, 3, 4, 1000]
    assert mean(values) == APPROX(202.0)
    assert percentile(values, 50) == APPROX(3.0)


# -------------------------------------------------------------- percentile
def test_median_of_an_even_length_list_interpolates():
    assert percentile([1, 2, 3, 4], 50) == APPROX(2.5)


def test_median_of_an_odd_length_list_is_the_middle_value():
    assert percentile([1, 2, 3], 50) == APPROX(2.0)


def test_percentile_zero_and_hundred_are_the_extremes():
    assert percentile([4, 1, 3, 2], 0) == APPROX(1.0)
    assert percentile([4, 1, 3, 2], 100) == APPROX(4.0)


def test_percentile_sorts_the_data_itself():
    """Порядок на входе не должен влиять на ответ."""
    assert percentile([3, 1, 4, 2], 50) == APPROX(percentile([1, 2, 3, 4], 50))


def test_percentile_does_not_mutate_the_caller_list():
    """Ловушка: values.sort() портит список вызывающего."""
    values = [3, 1, 4, 2]
    percentile(values, 50)
    assert values == [3, 1, 4, 2]


def test_percentiles_are_monotonic_in_q():
    values = [5, 1, 9, 3, 7, 2, 8]
    got = [percentile(values, q) for q in (0, 25, 50, 75, 100)]
    assert got == sorted(got)


def test_iqr_covers_the_middle_half():
    values = list(range(101))
    assert percentile(values, 75) - percentile(values, 25) == APPROX(50.0)


# ---------------------------------------------------------------- variance
def test_population_variance_of_a_textbook_sample():
    assert variance([2, 4, 4, 4, 5, 5, 7, 9], sample=False) == APPROX(4.0)


def test_sample_variance_is_larger_because_of_bessel():
    """Деление на n-1 вместо n поднимает оценку — она иначе занижена."""
    values = [2, 4, 4, 4, 5, 5, 7, 9]
    assert variance(values) == APPROX(4.571428571428571)
    assert variance(values) > variance(values, sample=False)


def test_variance_of_constant_data_is_zero():
    assert variance([7, 7, 7]) == APPROX(0.0)


def test_variance_ignores_a_constant_shift():
    base = [1.0, 2.0, 3.0, 4.0]
    assert variance([x + 500 for x in base]) == APPROX(variance(base))


def test_bessel_correction_barely_matters_on_large_samples():
    """На тысячах точек разница между n и n-1 неразличима, на десятке — нет."""
    big = list(range(2000))
    assert variance(big) == pytest.approx(variance(big, sample=False), rel=1e-3)


def test_sample_variance_of_one_value_is_an_error_not_zero():
    with pytest.raises(ValueError):
        variance([42.0])


# ----------------------------------------------------------------- pearson
def test_pearson_of_a_perfect_increasing_line_is_one():
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0, abs=1e-9)


def test_pearson_of_a_perfect_decreasing_line_is_minus_one():
    assert pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0, abs=1e-9)


def test_pearson_of_uncorrelated_data_is_zero():
    assert pearson([1, 2, 3, 4], [1, -1, -1, 1]) == APPROX(0.0)


def test_pearson_is_symmetric():
    x, y = [1, 4, 9, 3], [2, 2, 7, 1]
    assert pearson(x, y) == APPROX(pearson(y, x))


def test_pearson_ignores_rescaling_of_either_variable():
    """Коэффициент безразмерный: смена единиц измерения его не меняет."""
    x, y = [1, 4, 9, 3], [2, 2, 7, 1]
    assert pearson(x, [100 * b + 5 for b in y]) == APPROX(pearson(x, y))


def test_pearson_of_a_constant_series_does_not_divide_by_zero():
    assert pearson([1, 2, 3], [5, 5, 5]) == APPROX(0.0)


def test_pearson_underestimates_a_curved_but_monotonic_link():
    """y = x^3 — идеально монотонная связь, но не прямая линия."""
    assert pearson([1, 2, 3], [1, 8, 27]) < 0.99


# ---------------------------------------------------------------- spearman
def test_spearman_sees_a_monotonic_link_that_pearson_misses():
    x, y = [1, 2, 3], [1, 8, 27]
    assert spearman(x, y) == pytest.approx(1.0, abs=1e-9)
    assert spearman(x, y) > pearson(x, y)


def test_spearman_of_reversed_order_is_minus_one():
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0, abs=1e-9)


def test_spearman_averages_tied_ranks():
    """Два одинаковых значения делят ранги 1 и 2 пополам — по 1.5 каждому."""
    assert spearman([1, 1, 2], [1, 2, 3]) == APPROX(0.8660254037844387)


def test_spearman_does_not_depend_on_the_order_of_tied_values():
    """Без усреднения рангов ответ зависел бы от случайного порядка входа."""
    assert spearman([1, 1, 2], [1, 2, 3]) == APPROX(spearman([1, 1, 2], [2, 1, 3]))


def test_spearman_matches_pearson_on_a_straight_line():
    assert spearman([1, 2, 3, 4], [5, 7, 9, 11]) == pytest.approx(
        pearson([1, 2, 3, 4], [5, 7, 9, 11]), abs=1e-9
    )


def test_spearman_is_unmoved_by_an_extreme_outlier():
    """Ранги сплющивают выбросы — за это Спирмена и берут."""
    x = [1, 2, 3, 4]
    assert spearman(x, [1, 2, 3, 1000]) == APPROX(spearman(x, [1, 2, 3, 4]))


# ----------------------------------------------------------------- welch_t
def test_welch_t_of_identical_groups_is_zero():
    assert welch_t([1, 2, 3], [1, 2, 3]) == APPROX(0.0)


def test_welch_t_on_two_shifted_groups():
    assert welch_t([1, 2, 3, 4, 5], [2, 3, 4, 5, 6]) == APPROX(-1.0)


def test_swapping_the_groups_flips_the_sign():
    a, b = [1, 2, 3, 4, 5], [2, 3, 4, 5, 6]
    assert welch_t(a, b) == APPROX(-welch_t(b, a))


def test_the_sign_says_which_mean_is_bigger():
    assert welch_t([10, 11, 12], [1, 2, 3]) > 0


def test_more_data_inflates_the_statistic_for_the_same_difference():
    """С ростом n «значимым» становится любое различие — в этом ловушка."""
    small = welch_t([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
    big = welch_t([1, 2, 3, 4, 5] * 10, [2, 3, 4, 5, 6] * 10)
    assert abs(big) > 3 * abs(small)


def test_welch_t_of_distinct_constant_groups_is_signed_infinity():
    assert welch_t([1.0] * 3, [2.0] * 3) == float("-inf")
    assert welch_t([2.0] * 3, [1.0] * 3) == float("inf")


def test_welch_t_of_identical_constant_groups_is_undefined():
    assert math.isnan(welch_t([1.0] * 3, [1.0] * 3))


# ---------------------------------------------------------------- cohens_d
def test_cohens_d_of_identical_groups_is_zero():
    assert cohens_d([1, 2, 3], [1, 2, 3]) == APPROX(0.0)


def test_cohens_d_on_two_shifted_groups():
    assert cohens_d([1, 2, 3, 4, 5], [2, 3, 4, 5, 6]) == APPROX(-0.6324555320336759)


def test_cohens_d_barely_moves_when_the_sample_grows():
    """Ровно то, чем размер эффекта отличается от t: он про величину
    различия, а не про количество данных."""
    small = cohens_d([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
    big = cohens_d([1, 2, 3, 4, 5] * 10, [2, 3, 4, 5, 6] * 10)
    assert abs(big - small) < 0.15


def test_a_wider_spread_shrinks_the_effect_size():
    """То же различие средних на шумных данных значит меньше."""
    tight = cohens_d([9, 10, 11], [12, 13, 14])
    noisy = cohens_d([0, 10, 20], [3, 13, 23])
    assert abs(noisy) < abs(tight)


def test_swapping_the_groups_flips_the_effect_size():
    a, b = [1, 2, 3, 4, 5], [2, 3, 4, 5, 6]
    assert cohens_d(a, b) == APPROX(-cohens_d(b, a))


def test_cohens_d_of_distinct_constant_groups_is_signed_infinity():
    assert cohens_d([1.0] * 3, [2.0] * 3) == float("-inf")
    assert cohens_d([2.0] * 3, [1.0] * 3) == float("inf")


def test_cohens_d_of_identical_constant_groups_is_undefined():
    assert math.isnan(cohens_d([1.0] * 3, [1.0] * 3))


# ------------------------------------------------------------- bootstrap_ci
def test_bootstrap_ci_of_constant_data_has_zero_width():
    assert bootstrap_ci([5.0] * 20) == (APPROX(5.0), APPROX(5.0))


def test_bootstrap_ci_lower_bound_is_below_the_upper_bound():
    low, high = bootstrap_ci(list(range(101)))
    assert low < high


def test_bootstrap_ci_brackets_the_sample_mean():
    values = list(range(101))
    low, high = bootstrap_ci(values)
    assert low < mean(values) < high


def test_bootstrap_ci_is_reproducible_for_the_same_seed():
    """Свой random.Random(seed), а не глобальный random — иначе результат
    зависит от того, кто ещё дёргал генератор."""
    values = list(range(101))
    assert bootstrap_ci(values, seed=7) == bootstrap_ci(values, seed=7)


def test_a_different_seed_gives_a_different_interval():
    values = list(range(101))
    assert bootstrap_ci(values, seed=1) != bootstrap_ci(values, seed=2)


def test_more_data_narrows_the_interval():
    """Погрешность среднего падает как корень из объёма выборки."""
    small = bootstrap_ci([i / 2.0 for i in range(20)])
    big = bootstrap_ci([i / 40.0 for i in range(400)])
    assert (big[1] - big[0]) < (small[1] - small[0])


def test_a_looser_alpha_gives_a_narrower_interval():
    values = list(range(101))
    wide = bootstrap_ci(values, alpha=0.05)
    narrow = bootstrap_ci(values, alpha=0.5)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_bootstrap_works_for_a_statistic_without_any_formula():
    """Медиана — ровно тот случай, ради которого бутстрэп и придуман."""
    values = list(range(101))
    low, high = bootstrap_ci(values, statistic=lambda s: percentile(s, 50))
    assert low < 50.0 < high


def test_resampling_must_be_with_replacement():
    """Без возвращения каждая выборка равна исходной, разброса нет вовсе
    и интервал схлопывается в точку."""
    low, high = bootstrap_ci([1.0, 2.0, 3.0, 40.0])
    assert high - low > 1.0
