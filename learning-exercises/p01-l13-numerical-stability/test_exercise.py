"""Тесты к уроку «Численная стабильность». Правь exercise.py."""

import math

import pytest

from exercise import (
    clip_by_norm,
    cross_entropy,
    kahan_sum,
    log_softmax,
    logsumexp,
    relative_error,
    softmax,
    stable_variance,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def naive_sum(values):
    """Накопление в лоб — то, что мы и хотим победить."""
    total = 0.0
    for v in values:
        total += v
    return total


def naive_variance(values):
    """E[x^2] - E[x]^2: формула верная, реализация обречённая."""
    n = len(values)
    mean = naive_sum(values) / n
    return naive_sum([x * x for x in values]) / n - mean * mean


# --------------------------------------------------------------- kahan_sum
def test_kahan_sum_of_ten_tenths_is_exactly_one():
    """Наивный цикл здесь даёт 0.9999999999999999 — компенсация даёт ровно 1.0."""
    assert naive_sum([0.1] * 10) != 1.0
    assert kahan_sum([0.1] * 10) == 1.0


def test_kahan_sum_keeps_a_thousand_tiny_addends_that_the_naive_loop_loses():
    """Каждое 1e-16 по отдельности меньше половины ulp единицы: наивный
    цикл округляет их в ничто и возвращает ровно 1.0."""
    values = [1.0] + [1e-16] * 1000
    assert naive_sum(values) == 1.0
    assert kahan_sum(values) == pytest.approx(1.0 + 1e-13, rel=1e-6)


def test_kahan_sum_of_an_empty_list_is_zero():
    assert kahan_sum([]) == 0.0


def test_kahan_sum_agrees_with_exact_arithmetic_on_representable_numbers():
    assert kahan_sum([1.0, 2.0, 4.0, 8.0]) == APPROX(15.0)


def test_kahan_sum_does_not_depend_on_the_order_of_small_addends():
    """Наивная сумма зависит от порядка, компенсированная — практически нет."""
    values = [1.0] + [1e-16] * 1000
    assert kahan_sum(values) == pytest.approx(kahan_sum(values[::-1]), rel=1e-9)


# --------------------------------------------------------- stable_variance
def test_variance_of_a_textbook_sample():
    assert stable_variance([2, 4, 4, 4, 5, 5, 7, 9]) == APPROX(4.0)


def test_variance_of_constant_data_is_zero():
    assert stable_variance([7, 7, 7]) == APPROX(0.0)


def test_variance_survives_a_huge_mean_where_the_naive_formula_dies():
    """Классический пример из урока: правильный ответ 0.6667, наивная
    формула на этих числах уже врёт втрое."""
    values = [1e8, 1e8 + 1.0, 1e8 + 2.0]
    assert stable_variance(values) == pytest.approx(2.0 / 3.0, abs=1e-9)
    assert naive_variance(values) != pytest.approx(2.0 / 3.0, abs=1e-3)


def test_naive_formula_collapses_all_the_way_to_zero_at_1e9():
    """Здесь E[x^2] - E[x]^2 возвращает ровно 0.0: значащих цифр не осталось."""
    values = [1e9 + i for i in range(5)]
    assert naive_variance(values) == 0.0
    assert stable_variance(values) == pytest.approx(2.0, abs=1e-9)


def test_variance_does_not_change_when_all_values_shift():
    """Дисперсия инвариантна к сдвигу — это и есть причина центрировать."""
    base = [1.0, 2.0, 3.0, 4.0]
    shifted = [x + 1e6 for x in base]
    assert stable_variance(shifted) == pytest.approx(stable_variance(base), abs=1e-9)


def test_variance_is_never_negative():
    """Наивная формула умеет выдавать отрицательную дисперсию. Эта — нет."""
    assert stable_variance([1e8, 1e8, 1e8 + 1.0]) >= 0.0


# ----------------------------------------------------------------- logsumexp
def test_logsumexp_on_small_values_matches_the_direct_formula():
    assert logsumexp([1.0, 2.0, 3.0]) == APPROX(
        math.log(math.exp(1) + math.exp(2) + math.exp(3))
    )


def test_logsumexp_survives_large_values_that_overflow_exp():
    """math.exp(1000) — это OverflowError, а правильный ответ существует."""
    with pytest.raises(OverflowError):
        math.exp(1000.0)
    assert logsumexp([1000.0, 1001.0, 1002.0]) == pytest.approx(
        1002.0 + math.log(1 + math.exp(-1) + math.exp(-2)), abs=1e-9
    )


def test_logsumexp_survives_very_negative_values_that_underflow_to_zero():
    """Каждый exp(-1000) обнуляется, и наивный math.log(0.0) падает."""
    assert math.exp(-1000.0) == 0.0
    assert logsumexp([-1000.0, -1001.0]) == pytest.approx(
        -1000.0 + math.log(1 + math.exp(-1)), abs=1e-9
    )


def test_logsumexp_of_a_single_value_is_that_value():
    assert logsumexp([5.0]) == APPROX(5.0)


def test_logsumexp_of_n_equal_values_adds_log_n():
    assert logsumexp([2.0] * 4) == APPROX(2.0 + math.log(4))


def test_logsumexp_shifts_exactly_with_its_input():
    """Сдвиг всех значений на c сдвигает ответ ровно на c — на этом
    тождестве и построен весь приём."""
    base = [0.3, -1.2, 2.5]
    assert logsumexp([x + 700.0 for x in base]) == pytest.approx(
        logsumexp(base) + 700.0, abs=1e-9
    )


def test_logsumexp_with_positive_infinity_is_positive_infinity():
    assert logsumexp([1.0, float("inf"), -3.0]) == float("inf")


# --------------------------------------------------------------- log_softmax
def test_log_softmax_of_equal_logits():
    assert log_softmax([0.0, 0.0]) == APPROX([-math.log(2), -math.log(2)])


def test_log_softmax_equals_logits_minus_logsumexp():
    logits = [1.0, -2.0, 0.5]
    total = logsumexp(logits)
    assert log_softmax(logits) == APPROX([z - total for z in logits])


def test_log_softmax_stays_finite_on_huge_logits():
    result = log_softmax([1000.0, 1001.0, 1002.0])
    assert all(math.isfinite(x) for x in result)
    assert result == pytest.approx([-2.0, -1.0, 0.0], abs=0.5)


def test_log_softmax_values_are_never_positive():
    """Это логарифмы вероятностей — они не бывают больше нуля."""
    assert all(x <= 0.0 for x in log_softmax([5.0, -3.0, 0.0, 12.0]))


def test_log_softmax_splits_mass_between_multiple_positive_infinities():
    result = log_softmax([float("inf"), 3.0, float("inf")])
    assert result[0] == APPROX(-math.log(2))
    assert result[1] == float("-inf")
    assert result[2] == APPROX(-math.log(2))


# ------------------------------------------------------------------ softmax
def test_softmax_of_equal_logits_is_uniform():
    assert softmax([0.0, 0.0]) == APPROX([0.5, 0.5])


def test_softmax_sums_to_one():
    assert sum(softmax([1.0, -4.0, 3.5, 0.0])) == pytest.approx(1.0, abs=1e-12)


def test_softmax_survives_logits_that_overflow_the_naive_formula():
    """Наивное exp(1000)/sum падает, а ответ — обычные вероятности."""
    assert softmax([1000.0, 1001.0, 1002.0]) == pytest.approx(
        [0.09003057, 0.24472847, 0.66524096], abs=1e-7
    )


def test_softmax_ignores_a_constant_shift_of_all_logits():
    assert softmax([1000.0, 1001.0, 1002.0]) == pytest.approx(
        softmax([0.0, 1.0, 2.0]), abs=1e-12
    )


def test_softmax_keeps_the_order_of_the_logits():
    probs = softmax([0.5, 3.0, -1.0])
    assert probs[1] > probs[0] > probs[2]


def test_softmax_output_never_contains_nan_or_inf():
    """Стабильная версия не порождает ни inf, ни nan даже на диком разбросе."""
    probs = softmax([-800.0, 0.0, 800.0])
    assert all(math.isfinite(p) and p >= 0.0 for p in probs)
    assert sum(probs) == pytest.approx(1.0, abs=1e-12)


def test_softmax_assigns_all_mass_to_a_single_positive_infinity():
    assert softmax([1.0, float("inf"), -2.0]) == APPROX([0.0, 1.0, 0.0])


def test_softmax_splits_mass_between_multiple_positive_infinities():
    assert softmax([float("inf"), 1.0, float("inf")]) == APPROX([0.5, 0.0, 0.5])


# ------------------------------------------------------------ cross_entropy
def test_cross_entropy_on_uniform_logits_is_log_of_the_class_count():
    assert cross_entropy(0, [0.0, 0.0, 0.0]) == APPROX(math.log(3))


def test_cross_entropy_matches_minus_log_of_the_softmax_probability():
    logits = [2.0, 5.0, 1.0]
    assert cross_entropy(1, logits) == APPROX(-math.log(softmax(logits)[1]))


def test_cross_entropy_is_near_zero_for_a_confident_correct_answer():
    assert cross_entropy(0, [50.0, 0.0, 0.0]) < 1e-15


def test_cross_entropy_grows_when_the_answer_is_wrong():
    assert cross_entropy(2, [50.0, 0.0, 0.0]) > 40.0


def test_cross_entropy_stays_finite_on_huge_logits():
    value = cross_entropy(2, [1000.0, 1001.0, 1002.0])
    assert math.isfinite(value)
    assert value == pytest.approx(0.40760596, abs=1e-7)


# ----------------------------------------------------------- relative_error
def test_relative_error_of_identical_values_is_zero():
    assert relative_error(1.0, 1.0) == APPROX(0.0)


def test_relative_error_of_two_zeros_does_not_divide_by_zero():
    """Ловушка: голое |a-b|/|a| на нулевом градиенте роняет проверку."""
    assert relative_error(0.0, 0.0) == APPROX(0.0)


def test_relative_error_normalises_by_the_larger_magnitude():
    assert relative_error(1.0, 1.1) == pytest.approx(0.1 / 1.1, abs=1e-12)


def test_relative_error_is_symmetric():
    assert relative_error(3.0, 7.0) == APPROX(relative_error(7.0, 3.0))


def test_two_tiny_gradients_are_not_reported_as_a_huge_error():
    """1e-12 против 2e-12 отличаются вдвое, но оба практически ноль —
    порог 1e-8 в знаменателе не даёт поднять ложную тревогу."""
    assert relative_error(1e-12, 2e-12) < 1e-3


def test_a_sign_flip_is_reported_as_a_large_error():
    assert relative_error(1.0, -1.0) > 1.0


# ------------------------------------------------------------- clip_by_norm
def test_clip_leaves_a_short_gradient_alone():
    assert clip_by_norm([3.0, 4.0], 5.0) == APPROX([3.0, 4.0])


def test_clip_scales_a_long_gradient_down_to_the_threshold():
    clipped = clip_by_norm([3.0, 4.0], 1.0)
    assert math.sqrt(sum(g * g for g in clipped)) == pytest.approx(1.0, abs=1e-12)


def test_clip_preserves_the_direction():
    """Смысл обрезки по норме: шаг короче, но туда же. Поэлементный clamp
    этого не гарантирует."""
    original = [1.0, 2.0, -3.0]
    clipped = clip_by_norm(original, 0.5)
    ratios = [c / o for c, o in zip(clipped, original)]
    assert ratios == pytest.approx([ratios[0]] * 3, abs=1e-12)


def test_clip_of_a_zero_gradient_does_not_divide_by_zero():
    assert clip_by_norm([0.0, 0.0], 1.0) == APPROX([0.0, 0.0])


def test_clip_does_not_mutate_the_input_list():
    grads = [10.0, 20.0]
    clip_by_norm(grads, 1.0)
    assert grads == [10.0, 20.0]
