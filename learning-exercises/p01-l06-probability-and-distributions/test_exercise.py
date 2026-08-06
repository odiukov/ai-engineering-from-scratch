"""Тесты к уроку «Вероятность и распределения». Правь exercise.py."""

import math

import pytest

from exercise import (
    cross_entropy_loss,
    expected_value,
    is_independent,
    log_softmax,
    marginals,
    normal_pdf,
    softmax,
    variance,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

DIE_VALUES = [1, 2, 3, 4, 5, 6]
DIE_PROBS = [1 / 6] * 6


# --------------------------------------------------------- expected_value
def test_expected_value_of_a_fair_die():
    assert expected_value(DIE_VALUES, DIE_PROBS) == APPROX(3.5)


def test_expected_value_of_bernoulli_is_p():
    assert expected_value([0, 1], [0.7, 0.3]) == APPROX(0.3)


def test_expected_value_need_not_be_an_achievable_outcome():
    """3.5 на кости не выпадет никогда, а ожидание именно такое."""
    assert expected_value(DIE_VALUES, DIE_PROBS) not in DIE_VALUES


def test_expected_value_of_a_certain_outcome_is_that_outcome():
    assert expected_value([42, 7], [1.0, 0.0]) == APPROX(42.0)


def test_expected_value_is_linear_in_the_values():
    """E[2X] = 2 E[X] — свойство, а не совпадение."""
    doubled = [2 * v for v in DIE_VALUES]
    assert expected_value(doubled, DIE_PROBS) == APPROX(
        2 * expected_value(DIE_VALUES, DIE_PROBS)
    )


# ---------------------------------------------------------------- variance
def test_variance_of_a_fair_die():
    assert variance(DIE_VALUES, DIE_PROBS) == pytest.approx(35 / 12, abs=1e-9)


def test_variance_without_spread_is_zero():
    assert variance([5, 5], [0.5, 0.5]) == APPROX(0.0)


def test_variance_is_never_negative():
    assert variance([0, 1], [0.3, 0.7]) >= 0


def test_variance_of_bernoulli_is_p_times_one_minus_p():
    assert variance([0, 1], [0.7, 0.3]) == APPROX(0.3 * 0.7)


def test_variance_is_invariant_to_a_shift():
    """Сдвиг всех значений на константу разброс не меняет."""
    shifted = [v + 1000 for v in DIE_VALUES]
    assert variance(shifted, DIE_PROBS) == pytest.approx(
        variance(DIE_VALUES, DIE_PROBS), abs=1e-9
    )


# -------------------------------------------------------------- normal_pdf
def test_normal_pdf_peak_of_the_standard_normal():
    assert normal_pdf(0.0) == APPROX(1 / math.sqrt(2 * math.pi))


def test_normal_pdf_peaks_at_mu():
    assert normal_pdf(3.0, 3.0, 2.0) > normal_pdf(4.0, 3.0, 2.0)


def test_normal_pdf_is_symmetric_around_mu():
    assert normal_pdf(-1.5) == APPROX(normal_pdf(1.5))


def test_normal_pdf_can_exceed_one():
    """Плотность — не вероятность: при узкой сигме она больше единицы."""
    assert normal_pdf(0.0, 0.0, 0.1) > 1.0


def test_normal_pdf_integrates_to_one():
    """Численный интеграл по [-8, 8] с мелким шагом даёт единицу."""
    step = 0.001
    total = sum(normal_pdf(-8 + i * step) * step for i in range(16000))
    assert total == pytest.approx(1.0, abs=1e-6)


# ----------------------------------------------------------------- softmax
def test_softmax_of_equal_logits_is_uniform():
    assert softmax([0.0, 0.0, 0.0]) == pytest.approx([1 / 3] * 3, abs=1e-9)


def test_softmax_sums_to_one():
    assert sum(softmax([2.0, 1.0, 0.1])) == APPROX(1.0)


def test_softmax_preserves_the_order_of_logits():
    probs = softmax([0.5, 3.0, -2.0])
    assert probs[1] > probs[0] > probs[2]


def test_softmax_is_shift_invariant():
    """Прибавь константу ко всем logits — распределение не изменится."""
    a = softmax([1.0, 2.0, 3.0])
    b = softmax([101.0, 102.0, 103.0])
    assert a == pytest.approx(b, abs=1e-9)


def test_softmax_survives_huge_logits():
    """Наивный exp(1000) — это OverflowError, вычитание максимума спасает."""
    probs = softmax([1000.0, 1001.0, 1002.0])
    assert sum(probs) == APPROX(1.0)
    assert probs[2] > probs[1] > probs[0]


def test_softmax_of_two_logits_matches_the_sigmoid():
    assert softmax([1.0, 0.0])[0] == APPROX(1 / (1 + math.exp(-1.0)))


# ------------------------------------------------------------- log_softmax
def test_log_softmax_matches_log_of_softmax_on_tame_input():
    logits = [2.0, 1.0, 0.1]
    expected = [math.log(p) for p in softmax(logits)]
    assert log_softmax(logits) == pytest.approx(expected, abs=1e-9)


def test_log_softmax_is_always_non_positive():
    assert all(x <= 0 for x in log_softmax([5.0, -3.0, 0.0]))


def test_log_softmax_survives_a_probability_rounded_to_zero():
    """log(softmax(...)) здесь дал бы -inf: softmax уже округлил её до нуля."""
    result = log_softmax([0.0, 1000.0])
    assert math.isfinite(result[0])
    assert result[0] == pytest.approx(-1000.0, abs=1e-6)


def test_log_softmax_exponentiates_back_into_softmax():
    logits = [3.0, -1.0, 0.5]
    assert [math.exp(x) for x in log_softmax(logits)] == pytest.approx(
        softmax(logits), abs=1e-9
    )


# ------------------------------------------------------- cross_entropy_loss
def test_cross_entropy_loss_of_full_ignorance_is_log_of_class_count():
    assert cross_entropy_loss([0.0, 0.0], 0) == APPROX(math.log(2))


def test_cross_entropy_loss_is_small_when_confident_and_right():
    assert cross_entropy_loss([10.0, 0.0], 0) < 1e-4


def test_cross_entropy_loss_is_large_when_confident_and_wrong():
    assert cross_entropy_loss([0.0, 10.0], 0) > 9.0


def test_cross_entropy_loss_is_never_negative():
    assert cross_entropy_loss([2.0, 1.0, 0.1], 0) >= 0


def test_cross_entropy_loss_survives_huge_logits():
    """Ловушка урока: без log-space здесь получился бы inf или OverflowError."""
    assert math.isfinite(cross_entropy_loss([900.0, 1000.0], 1))


def test_cross_entropy_loss_ranks_targets_the_same_way_as_softmax():
    logits = [2.0, 1.0, 0.1]
    losses = [cross_entropy_loss(logits, i) for i in range(3)]
    assert losses[0] < losses[1] < losses[2]


# --------------------------------------------------------------- marginals
def test_marginals_of_the_weather_table():
    px, py = marginals([[0.40, 0.10], [0.05, 0.45]])
    assert px == pytest.approx([0.5, 0.5], abs=1e-9)
    assert py == pytest.approx([0.45, 0.55], abs=1e-9)


def test_marginals_each_sum_to_one():
    px, py = marginals([[0.1, 0.2, 0.1], [0.3, 0.2, 0.1]])
    assert sum(px) == APPROX(1.0)
    assert sum(py) == APPROX(1.0)


def test_marginals_handle_a_non_square_table():
    """Строк две, столбцов три — длины маргиналов разные."""
    px, py = marginals([[0.1, 0.2, 0.1], [0.3, 0.2, 0.1]])
    assert len(px) == 2
    assert len(py) == 3


def test_marginals_do_not_confuse_rows_with_columns():
    px, py = marginals([[0.5, 0.0], [0.0, 0.5]])
    assert px == pytest.approx([0.5, 0.5], abs=1e-9)
    assert py == pytest.approx([0.5, 0.5], abs=1e-9)


# ----------------------------------------------------------- is_independent
def test_independent_uniform_table():
    assert is_independent([[0.25, 0.25], [0.25, 0.25]]) is True


def test_rain_and_umbrella_are_not_independent():
    assert is_independent([[0.40, 0.10], [0.05, 0.45]]) is False


def test_independent_product_table_with_uneven_marginals():
    """P(X) = [0.2, 0.8], P(Y) = [0.5, 0.5] — таблица собрана как произведение."""
    joint = [[0.2 * 0.5, 0.2 * 0.5], [0.8 * 0.5, 0.8 * 0.5]]
    assert is_independent(joint) is True


def test_one_broken_cell_is_enough_to_lose_independence():
    joint = [[0.25, 0.25], [0.20, 0.30]]
    assert is_independent(joint) is False
