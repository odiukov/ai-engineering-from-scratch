"""Тесты к уроку «Теория информации». Правь exercise.py."""

import math

import pytest

from exercise import (
    conditional_entropy,
    cross_entropy,
    entropy,
    information_content,
    joint_entropy,
    kl_divergence,
    mutual_information,
    perplexity,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

INDEPENDENT = [[0.25, 0.25], [0.25, 0.25]]
DEPENDENT = [[0.45, 0.05], [0.05, 0.45]]
IDENTICAL = [[0.5, 0.0], [0.0, 0.5]]


# ---------------------------------------------------- information_content
def test_information_content_of_a_fair_coin_is_one_bit():
    assert information_content(0.5) == APPROX(1.0)


def test_information_content_of_a_certain_event_is_zero():
    assert information_content(1.0) == APPROX(0.0)


def test_information_content_of_an_impossible_event_is_infinite():
    """Ловушка: math.log(0) — это ValueError, а не бесконечность."""
    assert information_content(0.0) == math.inf


def test_information_content_grows_as_probability_falls():
    assert information_content(0.001) > information_content(0.1) > information_content(0.9)


def test_information_content_in_nats_is_smaller_than_in_bits():
    """1 бит = ln(2) ната — разница только в основании логарифма."""
    assert information_content(0.5, base=math.e) == APPROX(math.log(2))


# -------------------------------------------------------------- entropy
def test_entropy_of_a_fair_coin_is_one_bit():
    assert entropy([0.5, 0.5]) == APPROX(1.0)


def test_entropy_of_a_biased_coin_is_small():
    assert entropy([0.99, 0.01]) == pytest.approx(0.0807931, abs=1e-6)


def test_entropy_of_a_fair_die_is_log_of_six():
    assert entropy([1 / 6] * 6) == APPROX(math.log2(6))


def test_entropy_skips_zero_probability_outcomes():
    """Ловушка: 0 * inf даёт nan, слагаемое надо выбросить, а не считать."""
    result = entropy([1.0, 0.0])
    assert not math.isnan(result)
    assert result == APPROX(0.0)


def test_uniform_distribution_has_the_maximum_entropy():
    assert entropy([0.25] * 4) > entropy([0.7, 0.1, 0.1, 0.1])


def test_entropy_is_never_negative():
    assert entropy([0.7, 0.2, 0.1]) >= 0


# -------------------------------------------------------- cross_entropy
def test_cross_entropy_equals_entropy_when_the_model_is_perfect():
    p = [0.7, 0.2, 0.1]
    assert cross_entropy(p, p) == APPROX(entropy(p))


def test_cross_entropy_of_a_one_hot_target_against_a_uniform_model():
    assert cross_entropy([1.0, 0.0], [0.5, 0.5]) == APPROX(1.0)


def test_cross_entropy_punishes_the_worse_model():
    p = [0.7, 0.2, 0.1]
    assert cross_entropy(p, [0.6, 0.25, 0.15]) < cross_entropy(p, [0.1, 0.1, 0.8])


def test_cross_entropy_is_infinite_when_the_model_rules_out_a_real_outcome():
    assert cross_entropy([0.5, 0.5], [1.0, 0.0]) == math.inf


def test_cross_entropy_tolerates_a_zero_in_q_where_p_is_also_zero():
    """Событие не случается — то, что модель считает его невозможным, не беда."""
    assert cross_entropy([1.0, 0.0], [1.0, 0.0]) == APPROX(0.0)


def test_cross_entropy_is_not_symmetric():
    p, q = [0.7, 0.3], [0.1, 0.9]
    assert cross_entropy(p, q) != pytest.approx(cross_entropy(q, p), abs=1e-6)


# --------------------------------------------------------- kl_divergence
def test_kl_divergence_of_identical_distributions_is_zero():
    p = [0.7, 0.2, 0.1]
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-12)


def test_kl_divergence_is_never_negative():
    assert kl_divergence([0.7, 0.3], [0.6, 0.4]) >= 0
    assert kl_divergence([0.6, 0.4], [0.7, 0.3]) >= 0


def test_kl_divergence_equals_cross_entropy_minus_entropy():
    p, q = [0.7, 0.2, 0.1], [0.6, 0.25, 0.15]
    assert kl_divergence(p, q) == APPROX(cross_entropy(p, q) - entropy(p))


def test_kl_divergence_is_not_symmetric():
    """Главная ловушка: KL — не расстояние, порядок аргументов меняет ответ."""
    p, q = [0.7, 0.3], [0.1, 0.9]
    assert kl_divergence(p, q) != pytest.approx(kl_divergence(q, p), abs=1e-6)


def test_kl_divergence_grows_as_the_model_drifts_away():
    p = [0.7, 0.3]
    assert kl_divergence(p, [0.65, 0.35]) < kl_divergence(p, [0.3, 0.7])


# --------------------------------------------------------- joint_entropy
def test_joint_entropy_of_two_independent_fair_coins_is_two_bits():
    assert joint_entropy(INDEPENDENT) == APPROX(2.0)


def test_joint_entropy_of_a_perfectly_coupled_pair_is_one_bit():
    """Пара всегда одинакова — вторая переменная не добавляет ничего."""
    assert joint_entropy(IDENTICAL) == APPROX(1.0)


def test_joint_entropy_never_exceeds_the_sum_of_marginal_entropies():
    px = [sum(row) for row in DEPENDENT]
    py = [sum(col) for col in zip(*DEPENDENT)]
    assert joint_entropy(DEPENDENT) <= entropy(px) + entropy(py) + 1e-12


def test_joint_entropy_equals_the_sum_of_marginals_exactly_when_independent():
    px = [sum(row) for row in INDEPENDENT]
    py = [sum(col) for col in zip(*INDEPENDENT)]
    assert joint_entropy(INDEPENDENT) == APPROX(entropy(px) + entropy(py))


# --------------------------------------------------- conditional_entropy
def test_conditional_entropy_when_x_says_nothing_about_y():
    assert conditional_entropy(INDEPENDENT) == APPROX(1.0)


def test_conditional_entropy_when_x_determines_y_is_zero():
    assert conditional_entropy(IDENTICAL) == pytest.approx(0.0, abs=1e-12)


def test_conditional_entropy_never_exceeds_the_entropy_of_y():
    py = [sum(col) for col in zip(*DEPENDENT)]
    assert conditional_entropy(DEPENDENT) <= entropy(py) + 1e-12


def test_conditional_entropy_is_never_negative():
    assert conditional_entropy(DEPENDENT) >= -1e-12


def test_conditional_entropy_uses_rows_as_x_and_columns_as_y():
    """Несимметричная таблица: перепутаешь оси — получишь другое число."""
    joint = [[0.5, 0.0], [0.25, 0.25]]
    transposed = [list(col) for col in zip(*joint)]
    assert conditional_entropy(joint) != pytest.approx(
        conditional_entropy(transposed), abs=1e-6
    )


# --------------------------------------------------- mutual_information
def test_mutual_information_of_independent_variables_is_zero():
    assert mutual_information(INDEPENDENT) == pytest.approx(0.0, abs=1e-12)


def test_mutual_information_of_a_dependent_pair():
    assert mutual_information(DEPENDENT) == pytest.approx(0.5310044, abs=1e-6)


def test_mutual_information_of_a_variable_with_itself_is_its_entropy():
    assert mutual_information(IDENTICAL) == APPROX(1.0)


def test_mutual_information_is_symmetric():
    transposed = [list(col) for col in zip(*DEPENDENT)]
    assert mutual_information(DEPENDENT) == pytest.approx(
        mutual_information(transposed), abs=1e-9
    )


def test_mutual_information_is_never_negative():
    assert mutual_information([[0.1, 0.2], [0.3, 0.4]]) >= -1e-12


def test_mutual_information_ranks_the_informative_feature_higher():
    """Ровно так отбирают признаки: чем больше MI с таргетом, тем ценнее."""
    noise = [[0.25, 0.25], [0.25, 0.25]]
    signal = [[0.40, 0.10], [0.10, 0.40]]
    assert mutual_information(signal) > mutual_information(noise)


# ---------------------------------------------------------- perplexity
def test_perplexity_of_a_uniform_model_over_two_options_is_two():
    assert perplexity([1.0, 0.0], [0.5, 0.5]) == pytest.approx(2.0, abs=1e-9)


def test_perplexity_of_a_perfectly_certain_model_is_one():
    assert perplexity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0, abs=1e-9)


def test_perplexity_of_a_uniform_model_is_the_number_of_options():
    """Перплексия равномерной модели буквально равна размеру словаря."""
    assert perplexity([0.25] * 4, [0.25] * 4) == pytest.approx(4.0, abs=1e-9)


def test_perplexity_is_never_below_one():
    assert perplexity([0.7, 0.3], [0.6, 0.4]) >= 1.0


def test_perplexity_in_nats_matches_perplexity_in_bits():
    """Основание степени обязано совпадать с основанием логарифма."""
    p, q = [0.7, 0.3], [0.6, 0.4]
    assert perplexity(p, q, base=math.e) == pytest.approx(perplexity(p, q), abs=1e-9)


def test_perplexity_grows_when_the_model_gets_worse():
    p = [0.7, 0.3]
    assert perplexity(p, [0.65, 0.35]) < perplexity(p, [0.2, 0.8])
