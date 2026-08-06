"""Тесты к уроку «Методы сэмплирования». Правь exercise.py."""

import math
import random

import pytest

from exercise import (
    monte_carlo_pi,
    rejection_sample,
    sample_exponential,
    sample_index,
    sample_token,
    softmax_with_temperature,
    top_k_filter,
    top_p_filter,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


class FixedRng:
    """Поддельный генератор: всегда отдаёт одно и то же число.

    Нужен, чтобы детерминированно проверить крайние значения u, которые
    настоящий random.Random выдаёт раз в миллиард лет.
    """

    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


def finite_count(logits):
    """Сколько логитов не отфильтровано (не равны -inf)."""
    return sum(1 for z in logits if z != -math.inf)


# ------------------------------------------------ softmax_with_temperature
def test_softmax_probabilities_sum_to_one():
    """Свойство, которое обязано выполняться при любой температуре."""
    for temperature in (0.1, 0.5, 1.0, 2.0, 10.0):
        probs = softmax_with_temperature([2.0, 1.0, 0.0, -3.0], temperature)
        assert sum(probs) == pytest.approx(1.0, abs=1e-12)
        assert all(p >= 0.0 for p in probs)
    assert softmax_with_temperature([0.0, 0.0, 0.0]) == APPROX([1 / 3, 1 / 3, 1 / 3])


def test_temperature_below_one_sharpens_distribution():
    """T = 0.5 растягивает разрыв между логитами: лидер забирает больше массы."""
    base = softmax_with_temperature([2.0, 1.0])
    sharp = softmax_with_temperature([2.0, 1.0], 0.5)
    assert base == APPROX([0.7310585786300049, 0.2689414213699951])
    assert sharp == APPROX([0.8807970779778823, 0.11920292202211755])
    assert sharp[0] > base[0]


def test_temperature_above_one_flattens_distribution():
    """T = 5 подтягивает вероятности друг к другу, но порядок не меняет."""
    base = softmax_with_temperature([3.0, 0.0])
    flat = softmax_with_temperature([3.0, 0.0], 5.0)
    assert flat[0] < base[0]
    assert flat[0] > flat[1]


def test_softmax_does_not_overflow_on_huge_logits():
    """Ловушка: exp(1000) роняет программу, если не вычесть максимум."""
    probs = softmax_with_temperature([1000.0, 999.0, 998.0])
    assert sum(probs) == pytest.approx(1.0, abs=1e-12)
    assert probs == APPROX(softmax_with_temperature([2.0, 1.0, 0.0]))


def test_softmax_rejects_non_positive_temperature():
    """Ловушка: T = 0 это деление на ноль, а не «жадный выбор»."""
    with pytest.raises(ValueError):
        softmax_with_temperature([1.0, 2.0], 0.0)
    with pytest.raises(ValueError):
        softmax_with_temperature([1.0, 2.0], -1.0)


# ------------------------------------------------------------ sample_index
def test_sample_index_always_picks_the_only_possible_token():
    rng = random.Random(1)
    assert all(sample_index([1.0, 0.0, 0.0], rng) == 0 for _ in range(200))


def test_sample_index_never_picks_a_zero_probability_token():
    """Ловушка: при u = 0.0 сравнение через <= вернуло бы индекс 0 с p = 0."""
    assert sample_index([0.0, 1.0], FixedRng(0.0)) == 1
    assert sample_index([0.0, 0.0, 1.0], FixedRng(0.0)) == 2


def test_sample_index_frequencies_match_probabilities():
    """Доли выпадений сходятся к самим вероятностям."""
    rng = random.Random(0)
    probs = [0.5, 0.3, 0.2]
    counts = [0, 0, 0]
    n = 20000
    for _ in range(n):
        i = sample_index(probs, rng)
        assert 0 <= i < 3
        counts[i] += 1
    for got, want in zip(counts, probs):
        assert got / n == pytest.approx(want, abs=0.02)


def test_sample_index_survives_probabilities_that_sum_below_one():
    """Ловушка: из-за float сумма бывает 0.9999999 — вернуть индекс всё равно надо."""
    almost = [0.3333333333, 0.3333333333, 0.3333333333]
    rng = random.Random(3)
    for _ in range(500):
        assert sample_index(almost, rng) in (0, 1, 2)
    assert sample_index([0.5, 0.499999], FixedRng(0.9999999999)) == 1


# ------------------------------------------------------ sample_exponential
def test_exponential_mean_converges_to_one_over_lambda():
    rng = random.Random(0)
    values = [sample_exponential(2.0, rng) for _ in range(20000)]
    assert all(v >= 0.0 for v in values)
    assert sum(values) / len(values) == pytest.approx(0.5, abs=0.02)


def test_exponential_scales_inversely_with_lambda():
    """Один и тот же u при lam = 2 даёт ровно вдвое меньшее значение."""
    a = sample_exponential(1.0, random.Random(0))
    b = sample_exponential(2.0, random.Random(0))
    assert a == pytest.approx(1.8606071110652234, abs=1e-9)
    assert b == pytest.approx(a / 2, abs=1e-12)


def test_exponential_survives_u_equal_to_zero():
    """Ловушка: rng.random() может вернуть ровно 0.0, а ln(0) падает."""
    assert sample_exponential(1.0, FixedRng(0.0)) == APPROX(0.0)


def test_exponential_rejects_non_positive_lambda():
    with pytest.raises(ValueError):
        sample_exponential(0.0, random.Random(0))
    with pytest.raises(ValueError):
        sample_exponential(-1.0, random.Random(0))


# ----------------------------------------------------------- monte_carlo_pi
def test_monte_carlo_pi_is_close_to_pi():
    assert monte_carlo_pi(20000, random.Random(0)) == pytest.approx(math.pi, abs=0.05)


def test_monte_carlo_pi_returns_a_multiple_of_four_over_n():
    """Результат — доля попавших точек, умноженная на 4, значит кратен 4/n."""
    n = 500
    value = monte_carlo_pi(n, random.Random(11))
    assert 0.0 <= value <= 4.0
    assert (value * n / 4.0) == pytest.approx(round(value * n / 4.0), abs=1e-9)


def test_monte_carlo_pi_error_shrinks_with_more_samples():
    """Ошибка падает как O(1/sqrt(n)): в среднем по нескольким seed'ам заметно."""
    small = sum(abs(monte_carlo_pi(200, random.Random(s)) - math.pi) for s in range(8))
    large = sum(abs(monte_carlo_pi(20000, random.Random(s)) - math.pi) for s in range(8))
    assert large < small


def test_monte_carlo_pi_rejects_non_positive_n():
    """Ловушка: n = 0 это деление на ноль."""
    with pytest.raises(ValueError):
        monte_carlo_pi(0, random.Random(0))


# --------------------------------------------------------- rejection_sample
def test_rejection_sample_reproduces_a_triangular_density():
    """Для pdf(x) = 2x на [0, 1] среднее равно 2/3, а не 1/2."""
    rng = random.Random(1)
    values = [rejection_sample(lambda t: 2 * t, 0.0, 1.0, 2.0, rng) for _ in range(20000)]
    assert all(0.0 <= x <= 1.0 for x in values)
    assert sum(values) / len(values) == pytest.approx(2 / 3, abs=0.02)


def test_rejection_sample_with_flat_density_is_uniform_on_the_interval():
    """Равномерная цель на [-2, 3]: среднее в середине отрезка."""
    rng = random.Random(0)
    values = [rejection_sample(lambda t: 1.0, -2.0, 3.0, 1.0, rng) for _ in range(20000)]
    assert all(-2.0 <= x <= 3.0 for x in values)
    assert sum(values) / len(values) == pytest.approx(0.5, abs=0.05)


def test_rejection_sample_never_returns_where_density_is_zero():
    """Точное свойство при любом seed: нулевая плотность не принимается никогда."""
    pdf = lambda t: 1.0 if 0.4 <= t <= 0.6 else 0.0
    rng = random.Random(2)
    for _ in range(300):
        assert 0.4 <= rejection_sample(pdf, 0.0, 1.0, 1.0, rng) <= 0.6


def test_rejection_sample_raises_when_nothing_can_be_accepted():
    """Ловушка: с нулевой pdf цикл while True не кончится никогда.

    Сначала проверяем, что обычный вызов вообще что-то возвращает: иначе
    любая нереализованная функция «пройдёт» проверку на исключение.
    """
    assert rejection_sample(lambda t: 1.0, 0.0, 1.0, 1.0, random.Random(0)) is not None
    with pytest.raises(RuntimeError):
        rejection_sample(lambda t: 0.0, 0.0, 1.0, 1.0, random.Random(0), max_tries=50)


def test_rejection_sample_rejects_non_positive_bound():
    with pytest.raises(ValueError):
        rejection_sample(lambda t: 1.0, 0.0, 1.0, 0.0, random.Random(0))


# -------------------------------------------------------------- top_k_filter
def test_top_k_keeps_exactly_k_largest_logits():
    assert top_k_filter([1.0, 3.0, 2.0], 2) == [-math.inf, 3.0, 2.0]
    assert top_k_filter([1.0, 3.0, 2.0], 1) == [-math.inf, 3.0, -math.inf]
    assert finite_count(top_k_filter([0.1, 5.0, 2.0, -1.0, 3.0], 3)) == 3


def test_top_k_does_not_mutate_the_input():
    """Ловушка: логиты приходят из модели, портить их нельзя — даже при k
    больше длины словаря, когда фильтровать нечего."""
    logits = [1.0, 3.0, 2.0]
    top_k_filter(logits, 1)
    assert logits == [1.0, 3.0, 2.0]
    wide = top_k_filter(logits, 9)
    assert wide == [1.0, 3.0, 2.0]
    assert wide is not logits


def test_top_k_rejects_non_positive_k():
    """Ловушка: k = 0 оставил бы пустой набор кандидатов."""
    with pytest.raises(ValueError):
        top_k_filter([1.0, 2.0], 0)


def test_top_k_filtered_tokens_get_zero_probability():
    """-inf выбран не случайно: exp(-inf) = 0, индексы остальных не съезжают."""
    probs = softmax_with_temperature(top_k_filter([1.0, 3.0, 2.0], 1))
    assert probs == APPROX([0.0, 1.0, 0.0])


# -------------------------------------------------------------- top_p_filter
def test_top_p_keeps_the_smallest_set_reaching_p():
    """Вероятности [0.665, 0.245, 0.090]: до 0.7 хватает двух токенов."""
    assert top_p_filter([2.0, 1.0, 0.0], 0.7) == [2.0, 1.0, -math.inf]
    assert top_p_filter([2.0, 1.0, 0.0], 0.5) == [2.0, -math.inf, -math.inf]
    assert top_p_filter([2.0, 1.0, 0.0], 1.0) == [2.0, 1.0, 0.0]


def test_top_p_nucleus_is_never_empty():
    """Ловушка: если проверять сумму ДО добавления, при уверенной модели
    набор окажется пустым и сэмплировать будет не из чего."""
    out = top_p_filter([20.0, 0.0, 0.0], 0.5)
    assert finite_count(out) == 1
    assert out[0] == 20.0


def test_top_p_keeps_more_tokens_when_the_model_is_uncertain():
    """Главное отличие от top-k: размер набора подстраивается под уверенность."""
    confident = top_p_filter([10.0, 0.0, 0.0, 0.0, 0.0], 0.9)
    unsure = top_p_filter([0.0, 0.0, 0.0, 0.0, 0.0], 0.9)
    assert finite_count(confident) < finite_count(unsure)


def test_top_p_does_not_mutate_the_input():
    logits = [2.0, 1.0, 0.0]
    top_p_filter(logits, 0.5)
    assert logits == [2.0, 1.0, 0.0]


def test_top_p_rejects_p_outside_the_unit_interval():
    with pytest.raises(ValueError):
        top_p_filter([1.0, 2.0], 0.0)
    with pytest.raises(ValueError):
        top_p_filter([1.0, 2.0], 1.5)


# --------------------------------------------------------------- sample_token
def test_top_k_one_makes_decoding_greedy():
    """k = 1 — это жадный декодинг: тот же ответ при любом seed."""
    for seed in range(5):
        rng = random.Random(seed)
        assert sample_token([1.0, 5.0, 4.0], rng, top_k=1) == 1


def test_sample_token_never_picks_a_token_cut_by_the_filters():
    rng = random.Random(0)
    for _ in range(300):
        assert sample_token([1.0, 0.9, 0.8], rng, top_k=2) != 2
        assert sample_token([20.0, 0.0, 0.0], rng, top_p=0.5) == 0


def test_sample_token_without_filters_can_reach_every_token():
    """Ловушка: None означает «фильтр выключен», а не «оставить ноль токенов»."""
    rng = random.Random(0)
    seen = {sample_token([0.0, 0.0, 0.0], rng) for _ in range(300)}
    assert seen == {0, 1, 2}
    assert all(sample_token([1.0, 2.0, 3.0], rng) in (0, 1, 2) for _ in range(200))


def test_low_temperature_almost_always_picks_the_argmax():
    rng = random.Random(0)
    picks = [sample_token([5.0, 0.0, -1.0], rng, temperature=0.05) for _ in range(200)]
    assert set(picks) == {0}
