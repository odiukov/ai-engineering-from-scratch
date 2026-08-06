"""Тесты к уроку «Диффузионные модели: DDPM с нуля». Правь exercise.py."""

import math
import random

import pytest

from exercise import (
    alpha_bars_from_betas,
    ddpm_loss,
    forward_sample,
    linear_beta_schedule,
    predict_x0,
    reverse_step,
    sample_chain,
    sinusoidal_embedding,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def mean_std(values):
    """Выборочные среднее и стандартное отклонение — нужны для проверок статистики."""
    m = sum(values) / len(values)
    var = sum((v - m) ** 2 for v in values) / len(values)
    return m, math.sqrt(var)


def correlation(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b)) / len(a)
    sa = math.sqrt(sum((x - ma) ** 2 for x in a) / len(a))
    sb = math.sqrt(sum((y - mb) ** 2 for y in b) / len(b))
    return cov / (sa * sb)


# ------------------------------------------------------ linear_beta_schedule
def test_beta_schedule_spans_the_requested_range():
    betas = linear_beta_schedule(5, 0.0, 1.0)
    assert betas[0] == APPROX(0.0)
    assert betas[-1] == APPROX(1.0)


def test_beta_schedule_is_evenly_spaced():
    assert linear_beta_schedule(3, 0.0, 1.0) == pytest.approx([0.0, 0.5, 1.0])


def test_beta_schedule_increases_monotonically():
    """Шум обязан нарастать: ранние шаги мягкие, поздние — грубые."""
    betas = linear_beta_schedule(1000)
    assert all(betas[i] < betas[i + 1] for i in range(len(betas) - 1))


def test_beta_schedule_with_one_step_does_not_divide_by_zero():
    """Делитель T - 1 обнуляется при T = 1 — этот случай нужен отдельно."""
    assert linear_beta_schedule(1, 0.2, 0.9) == pytest.approx([0.2])


# ------------------------------------------------------ alpha_bars_from_betas
def test_alpha_bars_are_cumulative_products():
    assert alpha_bars_from_betas([0.1, 0.1]) == pytest.approx([0.9, 0.81])


def test_alpha_bars_start_at_one_minus_the_first_beta():
    assert alpha_bars_from_betas([0.25, 0.5, 0.5])[0] == APPROX(0.75)


def test_alpha_bars_decrease_monotonically():
    """Сигнала от x_0 в x_t становится только меньше, никогда больше."""
    bars = alpha_bars_from_betas(linear_beta_schedule(200))
    assert all(bars[i] > bars[i + 1] for i in range(len(bars) - 1))


def test_alpha_bars_reach_almost_zero_over_a_thousand_steps():
    """Смысл расписания DDPM: к t = T от исходных данных не остаётся ничего."""
    bars = alpha_bars_from_betas(linear_beta_schedule(1000))
    assert bars[-1] < 1e-3


# ------------------------------------------------------------ forward_sample
def test_forward_sample_without_noise_returns_the_source():
    x_t, _ = forward_sample(3.0, 0, [1.0], random.Random(0))
    assert x_t == APPROX(3.0)


def test_forward_sample_returns_the_noise_it_actually_used():
    """Без возвращённого eps нечего подставлять в loss — это не деталь, а контракт."""
    bars = [0.36]
    x_t, eps = forward_sample(2.0, 0, bars, random.Random(1))
    assert x_t == pytest.approx(math.sqrt(0.36) * 2.0 + math.sqrt(0.64) * eps, abs=1e-12)


def test_forward_sample_statistics_match_the_closed_form():
    """Среднее ~ sqrt(alpha_bar) * x_0, разброс ~ sqrt(1 - alpha_bar)."""
    bars = alpha_bars_from_betas(linear_beta_schedule(100))
    t, x0 = 99, 1.5
    rng = random.Random(3)
    xs = [forward_sample(x0, t, bars, rng)[0] for _ in range(3000)]
    m, s = mean_std(xs)
    assert m == pytest.approx(math.sqrt(bars[t]) * x0, abs=0.05)
    assert s == pytest.approx(math.sqrt(1 - bars[t]), abs=0.05)


def test_closed_form_matches_stepwise_noising():
    """Главное утверждение урока: один шаг по alpha_bar == t шагов цепочки q."""
    betas = linear_beta_schedule(100)
    bars = alpha_bars_from_betas(betas)
    t, x0, n = 99, 1.5, 3000

    rng = random.Random(3)
    closed = [forward_sample(x0, t, bars, rng)[0] for _ in range(n)]

    rng = random.Random(4)
    stepwise = []
    for _ in range(n):
        x = x0
        for s in range(t + 1):
            x = math.sqrt(1 - betas[s]) * x + math.sqrt(betas[s]) * rng.gauss(0, 1)
        stepwise.append(x)

    m_closed, s_closed = mean_std(closed)
    m_step, s_step = mean_std(stepwise)
    assert m_closed == pytest.approx(m_step, abs=0.05)
    assert s_closed == pytest.approx(s_step, abs=0.05)


def test_noised_sample_loses_correlation_with_the_source_at_large_t():
    """К концу расписания x_t уже не помнит, из чего его сделали."""
    bars = alpha_bars_from_betas(linear_beta_schedule(1000))
    rng = random.Random(1)
    sources = [rng.gauss(0, 1) for _ in range(2000)]

    rng = random.Random(7)
    early = [forward_sample(x, 0, bars, rng)[0] for x in sources]
    rng = random.Random(7)
    late = [forward_sample(x, 999, bars, rng)[0] for x in sources]

    assert correlation(sources, early) > 0.99
    assert abs(correlation(sources, late)) < 0.2


# ------------------------------------------------------- sinusoidal_embedding
def test_embedding_has_the_requested_length():
    assert len(sinusoidal_embedding(5, 8)) == 8


def test_embedding_at_step_zero_alternates_zero_and_one():
    """sin(0) = 0, cos(0) = 1 на всех частотах."""
    assert sinusoidal_embedding(0, 4) == pytest.approx([0.0, 1.0, 0.0, 1.0])


def test_embedding_stays_bounded():
    """Синусы и косинусы не выходят за [-1, 1] — вход сети не взрывается."""
    emb = sinusoidal_embedding(999, 16)
    assert all(-1.0 <= v <= 1.0 for v in emb)


def test_nearby_timesteps_have_more_similar_embeddings():
    """Смысл вложения: близкие уровни шума должны выглядеть похоже."""
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    e5 = sinusoidal_embedding(5, 8)
    assert dot(e5, sinusoidal_embedding(6, 8)) > dot(e5, sinusoidal_embedding(25, 8))


def test_embedding_of_dimension_two_does_not_divide_by_zero():
    """При dim = 2 показатель i / (half - 1) делит на ноль без защиты."""
    assert sinusoidal_embedding(3, 2) == pytest.approx([math.sin(3.0), math.cos(3.0)])


# ---------------------------------------------------------------- ddpm_loss
def test_loss_is_zero_for_a_perfect_prediction():
    assert ddpm_loss([1.0, -1.0], [1.0, -1.0]) == APPROX(0.0)


def test_loss_is_the_mean_squared_error():
    assert ddpm_loss([0.0, 0.0], [2.0, 0.0]) == APPROX(2.0)


def test_loss_does_not_grow_with_dimension():
    """Среднее, а не сумма: иначе 5-мерные данные «дороже» одномерных."""
    assert ddpm_loss([1.0] * 5, [0.0] * 5) == APPROX(ddpm_loss([1.0], [0.0]))


# --------------------------------------------------------------- predict_x0
def test_predict_x0_inverts_forward_sample():
    """Знаем настоящий eps — восстанавливаем x_0 точно."""
    bars = alpha_bars_from_betas(linear_beta_schedule(50))
    rng = random.Random(11)
    x0 = -0.7
    x_t, eps = forward_sample(x0, 30, bars, rng)
    assert predict_x0(x_t, 30, eps, bars) == pytest.approx(x0, abs=1e-9)


def test_predict_x0_is_identity_when_nothing_was_added():
    assert predict_x0(3.0, 0, 0.0, [1.0]) == APPROX(3.0)


def test_predict_x0_is_unstable_when_no_signal_is_left():
    """При alpha_bar -> 0 деление на sqrt(alpha_bar) разносит оценку. Так и надо."""
    assert abs(predict_x0(1.0, 0, 0.0, [1e-8])) > 1000


# -------------------------------------------------------------- reverse_step
def test_reverse_step_matches_the_posterior_mean_formula():
    betas, bars = [0.1], [0.9]
    expected = (0.5 - 0.1 / math.sqrt(1 - 0.9) * 0.2) / math.sqrt(0.9)
    assert reverse_step(0.5, 0, 0.2, betas, bars, random.Random(0)) == APPROX(expected)


def test_reverse_step_at_the_last_step_does_not_add_noise():
    """t == 0 обязан быть детерминированным и вообще не трогать rng."""
    betas, bars = [0.1], [0.9]
    used = random.Random(5)
    reverse_step(0.5, 0, 0.2, betas, bars, used)
    assert used.gauss(0, 1) == APPROX(random.Random(5).gauss(0, 1))


def test_reverse_step_adds_noise_while_t_is_positive():
    betas = linear_beta_schedule(10)
    bars = alpha_bars_from_betas(betas)
    a = reverse_step(0.5, 5, 0.2, betas, bars, random.Random(1))
    b = reverse_step(0.5, 5, 0.2, betas, bars, random.Random(2))
    assert a != b


def test_reverse_step_walks_back_along_the_noise_free_trajectory():
    """Если x_t ровно на траектории sqrt(alpha_bar)*c и шум предсказан нулём,
    последний шаг обязан вернуть ровно c."""
    betas = linear_beta_schedule(40)
    bars = alpha_bars_from_betas(betas)
    c = 1.7
    x_t = math.sqrt(bars[0]) * c
    assert reverse_step(x_t, 0, 0.0, betas, bars, random.Random(0)) == pytest.approx(c, abs=1e-12)


# -------------------------------------------------------------- sample_chain
def test_sample_chain_is_reproducible_for_a_fixed_seed():
    betas = linear_beta_schedule(20)
    bars = alpha_bars_from_betas(betas)
    model = lambda x, t: 0.0
    assert sample_chain(model, betas, bars, random.Random(9)) == APPROX(
        sample_chain(model, betas, bars, random.Random(9))
    )


def test_sample_chain_differs_across_seeds():
    """Сэмплер стохастический: разные seed — разные картинки."""
    betas = linear_beta_schedule(20)
    bars = alpha_bars_from_betas(betas)
    model = lambda x, t: 0.0
    a = sample_chain(model, betas, bars, random.Random(9))
    b = sample_chain(model, betas, bars, random.Random(10))
    assert a != b


def test_perfect_denoiser_lands_exactly_on_the_data_point():
    """Данные — одна точка c. Идеальный eps-предсказатель существует в явном
    виде, и обратная цепочка обязана сойтись ровно в c."""
    betas = linear_beta_schedule(1000)
    bars = alpha_bars_from_betas(betas)
    c = 1.7

    def oracle(x, t):
        return (x - math.sqrt(bars[t]) * c) / math.sqrt(1 - bars[t])

    rng = random.Random(0)
    samples = [sample_chain(oracle, betas, bars, rng) for _ in range(10)]
    assert all(s == pytest.approx(c, abs=1e-6) for s in samples)
