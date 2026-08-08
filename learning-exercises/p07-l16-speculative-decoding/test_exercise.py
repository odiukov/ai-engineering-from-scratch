"""Тесты к уроку «Speculative decoding: draft, verify, repeat». Правь exercise.py."""

import math
import random

import pytest

from exercise import (
    accept_draft,
    acceptance_probability,
    expected_tokens_per_verify,
    kl_divergence,
    residual_dist,
    rollback_kv,
    sample_from,
    spec_step,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# Распределение «большой» модели из урока.
Q = [0.35, 0.20, 0.15, 0.10, 0.08, 0.06, 0.04, 0.02]


def blend(dist, t):
    """Детерминированная «порча» распределения: смесь с равномерным на долю t."""
    u = 1.0 / len(dist)
    return [(1.0 - t) * x + t * u for x in dist]


def chi_square(counts, probs):
    """Статистика хи-квадрат наблюдённых частот против ожидаемых вероятностей."""
    n = sum(counts)
    return sum((c - n * p) ** 2 / (n * p) for c, p in zip(counts, probs) if p > 0)


# ------------------------------------------------------------ sample_from
def test_sample_from_degenerate_distribution_is_deterministic():
    rng = random.Random(0)
    assert [sample_from([0.0, 1.0, 0.0], rng) for _ in range(20)] == [1] * 20


def test_sample_from_never_returns_an_out_of_range_index():
    rng = random.Random(1)
    assert all(0 <= sample_from(Q, rng) < len(Q) for _ in range(500))


def test_sample_from_reproduces_the_distribution():
    """20000 тяг по фиксированному seed — частоты обязаны сойтись к Q."""
    rng = random.Random(7)
    counts = [0] * len(Q)
    for _ in range(20000):
        counts[sample_from(Q, rng)] += 1
    for c, p in zip(counts, Q):
        assert c / 20000 == pytest.approx(p, abs=0.012)


def test_sample_from_is_reproducible_for_the_same_seed():
    """Никакого глобального random: один seed — одна последовательность."""
    rng_a, rng_b = random.Random(42), random.Random(42)
    a = [sample_from(Q, rng_a) for _ in range(50)]
    b = [sample_from(Q, rng_b) for _ in range(50)]
    assert a == b
    assert len(set(a)) > 1  # иначе тест ничего не проверяет


# ----------------------------------------------------------- residual_dist
def test_residual_is_a_probability_distribution():
    r = residual_dist(Q, blend(Q, 0.5))
    assert sum(r) == APPROX(1.0)
    assert all(x >= 0.0 for x in r)


def test_residual_is_zero_where_the_draft_already_covers_the_verifier():
    """Там, где p >= q, добирать нечего — вероятность ровно 0.0."""
    q = [0.5, 0.5]
    p = [0.9, 0.1]
    assert residual_dist(q, p) == [0.0, 1.0]


def test_residual_of_identical_distributions_falls_back_to_q():
    """Все разности нули, нормировать нечего — делить на нуль нельзя."""
    assert residual_dist(Q, list(Q)) == pytest.approx(Q)


def test_min_plus_residual_reconstructs_the_verifier_exactly():
    """Тождество, на котором стоит вся теорема: min(p,q) + (q-p)+ == q."""
    p = blend(Q, 0.6)
    alpha = acceptance_probability(Q, p)
    r = residual_dist(Q, p)
    for i, qi in enumerate(Q):
        assert min(qi, p[i]) + r[i] * (1.0 - alpha) == pytest.approx(qi, abs=1e-12)


# ------------------------------------------------------------ accept_draft
def test_draft_is_always_accepted_when_verifier_likes_it_more():
    """q/p >= 1 -> min даёт 1 -> принимаем при любом u из [0, 1)."""
    assert accept_draft(0.4, 0.2, 0.0) is True
    assert accept_draft(0.4, 0.2, 0.999999) is True


def test_acceptance_threshold_is_the_probability_ratio():
    assert accept_draft(0.1, 0.4, 0.2) is True
    assert accept_draft(0.1, 0.4, 0.3) is False


def test_zero_draft_probability_is_accepted_instead_of_dividing_by_zero():
    """Квантованный draft умеет выдать токен с p = 0. Делить нельзя."""
    assert accept_draft(0.0001, 0.0, 0.9999) is True


def test_swapping_q_and_p_changes_the_verdict():
    """Порядок аргументов — не косметика: перепутал, и декодер тихо врёт."""
    assert accept_draft(0.1, 0.4, 0.5) is False
    assert accept_draft(0.4, 0.1, 0.5) is True


# --------------------------------------------------- acceptance_probability
def test_identical_models_accept_everything():
    assert acceptance_probability(Q, list(Q)) == APPROX(1.0)


def test_disjoint_models_accept_nothing():
    assert acceptance_probability([1.0, 0.0], [0.0, 1.0]) == APPROX(0.0)


def test_acceptance_is_one_minus_total_variation_distance():
    p = blend(Q, 0.5)
    tv = 0.5 * sum(abs(a - b) for a, b in zip(Q, p))
    assert acceptance_probability(Q, p) == pytest.approx(1.0 - tv, abs=1e-12)


def test_analytic_acceptance_matches_the_simulated_one():
    """Формула sum min(q, p) обязана совпасть с долей принятых в прогоне."""
    p = blend(Q, 0.5)
    rng = random.Random(2024)
    accepted = drafted = 0
    for _ in range(4000):
        _, n_acc = spec_step(Q, p, 1, rng)
        accepted += n_acc
        drafted += 1
    assert accepted / drafted == pytest.approx(acceptance_probability(Q, p), abs=0.02)


def test_worse_draft_accepts_less():
    rates = [acceptance_probability(Q, blend(Q, t)) for t in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert rates == sorted(rates, reverse=True)


# --------------------------------------------------------------- spec_step
def test_step_emits_one_more_token_than_it_accepted():
    """Инвариант: принятые черновики плюс один — либо исправление, либо бонус."""
    rng = random.Random(5)
    p = blend(Q, 0.5)
    for _ in range(300):
        emitted, n_acc = spec_step(Q, p, 4, rng)
        assert len(emitted) == n_acc + 1
        assert 0 <= n_acc <= 4


def test_identical_draft_accepts_every_token_and_adds_the_bonus():
    """p == q -> alpha = 1 -> все 5 черновиков приняты плюс бесплатный шестой."""
    rng = random.Random(6)
    for _ in range(100):
        emitted, n_acc = spec_step(Q, list(Q), 5, rng)
        assert n_acc == 5
        assert len(emitted) == 6


def test_disjoint_draft_accepts_nothing_and_falls_back_to_the_residual():
    """Черновик из другого множества токенов бесполезен: остаётся ровно один токен из q."""
    q = [0.5, 0.5, 0.0, 0.0]
    p = [0.0, 0.0, 0.5, 0.5]
    rng = random.Random(8)
    for _ in range(200):
        emitted, n_acc = spec_step(q, p, 5, rng)
        assert n_acc == 0
        assert emitted[0] in (0, 1)


def test_speculative_output_matches_the_verifier_distribution():
    """Теорема Leviathan: распределение выданных токенов — ровно q, не приближённо."""
    p = blend(Q, 0.5)
    rng = random.Random(1234)
    counts = [0] * len(Q)
    for _ in range(6000):
        emitted, _ = spec_step(Q, p, 4, rng)
        for token in emitted:
            counts[token] += 1
    assert sum(counts) > 20000
    assert chi_square(counts, Q) < 30.0


def test_bad_draft_still_matches_the_verifier_distribution():
    """Даже с плохим draft-ом качество не падает — падает только скорость."""
    p = blend(Q, 1.0)  # равномерный draft, alpha = 0.675
    rng = random.Random(999)
    counts = [0] * len(Q)
    for _ in range(6000):
        emitted, _ = spec_step(Q, p, 4, rng)
        for token in emitted:
            counts[token] += 1
    assert chi_square(counts, Q) < 30.0


def test_mean_tokens_per_step_matches_the_theory():
    """Среднее len(emitted) обязано совпасть с (1 - alpha^(N+1)) / (1 - alpha)."""
    p = blend(Q, 0.5)
    alpha = acceptance_probability(Q, p)
    rng = random.Random(1234)
    total = 0
    steps = 6000
    for _ in range(steps):
        emitted, _ = spec_step(Q, p, 4, rng)
        total += len(emitted)
    assert total / steps == pytest.approx(expected_tokens_per_verify(alpha, 4), abs=0.05)


def test_better_draft_produces_more_tokens_per_step():
    counts = []
    for t in (0.15, 0.5, 1.0):
        rng = random.Random(77)
        p = blend(Q, t)
        counts.append(sum(len(spec_step(Q, p, 5, rng)[0]) for _ in range(2000)))
    assert counts == sorted(counts, reverse=True)


# ---------------------------------------------- expected_tokens_per_verify
def test_zero_acceptance_still_yields_one_token():
    """Даже когда всё отвергнуто, шаг выдаёт исправляющий токен. Прогресс есть всегда."""
    assert expected_tokens_per_verify(0.0, 5) == APPROX(1.0)


def test_perfect_acceptance_hits_the_ceiling():
    """alpha = 1 — знаменатель нулевой, а ответ n_draft + 1."""
    assert expected_tokens_per_verify(1.0, 5) == APPROX(6.0)


def test_typical_production_setting():
    """alpha = 0.85, N = 5: примерно 4.15 токена на один forward большой модели."""
    assert expected_tokens_per_verify(0.85, 5) == pytest.approx(4.1523, abs=1e-4)


def test_gain_saturates_as_draft_length_grows():
    """После 1/(1-alpha) добавлять черновики бессмысленно — выигрыш выходит на плато."""
    alpha = 0.75
    ceiling = 1.0 / (1.0 - alpha)
    assert expected_tokens_per_verify(alpha, 100) == pytest.approx(ceiling, abs=1e-9)
    assert expected_tokens_per_verify(alpha, 40) < ceiling


def test_gain_grows_with_acceptance_rate():
    values = [expected_tokens_per_verify(a, 5) for a in (0.3, 0.5, 0.7, 0.85, 0.95)]
    assert values == sorted(values)


# ------------------------------------------------------------ kl_divergence
def test_kl_of_a_distribution_with_itself_is_zero():
    assert kl_divergence(Q, list(Q)) == APPROX(0.0)


def test_kl_is_never_negative():
    """Неравенство Гиббса: KL >= 0 для любой пары распределений."""
    for t in (0.1, 0.3, 0.6, 1.0):
        assert kl_divergence(Q, blend(Q, t)) >= 0.0


def test_kl_skips_zero_mass_in_the_first_distribution():
    """0 * log(0/q) равно нулю по соглашению, включая случай 0/0."""
    assert kl_divergence([0.0, 1.0], [0.5, 0.5]) == pytest.approx(0.6931471805599453)
    assert kl_divergence([0.0, 1.0], [0.0, 1.0]) == APPROX(0.0)


def test_kl_is_infinite_when_the_second_distribution_misses_support():
    """Положительную массу нельзя сравнить с нулём, просто выкинув слагаемое."""
    assert kl_divergence([0.25, 0.75], [0.0, 1.0]) == math.inf


def test_kl_grows_as_the_draft_drifts_away():
    kls = [kl_divergence(Q, blend(Q, t)) for t in (0.1, 0.25, 0.5, 0.75, 1.0)]
    assert kls == sorted(kls)


def test_acceptance_falls_as_kl_grows():
    """Та самая монотонная связь, которую урок просит построить."""
    pairs = [
        (kl_divergence(Q, blend(Q, t)), acceptance_probability(Q, blend(Q, t)))
        for t in (0.1, 0.25, 0.5, 0.75, 1.0)
    ]
    for (kl_a, alpha_a), (kl_b, alpha_b) in zip(pairs, pairs[1:]):
        assert kl_b > kl_a
        assert alpha_b < alpha_a


# -------------------------------------------------------------- rollback_kv
def test_rollback_keeps_prefix_plus_accepted_drafts():
    assert rollback_kv(["a", "b", "c", "d", "e"], 2, 1) == ["a", "b", "c"]


def test_rollback_of_a_full_acceptance_keeps_everything():
    cache = list(range(15))
    assert rollback_kv(cache, 10, 5) == cache


def test_rejection_at_position_three_leaves_two_drafts():
    """Сценарий из урока: префикс 10, черновиков 5, отказ на третьем."""
    cache = [f"kv{i}" for i in range(15)]
    assert rollback_kv(cache, 10, 2) == [f"kv{i}" for i in range(12)]


def test_rollback_does_not_mutate_the_original_cache():
    """Продакшн держит черновики в scratch-буфере именно поэтому."""
    cache = list(range(15))
    rolled = rollback_kv(cache, 10, 2)
    rolled.append(999)
    assert len(cache) == 15


def test_full_rejection_rolls_back_to_the_bare_prefix():
    assert rollback_kv(list(range(15)), 10, 0) == list(range(10))
