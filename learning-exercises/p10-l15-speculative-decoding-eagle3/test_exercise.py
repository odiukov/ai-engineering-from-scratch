"""Тесты к уроку «Спекулятивное декодирование и EAGLE-3». Правь exercise.py."""

import random

import pytest

from exercise import (
    accept,
    best_draft_length,
    expected_emitted,
    normalize,
    residual,
    sample_index,
    spec_step,
    time_per_token,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

Q = [0.4, 0.3, 0.2, 0.1]  # распределение верификатора
P = [0.35, 0.25, 0.25, 0.15]  # распределение черновика


# ---------------------------------------------------------------- normalize
def test_normalize_makes_a_distribution():
    assert normalize([1.0, 3.0]) == APPROX([0.25, 0.75])


def test_normalize_keeps_the_proportions():
    out = normalize([2.0, 4.0, 6.0])
    assert out[1] / out[0] == APPROX(2.0)
    assert sum(out) == APPROX(1.0)


def test_normalize_rejects_an_all_zero_vector():
    with pytest.raises(ValueError):
        normalize([0.0, 0.0])


def test_normalize_rejects_a_negative_weight_even_when_the_sum_is_positive():
    with pytest.raises(ValueError):
        normalize([-1.0, 3.0])


# ------------------------------------------------------------- sample_index
def test_sample_index_at_zero_takes_the_first_reachable_token():
    assert sample_index([0.5, 0.5], 0.0) == 0


def test_sample_index_crosses_the_boundary_at_the_cumulative_sum():
    assert sample_index([0.5, 0.5], 0.5) == 1


def test_sample_index_never_returns_a_zero_probability_token():
    """Строгое «>» вместо «>=»: токен с нулевой вероятностью не выдаётся."""
    assert sample_index([0.0, 1.0], 0.0) == 1
    assert sample_index([1.0, 0.0], 0.999999) == 0


def test_sample_index_reproduces_the_distribution():
    rng = random.Random(7)
    counts = [0] * 4
    for _ in range(20000):
        counts[sample_index(Q, rng.random())] += 1
    freq = [c / 20000 for c in counts]
    assert freq == pytest.approx(Q, abs=0.02)


# ------------------------------------------------------------------- accept
def test_accept_always_when_the_verifier_likes_it_more():
    assert accept(0.5, 0.2, 0.999) is True


def test_reject_when_the_draft_overshot():
    assert accept(0.1, 0.5, 0.9) is False


def test_accept_below_the_ratio():
    assert accept(0.1, 0.5, 0.1) is True


def test_accept_when_the_draft_could_not_have_produced_the_token():
    """p == 0 бывает только от округления; делить на ноль здесь нельзя."""
    assert accept(0.3, 0.0, 0.99) is True


# ----------------------------------------------------------------- residual
def test_residual_is_a_distribution():
    assert sum(residual(Q, P)) == APPROX(1.0)


def test_residual_is_zero_where_the_draft_already_overshot():
    out = residual([0.6, 0.4], [0.2, 0.8])
    assert out == APPROX([1.0, 0.0])


def test_residual_of_identical_distributions_falls_back_to_q():
    """Разность нулевая, нормировать нечего — возвращаем сам q."""
    assert residual([0.5, 0.5], [0.5, 0.5]) == APPROX([0.5, 0.5])


def test_leviathan_identity_reconstructs_q_exactly():
    """Ядро теоремы, без единой выборки.

    Принятая масса min(p, q) плюс масса отказа, размазанная по residual,
    должна давать в точности q. Если это равенство держится, выход цикла
    распределён как q при ЛЮБОМ черновике.
    """
    accepted_mass = [min(pi, qi) for pi, qi in zip(P, Q)]
    reject_prob = 1.0 - sum(accepted_mass)
    r = residual(Q, P)
    reconstructed = [a + reject_prob * ri for a, ri in zip(accepted_mass, r)]
    assert reconstructed == APPROX(Q)


def test_leviathan_identity_holds_for_a_terrible_draft():
    bad_p = [0.01, 0.01, 0.01, 0.97]
    accepted_mass = [min(pi, qi) for pi, qi in zip(bad_p, Q)]
    reject_prob = 1.0 - sum(accepted_mass)
    r = residual(Q, bad_p)
    reconstructed = [a + reject_prob * ri for a, ri in zip(accepted_mass, r)]
    assert reconstructed == APPROX(Q)


# ---------------------------------------------------------------- spec_step
def test_step_emits_between_one_and_n_plus_one_tokens():
    rng = random.Random(0)
    for _ in range(200):
        out = spec_step(P, Q, 4, rng)
        assert 1 <= len(out) <= 5


def test_perfect_draft_always_gets_the_bonus_token():
    """p == q значит q/p == 1, отказать невозможно — всегда n+1 токенов."""
    rng = random.Random(1)
    for _ in range(100):
        assert len(spec_step(Q, Q, 3, rng)) == 4


def test_hopeless_draft_still_emits_a_correct_token():
    """Черновик промахивается всегда, а выход всё равно из носителя q."""
    rng = random.Random(2)
    q = [1.0, 0.0]
    p = [0.0, 1.0]
    for _ in range(50):
        assert spec_step(p, q, 3, rng) == [0]


def test_output_distribution_equals_the_verifier_distribution():
    """Главное свойство урока: не «примерно как q», а ровно q."""
    rng = random.Random(12345)
    counts = [0] * len(Q)
    for _ in range(6000):
        for token in spec_step(P, Q, 4, rng):
            counts[token] += 1
    total = sum(counts)
    freq = [c / total for c in counts]
    assert freq == pytest.approx(Q, abs=0.025)


def test_a_better_draft_emits_more_tokens_per_verifier_pass():
    def mean_len(p, seed):
        rng = random.Random(seed)
        return sum(len(spec_step(p, Q, 5, rng)) for _ in range(2000)) / 2000

    poor = mean_len([0.05, 0.05, 0.05, 0.85], 3)
    good = mean_len(P, 3)
    assert good > poor


def test_step_is_reproducible_from_the_same_seed():
    """Тот же rng — тот же результат, иначе тесты не воспроизвести."""
    a = [spec_step(P, Q, 4, random.Random(99)) for _ in range(1)]
    b = [spec_step(P, Q, 4, random.Random(99)) for _ in range(1)]
    assert a == b


# --------------------------------------------------------- expected_emitted
def test_hopeless_draft_yields_exactly_one_token():
    assert expected_emitted(0.0, 5) == APPROX(1.0)


def test_perfect_draft_yields_n_plus_one():
    """alpha == 1 обнуляет знаменатель — этот случай надо разобрать руками."""
    assert expected_emitted(1.0, 5) == APPROX(6.0)


def test_expected_emitted_matches_the_worked_example():
    assert expected_emitted(0.8, 5) == pytest.approx(3.68928)


def test_expected_emitted_grows_with_acceptance_rate():
    """Скачок с 0.6 до 0.9 — это и есть весь выигрыш EAGLE-3."""
    assert expected_emitted(0.9, 5) > expected_emitted(0.6, 5) > expected_emitted(0.3, 5)


def test_expected_emitted_rejects_an_impossible_rate():
    with pytest.raises(ValueError):
        expected_emitted(1.5, 5)


# ------------------------------------------------------------ time_per_token
def test_no_speculation_costs_one_verifier_pass_per_token():
    assert time_per_token(0.0, 0, 0.05) == APPROX(1.0)


def test_good_draft_beats_plain_decoding():
    assert time_per_token(0.8, 5, 0.05) < 1.0


def test_time_per_token_matches_the_worked_example():
    assert time_per_token(0.8, 5, 0.05) == pytest.approx(0.33882, abs=1e-5)


def test_an_expensive_draft_eats_the_win():
    """При c близком к 1 черновик стоит как верификатор — смысла нет."""
    assert time_per_token(0.8, 5, 0.9) > 1.0


# -------------------------------------------------------- best_draft_length
def test_optimal_length_for_a_weak_draft():
    assert best_draft_length(0.6, 0.05) == 4


def test_optimal_length_for_a_strong_draft():
    assert best_draft_length(0.8, 0.05) == 8


def test_a_better_draft_deserves_a_longer_chain():
    assert best_draft_length(0.9, 0.05) > best_draft_length(0.5, 0.05)


def test_a_cheaper_draft_deserves_a_longer_chain():
    assert best_draft_length(0.8, 0.01) > best_draft_length(0.8, 0.3)


def test_the_optimum_really_is_the_minimum():
    best = best_draft_length(0.85, 0.04)
    at_best = time_per_token(0.85, best, 0.04)
    assert all(at_best <= time_per_token(0.85, n, 0.04) for n in range(1, 21))
