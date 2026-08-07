"""Тесты к уроку «RLHF: reward model и PPO». Правь exercise.py."""

import math

import pytest

from exercise import (
    bradley_terry_grad,
    bradley_terry_loss,
    kl_divergence,
    ppo_clipped_loss,
    reward_model_accuracy,
    rlhf_objective,
    sigmoid,
    softmax,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def numeric_partial(f, args, i, h=1e-6):
    """Численная производная f по i-му аргументу — центральная разность."""
    up, down = list(args), list(args)
    up[i] += h
    down[i] -= h
    return (f(*up) - f(*down)) / (2 * h)


# ---------------------------------------------------------------- sigmoid
def test_sigmoid_at_zero_is_half():
    assert sigmoid(0.0) == APPROX(0.5)


def test_sigmoid_survives_large_negative_input():
    """Наивная 1/(1+exp(-x)) падает с OverflowError на x = -1000."""
    assert sigmoid(-1000.0) == APPROX(0.0)


def test_sigmoid_survives_large_positive_input():
    assert sigmoid(1000.0) == APPROX(1.0)


# ---------------------------------------------------- bradley_terry_loss
def test_bradley_terry_loss_on_a_tie_is_log_two():
    """Равные награды — модель ничего не знает, лосс ровно log 2."""
    assert bradley_terry_loss(0.0, 0.0) == APPROX(math.log(2.0))


def test_bradley_terry_loss_falls_when_chosen_overtakes_rejected():
    """Главное смысловое свойство: обгоняет chosen — лосс падает."""
    behind = bradley_terry_loss(-2.0, 0.0)
    tie = bradley_terry_loss(0.0, 0.0)
    ahead = bradley_terry_loss(2.0, 0.0)
    assert behind > tie > ahead


def test_bradley_terry_loss_depends_only_on_the_difference():
    """Сдвиг обеих наград на константу ничего не меняет — шкала не определена."""
    assert bradley_terry_loss(1.0, 0.5) == APPROX(bradley_terry_loss(101.0, 100.5))


def test_bradley_terry_loss_survives_a_confidently_wrong_pair():
    """При diff = -800 наивный -log(sigmoid(diff)) даёт log(0) -> ValueError."""
    assert bradley_terry_loss(-800.0, 0.0) == pytest.approx(800.0, abs=1e-6)


# ---------------------------------------------------- bradley_terry_grad
def test_bradley_terry_grad_matches_numeric_derivative_for_chosen():
    args = (0.7, -0.4)
    numeric = numeric_partial(bradley_terry_loss, args, 0)
    assert bradley_terry_grad(*args)[0] == pytest.approx(numeric, abs=1e-6)


def test_bradley_terry_grad_matches_numeric_derivative_for_rejected():
    args = (0.7, -0.4)
    numeric = numeric_partial(bradley_terry_loss, args, 1)
    assert bradley_terry_grad(*args)[1] == pytest.approx(numeric, abs=1e-6)


def test_bradley_terry_grad_components_cancel_out():
    """Награды двигаются навстречу: сколько прибавили chosen, столько отняли у rejected."""
    g_chosen, g_rejected = bradley_terry_grad(1.3, 0.2)
    assert g_chosen + g_rejected == APPROX(0.0)


def test_bradley_terry_grad_pushes_chosen_up():
    """Градиентный спуск идёт против градиента, значит dL/dR_chosen < 0."""
    assert bradley_terry_grad(0.0, 0.0)[0] < 0


def test_bradley_terry_grad_vanishes_when_model_is_already_sure():
    """Пара решена уверенно — сигнала почти нет, обучение на ней стоит."""
    assert abs(bradley_terry_grad(20.0, 0.0)[0]) < 1e-8


# ------------------------------------------------- reward_model_accuracy
def test_reward_model_accuracy_all_correct():
    assert reward_model_accuracy([(1.0, 0.0), (2.0, 1.0)]) == APPROX(1.0)


def test_reward_model_accuracy_counts_a_tie_as_an_error():
    """Равные награды — выбора модель не сделала, засчитывать нечего."""
    assert reward_model_accuracy([(1.0, 1.0)]) == APPROX(0.0)


def test_reward_model_accuracy_of_empty_data_is_zero():
    assert reward_model_accuracy([]) == APPROX(0.0)


# ---------------------------------------------------------------- softmax
def test_softmax_of_equal_logits_is_uniform():
    assert softmax([1.0, 1.0, 1.0]) == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_softmax_sums_to_one():
    assert sum(softmax([3.0, -1.0, 0.5, 7.0])) == APPROX(1.0)


def test_softmax_survives_huge_logits():
    """Без вычитания максимума exp(1000) даёт OverflowError."""
    assert softmax([1000.0, 0.0]) == pytest.approx([1.0, 0.0])


def test_softmax_is_shift_invariant():
    assert softmax([1.0, 2.0, 3.0]) == pytest.approx(softmax([101.0, 102.0, 103.0]))


# ---------------------------------------------------------- kl_divergence
def test_kl_of_a_distribution_with_itself_is_zero():
    p = [0.2, 0.3, 0.5]
    assert kl_divergence(p, p) == APPROX(0.0)


def test_kl_is_never_negative():
    p, q = [0.7, 0.2, 0.1], [0.1, 0.3, 0.6]
    assert kl_divergence(p, q) > 0 and kl_divergence(q, p) > 0


def test_kl_is_not_symmetric():
    p, q = [0.9, 0.1], [0.5, 0.5]
    assert kl_divergence(p, q) != pytest.approx(kl_divergence(q, p), abs=1e-6)


def test_kl_skips_zero_probability_terms():
    """p_i == 0 даёт нулевое слагаемое, а не nan из math.log(0)."""
    assert kl_divergence([1.0, 0.0], [0.5, 0.5]) == APPROX(math.log(2.0))


def test_kl_grows_as_distributions_drift_apart():
    base = [0.5, 0.5]
    near = kl_divergence([0.6, 0.4], base)
    far = kl_divergence([0.9, 0.1], base)
    assert far > near > 0


# --------------------------------------------------------- rlhf_objective
def test_rlhf_objective_without_drift_equals_the_raw_reward():
    out = rlhf_objective(1.0, [0.3, -0.2], [0.3, -0.2])
    assert out["kl"] == APPROX(0.0)
    assert out["objective"] == APPROX(1.0)


def test_kl_penalty_keeps_the_policy_near_the_reference():
    """Главное свойство штрафа: ушёл от reference — получил меньше."""
    stayed = rlhf_objective(1.0, [0.0, 0.0], [0.0, 0.0], beta=0.5)
    drifted = rlhf_objective(1.0, [5.0, 0.0], [0.0, 0.0], beta=0.5)
    assert drifted["objective"] < stayed["objective"]


def test_bigger_beta_is_a_shorter_leash():
    soft = rlhf_objective(1.0, [3.0, 0.0], [0.0, 0.0], beta=0.02)
    hard = rlhf_objective(1.0, [3.0, 0.0], [0.0, 0.0], beta=1.0)
    assert hard["objective"] < soft["objective"]
    assert hard["kl"] == APPROX(soft["kl"])


def test_rlhf_objective_reports_the_penalty_it_subtracted():
    out = rlhf_objective(2.0, [1.0, 0.0], [0.0, 0.0], beta=0.3)
    assert out["objective"] == APPROX(out["reward"] - out["penalty"])
    assert out["penalty"] == APPROX(0.3 * out["kl"])


# ------------------------------------------------------ ppo_clipped_loss
def test_ppo_loss_at_ratio_one_is_minus_advantage():
    assert ppo_clipped_loss(1.0, 1.0) == APPROX(-1.0)


def test_ppo_clip_caps_a_too_large_step_on_a_good_action():
    """A > 0, ratio 1.5 — обрезано до 1+eps, дальше лосс не улучшается."""
    assert ppo_clipped_loss(1.5, 1.0) == APPROX(-1.2)
    assert ppo_clipped_loss(9.0, 1.0) == APPROX(-1.2)


def test_ppo_clip_caps_a_too_large_step_on_a_bad_action():
    """A < 0, ratio 0.5 — обрезано до 1-eps с другой стороны."""
    assert ppo_clipped_loss(0.5, -1.0) == APPROX(0.8)


def test_ppo_gradient_is_zero_beyond_the_clip_boundary():
    """Плоский участок — численная производная по ratio ровно ноль."""
    slope = numeric_partial(lambda r, a: ppo_clipped_loss(r, a), (1.5, 1.0), 0)
    assert slope == pytest.approx(0.0, abs=1e-9)


def test_ppo_gradient_survives_inside_the_trust_region():
    """Внутри [1-eps, 1+eps] обрезки нет, наклон равен -A."""
    slope = numeric_partial(lambda r, a: ppo_clipped_loss(r, a), (1.0, 1.0), 0)
    assert slope == pytest.approx(-1.0, abs=1e-6)


def test_ppo_does_not_clip_when_moving_away_from_a_bad_action():
    """Асимметрия: A < 0 и большой ratio — min берёт НЕобрезанную ветку."""
    slope = numeric_partial(lambda r, a: ppo_clipped_loss(r, a), (1.5, -1.0), 0)
    assert slope == pytest.approx(1.0, abs=1e-6)


def test_ppo_loss_with_zero_advantage_is_flat():
    assert ppo_clipped_loss(0.3, 0.0) == APPROX(0.0)
    assert ppo_clipped_loss(3.0, 0.0) == APPROX(0.0)
