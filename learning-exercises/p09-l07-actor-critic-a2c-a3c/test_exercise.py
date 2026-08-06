"""Тесты к уроку «Actor-critic: A2C и A3C». Правь exercise.py."""

import math

import pytest

from exercise import (
    actor_critic_step,
    discounted_returns,
    entropy,
    gae_advantages,
    grad_log_pi,
    normalize,
    softmax,
    td_residuals,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [v for row in M for v in row]


def variance(xs):
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def _logits(theta, x):
    return [sum(row[j] * x[j] for j in range(len(x))) for row in theta]


# ---------------------------------------------------------------- softmax
def test_softmax_sums_to_one():
    assert sum(softmax([1.0, -2.0, 3.5, 0.0])) == pytest.approx(1.0, abs=1e-12)


def test_softmax_of_equal_logits_is_uniform():
    assert softmax([1.0, 1.0, 1.0]) == APPROX([1 / 3, 1 / 3, 1 / 3])


def test_softmax_is_shift_invariant():
    """Прибавить константу ко всем логитам — распределение не меняется."""
    assert softmax([0.0, 1.0, 2.0]) == pytest.approx(softmax([10.0, 11.0, 12.0]))


def test_softmax_survives_huge_logits():
    """Наивный math.exp(1000) падает с OverflowError."""
    assert softmax([0.0, 1000.0]) == pytest.approx([0.0, 1.0], abs=1e-12)


# ---------------------------------------------------------------- entropy
def test_entropy_of_uniform_is_log_n():
    assert entropy([0.25] * 4) == pytest.approx(math.log(4), abs=1e-12)


def test_entropy_of_deterministic_policy_is_zero():
    """Ловушка: log(0) это -inf, нулевые вероятности надо пропускать."""
    assert entropy([1.0, 0.0, 0.0]) == APPROX(0.0)


def test_entropy_drops_as_the_policy_sharpens():
    """Схлопывание политики видно как падение энтропии — за этим и следят."""
    assert entropy([0.5, 0.5]) > entropy([0.9, 0.1]) > entropy([0.99, 0.01])


# ------------------------------------------------------------ grad_log_pi
def test_grad_log_pi_worked_example():
    assert grad_log_pi([0.2, 0.8], 1) == APPROX([-0.2, 0.2])


def test_grad_log_pi_sums_to_zero():
    """softmax нормирован: поднять одно действие можно только опустив другие."""
    assert sum(grad_log_pi([0.1, 0.3, 0.6], 2)) == APPROX(0.0)


def test_grad_log_pi_is_positive_for_the_chosen_action():
    g = grad_log_pi([0.25, 0.25, 0.5], 0)
    assert g[0] > 0
    assert g[1] < 0 and g[2] < 0


def test_grad_log_pi_matches_numeric_derivative():
    """Аналитический градиент против центральной разности по логитам."""
    z = [0.3, -1.2, 0.8]
    action = 1
    h = 1e-6
    numeric = []
    for i in range(len(z)):
        up, down = list(z), list(z)
        up[i] += h
        down[i] -= h
        numeric.append(
            (math.log(softmax(up)[action]) - math.log(softmax(down)[action])) / (2 * h)
        )
    assert grad_log_pi(softmax(z), action) == pytest.approx(numeric, abs=1e-6)


# ------------------------------------------------------- discounted_returns
def test_discounted_returns_worked_example():
    assert discounted_returns([1.0, 1.0], gamma=0.5) == APPROX([1.5, 1.0])


def test_discounted_returns_with_zero_gamma_is_just_the_rewards():
    """gamma = 0 — полная близорукость: возврат равен мгновенной награде."""
    assert discounted_returns([1.0, 2.0, 3.0], gamma=0.0) == APPROX([1.0, 2.0, 3.0])


def test_discounted_returns_bootstrap_closes_the_truncated_tail():
    assert discounted_returns([0.0], gamma=0.9, last_value=10.0) == APPROX([9.0])


def test_discounted_returns_with_gamma_one_is_a_plain_suffix_sum():
    assert discounted_returns([1.0, 2.0, 3.0], gamma=1.0) == APPROX([6.0, 5.0, 3.0])


# ---------------------------------------------------------- td_residuals
def test_td_residual_is_zero_when_the_critic_is_exact():
    """delta == 0 означает: критик уже знает ответ, учить его нечему."""
    assert td_residuals([1.0], [1.0], gamma=0.0) == APPROX([0.0])


def test_td_residuals_use_bootstrap_for_the_last_step():
    assert td_residuals([0.0, 0.0], [1.0, 1.0], gamma=1.0) == APPROX([0.0, -1.0])


def test_td_residual_is_positive_when_the_reward_beats_the_prediction():
    assert td_residuals([5.0], [0.0], gamma=0.9)[0] > 0


# --------------------------------------------------------- gae_advantages
def test_gae_with_lambda_zero_equals_one_step_td():
    rewards = [1.0, -2.0, 0.5, 3.0]
    values = [0.1, 0.2, -0.3, 0.4]
    advs, _ = gae_advantages(rewards, values, gamma=0.9, lam=0.0)
    assert advs == pytest.approx(td_residuals(rewards, values, 0.9), abs=1e-12)


def test_gae_with_lambda_one_equals_monte_carlo_minus_baseline():
    rewards = [1.0, -2.0, 0.5, 3.0]
    values = [0.1, 0.2, -0.3, 0.4]
    advs, _ = gae_advantages(rewards, values, gamma=0.9, lam=1.0)
    mc = discounted_returns(rewards, gamma=0.9)
    assert advs == pytest.approx([g - v for g, v in zip(mc, values)], abs=1e-9)


def test_gae_returns_are_the_critic_target():
    """returns[t] == advantages[t] + values[t] — иначе критик учится не туда."""
    rewards = [1.0, 2.0, -1.0]
    values = [0.3, -0.7, 1.1]
    advs, rets = gae_advantages(rewards, values, gamma=0.95, lam=0.9)
    assert rets == pytest.approx([a + v for a, v in zip(advs, values)], abs=1e-12)


def test_gae_with_lambda_one_reproduces_monte_carlo_returns():
    rewards = [1.0, 1.0, 1.0]
    values = [0.5, 0.5, 0.5]
    _, rets = gae_advantages(rewards, values, gamma=1.0, lam=1.0)
    assert rets == pytest.approx(discounted_returns(rewards, gamma=1.0), abs=1e-9)


def test_gae_lambda_dial_moves_between_td_and_mc():
    """Промежуточная lambda лежит МЕЖДУ двумя крайностями, а не снаружи."""
    rewards = [1.0, 1.0, 1.0]
    values = [0.0, 0.0, 0.0]
    td, _ = gae_advantages(rewards, values, gamma=1.0, lam=0.0)
    mid, _ = gae_advantages(rewards, values, gamma=1.0, lam=0.5)
    mc, _ = gae_advantages(rewards, values, gamma=1.0, lam=1.0)
    assert td[0] < mid[0] < mc[0]


def test_a_state_dependent_baseline_lowers_the_variance_of_the_advantage():
    """Вся суть критика: он срезает дисперсию сигнала, который умножает grad log pi.

    Критик, следящий за возвратом, оставляет только остаток. Постоянный
    baseline так не умеет — он лишь сдвигает среднее, дисперсия та же.
    """
    rewards = [5.0, -5.0, 4.0, -6.0, 5.0, -3.0]
    no_baseline, _ = gae_advantages(rewards, [0.0] * 6, gamma=0.0, lam=0.0)
    tracking = [0.9 * r for r in rewards]
    with_critic, _ = gae_advantages(rewards, tracking, gamma=0.0, lam=0.0)
    assert variance(with_critic) < variance(no_baseline)

    const = sum(rewards) / len(rewards)
    flat_baseline, _ = gae_advantages(rewards, [const] * 6, gamma=0.0, lam=0.0)
    assert variance(flat_baseline) == pytest.approx(variance(no_baseline), abs=1e-9)


# -------------------------------------------------------------- normalize
def test_normalize_gives_zero_mean_and_unit_std():
    out = normalize([1.0, 2.0, 3.0, 10.0])
    assert sum(out) / len(out) == pytest.approx(0.0, abs=1e-7)
    assert variance(out) == pytest.approx(1.0, abs=1e-6)


def test_normalize_worked_example():
    assert normalize([1.0, 2.0, 3.0]) == pytest.approx(
        [-math.sqrt(1.5), 0.0, math.sqrt(1.5)], abs=1e-6
    )


def test_normalize_of_a_constant_batch_does_not_divide_by_zero():
    """Ловушка: sd == 0 на константном advantage-батче."""
    assert normalize([2.0, 2.0, 2.0]) == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)


def test_normalize_of_a_single_element_leaves_it_alone():
    assert normalize([5.0]) == APPROX([5.0])


# -------------------------------------------------------- actor_critic_step
def test_actor_critic_step_worked_example():
    theta, w = actor_critic_step([[0.0], [0.0]], [0.0], [1.0], 0, 1.0, 2.0)
    assert flat(theta) == APPROX([0.025, -0.025])
    assert w == APPROX([0.2])


def test_positive_advantage_raises_the_probability_of_the_taken_action():
    theta = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    x = [1.0, 0.0]
    before = softmax(_logits(theta, x))[1]
    new_theta, _ = actor_critic_step(theta, [0.0, 0.0], x, 1, +1.0, 0.0)
    after = softmax(_logits(new_theta, x))[1]
    assert after > before


def test_negative_advantage_lowers_the_probability_of_the_taken_action():
    theta = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    x = [1.0, 0.0]
    before = softmax(_logits(theta, x))[1]
    new_theta, _ = actor_critic_step(theta, [0.0, 0.0], x, 1, -1.0, 0.0)
    after = softmax(_logits(new_theta, x))[1]
    assert after < before


def test_critic_moves_toward_its_target():
    theta = [[0.0], [0.0]]
    x = [1.0]
    _, w1 = actor_critic_step(theta, [0.0], x, 0, 0.0, 5.0)
    _, w2 = actor_critic_step(theta, w1, x, 0, 0.0, 5.0)
    assert abs(5.0 - w1[0]) < abs(5.0 - 0.0)
    assert abs(5.0 - w2[0]) < abs(5.0 - w1[0])


def test_actor_update_matches_the_numeric_policy_gradient():
    """Сдвиг theta / lr_a обязан совпасть с численной производной adv*log pi."""
    theta = [[0.4, -0.2], [0.1, 0.7], [-0.5, 0.3]]
    x = [0.6, -1.1]
    action, adv, lr_a = 2, 1.7, 0.05
    new_theta, _ = actor_critic_step(theta, [0.0, 0.0], x, action, adv, 0.0, lr_a=lr_a)

    h = 1e-6
    for i in range(len(theta)):
        for j in range(len(x)):
            up = [row[:] for row in theta]
            down = [row[:] for row in theta]
            up[i][j] += h
            down[i][j] -= h
            f_up = adv * math.log(softmax(_logits(up, x))[action])
            f_down = adv * math.log(softmax(_logits(down, x))[action])
            numeric = (f_up - f_down) / (2 * h)
            got = (new_theta[i][j] - theta[i][j]) / lr_a
            assert got == pytest.approx(numeric, abs=1e-5)


def test_critic_update_matches_the_numeric_mse_gradient():
    """Критик спускается по -(1/2) d/dw (target - V)^2, проверяем численно."""
    w = [0.3, -0.8, 0.5]
    x = [1.0, 0.5, -2.0]
    target, lr_v = 4.0, 0.1
    _, new_w = actor_critic_step([[0.0] * 3, [0.0] * 3], w, x, 0, 0.0, target, lr_v=lr_v)

    h = 1e-6
    for j in range(len(w)):
        up, down = list(w), list(w)
        up[j] += h
        down[j] -= h
        loss_up = (target - sum(a * b for a, b in zip(up, x))) ** 2
        loss_down = (target - sum(a * b for a, b in zip(down, x))) ** 2
        numeric = (loss_up - loss_down) / (2 * h)
        got = (new_w[j] - w[j]) / lr_v
        assert got == pytest.approx(-0.5 * numeric, abs=1e-5)


def test_actor_critic_step_does_not_mutate_its_inputs():
    theta = [[0.0], [0.0]]
    w = [0.0]
    actor_critic_step(theta, w, [1.0], 0, 1.0, 2.0)
    assert flat(theta) == APPROX([0.0, 0.0])
    assert w == APPROX([0.0])
