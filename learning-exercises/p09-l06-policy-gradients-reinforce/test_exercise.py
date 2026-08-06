"""Тесты к уроку «Policy gradient: REINFORCE с нуля». Правь exercise.py."""

import math
import random
import statistics

import pytest

from exercise import (
    grad_log_pi,
    grid_rollout,
    policy_probs,
    reinforce_grad,
    returns_to_go,
    sample_action,
    softmax,
    train_reinforce,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# 0 up, 1 down, 2 left, 3 right
N_ACTIONS = 4
N_FEATURES = 16

OPTIMAL_G = -(1 - 0.99 ** 6) / 0.01


def zero_theta():
    return [[0.0] * N_FEATURES for _ in range(N_ACTIONS)]


def staircase_theta():
    """Почти детерминированная политика «вниз, пока не нижний ряд, потом вправо»."""
    theta = zero_theta()
    for r in range(4):
        for c in range(4):
            action = 1 if r < 3 else 3
            theta[action][r * 4 + c] = 50.0
    return theta


def random_theta(seed):
    rng = random.Random(seed)
    return [[rng.uniform(-1.0, 1.0) for _ in range(N_FEATURES)] for _ in range(N_ACTIONS)]


def flat(matrix):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [v for row in matrix for v in row]


def surrogate(theta, trajectory, gamma, baseline):
    """L(theta) = sum_t (G_t - b) * log pi_theta(a_t|s_t).

    Градиент именно этой функции и обязан возвращать reinforce_grad — по ней
    и считаем численную производную.
    """
    returns = returns_to_go([r for _, _, r in trajectory], gamma)
    total = 0.0
    for (features, action, _), G in zip(trajectory, returns):
        total += (G - baseline) * math.log(policy_probs(theta, features)[action])
    return total


# ----------------------------------------------------------------- softmax
def test_softmax_of_equal_logits_is_uniform():
    assert softmax([0.0, 0.0]) == pytest.approx([0.5, 0.5], abs=1e-12)
    assert softmax([7.0] * 4) == pytest.approx([0.25] * 4, abs=1e-12)


def test_softmax_always_sums_to_one():
    for logits in ([1.0, 0.0], [-3.0, 2.5, 0.1], [0.0] * 7):
        assert sum(softmax(logits)) == pytest.approx(1.0, abs=1e-12)


def test_softmax_matches_the_hand_computed_pair():
    p = softmax([1.0, 0.0])
    assert p[0] == pytest.approx(math.e / (math.e + 1), abs=1e-12)


def test_softmax_is_shift_invariant():
    """softmax(l) == softmax(l - max l): на этом и стоит численная защита."""
    assert softmax([1001.0, 1000.0]) == pytest.approx(softmax([1.0, 0.0]), abs=1e-12)


def test_softmax_survives_logits_that_overflow_exp():
    """Наивный exp(1000) падает с OverflowError, exp(-1000) даёт деление на ноль."""
    assert softmax([1000.0, 0.0]) == pytest.approx([1.0, 0.0], abs=1e-9)
    assert softmax([-1000.0, -1000.0]) == pytest.approx([0.5, 0.5], abs=1e-12)


# ------------------------------------------------------------- policy_probs
def test_zero_parameters_give_a_uniform_policy():
    """Нулевая theta — максимальная энтропия, агент честно пробует всё."""
    probs = policy_probs(zero_theta(), [1.0] + [0.0] * 15)
    assert probs == pytest.approx([0.25] * 4, abs=1e-12)


def test_policy_probs_equals_softmax_of_the_dot_products():
    theta, features = random_theta(1), [0.0] * 16
    features[5] = 1.0
    logits = [sum(w * x for w, x in zip(row, features)) for row in theta]
    assert policy_probs(theta, features) == pytest.approx(softmax(logits), abs=1e-12)


def test_a_large_weight_makes_its_action_almost_certain():
    theta = zero_theta()
    theta[1][0] = 20.0
    probs = policy_probs(theta, [1.0] + [0.0] * 15)
    assert probs[1] > 0.999
    assert sum(probs) == pytest.approx(1.0, abs=1e-12)


# ------------------------------------------------------------- sample_action
def test_sample_action_with_a_certain_distribution_is_deterministic():
    rng = random.Random(0)
    assert {sample_action([0.0, 1.0, 0.0], rng) for _ in range(200)} == {1}


def test_sample_action_frequencies_match_the_probabilities():
    rng = random.Random(3)
    n = 20000
    draws = [sample_action([0.1, 0.6, 0.3], rng) for _ in range(n)]
    assert draws.count(1) / n == pytest.approx(0.6, abs=0.02)
    assert draws.count(0) / n == pytest.approx(0.1, abs=0.02)


def test_sample_action_never_leaves_the_index_range():
    rng = random.Random(4)
    assert all(0 <= sample_action([0.25] * 4, rng) < 4 for _ in range(500))


def test_sample_action_is_reproducible_for_the_same_seed():
    probs = [0.25] * 4
    rng_a, rng_b = random.Random(9), random.Random(9)
    draws = [sample_action(probs, rng_a) for _ in range(40)]
    assert draws == [sample_action(probs, rng_b) for _ in range(40)]
    assert len(set(draws)) > 1


# -------------------------------------------------------------- grad_log_pi
def test_grad_log_pi_of_a_uniform_policy():
    assert grad_log_pi([0.25] * 4, 1) == pytest.approx(
        [-0.25, 0.75, -0.25, -0.25], abs=1e-12
    )


def test_grad_log_pi_components_sum_to_zero():
    """Вероятность перекладывается между действиями, а не создаётся."""
    for seed in (0, 1, 2):
        probs = policy_probs(random_theta(seed), [1.0] + [0.0] * 15)
        for action in range(4):
            assert sum(grad_log_pi(probs, action)) == pytest.approx(0.0, abs=1e-12)


def test_grad_log_pi_pushes_the_taken_action_up_and_the_rest_down():
    probs = policy_probs(random_theta(5), [1.0] + [0.0] * 15)
    grad = grad_log_pi(probs, 2)
    assert grad[2] == pytest.approx(1.0 - probs[2], abs=1e-12)
    assert all(grad[i] < 0 for i in (0, 1, 3))


def test_grad_log_pi_matches_the_central_difference_on_the_logits():
    """Аналитика против численной производной d log softmax(z)[a] / d z_k."""
    logits = [0.7, -1.2, 0.3, 2.0]
    h = 1e-6
    analytic = grad_log_pi(softmax(logits), 2)
    for k in range(4):
        up, down = list(logits), list(logits)
        up[k] += h
        down[k] -= h
        numeric = (math.log(softmax(up)[2]) - math.log(softmax(down)[2])) / (2 * h)
        assert analytic[k] == pytest.approx(numeric, abs=1e-6)


def test_grad_log_pi_is_almost_zero_for_an_already_certain_action():
    """Если действие и так берётся с вероятностью 1, толкать его некуда."""
    probs = softmax([100.0, 0.0])
    assert grad_log_pi(probs, 0)[0] == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------------ returns_to_go
def test_returns_to_go_with_gamma_one_is_suffix_sums():
    assert returns_to_go([-1.0] * 3, 1.0) == pytest.approx([-3.0, -2.0, -1.0], abs=1e-12)


def test_returns_to_go_discounts_the_tail():
    assert returns_to_go([1.0, 1.0], 0.5) == pytest.approx([1.5, 1.0], abs=1e-12)


def test_returns_to_go_satisfies_the_backward_recurrence():
    rng = random.Random(2)
    rewards = [rng.uniform(-2, 2) for _ in range(30)]
    gamma = 0.9
    G = returns_to_go(rewards, gamma)
    for t in range(len(rewards) - 1):
        assert G[t] == pytest.approx(rewards[t] + gamma * G[t + 1], abs=1e-9)


def test_returns_to_go_of_the_optimal_path():
    assert returns_to_go([-1.0] * 6, 0.99)[0] == APPROX(OPTIMAL_G)
    assert returns_to_go([-1.0] * 6, 0.99)[-1] == APPROX(-1.0)


# ------------------------------------------------------------- grid_rollout
def test_grid_rollout_of_a_near_deterministic_policy_is_six_steps():
    trajectory = grid_rollout(staircase_theta(), random.Random(0))
    assert len(trajectory) == 6
    assert [a for _, a, _ in trajectory] == [1, 1, 1, 3, 3, 3]
    assert all(r == -1.0 for _, _, r in trajectory)


def test_grid_rollout_features_are_one_hot_of_the_visited_state():
    trajectory = grid_rollout(staircase_theta(), random.Random(0))
    visited = [row.index(1.0) for row, _, _ in trajectory]
    assert visited == [0, 4, 8, 12, 13, 14]
    assert all(sum(row) == 1.0 and len(row) == 16 for row, _, _ in trajectory)


def test_grid_rollout_stops_at_max_steps():
    """Политика «всегда вверх» из угла не выйдет — спасает только cap."""
    theta = zero_theta()
    for j in range(16):
        theta[0][j] = 50.0
    assert len(grid_rollout(theta, random.Random(0), max_steps=17)) == 17


def test_grid_rollout_is_reproducible_for_the_same_seed():
    theta = random_theta(6)
    assert grid_rollout(theta, random.Random(11)) == grid_rollout(theta, random.Random(11))


# ----------------------------------------------------------- reinforce_grad
def test_reinforce_grad_has_the_same_shape_as_theta():
    theta = random_theta(7)
    grad = reinforce_grad(theta, grid_rollout(theta, random.Random(0)))
    assert len(grad) == len(theta)
    assert [len(row) for row in grad] == [len(row) for row in theta]


def test_reinforce_grad_matches_the_numeric_gradient_of_the_surrogate():
    """Аналитический градиент против центральной разности по самой theta."""
    theta = random_theta(8)
    trajectory = grid_rollout(theta, random.Random(1))
    gamma, h = 0.99, 1e-5
    grad = reinforce_grad(theta, trajectory, gamma, 0.0)
    for i in range(N_ACTIONS):
        for j in (0, 4, 5):
            up = [row[:] for row in theta]
            down = [row[:] for row in theta]
            up[i][j] += h
            down[i][j] -= h
            numeric = (
                surrogate(up, trajectory, gamma, 0.0)
                - surrogate(down, trajectory, gamma, 0.0)
            ) / (2 * h)
            assert grad[i][j] == pytest.approx(numeric, abs=1e-4)


def test_reinforce_grad_with_a_baseline_still_matches_the_numeric_gradient():
    theta = random_theta(9)
    trajectory = grid_rollout(theta, random.Random(2))
    gamma, h, baseline = 0.99, 1e-5, -30.0
    grad = reinforce_grad(theta, trajectory, gamma, baseline)
    for i in range(N_ACTIONS):
        for j in (0, 4):
            up = [row[:] for row in theta]
            down = [row[:] for row in theta]
            up[i][j] += h
            down[i][j] -= h
            numeric = (
                surrogate(up, trajectory, gamma, baseline)
                - surrogate(down, trajectory, gamma, baseline)
            ) / (2 * h)
            assert grad[i][j] == pytest.approx(numeric, abs=1e-4)


def test_a_zero_advantage_gives_a_zero_gradient():
    """Все advantage нулевые — двигаться некуда, хотя эпизод и был."""
    theta = zero_theta()
    trajectory = grid_rollout(theta, random.Random(3))
    gamma = 1.0
    returns = returns_to_go([r for _, _, r in trajectory], gamma)
    constant = [(f, a, 0.0) for f, a, _ in trajectory]
    grad = reinforce_grad(theta, constant, gamma, 0.0)
    assert flat(grad) == pytest.approx([0.0] * (N_ACTIONS * N_FEATURES), abs=1e-12)
    assert len(returns) == len(trajectory)


def test_baseline_shifts_the_gradient_by_a_predictable_amount():
    """Вычитание константы b меняет градиент ровно на -b * sum_t dlog * x."""
    theta = random_theta(10)
    trajectory = grid_rollout(theta, random.Random(4))
    plain = reinforce_grad(theta, trajectory, 0.99, 0.0)
    shifted = reinforce_grad(theta, trajectory, 0.99, 5.0)
    zero_adv = reinforce_grad(
        theta, [(f, a, 0.0) for f, a, _ in trajectory], 0.99, 5.0
    )
    for i in range(N_ACTIONS):
        for j in range(N_FEATURES):
            assert shifted[i][j] == pytest.approx(plain[i][j] + zero_adv[i][j], abs=1e-9)


def test_a_baseline_cuts_the_variance_of_the_gradient_estimate():
    """Главное свойство baseline: смещения нет, а дисперсия падает в разы.

    Собираем 300 эпизодов одной и той же (равномерной) политикой и смотрим
    разброс покомпонентных оценок градиента с baseline и без.
    """
    theta = zero_theta()
    gamma = 0.99
    trajectories = [grid_rollout(theta, random.Random(s)) for s in range(300)]
    all_returns = [
        G for t in trajectories for G in returns_to_go([r for _, _, r in t], gamma)
    ]
    baseline = sum(all_returns) / len(all_returns)

    raw = [reinforce_grad(theta, t, gamma, 0.0) for t in trajectories]
    fixed = [reinforce_grad(theta, t, gamma, baseline) for t in trajectories]

    def total_variance(samples):
        return sum(
            statistics.pvariance([g[i][j] for g in samples])
            for i in range(N_ACTIONS)
            for j in range(N_FEATURES)
        )

    assert total_variance(fixed) * 3 < total_variance(raw)
    assert baseline < -10.0


# ---------------------------------------------------------- train_reinforce
def test_train_reinforce_learning_curve_goes_up():
    _, log = train_reinforce(1200, lr=0.05, rng=random.Random(0))
    assert sum(log[:100]) / 100 < -10.0
    assert sum(log[-100:]) / 100 > sum(log[:100]) / 100
    assert sum(log[-100:]) / 100 > -8.0


def test_train_reinforce_ends_up_walking_the_short_path():
    """Проверяем поведением: обученная политика доходит почти оптимально."""
    theta, _ = train_reinforce(1200, lr=0.05, rng=random.Random(0))
    lengths = [len(grid_rollout(theta, random.Random(s))) for s in range(20)]
    assert min(lengths) == 6, "оптимальный путь ни разу не получился"
    assert statistics.mean(lengths) < 8.0


def test_train_reinforce_works_without_a_baseline_too():
    """Baseline снижает дисперсию, но не является условием обучаемости."""
    _, log = train_reinforce(1200, lr=0.01, use_baseline=False, rng=random.Random(0))
    assert sum(log[-100:]) / 100 > -9.0


def test_train_reinforce_is_reproducible_for_the_same_seed():
    assert (
        train_reinforce(60, rng=random.Random(5))[1]
        == train_reinforce(60, rng=random.Random(5))[1]
    )
