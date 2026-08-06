"""Тесты к уроку «Proximal Policy Optimization (PPO)». Правь exercise.py."""

import math

import pytest

from exercise import (
    action_probs,
    approx_kl,
    clip_fraction,
    clipped_surrogate,
    importance_ratio,
    ppo_actor_step,
    ppo_update,
    surrogate_gradient_scale,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [v for row in M for v in row]


def snapshot(theta, x, action):
    """log pi_old(a|s) при текущей theta — как его снимают во время роллаута."""
    return math.log(action_probs(theta, x)[action])


# ----------------------------------------------------------- action_probs
def test_action_probs_sums_to_one():
    theta = [[0.3, -1.0], [2.0, 0.5], [-0.7, 0.1]]
    assert sum(action_probs(theta, [1.0, 2.0])) == pytest.approx(1.0, abs=1e-12)


def test_zero_weights_give_a_uniform_policy():
    assert action_probs([[0.0], [0.0], [0.0], [0.0]], [1.0]) == APPROX([0.25] * 4)


def test_action_probs_worked_example():
    p = action_probs([[1.0], [0.0]], [1.0])
    assert p == pytest.approx([math.e / (math.e + 1), 1 / (math.e + 1)], abs=1e-12)


def test_action_probs_survives_huge_features():
    """Наивный math.exp(1000) падает с OverflowError."""
    assert action_probs([[1.0], [0.0]], [1000.0]) == pytest.approx([1.0, 0.0], abs=1e-12)


# -------------------------------------------------------- importance_ratio
def test_ratio_of_an_unchanged_policy_is_one():
    assert importance_ratio(-1.234, -1.234) == APPROX(1.0)


def test_ratio_is_the_quotient_of_probabilities():
    assert importance_ratio(math.log(0.4), math.log(0.2)) == pytest.approx(2.0, abs=1e-12)


def test_ratio_survives_tiny_probabilities():
    """Длинный ответ LLM: сами вероятности занулились бы, разность логов — нет."""
    assert importance_ratio(-800.0, -801.0) == pytest.approx(math.e, abs=1e-9)


# ------------------------------------------------------- clipped_surrogate
def test_surrogate_at_ratio_one_is_just_the_advantage():
    assert clipped_surrogate(1.0, 3.0) == APPROX(3.0)


def test_positive_advantage_is_capped_above_the_corridor():
    assert clipped_surrogate(2.0, 3.0) == APPROX(1.2 * 3.0)


def test_negative_advantage_is_floored_below_the_corridor():
    assert clipped_surrogate(0.5, -3.0) == APPROX(0.8 * -3.0)


def test_the_min_lets_the_useful_direction_through():
    """A<0 и r>1+eps — политика ушла НЕ туда, и градиент обязан пройти целиком."""
    assert clipped_surrogate(2.0, -3.0) == APPROX(-6.0)


# ------------------------------------------------- surrogate_gradient_scale
def test_unclipped_scale_is_ratio_times_advantage():
    assert surrogate_gradient_scale(1.1, 2.0) == APPROX(2.2)


def test_clipped_objective_ignores_large_ratio_improvements():
    """Хорошее действие уже поднято на +eps — дальше двигать его нельзя."""
    assert surrogate_gradient_scale(2.0, 3.0) == APPROX(0.0)


def test_clipped_objective_ignores_over_punished_bad_actions():
    """Плохое действие уже опущено на -eps — дальше давить его нельзя."""
    assert surrogate_gradient_scale(0.5, -3.0) == APPROX(0.0)


def test_gradient_still_flows_on_the_side_that_needs_fixing():
    assert surrogate_gradient_scale(2.0, -3.0) == APPROX(-6.0)
    assert surrogate_gradient_scale(0.5, 3.0) == APPROX(1.5)


def test_scale_is_continuous_at_the_corridor_edge():
    """Ровно на границе градиент ещё есть, чуть за ней — уже нет."""
    assert surrogate_gradient_scale(1.2, 1.0) == APPROX(1.2)
    assert surrogate_gradient_scale(1.2 + 1e-9, 1.0) == APPROX(0.0)


# ----------------------------------------------------------- clip_fraction
def test_clip_fraction_of_a_fresh_rollout_is_zero():
    assert clip_fraction([1.0, 1.0, 1.0], [1.0, -1.0, 2.0]) == APPROX(0.0)


def test_clip_fraction_counts_only_the_clipped_side():
    ratios = [2.0, 2.0, 0.5, 0.5]
    advantages = [1.0, -1.0, 1.0, -1.0]
    # обрезаны только (r=2, A>0) и (r=0.5, A<0)
    assert clip_fraction(ratios, advantages) == APPROX(0.5)


def test_clip_fraction_of_an_empty_batch_is_zero():
    """Ловушка: деление на len([]) это ZeroDivisionError."""
    assert clip_fraction([], []) == APPROX(0.0)


def test_clip_fraction_grows_as_the_policy_drifts():
    advantages = [1.0] * 4
    near = clip_fraction([1.0, 1.05, 1.1, 1.15], advantages)
    far = clip_fraction([1.0, 1.5, 2.0, 3.0], advantages)
    assert near < far


# --------------------------------------------------------------- approx_kl
def test_kl_of_an_identical_policy_is_zero():
    assert approx_kl([-1.0, -2.0, -0.5], [-1.0, -2.0, -0.5]) == APPROX(0.0)


def test_kl_worked_example():
    assert approx_kl([-1.0, -1.0], [-2.0, -3.0]) == APPROX(1.5)


def test_kl_of_an_empty_batch_is_zero():
    assert approx_kl([], []) == APPROX(0.0)


# ---------------------------------------------------------- ppo_actor_step
def test_ppo_actor_step_worked_example():
    theta = [[0.0], [0.0]]
    new = ppo_actor_step(theta, [1.0], 0, -math.log(2), 1.0)
    assert flat(new) == APPROX([0.025, -0.025])


def test_positive_advantage_raises_the_probability_of_the_taken_action():
    theta = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    x = [1.0, 0.5]
    old = snapshot(theta, x, 1)
    before = action_probs(theta, x)[1]
    after = action_probs(ppo_actor_step(theta, x, 1, old, +1.0), x)[1]
    assert after > before


def test_negative_advantage_lowers_the_probability_of_the_taken_action():
    theta = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    x = [1.0, 0.5]
    old = snapshot(theta, x, 1)
    before = action_probs(theta, x)[1]
    after = action_probs(ppo_actor_step(theta, x, 1, old, -1.0), x)[1]
    assert after < before


def test_a_clipped_sample_leaves_theta_untouched():
    """Главное свойство PPO: за коридором градиент ровно ноль, а не «маленький»."""
    theta = [[0.5], [0.0]]
    x = [1.0]
    # log_pi_old настолько ниже текущего, что ratio далеко за 1+eps
    stale = snapshot(theta, x, 0) - 5.0
    assert flat(ppo_actor_step(theta, x, 0, stale, +2.0)) == APPROX(flat(theta))


def test_ppo_actor_step_matches_the_numeric_surrogate_gradient():
    """Сдвиг theta / lr обязан совпасть с численной производной surrogate."""
    theta = [[0.4, -0.2], [0.1, 0.7], [-0.5, 0.3]]
    x = [0.6, -1.1]
    action, adv, lr = 2, 1.7, 0.05
    old = snapshot(theta, x, action) - 0.05  # ratio чуть больше 1, внутри коридора
    new = ppo_actor_step(theta, x, action, old, adv, lr=lr)

    def objective(th):
        logp = math.log(action_probs(th, x)[action])
        return clipped_surrogate(importance_ratio(logp, old), adv)

    h = 1e-6
    for i in range(len(theta)):
        for j in range(len(x)):
            up = [row[:] for row in theta]
            down = [row[:] for row in theta]
            up[i][j] += h
            down[i][j] -= h
            numeric = (objective(up) - objective(down)) / (2 * h)
            got = (new[i][j] - theta[i][j]) / lr
            assert got == pytest.approx(numeric, abs=1e-5)


def test_ppo_actor_step_does_not_mutate_theta():
    theta = [[0.0], [0.0]]
    ppo_actor_step(theta, [1.0], 0, -math.log(2), 1.0)
    assert flat(theta) == APPROX([0.0, 0.0])


# ------------------------------------------------------------- ppo_update
def _batch(theta):
    x0, x1 = [1.0, 0.0], [0.0, 1.0]
    return [
        {"x": x0, "a": 0, "log_pi_old": snapshot(theta, x0, 0), "adv": +1.0},
        {"x": x1, "a": 1, "log_pi_old": snapshot(theta, x1, 1), "adv": -1.0},
    ]


def test_first_epoch_on_a_fresh_rollout_never_clips():
    """Первый сэмпл первой эпохи всегда идёт с ratio == 1: коридор не при чём."""
    theta = [[0.0, 0.0], [0.0, 0.0]]
    batch = [{"x": [1.0, 0.0], "a": 0, "log_pi_old": snapshot(theta, [1.0, 0.0], 0),
              "adv": +1.0}]
    _, mean_kl, clip_frac = ppo_update(theta, batch, epochs=1)
    assert clip_frac == APPROX(0.0)
    assert mean_kl == APPROX(0.0)


def test_more_epochs_drive_the_policy_further_from_pi_old():
    theta = [[0.0, 0.0], [0.0, 0.0]]
    batch = _batch(theta)
    _, kl_1, _ = ppo_update(theta, batch, epochs=1)
    _, kl_10, _ = ppo_update(theta, batch, epochs=10)
    assert abs(kl_10) > abs(kl_1)


def test_clipping_eventually_engages_over_many_epochs():
    """При K=1 clip не срабатывает, при большом K политика уезжает за коридор."""
    theta = [[0.0, 0.0], [0.0, 0.0]]
    batch = _batch(theta)
    _, _, frac_1 = ppo_update(theta, batch, lr=0.5, epochs=1)
    _, _, frac_20 = ppo_update(theta, batch, lr=0.5, epochs=20)
    assert frac_1 == APPROX(0.0)
    assert frac_20 > 0.0


def test_ppo_update_moves_the_policy_toward_the_positive_advantage_action():
    theta = [[0.0, 0.0], [0.0, 0.0]]
    batch = _batch(theta)
    new_theta, _, _ = ppo_update(theta, batch, epochs=4)
    x0 = [1.0, 0.0]
    assert action_probs(new_theta, x0)[0] > action_probs(theta, x0)[0]


def test_ppo_update_does_not_mutate_the_input_theta():
    theta = [[0.0, 0.0], [0.0, 0.0]]
    ppo_update(theta, _batch(theta), epochs=3)
    assert flat(theta) == APPROX([0.0, 0.0, 0.0, 0.0])


def test_ppo_update_with_zero_advantages_changes_nothing():
    """Нулевой advantage — нулевой градиент, сколько эпох ни крути."""
    theta = [[0.2, -0.4], [0.1, 0.9]]
    x = [1.0, 0.3]
    batch = [{"x": x, "a": 0, "log_pi_old": snapshot(theta, x, 0), "adv": 0.0}]
    new_theta, _, _ = ppo_update(theta, batch, epochs=8)
    assert flat(new_theta) == APPROX(flat(theta))
