"""Тесты к уроку «MDP: состояния, действия, награды». Правь exercise.py."""

import random

import pytest

from exercise import (
    discounted_return,
    effective_horizon,
    greedy_from_q,
    grid_step,
    policy_evaluation,
    q_from_v,
    rollout,
    sample_action,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

ACTIONS = ("up", "down", "left", "right")
TERMINAL = (3, 3)

# G оптимального пути: шесть шагов по -1 с gamma=0.99
OPTIMAL_G = -(1 - 0.99 ** 6) / 0.01


def uniform(_state):
    return {a: 0.25 for a in ACTIONS}


def staircase(state):
    """Оптимальная политика: сначала вниз до нижнего ряда, потом вправо."""
    if state[0] < 3:
        return {"up": 0.0, "down": 1.0, "left": 0.0, "right": 0.0}
    return {"up": 0.0, "down": 0.0, "left": 0.0, "right": 1.0}


def always_up(_state):
    return {"up": 1.0, "down": 0.0, "left": 0.0, "right": 0.0}


def flat(mapping):
    """Развернуть {state: {action: value}} в плоский список: approx не умеет вложенное."""
    return [mapping[s][a] for s in sorted(mapping) for a in ACTIONS]


# --------------------------------------------------------------- grid_step
def test_grid_step_moves_in_the_requested_direction():
    assert grid_step((0, 0), "down") == ((1, 0), -1.0, False)
    assert grid_step((0, 0), "right") == ((0, 1), -1.0, False)


def test_grid_step_clamps_against_the_wall():
    """За границу сетки уйти нельзя, но шаг всё равно стоит -1."""
    assert grid_step((0, 0), "up") == ((0, 0), -1.0, False)
    assert grid_step((0, 0), "left") == ((0, 0), -1.0, False)
    assert grid_step((3, 3 - 1), "down") == ((3, 2), -1.0, False)


def test_grid_step_marks_done_when_the_terminal_is_reached():
    assert grid_step((3, 2), "right") == ((3, 3), -1.0, True)
    assert grid_step((2, 3), "down") == ((3, 3), -1.0, True)


def test_grid_step_terminal_is_absorbing_and_free():
    """Из терминала не выйти, и награда там уже нулевая, а не -1."""
    for action in ACTIONS:
        assert grid_step(TERMINAL, action) == (TERMINAL, 0.0, True)


def test_grid_step_reward_is_minus_one_on_every_non_terminal_state():
    for r in range(4):
        for c in range(4):
            if (r, c) == TERMINAL:
                continue
            assert grid_step((r, c), "up")[1] == APPROX(-1.0)


# -------------------------------------------------------- discounted_return
def test_discounted_return_weights_the_second_reward_by_gamma():
    assert discounted_return([1.0, 1.0], 0.5) == APPROX(1.5)


def test_discounted_return_with_gamma_zero_sees_only_the_first_step():
    """gamma=0 делает агента полностью близоруким."""
    assert discounted_return([5.0, 100.0, 100.0], 0.0) == APPROX(5.0)


def test_discounted_return_with_gamma_one_is_a_plain_sum():
    assert discounted_return([-1.0] * 6, 1.0) == APPROX(-6.0)


def test_discounted_return_of_the_optimal_path():
    assert discounted_return([-1.0] * 6, 0.99) == APPROX(OPTIMAL_G)


def test_higher_gamma_values_a_distant_reward_more():
    """Одна и та же награда на шестом шаге весит тем больше, чем выше gamma."""
    late = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    assert discounted_return(late, 0.99) > discounted_return(late, 0.9)
    assert discounted_return(late, 0.5) < 0.05


def test_discounted_return_of_an_empty_episode_is_zero():
    assert discounted_return([], 0.9) == APPROX(0.0)


# ------------------------------------------------------- effective_horizon
def test_effective_horizon_of_099_is_about_a_hundred_steps():
    assert effective_horizon(0.99) == pytest.approx(100.0, abs=1e-6)
    assert effective_horizon(0.9) == pytest.approx(10.0, abs=1e-6)


def test_effective_horizon_of_zero_gamma_is_one_step():
    assert effective_horizon(0.0) == APPROX(1.0)


def test_effective_horizon_grows_as_gamma_approaches_one():
    assert effective_horizon(0.9) < effective_horizon(0.99) < effective_horizon(0.999)


def test_effective_horizon_rejects_gamma_one_as_a_broken_task():
    """gamma=1 на бесконечном горизонте — не «большой горизонт», а расходимость."""
    with pytest.raises(ValueError):
        effective_horizon(1.0)
    with pytest.raises(ValueError):
        effective_horizon(-0.5)


# ---------------------------------------------------------- sample_action
def test_sample_action_with_a_certain_distribution_is_deterministic():
    rng = random.Random(0)
    assert [sample_action({"up": 1.0, "down": 0.0}, rng) for _ in range(50)] == ["up"] * 50


def test_sample_action_never_returns_a_zero_probability_action():
    rng = random.Random(1)
    dist = {"up": 0.0, "down": 0.5, "left": 0.0, "right": 0.5}
    picked = {sample_action(dist, rng) for _ in range(500)}
    assert picked == {"down", "right"}


def test_sample_action_frequencies_match_the_probabilities():
    rng = random.Random(7)
    dist = {"a": 0.7, "b": 0.3}
    draws = [sample_action(dist, rng) for _ in range(20000)]
    assert draws.count("a") / 20000 == pytest.approx(0.7, abs=0.02)


def test_sample_action_is_reproducible_for_the_same_seed():
    """Без воспроизводимости RL нельзя отлаживать: тот же seed — та же выборка."""
    dist = {a: 0.25 for a in ACTIONS}
    rng_a, rng_b = random.Random(11), random.Random(11)
    draws_a = [sample_action(dist, rng_a) for _ in range(30)]
    draws_b = [sample_action(dist, rng_b) for _ in range(30)]
    assert draws_a == draws_b
    assert set(draws_a) <= set(ACTIONS) and len(set(draws_a)) > 1


# ---------------------------------------------------------------- rollout
def test_rollout_of_the_optimal_policy_takes_exactly_six_steps():
    total, steps = rollout(staircase, random.Random(0))
    assert (total, steps) == (APPROX(-6.0), 6)


def test_rollout_of_a_random_policy_is_never_better_than_optimal():
    rng = random.Random(5)
    for _ in range(20):
        total, steps = rollout(uniform, rng)
        assert total <= -6.0
        assert steps >= 6


def test_rollout_stops_at_max_steps_when_the_policy_never_terminates():
    """Политика «всегда вверх» из угла не выйдет никогда — спасает только cap."""
    total, steps = rollout(always_up, random.Random(0), max_steps=10)
    assert (total, steps) == (APPROX(-10.0), 10)


def test_rollout_is_reproducible_for_the_same_seed():
    assert rollout(uniform, random.Random(42)) == rollout(uniform, random.Random(42))


# ------------------------------------------------------- policy_evaluation
def test_policy_evaluation_keeps_the_terminal_at_zero():
    V = policy_evaluation(uniform)
    assert V[TERMINAL] == APPROX(0.0)


def test_policy_evaluation_matches_the_closed_form_on_the_optimal_path():
    """Политика «вниз, потом вправо» проходит ровно шесть шагов — V(0,0) считается вручную."""
    V = policy_evaluation(staircase, gamma=0.99)
    assert V[(0, 0)] == pytest.approx(OPTIMAL_G, abs=1e-6)


def test_policy_evaluation_satisfies_the_bellman_equation():
    """Главное свойство: V — неподвижная точка своего же уравнения."""
    gamma = 0.9
    V = policy_evaluation(uniform, gamma=gamma)
    residual = 0.0
    for state in V:
        if state == TERMINAL:
            continue
        v = 0.0
        for action, pi_a in uniform(state).items():
            s_next, reward, _ = grid_step(state, action)
            v += pi_a * (reward + gamma * V[s_next])
        residual = max(residual, abs(v - V[state]))
    assert residual < 1e-6


def test_higher_gamma_makes_the_random_policy_look_worse():
    """Чем выше gamma, тем больше будущих -1 попадает в счёт."""
    near = policy_evaluation(uniform, gamma=0.5)
    far = policy_evaluation(uniform, gamma=0.99)
    assert far[(0, 0)] < near[(0, 0)] < 0.0


def test_the_optimal_policy_beats_the_uniform_one_in_every_state():
    good = policy_evaluation(staircase, gamma=0.99)
    bad = policy_evaluation(uniform, gamma=0.99)
    assert all(good[s] >= bad[s] - 1e-9 for s in good)
    assert good[(0, 0)] > bad[(0, 0)] + 10.0


# ---------------------------------------------------------------- q_from_v
def test_q_from_v_is_reward_plus_discounted_next_value():
    V = policy_evaluation(uniform, gamma=0.99)
    Q = q_from_v(V, gamma=0.99)
    assert Q[(0, 0)]["down"] == APPROX(-1.0 + 0.99 * V[(1, 0)])
    assert Q[(0, 0)]["up"] == APPROX(-1.0 + 0.99 * V[(0, 0)])


def test_q_from_v_is_zero_in_the_terminal():
    Q = q_from_v(policy_evaluation(uniform))
    assert flat({TERMINAL: Q[TERMINAL]}) == pytest.approx([0.0] * 4, abs=1e-12)


def test_q_and_v_agree_under_the_evaluated_policy():
    """V(s) == sum_a pi(a|s) Q(s,a). Если нет — потерян gamma или награда."""
    gamma = 0.95
    V = policy_evaluation(uniform, gamma=gamma)
    Q = q_from_v(V, gamma=gamma)
    for state in V:
        if state == TERMINAL:
            continue
        mixed = sum(0.25 * Q[state][a] for a in ACTIONS)
        assert mixed == pytest.approx(V[state], abs=1e-6)


# ------------------------------------------------------------ greedy_from_q
def test_greedy_from_q_picks_the_argmax():
    Q = {(0, 0): {"up": -9.0, "down": -5.0, "left": -7.0, "right": -6.0}}
    assert greedy_from_q(Q) == {(0, 0): "down"}


def test_greedy_from_q_breaks_ties_by_the_first_action():
    """Стабильный тай-брейк: иначе policy iteration колеблется вечно."""
    Q = {(1, 1): {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0}}
    assert greedy_from_q(Q) == {(1, 1): "up"}


def test_greedy_policy_on_optimal_values_walks_toward_the_terminal():
    """Жадная политика по V* обязана давать оптимальные действия."""
    V = policy_evaluation(staircase, gamma=0.99)
    policy = greedy_from_q(q_from_v(V, gamma=0.99))
    assert policy[(0, 0)] in ("down", "right")
    assert policy[(0, 3)] == "down"
    assert policy[(3, 0)] == "right"


def test_greedy_policy_over_the_random_policy_already_improves_it():
    """Один шаг policy improvement: жадность по V^uniform уже лучше uniform."""
    gamma = 0.99
    V_random = policy_evaluation(uniform, gamma=gamma)
    greedy = greedy_from_q(q_from_v(V_random, gamma=gamma))
    V_greedy = policy_evaluation(lambda s: {greedy[s]: 1.0}, gamma=gamma)
    assert V_greedy[(0, 0)] > V_random[(0, 0)]
    assert V_greedy[(0, 0)] == pytest.approx(OPTIMAL_G, abs=1e-6)
