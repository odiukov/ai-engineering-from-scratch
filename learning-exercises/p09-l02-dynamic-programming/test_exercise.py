"""Тесты к уроку «Динамическое программирование: policy iteration и value iteration»."""

import random

import pytest

from exercise import (
    bellman_sweep,
    greedy_policy,
    policy_evaluation,
    policy_iteration,
    q_value,
    sup_norm,
    transitions,
    value_iteration,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

ACTIONS = ("up", "down", "left", "right")
STATES = [(r, c) for r in range(4) for c in range(4)]
TERMINAL = (3, 3)
ZEROS = {s: 0.0 for s in STATES}

# V*(0,0) без скольжения: шесть шагов по -1 с gamma=0.99
OPTIMAL_V0 = -(1 - 0.99 ** 6) / 0.01


def staircase(state):
    """Оптимальная политика на детерминированной сетке: вниз, потом вправо."""
    return {"down": 1.0} if state[0] < 3 else {"right": 1.0}


def uniform(_state):
    return {a: 0.25 for a in ACTIONS}


def random_values(seed):
    rng = random.Random(seed)
    return {s: rng.uniform(-20.0, 20.0) for s in STATES}


def flat(mapping):
    """approx не умеет вложенные структуры — разворачиваем в плоский список."""
    return [mapping[s] for s in sorted(mapping)]


# ------------------------------------------------------------- transitions
def test_transitions_are_deterministic_without_slip():
    assert transitions((0, 0), "down") == [((1, 0), -1.0, 1.0)]
    assert transitions((0, 0), "up") == [((0, 0), -1.0, 1.0)]


def test_transition_probabilities_always_sum_to_one():
    for slip in (0.0, 0.1, 0.5):
        for state in STATES:
            for action in ACTIONS:
                total = sum(p for _, _, p in transitions(state, action, slip))
                assert total == pytest.approx(1.0, abs=1e-12)


def test_slip_splits_the_leftover_probability_between_the_perpendiculars():
    outcomes = transitions((1, 1), "down", slip=0.1)
    by_state = {s: p for s, _, p in outcomes}
    assert by_state[(2, 1)] == APPROX(0.9)
    assert by_state[(1, 0)] == APPROX(0.05)
    assert by_state[(1, 2)] == APPROX(0.05)


def test_slip_never_reverses_the_action():
    """Соскальзывание — только в перпендикуляр. Назад агента не отбрасывает."""
    reachable = {s for s, _, p in transitions((1, 1), "down", slip=0.5) if p > 0}
    assert (0, 1) not in reachable
    reachable = {s for s, _, p in transitions((1, 1), "right", slip=0.5) if p > 0}
    assert (1, 0) not in reachable


def test_transitions_from_the_terminal_are_absorbing_and_free():
    for action in ACTIONS:
        assert transitions(TERMINAL, action, slip=0.3) == [(TERMINAL, 0.0, 1.0)]


# ---------------------------------------------------------------- sup_norm
def test_sup_norm_is_the_largest_gap():
    assert sup_norm({1: 0.0, 2: 0.0}, {1: 3.0, 2: -1.0}) == APPROX(3.0)


def test_sup_norm_of_identical_values_is_zero():
    V = random_values(1)
    assert sup_norm(V, V) == APPROX(0.0)


def test_sup_norm_takes_the_max_not_the_mean():
    """Одно несошедшееся состояние обязано быть видно, а среднее его прячет."""
    a = {i: 0.0 for i in range(100)}
    b = dict(a)
    b[57] = 5.0
    assert sup_norm(a, b) == APPROX(5.0)


# ----------------------------------------------------------------- q_value
def test_q_value_from_zero_values_is_just_the_step_reward():
    assert q_value((0, 0), "down", ZEROS) == APPROX(-1.0)
    assert q_value((3, 2), "right", ZEROS) == APPROX(-1.0)


def test_q_value_discounts_only_the_next_value_not_the_reward():
    """gamma умножает V(s'), но не награду за шаг."""
    V = dict(ZEROS)
    V[(1, 0)] = -10.0
    assert q_value((0, 0), "down", V, gamma=0.5) == APPROX(-1.0 + 0.5 * -10.0)


def test_q_value_mixes_outcomes_by_their_probability():
    V = dict(ZEROS)
    V[(2, 1)] = -10.0
    V[(1, 0)] = -2.0
    V[(1, 2)] = -4.0
    expected = 0.8 * (-1 + 0.9 * -10.0) + 0.1 * (-1 + 0.9 * -2.0) + 0.1 * (-1 + 0.9 * -4.0)
    assert q_value((1, 1), "down", V, gamma=0.9, slip=0.2) == APPROX(expected)


def test_q_value_at_the_terminal_ignores_the_rest_of_the_board():
    """Терминал absorbing: награда 0, переход в себя. Соседние значения ни при чём."""
    V = random_values(2)
    V[TERMINAL] = 0.0
    assert q_value(TERMINAL, "up", V, gamma=0.99) == APPROX(0.0)
    assert q_value(TERMINAL, "left", V, gamma=0.99) == APPROX(0.0)


# ----------------------------------------------------------- bellman_sweep
def test_bellman_sweep_from_zeros_gives_the_one_step_reward():
    V = bellman_sweep(ZEROS)
    assert V[TERMINAL] == APPROX(0.0)
    assert flat({s: V[s] for s in STATES if s != TERMINAL}) == pytest.approx(
        [-1.0] * 15, abs=1e-12
    )


def test_bellman_sweep_does_not_mutate_its_input():
    V = random_values(3)
    snapshot = dict(V)
    bellman_sweep(V)
    assert flat(V) == pytest.approx(flat(snapshot), abs=1e-12)


def test_bellman_sweep_takes_the_max_not_the_average():
    """Оператор оптимальности берёт лучшее действие, а не усредняет по действиям."""
    V = dict(ZEROS)
    V[(1, 0)] = 100.0  # вниз из (0,0) внезапно очень выгодно
    swept = bellman_sweep(V, gamma=0.9)
    assert swept[(0, 0)] == APPROX(-1.0 + 0.9 * 100.0)


def test_bellman_sweep_is_a_gamma_contraction_in_sup_norm():
    """Причина, по которой DP вообще сходится: расстояние сжимается в gamma раз."""
    gamma = 0.9
    for seed in (0, 1, 2, 3):
        v_a, v_b = random_values(seed), random_values(seed + 100)
        before = sup_norm(v_a, v_b)
        after = sup_norm(bellman_sweep(v_a, gamma), bellman_sweep(v_b, gamma))
        assert after <= gamma * before + 1e-12
        assert before > 0.0


def test_bellman_sweep_leaves_the_optimal_values_alone():
    """V* — неподвижная точка: ещё один проход её не двигает."""
    V_star, _, _ = value_iteration(gamma=0.99)
    assert sup_norm(bellman_sweep(V_star, gamma=0.99), V_star) < 1e-9


# -------------------------------------------------------- policy_evaluation
def test_policy_evaluation_matches_the_closed_form_for_the_optimal_policy():
    V = policy_evaluation(staircase, gamma=0.99)
    assert V[(0, 0)] == pytest.approx(OPTIMAL_V0, abs=1e-6)
    assert V[TERMINAL] == APPROX(0.0)


def test_policy_evaluation_is_a_fixed_point_of_its_own_equation():
    gamma = 0.95
    V = policy_evaluation(uniform, gamma=gamma)
    residual = 0.0
    for state in STATES:
        if state == TERMINAL:
            continue
        v = sum(0.25 * q_value(state, a, V, gamma) for a in ACTIONS)
        residual = max(residual, abs(v - V[state]))
    assert residual < 1e-6


def test_slip_makes_every_policy_worse():
    """Шум в динамике не может помочь: та же политика на скользкой сетке хуже."""
    dry = policy_evaluation(staircase, gamma=0.99, slip=0.0)
    icy = policy_evaluation(staircase, gamma=0.99, slip=0.2)
    assert icy[(0, 0)] < dry[(0, 0)]


def test_policy_evaluation_ranks_the_optimal_policy_above_the_random_one():
    good = policy_evaluation(staircase, gamma=0.99)
    bad = policy_evaluation(uniform, gamma=0.99)
    assert all(good[s] >= bad[s] - 1e-9 for s in STATES)
    assert good[(0, 0)] > bad[(0, 0)] + 10.0


# ----------------------------------------------------------- greedy_policy
def test_greedy_policy_on_optimal_values_walks_toward_the_terminal():
    V_star, _, _ = value_iteration(gamma=0.99)
    policy = greedy_policy(V_star, gamma=0.99)
    assert policy[(0, 0)] in ("down", "right")
    assert policy[(0, 3)] == "down"
    assert policy[(3, 0)] == "right"
    assert policy[(2, 3)] == "down"


def test_greedy_policy_breaks_ties_by_action_order():
    """Все Q равны — берём первое действие, и так каждый раз."""
    policy = greedy_policy(ZEROS)
    assert policy[(1, 1)] == "up"
    assert greedy_policy(ZEROS) == policy


def test_greedy_policy_improves_the_random_policy_in_one_step():
    gamma = 0.99
    V_random = policy_evaluation(uniform, gamma=gamma)
    improved = greedy_policy(V_random, gamma=gamma)
    V_improved = policy_evaluation(lambda s: {improved[s]: 1.0}, gamma=gamma)
    assert V_improved[(0, 0)] > V_random[(0, 0)]


def test_greedy_policy_assigns_an_action_to_every_state():
    policy = greedy_policy(ZEROS)
    assert set(policy) == set(STATES)


# ---------------------------------------------------------- value_iteration
def test_value_iteration_converges_to_the_bellman_fixed_point():
    V_star, _, sweeps = value_iteration(gamma=0.9)
    assert sup_norm(bellman_sweep(V_star, gamma=0.9), V_star) < 1e-9
    assert sweeps > 1


def test_value_iteration_matches_the_closed_form_without_slip():
    V_star, policy, _ = value_iteration(gamma=0.99)
    assert V_star[(0, 0)] == pytest.approx(OPTIMAL_V0, abs=1e-6)
    assert policy[(0, 0)] in ("down", "right")


def test_optimal_values_dominate_every_policy():
    """V* — верхняя граница: никакая политика не может её обойти."""
    V_star, _, _ = value_iteration(gamma=0.99)
    for policy in (uniform, staircase):
        V_pi = policy_evaluation(policy, gamma=0.99)
        assert all(V_star[s] >= V_pi[s] - 1e-6 for s in STATES)


def test_value_iteration_needs_more_sweeps_when_gamma_is_higher():
    """Ошибка падает как gamma^n, поэтому 0.99 сходится заметно дольше 0.9.

    Сетку берём скользкую: на детерминированной значения становятся точными
    за диаметр сетки шагов, и разницу между gamma не увидеть.
    """
    sweeps = [value_iteration(gamma=g, slip=0.2, tol=1e-9)[2] for g in (0.5, 0.9, 0.99)]
    assert sweeps[0] < sweeps[1] < sweeps[2]


def test_value_iteration_on_the_slippery_grid_is_worse_but_still_sane():
    dry, _, _ = value_iteration(gamma=0.99, slip=0.0)
    icy, policy, _ = value_iteration(gamma=0.99, slip=0.3)
    assert icy[(0, 0)] < dry[(0, 0)]
    assert policy[(0, 0)] in ("down", "right")


# --------------------------------------------------------- policy_iteration
def test_policy_iteration_lands_on_the_same_fixed_point_as_value_iteration():
    """Неподвижная точка одна, значит и V, и политика обязаны совпасть."""
    V_pi, pol_pi, _ = policy_iteration(gamma=0.99)
    V_vi, pol_vi, _ = value_iteration(gamma=0.99)
    assert sup_norm(V_pi, V_vi) < 1e-6
    assert pol_pi == pol_vi


def test_policy_iteration_converges_in_a_handful_of_outer_iterations():
    _, _, outer = policy_iteration(gamma=0.99)
    assert 1 < outer <= 10


def test_policy_iteration_result_is_greedy_with_respect_to_its_own_values():
    """Критерий остановки: улучшать больше нечего."""
    V, policy, _ = policy_iteration(gamma=0.99, slip=0.2)
    assert greedy_policy(V, gamma=0.99, slip=0.2) == policy


def test_policy_iteration_escapes_the_useless_starting_policy():
    """Старт — «всегда вверх», из угла он никуда не ведёт. Алгоритм обязан выбраться."""
    V, policy, _ = policy_iteration(gamma=0.99)
    stuck = policy_evaluation(lambda s: {"up": 1.0}, gamma=0.99)
    assert V[(0, 0)] > stuck[(0, 0)]
    assert policy[(0, 0)] != "up"
