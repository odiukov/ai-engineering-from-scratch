"""Тесты к уроку «Temporal Difference: Q-learning и SARSA». Правь exercise.py."""

import random

import pytest

from exercise import (
    bootstrap_expected_sarsa,
    bootstrap_q_learning,
    bootstrap_sarsa,
    epsilon_greedy_action,
    grid_step,
    q_learning,
    sarsa,
    td_error,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

ACTIONS = ("up", "down", "left", "right")
TERMINAL = (3, 3)

# V*(0,0) = шесть шагов по -1 с gamma=0.99
OPTIMAL_G = -(1 - 0.99 ** 6) / 0.01

ROW = {"up": -9.0, "down": -1.0, "left": -5.0, "right": -3.0}


def greedy_return(Q, start=(0, 0), limit=60):
    """Прогнать жадную по Q политику и вернуть сумму награды — проверка поведением."""
    state, total = start, 0.0
    for _ in range(limit):
        if state not in Q:
            return float("-inf")
        action = max(Q[state], key=Q[state].get)
        state, reward, done = grid_step(state, action)
        total += reward
        if done:
            return total
    return total


# --------------------------------------------------------------- grid_step
def test_grid_step_moves_and_bumps_into_walls():
    assert grid_step((0, 0), "down") == ((1, 0), -1.0, False)
    assert grid_step((0, 0), "left") == ((0, 0), -1.0, False)


def test_grid_step_flags_done_on_arrival():
    assert grid_step((2, 3), "down") == (TERMINAL, -1.0, True)


def test_grid_step_terminal_is_absorbing_and_free():
    for action in ACTIONS:
        assert grid_step(TERMINAL, action) == (TERMINAL, 0.0, True)


# ---------------------------------------------------------------- td_error
def test_td_error_is_reward_plus_discounted_next_minus_current():
    assert td_error(0.0, 1.0, 10.0, 0.0) == APPROX(10.0)
    assert td_error(-1.0, 0.5, -4.0, -2.0) == APPROX(-1.0 + 0.5 * -4.0 + 2.0)


def test_td_error_is_zero_at_the_bellman_fixed_point():
    """Оценка согласована с одним шагом среды — двигать её незачем."""
    gamma, value_next, reward = 0.9, -5.0, -1.0
    consistent = reward + gamma * value_next
    assert td_error(reward, gamma, value_next, consistent) == APPROX(0.0)


def test_td_error_sign_says_better_or_worse_than_expected():
    assert td_error(-1.0, 0.9, -1.0, -10.0) > 0  # оказалось лучше ожиданий
    assert td_error(-1.0, 0.9, -100.0, -1.0) < 0  # хуже


def test_td_error_with_gamma_zero_ignores_the_future():
    assert td_error(-1.0, 0.0, -99.0, -1.0) == APPROX(0.0)


# ----------------------------------------------------- epsilon_greedy_action
def test_epsilon_zero_is_fully_deterministic():
    rng = random.Random(0)
    assert {epsilon_greedy_action(ROW, rng, 0.0) for _ in range(200)} == {"down"}


def test_epsilon_one_eventually_tries_every_action():
    rng = random.Random(0)
    assert {epsilon_greedy_action(ROW, rng, 1.0) for _ in range(300)} == set(ACTIONS)


def test_epsilon_greedy_picks_the_greedy_action_with_the_right_frequency():
    """Жадное берётся с (1-eps) + eps/n: случайный выбор его тоже может вытянуть."""
    rng = random.Random(5)
    n = 20000
    hits = sum(epsilon_greedy_action(ROW, rng, 0.2) == "down" for _ in range(n))
    assert hits / n == pytest.approx(0.8 + 0.2 / 4, abs=0.02)


def test_epsilon_greedy_breaks_ties_on_the_first_action():
    flat_row = {a: 0.0 for a in ACTIONS}
    assert epsilon_greedy_action(flat_row, random.Random(0), 0.0) == "up"


# --------------------------------------------------------- bootstrap targets
def test_q_learning_bootstrap_takes_the_best_next_action():
    assert bootstrap_q_learning(ROW, False) == APPROX(-1.0)


def test_q_learning_bootstrap_is_zero_in_the_terminal():
    """Будущего нет — bootstrap строго 0.0, что бы ни лежало в строке."""
    assert bootstrap_q_learning(ROW, True) == APPROX(0.0)


def test_q_learning_bootstrap_does_not_depend_on_behaviour():
    """Off-policy: цель одна и та же, какое бы действие агент ни выбрал дальше."""
    assert bootstrap_q_learning(ROW, False) == APPROX(
        bootstrap_sarsa(ROW, "down", False)
    )
    assert bootstrap_q_learning(ROW, False) != bootstrap_sarsa(ROW, "up", False)


def test_q_learning_bootstrap_is_never_below_sarsa_bootstrap():
    """max по строке не может быть меньше конкретного элемента этой строки."""
    for action in ACTIONS:
        assert bootstrap_q_learning(ROW, False) >= bootstrap_sarsa(ROW, action, False)


def test_sarsa_bootstrap_uses_the_action_it_was_given():
    assert bootstrap_sarsa(ROW, "up", False) == APPROX(-9.0)
    assert bootstrap_sarsa(ROW, "right", False) == APPROX(-3.0)


def test_sarsa_bootstrap_is_zero_in_the_terminal():
    assert bootstrap_sarsa(ROW, None, True) == APPROX(0.0)


def test_expected_sarsa_bootstrap_with_greedy_probs_equals_q_learning():
    """eps -> 0 стирает разницу между Expected SARSA и Q-learning."""
    probs = {a: 0.0 for a in ACTIONS}
    probs["down"] = 1.0
    assert bootstrap_expected_sarsa(ROW, probs, False) == APPROX(
        bootstrap_q_learning(ROW, False)
    )


def test_expected_sarsa_bootstrap_with_uniform_probs_is_the_row_mean():
    probs = {a: 0.25 for a in ACTIONS}
    assert bootstrap_expected_sarsa(ROW, probs, False) == APPROX(
        sum(ROW.values()) / 4
    )


def test_expected_sarsa_bootstrap_stays_between_the_worst_and_best_action():
    probs = {"up": 0.1, "down": 0.4, "left": 0.2, "right": 0.3}
    value = bootstrap_expected_sarsa(ROW, probs, False)
    assert min(ROW.values()) <= value <= max(ROW.values())


def test_expected_sarsa_bootstrap_is_the_average_of_sampled_sarsa_bootstraps():
    """Expected SARSA — это в точности среднее SARSA по a', только без выборки."""
    probs = {"up": 0.1, "down": 0.4, "left": 0.2, "right": 0.3}
    rng = random.Random(7)
    names, weights = list(probs), [probs[a] for a in probs]
    sampled = [
        bootstrap_sarsa(ROW, rng.choices(names, weights)[0], False) for _ in range(30000)
    ]
    assert sum(sampled) / len(sampled) == pytest.approx(
        bootstrap_expected_sarsa(ROW, probs, False), abs=0.05
    )


def test_expected_sarsa_bootstrap_is_zero_in_the_terminal():
    probs = {a: 0.25 for a in ACTIONS}
    assert bootstrap_expected_sarsa(ROW, probs, True) == APPROX(0.0)


# -------------------------------------------------------------- q_learning
def test_q_learning_learning_curve_goes_up():
    _, returns = q_learning(1500, rng=random.Random(0))
    assert sum(returns[-200:]) / 200 > sum(returns[:200]) / 200


def test_q_learning_greedy_policy_walks_the_optimal_path():
    Q, _ = q_learning(3000, rng=random.Random(0))
    assert greedy_return(Q) == APPROX(-6.0)


def test_q_learning_q_values_converge_to_the_optimal_ones():
    Q, _ = q_learning(3000, rng=random.Random(0))
    assert max(Q[(0, 0)].values()) == pytest.approx(OPTIMAL_G, abs=0.5)
    assert max(Q[(3, 2)].values()) == pytest.approx(-1.0, abs=0.1)


def test_q_learning_finds_the_optimal_policy_from_purely_random_data():
    """epsilon=1 — поведение чисто случайное. Off-policy это не мешает: цель
    считается по max, а не по тому, что агент делал. SARSA на тех же данных
    выучит ценность случайной политики и промахнётся на десятки единиц."""
    Q_off, _ = q_learning(4000, epsilon=1.0, rng=random.Random(0))
    Q_on, _ = sarsa(4000, epsilon=1.0, rng=random.Random(0))
    assert max(Q_off[(0, 0)].values()) == pytest.approx(OPTIMAL_G, abs=1.0)
    assert greedy_return(Q_off) == APPROX(-6.0)
    assert max(Q_on[(0, 0)].values()) < OPTIMAL_G - 10.0


def test_q_learning_is_reproducible_for_the_same_seed():
    a = q_learning(200, rng=random.Random(3))[1]
    b = q_learning(200, rng=random.Random(3))[1]
    assert a == b


# -------------------------------------------------------------------- sarsa
def test_sarsa_learning_curve_goes_up():
    _, returns = sarsa(1500, rng=random.Random(0))
    assert sum(returns[-200:]) / 200 > sum(returns[:200]) / 200


def test_sarsa_greedy_policy_walks_the_optimal_path():
    Q, _ = sarsa(3000, rng=random.Random(0))
    assert greedy_return(Q) == APPROX(-6.0)


def test_sarsa_values_are_more_pessimistic_than_q_learning_at_the_same_epsilon():
    """On-policy включает в оценку стоимость собственных случайных шагов."""
    Q_sarsa, _ = sarsa(3000, epsilon=0.3, rng=random.Random(0))
    Q_ql, _ = q_learning(3000, epsilon=0.3, rng=random.Random(0))
    assert max(Q_sarsa[(0, 0)].values()) < max(Q_ql[(0, 0)].values())


def test_sarsa_is_reproducible_for_the_same_seed():
    assert sarsa(200, rng=random.Random(3))[1] == sarsa(200, rng=random.Random(3))[1]
