"""Тесты к уроку «Monte Carlo: обучение по полным эпизодам». Правь exercise.py."""

import random

import pytest

from exercise import (
    constant_alpha_mc,
    epsilon_greedy_action,
    grid_step,
    incremental_mean,
    mc_control,
    mc_evaluate,
    returns_from,
    rollout,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

ACTIONS = ("up", "down", "left", "right")
STATES = [(r, c) for r in range(4) for c in range(4)]
TERMINAL = (3, 3)

# G оптимального пути из (0,0): шесть шагов по -1 с gamma=0.99
OPTIMAL_G = -(1 - 0.99 ** 6) / 0.01


def optimal(state, _rng):
    """Вниз до нижнего ряда, потом вправо — ровно шесть шагов из (0,0)."""
    return "down" if state[0] < 3 else "right"


def uniform(_state, rng):
    return rng.choice(ACTIONS)


def always_up(_state, _rng):
    return "up"


def dp_values(gamma, policy_probs):
    """Точное V^pi через уравнение Беллмана — эталон, с которым сверяем MC."""
    V = {s: 0.0 for s in STATES}
    for _ in range(20000):
        delta = 0.0
        for state in STATES:
            if state == TERMINAL:
                continue
            v = 0.0
            for action, p in policy_probs.items():
                s_next, reward, _ = grid_step(state, action)
                v += p * (reward + gamma * V[s_next])
            delta = max(delta, abs(v - V[state]))
            V[state] = v
        if delta < 1e-12:
            break
    return V


# --------------------------------------------------------------- grid_step
def test_grid_step_moves_and_bumps_into_walls():
    assert grid_step((0, 0), "down") == ((1, 0), -1.0, False)
    assert grid_step((0, 0), "up") == ((0, 0), -1.0, False)


def test_grid_step_terminal_is_absorbing_and_free():
    for action in ACTIONS:
        assert grid_step(TERMINAL, action) == (TERMINAL, 0.0, True)


def test_grid_step_flags_done_on_arrival():
    assert grid_step((3, 2), "right") == (TERMINAL, -1.0, True)


# ------------------------------------------------------------- returns_from
def test_returns_from_discounts_the_tail():
    traj = [((0, 0), "down", 1.0), ((1, 0), "down", 1.0)]
    assert returns_from(traj, gamma=0.5) == pytest.approx([1.5, 1.0], abs=1e-12)


def test_returns_from_with_gamma_one_is_suffix_sums():
    traj = [((0, 0), "down", -1.0)] * 6
    assert returns_from(traj, gamma=1.0) == pytest.approx(
        [-6.0, -5.0, -4.0, -3.0, -2.0, -1.0], abs=1e-12
    )


def test_returns_from_last_element_is_the_last_reward():
    traj = [((0, 0), "down", -1.0), ((1, 0), "down", 7.0)]
    assert returns_from(traj, gamma=0.99)[-1] == APPROX(7.0)


def test_returns_from_satisfies_the_backward_recurrence():
    """G_t = r_t + gamma * G_{t+1} — это и есть определение, проверим его прямо."""
    rng = random.Random(4)
    traj = [((0, 0), "up", rng.uniform(-3, 3)) for _ in range(25)]
    gamma = 0.9
    G = returns_from(traj, gamma)
    for t in range(len(traj) - 1):
        assert G[t] == pytest.approx(traj[t][2] + gamma * G[t + 1], abs=1e-9)


def test_returns_from_of_the_optimal_path():
    traj = [((0, 0), "down", -1.0)] * 6
    assert returns_from(traj, gamma=0.99)[0] == APPROX(OPTIMAL_G)
    assert len(returns_from(traj, 0.99)) == 6


# --------------------------------------------------------- incremental_mean
def test_incremental_mean_of_the_first_sample_is_the_sample():
    assert incremental_mean(0.0, 10.0, 1) == APPROX(10.0)


def test_incremental_mean_reproduces_the_arithmetic_mean():
    """Применённое подряд, оно обязано дать в точности обычное среднее."""
    values = [3.0, -1.0, 7.5, 0.25, -12.0]
    mean = 0.0
    for i, v in enumerate(values, start=1):
        mean = incremental_mean(mean, v, i)
    assert mean == pytest.approx(sum(values) / len(values), abs=1e-12)


def test_incremental_mean_moves_less_as_the_count_grows():
    early = abs(incremental_mean(0.0, 100.0, 2) - 0.0)
    late = abs(incremental_mean(0.0, 100.0, 500) - 0.0)
    assert late < early / 100


def test_incremental_mean_ignores_a_sample_equal_to_the_mean():
    assert incremental_mean(5.0, 5.0, 7) == APPROX(5.0)


# ----------------------------------------------------------------- rollout
def test_rollout_of_the_optimal_policy_is_six_steps_long():
    traj = rollout(optimal, random.Random(0))
    assert len(traj) == 6
    assert [s for s, _, _ in traj] == [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2)]
    assert all(r == -1.0 for _, _, r in traj)


def test_rollout_does_not_include_the_terminal_as_a_visited_state():
    """Из терминала действий не делают, значит его в траектории быть не должно."""
    traj = rollout(optimal, random.Random(0))
    assert TERMINAL not in [s for s, _, _ in traj]


def test_rollout_stops_at_max_steps_for_a_stuck_policy():
    assert len(rollout(always_up, random.Random(0), max_steps=12)) == 12


def test_rollout_is_reproducible_for_the_same_seed():
    assert rollout(uniform, random.Random(9)) == rollout(uniform, random.Random(9))


# -------------------------------------------------------------- mc_evaluate
def test_mc_evaluate_is_exact_when_the_policy_has_no_randomness():
    """Детерминированная политика даёт один и тот же G, дисперсии нет вовсе."""
    V, counts = mc_evaluate(optimal, 20, gamma=0.99, rng=random.Random(0))
    assert V[(0, 0)] == pytest.approx(OPTIMAL_G, abs=1e-9)
    assert counts[(0, 0)] == 20


def test_mc_evaluate_only_knows_the_states_it_visited():
    """DP видит всю сетку по построению, MC — только то, куда его занесло."""
    V, _ = mc_evaluate(optimal, 20, rng=random.Random(0))
    assert set(V) == {(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2)}
    assert TERMINAL not in V


def test_mc_evaluate_converges_to_the_dp_answer():
    gamma = 0.9
    V, _ = mc_evaluate(uniform, 4000, gamma=gamma, rng=random.Random(3))
    exact = dp_values(gamma, {a: 0.25 for a in ACTIONS})
    assert V[(0, 0)] == pytest.approx(exact[(0, 0)], abs=0.5)
    assert V[(3, 2)] == pytest.approx(exact[(3, 2)], abs=0.5)


def test_first_visit_counts_a_state_once_per_episode():
    _, counts = mc_evaluate(uniform, 300, rng=random.Random(1), first_visit=True)
    assert all(n <= 300 for n in counts.values())
    assert counts[(0, 0)] == 300


def test_every_visit_uses_more_samples_than_first_visit():
    """Случайная политика возвращается в старт по многу раз — every-visit это считает."""
    _, first = mc_evaluate(uniform, 300, rng=random.Random(1), first_visit=True)
    _, every = mc_evaluate(uniform, 300, rng=random.Random(1), first_visit=False)
    assert every[(0, 0)] > first[(0, 0)]
    assert sum(every.values()) > sum(first.values())


# --------------------------------------------------------- constant_alpha_mc
def test_constant_alpha_one_remembers_only_the_last_episode():
    """alpha=1 полностью выкидывает историю: V(s) = G последнего визита."""
    V = constant_alpha_mc(optimal, 50, alpha=1.0, gamma=0.99, rng=random.Random(0))
    assert V[(0, 0)] == pytest.approx(OPTIMAL_G, abs=1e-9)
    assert V[(3, 2)] == pytest.approx(-1.0, abs=1e-9)


def test_constant_alpha_estimate_stays_inside_the_observed_returns():
    """При 0 < alpha <= 1 оценка — выпуклая комбинация G, вылететь она не может."""
    V = constant_alpha_mc(uniform, 400, alpha=0.2, gamma=0.9, rng=random.Random(2))
    assert all(-10.0 <= v <= 0.0 for v in V.values())


def test_constant_alpha_matches_the_plain_mean_on_a_noiseless_policy():
    smooth = constant_alpha_mc(optimal, 200, alpha=0.3, gamma=0.99, rng=random.Random(0))
    averaged, _ = mc_evaluate(optimal, 200, gamma=0.99, rng=random.Random(0))
    assert smooth[(0, 0)] == pytest.approx(averaged[(0, 0)], abs=1e-6)


def test_small_alpha_leaves_the_estimate_close_to_its_zero_start():
    """С крошечной alpha за один эпизод оценка почти не сдвинется с нуля."""
    V = constant_alpha_mc(optimal, 1, alpha=0.001, gamma=0.99, rng=random.Random(0))
    assert V[(0, 0)] == pytest.approx(0.001 * OPTIMAL_G, abs=1e-12)


# ----------------------------------------------------- epsilon_greedy_action
def test_epsilon_zero_is_fully_deterministic():
    row = {"up": -9.0, "down": -1.0, "left": -5.0, "right": -3.0}
    rng = random.Random(0)
    assert {epsilon_greedy_action(row, rng, 0.0) for _ in range(200)} == {"down"}


def test_epsilon_one_eventually_tries_every_action():
    row = {"up": -9.0, "down": -1.0, "left": -5.0, "right": -3.0}
    rng = random.Random(0)
    assert {epsilon_greedy_action(row, rng, 1.0) for _ in range(200)} == set(ACTIONS)


def test_epsilon_greedy_picks_the_greedy_action_with_the_right_frequency():
    """Жадное берётся с (1-eps) + eps/n, а не с (1-eps): случайный выбор его тоже включает."""
    row = {a: 0.0 for a in ACTIONS}
    row["down"] = 1.0
    rng = random.Random(5)
    n = 20000
    hits = sum(epsilon_greedy_action(row, rng, 0.2) == "down" for _ in range(n))
    assert hits / n == pytest.approx(0.8 + 0.2 / 4, abs=0.02)


def test_epsilon_greedy_breaks_ties_on_the_first_action():
    row = {a: 0.0 for a in ACTIONS}
    assert epsilon_greedy_action(row, random.Random(0), 0.0) == "up"


def test_epsilon_greedy_is_reproducible_for_the_same_seed():
    row = {"up": 1.0, "down": 1.0, "left": 0.0, "right": 0.0}
    a = [epsilon_greedy_action(row, random.Random(8), 0.5) for _ in range(1)]
    rng_1, rng_2 = random.Random(13), random.Random(13)
    assert [epsilon_greedy_action(row, rng_1, 0.5) for _ in range(40)] == [
        epsilon_greedy_action(row, rng_2, 0.5) for _ in range(40)
    ]
    assert a[0] in ACTIONS


# --------------------------------------------------------------- mc_control
def test_mc_control_recovers_a_policy_that_heads_for_the_terminal():
    _, greedy = mc_control(6000, gamma=0.99, epsilon=0.2, rng=random.Random(0))
    assert greedy[(0, 0)] in ("down", "right")
    assert greedy[(3, 2)] == "right"
    assert greedy[(2, 3)] == "down"


def test_mc_control_greedy_policy_walks_the_optimal_path():
    """Итог обучения проверяем поведением: жадная политика обязана дать -6."""
    _, greedy = mc_control(6000, gamma=0.99, epsilon=0.2, rng=random.Random(0))
    total, state = 0.0, (0, 0)
    for _ in range(50):
        state, reward, done = grid_step(state, greedy[state])
        total += reward
        if done:
            break
    assert total == APPROX(-6.0)


def test_mc_control_learns_the_value_of_its_own_epsilon_greedy_policy():
    """On-policy: Q сходится к Q^pi текущей eps-жадной политики, а НЕ к Q*.

    Плата за исследование сидит прямо в оценке, поэтому Q(0,0) хуже
    оптимального -5.85, и тем хуже, чем больше epsilon. Off-policy
    Q-learning из урока 04 этой платы не платит.
    """
    loose, _ = mc_control(6000, gamma=0.99, epsilon=0.3, rng=random.Random(0))
    tight, _ = mc_control(6000, gamma=0.99, epsilon=0.05, rng=random.Random(0))
    assert max(loose[(0, 0)].values()) < max(tight[(0, 0)].values()) < OPTIMAL_G
    assert max(tight[(0, 0)].values()) == pytest.approx(OPTIMAL_G, abs=1.0)
    assert max(loose[(3, 2)].values()) == pytest.approx(-1.0, abs=1e-9)


def test_mc_control_never_reports_a_positive_q_on_a_penalty_only_grid():
    """Все награды -1 или 0, значит положительного Q взяться неоткуда."""
    Q, _ = mc_control(500, gamma=0.99, epsilon=0.2, rng=random.Random(0))
    assert all(v <= 0.0 for row in Q.values() for v in row.values())
