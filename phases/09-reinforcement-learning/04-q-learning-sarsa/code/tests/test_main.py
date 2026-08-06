"""Tests for tabular TD control: Q-learning, SARSA, epsilon-greedy behavior.

Covers phases/09-reinforcement-learning/04-q-learning-sarsa/docs/en.md.
Key properties: the TD error vanishes on the exact Q*, epsilon=0 makes action
selection deterministic, off-policy Q-learning reaches Q*(s_0) while on-policy
SARSA converges to Q^pi for the epsilon-greedy policy and stays short of it.
"""

from __future__ import annotations

import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from main import (  # noqa: E402
    ACTIONS,
    GRID,
    TERMINAL,
    block_means,
    epsilon_greedy,
    greedy_policy,
    q_learning,
    reset,
    sarsa,
    step,
)

GAMMA = 0.99
STATES = [(r, c) for r in range(GRID) for c in range(GRID)]
OPTIMAL_START_VALUE = -(1.0 - GAMMA**6) / (1.0 - GAMMA)  # -5.851985...


def exact_q_star(gamma=GAMMA):
    """Value iteration on the deterministic env defined by main.step, then Q* from V*."""
    values = {s: 0.0 for s in STATES}
    for _ in range(20000):
        delta = 0.0
        for state in STATES:
            if state == TERMINAL:
                continue
            best = max(
                reward + (0.0 if done else gamma * values[s_next])
                for s_next, reward, done in (step(state, a) for a in ACTIONS)
            )
            delta = max(delta, abs(best - values[state]))
            values[state] = best
        if delta < 1e-13:
            break
    q_star = {}
    for state in STATES:
        for action in ACTIONS:
            s_next, reward, done = step(state, action)
            q_star[(state, action)] = reward + (0.0 if done else gamma * values[s_next])
    return values, q_star


class TestExactQStarIsATdFixedPoint(unittest.TestCase):
    def test_td_error_is_zero_everywhere_on_the_exact_q_star(self) -> None:
        _values, q_star = exact_q_star()
        worst = 0.0
        for state in STATES:
            if state == TERMINAL:
                continue
            for action in ACTIONS:
                s_next, reward, done = step(state, action)
                target = reward if done else reward + GAMMA * max(
                    q_star[(s_next, a)] for a in ACTIONS
                )
                worst = max(worst, abs(target - q_star[(state, action)]))
        self.assertLess(worst, 1e-9)

    def test_optimal_start_value_matches_the_six_step_geometric_sum(self) -> None:
        values, _q_star = exact_q_star()
        self.assertAlmostEqual(values[reset()], OPTIMAL_START_VALUE, places=9)
        self.assertAlmostEqual(values[reset()], -5.851985, places=5)


class TestEpsilonGreedy(unittest.TestCase):
    def test_epsilon_zero_always_returns_the_argmax(self) -> None:
        table = {(0, 0): {"up": -3.0, "down": -1.0, "left": -9.0, "right": -2.0}}
        rng = random.Random(0)
        for _ in range(200):
            self.assertEqual(epsilon_greedy(table, (0, 0), rng, 0.0), "down")

    def test_epsilon_one_explores_every_action(self) -> None:
        table = {(0, 0): {a: 0.0 for a in ACTIONS}}
        rng = random.Random(1)
        picked = {epsilon_greedy(table, (0, 0), rng, 1.0) for _ in range(400)}
        self.assertEqual(picked, set(ACTIONS))

    def test_selection_is_reproducible_under_a_seeded_rng(self) -> None:
        table = {(0, 0): {a: float(i) for i, a in enumerate(ACTIONS)}}
        a = [epsilon_greedy(table, (0, 0), random.Random(5), 0.5) for _ in range(1)]
        b = [epsilon_greedy(table, (0, 0), random.Random(5), 0.5) for _ in range(1)]
        self.assertEqual(a, b)

    def test_ties_break_on_the_first_action_in_fixed_order(self) -> None:
        table = {(1, 1): {a: -4.0 for a in ACTIONS}}
        self.assertEqual(epsilon_greedy(table, (1, 1), random.Random(0), 0.0), ACTIONS[0])


class TestQLearning(unittest.TestCase):
    def test_learns_the_exact_value_of_the_start_state(self) -> None:
        table, _log = q_learning(6000, rng=random.Random(7))
        self.assertAlmostEqual(max(table[reset()].values()), OPTIMAL_START_VALUE, delta=0.01)

    def test_transition_into_the_terminal_cell_is_learned_exactly(self) -> None:
        table, _log = q_learning(6000, rng=random.Random(7))
        self.assertAlmostEqual(table[(3, 2)]["right"], -1.0, delta=0.02)
        self.assertAlmostEqual(table[(2, 3)]["down"], -1.0, delta=0.02)

    def test_residual_against_q_star_is_bounded_and_lives_in_rare_corners(self) -> None:
        _values, q_star = exact_q_star()
        table, _log = q_learning(10000, rng=random.Random(7))
        errors = {
            (s, a): abs(table[s][a] - q_star[(s, a)])
            for s in STATES
            if s != TERMINAL
            for a in ACTIONS
        }
        self.assertLess(max(errors.values()), 1.5)
        worst_state = max(errors, key=errors.get)[0]
        self.assertGreater(sum(worst_state), 1, "worst residual should not sit on the optimal path")

    def test_greedy_policy_reaches_the_goal_in_the_optimal_six_steps(self) -> None:
        table, _log = q_learning(4000, rng=random.Random(7))
        policy = greedy_policy(table)
        state, steps = reset(), 0
        while state != TERMINAL and steps < 50:
            state, _r, _done = step(state, policy[state])
            steps += 1
        self.assertEqual(state, TERMINAL)
        self.assertEqual(steps, 6)

    def test_training_is_reproducible_under_a_seeded_rng(self) -> None:
        q1, l1 = q_learning(300, rng=random.Random(3))
        q2, l2 = q_learning(300, rng=random.Random(3))
        self.assertEqual(l1, l2)
        self.assertEqual({s: dict(v) for s, v in q1.items()}, {s: dict(v) for s, v in q2.items()})


class TestSarsaVersusQLearning(unittest.TestCase):
    def test_sarsa_converges_to_the_epsilon_greedy_value_not_to_q_star(self) -> None:
        table_sarsa, _ = sarsa(10000, epsilon=0.1, rng=random.Random(7))
        table_ql, _ = q_learning(10000, epsilon=0.1, rng=random.Random(7))
        v_sarsa = max(table_sarsa[reset()].values())
        v_ql = max(table_ql[reset()].values())
        self.assertLess(v_sarsa, OPTIMAL_START_VALUE - 0.1)
        self.assertGreater(v_ql, v_sarsa)
        self.assertAlmostEqual(v_ql, OPTIMAL_START_VALUE, delta=0.01)

    def test_both_methods_improve_their_return_over_training(self) -> None:
        for learner in (sarsa, q_learning):
            _table, log = learner(3000, rng=random.Random(42))
            blocks = block_means(log, 500)
            self.assertGreater(blocks[-1], blocks[0] + 2.0, f"{learner.__name__} did not improve")
            self.assertLess(blocks[-1], -6.0)

    def test_neither_method_beats_the_undiscounted_optimum_of_minus_six(self) -> None:
        for learner in (sarsa, q_learning):
            _table, log = learner(2000, rng=random.Random(11))
            self.assertLessEqual(max(log), -6.0)

    def test_sarsa_training_is_reproducible_under_a_seeded_rng(self) -> None:
        _q1, l1 = sarsa(300, rng=random.Random(9))
        _q2, l2 = sarsa(300, rng=random.Random(9))
        self.assertEqual(l1, l2)


if __name__ == "__main__":
    unittest.main()
