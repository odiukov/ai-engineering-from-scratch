"""Tests for the 4x4 GridWorld MDP: transitions, rollouts, Bellman fixed point.

Covers phases/09-reinforcement-learning/01-mdps-states-actions-rewards/docs/en.md.
Every property here is a claim the lesson makes about (S, A, P, R, gamma):
absorbing terminal, clamping walls, and V^pi as the Bellman fixed point.
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
    all_states,
    down_right_policy,
    policy_evaluation,
    rollout,
    sample_action,
    step,
    uniform_policy,
)


def straight_policy(state):
    """Deterministic optimal policy: go down until the bottom row, then right."""
    row, _ = state
    return {"down": 1.0} if row < GRID - 1 else {"right": 1.0}


def bellman_residual(policy, values, gamma):
    """max_s |V(s) - sum_a pi(a|s) (r + gamma V(s'))| over non-terminal states."""
    worst = 0.0
    for state in all_states():
        if state == TERMINAL:
            continue
        backup = 0.0
        for action, pi_a in policy(state).items():
            s_next, reward, _ = step(state, action)
            backup += pi_a * (reward + gamma * values[s_next])
        worst = max(worst, abs(backup - values[state]))
    return worst


class TestEnvironmentIsAWellFormedMDP(unittest.TestCase):
    def test_terminal_state_is_absorbing_and_reward_free(self) -> None:
        for action in ACTIONS:
            s_next, reward, done = step(TERMINAL, action)
            self.assertEqual(s_next, TERMINAL)
            self.assertEqual(reward, 0.0)
            self.assertTrue(done)

    def test_walls_clamp_instead_of_leaving_the_grid(self) -> None:
        self.assertEqual(step((0, 0), "up")[0], (0, 0))
        self.assertEqual(step((0, 0), "left")[0], (0, 0))
        self.assertEqual(step((3, 2), "down")[0], (3, 2))
        self.assertEqual(step((0, 3), "right")[0], (0, 3))

    def test_every_non_terminal_step_costs_exactly_one(self) -> None:
        for state in all_states():
            if state == TERMINAL:
                continue
            for action in ACTIONS:
                self.assertEqual(step(state, action)[1], -1.0)

    def test_done_flag_is_set_only_on_entering_the_terminal_cell(self) -> None:
        self.assertTrue(step((3, 2), "right")[2])
        self.assertTrue(step((2, 3), "down")[2])
        self.assertFalse(step((3, 1), "right")[2])

    def test_policies_are_normalized_distributions(self) -> None:
        for policy in (uniform_policy, down_right_policy):
            dist = policy((1, 1))
            self.assertAlmostEqual(sum(dist.values()), 1.0, places=12)
            self.assertTrue(all(p >= 0.0 for p in dist.values()))


class TestSamplingAndRollout(unittest.TestCase):
    def test_degenerate_distribution_is_sampled_deterministically(self) -> None:
        rng = random.Random(0)
        for _ in range(50):
            self.assertEqual(sample_action({"down": 1.0, "up": 0.0}, rng), "down")

    def test_sample_action_frequencies_track_the_distribution(self) -> None:
        rng = random.Random(7)
        counts = {a: 0 for a in ACTIONS}
        draws = 20000
        for _ in range(draws):
            counts[sample_action(down_right_policy(None), rng)] += 1
        self.assertAlmostEqual(counts["down"] / draws, 0.5, delta=0.02)
        self.assertAlmostEqual(counts["right"] / draws, 0.5, delta=0.02)
        self.assertEqual(counts["up"], 0)
        self.assertEqual(counts["left"], 0)

    def test_optimal_straight_line_rollout_costs_six_steps(self) -> None:
        total, steps = rollout(straight_policy, random.Random(0))
        self.assertEqual(steps, 6)
        self.assertEqual(total, -6.0)

    def test_rollout_is_reproducible_under_a_seeded_rng(self) -> None:
        first = [rollout(uniform_policy, random.Random(123)) for _ in range(3)]
        second = [rollout(uniform_policy, random.Random(123)) for _ in range(3)]
        self.assertEqual(first, second)

    def test_random_policy_is_far_worse_than_the_optimal_six_steps(self) -> None:
        rng = random.Random(42)
        returns = [rollout(uniform_policy, rng)[0] for _ in range(2000)]
        mean = sum(returns) / len(returns)
        self.assertLess(mean, -30.0)
        self.assertGreater(mean, -100.0)


class TestPolicyEvaluationIsABellmanFixedPoint(unittest.TestCase):
    def test_terminal_value_stays_zero(self) -> None:
        for gamma in (0.5, 0.9, 0.99):
            self.assertEqual(policy_evaluation(uniform_policy, gamma=gamma)[TERMINAL], 0.0)

    def test_uniform_policy_values_satisfy_the_bellman_equation(self) -> None:
        for gamma in (0.5, 0.9, 0.99):
            values = policy_evaluation(uniform_policy, gamma=gamma)
            self.assertLess(bellman_residual(uniform_policy, values, gamma), 1e-5)

    def test_fixed_point_is_independent_of_the_iteration_budget(self) -> None:
        loose = policy_evaluation(uniform_policy, gamma=0.9, tol=1e-9, max_iter=300)
        tight = policy_evaluation(uniform_policy, gamma=0.9, tol=1e-12, max_iter=3000)
        self.assertLess(max(abs(loose[s] - tight[s]) for s in all_states()), 1e-6)

    def test_straight_line_policy_value_matches_the_closed_form_geometric_sum(self) -> None:
        gamma = 0.99
        values = policy_evaluation(straight_policy, gamma=gamma)
        expected = -(1.0 - gamma**6) / (1.0 - gamma)
        self.assertAlmostEqual(values[(0, 0)], expected, places=6)
        self.assertAlmostEqual(values[(0, 0)], -5.851985, places=5)

    def test_larger_gamma_makes_the_start_state_more_negative(self) -> None:
        v_half = policy_evaluation(uniform_policy, gamma=0.5)[(0, 0)]
        v_nine = policy_evaluation(uniform_policy, gamma=0.9)[(0, 0)]
        v_ninetynine = policy_evaluation(uniform_policy, gamma=0.99)[(0, 0)]
        self.assertGreater(v_half, v_nine)
        self.assertGreater(v_nine, v_ninetynine)
        self.assertGreater(v_half, -2.0 / (1.0 - 0.5) - 1e-9)

    def test_better_policy_has_higher_value_everywhere(self) -> None:
        gamma = 0.99
        uniform = policy_evaluation(uniform_policy, gamma=gamma)
        down_right = policy_evaluation(down_right_policy, gamma=gamma)
        straight = policy_evaluation(straight_policy, gamma=gamma)
        for state in all_states():
            self.assertGreaterEqual(down_right[state], uniform[state] - 1e-9)
            self.assertGreaterEqual(straight[state], down_right[state] - 1e-9)

    def test_no_policy_beats_the_optimal_start_value(self) -> None:
        gamma = 0.99
        optimal = -(1.0 - gamma**6) / (1.0 - gamma)
        for policy in (uniform_policy, down_right_policy, straight_policy):
            self.assertLessEqual(policy_evaluation(policy, gamma=gamma)[(0, 0)], optimal + 1e-6)


if __name__ == "__main__":
    unittest.main()
