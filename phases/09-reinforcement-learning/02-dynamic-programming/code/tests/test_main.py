"""Tests for policy iteration and value iteration on the slippery 4x4 GridWorld.

Covers phases/09-reinforcement-learning/02-dynamic-programming/docs/en.md.
The properties checked are the ones the lesson rests its guarantees on: the
model is a proper distribution, the Bellman optimality operator is a
gamma-contraction in sup-norm, and both DP algorithms land on the same
fixed point with the greedy policy attaining it.
"""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from main import (  # noqa: E402
    ACTIONS,
    GRID,
    SLIP,
    TERMINAL,
    apply_move,
    greedy_from_V,
    perpendiculars,
    policy_evaluation,
    policy_iteration,
    q_value,
    states,
    transitions,
    value_iteration,
)

DETERMINISTIC_OPTIMAL_START = -(1.0 - 0.99**6) / 0.01  # -5.851985..., lesson 01's V*(0,0)


def bellman_optimality_operator(values, gamma):
    """One synchronous sweep of (T V)(s) = max_a sum P (r + gamma V(s'))."""
    out = {}
    for state in states():
        if state == TERMINAL:
            out[state] = 0.0
            continue
        out[state] = max(q_value(state, action, values, gamma) for action in ACTIONS)
    return out


def sup_norm(a, b):
    return max(abs(a[s] - b[s]) for s in states())


class TestTransitionModel(unittest.TestCase):
    def test_probabilities_sum_to_one_for_every_state_action(self) -> None:
        for state in states():
            for action in ACTIONS:
                total = sum(p for _, _, p in transitions(state, action))
                self.assertAlmostEqual(total, 1.0, places=12)

    def test_intended_direction_keeps_one_minus_slip_of_the_mass(self) -> None:
        outcomes = transitions((1, 1), "down")
        by_prob = {p for _, _, p in outcomes}
        self.assertIn(1.0 - SLIP, by_prob)
        self.assertEqual(sorted(p for _, _, p in outcomes), sorted([0.9, 0.05, 0.05]))

    def test_slip_directions_are_perpendicular_never_the_reverse(self) -> None:
        self.assertEqual(set(perpendiculars("up")), {"left", "right"})
        self.assertEqual(set(perpendiculars("right")), {"up", "down"})
        for action in ACTIONS:
            self.assertNotIn(action, perpendiculars(action))

    def test_terminal_is_absorbing_and_reward_free(self) -> None:
        for action in ACTIONS:
            self.assertEqual(transitions(TERMINAL, action), [(TERMINAL, 0.0, 1.0)])

    def test_moves_clamp_at_the_grid_border(self) -> None:
        self.assertEqual(apply_move((0, 0), "up"), (0, 0))
        self.assertEqual(apply_move((GRID - 1, GRID - 1), "down"), (GRID - 1, GRID - 1))
        self.assertEqual(apply_move((2, 2), "right"), (2, 3))


class TestBellmanOperatorIsAContraction(unittest.TestCase):
    def test_sup_norm_distance_shrinks_by_at_most_gamma(self) -> None:
        gamma = 0.9
        v1 = {s: 0.0 for s in states()}
        v2 = {s: (-7.0 if (s[0] + s[1]) % 2 else 11.0) for s in states()}
        v2[TERMINAL] = 0.0
        before = sup_norm(v1, v2)
        for _ in range(5):
            v1, v2 = bellman_optimality_operator(v1, gamma), bellman_optimality_operator(v2, gamma)
            after = sup_norm(v1, v2)
            self.assertLessEqual(after, gamma * before + 1e-12)
            before = after

    def test_value_iteration_result_is_a_fixed_point_of_the_operator(self) -> None:
        for gamma in (0.9, 0.99):
            values, _, _ = value_iteration(gamma=gamma)
            self.assertLess(sup_norm(values, bellman_optimality_operator(values, gamma)), 1e-4)

    def test_iteration_from_a_wildly_wrong_start_reaches_the_same_fixed_point(self) -> None:
        gamma = 0.9
        reference, _, _ = value_iteration(gamma=gamma)
        values = {s: 100.0 for s in states()}
        values[TERMINAL] = 0.0
        for _ in range(600):
            values = bellman_optimality_operator(values, gamma)
        self.assertLess(sup_norm(values, reference), 1e-4)


class TestValueIterationAndPolicyIteration(unittest.TestCase):
    def test_both_algorithms_agree_on_v_star_and_the_policy(self) -> None:
        v_vi, pi_vi, _ = value_iteration(gamma=0.99)
        v_pi, pi_pi, _ = policy_iteration(gamma=0.99)
        self.assertLess(sup_norm(v_vi, v_pi), 1e-6)
        self.assertEqual(pi_vi, pi_pi)

    def test_policy_iteration_needs_fewer_outer_steps_than_vi_needs_sweeps(self) -> None:
        _, _, sweeps_vi = value_iteration(gamma=0.99)
        _, _, iters_pi = policy_iteration(gamma=0.99)
        self.assertLess(iters_pi, sweeps_vi)
        self.assertLessEqual(iters_pi, 20)

    def test_greedy_policy_evaluates_back_to_v_star(self) -> None:
        gamma = 0.99
        v_star, policy, _ = value_iteration(gamma=gamma)
        v_greedy = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma=gamma)
        self.assertLess(sup_norm(v_star, v_greedy), 1e-4)

    def test_v_star_dominates_every_fixed_policy(self) -> None:
        gamma = 0.99
        v_star, _, _ = value_iteration(gamma=gamma)
        for action in ACTIONS:
            v_fixed = policy_evaluation(lambda _s, a=action: {a: 1.0}, gamma=gamma)
            for state in states():
                self.assertGreaterEqual(v_star[state], v_fixed[state] - 1e-6)

    def test_terminal_value_is_exactly_zero(self) -> None:
        v_star, _, _ = value_iteration(gamma=0.99)
        self.assertEqual(v_star[TERMINAL], 0.0)

    def test_slipping_costs_value_relative_to_the_deterministic_grid(self) -> None:
        v_star, _, _ = value_iteration(gamma=0.99)
        self.assertLess(v_star[(0, 0)], DETERMINISTIC_OPTIMAL_START)
        self.assertAlmostEqual(v_star[(0, 0)], -6.43, delta=0.01)

    def test_smaller_gamma_yields_a_less_negative_start_value(self) -> None:
        v_nine, _, _ = value_iteration(gamma=0.9)
        v_ninetynine, _, _ = value_iteration(gamma=0.99)
        self.assertGreater(v_nine[(0, 0)], v_ninetynine[(0, 0)])

    def test_greedy_action_is_an_argmax_of_the_q_values(self) -> None:
        gamma = 0.99
        v_star, policy, _ = value_iteration(gamma=gamma)
        for state in states():
            if state == TERMINAL:
                continue
            best = max(q_value(state, a, v_star, gamma) for a in ACTIONS)
            self.assertAlmostEqual(q_value(state, policy[state], v_star, gamma), best, places=9)

    def test_greedy_from_v_never_walks_away_from_the_goal(self) -> None:
        v_star, policy, _ = value_iteration(gamma=0.99)
        self.assertEqual(set(greedy_from_V(v_star).items()), set(policy.items()))
        for state in states():
            if state == TERMINAL:
                continue
            self.assertIn(policy[state], ("down", "right"))


if __name__ == "__main__":
    unittest.main()
