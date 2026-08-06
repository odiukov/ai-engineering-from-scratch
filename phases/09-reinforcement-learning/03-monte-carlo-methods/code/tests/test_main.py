"""Tests for first-visit Monte Carlo evaluation and epsilon-greedy MC control.

Covers phases/09-reinforcement-learning/03-monte-carlo-methods/docs/en.md.
The lesson's claims under test: the backward return recurrence equals the
explicit discounted sum, the incremental mean equals the batch average,
first-visit MC counts a state once per episode, and MC control drives the
greedy policy toward the DP optimum.
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
    mc_control,
    mc_policy_evaluation,
    reset,
    returns_from,
    rollout,
    states,
    step,
    uniform_policy,
)

OPTIMAL_DISCOUNTED_RETURN = -(1.0 - 0.99**6) / 0.01  # -5.851985..., the 6-step straight line


def straight_policy(state, _rng):
    """Deterministic optimal policy: down until the bottom row, then right."""
    row, _ = state
    return "down" if row < GRID - 1 else "right"


def exact_policy_evaluation(action_probs, gamma, tol=1e-12, max_iter=20000):
    """Iterative policy evaluation on the deterministic grid — the DP reference."""
    values = {s: 0.0 for s in states()}
    for _ in range(max_iter):
        delta = 0.0
        for state in states():
            if state == TERMINAL:
                continue
            backup = 0.0
            for action, prob in action_probs.items():
                s_next, reward, _ = step(state, action)
                backup += prob * (reward + gamma * values[s_next])
            delta = max(delta, abs(backup - values[state]))
            values[state] = backup
        if delta < tol:
            break
    return values


class TestReturnRecurrence(unittest.TestCase):
    def test_matches_the_explicit_discounted_sum(self) -> None:
        gamma = 0.9
        rewards = [-1.0, 2.0, -3.0, 4.0]
        traj = [((0, 0), "up", r) for r in rewards]
        got = returns_from(traj, gamma)
        for t in range(len(rewards)):
            expected = sum(gamma**k * rewards[t + k] for k in range(len(rewards) - t))
            self.assertAlmostEqual(got[t], expected, places=12)

    def test_gamma_zero_keeps_only_the_immediate_reward(self) -> None:
        traj = [((0, 0), "up", -1.0), ((0, 1), "up", 5.0), ((0, 2), "up", -2.0)]
        self.assertEqual(returns_from(traj, 0.0), [-1.0, 5.0, -2.0])

    def test_gamma_one_gives_plain_suffix_sums(self) -> None:
        traj = [((0, 0), "up", -1.0)] * 6
        self.assertEqual(returns_from(traj, 1.0), [-6.0, -5.0, -4.0, -3.0, -2.0, -1.0])

    def test_empty_trajectory_has_no_returns(self) -> None:
        self.assertEqual(returns_from([], 0.99), [])


class TestRollout(unittest.TestCase):
    def test_optimal_policy_terminates_in_six_steps_from_the_start(self) -> None:
        traj = rollout(straight_policy, random.Random(0))
        self.assertEqual(len(traj), 6)
        self.assertEqual(traj[0][0], reset())
        self.assertEqual(traj[-1][0], (3, 2))
        self.assertTrue(all(r == -1.0 for _, _, r in traj))

    def test_random_rollout_is_reproducible_under_a_seeded_rng(self) -> None:
        a = rollout(uniform_policy, random.Random(99))
        b = rollout(uniform_policy, random.Random(99))
        self.assertEqual(a, b)

    def test_step_cap_bounds_a_non_terminating_rollout(self) -> None:
        def bounce(_state, _rng):
            return "up"

        traj = rollout(bounce, random.Random(0), max_steps=17)
        self.assertEqual(len(traj), 17)


class TestFirstVisitEvaluation(unittest.TestCase):
    def test_deterministic_policy_recovers_the_exact_returns(self) -> None:
        gamma = 0.99
        values, counts = mc_policy_evaluation(straight_policy, 5, gamma=gamma, rng=random.Random(0))
        self.assertAlmostEqual(values[(0, 0)], OPTIMAL_DISCOUNTED_RETURN, places=9)
        self.assertAlmostEqual(values[(3, 2)], -1.0, places=12)
        self.assertEqual(counts[(0, 0)], 5)

    def test_a_state_is_counted_at_most_once_per_episode(self) -> None:
        episodes = 200
        _values, counts = mc_policy_evaluation(
            uniform_policy, episodes, gamma=0.99, rng=random.Random(3)
        )
        self.assertTrue(counts)
        for state, n in counts.items():
            self.assertLessEqual(n, episodes, f"{state} counted more than once per episode")
        self.assertEqual(counts[(0, 0)], episodes)

    def test_incremental_mean_equals_the_batch_average_of_first_visit_returns(self) -> None:
        episodes, gamma = 300, 0.99
        values, counts = mc_policy_evaluation(
            uniform_policy, episodes, gamma=gamma, rng=random.Random(11)
        )

        replay = random.Random(11)
        collected: dict[tuple[int, int], list[float]] = {}
        for _ in range(episodes):
            traj = rollout(uniform_policy, replay)
            returns = returns_from(traj, gamma)
            seen = set()
            for (state, _a, _r), G in zip(traj, returns):
                if state in seen:
                    continue
                seen.add(state)
                collected.setdefault(state, []).append(G)

        self.assertEqual(set(collected), set(counts))
        for state, samples in collected.items():
            self.assertEqual(len(samples), counts[state])
            self.assertAlmostEqual(values[state], sum(samples) / len(samples), places=9)

    def test_estimate_is_reproducible_under_a_seeded_rng(self) -> None:
        a, _ = mc_policy_evaluation(uniform_policy, 150, rng=random.Random(5))
        b, _ = mc_policy_evaluation(uniform_policy, 150, rng=random.Random(5))
        self.assertEqual(dict(a), dict(b))

    def test_estimate_converges_toward_the_dp_reference(self) -> None:
        gamma = 0.99
        reference = exact_policy_evaluation({a: 1.0 / len(ACTIONS) for a in ACTIONS}, gamma)
        self.assertAlmostEqual(reference[(0, 0)], -39.41, delta=0.01)

        coarse, _ = mc_policy_evaluation(uniform_policy, 200, gamma=gamma, rng=random.Random(2))
        fine, _ = mc_policy_evaluation(uniform_policy, 8000, gamma=gamma, rng=random.Random(2))
        coarse_err = abs(coarse[(0, 0)] - reference[(0, 0)])
        fine_err = abs(fine[(0, 0)] - reference[(0, 0)])
        self.assertLess(fine_err, coarse_err)
        self.assertLess(fine_err, 2.0)


class TestMonteCarloControl(unittest.TestCase):
    def test_greedy_policy_recovered_for_every_visited_state(self) -> None:
        _Q, greedy, log = mc_control(2000, epsilon=0.2, rng=random.Random(4))
        self.assertEqual(len(log), 2000)
        self.assertIn((0, 0), greedy)
        for state, action in greedy.items():
            self.assertIn(action, ACTIONS)
            self.assertNotEqual(state, TERMINAL)

    def test_control_improves_return_over_training(self) -> None:
        _Q, _greedy, log = mc_control(4000, epsilon=0.1, rng=random.Random(6))
        first = sum(log[:50]) / 50
        last = sum(log[-500:]) / 500
        self.assertGreater(last, first + 5.0)
        self.assertLess(last, OPTIMAL_DISCOUNTED_RETURN + 1.5)

    def test_greedy_policy_moves_toward_the_goal_from_the_start(self) -> None:
        _Q, greedy, _log = mc_control(6000, epsilon=0.1, rng=random.Random(6))
        self.assertIn(greedy[(0, 0)], ("down", "right"))
        self.assertEqual(greedy[(3, 2)], "right")
        self.assertEqual(greedy[(2, 3)], "down")

    def test_control_is_reproducible_under_a_seeded_rng(self) -> None:
        _q1, g1, l1 = mc_control(400, rng=random.Random(8))
        _q2, g2, l2 = mc_control(400, rng=random.Random(8))
        self.assertEqual(g1, g2)
        self.assertEqual(l1, l2)


if __name__ == "__main__":
    unittest.main()
