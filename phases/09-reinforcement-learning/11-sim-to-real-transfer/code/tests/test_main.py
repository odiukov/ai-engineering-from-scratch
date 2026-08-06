"""Tests for the domain-randomization demo: the slip parameter, DR vs fixed training.

Covers docs/en.md "Build It" steps 1-4. The 5x5 grid's best possible return at
slip = 0 is -8 (eight moves from (0, 0) to (4, 4), one penalty each), which pins
the in-distribution assertions to an exact number instead of a threshold.
"""

from __future__ import annotations

import os
import random
import sys
import unittest
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from main import (  # noqa: E402
    ACTIONS,
    GRID,
    TERMINAL,
    default_q,
    epsilon_greedy,
    evaluate,
    step,
    train_dr,
    train_fixed,
)

OPTIMAL_NO_SLIP_RETURN = -float((GRID - 1) * 2)


class TestSlipDynamics(unittest.TestCase):
    def test_zero_slip_executes_the_commanded_action(self):
        rng = random.Random(0)
        for _ in range(20):
            s_next, reward, done = step((2, 2), "up", 0.0, rng)
            self.assertEqual(s_next, (1, 2))
            self.assertEqual(reward, -1.0)
            self.assertFalse(done)

    def test_certain_slip_diverts_a_vertical_action_sideways(self):
        rng = random.Random(1)
        seen = set()
        for _ in range(200):
            s_next, _, _ = step((2, 2), "up", 1.0, rng)
            seen.add(s_next)
        self.assertEqual(seen, {(2, 1), (2, 3)})

    def test_certain_slip_diverts_a_horizontal_action_vertically(self):
        rng = random.Random(2)
        seen = set()
        for _ in range(200):
            s_next, _, _ = step((2, 2), "left", 1.0, rng)
            seen.add(s_next)
        self.assertEqual(seen, {(1, 2), (3, 2)})

    def test_intermediate_slip_mixes_both_outcomes(self):
        rng = random.Random(3)
        commanded = 0
        trials = 2000
        for _ in range(trials):
            s_next, _, _ = step((2, 2), "up", 0.3, rng)
            if s_next == (1, 2):
                commanded += 1
        self.assertAlmostEqual(commanded / trials, 0.7, delta=0.04)

    def test_walls_clamp_even_after_a_slip(self):
        rng = random.Random(4)
        for _ in range(50):
            s_next, _, _ = step((0, 0), "up", 1.0, rng)
            self.assertIn(s_next, {(0, 0), (0, 1)})

    def test_terminal_state_is_absorbing_and_free(self):
        rng = random.Random(5)
        for action in ACTIONS:
            s_next, reward, done = step(TERMINAL, action, 0.9, rng)
            self.assertEqual((s_next, reward, done), (TERMINAL, 0.0, True))

    def test_stepping_onto_the_goal_flags_done(self):
        rng = random.Random(6)
        s_next, reward, done = step((GRID - 2, GRID - 1), "down", 0.0, rng)
        self.assertEqual(s_next, TERMINAL)
        self.assertEqual(reward, -1.0)
        self.assertTrue(done)


class TestEpsilonGreedy(unittest.TestCase):
    def test_zero_epsilon_takes_the_argmax(self):
        Q = defaultdict(default_q)
        Q[(0, 0)]["right"] = 1.0
        rng = random.Random(0)
        for _ in range(20):
            self.assertEqual(epsilon_greedy(Q, (0, 0), rng, 0.0), "right")

    def test_full_epsilon_covers_the_action_set(self):
        Q = defaultdict(default_q)
        rng = random.Random(1)
        seen = {epsilon_greedy(Q, (0, 0), rng, 1.0) for _ in range(200)}
        self.assertEqual(seen, set(ACTIONS))


class TestEvaluation(unittest.TestCase):
    def test_untrained_greedy_policy_stalls_against_the_top_wall(self):
        empty = defaultdict(default_q)
        self.assertEqual(evaluate(empty, 0.0, episodes=3, rng=random.Random(0)), -100.0)

    def test_evaluation_is_reproducible_for_a_given_seed(self):
        Q = train_fixed(0.0, episodes=200, rng=random.Random(0))
        first = evaluate(Q, 0.3, episodes=40, rng=random.Random(8))
        second = evaluate(Q, 0.3, episodes=40, rng=random.Random(8))
        self.assertEqual(first, second)


class TestDomainRandomization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.q_fixed = train_fixed(0.0, rng=random.Random(1))
        cls.q_dr = train_dr(0.0, 0.3, rng=random.Random(1))

    def _score(self, Q, slip):
        return evaluate(Q, slip, episodes=200, rng=random.Random(42))

    def test_both_policies_are_optimal_in_the_frictionless_condition(self):
        self.assertEqual(self._score(self.q_fixed, 0.0), OPTIMAL_NO_SLIP_RETURN)
        self.assertEqual(self._score(self.q_dr, 0.0), OPTIMAL_NO_SLIP_RETURN)

    def test_no_policy_can_beat_the_shortest_path(self):
        for slip in (0.0, 0.2, 0.5):
            self.assertLessEqual(self._score(self.q_dr, slip), OPTIMAL_NO_SLIP_RETURN)
            self.assertLessEqual(self._score(self.q_fixed, slip), OPTIMAL_NO_SLIP_RETURN)

    def test_return_degrades_as_the_real_slip_rises(self):
        scores = [self._score(self.q_dr, slip) for slip in (0.0, 0.3, 0.7)]
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[1], scores[2])

    def test_dr_policy_beats_the_fixed_slip_policy_out_of_support(self):
        self.assertGreater(self._score(self.q_dr, 0.5), self._score(self.q_fixed, 0.5))

    def test_dr_policy_matches_the_fixed_policy_inside_the_easy_condition(self):
        """DR is not a free win: at slip = 0 the narrow policy is already optimal."""
        self.assertEqual(self._score(self.q_dr, 0.0), self._score(self.q_fixed, 0.0))

    def test_training_visits_only_reachable_grid_cells(self):
        for state in self.q_dr:
            r, c = state
            self.assertTrue(0 <= r < GRID)
            self.assertTrue(0 <= c < GRID)

    def test_widening_the_range_makes_the_policy_more_conservative_at_high_slip(self):
        wide = train_dr(0.0, 0.6, rng=random.Random(1))
        self.assertGreaterEqual(self._score(wide, 0.7), self._score(self.q_fixed, 0.7))


if __name__ == "__main__":
    unittest.main()
