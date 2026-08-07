"""Tests for the domain-randomization demo: slip dynamics, the cliff, DR vs fixed.

Covers docs/en.md "Build It" steps 1-4. Two numbers anchor the assertions and
both follow from the geometry rather than from a training run: the shortest safe
route from S=(3, 0) to G=(3, 5) is 7 moves (up, five right, down), so the best
possible return at slip = 0 is -7; and an episode that never terminates is
capped at MAX_STEPS moves of -1, i.e. -100.

Everything about the *comparison* between policies is asserted as a property --
who beats whom, where the route runs, which direction the curve moves -- never
as a transcript of the numbers main.py happens to print.
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
    CLIFF,
    CLIFF_COST,
    COLS,
    GOAL,
    MAX_STEPS,
    ROWS,
    START,
    default_q,
    epsilon_greedy,
    evaluate,
    greedy_path,
    step,
    train_dr,
    train_fixed,
)

# up + (COLS - 1) rights + down: the cliff forces a one-row detour.
OPTIMAL_NO_SLIP_RETURN = -float(COLS + 1)
STALLED_RETURN = -float(MAX_STEPS)
# The row that runs directly above the cliff -- the risky edge.
EDGE_ROW = {(ROWS - 2, c) for c in range(1, COLS - 1)}


class TestGridLayout(unittest.TestCase):
    def test_the_cliff_separates_the_start_from_the_goal(self):
        self.assertEqual(CLIFF, {(ROWS - 1, c) for c in range(1, COLS - 1)})
        self.assertNotIn(START, CLIFF)
        self.assertNotIn(GOAL, CLIFF)

    def test_the_cliff_costs_much_more_than_a_detour(self):
        """One fall must outweigh walking the long way round, or there is no dilemma."""
        detour_cost = 4.0  # the widest sensible detour is a handful of extra steps
        self.assertLess(CLIFF_COST, -detour_cost)


class TestSlipDynamics(unittest.TestCase):
    def test_zero_slip_executes_the_commanded_action(self):
        rng = random.Random(0)
        for _ in range(20):
            s_next, reward, done = step((1, 2), "up", 0.0, rng)
            self.assertEqual(s_next, (0, 2))
            self.assertEqual(reward, -1.0)
            self.assertFalse(done)

    def test_certain_slip_diverts_a_vertical_action_sideways(self):
        rng = random.Random(1)
        seen = {step((1, 2), "up", 1.0, rng)[0] for _ in range(200)}
        self.assertEqual(seen, {(1, 1), (1, 3)})

    def test_certain_slip_diverts_a_horizontal_action_vertically(self):
        rng = random.Random(2)
        seen = {step((1, 2), "left", 1.0, rng)[0] for _ in range(200)}
        self.assertEqual(seen, {(0, 2), (2, 2)})

    def test_intermediate_slip_mixes_both_outcomes(self):
        rng = random.Random(3)
        commanded = 0
        trials = 2000
        for _ in range(trials):
            if step((1, 2), "up", 0.3, rng)[0] == (0, 2):
                commanded += 1
        self.assertAlmostEqual(commanded / trials, 0.7, delta=0.04)

    def test_the_slip_die_is_rolled_even_at_zero_slip(self):
        """Otherwise the rng stream depends on slip and the two policies are
        no longer evaluated on the same rollouts."""
        control = random.Random(9)
        control.random()  # exactly one die
        expected = control.random()

        rng = random.Random(9)
        step((1, 1), "right", 0.0, rng)
        self.assertEqual(rng.random(), expected)

    def test_walls_clamp_even_after_a_slip(self):
        rng = random.Random(4)
        for _ in range(50):
            self.assertIn(step((0, 0), "up", 1.0, rng)[0], {(0, 0), (0, 1)})

    def test_falling_into_the_cliff_costs_a_lot_and_teleports_home(self):
        """The fall does not end the episode -- the agent restarts carrying the debt."""
        rng = random.Random(5)
        s_next, reward, done = step((ROWS - 2, 1), "down", 0.0, rng)
        self.assertEqual(s_next, START)
        self.assertEqual(reward, CLIFF_COST)
        self.assertFalse(done)

    def test_the_agent_never_ends_a_step_inside_the_cliff(self):
        rng = random.Random(6)
        for _ in range(500):
            state = (ROWS - 2, rng.randrange(COLS))
            s_next, _, _ = step(state, rng.choice(ACTIONS), 0.5, rng)
            self.assertNotIn(s_next, CLIFF)

    def test_stepping_onto_the_goal_flags_done(self):
        rng = random.Random(7)
        s_next, reward, done = step((ROWS - 2, COLS - 1), "down", 0.0, rng)
        self.assertEqual(s_next, GOAL)
        self.assertEqual(reward, -1.0)
        self.assertTrue(done)


class TestEpsilonGreedy(unittest.TestCase):
    def test_zero_epsilon_takes_the_argmax(self):
        Q = defaultdict(default_q)
        Q[START]["right"] = 1.0
        rng = random.Random(0)
        for _ in range(20):
            self.assertEqual(epsilon_greedy(Q, START, rng, 0.0), "right")

    def test_full_epsilon_covers_the_action_set(self):
        Q = defaultdict(default_q)
        rng = random.Random(1)
        seen = {epsilon_greedy(Q, START, rng, 1.0) for _ in range(200)}
        self.assertEqual(seen, set(ACTIONS))


class TestEvaluation(unittest.TestCase):
    def test_untrained_greedy_policy_stalls_against_the_left_wall(self):
        empty = defaultdict(default_q)
        self.assertEqual(
            evaluate(empty, 0.0, episodes=3, rng=random.Random(0)), STALLED_RETURN
        )

    def test_evaluation_is_reproducible_for_a_given_seed(self):
        Q = train_fixed(0.0, episodes=200, rng=random.Random(0))
        first = evaluate(Q, 0.3, episodes=40, rng=random.Random(8))
        second = evaluate(Q, 0.3, episodes=40, rng=random.Random(8))
        self.assertEqual(first, second)

    def test_a_fixed_slip_is_only_one_degenerate_randomization_range(self):
        """train_fixed must be train_dr with a zero-width range, same rng draws."""
        a = train_fixed(0.2, episodes=150, rng=random.Random(3))
        b = train_dr(0.2, 0.2, episodes=150, rng=random.Random(3))
        self.assertEqual(a, b)


class TestDomainRandomization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.q_fixed = train_fixed(0.0, rng=random.Random(1))
        cls.q_dr = train_dr(0.0, 0.3, rng=random.Random(1))

    def _score(self, Q, slip):
        return evaluate(Q, slip, episodes=200, rng=random.Random(42))

    def test_the_fixed_slip_policy_is_optimal_in_the_condition_it_trained_on(self):
        self.assertEqual(self._score(self.q_fixed, 0.0), OPTIMAL_NO_SLIP_RETURN)

    def test_no_policy_can_beat_the_shortest_path(self):
        for slip in (0.0, 0.2, 0.5):
            self.assertLessEqual(self._score(self.q_dr, slip), OPTIMAL_NO_SLIP_RETURN)
            self.assertLessEqual(self._score(self.q_fixed, slip), OPTIMAL_NO_SLIP_RETURN)

    def test_the_fixed_slip_policy_walks_the_cliff_edge(self):
        """Nothing punished the risk during training, so it took the shortest route."""
        path = greedy_path(self.q_fixed)
        self.assertEqual(path[-1], GOAL)
        self.assertTrue(EDGE_ROW <= set(path))

    def test_the_dr_policy_buys_a_margin_away_from_the_cliff(self):
        """It still finishes, but it pays extra steps to stay off the edge row."""
        dr_path = greedy_path(self.q_dr)
        fixed_path = greedy_path(self.q_fixed)
        self.assertEqual(dr_path[-1], GOAL)
        self.assertLess(
            len(EDGE_ROW & set(dr_path)), len(EDGE_ROW & set(fixed_path))
        )
        self.assertGreater(len(dr_path) - 1, -OPTIMAL_NO_SLIP_RETURN)

    def test_return_degrades_as_the_real_slip_rises(self):
        scores = [self._score(self.q_dr, slip) for slip in (0.0, 0.3, 0.7)]
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[1], scores[2])

    def test_dr_policy_beats_the_fixed_slip_policy_out_of_support(self):
        """The headline claim: outside the training range the gap is not marginal.

        Across seeds the ratio lands between 2.7x and 5.5x, so the assertion is
        pinned at the low end of that band rather than at the seed we ship.
        """
        for slip in (0.5, 0.7):
            dr = self._score(self.q_dr, slip)
            fixed = self._score(self.q_fixed, slip)
            self.assertGreater(dr, fixed)
            self.assertGreater(dr, 2 * fixed)  # returns are negative: 2x the gap

    def test_dr_policy_also_wins_everywhere_inside_support_except_at_zero_slip(self):
        for slip in (0.1, 0.2, 0.3):
            self.assertGreater(self._score(self.q_dr, slip), self._score(self.q_fixed, slip))

    def test_dr_pays_for_the_margin_when_the_motors_are_perfect(self):
        """DR is not a free win: the detour costs real steps at slip = 0.

        This is the honest price of robustness, and it is small -- a couple of
        steps -- next to the collapse it prevents.
        """
        dr = self._score(self.q_dr, 0.0)
        fixed = self._score(self.q_fixed, 0.0)
        self.assertLess(dr, fixed)
        self.assertGreaterEqual(dr, fixed - 4.0)

    def test_training_visits_only_reachable_grid_cells(self):
        for state in self.q_dr:
            r, c = state
            self.assertTrue(0 <= r < ROWS)
            self.assertTrue(0 <= c < COLS)
            self.assertNotIn(state, CLIFF)

    def test_over_randomizing_trades_home_performance_for_the_extremes(self):
        """The mirror-image pitfall: U[0, 0.9] is safest at slip 0.7 and useless at 0."""
        over = train_dr(0.0, 0.9, rng=random.Random(1))
        self.assertGreater(self._score(over, 0.7), self._score(self.q_fixed, 0.7))
        self.assertLess(self._score(over, 0.0), self._score(self.q_dr, 0.0))


if __name__ == "__main__":
    unittest.main()
