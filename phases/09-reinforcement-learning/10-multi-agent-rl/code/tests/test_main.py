"""Tests for the cooperative 2-agent GridWorld: shared reward, independent vs joint Q.

Covers docs/en.md "Build It" steps 1-3. The optimal joint return on this task is
+3: the far agent needs 8 moves to reach (4, 4), so 7 penalised steps precede the
terminal +10. That bound is used as an upper limit no learner may exceed.
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
    GOAL,
    GRID,
    block_mean,
    default_q,
    epsilon_greedy,
    evaluate_ind,
    evaluate_joint,
    independent_q,
    joint_q_learning,
    move,
    reset,
    step,
)

OPTIMAL_RETURN = 3.0


class TestEnvironment(unittest.TestCase):
    def test_move_clamps_at_every_wall(self):
        self.assertEqual(move((0, 0), "up"), (0, 0))
        self.assertEqual(move((0, 0), "left"), (0, 0))
        self.assertEqual(move((GRID - 1, GRID - 1), "down"), (GRID - 1, GRID - 1))
        self.assertEqual(move((GRID - 1, GRID - 1), "right"), (GRID - 1, GRID - 1))

    def test_move_is_a_single_cell_step_inside_the_grid(self):
        self.assertEqual(move((2, 2), "up"), (1, 2))
        self.assertEqual(move((2, 2), "down"), (3, 2))
        self.assertEqual(move((2, 2), "left"), (2, 1))
        self.assertEqual(move((2, 2), "right"), (2, 3))

    def test_agents_start_in_different_corners(self):
        a1, a2 = reset()
        self.assertEqual(a1, (0, 0))
        self.assertEqual(a2, (GRID - 1, 0))
        self.assertNotEqual(a1, a2)

    def test_one_agent_on_the_goal_is_not_enough(self):
        state = ((GRID - 2, GRID - 1), (0, 0))
        (new1, new2), reward, done = step(state, ("down", "up"))
        self.assertEqual(new1, GOAL)
        self.assertNotEqual(new2, GOAL)
        self.assertEqual(reward, -1.0)
        self.assertFalse(done)

    def test_both_agents_arriving_together_pays_the_bonus(self):
        state = ((GRID - 2, GRID - 1), (GRID - 1, GRID - 2))
        (new1, new2), reward, done = step(state, ("down", "right"))
        self.assertEqual((new1, new2), (GOAL, GOAL))
        self.assertEqual(reward, 10.0)
        self.assertTrue(done)

    def test_reward_is_shared_so_both_agents_see_the_same_scalar(self):
        _, reward_a, _ = step(((1, 1), (3, 1)), ("down", "up"))
        _, reward_b, _ = step(((1, 1), (3, 1)), ("right", "right"))
        self.assertEqual(reward_a, reward_b)


class TestEpsilonGreedy(unittest.TestCase):
    def test_zero_epsilon_always_takes_the_argmax(self):
        table = defaultdict(default_q)
        table[(0, 0)]["left"] = 5.0
        rng = random.Random(0)
        for _ in range(20):
            self.assertEqual(epsilon_greedy(table, (0, 0), rng, 0.0), "left")

    def test_epsilon_one_explores_the_whole_action_set(self):
        table = defaultdict(default_q)
        table[(0, 0)]["left"] = 5.0
        rng = random.Random(1)
        seen = {epsilon_greedy(table, (0, 0), rng, 1.0) for _ in range(200)}
        self.assertEqual(seen, set(ACTIONS))

    def test_default_q_is_flat_over_the_action_set(self):
        self.assertEqual(set(default_q()), set(ACTIONS))
        self.assertEqual(set(default_q().values()), {0.0})


class TestIndependentQ(unittest.TestCase):
    def test_each_agent_keeps_its_own_table(self):
        Q1, Q2, log = independent_q(episodes=30, rng=random.Random(0))
        self.assertIsNot(Q1, Q2)
        self.assertEqual(len(log), 30)

    def test_tables_are_keyed_on_the_joint_state(self):
        Q1, _, _ = independent_q(episodes=20, rng=random.Random(0))
        for key in Q1:
            self.assertEqual(len(key), 2)
            (r1, c1), (r2, c2) = key
            for coord in (r1, c1, r2, c2):
                self.assertTrue(0 <= coord < GRID)

    def test_untrained_greedy_agents_never_reach_the_goal(self):
        empty1, empty2 = defaultdict(default_q), defaultdict(default_q)
        self.assertEqual(evaluate_ind(empty1, empty2, episodes=3, rng=random.Random(0)), -100.0)

    def test_training_improves_the_shared_return(self):
        _, _, log = independent_q(episodes=1200, rng=random.Random(1))
        early = sum(log[:150]) / 150.0
        late = sum(log[-150:]) / 150.0
        self.assertGreater(late, early)

    def test_learned_policy_beats_zero_return_without_exceeding_the_optimum(self):
        Q1, Q2, _ = independent_q(episodes=1500, rng=random.Random(1))
        got = evaluate_ind(Q1, Q2, episodes=50, rng=random.Random(9))
        self.assertGreater(got, 0.0)
        self.assertLessEqual(got, OPTIMAL_RETURN)


class TestJointQ(unittest.TestCase):
    def test_joint_action_space_is_the_square_of_one_agent_s(self):
        Q, _ = joint_q_learning(episodes=5, rng=random.Random(0))
        any_state = next(iter(Q))
        self.assertEqual(len(Q[any_state]), len(ACTIONS) ** 2)
        for key in Q[any_state]:
            self.assertEqual(len(key), 2)
            self.assertTrue(set(key) <= set(ACTIONS))

    def test_training_improves_the_shared_return(self):
        _, log = joint_q_learning(episodes=1200, rng=random.Random(1))
        early = sum(log[:150]) / 150.0
        late = sum(log[-150:]) / 150.0
        self.assertGreater(late, early)

    def test_learned_policy_beats_zero_return_without_exceeding_the_optimum(self):
        Q, _ = joint_q_learning(episodes=1500, rng=random.Random(1))
        got = evaluate_joint(Q, episodes=50, rng=random.Random(9))
        self.assertGreater(got, 0.0)
        self.assertLessEqual(got, OPTIMAL_RETURN)

    def test_evaluation_is_reproducible_for_a_given_seed(self):
        Q, _ = joint_q_learning(episodes=200, rng=random.Random(4))
        first = evaluate_joint(Q, episodes=20, rng=random.Random(77))
        second = evaluate_joint(Q, episodes=20, rng=random.Random(77))
        self.assertEqual(first, second)


class TestReporting(unittest.TestCase):
    def test_block_mean_averages_whole_blocks_and_drops_the_remainder(self):
        self.assertEqual(block_mean([1.0, 2.0, 3.0, 4.0, 5.0], 2), [1.5, 3.5])
        self.assertEqual(block_mean([2.0, 4.0, 6.0], 3), [4.0])


if __name__ == "__main__":
    unittest.main()
