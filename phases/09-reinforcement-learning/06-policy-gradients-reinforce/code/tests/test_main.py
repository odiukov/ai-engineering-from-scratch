"""Tests for REINFORCE: softmax policy, score function, return-weighted update.

Covers phases/09-reinforcement-learning/06-policy-gradients-reinforce/docs/en.md.
Both analytic gradients the lesson derives by hand are checked against central
finite differences: grad log pi = onehot(a) - pi, and the full REINFORCE step as
the gradient of the return-weighted log-likelihood surrogate.
"""

from __future__ import annotations

import math
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from main import (  # noqa: E402
    ACTIONS,
    GRID,
    N_ACTIONS,
    N_FEAT,
    TERMINAL,
    features,
    grad_log_pi,
    greedy_policy,
    init_theta,
    logits,
    reinforce,
    reinforce_step,
    reset,
    returns_to_go,
    rollout,
    sample,
    softmax,
    step,
)

OPTIMAL_DISCOUNTED_RETURN = -(1.0 - 0.99**6) / 0.01  # -5.851985...


def copy_theta(theta):
    return [row[:] for row in theta]


def log_pi(theta, x, a):
    return math.log(softmax(logits(theta, x))[a])


def surrogate(theta, traj, advantages):
    """sum_t A_t * log pi_theta(a_t | s_t) — the objective REINFORCE ascends."""
    return sum(adv * log_pi(theta, x, a) for (x, a, _r, _p), adv in zip(traj, advantages))


def numerical_theta_grad(fn, theta, eps=1e-6):
    grad = [[0.0] * N_FEAT for _ in range(N_ACTIONS)]
    for i in range(N_ACTIONS):
        for j in range(N_FEAT):
            original = theta[i][j]
            theta[i][j] = original + eps
            plus = fn(theta)
            theta[i][j] = original - eps
            minus = fn(theta)
            theta[i][j] = original
            grad[i][j] = (plus - minus) / (2 * eps)
    return grad


class TestEnvironmentAndFeatures(unittest.TestCase):
    def test_actions_are_addressed_by_index(self) -> None:
        self.assertEqual(step((0, 0), ACTIONS.index("down"))[0], (1, 0))
        self.assertEqual(step((0, 0), ACTIONS.index("up"))[0], (0, 0))
        self.assertEqual(step((3, 2), ACTIONS.index("right")), (TERMINAL, -1.0, True))

    def test_terminal_state_is_absorbing_and_reward_free(self) -> None:
        for a in range(N_ACTIONS):
            self.assertEqual(step(TERMINAL, a), (TERMINAL, 0.0, True))

    def test_features_are_one_hot(self) -> None:
        x = features((1, 3))
        self.assertEqual(len(x), N_FEAT)
        self.assertEqual(sum(x), 1.0)
        self.assertEqual(x[1 * GRID + 3], 1.0)


class TestSoftmaxAndSampling(unittest.TestCase):
    def test_probabilities_are_normalized_and_positive(self) -> None:
        probs = softmax([2.0, -1.0, 0.0, 5.0])
        self.assertAlmostEqual(sum(probs), 1.0, places=12)
        self.assertTrue(all(p > 0.0 for p in probs))
        self.assertEqual(max(range(4), key=lambda i: probs[i]), 3)

    def test_softmax_is_invariant_to_a_constant_logit_shift(self) -> None:
        base = softmax([0.3, -1.2, 4.0, 0.9])
        shifted = softmax([z + 1000.0 for z in [0.3, -1.2, 4.0, 0.9]])
        for a, b in zip(base, shifted):
            self.assertAlmostEqual(a, b, places=12)

    def test_equal_logits_give_a_uniform_policy(self) -> None:
        for p in softmax([1.7] * N_ACTIONS):
            self.assertAlmostEqual(p, 1.0 / N_ACTIONS, places=12)

    def test_degenerate_distribution_is_sampled_deterministically(self) -> None:
        rng = random.Random(0)
        for _ in range(100):
            self.assertEqual(sample([0.0, 1.0, 0.0, 0.0], rng), 1)

    def test_sample_frequencies_track_the_distribution(self) -> None:
        probs = [0.1, 0.2, 0.3, 0.4]
        rng = random.Random(3)
        counts = [0] * N_ACTIONS
        draws = 40000
        for _ in range(draws):
            counts[sample(probs, rng)] += 1
        for observed, expected in zip(counts, probs):
            self.assertAlmostEqual(observed / draws, expected, delta=0.01)


class TestReturnsToGo(unittest.TestCase):
    def test_matches_the_explicit_discounted_suffix_sum(self) -> None:
        gamma = 0.9
        rewards = [-1.0, -1.0, 3.0, -1.0, 2.0]
        traj = [(features((0, 0)), 0, r, [0.25] * 4) for r in rewards]
        got = returns_to_go(traj, gamma)
        for t in range(len(rewards)):
            expected = sum(gamma**k * rewards[t + k] for k in range(len(rewards) - t))
            self.assertAlmostEqual(got[t], expected, places=12)

    def test_gamma_one_gives_plain_suffix_sums(self) -> None:
        traj = [(features((0, 0)), 0, -1.0, [0.25] * 4)] * 4
        self.assertEqual(returns_to_go(traj, 1.0), [-4.0, -3.0, -2.0, -1.0])

    def test_six_step_episode_at_gamma_099_matches_the_lesson_optimum(self) -> None:
        traj = [(features((0, 0)), 0, -1.0, [0.25] * 4)] * 6
        self.assertAlmostEqual(returns_to_go(traj, 0.99)[0], OPTIMAL_DISCOUNTED_RETURN, places=9)


class TestScoreFunction(unittest.TestCase):
    def test_grad_log_pi_sums_to_zero(self) -> None:
        for probs in ([0.25] * 4, [0.7, 0.1, 0.15, 0.05], [0.99, 0.005, 0.004, 0.001]):
            for a in range(N_ACTIONS):
                self.assertAlmostEqual(sum(grad_log_pi(probs, a)), 0.0, places=12)

    def test_score_function_has_zero_mean_under_the_policy(self) -> None:
        probs = softmax([0.4, -2.1, 1.3, 0.0])
        for i in range(N_ACTIONS):
            expected = sum(probs[a] * grad_log_pi(probs, a)[i] for a in range(N_ACTIONS))
            self.assertAlmostEqual(expected, 0.0, places=12)

    def test_taken_action_gets_positive_score_others_negative(self) -> None:
        probs = softmax([0.4, -2.1, 1.3, 0.0])
        grads = grad_log_pi(probs, 2)
        self.assertGreater(grads[2], 0.0)
        self.assertTrue(all(g < 0.0 for i, g in enumerate(grads) if i != 2))

    def test_grad_log_pi_matches_central_differences_in_theta(self) -> None:
        theta = init_theta(random.Random(4))
        for state, a in (((0, 0), 1), ((2, 3), 3), ((1, 1), 0)):
            x = features(state)
            numeric = numerical_theta_grad(lambda t, x=x, a=a: log_pi(t, x, a), theta)
            probs = softmax(logits(theta, x))
            analytic = grad_log_pi(probs, a)
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    self.assertAlmostEqual(numeric[i][j], analytic[i] * x[j], delta=1e-6)


class TestReinforceStepIsAGradientAscentStep(unittest.TestCase):
    def _fixture(self, seed=1, gamma=0.99):
        theta = init_theta(random.Random(seed))
        traj = rollout(theta, random.Random(seed + 100), max_steps=25)
        returns = returns_to_go(traj, gamma)
        return theta, traj, returns

    def test_update_matches_central_differences_without_a_baseline(self) -> None:
        theta, traj, returns = self._fixture(seed=1)
        numeric = numerical_theta_grad(lambda t: surrogate(t, traj, returns), copy_theta(theta))
        lr = 1.0
        updated = copy_theta(theta)
        reinforce_step(updated, traj, returns, lr)
        for i in range(N_ACTIONS):
            for j in range(N_FEAT):
                analytic = (updated[i][j] - theta[i][j]) / lr
                self.assertAlmostEqual(analytic, numeric[i][j], delta=1e-5)

    def test_update_matches_central_differences_with_a_baseline(self) -> None:
        theta, traj, returns = self._fixture(seed=2)
        baseline = sum(returns) / len(returns)
        advantages = [G - baseline for G in returns]
        numeric = numerical_theta_grad(lambda t: surrogate(t, traj, advantages), copy_theta(theta))
        lr = 1.0
        updated = copy_theta(theta)
        reinforce_step(updated, traj, returns, lr, baseline)
        for i in range(N_ACTIONS):
            for j in range(N_FEAT):
                analytic = (updated[i][j] - theta[i][j]) / lr
                self.assertAlmostEqual(analytic, numeric[i][j], delta=1e-5)

    def test_zero_advantage_leaves_theta_untouched(self) -> None:
        theta, traj, returns = self._fixture(seed=3)
        before = copy_theta(theta)
        reinforce_step(theta, traj, [0.0] * len(returns), 0.5)
        self.assertEqual(theta, before)

    def test_update_only_touches_features_that_were_visited(self) -> None:
        theta = init_theta(random.Random(5))
        before = copy_theta(theta)
        x = features((1, 2))
        traj = [(x, 0, -1.0, softmax(logits(theta, x)))]
        reinforce_step(theta, traj, [-1.0], 0.5)
        touched = {j for i in range(N_ACTIONS) for j in range(N_FEAT) if theta[i][j] != before[i][j]}
        self.assertEqual(touched, {1 * GRID + 2})

    def test_ascending_raises_the_log_probability_of_a_positively_weighted_action(self) -> None:
        theta = init_theta(random.Random(6))
        x = features((0, 0))
        probs = softmax(logits(theta, x))
        traj = [(x, 2, -1.0, probs)]
        before = log_pi(theta, x, 2)
        reinforce_step(theta, traj, [1.0], 0.1)  # positive advantage -> make action 2 likelier
        self.assertGreater(log_pi(theta, x, 2), before)


class TestTraining(unittest.TestCase):
    def test_training_is_reproducible_under_a_seeded_rng(self) -> None:
        t1, l1 = reinforce(200, rng=random.Random(0))
        t2, l2 = reinforce(200, rng=random.Random(0))
        self.assertEqual(l1, l2)
        self.assertEqual(t1, t2)

    def test_return_improves_and_approaches_the_discounted_optimum(self) -> None:
        _theta, log = reinforce(1200, rng=random.Random(42))
        early = sum(log[:100]) / 100
        late = sum(log[-200:]) / 200
        self.assertGreater(late, early + 2.0)
        self.assertAlmostEqual(late, OPTIMAL_DISCOUNTED_RETURN, delta=0.4)
        self.assertLessEqual(max(log), OPTIMAL_DISCOUNTED_RETURN + 1e-9)

    def test_learned_policy_walks_toward_the_goal_along_the_visited_path(self) -> None:
        theta, _log = reinforce(1200, use_baseline=True, rng=random.Random(42))
        policy = greedy_policy(theta)
        state, steps = reset(), 0
        while state != TERMINAL and steps < 20:
            state, _r, _done = step(state, ACTIONS.index(policy[state]))
            steps += 1
        self.assertEqual(state, TERMINAL)
        self.assertEqual(steps, 6)

    def test_greedy_policy_covers_every_non_terminal_cell(self) -> None:
        theta, _log = reinforce(50, rng=random.Random(1))
        policy = greedy_policy(theta)
        self.assertEqual(len(policy), GRID * GRID - 1)
        self.assertNotIn(TERMINAL, policy)


if __name__ == "__main__":
    unittest.main()
