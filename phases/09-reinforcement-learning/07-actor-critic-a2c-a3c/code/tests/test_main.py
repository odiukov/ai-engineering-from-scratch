"""Tests for the A2C-style actor-critic: env contract, GAE dial, entropy gradient.

Covers docs/en.md "Build It" steps 1-3. The entropy-bonus gradient is checked
against a central-difference estimate of dH/dz because the analytic form
-p_i * (log p_i + H) is easy to get subtly wrong (dropping H leaves a residual
that does not sum to zero).
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
    actor_critic,
    block_mean,
    features,
    gae_advantages,
    greedy_policy,
    init_theta,
    init_w,
    logits,
    normalize,
    rollout,
    sample,
    softmax,
    step,
    value,
)


def node(reward, val, done=False):
    return {"r": reward, "v": val, "done": done}


def entropy(probs):
    return -sum(p * math.log(p) for p in probs if p > 0.0)


def numeric_entropy_grad(z, i, h=1e-5):
    """Central difference of H(softmax(z)) with respect to logit z_i."""
    z_plus = list(z)
    z_plus[i] += h
    z_minus = list(z)
    z_minus[i] -= h
    return (entropy(softmax(z_plus)) - entropy(softmax(z_minus))) / (2.0 * h)


class TestGridEnv(unittest.TestCase):
    def test_action_into_wall_clamps_and_still_costs_a_step(self):
        s_next, reward, done = step((0, 0), ACTIONS.index("up"))
        self.assertEqual(s_next, (0, 0))
        self.assertEqual(reward, -1.0)
        self.assertFalse(done)

    def test_terminal_state_is_absorbing_and_free(self):
        for a in range(N_ACTIONS):
            s_next, reward, done = step(TERMINAL, a)
            self.assertEqual(s_next, TERMINAL)
            self.assertEqual(reward, 0.0)
            self.assertTrue(done)

    def test_stepping_onto_goal_sets_done(self):
        s_next, reward, done = step((GRID - 2, GRID - 1), ACTIONS.index("down"))
        self.assertEqual(s_next, TERMINAL)
        self.assertEqual(reward, -1.0)
        self.assertTrue(done)

    def test_features_are_one_hot_over_the_grid(self):
        x = features((1, 2))
        self.assertEqual(len(x), N_FEAT)
        self.assertEqual(sum(x), 1.0)
        self.assertEqual(x[1 * GRID + 2], 1.0)

    def test_value_of_one_hot_state_reads_a_single_weight(self):
        w = [float(j) for j in range(N_FEAT)]
        self.assertEqual(value(w, features((2, 1))), float(2 * GRID + 1))


class TestSoftmaxAndSampling(unittest.TestCase):
    def test_softmax_normalizes_and_is_shift_invariant(self):
        z = [0.4, -1.2, 3.0, 0.0]
        p = softmax(z)
        self.assertAlmostEqual(sum(p), 1.0, places=12)
        shifted = softmax([zi + 100.0 for zi in z])
        for a, b in zip(p, shifted):
            self.assertAlmostEqual(a, b, places=12)

    def test_sample_always_returns_the_only_supported_action(self):
        rng = random.Random(3)
        probs = [0.0, 1.0, 0.0, 0.0]
        self.assertEqual({sample(probs, rng) for _ in range(50)}, {1})

    def test_sample_frequencies_track_the_distribution(self):
        rng = random.Random(17)
        probs = [0.7, 0.1, 0.1, 0.1]
        counts = [0] * N_ACTIONS
        for _ in range(4000):
            counts[sample(probs, rng)] += 1
        self.assertAlmostEqual(counts[0] / 4000.0, 0.7, delta=0.03)


class TestGAE(unittest.TestCase):
    def test_lambda_zero_is_the_one_step_td_residual(self):
        gamma = 0.9
        traj = [node(-1.0, 2.0), node(-1.0, 3.0), node(-1.0, 5.0, done=True)]
        advs, _ = gae_advantages(traj, gamma=gamma, lam=0.0)
        self.assertAlmostEqual(advs[0], -1.0 + gamma * 3.0 - 2.0, places=12)
        self.assertAlmostEqual(advs[1], -1.0 + gamma * 5.0 - 3.0, places=12)
        self.assertAlmostEqual(advs[2], -1.0 - 5.0, places=12)

    def test_lambda_one_is_monte_carlo_return_minus_value(self):
        gamma = 0.9
        rewards = [-1.0, -2.0, -3.0, 10.0]
        values = [0.5, -1.5, 4.0, 2.25]
        traj = [node(r, v) for r, v in zip(rewards, values)]
        traj[-1]["done"] = True
        advs, returns = gae_advantages(traj, gamma=gamma, lam=1.0)
        for t in range(len(rewards)):
            mc = 0.0
            for r in reversed(rewards[t:]):
                mc = r + gamma * mc
            self.assertAlmostEqual(advs[t], mc - values[t], places=10)
            self.assertAlmostEqual(returns[t], mc, places=10)

    def test_done_step_does_not_bootstrap_the_next_value(self):
        gamma = 0.9
        traj = [node(1.0, 5.0), node(2.0, 7.0, done=True), node(3.0, 11.0)]
        advs, _ = gae_advantages(traj, gamma=gamma, lam=0.0)
        self.assertAlmostEqual(advs[1], 2.0 - 7.0, places=12)

    def test_returns_are_advantages_plus_values(self):
        traj = [node(-1.0, 0.25), node(-1.0, -0.75), node(4.0, 1.5, done=True)]
        advs, returns = gae_advantages(traj)
        for a, r, n in zip(advs, returns, traj):
            self.assertAlmostEqual(r, a + n["v"], places=12)


class TestNormalize(unittest.TestCase):
    def test_batch_becomes_zero_mean_unit_std(self):
        xs = normalize([3.0, -1.0, 7.5, 0.25, -4.0])
        mean = sum(xs) / len(xs)
        var = sum((x - mean) ** 2 for x in xs) / len(xs)
        self.assertAlmostEqual(mean, 0.0, places=10)
        self.assertAlmostEqual(var, 1.0, places=6)

    def test_single_element_batch_is_passed_through_untouched(self):
        self.assertEqual(normalize([4.2]), [4.2])

    def test_constant_batch_does_not_divide_by_zero(self):
        xs = normalize([2.0, 2.0, 2.0])
        self.assertEqual(xs, [0.0, 0.0, 0.0])


class TestEntropyGradient(unittest.TestCase):
    """The entropy bonus is only isolable by differencing two runs of the same
    episode: the rollout is generated before any update, so with a fixed seed the
    trajectory, the stored probs and the critic path are identical and the whole
    theta difference is the entropy term."""

    SEED = 11
    LR_A = 0.05
    ENT = 0.7

    def _theta_delta_and_trajectory(self):
        theta_no_ent, _, _ = actor_critic(
            1, lr_a=self.LR_A, ent_coef=0.0, rng=random.Random(self.SEED)
        )
        theta_ent, _, _ = actor_critic(
            1, lr_a=self.LR_A, ent_coef=self.ENT, rng=random.Random(self.SEED)
        )
        delta = [
            [theta_ent[i][j] - theta_no_ent[i][j] for j in range(N_FEAT)]
            for i in range(N_ACTIONS)
        ]
        replay = random.Random(self.SEED)
        theta0 = init_theta(replay)
        w0 = init_w(replay)
        traj = rollout(theta0, w0, replay)
        return delta, theta0, traj

    def test_entropy_gradient_matches_central_difference(self):
        delta, theta0, traj = self._theta_delta_and_trajectory()
        visits = {}
        for n in traj:
            j = n["x"].index(1.0)
            visits[j] = visits.get(j, 0) + 1
        self.assertGreaterEqual(len(visits), 3, "trajectory too short to be a real check")

        for j, count in visits.items():
            x = [0.0] * N_FEAT
            x[j] = 1.0
            z = logits(theta0, x)
            for i in range(N_ACTIONS):
                expected = self.LR_A * self.ENT * count * numeric_entropy_grad(z, i)
                self.assertAlmostEqual(delta[i][j], expected, delta=1e-9)

    def test_entropy_gradient_components_sum_to_zero_per_state(self):
        delta, _, traj = self._theta_delta_and_trajectory()
        for n in traj:
            j = n["x"].index(1.0)
            self.assertAlmostEqual(sum(delta[i][j] for i in range(N_ACTIONS)), 0.0, delta=1e-12)

    def test_entropy_bonus_pushes_toward_the_uniform_policy(self):
        delta, theta0, traj = self._theta_delta_and_trajectory()
        j = traj[0]["x"].index(1.0)
        x = [0.0] * N_FEAT
        x[j] = 1.0
        before = softmax(logits(theta0, x))
        after = softmax([logits(theta0, x)[i] + delta[i][j] for i in range(N_ACTIONS)])
        self.assertGreater(entropy(after), entropy(before))


class TestTrainingLoop(unittest.TestCase):
    def test_actor_critic_improves_return_over_training(self):
        _, _, log = actor_critic(600, rng=random.Random(5))
        self.assertEqual(len(log), 600)
        early = sum(log[:100]) / 100.0
        late = sum(log[-100:]) / 100.0
        self.assertGreater(late, early)

    def test_critic_ranks_the_goal_neighbourhood_above_the_start(self):
        _, w, _ = actor_critic(1200, rng=random.Random(7))
        near_goal = value(w, features((GRID - 1, GRID - 2)))
        start = value(w, features((0, 0)))
        self.assertGreater(near_goal, start)

    def test_greedy_policy_names_an_action_for_every_non_terminal_state(self):
        theta, _, _ = actor_critic(50, rng=random.Random(2))
        policy = greedy_policy(theta)
        self.assertEqual(len(policy), GRID * GRID - 1)
        self.assertNotIn(TERMINAL, policy)
        self.assertTrue(set(policy.values()) <= set(ACTIONS))

    def test_block_mean_drops_the_ragged_tail(self):
        self.assertEqual(block_mean([1.0, 3.0, 5.0, 7.0, 9.0], 2), [2.0, 6.0])


if __name__ == "__main__":
    unittest.main()
