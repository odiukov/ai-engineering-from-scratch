"""Tests for PPO: rollout bookkeeping, GAE, and the clipped surrogate.

Covers docs/en.md "Build It" steps 1-5. The clip is exercised from both sides:
with A_t > 0 the ceiling at 1 + eps binds, with A_t < 0 the floor at 1 - eps
binds, and in both cases a clipped sample must contribute exactly zero policy
gradient. A single-record buffer isolates one update so the arithmetic is
checkable in closed form.
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
    collect_rollout,
    evaluate,
    features,
    gae,
    greedy_policy,
    init_theta,
    init_w,
    logits,
    normalize,
    ppo_update,
    softmax,
    step,
    value,
)

EPS = 0.2
LR_A = 0.1
UNIFORM = 1.0 / N_ACTIONS


def zero_theta():
    """All-zero logits, so every state's policy is uniform and probs are exact."""
    return [[0.0] * N_FEAT for _ in range(N_ACTIONS)]


def buffer_with_ratio(ratio, action=0, state=(0, 0)):
    """One record whose importance ratio under a uniform policy equals `ratio`."""
    logp = math.log(UNIFORM)
    return [
        {
            "x": features(state),
            "a": action,
            "r": -1.0,
            "done": False,
            "v_old": 0.0,
            "log_pi_old": logp - math.log(ratio),
        }
    ]


def run_single_update(ratio, adv, eps=EPS):
    theta = zero_theta()
    w = init_w(None)
    buf = buffer_with_ratio(ratio)
    mean_kl, clip_frac = ppo_update(
        theta,
        w,
        buf,
        [adv],
        [0.0],
        lr_a=LR_A,
        lr_v=0.0,
        eps=eps,
        epochs=1,
        batch=8,
        rng=random.Random(0),
    )
    return theta, mean_kl, clip_frac


class TestRolloutBookkeeping(unittest.TestCase):
    def test_log_pi_old_matches_the_probability_of_the_sampled_action(self):
        rng = random.Random(4)
        theta = init_theta(rng)
        w = init_w(rng)
        buf = collect_rollout(theta, w, rng, horizon=20, n_envs=3)
        self.assertGreater(len(buf), 0)
        for rec in buf:
            probs = softmax(logits(theta, rec["x"]))
            self.assertAlmostEqual(math.exp(rec["log_pi_old"]), probs[rec["a"]], places=12)

    def test_v_old_is_the_critic_value_frozen_at_rollout_time(self):
        rng = random.Random(9)
        theta = init_theta(rng)
        w = [0.5 * j for j in range(N_FEAT)]
        buf = collect_rollout(theta, w, rng, horizon=10, n_envs=2)
        for rec in buf:
            self.assertAlmostEqual(rec["v_old"], value(w, rec["x"]), places=12)

    def test_episode_stops_at_the_goal(self):
        rng = random.Random(2)
        theta = init_theta(rng)
        w = init_w(rng)
        buf = collect_rollout(theta, w, rng, horizon=200, n_envs=1)
        dones = [i for i, rec in enumerate(buf) if rec["done"]]
        self.assertLessEqual(len(dones), 1)
        if dones:
            self.assertEqual(dones[0], len(buf) - 1)

    def test_terminal_state_is_absorbing(self):
        for a in range(N_ACTIONS):
            s_next, reward, done = step(TERMINAL, a)
            self.assertEqual((s_next, reward, done), (TERMINAL, 0.0, True))


class TestGAE(unittest.TestCase):
    def test_lambda_zero_is_the_one_step_td_residual(self):
        gamma = 0.9
        buf = [
            {"r": -1.0, "v_old": 2.0, "done": False},
            {"r": -1.0, "v_old": 3.0, "done": False},
            {"r": 5.0, "v_old": 4.0, "done": True},
        ]
        advs, _ = gae(buf, gamma=gamma, lam=0.0)
        self.assertAlmostEqual(advs[0], -1.0 + gamma * 3.0 - 2.0, places=12)
        self.assertAlmostEqual(advs[1], -1.0 + gamma * 4.0 - 3.0, places=12)
        self.assertAlmostEqual(advs[2], 5.0 - 4.0, places=12)

    def test_lambda_one_is_monte_carlo_return_minus_value(self):
        gamma = 0.95
        rewards = [-1.0, -1.0, -1.0, 8.0]
        values = [1.0, -2.0, 0.5, 3.0]
        buf = [
            {"r": r, "v_old": v, "done": i == len(rewards) - 1}
            for i, (r, v) in enumerate(zip(rewards, values))
        ]
        advs, returns = gae(buf, gamma=gamma, lam=1.0)
        for t in range(len(rewards)):
            mc = 0.0
            for r in reversed(rewards[t:]):
                mc = r + gamma * mc
            self.assertAlmostEqual(advs[t], mc - values[t], places=10)
            self.assertAlmostEqual(returns[t], mc, places=10)

    def test_advantage_normalization_gives_zero_mean_unit_std(self):
        xs = normalize([1.0, -3.0, 0.5, 8.0])
        mean = sum(xs) / len(xs)
        var = sum((x - mean) ** 2 for x in xs) / len(xs)
        self.assertAlmostEqual(mean, 0.0, places=10)
        self.assertAlmostEqual(var, 1.0, places=6)


class TestClipUpperSide(unittest.TestCase):
    """A_t > 0: the ceiling at 1 + eps is the binding boundary."""

    def test_ratio_above_one_plus_eps_gives_exactly_zero_policy_gradient(self):
        theta, _, clip_frac = run_single_update(ratio=1.5, adv=1.0)
        self.assertEqual(clip_frac, 1.0)
        self.assertEqual(theta, zero_theta())

    def test_ratio_inside_the_corridor_updates_by_ratio_times_advantage(self):
        ratio, adv = 1.1, 1.0
        theta, _, clip_frac = run_single_update(ratio=ratio, adv=adv)
        self.assertEqual(clip_frac, 0.0)
        j = features((0, 0)).index(1.0)
        self.assertAlmostEqual(theta[0][j], LR_A * ratio * adv * (1.0 - UNIFORM), places=9)
        for other in range(1, N_ACTIONS):
            self.assertAlmostEqual(theta[other][j], LR_A * ratio * adv * (-UNIFORM), places=9)

    def test_upper_boundary_sits_exactly_at_one_plus_eps(self):
        inside, _, frac_inside = run_single_update(ratio=(1 + EPS) * (1 - 1e-6), adv=1.0)
        outside, _, frac_outside = run_single_update(ratio=(1 + EPS) * (1 + 1e-6), adv=1.0)
        self.assertEqual(frac_inside, 0.0)
        self.assertEqual(frac_outside, 1.0)
        self.assertNotEqual(inside, zero_theta())
        self.assertEqual(outside, zero_theta())

    def test_low_ratio_with_positive_advantage_is_the_beneficial_side_and_still_updates(self):
        theta, _, clip_frac = run_single_update(ratio=0.4, adv=1.0)
        self.assertEqual(clip_frac, 0.0)
        self.assertNotEqual(theta, zero_theta())


class TestClipLowerSide(unittest.TestCase):
    """A_t < 0: the floor at 1 - eps is the binding boundary."""

    def test_ratio_below_one_minus_eps_gives_exactly_zero_policy_gradient(self):
        theta, _, clip_frac = run_single_update(ratio=0.5, adv=-1.0)
        self.assertEqual(clip_frac, 1.0)
        self.assertEqual(theta, zero_theta())

    def test_lower_boundary_sits_exactly_at_one_minus_eps(self):
        inside, _, frac_inside = run_single_update(ratio=(1 - EPS) * (1 + 1e-6), adv=-1.0)
        outside, _, frac_outside = run_single_update(ratio=(1 - EPS) * (1 - 1e-6), adv=-1.0)
        self.assertEqual(frac_inside, 0.0)
        self.assertEqual(frac_outside, 1.0)
        self.assertNotEqual(inside, zero_theta())
        self.assertEqual(outside, zero_theta())

    def test_high_ratio_with_negative_advantage_is_not_clipped(self):
        theta, _, clip_frac = run_single_update(ratio=1.9, adv=-1.0)
        self.assertEqual(clip_frac, 0.0)
        j = features((0, 0)).index(1.0)
        self.assertLess(theta[0][j], 0.0)

    def test_negative_advantage_pushes_the_taken_action_down(self):
        theta, _, _ = run_single_update(ratio=1.0, adv=-1.0)
        j = features((0, 0)).index(1.0)
        self.assertLess(theta[0][j], 0.0)
        for other in range(1, N_ACTIONS):
            self.assertGreater(theta[other][j], 0.0)

    def test_epsilon_widens_the_corridor(self):
        _, _, tight = run_single_update(ratio=0.75, adv=-1.0, eps=0.2)
        _, _, loose = run_single_update(ratio=0.75, adv=-1.0, eps=0.4)
        self.assertEqual(tight, 1.0)
        self.assertEqual(loose, 0.0)


class TestDiagnostics(unittest.TestCase):
    def test_mean_kl_is_minus_log_of_the_ratio(self):
        _, mean_kl, _ = run_single_update(ratio=1.1, adv=1.0)
        self.assertAlmostEqual(mean_kl, -math.log(1.1), places=10)

    def test_unmoved_policy_reports_zero_kl_and_zero_clipping(self):
        _, mean_kl, clip_frac = run_single_update(ratio=1.0, adv=1.0)
        self.assertAlmostEqual(mean_kl, 0.0, places=12)
        self.assertEqual(clip_frac, 0.0)

    def test_clip_fraction_stays_a_probability_on_a_real_rollout(self):
        rng = random.Random(21)
        theta = init_theta(rng)
        w = init_w(rng)
        buf = collect_rollout(theta, w, rng, horizon=30, n_envs=4)
        advs, rets = gae(buf)
        _, clip_frac = ppo_update(theta, w, buf, advs, rets, rng=random.Random(1))
        self.assertGreaterEqual(clip_frac, 0.0)
        self.assertLessEqual(clip_frac, 1.0)


class TestTrainingLoop(unittest.TestCase):
    def test_repeated_updates_move_the_policy_further_than_one(self):
        base = zero_theta()
        buf = buffer_with_ratio(1.0)
        theta_one = zero_theta()
        ppo_update(theta_one, init_w(None), buf, [1.0], [0.0], lr_v=0.0, epochs=1, rng=random.Random(0))
        theta_many = zero_theta()
        ppo_update(theta_many, init_w(None), buffer_with_ratio(1.0), [1.0], [0.0], lr_v=0.0, epochs=3, rng=random.Random(0))
        j = features((0, 0)).index(1.0)
        self.assertGreater(theta_many[0][j], theta_one[0][j])
        self.assertNotEqual(theta_one, base)

    def test_critic_regresses_toward_the_return_target(self):
        theta = zero_theta()
        w = init_w(None)
        buf = buffer_with_ratio(1.0)
        ppo_update(theta, w, buf, [0.0], [10.0], lr_v=0.1, epochs=1, rng=random.Random(0))
        self.assertGreater(value(w, features((0, 0))), 0.0)
        self.assertLess(value(w, features((0, 0))), 10.0)

    def test_ppo_improves_evaluated_return(self):
        rng = random.Random(123)
        theta = init_theta(rng)
        w = init_w(rng)
        before = evaluate(theta, random.Random(0), episodes=100)
        for _ in range(25):
            buf = collect_rollout(theta, w, rng)
            advs, rets = gae(buf)
            ppo_update(theta, w, buf, advs, rets, rng=rng)
        after = evaluate(theta, random.Random(0), episodes=100)
        self.assertGreater(after, before)

    def test_evaluate_is_reproducible_for_a_given_seed(self):
        theta = init_theta(random.Random(6))
        first = evaluate(theta, random.Random(31), episodes=40)
        second = evaluate(theta, random.Random(31), episodes=40)
        self.assertEqual(first, second)

    def test_greedy_policy_covers_every_non_terminal_state(self):
        policy = greedy_policy(init_theta(random.Random(8)))
        self.assertEqual(len(policy), GRID * GRID - 1)
        self.assertNotIn(TERMINAL, policy)
        self.assertTrue(set(policy.values()) <= set(ACTIONS))


if __name__ == "__main__":
    unittest.main()
