"""Tests for GRPO in miniature: verifier reward, group-relative advantage, KL term.

Covers docs/en.md "Build It" steps 1-5. The KL penalty's analytic logit gradient
d KL/dz_i = p_i * (log(p_i / q_i) - KL) is checked against a central difference,
and separately checked to sum to zero over i — the property that the -KL term
exists to guarantee. Setting lr = 0 isolates the penalty from the policy update.
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
    N_ANSWERS,
    N_PROMPTS,
    QUESTIONS,
    evaluate,
    grpo_step,
    policy_probs,
    reinforce_step,
    sample,
    softmax,
    train_grpo,
    train_reinforce,
    verify,
)


def kl_of_logits(z, q):
    p = softmax(z)
    return sum(pi * (math.log(pi) - math.log(qi)) for pi, qi in zip(p, q))


def numeric_kl_grad(z, q, i, h=1e-6):
    z_plus = list(z)
    z_plus[i] += h
    z_minus = list(z)
    z_minus[i] -= h
    return (kl_of_logits(z_plus, q) - kl_of_logits(z_minus, q)) / (2.0 * h)


def rows(spec):
    return [list(row) for row in spec]


class TestVerifier(unittest.TestCase):
    def test_only_the_recorded_answer_scores_one(self):
        for p_idx, question in enumerate(QUESTIONS):
            for answer in range(N_ANSWERS):
                expected = 1.0 if answer == question["correct"] else 0.0
                self.assertEqual(verify(p_idx, answer), expected)

    def test_every_prompt_has_exactly_one_correct_answer_in_range(self):
        self.assertEqual(len(QUESTIONS), N_PROMPTS)
        for question in QUESTIONS:
            self.assertTrue(0 <= question["correct"] < N_ANSWERS)


class TestPolicy(unittest.TestCase):
    def test_policy_probs_normalize_per_prompt(self):
        theta = rows([[0.4, -0.3, 1.2, 0.1]] * N_PROMPTS)
        for p_idx in range(N_PROMPTS):
            self.assertAlmostEqual(sum(policy_probs(theta, p_idx)), 1.0, places=12)

    def test_sample_returns_the_only_supported_answer(self):
        rng = random.Random(0)
        probs = [0.0, 0.0, 1.0, 0.0]
        self.assertEqual({sample(probs, rng) for _ in range(40)}, {2})

    def test_evaluate_takes_the_greedy_argmax(self):
        theta = [[0.0] * N_ANSWERS for _ in range(N_PROMPTS)]
        for p_idx, question in enumerate(QUESTIONS):
            theta[p_idx][question["correct"]] = 10.0
        self.assertEqual(evaluate(theta, episodes=60, rng=random.Random(0)), 1.0)

    def test_evaluate_scores_zero_for_a_uniformly_wrong_policy(self):
        theta = [[0.0] * N_ANSWERS for _ in range(N_PROMPTS)]
        for p_idx, question in enumerate(QUESTIONS):
            wrong = (question["correct"] + 1) % N_ANSWERS
            theta[p_idx][wrong] = 10.0
        self.assertEqual(evaluate(theta, episodes=60, rng=random.Random(0)), 0.0)


class TestKLGradient(unittest.TestCase):
    BETA = 0.5

    def _isolated_kl_step(self, seed):
        """lr = 0 zeroes the group-relative term, leaving only the KL descent."""
        before = rows([[0.4, -0.3, 1.2, 0.1], [0.0, 0.5, -0.5, 0.2], [1.0, 0.0, 0.0, -1.0]])
        reference = rows([[0.0] * N_ANSWERS] * N_PROMPTS)
        theta = rows(before)
        grpo_step(theta, reference, random.Random(seed), G=8, beta=self.BETA, lr=0.0)
        changed = [p for p in range(N_PROMPTS) if theta[p] != before[p]]
        self.assertEqual(len(changed), 1, "exactly one prompt row should move")
        p_idx = changed[0]
        delta = [theta[p_idx][i] - before[p_idx][i] for i in range(N_ANSWERS)]
        return before[p_idx], policy_probs(reference, p_idx), delta

    def test_kl_gradient_matches_central_difference(self):
        for seed in (0, 1, 4):
            z, q, delta = self._isolated_kl_step(seed)
            for i in range(N_ANSWERS):
                expected = -self.BETA * numeric_kl_grad(z, q, i)
                self.assertAlmostEqual(delta[i], expected, delta=1e-9)

    def test_kl_gradient_components_sum_to_zero(self):
        for seed in (0, 1, 4):
            _, _, delta = self._isolated_kl_step(seed)
            self.assertAlmostEqual(sum(delta), 0.0, delta=1e-12)

    def test_kl_descent_moves_the_policy_toward_the_reference(self):
        z, q, delta = self._isolated_kl_step(0)
        before = kl_of_logits(z, q)
        after = kl_of_logits([z[i] + delta[i] for i in range(N_ANSWERS)], q)
        self.assertLess(after, before)

    def test_policy_equal_to_reference_reports_zero_kl(self):
        theta = rows([[0.4, -0.3, 1.2, 0.1]] * N_PROMPTS)
        reference = rows(theta)
        _, reported_kl = grpo_step(theta, reference, random.Random(0), beta=0.1)
        self.assertAlmostEqual(reported_kl, 0.0, places=12)


class TestGroupRelativeAdvantage(unittest.TestCase):
    def test_a_group_with_no_reward_spread_produces_no_policy_gradient(self):
        """Every sample scores the same, so the group baseline cancels it exactly."""
        theta = rows([[50.0, 0.0, 0.0, 0.0]] * N_PROMPTS)
        reference = rows(theta)
        before = rows(theta)
        mean_r, _ = grpo_step(theta, reference, random.Random(0), G=8, beta=0.0, lr=0.5)
        self.assertEqual(theta, before)
        self.assertIn(mean_r, (0.0, 1.0))

    def test_mean_reward_is_the_group_pass_rate(self):
        theta = rows([[0.0] * N_ANSWERS] * N_PROMPTS)
        reference = rows(theta)
        mean_r, _ = grpo_step(theta, reference, random.Random(3), G=8, beta=0.0)
        self.assertIn(round(mean_r * 8), range(9))
        self.assertGreaterEqual(mean_r, 0.0)
        self.assertLessEqual(mean_r, 1.0)

    def test_correct_answer_gains_logit_mass_when_the_group_is_mixed(self):
        theta = rows([[0.0] * N_ANSWERS] * N_PROMPTS)
        reference = rows(theta)
        before = rows(theta)
        grpo_step(theta, reference, random.Random(3), G=16, beta=0.0, lr=1.0)
        moved = [p for p in range(N_PROMPTS) if theta[p] != before[p]]
        self.assertEqual(len(moved), 1)
        p_idx = moved[0]
        correct = QUESTIONS[p_idx]["correct"]
        self.assertGreater(theta[p_idx][correct], before[p_idx][correct])

    def test_updates_touch_only_the_sampled_prompt(self):
        theta = rows([[0.2, -0.4, 0.9, 0.0]] * N_PROMPTS)
        reference = rows([[0.0] * N_ANSWERS] * N_PROMPTS)
        before = rows(theta)
        grpo_step(theta, reference, random.Random(7), G=8, beta=0.3, lr=0.2)
        moved = [p for p in range(N_PROMPTS) if theta[p] != before[p]]
        self.assertEqual(len(moved), 1)


class TestReinforceBaseline(unittest.TestCase):
    def test_a_failed_sample_carries_no_gradient_without_a_baseline(self):
        """Reward 0 kills the whole update — the variance problem a baseline fixes."""
        theta = [[0.0] * N_ANSWERS for _ in range(N_PROMPTS)]
        for p_idx, question in enumerate(QUESTIONS):
            wrong = (question["correct"] + 1) % N_ANSWERS
            theta[p_idx][wrong] = 30.0
        before = rows(theta)
        reward = reinforce_step(theta, random.Random(0), lr=1.0)
        self.assertEqual(reward, 0.0)
        self.assertEqual(theta, before)

    def test_a_successful_sample_reinforces_the_answer_taken(self):
        theta = [[0.0] * N_ANSWERS for _ in range(N_PROMPTS)]
        for p_idx, question in enumerate(QUESTIONS):
            theta[p_idx][question["correct"]] = 30.0
        before = rows(theta)
        reward = reinforce_step(theta, random.Random(0), lr=1.0)
        self.assertEqual(reward, 1.0)
        moved = [p for p in range(N_PROMPTS) if theta[p] != before[p]]
        self.assertEqual(len(moved), 1)


class TestTraining(unittest.TestCase):
    def test_grpo_solves_the_bandit(self):
        theta, history = train_grpo(updates=400, rng=random.Random(3))
        self.assertEqual(len(history), 400)
        self.assertEqual(evaluate(theta, episodes=200, rng=random.Random(42)), 1.0)

    def test_grpo_pass_rate_rises_over_training(self):
        _, history = train_grpo(updates=400, rng=random.Random(3))
        early = sum(m for m, _ in history[:50]) / 50.0
        late = sum(m for m, _ in history[-50:]) / 50.0
        self.assertGreater(late, early)

    def test_kl_to_reference_grows_from_zero_as_the_policy_moves(self):
        _, history = train_grpo(updates=200, rng=random.Random(3))
        self.assertAlmostEqual(history[0][1], 0.0, places=12)
        self.assertGreater(history[-1][1], history[0][1])

    def test_reinforce_also_learns_but_from_successes_only(self):
        theta, history = train_reinforce(updates=400, rng=random.Random(3))
        early = sum(history[:50]) / 50.0
        late = sum(history[-50:]) / 50.0
        self.assertGreater(late, early)
        self.assertGreater(evaluate(theta, episodes=200, rng=random.Random(42)), 0.5)

    def test_training_is_reproducible_for_a_given_seed(self):
        first = train_grpo(updates=50, rng=random.Random(2))[1]
        second = train_grpo(updates=50, rng=random.Random(2))[1]
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
