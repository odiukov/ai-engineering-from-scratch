"""Tests for the DPO pieces in phases/10-llms-from-scratch/08-dpo/code/main.py.

Pins the DPO loss at its zero-margin value, its monotonicity in the log-ratio
gap, its analytic gradient against a central difference, the beta -> 0 limit
where the loss stops caring about the gap, and the sequence log-probability
against an independent log-softmax computation.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
sys.path.insert(0, CODE_DIR)


def _load_lesson_main():
    """Load this lesson's main.py under an explicit module name.

    main.py itself does `from main import MiniGPT` to pull lesson 04's model, so
    importing it as `main` would make it shadow its own dependency. Giving it a
    distinct name keeps lesson 04's `main` resolvable.
    """
    path = os.path.join(CODE_DIR, "main.py")
    spec = importlib.util.spec_from_file_location("dpo_main", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


main = _load_lesson_main()

MAX_SEQ_LEN = 64


def small_gpt(seed=0):
    np.random.seed(seed)
    return main.MiniGPT(
        vocab_size=256, embed_dim=32, num_heads=4, num_layers=2,
        max_seq_len=MAX_SEQ_LEN, ff_dim=64,
    )


def loss_only(policy_w, policy_l, ref_w, ref_l, beta):
    return float(main.dpo_loss(policy_w, policy_l, ref_w, ref_l, beta)[0])


class TestDPOLoss(unittest.TestCase):
    def test_loss_is_log_two_when_policy_equals_reference(self):
        loss, metrics = main.dpo_loss(-3.0, -8.0, -3.0, -8.0, beta=0.1)
        self.assertAlmostEqual(float(loss), float(np.log(2)), places=6)
        self.assertAlmostEqual(metrics["logit"], 0.0)
        self.assertAlmostEqual(metrics["implicit_preferred_reward"], 0.0)
        self.assertAlmostEqual(metrics["implicit_rejected_reward"], 0.0)
        self.assertAlmostEqual(metrics["reward_margin"], 0.0)

    def test_loss_falls_monotonically_in_the_log_ratio_gap(self):
        beta = 0.1
        gaps = [-8.0, -2.0, 0.0, 2.0, 8.0, 40.0]
        losses = [loss_only(gap, 0.0, 0.0, 0.0, beta) for gap in gaps]
        for earlier, later in zip(losses, losses[1:]):
            self.assertGreater(earlier, later)
        self.assertGreater(losses[0], float(np.log(2)))
        self.assertLess(losses[-1], 0.02)

    def test_implicit_rewards_are_beta_times_the_log_ratios(self):
        beta = 0.25
        policy_w, policy_l, ref_w, ref_l = -4.0, -6.5, -5.0, -6.0
        _, metrics = main.dpo_loss(policy_w, policy_l, ref_w, ref_l, beta)
        self.assertAlmostEqual(metrics["preferred_ratio"], policy_w - ref_w)
        self.assertAlmostEqual(metrics["rejected_ratio"], policy_l - ref_l)
        self.assertAlmostEqual(metrics["implicit_preferred_reward"], beta * (policy_w - ref_w))
        self.assertAlmostEqual(metrics["implicit_rejected_reward"], beta * (policy_l - ref_l))
        self.assertAlmostEqual(metrics["reward_margin"], metrics["logit"])

    def test_gradient_matches_central_difference(self):
        # d/d(logpi_w) [-log sigmoid(beta * gap)] = -beta * sigmoid(-beta * gap),
        # and the rejected side is the exact mirror image.
        step = 1e-6
        ref_w, ref_l = -0.4, -1.1
        for beta in (0.05, 0.1, 0.5):
            for policy_w, policy_l in ((-0.2, -1.5), (-2.0, -0.5), (0.3, 0.3)):
                logit = beta * ((policy_w - ref_w) - (policy_l - ref_l))
                analytic = -beta * float(main.sigmoid(-logit))
                numeric_w = (
                    loss_only(policy_w + step, policy_l, ref_w, ref_l, beta)
                    - loss_only(policy_w - step, policy_l, ref_w, ref_l, beta)
                ) / (2 * step)
                numeric_l = (
                    loss_only(policy_w, policy_l + step, ref_w, ref_l, beta)
                    - loss_only(policy_w, policy_l - step, ref_w, ref_l, beta)
                ) / (2 * step)
                self.assertAlmostEqual(numeric_w, analytic, places=6)
                self.assertAlmostEqual(numeric_l, -analytic, places=6)

    def test_small_beta_makes_the_loss_insensitive_to_the_gap(self):
        gap = 12.0
        losses = [loss_only(gap, 0.0, 0.0, 0.0, beta) for beta in (1.0, 0.1, 0.01, 1e-4)]
        for earlier, later in zip(losses, losses[1:]):
            self.assertGreater(later, earlier)
        self.assertAlmostEqual(losses[-1], float(np.log(2)), places=4)

        # The gradient magnitude collapses with beta, which is why a tiny beta
        # barely moves the policy away from the reference.
        step = 1e-6
        magnitudes = []
        for beta in (1.0, 0.1, 0.01):
            grad = (
                loss_only(gap + step, 0.0, 0.0, 0.0, beta)
                - loss_only(gap - step, 0.0, 0.0, 0.0, beta)
            ) / (2 * step)
            magnitudes.append(abs(grad))
        for earlier, later in zip(magnitudes, magnitudes[1:]):
            self.assertLess(later, earlier)

    def test_winning_the_pair_needs_the_policy_to_outrun_the_reference(self):
        beta = 0.1
        ref_w, ref_l = -5.0, -5.0
        baseline = main.dpo_loss(-5.0, -5.0, ref_w, ref_l, beta)
        boosted = main.dpo_loss(-4.0, -5.0, ref_w, ref_l, beta)
        self.assertLess(float(boosted[0]), float(baseline[0]))
        self.assertGreater(boosted[1]["implicit_preferred_reward"], 0.0)
        self.assertGreater(boosted[1]["reward_margin"], baseline[1]["reward_margin"])


class TestSequenceLogProb(unittest.TestCase):
    def setUp(self):
        self.model = small_gpt(seed=1)
        self.prompt = main.tokenize_sequence("What is 2 + 2?")
        self.response = main.tokenize_sequence(" 4.")

    def test_matches_an_independent_log_softmax_sum(self):
        full = self.prompt + self.response
        logits = self.model.forward(np.array([full[:-1]]))[0]
        shifted = logits - logits.max(axis=-1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        targets = full[1:]
        expected = sum(
            log_probs[i, targets[i]] for i in range(len(self.prompt) - 1, len(targets))
        )
        actual = main.compute_sequence_log_prob(
            self.model, self.prompt, self.response, MAX_SEQ_LEN
        )
        self.assertAlmostEqual(float(actual), float(expected), places=10)
        self.assertLess(float(actual), 0.0)

    def test_empty_response_scores_zero(self):
        self.assertEqual(
            main.compute_sequence_log_prob(self.model, self.prompt, [], MAX_SEQ_LEN), 0.0
        )
        self.assertEqual(
            main.compute_sequence_log_prob(self.model, [7], [], MAX_SEQ_LEN), 0.0
        )


class TestTrainingLoop(unittest.TestCase):
    def setUp(self):
        self.policy = small_gpt(seed=2)
        self.reference = small_gpt(seed=3)
        main.copy_model_weights(self.policy, self.reference)
        self.pairs = main.PREFERENCE_DATA[:2]

    def test_a_fresh_clone_carries_no_preference_signal(self):
        pair = self.pairs[0]
        prompt = main.tokenize_sequence(pair["prompt"])
        preferred = main.tokenize_sequence(pair["preferred"])
        policy_lp = main.compute_sequence_log_prob(self.policy, prompt, preferred, MAX_SEQ_LEN)
        ref_lp = main.compute_sequence_log_prob(self.reference, prompt, preferred, MAX_SEQ_LEN)
        self.assertEqual(float(policy_lp), float(ref_lp))
        accuracy = main.evaluate_preference_accuracy(
            self.policy, self.reference, self.pairs, beta=0.1, max_seq_len=MAX_SEQ_LEN
        )
        self.assertEqual(accuracy, 0.0)

    def test_training_updates_the_policy_and_freezes_the_reference(self):
        ref_before = self.reference.blocks[0].ffn.W1.copy()
        policy_before = self.policy.blocks[0].ffn.W1.copy()
        np.random.seed(9)
        with contextlib.redirect_stdout(io.StringIO()):
            trained, losses, margins = main.dpo_train(
                self.policy, self.reference, self.pairs,
                num_epochs=2, lr=5e-3, beta=0.1, max_seq_len=MAX_SEQ_LEN,
            )
        self.assertIs(trained, self.policy)
        self.assertEqual(len(losses), 2 * len(self.pairs))
        self.assertEqual(len(margins), 2 * len(self.pairs))
        self.assertTrue(np.array_equal(ref_before, self.reference.blocks[0].ffn.W1))
        self.assertFalse(np.array_equal(policy_before, self.policy.blocks[0].ffn.W1))


if __name__ == "__main__":
    unittest.main()
