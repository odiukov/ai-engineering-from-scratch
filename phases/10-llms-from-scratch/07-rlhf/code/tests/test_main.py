"""Tests for the RLHF pieces in phases/10-llms-from-scratch/07-rlhf/code/main.py.

Pins the Bradley-Terry loss against a numerical derivative of itself, the KL
penalty's zero point and shift invariance, the reward head as a linear read-out
of the final hidden state, and the reference clone that makes the KL term start
at exactly zero.
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
    spec = importlib.util.spec_from_file_location("rlhf_main", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


main = _load_lesson_main()


def causal_mask(seq_len):
    return np.triu(np.full((seq_len, seq_len), -1e9), k=1)


def last_hidden(reward_model, token_ids):
    """Everything RewardModel.forward does except the scalar projection."""
    mask = causal_mask(token_ids.shape[-1])
    x = reward_model.embedding.forward(token_ids)
    for block in reward_model.blocks:
        x = block.forward(x, mask)
    x = reward_model.ln_f.forward(x)
    return x[:, -1, :]


def small_reward_model(seed=0):
    np.random.seed(seed)
    return main.RewardModel(
        vocab_size=256, embed_dim=32, num_heads=4, num_layers=2,
        max_seq_len=64, ff_dim=64,
    )


def small_gpt(seed=0):
    np.random.seed(seed)
    return main.MiniGPT(
        vocab_size=256, embed_dim=32, num_heads=4, num_layers=2,
        max_seq_len=64, ff_dim=64,
    )


class TestBradleyTerryLoss(unittest.TestCase):
    def test_sigmoid_stays_finite_at_extreme_logits(self):
        values = np.array([-800.0, -1.0, 0.0, 1.0, 800.0])
        # np.where evaluates both branches, so the discarded one still overflows
        # at +-800; the selected branch is the finite one.
        with np.errstate(over="ignore", invalid="ignore"):
            out = main.sigmoid(values)
            mirrored = main.sigmoid(-values)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertTrue(np.all((out >= 0.0) & (out <= 1.0)))
        self.assertAlmostEqual(float(main.sigmoid(0.0)), 0.5)
        # sigmoid(-x) == 1 - sigmoid(x) is what makes the two-branch form safe.
        self.assertTrue(np.allclose(mirrored, 1.0 - out))

    def test_loss_is_log_two_when_the_pair_is_indistinguishable(self):
        loss = main.bradley_terry_loss(0.7, 0.7)
        self.assertAlmostEqual(float(loss), float(np.log(2)), places=6)

    def test_loss_falls_as_the_preferred_margin_grows(self):
        margins = [-2.0, -0.5, 0.0, 0.5, 2.0, 6.0]
        losses = [float(main.bradley_terry_loss(m, 0.0)) for m in margins]
        for earlier, later in zip(losses, losses[1:]):
            self.assertGreater(earlier, later)
        self.assertGreater(losses[0], 2.0)
        self.assertLess(losses[-1], 0.01)

    def test_gradient_formula_matches_central_difference(self):
        # train_reward_model uses grad = sigmoid(diff) - 1 as d(loss)/d(diff).
        step = 1e-5
        for diff in (-3.0, -0.4, 0.0, 0.9, 2.5):
            analytic = float(main.sigmoid(diff) - 1.0)
            numeric = float(
                main.bradley_terry_loss(diff + step, 0.0)
                - main.bradley_terry_loss(diff - step, 0.0)
            ) / (2 * step)
            self.assertAlmostEqual(numeric, analytic, places=6)
            self.assertLess(analytic, 0.0)


class TestKLPenalty(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.logits = rng.normal(size=(1, 6, 16))

    def test_kl_is_zero_for_identical_policies(self):
        self.assertAlmostEqual(
            float(main.compute_kl_divergence(self.logits, self.logits)), 0.0, places=12
        )

    def test_kl_ignores_a_constant_logit_shift(self):
        shifted = self.logits + 7.5
        self.assertAlmostEqual(
            float(main.compute_kl_divergence(self.logits, shifted)), 0.0, places=10
        )

    def test_kl_grows_with_policy_divergence(self):
        divergences = [
            float(main.compute_kl_divergence(self.logits * (1 + s), self.logits))
            for s in (0.1, 0.5, 1.0, 2.0)
        ]
        for value in divergences:
            self.assertGreaterEqual(value, 0.0)
        for earlier, later in zip(divergences, divergences[1:]):
            self.assertGreater(later, earlier)

    def test_reference_clone_starts_the_penalty_at_zero(self):
        policy = small_gpt(seed=3)
        reference = small_gpt(seed=4)
        token_ids = np.array([[5, 9, 12, 40, 7]])
        self.assertGreater(
            float(main.compute_kl_divergence(policy.forward(token_ids),
                                             reference.forward(token_ids))),
            0.0,
        )
        main.copy_model_weights(policy, reference)
        self.assertTrue(
            np.array_equal(policy.forward(token_ids), reference.forward(token_ids))
        )
        self.assertEqual(
            float(main.compute_kl_divergence(policy.forward(token_ids),
                                             reference.forward(token_ids))),
            0.0,
        )


class TestRewardModel(unittest.TestCase):
    def test_reward_is_a_linear_readout_of_the_last_hidden_state(self):
        rm = small_reward_model(seed=1)
        token_ids = np.array([main.tokenize_for_reward("What is 2 + 2?", "4")])
        hidden = last_hidden(rm, token_ids)
        self.assertTrue(
            np.allclose(rm.forward(token_ids), hidden @ rm.reward_head)
        )

    def test_reward_head_can_separate_a_preference_pair(self):
        rm = small_reward_model(seed=2)
        pair = main.PREFERENCE_DATA[0]
        preferred = np.array([main.tokenize_for_reward(pair["prompt"], pair["preferred"])[:64]])
        rejected = np.array([main.tokenize_for_reward(pair["prompt"], pair["rejected"])[:64]])

        # The head is the only trainable direction, so pointing it along
        # (h_preferred - h_rejected) has to score the preferred answer higher.
        rm.reward_head = (last_hidden(rm, preferred) - last_hidden(rm, rejected)).flatten()
        r_preferred = float(rm.forward(preferred)[0])
        r_rejected = float(rm.forward(rejected)[0])

        self.assertGreater(r_preferred, r_rejected)
        self.assertLess(float(main.bradley_terry_loss(r_preferred, r_rejected)), np.log(2))

    def test_scores_do_not_leak_across_batch_rows(self):
        rm = small_reward_model(seed=5)
        row_a = main.tokenize_for_reward("Name a color.", "Blue.")
        row_b = main.tokenize_for_reward("Name a fruit.", "Pear.")
        self.assertEqual(len(row_a), len(row_b))
        batched = rm.forward(np.array([row_a, row_b]))
        alone_a = rm.forward(np.array([row_a]))
        alone_b = rm.forward(np.array([row_b]))
        self.assertEqual(batched.shape, (2,))
        self.assertTrue(np.allclose(batched[0], alone_a[0]))
        self.assertTrue(np.allclose(batched[1], alone_b[0]))

    def test_tokenizer_joins_prompt_and_response_with_a_separator(self):
        prompt, response = "Hi?", "Yes."
        tokens = main.tokenize_for_reward(prompt, response, vocab_size=256)
        prompt_len = len(prompt.encode("utf-8"))
        self.assertEqual(tokens[:prompt_len], list(prompt.encode("utf-8")))
        self.assertEqual(tokens[prompt_len], 0)
        self.assertEqual(tokens[prompt_len + 1:], list(response.encode("utf-8")))
        self.assertTrue(all(0 <= t < 256 for t in tokens))

    def test_training_reports_one_metric_per_epoch_and_moves_the_head(self):
        rm = small_reward_model(seed=6)
        before = rm.reward_head.copy()
        np.random.seed(7)
        with contextlib.redirect_stdout(io.StringIO()):
            trained, losses, accuracies = main.train_reward_model(
                rm, main.PREFERENCE_DATA[:3], num_epochs=3, lr=1e-3, max_seq_len=64
            )
        self.assertIs(trained, rm)
        self.assertEqual(len(losses), 3)
        self.assertEqual(len(accuracies), 3)
        self.assertTrue(all(0.0 <= a <= 1.0 for a in accuracies))
        self.assertFalse(np.array_equal(before, rm.reward_head))


if __name__ == "__main__":
    unittest.main()
