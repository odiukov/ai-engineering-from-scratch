"""Tests for the from-scratch mini GPT.

Covers: the causal mask actually forbidding lookahead, LayerNorm statistics,
numerically stable cross-entropy, analytic LayerNorm/FFN gradients checked
against central differences, the parameter count, and loss going down when the
model overfits a single fixed batch.
Reference: ../../docs/en.md
"""

import io
import os
import re
import sys
import unittest
from contextlib import redirect_stdout

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from main import (  # noqa: E402
    FeedForward,
    LayerNorm,
    MiniGPT,
    MultiHeadAttention,
    cross_entropy_loss,
    ffn_backward,
    generate,
    layernorm_backward,
    parameter_breakdown,
    train_mini_gpt,
)

TINY = dict(
    vocab_size=16,
    embed_dim=8,
    num_heads=2,
    num_layers=2,
    max_seq_len=6,
    ff_dim=16,
)


def tiny_model(seed=0):
    np.random.seed(seed)
    return MiniGPT(**TINY)


def numerical_gradient(loss_fn, array, eps=1e-6):
    """Central-difference gradient of loss_fn() with respect to array in place."""
    grad = np.zeros_like(array)
    it = np.nditer(array, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        original = array[idx]
        array[idx] = original + eps
        plus = loss_fn()
        array[idx] = original - eps
        minus = loss_fn()
        array[idx] = original
        grad[idx] = (plus - minus) / (2 * eps)
        it.iternext()
    return grad


class TestCausalMask(unittest.TestCase):
    def test_changing_a_future_token_leaves_earlier_logits_alone(self):
        model = tiny_model()
        base = np.array([[1, 2, 3, 4]])
        changed = np.array([[1, 2, 3, 9]])
        left = model.forward(base)
        right = model.forward(changed)
        np.testing.assert_allclose(left[:, :3, :], right[:, :3, :], rtol=0, atol=0)
        self.assertFalse(np.allclose(left[:, 3, :], right[:, 3, :]))

    def test_permuting_future_tokens_leaves_earlier_logits_alone(self):
        model = tiny_model()
        original = model.forward(np.array([[5, 6, 7, 8]]))
        permuted = model.forward(np.array([[5, 6, 8, 7]]))
        np.testing.assert_allclose(original[:, :2, :], permuted[:, :2, :], atol=0)

    def test_a_prefix_predicts_the_same_way_inside_a_longer_sequence(self):
        model = tiny_model()
        short = model.forward(np.array([[1, 2, 3]]))
        long = model.forward(np.array([[1, 2, 3, 4, 5]]))
        np.testing.assert_allclose(short, long[:, :3, :], rtol=1e-10, atol=1e-12)

    def test_masked_attention_stays_finite(self):
        np.random.seed(1)
        attn = MultiHeadAttention(8, 2)
        x = np.random.randn(1, 4, 8)
        mask = np.triu(np.full((4, 4), -1e9), k=1)
        out = attn.forward(x, mask)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_head_split_must_divide_the_embedding(self):
        with self.assertRaises(AssertionError):
            MultiHeadAttention(8, 3)


class TestLayerNorm(unittest.TestCase):
    def test_output_is_zero_mean_and_unit_variance_per_position(self):
        np.random.seed(2)
        ln = LayerNorm(6)
        x = np.random.randn(2, 3, 6) * 7.0 + 4.0
        y = ln.forward(x)
        np.testing.assert_allclose(y.mean(axis=-1), 0.0, atol=1e-12)
        np.testing.assert_allclose(y.var(axis=-1), 1.0, atol=1e-4)

    def test_gamma_and_beta_apply_affinely(self):
        np.random.seed(3)
        ln = LayerNorm(4)
        x = np.random.randn(1, 2, 4)
        baseline = ln.forward(x)
        ln.gamma = np.full(4, 3.0)
        ln.beta = np.full(4, -1.0)
        np.testing.assert_allclose(ln.forward(x), 3.0 * baseline - 1.0, atol=1e-12)

    def test_constant_input_does_not_divide_by_zero(self):
        ln = LayerNorm(4)
        out = ln.forward(np.ones((1, 2, 4)))
        self.assertTrue(np.all(np.isfinite(out)))
        np.testing.assert_allclose(out, 0.0, atol=1e-12)


class TestCrossEntropy(unittest.TestCase):
    def test_uniform_logits_give_log_vocab_size(self):
        logits = np.zeros((1, 3, 10))
        targets = np.array([[0, 1, 2]])
        self.assertAlmostEqual(cross_entropy_loss(logits, targets), np.log(10), places=12)

    def test_confident_and_correct_costs_almost_nothing(self):
        logits = np.zeros((1, 2, 5))
        logits[0, 0, 3] = 40.0
        logits[0, 1, 1] = 40.0
        loss = cross_entropy_loss(logits, np.array([[3, 1]]))
        self.assertLess(loss, 1e-8)

    def test_large_logits_stay_finite(self):
        logits = np.array([[[1e5, -1e5, 0.0, 5e4]]])
        loss = cross_entropy_loss(logits, np.array([[2]]))
        self.assertTrue(np.isfinite(loss))
        self.assertAlmostEqual(loss, 1e5, places=4)

    def test_loss_is_invariant_to_a_constant_logit_shift(self):
        np.random.seed(4)
        logits = np.random.randn(2, 3, 7)
        targets = np.random.randint(0, 7, size=(2, 3))
        base = cross_entropy_loss(logits, targets)
        shifted = cross_entropy_loss(logits + 1234.5, targets)
        self.assertAlmostEqual(base, shifted, places=10)


class TestLayerNormBackward(unittest.TestCase):
    def setUp(self):
        np.random.seed(5)
        self.ln = LayerNorm(5)
        self.ln.gamma = np.random.randn(5)
        self.ln.beta = np.random.randn(5)
        self.x = np.random.randn(2, 3, 5)
        self.dy = np.random.randn(2, 3, 5)
        self.dx, self.grad_gamma, self.grad_beta = layernorm_backward(
            self.dy, self.x, self.ln
        )

    def _loss(self):
        return float(np.sum(self.dy * self.ln.forward(self.x)))

    def test_input_gradient_matches_central_differences(self):
        np.testing.assert_allclose(
            self.dx, numerical_gradient(self._loss, self.x), rtol=1e-5, atol=1e-7
        )

    def test_gamma_gradient_matches_central_differences(self):
        np.testing.assert_allclose(
            self.grad_gamma,
            numerical_gradient(self._loss, self.ln.gamma),
            rtol=1e-5,
            atol=1e-7,
        )

    def test_beta_gradient_matches_central_differences(self):
        np.testing.assert_allclose(
            self.grad_beta,
            numerical_gradient(self._loss, self.ln.beta),
            rtol=1e-5,
            atol=1e-7,
        )


class TestFeedForwardBackward(unittest.TestCase):
    def setUp(self):
        np.random.seed(6)
        self.ffn = FeedForward(4, 7)
        self.ffn.W1 = np.random.randn(4, 7)
        self.ffn.W2 = np.random.randn(7, 4)
        self.ffn.b1 = np.random.randn(7)
        self.ffn.b2 = np.random.randn(4)
        self.x = np.random.randn(2, 3, 4)
        self.dy = np.random.randn(2, 3, 4)
        (
            self.dx,
            self.grad_W1,
            self.grad_b1,
            self.grad_W2,
            self.grad_b2,
        ) = ffn_backward(self.dy, self.x, self.ffn)

    def _loss(self):
        return float(np.sum(self.dy * self.ffn.forward(self.x)))

    def test_input_gradient_matches_central_differences(self):
        np.testing.assert_allclose(
            self.dx, numerical_gradient(self._loss, self.x), rtol=1e-5, atol=1e-7
        )

    def test_weight_gradients_match_central_differences(self):
        for analytic, param in (
            (self.grad_W1, self.ffn.W1),
            (self.grad_W2, self.ffn.W2),
        ):
            np.testing.assert_allclose(
                analytic, numerical_gradient(self._loss, param), rtol=1e-5, atol=1e-7
            )

    def test_bias_gradients_match_central_differences(self):
        for analytic, param in (
            (self.grad_b1, self.ffn.b1),
            (self.grad_b2, self.ffn.b2),
        ):
            np.testing.assert_allclose(
                analytic, numerical_gradient(self._loss, param), rtol=1e-5, atol=1e-7
            )

    def test_relu_blocks_gradient_on_dead_units(self):
        np.random.seed(7)
        ffn = FeedForward(3, 5)
        ffn.W1 = np.zeros((3, 5))
        ffn.b1 = np.full(5, -1.0)  # every pre-activation is negative
        ffn.W2 = np.random.randn(5, 3)
        x = np.random.randn(1, 2, 3)
        dy = np.random.randn(1, 2, 3)
        dx, grad_W1, grad_b1, _, _ = ffn_backward(dy, x, ffn)
        np.testing.assert_allclose(dx, 0.0, atol=1e-12)
        np.testing.assert_allclose(grad_W1, 0.0, atol=1e-12)
        np.testing.assert_allclose(grad_b1, 0.0, atol=1e-12)


class TestParameterCount(unittest.TestCase):
    def test_count_matches_the_closed_form_used_in_the_breakdown(self):
        vocab, dim, layers, seq_len, ff = (
            TINY["vocab_size"],
            TINY["embed_dim"],
            TINY["num_layers"],
            TINY["max_seq_len"],
            TINY["ff_dim"],
        )
        per_block = 4 * dim * dim + (2 * dim * ff + dim + ff) + 4 * dim
        expected = vocab * dim + seq_len * dim + layers * per_block + 2 * dim
        self.assertEqual(tiny_model().count_parameters(), expected)

    def test_breakdown_reports_the_known_gpt2_small_size(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            parameter_breakdown()
        line = next(l for l in buf.getvalue().splitlines() if "GPT-2 Small" in l)
        self.assertIn("124,", line.replace(" ", ""))


class TestGeneration(unittest.TestCase):
    def test_generate_appends_exactly_the_requested_tokens(self):
        model = tiny_model()
        np.random.seed(8)
        prompt = [1, 2]
        out = generate(model, prompt, max_new_tokens=5, temperature=0.9)
        self.assertEqual(len(out), len(prompt) + 5)
        self.assertEqual(out[: len(prompt)], prompt)
        self.assertTrue(all(0 <= t < TINY["vocab_size"] for t in out))

    def test_generation_survives_a_prompt_longer_than_the_context(self):
        model = tiny_model()
        np.random.seed(9)
        prompt = list(range(TINY["max_seq_len"] + 4))
        prompt = [t % TINY["vocab_size"] for t in prompt]
        out = generate(model, prompt, max_new_tokens=3)
        self.assertEqual(len(out), len(prompt) + 3)

    def test_generation_is_reproducible_under_a_fixed_seed(self):
        model = tiny_model()
        np.random.seed(10)
        first = generate(model, [1, 2], max_new_tokens=4)
        np.random.seed(10)
        second = generate(model, [1, 2], max_new_tokens=4)
        self.assertEqual(first, second)


class TestTrainingLoop(unittest.TestCase):
    def test_loss_falls_when_overfitting_one_fixed_batch(self):
        # 18 bytes with seq_len 16 leaves a single legal window, so every step
        # sees the same batch.
        text = "abcabcabcabcabcabc"
        seq_len = 16
        self.assertLessEqual(len(text.encode("utf-8")), seq_len + 2)

        np.random.seed(11)
        buf = io.StringIO()
        with redirect_stdout(buf):
            train_mini_gpt(
                text,
                vocab_size=256,
                embed_dim=32,
                num_heads=2,
                num_layers=2,
                seq_len=seq_len,
                num_steps=41,
                lr=1e-2,
            )
        losses = [float(m) for m in re.findall(r"Loss: ([0-9.]+)", buf.getvalue())]
        self.assertEqual(len(losses), 3)
        self.assertEqual(losses, sorted(losses, reverse=True))
        self.assertLess(losses[-1], losses[0] / 2)

    def test_training_reports_the_parameter_count_it_will_update(self):
        np.random.seed(12)
        buf = io.StringIO()
        with redirect_stdout(buf):
            model = train_mini_gpt(
                "abcabcabcabcabcabc",
                vocab_size=256,
                embed_dim=32,
                num_heads=2,
                num_layers=2,
                seq_len=16,
                num_steps=1,
                lr=1e-2,
            )
        self.assertIn(f"{model.count_parameters():,}", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
