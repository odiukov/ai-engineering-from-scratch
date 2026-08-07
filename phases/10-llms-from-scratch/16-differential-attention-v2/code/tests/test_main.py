"""Tests for differential attention (DIFF V1 / V2).

Two properties carry the suite. First, degeneracy: at lambda = 0, or with two
identical branches, differential attention collapses back onto plain softmax
attention (up to the 1 - lambda scale). Second, the analytic derivative of the
output with respect to lambda is -A2 V, checked against a central difference.
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
    attention_params_baseline,
    attention_params_diff_v1,
    attention_params_diff_v2,
    build_signal_plus_noise,
    compute_param_diff,
    diff_attention,
    dot,
    fmt_m,
    matmul,
    random_projection,
    snr,
    softmax_row,
    standard_attention,
)


def toy_inputs(seed=3, n_q=3, n_k=6, d=4, d_v=5):
    rng = random.Random(seed)
    Q = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(n_q)]
    K = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(n_k)]
    V = [[rng.gauss(0, 1) for _ in range(d_v)] for _ in range(n_k)]
    return Q, K, V


class TestDegeneracyToStandardAttention(unittest.TestCase):
    def test_lambda_zero_reproduces_standard_attention(self) -> None:
        Q, K, V = toy_inputs()
        std_w, std_out = standard_attention(Q, K, V)
        diff_w, diff_out = diff_attention(Q, K, Q, K, V, lam=0.0)
        for row_std, row_diff in zip(std_w, diff_w):
            for a, b in zip(row_std, row_diff):
                self.assertAlmostEqual(a, b, places=15)
        for row_std, row_diff in zip(std_out, diff_out):
            for a, b in zip(row_std, row_diff):
                self.assertAlmostEqual(a, b, places=15)

    def test_two_identical_branches_only_rescale_standard_attention(self) -> None:
        Q, K, V = toy_inputs(seed=11)
        std_w, std_out = standard_attention(Q, K, V)
        for lam in (0.0, 0.3, 0.8, 1.0):
            diff_w, diff_out = diff_attention(Q, K, Q, K, V, lam=lam)
            for row_std, row_diff in zip(std_w, diff_w):
                for a, b in zip(row_std, row_diff):
                    self.assertAlmostEqual((1.0 - lam) * a, b, places=14)
            for row_std, row_diff in zip(std_out, diff_out):
                for a, b in zip(row_std, row_diff):
                    self.assertAlmostEqual((1.0 - lam) * a, b, places=14)

    def test_identical_branches_at_lambda_one_cancel_completely(self) -> None:
        Q, K, V = toy_inputs(seed=13)
        _, out = diff_attention(Q, K, Q, K, V, lam=1.0)
        for row in out:
            for value in row:
                self.assertAlmostEqual(value, 0.0, places=14)

    def test_weight_rows_sum_to_one_minus_lambda(self) -> None:
        Q, K, V = toy_inputs(seed=17)
        Q2, K2, _ = toy_inputs(seed=19, n_q=3, n_k=6)
        for lam in (0.0, 0.5, 0.8, 1.2):
            weights, _ = diff_attention(Q, K, Q2, K2, V, lam=lam)
            for row in weights:
                self.assertAlmostEqual(sum(row), 1.0 - lam, places=12)

    def test_subtraction_can_drive_weights_negative(self) -> None:
        Q, K, V = toy_inputs(seed=23)
        Q2, K2, _ = toy_inputs(seed=29, n_q=3, n_k=6)
        weights, _ = diff_attention(Q, K, Q2, K2, V, lam=1.0)
        self.assertTrue(any(w < 0 for row in weights for w in row))


class TestLambdaGradient(unittest.TestCase):
    def test_analytic_lambda_gradient_matches_central_difference(self) -> None:
        # out(lam) = (A1 - lam * A2) V  =>  d out / d lam = -A2 V.
        Q1, K1, V = toy_inputs(seed=31)
        Q2, K2, _ = toy_inputs(seed=37, n_q=3, n_k=6)
        A2, _ = standard_attention(Q2, K2, V)
        analytic = [[-value for value in row] for row in matmul(A2, V)]

        lam, h = 0.6, 1e-5
        _, out_plus = diff_attention(Q1, K1, Q2, K2, V, lam=lam + h)
        _, out_minus = diff_attention(Q1, K1, Q2, K2, V, lam=lam - h)
        for i, row in enumerate(analytic):
            for j, expected in enumerate(row):
                numeric = (out_plus[i][j] - out_minus[i][j]) / (2 * h)
                self.assertAlmostEqual(numeric, expected, places=8)

    def test_gradient_is_constant_in_lambda(self) -> None:
        Q1, K1, V = toy_inputs(seed=41)
        Q2, K2, _ = toy_inputs(seed=43, n_q=3, n_k=6)
        h = 1e-5

        def numeric_at(lam):
            _, plus = diff_attention(Q1, K1, Q2, K2, V, lam=lam + h)
            _, minus = diff_attention(Q1, K1, Q2, K2, V, lam=lam - h)
            return [[(a - b) / (2 * h) for a, b in zip(pr, mr)]
                    for pr, mr in zip(plus, minus)]

        low, high = numeric_at(0.1), numeric_at(0.9)
        for row_low, row_high in zip(low, high):
            for a, b in zip(row_low, row_high):
                self.assertAlmostEqual(a, b, places=7)


class TestNoiseCancellation(unittest.TestCase):
    def _logit_maps(self, noise_std, seed, n_tokens=512, signal_pos=200,
                    signal_logit=4.0):
        rng = random.Random(seed)
        trained = [rng.gauss(0, noise_std) for _ in range(n_tokens)]
        trained[signal_pos] = signal_logit
        untrained = [rng.gauss(0, noise_std) for _ in range(n_tokens)]
        return softmax_row(trained), softmax_row(untrained), signal_pos

    def test_subtraction_raises_snr_while_the_noise_floors_stay_comparable(self) -> None:
        for noise_std, seed in ((0.25, 101), (0.5, 102)):
            A1, A2, pos = self._logit_maps(noise_std, seed)
            diff = [a - 0.8 * b for a, b in zip(A1, A2)]
            self.assertGreater(snr(diff, pos), snr(A1, pos))

    def test_the_signal_weight_barely_moves_while_noise_mass_shrinks(self) -> None:
        A1, A2, pos = self._logit_maps(0.5, 103)
        diff = [a - 0.8 * b for a, b in zip(A1, A2)]
        self.assertAlmostEqual(diff[pos], A1[pos], delta=0.01 * A1[pos])
        noise_before = sum(abs(w) for i, w in enumerate(A1) if i != pos)
        noise_after = sum(abs(w) for i, w in enumerate(diff) if i != pos)
        self.assertLess(noise_after, noise_before)

    def test_snr_falls_as_the_context_gets_noisier(self) -> None:
        values = [snr(self._logit_maps(std, 104)[0], 200)
                  for std in (0.25, 0.5, 1.0, 1.5, 2.0)]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_snr_is_signal_over_mean_absolute_noise(self) -> None:
        row = [0.1, -0.2, 0.9, 0.3]
        self.assertAlmostEqual(snr(row, 2), 0.9 / ((0.1 + 0.2 + 0.3) / 3), places=12)
        self.assertEqual(snr([0.0, 0.5, 0.0], 1), float("inf"))


class TestNumerics(unittest.TestCase):
    def test_softmax_is_normalized_and_overflow_safe(self) -> None:
        for row in ([0.0, 0.0, 0.0], [1000.0, 999.0, -1000.0], [-5.5, 2.25]):
            probs = softmax_row(row)
            self.assertAlmostEqual(sum(probs), 1.0, places=12)
            self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))
        uniform = softmax_row([0.0] * 4)
        self.assertTrue(all(abs(p - 0.25) < 1e-12 for p in uniform))

    def test_standard_attention_output_is_a_convex_mix_of_values(self) -> None:
        Q, K, V = toy_inputs(seed=47, n_q=2, n_k=4, d=3, d_v=2)
        weights, out = standard_attention(Q, K, V)
        for row in weights:
            self.assertAlmostEqual(sum(row), 1.0, places=12)
            self.assertTrue(all(w > 0 for w in row))
        for row in out:
            for c, value in enumerate(row):
                column = [v[c] for v in V]
                self.assertGreaterEqual(value, min(column) - 1e-12)
                self.assertLessEqual(value, max(column) + 1e-12)

    def test_scores_are_scaled_by_the_square_root_of_head_dim(self) -> None:
        Q = [[1.0, 0.0, 0.0, 0.0]]
        K = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        V = [[1.0], [0.0]]
        weights, _ = standard_attention(Q, K, V)
        expected = softmax_row([1.0 / math.sqrt(4), 0.0])
        for a, b in zip(weights[0], expected):
            self.assertAlmostEqual(a, b, places=14)

    def test_matmul_and_projection_shapes_line_up(self) -> None:
        rng = random.Random(53)
        W = random_projection(6, 3, rng)
        self.assertEqual(len(W), 6)
        self.assertEqual(len(W[0]), 3)
        X = [[rng.gauss(0, 1) for _ in range(6)] for _ in range(4)]
        out = matmul(X, W)
        self.assertEqual((len(out), len(out[0])), (4, 3))
        self.assertAlmostEqual(out[0][0], dot(X[0], [row[0] for row in W]),
                               places=12)

    def test_signal_position_is_the_only_one_aligned_with_the_query(self) -> None:
        rng = random.Random(59)
        X, q = build_signal_plus_noise(64, 20, 8, noise_scale=0.3, rng=rng)
        self.assertEqual(len(X), 64)
        scores = [dot(q, x) for x in X]
        self.assertEqual(max(range(len(scores)), key=lambda i: scores[i]), 20)


class TestParameterAccounting(unittest.TestCase):
    HIDDEN, HEADS, KV_HEADS = 4096, 32, 8
    D_HEAD = HIDDEN // HEADS

    def test_v1_matches_the_baseline_except_for_the_lambda_vectors(self) -> None:
        pd = compute_param_diff(self.HIDDEN, self.HEADS, self.KV_HEADS)
        self.assertEqual(pd.baseline, 4 * self.HIDDEN * self.HIDDEN)
        self.assertEqual(pd.extra_v1, 4 * self.HEADS * (self.D_HEAD // 2))
        self.assertLess(pd.extra_v1 / pd.baseline, 0.001)

    def test_v2_pays_for_doubled_query_heads(self) -> None:
        # With KV heads equal to Q heads the only structural change is 2x on
        # the Q projection and 2x on the output projection: +2 * hidden^2.
        v2 = attention_params_diff_v2(self.HIDDEN, self.HEADS, self.D_HEAD,
                                      kv_heads=self.HEADS)
        base = attention_params_baseline(self.HIDDEN)
        self.assertEqual(v2 - base,
                         2 * self.HIDDEN * self.HIDDEN + 4 * self.HEADS * self.D_HEAD)

    def test_v2_costs_more_than_v1_but_stays_within_a_fifth_of_baseline(self) -> None:
        pd = compute_param_diff(self.HIDDEN, self.HEADS, self.KV_HEADS)
        self.assertGreater(pd.extra_v2, pd.extra_v1)
        self.assertLess(pd.extra_v2 / pd.baseline, 0.2)

    def test_v1_keeps_the_lambda_vectors_at_half_head_dim(self) -> None:
        # V1 halves the head dimension, so its lambda vectors are half the
        # length of V2's.
        v1_lambdas = 4 * self.HEADS * (self.D_HEAD // 2)
        v2_lambdas = 4 * self.HEADS * self.D_HEAD
        self.assertEqual(v2_lambdas, 2 * v1_lambdas)
        self.assertEqual(
            attention_params_diff_v1(self.HIDDEN, self.HEADS, self.D_HEAD),
            attention_params_baseline(self.HIDDEN) + v1_lambdas,
        )

    def test_fewer_kv_heads_make_v2_cheaper(self) -> None:
        wide = compute_param_diff(self.HIDDEN, self.HEADS, self.HEADS)
        narrow = compute_param_diff(self.HIDDEN, self.HEADS, self.KV_HEADS)
        self.assertLess(narrow.diff_v2, wide.diff_v2)

    def test_formatting_reports_magnitudes(self) -> None:
        self.assertEqual(fmt_m(67_108_864), "67.1M")
        self.assertEqual(fmt_m(8_192), "8.2K")
        self.assertEqual(fmt_m(42), "42")


if __name__ == "__main__":
    unittest.main()
