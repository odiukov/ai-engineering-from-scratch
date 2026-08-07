"""Tests for speculative decoding with N-token drafts and KV rollback.

The load-bearing test is the Leviathan invariant: the token the loop emits is
distributed exactly as the verifier's q, for any draft p — a good one, a bad
one, or one that is actively wrong. Everything else (alpha, expected tokens
per forward, KV bookkeeping) only matters once that holds.
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
    KVBuffer,
    chi_square,
    expected_tokens_per_verify,
    kl,
    measure_alpha,
    perturb,
    residual,
    sample,
    spec_step,
    wall_time_per_token,
)


Q = [0.30, 0.22, 0.15, 0.10, 0.08, 0.07, 0.05, 0.03]


def normalized(values: list[float]) -> list[float]:
    total = sum(values)
    return [v / total for v in values]


def empirical_first_token(q, p, trials, seed):
    rng = random.Random(seed)
    counts = [0] * len(q)
    for _ in range(trials):
        tokens, _ = spec_step(q, p, N=1, kv=KVBuffer(), rng=rng)
        counts[tokens[0]] += 1
    return [c / trials for c in counts]


def total_variation(a, b):
    return 0.5 * sum(abs(x - y) for x, y in zip(a, b))


class TestLeviathanEquivalence(unittest.TestCase):
    """The first emitted token is distributed as q whatever the draft does."""

    TRIALS = 40_000
    TOLERANCE = 0.02

    def test_accept_plus_residual_reconstructs_the_verifier_exactly(self) -> None:
        # Algebraic form of the theorem: P(emit x) = min(p,q)(x) + P(reject) *
        # residual(x), and that must equal q(x) term by term.
        for p in (normalized([1.0] * len(Q)), list(reversed(Q)),
                  perturb(Q, 0.03, random.Random(5))):
            overlap = sum(min(pi, qi) for pi, qi in zip(p, Q))
            res = residual(Q, p)
            for x, q_x in enumerate(Q):
                emitted = min(p[x], Q[x]) + (1.0 - overlap) * res[x]
                self.assertAlmostEqual(emitted, q_x, places=12)

    def test_near_perfect_draft_emits_the_verifier_distribution(self) -> None:
        p = perturb(Q, amount=0.005, rng=random.Random(1))
        observed = empirical_first_token(Q, p, self.TRIALS, seed=101)
        self.assertLess(total_variation(observed, Q), self.TOLERANCE)

    def test_poor_draft_emits_the_verifier_distribution_too(self) -> None:
        p = perturb(Q, amount=0.08, rng=random.Random(3))
        observed = empirical_first_token(Q, p, self.TRIALS, seed=102)
        self.assertLess(total_variation(observed, Q), self.TOLERANCE)

    def test_adversarial_reversed_draft_still_emits_the_verifier(self) -> None:
        p = normalized(list(reversed(Q)))
        observed = empirical_first_token(Q, p, self.TRIALS, seed=103)
        self.assertLess(total_variation(observed, Q), self.TOLERANCE)

    def test_uniform_draft_still_emits_the_verifier(self) -> None:
        p = normalized([1.0] * len(Q))
        observed = empirical_first_token(Q, p, self.TRIALS, seed=104)
        self.assertLess(total_variation(observed, Q), self.TOLERANCE)

    def test_chi_square_against_direct_sampling_clears_the_critical_value(self) -> None:
        rng = random.Random(202)
        spec_counts = [0] * len(Q)
        direct_counts = [0] * len(Q)
        p = perturb(Q, amount=0.02, rng=random.Random(2))
        for _ in range(20_000):
            tokens, _ = spec_step(Q, p, N=1, kv=KVBuffer(), rng=rng)
            spec_counts[tokens[0]] += 1
            direct_counts[sample(Q, rng)] += 1
        self.assertLess(chi_square(spec_counts, direct_counts), 14.07)


class TestResidual(unittest.TestCase):
    def test_residual_is_a_distribution_supported_where_q_exceeds_p(self) -> None:
        p = perturb(Q, amount=0.05, rng=random.Random(9))
        res = residual(Q, p)
        self.assertAlmostEqual(sum(res), 1.0, places=12)
        for i, r in enumerate(res):
            self.assertGreaterEqual(r, 0.0)
            if Q[i] <= p[i]:
                self.assertEqual(r, 0.0)

    def test_identical_distributions_fall_back_to_q(self) -> None:
        self.assertEqual(residual(Q, list(Q)), list(Q))


class TestSpecStep(unittest.TestCase):
    def test_emitted_count_stays_between_one_and_n_plus_one(self) -> None:
        rng = random.Random(7)
        p = perturb(Q, amount=0.05, rng=random.Random(4))
        for n in (1, 2, 5, 8):
            for _ in range(200):
                tokens, forwards = spec_step(Q, p, N=n, kv=KVBuffer(), rng=rng)
                self.assertGreaterEqual(len(tokens), 1)
                self.assertLessEqual(len(tokens), n + 1)
                self.assertEqual(forwards, 1)

    def test_one_verifier_forward_serves_the_whole_draft(self) -> None:
        rng = random.Random(11)
        p = perturb(Q, amount=0.05, rng=random.Random(4))
        emitted = forwards = 0
        for _ in range(500):
            tokens, used = spec_step(Q, p, N=5, kv=KVBuffer(), rng=rng)
            emitted += len(tokens)
            forwards += used
        self.assertEqual(forwards, 500)
        self.assertGreater(emitted / forwards, 1.0)

    def test_a_perfect_draft_never_rejects_and_always_takes_the_bonus(self) -> None:
        rng = random.Random(13)
        for _ in range(200):
            tokens, _ = spec_step(Q, list(Q), N=4, kv=KVBuffer(), rng=rng)
            self.assertEqual(len(tokens), 5)

    def test_kv_length_always_equals_the_accepted_prefix(self) -> None:
        rng = random.Random(17)
        p = perturb(Q, amount=0.06, rng=random.Random(6))
        kv = KVBuffer(length=12)
        for _ in range(400):
            before = kv.length
            tokens, _ = spec_step(Q, p, N=5, kv=kv, rng=rng)
            self.assertEqual(kv.length, before + len(tokens))

    def test_rollback_discards_the_speculated_tail(self) -> None:
        # A draft that can only be rejected: p puts all its mass on the token
        # q is least likely to keep.
        p = [0.0] * len(Q)
        p[7] = 1.0
        kv = KVBuffer(length=100)
        rng = random.Random(19)
        tokens, _ = spec_step(Q, p, N=6, kv=kv, rng=rng)
        self.assertLessEqual(len(tokens), 7)
        self.assertEqual(kv.length, 100 + len(tokens))


class TestAcceptanceRate(unittest.TestCase):
    def test_alpha_rises_as_the_draft_moves_closer_to_the_verifier(self) -> None:
        drafts = [
            ("vanilla", perturb(Q, 0.08, random.Random(3))),
            ("eagle-1", perturb(Q, 0.02, random.Random(2))),
            ("eagle-3", perturb(Q, 0.005, random.Random(1))),
        ]
        divergences = [kl(Q, p) for _, p in drafts]
        alphas = [measure_alpha(Q, p, 20_000, random.Random(7)) for _, p in drafts]
        self.assertEqual(divergences, sorted(divergences, reverse=True))
        self.assertEqual(alphas, sorted(alphas))
        self.assertGreater(alphas[-1], 0.95)

    def test_alpha_equals_the_overlap_between_draft_and_verifier(self) -> None:
        # E[accept] = sum_x p(x) * min(1, q(x)/p(x)) = sum_x min(p(x), q(x)).
        p = perturb(Q, amount=0.04, rng=random.Random(8))
        overlap = sum(min(pi, qi) for pi, qi in zip(p, Q))
        measured = measure_alpha(Q, p, 60_000, random.Random(23))
        self.assertAlmostEqual(measured, overlap, delta=0.01)

    def test_a_perfect_draft_is_always_accepted(self) -> None:
        self.assertEqual(measure_alpha(Q, list(Q), 2_000, random.Random(29)), 1.0)


class TestSpeedupMath(unittest.TestCase):
    def test_expected_tokens_is_the_geometric_series_through_the_bonus(self) -> None:
        for alpha in (0.55, 0.7, 0.8, 0.9, 0.95):
            for n in (1, 3, 5, 7, 10):
                series = sum(alpha ** i for i in range(n + 1))
                self.assertAlmostEqual(
                    expected_tokens_per_verify(alpha, n), series, places=10
                )

    def test_boundary_acceptance_rates_are_handled(self) -> None:
        self.assertEqual(expected_tokens_per_verify(1.0, 5), 6)
        self.assertEqual(expected_tokens_per_verify(0.0, 5), 1.0)

    def test_expected_tokens_grows_with_alpha_and_with_n(self) -> None:
        for n in (1, 5, 10):
            values = [expected_tokens_per_verify(a, n) for a in (0.5, 0.7, 0.9)]
            self.assertEqual(values, sorted(values))
        for alpha in (0.5, 0.9):
            values = [expected_tokens_per_verify(alpha, n) for n in (1, 3, 5, 10)]
            self.assertEqual(values, sorted(values))

    def test_expected_tokens_saturates_at_the_one_over_one_minus_alpha_ceiling(self) -> None:
        alpha = 0.8
        ceiling = 1 / (1 - alpha)
        self.assertLess(expected_tokens_per_verify(alpha, 50), ceiling)
        self.assertAlmostEqual(
            expected_tokens_per_verify(alpha, 50), ceiling, places=3
        )

    def test_high_alpha_beats_the_no_speculation_baseline(self) -> None:
        self.assertLess(wall_time_per_token(0.9, 5, c=0.04), 1.0)
        self.assertLess(wall_time_per_token(0.95, 5, c=0.04), 1.0)

    def test_a_costly_draft_with_a_bad_acceptance_rate_loses_to_the_baseline(self) -> None:
        self.assertGreater(wall_time_per_token(0.1, 5, c=0.5), 1.0)

    def test_wall_time_is_unimodal_in_n_so_a_best_draft_length_exists(self) -> None:
        alpha, c = 0.8, 0.05
        times = [wall_time_per_token(alpha, n, c) for n in range(1, 30)]
        best = times.index(min(times))
        self.assertEqual(times[: best + 1], sorted(times[: best + 1], reverse=True))
        self.assertEqual(times[best:], sorted(times[best:]))

    def test_cheaper_drafts_justify_longer_speculation(self) -> None:
        def best_n(alpha, c):
            times = [wall_time_per_token(alpha, n, c) for n in range(1, 60)]
            return times.index(min(times)) + 1

        self.assertGreater(best_n(0.95, 0.02), best_n(0.8, 0.05))


class TestDivergence(unittest.TestCase):
    def test_kl_is_zero_only_when_the_draft_equals_the_verifier(self) -> None:
        self.assertAlmostEqual(kl(Q, list(Q)), 0.0, places=12)
        self.assertGreater(kl(Q, perturb(Q, 0.05, random.Random(31))), 0.0)

    def test_perturb_returns_a_valid_distribution(self) -> None:
        p = perturb(Q, amount=0.1, rng=random.Random(37))
        self.assertAlmostEqual(sum(p), 1.0, places=12)
        self.assertTrue(all(x > 0 for x in p))

    def test_larger_perturbation_moves_the_draft_further_away(self) -> None:
        near = kl(Q, perturb(Q, 0.005, random.Random(41)))
        far = kl(Q, perturb(Q, 0.08, random.Random(41)))
        self.assertGreater(far, near)

    def test_chi_square_of_a_distribution_against_itself_is_zero(self) -> None:
        counts = [300, 220, 150, 100, 80, 70, 50, 30]
        self.assertAlmostEqual(chi_square(counts, counts), 0.0, places=12)
        self.assertEqual(chi_square([], []), 0.0)


class TestSampling(unittest.TestCase):
    def test_sample_respects_the_cumulative_distribution(self) -> None:
        class FixedRng:
            def __init__(self, u): self.u = u
            def random(self): return self.u

        self.assertEqual(sample(Q, FixedRng(0.0)), 0)
        self.assertEqual(sample(Q, FixedRng(0.29)), 0)
        self.assertEqual(sample(Q, FixedRng(0.31)), 1)
        self.assertEqual(sample(Q, FixedRng(0.999999)), len(Q) - 1)

    def test_sample_never_leaves_the_support(self) -> None:
        rng = random.Random(43)
        for _ in range(1000):
            self.assertIn(sample(Q, rng), range(len(Q)))

    def test_empirical_mean_of_direct_sampling_matches_q(self) -> None:
        rng = random.Random(47)
        counts = [0] * len(Q)
        for _ in range(20_000):
            counts[sample(Q, rng)] += 1
        observed = [c / 20_000 for c in counts]
        self.assertLess(total_variation(observed, Q), 0.02)

    def test_math_import_is_used_for_log_based_divergence(self) -> None:
        self.assertAlmostEqual(kl([1.0, 0.0], [0.5, 0.5]), math.log(2.0), places=12)


if __name__ == "__main__":
    unittest.main()
