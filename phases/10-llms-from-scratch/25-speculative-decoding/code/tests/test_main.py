"""Tests for the speculative-decoding harness.

The property that matters is exactness: the accept/reject rule must reproduce
the target's next-token distribution at every acceptance rate and every draft
length, including the degenerate cases of a perfect draft (all K+1 tokens
accepted) and a draft with disjoint support (always rejected, corrected from
the residual). The tree tests pin that a verified node attends to its
ancestors and nothing else.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from main import (  # noqa: E402
    _positive_int,
    _unit_float,
    build_tree,
    empirical_dist,
    expected_tokens,
    make_draft,
    make_target,
    measure_alpha,
    measure_throughput,
    speculative_step,
    total_variation,
    tree_attention_mask,
    validate_tree_mask,
)


def make_pair(vocab: int, alpha_hint: float, seed: int):
    rng = np.random.default_rng(seed)
    target = make_target(vocab, rng)
    draft = make_draft(target, alpha_hint, rng)
    return target, draft


class TestExactness(unittest.TestCase):
    def test_first_token_matches_target_at_every_acceptance_rate(self) -> None:
        """Whatever the draft quality, the token the round emits first is
        distributed exactly as a plain target sample would be."""
        vocab, n_samples = 8, 4000
        for alpha_hint, k in ((0.1, 1), (0.5, 4), (0.95, 4)):
            with self.subTest(alpha_hint=alpha_hint, K=k):
                target, draft = make_pair(vocab, alpha_hint, seed=11)
                rng = np.random.default_rng(202)
                spec = [speculative_step(target, draft, k, rng)[0]
                        for _ in range(n_samples)]
                tv = total_variation(empirical_dist(spec, vocab), target)
                self.assertLess(tv, 0.05)

    def test_speculative_sampling_is_no_further_from_target_than_plain(self) -> None:
        vocab, n_samples = 8, 4000
        target, draft = make_pair(vocab, 0.4, seed=3)
        rng = np.random.default_rng(5)
        plain = rng.choice(vocab, size=n_samples, p=target).tolist()
        spec = [speculative_step(target, draft, 3, rng)[0] for _ in range(n_samples)]
        tv_plain = total_variation(empirical_dist(plain, vocab), target)
        tv_spec = total_variation(empirical_dist(spec, vocab), target)
        self.assertLess(tv_spec, tv_plain + 0.03)

    def test_perfect_draft_accepts_the_whole_block_plus_a_bonus_token(self) -> None:
        target, _ = make_pair(16, 0.9, seed=1)
        rng = np.random.default_rng(0)
        for k in (1, 3, 7):
            with self.subTest(K=k):
                for _ in range(20):
                    self.assertEqual(len(speculative_step(target, target, k, rng)), k + 1)

    def test_disjoint_draft_is_always_corrected_from_the_residual(self) -> None:
        target = np.array([0.0, 0.5, 0.5])
        draft = np.array([1.0, 0.0, 0.0])
        rng = np.random.default_rng(9)
        for _ in range(50):
            tokens = speculative_step(target, draft, 4, rng)
            self.assertEqual(len(tokens), 1)
            self.assertIn(tokens[0], (1, 2))

    def test_a_round_never_emits_more_than_k_plus_one_tokens(self) -> None:
        target, draft = make_pair(12, 0.6, seed=4)
        rng = np.random.default_rng(6)
        for k in (1, 2, 5):
            for _ in range(50):
                length = len(speculative_step(target, draft, k, rng))
                with self.subTest(K=k):
                    self.assertGreaterEqual(length, 1)
                    self.assertLessEqual(length, k + 1)

    def test_identical_distributions_accept_everything(self) -> None:
        target, _ = make_pair(16, 0.5, seed=2)
        rng = np.random.default_rng(7)
        self.assertEqual(measure_alpha(target, target, 500, rng), 1.0)


class TestDraftQuality(unittest.TestCase):
    def test_distributions_are_normalized(self) -> None:
        target, draft = make_pair(32, 0.75, seed=8)
        self.assertAlmostEqual(float(target.sum()), 1.0)
        self.assertAlmostEqual(float(draft.sum()), 1.0)
        self.assertTrue(bool((draft > 0).all()))

    def test_higher_hint_puts_the_draft_closer_to_the_target(self) -> None:
        target, close = make_pair(32, 0.95, seed=10)
        _, far = make_draft(target, 0.05, np.random.default_rng(10)), None
        self.assertLess(total_variation(close, target), total_variation(far, target))

    def test_measured_acceptance_tracks_the_draft_hint(self) -> None:
        alphas = []
        for hint in (0.1, 0.5, 0.9):
            target, draft = make_pair(32, hint, seed=13)
            alphas.append(measure_alpha(target, draft, 3000,
                                        np.random.default_rng(21)))
        self.assertEqual(alphas, sorted(alphas))
        self.assertLess(alphas[0], alphas[-1])


class TestThroughputModel(unittest.TestCase):
    def test_expected_tokens_is_the_truncated_geometric_sum(self) -> None:
        for alpha in (0.3, 0.5, 0.8):
            for k in (1, 4, 8):
                with self.subTest(alpha=alpha, K=k):
                    self.assertAlmostEqual(
                        expected_tokens(alpha, k),
                        sum(alpha ** i for i in range(k + 1)),
                        places=9,
                    )

    def test_boundary_acceptance_rates(self) -> None:
        self.assertAlmostEqual(expected_tokens(0.0, 4), 1.0)
        self.assertAlmostEqual(expected_tokens(1.0, 4), 5.0)
        self.assertAlmostEqual(expected_tokens(0.8, 4), 3.3616, places=4)

    def test_expected_tokens_rises_with_alpha_and_with_k(self) -> None:
        by_alpha = [expected_tokens(a, 4) for a in (0.1, 0.3, 0.5, 0.7, 0.9)]
        self.assertEqual(by_alpha, sorted(by_alpha))
        by_k = [expected_tokens(0.7, k) for k in (1, 2, 4, 8)]
        self.assertEqual(by_k, sorted(by_k))

    def test_measured_throughput_matches_the_formula(self) -> None:
        target, draft = make_pair(16, 0.7, seed=14)
        rng = np.random.default_rng(31)
        alpha = measure_alpha(target, draft, 4000, rng)
        measured = measure_throughput(target, draft, 4, 2000, rng)
        self.assertAlmostEqual(measured, expected_tokens(alpha, 4), delta=0.25)


class TestTreeDrafting(unittest.TestCase):
    def test_node_count_follows_the_branch_factors(self) -> None:
        self.assertEqual(len(build_tree((3, 2, 2))), 1 + 3 + 6 + 12)
        self.assertEqual(len(build_tree((2, 2))), 1 + 2 + 4)
        self.assertEqual(len(build_tree(())), 1)

    def test_every_node_attends_to_exactly_its_ancestors(self) -> None:
        for branch in ((2, 2), (3, 2, 2), (4,)):
            with self.subTest(branch=branch):
                tree = build_tree(branch)
                mask = tree_attention_mask(tree)
                self.assertTrue(validate_tree_mask(mask, tree))

    def test_mask_is_causal_in_construction_order(self) -> None:
        tree = build_tree((3, 2))
        mask = tree_attention_mask(tree)
        n = len(tree)
        self.assertEqual(mask.shape, (n, n))
        for i in range(n):
            self.assertEqual(mask[i, i], 1)
            for j in range(i + 1, n):
                self.assertEqual(mask[i, j], 0, f"node {i} attends to later node {j}")

    def test_attention_span_equals_tree_depth(self) -> None:
        """Root attends to itself only; a depth-d node attends to d+1 nodes."""
        tree = build_tree((2, 2, 2))
        mask = tree_attention_mask(tree)
        spans = mask.sum(axis=1).tolist()
        self.assertEqual(spans[0], 1)
        self.assertEqual(sorted(spans), spans)
        self.assertEqual(max(spans), 4)

    def test_a_broken_mask_is_rejected(self) -> None:
        tree = build_tree((2, 2))
        mask = tree_attention_mask(tree)
        mask[-1, 0] = 0  # drop the root from a leaf's ancestor set
        self.assertFalse(validate_tree_mask(mask, tree))


class TestCliValidation(unittest.TestCase):
    def test_acceptance_rate_must_be_a_probability(self) -> None:
        self.assertAlmostEqual(_unit_float("0.75"), 0.75)
        for bad in ("0", "1.5", "-0.2"):
            with self.subTest(value=bad):
                with self.assertRaises(Exception):
                    _unit_float(bad)

    def test_draft_length_must_be_positive(self) -> None:
        self.assertEqual(_positive_int("4"), 4)
        with self.assertRaises(Exception):
            _positive_int("0")
        with self.assertRaises(Exception):
            _positive_int("1", minimum=2)


if __name__ == "__main__":
    unittest.main()
