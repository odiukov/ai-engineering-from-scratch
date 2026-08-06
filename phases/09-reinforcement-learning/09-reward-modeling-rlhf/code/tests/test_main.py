"""Tests for the RLHF toy pipeline: Bradley-Terry reward model, KL, PPO-RLHF loop.

Covers docs/en.md "Build It" steps 1-4. The Bradley-Terry update is checked
against a central-difference gradient of -log sigmoid(R+ - R-) taken at a
non-trivial w, so a (1 - p) / p mix-up or a sign flip cannot slip through: at
w = 0 the two would coincide, which is why the check runs on the *second*
training pair rather than the first.
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
    BAD,
    GOOD,
    PROMPTS,
    VOCAB,
    bag,
    kl,
    policy_probs,
    rlhf_loop,
    rm_accuracy,
    sample_pair,
    sample_token,
    score,
    sigmoid,
    softmax,
    train_rm,
)


def bt_loss(weights, y_pos, y_neg):
    """-log sigmoid(R(y_pos) - R(y_neg)), the Bradley-Terry pairwise logistic."""
    margin = score(weights, y_pos) - score(weights, y_neg)
    return -math.log(sigmoid(margin))


def numeric_bt_grad(weights, y_pos, y_neg, token, h=1e-6):
    plus = dict(weights)
    plus[token] = plus.get(token, 0.0) + h
    minus = dict(weights)
    minus[token] = minus.get(token, 0.0) - h
    return (bt_loss(plus, y_pos, y_neg) - bt_loss(minus, y_pos, y_neg)) / (2.0 * h)


class TestSigmoid(unittest.TestCase):
    def test_zero_maps_to_one_half_and_is_symmetric(self):
        self.assertAlmostEqual(sigmoid(0.0), 0.5, places=12)
        for x in (0.3, 2.0, 17.0):
            self.assertAlmostEqual(sigmoid(x) + sigmoid(-x), 1.0, places=12)

    def test_large_magnitudes_do_not_overflow(self):
        self.assertAlmostEqual(sigmoid(1000.0), 1.0, places=12)
        self.assertAlmostEqual(sigmoid(-1000.0), 0.0, places=12)


class TestScoring(unittest.TestCase):
    def test_repeated_tokens_are_counted_not_deduplicated(self):
        w = {"clear": 0.5}
        self.assertAlmostEqual(score(w, ("clear", "clear")), 1.0, places=12)
        self.assertAlmostEqual(score(w, ("clear",)), 0.5, places=12)

    def test_unseen_tokens_contribute_zero(self):
        self.assertEqual(score({"clear": 2.0}, ("unknown", "words")), 0.0)

    def test_bag_is_a_multiset_of_tokens(self):
        self.assertEqual(dict(bag(("a", "b", "a"))), {"a": 2, "b": 1})

    def test_vocab_is_the_union_of_good_and_bad_words(self):
        self.assertEqual(set(VOCAB), set(GOOD) | set(BAD))
        self.assertEqual(len(VOCAB), len(set(GOOD)) + len(set(BAD)))


class TestBradleyTerryGradient(unittest.TestCase):
    SEED = 5
    LR = 0.1

    def test_update_equals_minus_lr_times_the_numerical_loss_gradient(self):
        w_one = dict(train_rm(n_pairs=1, lr=self.LR, rng=random.Random(self.SEED)))
        w_two = dict(train_rm(n_pairs=2, lr=self.LR, rng=random.Random(self.SEED)))

        replay = random.Random(self.SEED)
        sample_pair(replay)
        _, y_pos, y_neg = sample_pair(replay)

        margin = score(w_one, y_pos) - score(w_one, y_neg)
        self.assertNotAlmostEqual(
            sigmoid(margin), 0.5, places=3, msg="second pair must start from a non-trivial w"
        )

        tokens = set(w_one) | set(w_two) | set(y_pos) | set(y_neg)
        for token in tokens:
            step = w_two.get(token, 0.0) - w_one.get(token, 0.0)
            expected = -self.LR * numeric_bt_grad(w_one, y_pos, y_neg, token)
            self.assertAlmostEqual(step, expected, delta=1e-8)

    def test_preferred_response_gains_weight_and_rejected_loses_it(self):
        replay = random.Random(self.SEED)
        _, y_pos, y_neg = sample_pair(replay)
        w = train_rm(n_pairs=1, lr=self.LR, rng=random.Random(self.SEED))
        for token in set(y_pos):
            self.assertGreater(w[token], 0.0)
        for token in set(y_neg):
            self.assertLess(w[token], 0.0)

    def test_confidently_ranked_pairs_produce_a_smaller_step(self):
        """The (1 - p) factor means an already-correct pair barely moves w."""
        replay = random.Random(self.SEED)
        _, y_pos, y_neg = sample_pair(replay)
        after_one = dict(train_rm(n_pairs=1, lr=self.LR, rng=random.Random(self.SEED)))
        after_many = dict(train_rm(n_pairs=400, lr=self.LR, rng=random.Random(self.SEED)))
        after_more = dict(train_rm(n_pairs=401, lr=self.LR, rng=random.Random(self.SEED)))

        early_move = abs(after_one.get(y_pos[0], 0.0))
        late_move = max(
            abs(after_more.get(t, 0.0) - after_many.get(t, 0.0))
            for t in set(after_more) | set(after_many)
        )
        self.assertGreater(
            sigmoid(score(after_many, y_pos) - score(after_many, y_neg)), sigmoid(0.0)
        )
        self.assertLess(late_move, early_move)


class TestRewardModelQuality(unittest.TestCase):
    def test_good_words_outrank_bad_words_after_training(self):
        w = train_rm(n_pairs=600, rng=random.Random(42))
        for token in GOOD:
            self.assertGreater(w[token], 0.0, token)
        for token in BAD:
            self.assertLess(w[token], 0.0, token)

    def test_holdout_pairwise_accuracy_clears_ninety_percent(self):
        w = train_rm(n_pairs=600, rng=random.Random(42))
        self.assertGreater(rm_accuracy(w, n_pairs=200, rng=random.Random(1)), 0.9)

    def test_untrained_model_is_at_chance_and_never_prefers_anything(self):
        empty = {}
        self.assertEqual(rm_accuracy(empty, n_pairs=50, rng=random.Random(1)), 0.0)


class TestKL(unittest.TestCase):
    def test_identical_distributions_have_zero_divergence(self):
        p = softmax([0.3, -1.0, 2.0, 0.0])
        self.assertAlmostEqual(kl(p, p), 0.0, places=12)

    def test_divergence_is_positive_and_asymmetric(self):
        p = softmax([2.0, 0.0, 0.0, 0.0])
        q = softmax([0.0, 0.0, 0.0, 0.0])
        self.assertGreater(kl(p, q), 0.0)
        self.assertNotAlmostEqual(kl(p, q), kl(q, p), places=6)

    def test_zero_probability_terms_are_skipped_not_infinite(self):
        self.assertAlmostEqual(kl([0.0, 1.0], [0.5, 0.5]), math.log(2.0), places=12)


class TestPolicyAndSampling(unittest.TestCase):
    def test_policy_probs_normalize_per_prompt(self):
        theta = [[0.1 * i for i in range(len(VOCAB))] for _ in PROMPTS]
        for p_idx in range(len(PROMPTS)):
            self.assertAlmostEqual(sum(policy_probs(theta, p_idx)), 1.0, places=12)

    def test_sample_token_honours_a_degenerate_distribution(self):
        rng = random.Random(0)
        probs = [0.0] * len(VOCAB)
        probs[3] = 1.0
        self.assertEqual({sample_token(probs, rng) for _ in range(40)}, {3})


class TestRLHFLoop(unittest.TestCase):
    def test_reference_starts_equal_to_the_policy_so_first_kl_is_zero(self):
        w = train_rm(n_pairs=200, rng=random.Random(42))
        _, hist = rlhf_loop(w, updates=3, beta=0.1, rng=random.Random(0))
        self.assertEqual(len(hist), 3)
        self.assertAlmostEqual(hist[0][2], 0.0, places=12)

    def test_reported_rm_score_excludes_the_kl_penalty(self):
        w = train_rm(n_pairs=200, rng=random.Random(42))
        _, low = rlhf_loop(w, updates=1, beta=0.0, rng=random.Random(0))
        _, high = rlhf_loop(w, updates=1, beta=5.0, rng=random.Random(0))
        self.assertAlmostEqual(low[0][1], high[0][1], places=12)

    def test_policy_climbs_the_reward_model_score(self):
        w = train_rm(n_pairs=600, rng=random.Random(42))
        _, hist = rlhf_loop(w, updates=150, beta=0.1, rng=random.Random(0))
        self.assertGreater(hist[-1][1], hist[0][1])

    def test_larger_beta_holds_the_policy_closer_to_the_reference(self):
        w = train_rm(n_pairs=600, rng=random.Random(42))
        _, loose = rlhf_loop(w, updates=150, beta=0.0, rng=random.Random(0))
        _, tight = rlhf_loop(w, updates=150, beta=5.0, rng=random.Random(0))
        self.assertLess(tight[-1][2], loose[-1][2])

    def test_loop_is_reproducible_for_a_given_seed(self):
        w = train_rm(n_pairs=200, rng=random.Random(42))
        first = rlhf_loop(w, updates=20, beta=0.1, rng=random.Random(11))[1]
        second = rlhf_loop(w, updates=20, beta=0.1, rng=random.Random(11))[1]
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
