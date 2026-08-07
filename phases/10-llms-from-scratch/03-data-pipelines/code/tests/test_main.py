"""Tests for the pre-training data pipeline.

Covers: cleaning, quality filtering, MinHash/LSH near-duplicate detection and
its idempotence, sequence packing that neither loses nor duplicates a token,
and a data loader that serves every sequence exactly once.
Reference: ../../docs/en.md
"""

import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from main import (  # noqa: E402
    PreTrainingDataLoader,
    SimpleTokenizer,
    clean_text,
    compute_statistics,
    deduplicate,
    generate_sample_corpus,
    get_shingles,
    lsh_buckets,
    minhash_signature,
    pack_sequences,
    quality_filter,
    tokenize_corpus,
)

DOC_A = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
    "omicron pi rho sigma tau upsilon phi chi psi omega"
)
DOC_B = (
    "one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen twenty"
)
DOC_A_NEAR = DOC_A + " and one extra tail phrase here"

DEDUP_KW = {"threshold": 0.8, "num_hashes": 32, "bands": 8}


class TestCleaning(unittest.TestCase):
    def test_html_tags_and_urls_are_removed(self):
        out = clean_text("<p>hello</p> see http://example.com/page now")
        self.assertNotIn("<p>", out)
        self.assertNotIn("http", out)
        self.assertIn("hello", out)

    def test_runs_of_spaces_and_blank_lines_are_collapsed(self):
        self.assertEqual(clean_text("a     b"), "a b")
        self.assertEqual(clean_text("a\n\n\n\n\nb"), "a\n\nb")

    def test_non_printable_characters_are_dropped(self):
        self.assertEqual(clean_text("café bar"), "caf bar")


class TestQualityFilter(unittest.TestCase):
    def test_too_short_documents_are_rejected(self):
        self.assertFalse(quality_filter("only a handful of words here"))

    def test_shouting_documents_are_rejected(self):
        self.assertFalse(quality_filter(" ".join(["SPAM"] * 80)))

    def test_ordinary_prose_is_kept(self):
        self.assertTrue(quality_filter(" ".join(["word"] * 80)))

    def test_thresholds_are_honoured(self):
        text = " ".join(["word"] * 60)
        self.assertFalse(quality_filter(text, min_words=100))
        self.assertTrue(quality_filter(text, min_words=10))


class TestShinglesAndMinHash(unittest.TestCase):
    def test_shingle_count_is_words_minus_k_plus_one(self):
        text = " ".join(str(i) for i in range(20))
        self.assertEqual(len(get_shingles(text, k=5)), 20 - 5 + 1)

    def test_documents_shorter_than_k_have_no_shingles(self):
        self.assertEqual(get_shingles("a b c", k=5), set())

    def test_signature_is_deterministic_and_shingle_order_independent(self):
        shingles = get_shingles(DOC_A)
        first = minhash_signature(shingles, 16)
        second = minhash_signature(set(reversed(sorted(shingles))), 16)
        self.assertEqual(first, second)

    def test_signature_agreement_tracks_similarity(self):
        sig_a = minhash_signature(get_shingles(DOC_A), 64)
        sig_near = minhash_signature(get_shingles(DOC_A_NEAR), 64)
        sig_far = minhash_signature(get_shingles(DOC_B), 64)

        def agreement(x, y):
            return sum(1 for p, q in zip(x, y) if p == q) / len(x)

        self.assertGreater(agreement(sig_a, sig_near), agreement(sig_a, sig_far))
        self.assertEqual(agreement(sig_a, sig_a), 1.0)

    def test_empty_shingle_set_yields_a_zero_signature(self):
        self.assertEqual(minhash_signature(set(), 8), [0] * 8)

    def test_lsh_emits_one_bucket_per_band(self):
        sig = minhash_signature(get_shingles(DOC_A), 32)
        buckets = lsh_buckets(sig, bands=8)
        self.assertEqual(len(buckets), 8)
        self.assertEqual([b for b, _ in buckets], list(range(8)))
        self.assertEqual(lsh_buckets(sig, bands=8), buckets)


class TestDeduplication(unittest.TestCase):
    def test_exact_duplicate_is_removed_and_the_first_copy_kept(self):
        kept, removed = deduplicate([DOC_A, DOC_B, DOC_A], **DEDUP_KW)
        self.assertEqual(removed, 1)
        self.assertEqual(kept, [DOC_A, DOC_B])

    def test_deduplication_is_idempotent(self):
        once, first_removed = deduplicate([DOC_A, DOC_A, DOC_B], **DEDUP_KW)
        twice, second_removed = deduplicate(once, **DEDUP_KW)
        self.assertEqual(first_removed, 1)
        self.assertEqual(second_removed, 0)
        self.assertEqual(twice, once)

    def test_distinct_documents_all_survive(self):
        kept, removed = deduplicate([DOC_A, DOC_B], **DEDUP_KW)
        self.assertEqual(removed, 0)
        self.assertEqual(kept, [DOC_A, DOC_B])

    def test_output_is_a_subsequence_of_the_input(self):
        docs = [DOC_A, DOC_B, DOC_A, DOC_A_NEAR]
        kept, _ = deduplicate(docs, **DEDUP_KW)
        positions = [docs.index(d) for d in kept]
        self.assertEqual(positions, sorted(positions))
        self.assertTrue(set(kept).issubset(set(docs)))


class TestPacking(unittest.TestCase):
    def test_packing_loses_and_duplicates_nothing(self):
        token_ids = list(range(1, 24))
        sequences, masks = pack_sequences(token_ids, 5, pad_id=0)
        recovered = [
            tok
            for seq, mask in zip(sequences, masks)
            for tok, keep in zip(seq, mask)
            if keep
        ]
        self.assertEqual(recovered, token_ids)

    def test_every_sequence_has_the_requested_length(self):
        sequences, masks = pack_sequences(list(range(23)), 5)
        self.assertTrue(all(len(s) == 5 for s in sequences))
        self.assertTrue(all(len(m) == 5 for m in masks))

    def test_only_the_tail_is_padded(self):
        sequences, masks = pack_sequences(list(range(1, 12)), 4, pad_id=0)
        self.assertEqual(masks[:-1], [[1, 1, 1, 1], [1, 1, 1, 1]])
        self.assertEqual(masks[-1], [1, 1, 1, 0])
        self.assertEqual(sequences[-1][-1], 0)

    def test_exact_multiple_needs_no_padding(self):
        _, masks = pack_sequences(list(range(12)), 4)
        self.assertTrue(all(all(m) for m in masks))


class TestDataLoader(unittest.TestCase):
    def _loader_sequences(self, shuffle):
        sequences, masks = pack_sequences(list(range(1, 41)), 4)
        loader = PreTrainingDataLoader(sequences, masks, batch_size=3, shuffle=shuffle)
        served = []
        for batch_seqs, batch_masks in loader:
            self.assertEqual(len(batch_seqs), len(batch_masks))
            served.extend(batch_seqs)
        return sequences, served

    def test_every_sequence_is_served_exactly_once(self):
        random.seed(20240607)
        for shuffle in (False, True):
            sequences, served = self._loader_sequences(shuffle)
            self.assertEqual(len(served), len(sequences))
            self.assertEqual(sorted(served), sorted(sequences))

    def test_unshuffled_order_matches_the_input(self):
        sequences, served = self._loader_sequences(shuffle=False)
        self.assertEqual(served, sequences)

    def test_batch_count_covers_the_partial_tail(self):
        sequences, masks = pack_sequences(list(range(1, 41)), 4)
        loader = PreTrainingDataLoader(sequences, masks, batch_size=3, shuffle=False)
        self.assertEqual(len(loader), 4)
        self.assertEqual(len(list(loader)), len(loader))

    def test_masks_travel_with_their_sequences(self):
        random.seed(7)
        sequences, masks = pack_sequences(list(range(1, 15)), 4)
        pairing = {tuple(s): tuple(m) for s, m in zip(sequences, masks)}
        loader = PreTrainingDataLoader(sequences, masks, batch_size=2, shuffle=True)
        for batch_seqs, batch_masks in loader:
            for seq, mask in zip(batch_seqs, batch_masks):
                self.assertEqual(pairing[tuple(seq)], tuple(mask))


class TestTokenizerStage(unittest.TestCase):
    def test_eos_is_appended_once_per_document(self):
        tok = SimpleTokenizer()
        tok.train_bpe(DOC_A + " " + DOC_B, num_merges=20)
        ids = tokenize_corpus([DOC_A, DOC_B], tok)
        self.assertEqual(ids.count(tok.eos_id), 2)
        self.assertEqual(ids[-1], tok.eos_id)

    def test_roundtrip_ignores_eos_and_recovers_the_text(self):
        tok = SimpleTokenizer()
        tok.train_bpe(DOC_A, num_merges=30)
        self.assertEqual(tok.decode(tok.encode(DOC_A) + [tok.eos_id]), DOC_A)

    def test_vocab_grows_by_one_per_merge_plus_the_eos(self):
        for k in (0, 5, 25):
            tok = SimpleTokenizer()
            tok.train_bpe(DOC_A, num_merges=k)
            self.assertEqual(tok.vocab_size(), 256 + k + 1)


class TestStatistics(unittest.TestCase):
    def test_utilization_and_totals_are_consistent(self):
        docs = [DOC_A, DOC_B]
        tok = SimpleTokenizer()
        tok.train_bpe(" ".join(docs), num_merges=20)
        ids = tokenize_corpus(docs, tok)
        sequences, _ = pack_sequences(ids, 32)

        stats = compute_statistics(docs, ids, sequences, tok.vocab_size())
        self.assertEqual(stats["total_documents"], 2)
        self.assertEqual(stats["total_tokens"], len(ids))
        self.assertEqual(stats["unique_tokens"], len(set(ids)))
        self.assertEqual(stats["num_sequences"], len(sequences))
        self.assertLessEqual(stats["sequence_utilization"], 1.0)
        self.assertGreater(stats["sequence_utilization"], 0.0)
        self.assertLessEqual(stats["vocab_utilization"], 1.0)

    def test_sample_corpus_has_something_for_every_filter_stage(self):
        docs = generate_sample_corpus()
        cleaned = [clean_text(d) for d in docs]
        kept = [d for d in cleaned if quality_filter(d)]
        self.assertLess(len(kept), len(cleaned))
        deduped, removed = deduplicate(kept, **DEDUP_KW)
        self.assertGreater(removed, 0)
        self.assertEqual(len(deduped), len(kept) - removed)


if __name__ == "__main__":
    unittest.main()
