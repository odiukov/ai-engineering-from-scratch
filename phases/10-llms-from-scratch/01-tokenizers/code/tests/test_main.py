"""Tests for character and byte-level BPE tokenizers.

Covers: lossless round-trips, merge determinism, monotonic vocab growth,
non-overlapping left-to-right merge application, and compression accounting.
Reference: ../../docs/en.md
"""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from main import (  # noqa: E402
    BPETokenizer,
    CharTokenizer,
    compression_ratio,
    vocabulary_stats,
)


CORPUS = (
    "The cat sat on the mat. The cat ate the rat. "
    "The dog sat on the log. The dog ate the frog. "
    "Tokenization is the first step in any NLP pipeline. "
    "Language models read tokens, not words."
)


def trained(num_merges):
    return BPETokenizer().train(CORPUS, num_merges=num_merges)


class TestCharTokenizer(unittest.TestCase):
    def test_roundtrip_is_lossless_including_non_ascii(self):
        ct = CharTokenizer()
        for text in ["hello", "Hello, world!", "GPT-4", "ünïcödé", "你好", ""]:
            self.assertEqual(ct.decode(ct.encode(text)), text)

    def test_one_token_per_character(self):
        ct = CharTokenizer()
        text = "abc 你好"
        self.assertEqual(len(ct.encode(text)), len(text))


class TestBPEMergeMechanics(unittest.TestCase):
    def test_merge_is_non_overlapping_and_left_to_right(self):
        tok = BPETokenizer()
        # Three identical symbols admit only one non-overlapping merge.
        self.assertEqual(tok._merge_pair([1, 1, 1], (1, 1), 99), [99, 1])
        self.assertEqual(tok._merge_pair([1, 1, 1, 1], (1, 1), 99), [99, 99])
        self.assertEqual(tok._merge_pair([0, 1, 1, 2], (1, 1), 99), [0, 99, 2])

    def test_merge_leaves_unrelated_pairs_untouched(self):
        tok = BPETokenizer()
        self.assertEqual(tok._merge_pair([7, 8, 9], (1, 2), 99), [7, 8, 9])

    def test_training_is_deterministic(self):
        a = trained(30)
        b = trained(30)
        self.assertEqual(a.merges, b.merges)
        self.assertEqual(a.vocab, b.vocab)


class TestBPERoundTrip(unittest.TestCase):
    def test_decode_encode_is_identity_on_training_text(self):
        tok = trained(50)
        self.assertEqual(tok.decode(tok.encode(CORPUS)), CORPUS)

    def test_decode_encode_is_identity_on_unseen_text(self):
        tok = trained(50)
        for text in [
            "The dog ate the frog.",
            "unhappiness",
            "Geschwindigkeitsbegrenzung",
            "你好世界",
            "\U0001f525 emoji tail",
        ]:
            self.assertEqual(tok.decode(tok.encode(text)), text)

    def test_untrained_tokenizer_is_plain_bytes(self):
        tok = BPETokenizer()
        tok.train("", num_merges=5)
        self.assertEqual(tok.encode("abc"), [97, 98, 99])
        self.assertEqual(tok.vocab_size(), 256)


class TestVocabGrowth(unittest.TestCase):
    def test_vocab_grows_by_exactly_one_per_merge(self):
        sizes = [trained(k).vocab_size() for k in (0, 1, 5, 20, 50)]
        self.assertEqual(sizes, [256, 257, 261, 276, 306])

    def test_vocab_size_is_monotonic_in_num_merges(self):
        sizes = [trained(k).vocab_size() for k in range(0, 40, 8)]
        self.assertEqual(sizes, sorted(sizes))
        self.assertEqual(len(set(sizes)), len(sizes))

    def test_merged_token_bytes_are_concatenation_of_parts(self):
        tok = trained(20)
        for (left, right), new_id in tok.merges.items():
            self.assertEqual(tok.vocab[new_id], tok.vocab[left] + tok.vocab[right])

    def test_more_merges_never_lengthen_the_encoding(self):
        few = len(trained(5).encode(CORPUS))
        many = len(trained(50).encode(CORPUS))
        self.assertLess(many, few)


class TestCompressionAccounting(unittest.TestCase):
    def test_compression_ratio_matches_tokens_over_bytes(self):
        tok = trained(50)
        text = "The cat sat on the mat."
        expected = len(tok.encode(text)) / len(text.encode("utf-8"))
        self.assertAlmostEqual(compression_ratio(tok, text), expected, places=12)

    def test_trained_tokenizer_compresses_its_corpus(self):
        self.assertLess(compression_ratio(trained(50), CORPUS), 1.0)

    def test_untrained_tokenizer_has_ratio_one(self):
        tok = BPETokenizer()
        tok.train("", num_merges=0)
        self.assertAlmostEqual(compression_ratio(tok, "plain ascii"), 1.0, places=12)


class TestReporting(unittest.TestCase):
    def test_token_to_str_falls_back_for_unknown_ids(self):
        tok = trained(5)
        self.assertEqual(tok.token_to_str(99999), "<?>")

    def test_vocabulary_stats_reports_unused_tokens(self):
        tok = trained(50)
        buf = io.StringIO()
        with redirect_stdout(buf):
            vocabulary_stats(tok, [CORPUS])
        out = buf.getvalue()
        self.assertIn(f"Vocabulary size: {tok.vocab_size()}", out)
        self.assertIn("Unused tokens:", out)


if __name__ == "__main__":
    unittest.main()
