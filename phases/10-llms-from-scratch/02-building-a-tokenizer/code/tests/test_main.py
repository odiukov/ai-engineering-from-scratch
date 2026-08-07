"""Tests for the production byte-level BPE tokenizer.

Covers: pre-tokenization losslessness, NFKC normalization, special tokens that
are never split, non-overlapping merges, deterministic training, and vocab
growth of exactly one entry per merge.
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
    ProductionTokenizer,
    SpecialTokenHandler,
    apply_merge,
    pre_tokenize,
)

try:  # the GPT-2 pattern needs \p{L}; the re fallback is ASCII-only
    import regex  # noqa: F401

    HAS_REGEX = True
except ImportError:  # pragma: no cover
    HAS_REGEX = False


CORPUS = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Machine learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
)


def trained(num_merges=40):
    tok = ProductionTokenizer()
    with redirect_stdout(io.StringIO()):
        tok.train(CORPUS, num_merges=num_merges)
    return tok


class TestPreTokenization(unittest.TestCase):
    def test_chunks_concatenate_back_to_the_input(self):
        for text in [
            "Hello, world! Don't stop.",
            "def train(model, data):",
            "The price is $3.14 per unit.",
            "  multiple   spaces   here  ",
        ]:
            self.assertEqual("".join(pre_tokenize(text)), text)

    @unittest.skipUnless(HAS_REGEX, "unicode property classes need the regex module")
    def test_chunks_concatenate_back_to_non_ascii_input(self):
        for text in ["你好世界 Hello World", "\U0001f525\U0001f30d\U0001f680", "naïve café"]:
            self.assertEqual("".join(pre_tokenize(text)), text)

    def test_leading_space_stays_with_the_following_word(self):
        self.assertEqual(pre_tokenize("a b"), ["a", " b"])


class TestMergeMechanics(unittest.TestCase):
    def test_merge_is_non_overlapping_and_left_to_right(self):
        self.assertEqual(apply_merge([1, 1, 1], (1, 1), 99), [99, 1])
        self.assertEqual(apply_merge([1, 1, 1, 1], (1, 1), 99), [99, 99])
        self.assertEqual(apply_merge([0, 1, 1, 2], (1, 1), 99), [0, 99, 2])

    def test_merge_of_absent_pair_is_the_identity(self):
        self.assertEqual(apply_merge([7, 8, 9], (1, 2), 99), [7, 8, 9])

    def test_training_is_deterministic(self):
        self.assertEqual(trained(30).merges, trained(30).merges)


class TestVocabGrowth(unittest.TestCase):
    def test_vocab_grows_by_exactly_one_per_merge(self):
        for k in (0, 1, 10, 40):
            self.assertEqual(trained(k).vocab_size(), 256 + k)

    def test_merge_ids_are_assigned_in_order_from_256(self):
        tok = trained(15)
        self.assertEqual(sorted(tok.merges.values()), list(range(256, 271)))

    def test_merged_token_bytes_are_concatenation_of_parts(self):
        tok = trained(20)
        for (left, right), new_id in tok.merges.items():
            self.assertEqual(tok.vocab[new_id], tok.vocab[left] + tok.vocab[right])

    def test_special_tokens_take_ids_after_the_merges(self):
        tok = trained(20)
        before = tok.vocab_size()
        bos = tok.add_special_token("<|begin|>")
        eos = tok.add_special_token("<|end|>")
        self.assertEqual([bos, eos], [276, 277])
        self.assertEqual(tok.vocab_size(), before + 2)


class TestRoundTrip(unittest.TestCase):
    def test_decode_encode_is_identity_on_ascii(self):
        tok = trained()
        for text in [
            "The quick brown fox.",
            "def foo(x): return x + 1",
            "Machine learning is powerful.",
        ]:
            self.assertEqual(tok.decode(tok.encode(text)), text)

    @unittest.skipUnless(HAS_REGEX, "unicode property classes need the regex module")
    def test_decode_encode_is_identity_on_non_ascii(self):
        tok = trained()
        for text in ["你好世界 Hello World", "\U0001f525\U0001f30d\U0001f680"]:
            self.assertEqual(tok.decode(tok.encode(text)), text)

    def test_encoding_never_exceeds_the_raw_byte_count(self):
        tok = trained()
        for text in ["The quick brown fox.", "Machine learning models"]:
            self.assertLessEqual(len(tok.encode(text)), len(text.encode("utf-8")))


class TestNormalization(unittest.TestCase):
    def test_nfkc_collapses_the_ligature_to_its_letters(self):
        tok = trained()
        self.assertEqual(tok.encode("ﬁne"), tok.encode("fine"))

    def test_normalize_is_idempotent(self):
        tok = ProductionTokenizer()
        once = tok.normalize("ﬁne café")
        self.assertEqual(tok.normalize(once), once)


class TestSpecialTokens(unittest.TestCase):
    def test_special_token_encodes_as_a_single_id(self):
        tok = trained()
        eos = tok.add_special_token("<|end|>")
        ids = tok.encode("The quick<|end|>brown")
        self.assertEqual(ids.count(eos), 1)
        plain = tok.encode("The quickbrown")
        self.assertEqual(len(ids), len(plain) + 1)

    def test_special_token_bytes_are_never_reached_by_bpe(self):
        tok = trained()
        eos = tok.add_special_token("<|end|>")
        # Every id in the payload halves is below the special id.
        ids = tok.encode("<|end|>")
        self.assertEqual(ids, [eos])

    def test_longest_special_token_wins_over_its_prefix(self):
        handler = SpecialTokenHandler()
        handler.add_token("<|end|>", 500)
        handler.add_token("<|endoftext|>", 501)
        parts = handler.split_with_specials("a<|endoftext|>b")
        self.assertEqual(parts, [("a", False), ("<|endoftext|>", True), ("b", False)])

    def test_split_preserves_the_whole_input(self):
        handler = SpecialTokenHandler()
        handler.add_token("<|user|>", 500)
        text = "<|user|>hi there<|user|>"
        parts = handler.split_with_specials(text)
        self.assertEqual("".join(p for p, _ in parts), text)

    def test_handler_without_tokens_returns_one_plain_part(self):
        self.assertEqual(
            SpecialTokenHandler().split_with_specials("abc"), [("abc", False)]
        )


class TestDecodeEdgeCases(unittest.TestCase):
    def test_unknown_ids_are_dropped_rather_than_raising(self):
        tok = trained()
        self.assertEqual(tok.decode([99999] + tok.encode("ab")), "ab")

    def test_get_token_bytes_falls_back_for_unknown_ids(self):
        self.assertEqual(trained(5).get_token_bytes(99999), b"<?>")


if __name__ == "__main__":
    unittest.main()
