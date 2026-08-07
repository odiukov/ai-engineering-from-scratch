"""Входные данные для замера скорости."""

import random
from collections import Counter

random.seed(0)

_WORDS = [
    "tokenization", "subword", "encoding", "language", "model", "byte",
    "pair", "merge", "vocabulary", "compression", "sentence", "piece",
    "the", "of", "and", "in", "text", "rare", "words", "pieces",
]

_text = " ".join(random.choice(_WORDS) for _ in range(4000))


def _initial_vocab(text):
    """Стартовый словарь BPE: слово в байтовых символах -> его частота."""
    vocab = Counter()
    for word, freq in Counter(text.split(" ")).items():
        symbols = tuple(chr(b) for b in word.encode("utf-8")) + ("</w>",)
        vocab[symbols] += freq
    return dict(vocab)


_vocab = _initial_vocab(_text)
_merges = [("t", "i"), ("ti", "o"), ("tio", "n"), ("e", "n"), ("o", "d")]
_sample = " ".join(random.choice(_WORDS) for _ in range(300))
_tokens = [chr(b) for b in _sample.encode("utf-8")] + ["</w>"]

BENCH = {
    "pre_tokenize": (_text,),
    "word_to_symbols": ("tokenization" * 20,),
    "pair_counts": (_vocab,),
    "merge_vocab": (_vocab, ("t", "i")),
    "train_bpe": (_sample, 40),
    "encode": (_sample, _merges),
    "decode": (_tokens,),
    "tokenizer_stats": (_sample, _merges),
}
