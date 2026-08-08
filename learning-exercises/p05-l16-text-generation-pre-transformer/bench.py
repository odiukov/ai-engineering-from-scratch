"""Входные данные для замера скорости."""

import random

random.seed(0)

_VOCAB = [
    "the", "cat", "sat", "on", "mat", "dog", "ran", "fast",
    "san", "francisco", "is", "cold", "today", "and", "tomorrow",
]

_sentences = [
    [random.choice(_VOCAB) for _ in range(random.randint(5, 15))] for _ in range(400)
]
_held_out = _sentences[:40]

_ngrams, _contexts = None, None


def _counts():
    """Триграммные счётчики считаем раз, чтобы benchmark мерил интеграцию."""
    ngrams = {}
    contexts = {}
    for sentence in _sentences:
        padded = ["<s>", "<s>"] + sentence + ["</s>"]
        for i in range(2, len(padded)):
            ctx = tuple(padded[i - 2 : i])
            ngrams[ctx + (padded[i],)] = ngrams.get(ctx + (padded[i],), 0) + 1
            contexts[ctx] = contexts.get(ctx, 0) + 1
    return ngrams, contexts


_ngrams, _contexts = _counts()
_MODEL_VOCAB = _VOCAB + ["</s>"]


def _trigram(context, word):
    """Laplace trigram that makes a stale scalar/one-token context fail loudly."""
    if not isinstance(context, tuple) or len(context) != 2:
        raise ValueError(f"expected a two-token tuple, got {context!r}")
    return (_ngrams.get(context + (word,), 0) + 1) / (
        _contexts.get(context, 0) + len(_MODEL_VOCAB)
    )

BENCH = {
    "train_ngram": (_sentences, 3),
    "raw_probability": (_ngrams, _contexts, ("the",), "cat"),
    "laplace_probability": (_ngrams, _contexts, len(_VOCAB), ("the",), "cat"),
    "continuation_probability": (_sentences,),
    "kneser_ney_bigram": (_sentences, 0.75),
    "bits_per_token": (_trigram, _held_out, 3),
    "perplexity": (_trigram, _held_out, 3),
    "generate": (_trigram, _MODEL_VOCAB, ["<s>"], random.Random(0), 30, 3),
}
