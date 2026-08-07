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
    """Счётчики считаем один раз, чтобы замер мерил измеряемую функцию."""
    ngrams = {}
    contexts = {}
    for sentence in _sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            ctx = (padded[i - 1],)
            ngrams[ctx + (padded[i],)] = ngrams.get(ctx + (padded[i],), 0) + 1
            contexts[ctx] = contexts.get(ctx, 0) + 1
    return ngrams, contexts


_ngrams, _contexts = _counts()
_uniform = lambda prev, w: 1.0 / len(_VOCAB)

BENCH = {
    "train_ngram": (_sentences, 3),
    "raw_probability": (_ngrams, _contexts, ("the",), "cat"),
    "laplace_probability": (_ngrams, _contexts, len(_VOCAB), ("the",), "cat"),
    "continuation_probability": (_sentences,),
    "kneser_ney_bigram": (_sentences, 0.75),
    "bits_per_token": (_uniform, _held_out),
    "perplexity": (_uniform, _held_out),
    "generate": (_uniform, _VOCAB, ["<s>"], random.Random(0), 30),
}
