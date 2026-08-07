"""Входные данные для замера скорости."""

import random

random.seed(0)

_WORDS = [f"w{i}" for i in range(400)]


def _sentence(length):
    return [random.choice(_WORDS) for _ in range(length)]


_hyp = _sentence(60)
_ref = _sentence(60)
_corpus_hyp = [_sentence(25) for _ in range(200)]
_corpus_ref = [[_sentence(25)] for _ in range(200)]

_text_hyp = " ".join(_sentence(120))
_text_ref = " ".join(_sentence(120))

_glossary = {f"w{i}": f"t{i}" for i in range(200)}

BENCH = {
    "ngrams": (_hyp, 4),
    "clipped_ngram_counts": (_hyp, [_ref], 4),
    "brevity_penalty": (55, 60),
    "sentence_bleu": (_hyp, [_ref]),
    "corpus_bleu": (_corpus_hyp, _corpus_ref),
    "chrf": (_text_hyp, _text_ref),
    "flag_length_explosion": (_hyp, _ref),
    "glossary_violations": (_text_hyp, _text_ref, _glossary),
}
