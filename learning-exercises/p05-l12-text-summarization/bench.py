"""Входные данные для замера скорости."""

import random

random.seed(0)

_VOCAB = [f"w{i}" for i in range(300)]


def _sentence(length):
    return " ".join(random.choice(_VOCAB) for _ in range(length))


_sentences = [_sentence(random.randint(8, 20)) for _ in range(120)]
_article = ". ".join(_sentences) + "."

_candidate = _sentence(180).split()
_reference = _sentence(180).split()

_source_text = ". ".join(
    " ".join(random.choice(_VOCAB + ["Smith", "Brown", "25,000"]) for _ in range(15))
    for _ in range(80)
) + "."
_summary_text = ". ".join(
    " ".join(random.choice(_VOCAB + ["Smith", "Jones", "42"]) for _ in range(15))
    for _ in range(20)
) + "."

BENCH = {
    "sentence_split": (_article,),
    "similarity": (_sentences[0], _sentences[1]),
    "textrank_scores": (_sentences[:60],),
    "textrank_summary": (". ".join(_sentences[:60]) + ".", 3),
    "lcs_length": (_candidate, _reference),
    "rouge_n": (_candidate, _reference, 2),
    "rouge_l": (_candidate, _reference),
    "hallucinated_entities": (_source_text, _summary_text),
}
