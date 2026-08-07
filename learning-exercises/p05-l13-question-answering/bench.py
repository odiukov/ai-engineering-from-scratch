"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_WORDS = ["iphone", "apple", "released", "june", "2007", "android", "google",
          "steve", "jobs", "macworld", "ipod", "launched", "mobile", "system"]

_corpus = [
    " ".join(random.choice(_WORDS) for _ in range(40))
    for _ in range(400)
]
_question = "when was the first iphone released by apple"
_rankings = [[(0.5, i) for i in range(20)] for _ in range(200)]
_gold = [random.randrange(400) for _ in range(200)]
_starts = [random.random() for _ in range(400)]
_ends = [random.random() for _ in range(400)]
_tokens = [random.choice(_WORDS) for _ in range(400)]

BENCH = {
    "normalize_answer": (" ".join(_WORDS * 200),),
    "exact_match": ("June 29, 2007", "june 29 2007"),
    "token_f1": (" ".join(_WORDS * 50), " ".join(_WORDS * 50)),
    "best_span": (_starts, _ends, 15),
    "answer_span": (_tokens, _starts, _ends, 15),
    "retrieve_top_k": (_question, _corpus, 5),
    "recall_at_k": (_rankings, _gold, 10),
    "answer_with_refusal": (_question, _corpus, 0.5),
}
