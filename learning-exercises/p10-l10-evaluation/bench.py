"""Входные данные для замера скорости."""

import math
import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_WORDS = [f"w{i}" for i in range(300)]


def _sentence(n):
    return " ".join(random.choice(_WORDS) for _ in range(n))


_pred = _sentence(400)
_exp = _sentence(400)

_log_probs = [math.log(random.uniform(0.01, 0.99)) for _ in range(50000)]

_names = [f"model_{i}" for i in range(40)]
_matches = [
    (random.choice(_names), random.choice(_names), random.choice(["a", "b", "tie"]))
    for _ in range(20000)
]

_cases = [(_sentence(8), _sentence(8)) for _ in range(2000)]
_answers = {q: a for q, a in _cases}


def _exact_match(prediction, expected):
    return 1.0 if prediction.strip().lower() == expected.strip().lower() else 0.0


_results = [{"scores": {"em": random.random(), "f1": random.random()}} for _ in range(50000)]

BENCH = {
    "exact_match": (_pred, _exp),
    "token_f1": (_pred, _exp),
    "perplexity": (_log_probs,),
    "expected_score": (1720.0, 1480.0),
    "elo_update": (1720.0, 1480.0, "a"),
    "elo_tournament": (_matches,),
    "run_suite": (_cases, _answers.get, {"em": _exact_match}),
    "summarize": (_results,),
}
