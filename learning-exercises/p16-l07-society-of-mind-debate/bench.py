"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_N = 120
_answers = [random.uniform(0.0, 100.0) for _ in range(_N)]
_confidences = [random.uniform(0.1, 1.0) for _ in range(_N)]

BENCH = {
    "weighted_mean": (_answers, _confidences),
    "spread": (_answers,),
    "agreement_score": (_answers, 1.0),
    "debate_round": (_answers, _confidences, 10.0, 0.3),
    "run_debate": (_answers, _confidences, 15, 10.0, 0.3),
    "rounds_to_consensus": (_answers, _confidences, 0.5, 1e9, 0.5, 40),
    "opinion_clusters": (_answers, 1.0),
    "sycophancy_collapse": (_answers, _confidences),
}
