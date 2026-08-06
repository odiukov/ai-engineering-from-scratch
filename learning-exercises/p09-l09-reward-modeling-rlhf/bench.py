"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_GOOD = ("clear", "specific", "kind", "thorough", "precise", "helpful")
_BAD = ("vague", "rude", "wrong", "short", "cold", "careless")
_FILLER = tuple(f"tok{i}" for i in range(40))

_w = {t: random.gauss(0.0, 1.0) for t in _GOOD + _BAD + _FILLER}
_response = [random.choice(_GOOD + _FILLER) for _ in range(64)]

# 500 пар — размер, на котором RM урока уже уверенно разделяет токены
_pairs = [
    (
        [random.choice(_GOOD), random.choice(_GOOD), random.choice(_FILLER)],
        [random.choice(_BAD), random.choice(_BAD), random.choice(_FILLER)],
    )
    for _ in range(500)
]

_p = [0.4, 0.3, 0.2, 0.1]
_q = [0.25, 0.25, 0.25, 0.25]

BENCH = {
    "sigmoid": (-3.5,),
    "reward_score": (_w, _response),
    "bt_loss": (_w, _pairs[0][0], _pairs[0][1]),
    "bt_gradient": (_w, _pairs[0][0], _pairs[0][1]),
    "train_reward_model": (_pairs, 0.1, 1),
    "pairwise_accuracy": (_w, _pairs),
    "kl_divergence": (_p, _q),
    "penalized_reward": (1.0, _p, _q, 0.1),
}
