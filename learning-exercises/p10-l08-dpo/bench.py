"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_VOCAB = 512
_STEPS = 400

_logits_per_step = [
    [random.gauss(0.0, 2.0) for _ in range(_VOCAB)] for _ in range(_STEPS)
]
_targets = [random.randrange(_VOCAB) for _ in range(_STEPS)]

_rows = [
    (
        random.gauss(-200.0, 5.0),
        random.gauss(-210.0, 5.0),
        random.gauss(-205.0, 5.0),
        random.gauss(-205.0, 5.0),
    )
    for _ in range(20000)
]

BENCH = {
    "sigmoid": (0.7,),
    "log_softmax": (_logits_per_step[0],),
    "sequence_logprob": (_logits_per_step, _targets),
    "dpo_logit": (-1.0, -4.0, -2.0, -3.0, 0.1),
    "dpo_loss": (-1.0, -4.0, -2.0, -3.0, 0.1),
    "dpo_grad": (-1.0, -4.0, -2.0, -3.0, 0.1),
    "implicit_rewards": (-1.0, -4.0, -2.0, -3.0, 0.1),
    "preference_accuracy": (_rows,),
}
