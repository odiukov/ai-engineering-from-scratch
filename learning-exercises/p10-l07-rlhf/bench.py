"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_pairs = [(random.gauss(0.5, 1.0), random.gauss(0.0, 1.0)) for _ in range(20000)]
_logits = [random.gauss(0.0, 3.0) for _ in range(4000)]
_ref_logits = [random.gauss(0.0, 3.0) for _ in range(4000)]

_p = [1.0 / 4000] * 4000
_q = [1.0 / 4000] * 4000

BENCH = {
    "sigmoid": (0.7,),
    "bradley_terry_loss": (1.5, 0.25),
    "bradley_terry_grad": (1.5, 0.25),
    "reward_model_accuracy": (_pairs,),
    "softmax": (_logits,),
    "kl_divergence": (_p, _q),
    "rlhf_objective": (1.0, _logits, _ref_logits, 0.02),
    "ppo_clipped_loss": (1.3, 0.8),
}
