"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_VOCAB = 64
_logits = [random.gauss(0.0, 1.0) for _ in range(_VOCAB)]
_probs = [1.0 / _VOCAB] * _VOCAB
_ref = [1.0 / _VOCAB] * _VOCAB

_G = 64  # верхняя граница размера группы в GRPO
_samples = [random.randrange(_VOCAB) for _ in range(_G)]
_rewards = [1.0 if random.random() < 0.3 else 0.0 for _ in range(_G)]

BENCH = {
    "softmax": (_logits,),
    "group_advantages": (_rewards,),
    "kl_penalty_gradient": (_probs, _ref),
    "grpo_step": (_logits, _samples, _rewards, _ref, 0.1, 0.01),
    "winner": ("xoxoxoox.",),
    "minimax_value": ("." * 9, "x"),
    "best_move": ("x...o....", "x"),
    "puct_score": (0.4, 0.2, 100, 7),
}
