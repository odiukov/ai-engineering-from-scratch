"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

MASK = -1

_VOCAB = 128
_SEQ_LEN = 256  # сетка 16x16 VQ-токенов: столько же, сколько патчей у ViT

_TOKENS = [MASK] * _SEQ_LEN
_LOGITS = [[random.gauss(0.0, 1.0) for _ in range(_VOCAB)] for _ in range(_SEQ_LEN)]


def _predict(tokens):
    # заглушка вместо transformer: логиты не зависят от состояния, зато
    # замер меряет ровно стоимость самого сэмплера
    return _LOGITS[: len(tokens)]


BENCH = {
    "cosine_schedule": (128,),
    "linear_schedule": (128,),
    "unmask_counts": ([1.0 - i / 128 for i in range(129)], _SEQ_LEN),
    "softmax": (_LOGITS[0],),
    "top_k_confident": (_TOKENS, _LOGITS, 32),
    "unmask_step": (_TOKENS, _LOGITS, 32),
    "sample_masked": (_predict, _TOKENS, 16),
    "compression_ratio": (512, 512, 1024, 16384),
}
