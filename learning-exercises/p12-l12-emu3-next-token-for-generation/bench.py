"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

# Словарь размером с IBQ-токенизатор Emu3: на такой длине видно разницу
# между одним проходом по логитам и пересчётом суммы на каждый элемент.
_VOCAB = 32768
_cond = [random.uniform(-8.0, 8.0) for _ in range(_VOCAB)]
_uncond = [random.uniform(-8.0, 8.0) for _ in range(_VOCAB)]

# Семплер вызывает softmax на каждый токен, поэтому ему нужен словарь
# поменьше: 512 токенов по словарю в 512 записей — те самые десятки
# миллисекунд на один вызов.
_small_cond = _cond[:512]
_small_uncond = _uncond[:512]

BENCH = {
    "image_tokens": (512, 512, 8),
    "video_tokens": (256, 256, 32, 4, 4),
    "frames_in_clip": (4.0, 8),
    "generation_seconds": (4096, 30),
    "cfg_logits": (_cond, _uncond, 5.0),
    "softmax": (_cond, 0.8),
    "sample_token": (_cond, random.Random(0), 0.8),
    "sample_image_tokens": (512, _small_cond, _small_uncond, random.Random(0)),
}
