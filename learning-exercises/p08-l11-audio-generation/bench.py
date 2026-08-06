"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_rng = random.Random(0)
_VOCAB = 64

_books = [
    [i / 8.0 for i in range(9)],
    [(i - 8) / 40.0 for i in range(17)],
    [(i - 8) / 200.0 for i in range(17)],
]
_sequences = [[_rng.randrange(_VOCAB) for _ in range(200)] for _ in range(200)]
_counts = [[1.0 + _rng.random() for _ in range(_VOCAB)] for _ in range(_VOCAB)]
_streams = [[_rng.randrange(1024) for _ in range(2000)] for _ in range(8)]
_delayed = [[_rng.randrange(1024) for _ in range(2007)] for _ in range(8)]

BENCH = {
    "codec_token_count": (30, 75, 8),
    "rvq_encode": (0.617, _books),
    "rvq_decode": ([3, 5, 9], _books),
    "train_bigram": (_sequences, _VOCAB),
    "next_token_probs": (_counts, 7, 0.8),
    "generate_tokens": (_counts, 0, 3000, random.Random(1), 0.8),
    "delay_streams": (_streams, -1),
    "undelay_streams": (_delayed,),
}
