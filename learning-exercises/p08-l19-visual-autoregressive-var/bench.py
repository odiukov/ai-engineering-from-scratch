"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_rng = random.Random(0)
_SIZE = 32
_SCALES = (1, 2, 4, 8, 16, 32)
_BOOK = [i / 16.0 - 1.0 for i in range(33)]
_BOOKS = [_BOOK] * len(_SCALES)

_img = [[_rng.uniform(0.0, 1.0) for _ in range(_SIZE)] for _ in range(_SIZE)]
_grid = [[_rng.uniform(0.0, 1.0) for _ in range(4)] for _ in range(4)]
_tokens = [[[_rng.randrange(len(_BOOK)) for _ in range(s)] for _ in range(s)]
           for s in _SCALES]
# маска квадратична по числу токенов: на 6 масштабах это уже 1365x1365
_MASK_SCALES = (1, 2, 4, 8, 16)

BENCH = {
    "downsample": (_img, 4),
    "upsample": (_grid, _SIZE),
    "encode_grid": (_img, _BOOK),
    "tokenize_multiscale": (_img, _BOOKS, _SCALES),
    "detokenize_multiscale": (_tokens, _BOOKS, _SIZE),
    "scale_positions": (_SCALES,),
    "scale_causal_mask": (_MASK_SCALES,),
    "generate_scales": (lambda k, prev: [1.0 / 33] * 33, _SCALES, random.Random(1)),
}
