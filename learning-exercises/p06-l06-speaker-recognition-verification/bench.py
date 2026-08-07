"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)          # обязательно: замер должен быть воспроизводим

_DIM = 192                       # размерность эмбеддинга ECAPA-TDNN

_vec = [_rng.gauss(0, 1) for _ in range(_DIM)]
_test = [_rng.gauss(0, 1) for _ in range(_DIM)]
_samples = [[_rng.gauss(0, 1) for _ in range(_DIM)] for _ in range(5)]

_bank = [[_rng.gauss(0, 1) for _ in range(_DIM)] for _ in range(500)]
_labels = [f"spk{i}" for i in range(500)]

_same = [_rng.gauss(0.6, 0.15) for _ in range(1000)]
_diff = [_rng.gauss(0.1, 0.15) for _ in range(1000)]

BENCH = {
    "l2_normalize": (_vec,),
    "cosine_score": (_vec, _test),
    "enroll_speaker": (_samples,),
    "verify": (_vec, _test, 0.25),
    "false_rates": (_same, _diff, 0.35),
    "equal_error_rate": (_same, _diff),
    "identify": (_test, _bank, _labels, 0.25),
    "aam_margin_logit": (0.87, 0.2, 30.0),
}
