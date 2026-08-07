"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)  # обязательно: замер должен быть воспроизводим

_AUDIO_DIM, _HIDDEN, _LLM_DIM = 128, 96, 192

_x = [_rng.gauss(0.0, 1.0) for _ in range(_AUDIO_DIM)]
_W1 = [[_rng.gauss(0.0, 0.1) for _ in range(_AUDIO_DIM)] for _ in range(_HIDDEN)]
_b1 = [0.0] * _HIDDEN
_W2 = [[_rng.gauss(0.0, 0.1) for _ in range(_HIDDEN)] for _ in range(_LLM_DIM)]
_b2 = [0.0] * _LLM_DIM

_frames = [[_rng.gauss(0.0, 1.0) for _ in range(_AUDIO_DIM)] for _ in range(60)]
_wide = [_rng.gauss(0.0, 1.0) for _ in range(20000)]

_embed = lambda text: [[float(ord(ch))] * 1 for ch in text]
_parts = [("text", "what sounds do you hear" * 20), ("audio", [[1.0]] * 2000)]

_modules = {f"block_{i}": float(i * 1000) for i in range(500)}
_trainable = [f"block_{i}" for i in range(0, 500, 2)]

_items = [
    {
        "category": ["speech", "sound", "music", "multi"][i % 4],
        "predicted": "a",
        "correct": "a" if i % 3 else "b",
    }
    for i in range(20000)
]

_samples = [_rng.uniform(-1.0, 1.0) for _ in range(200000)]

BENCH = {
    "linear": (_x, _W1, _b1),
    "gelu": (_wide,),
    "project": (_frames, [(_W1, _b1), (_W2, _b2)]),
    "build_lm_sequence": (_parts, _embed, 1),
    "trainable_parameter_count": (_modules, _trainable),
    "accuracy_by_category": (_items,),
    "is_above_chance": (0.52, 4),
    "gate_on_silence": ("dog barks", _samples, 0.01),
}
