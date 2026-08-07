"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

# словарь MIO целиком: 48k слотов, по ним и ищется владелец id
_PLAN = [
    ("text", 32000),
    ("image", 4096),
    ("speech", 4096),
    ("music", 8192),
] + [(f"sep{i}", 1) for i in range(10)]

_SLOTS = []
_cursor = 0
for _name, _size in _PLAN:
    _SLOTS.append((_name, _cursor, _cursor + _size))
    _cursor += _size

_STAGES = [(f"stage{i}", random.uniform(0.0, 50.0)) for i in range(2000)]

BENCH = {
    "allocate_vocab": (_PLAN,),
    "modality_of": (_SLOTS, _cursor - 1),
    "route_modality": ("voice",),
    "embedding_params": (_cursor, 4096),
    "residual_vq_tokens": (3600.0, 20, 8),
    "latency_trace": (_STAGES,),
    "latency_verdict": (612.0,),
    "curriculum_gap": (["alignment", "sft"],),
}
