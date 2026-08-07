"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)  # обязательно: замер должен быть воспроизводим

_weights = {"sound": 0.2, "shortcut": 0.5, "random": 0.3}

_traces = [
    {
        "strategy": _rng.choice(["sound", "shortcut", "random"]),
        "answer_correct": _rng.random() < 0.5,
        "rationale_sound": _rng.random() < 0.3,
    }
    for _ in range(4000)
]

BENCH = {
    "pick_strategy": (_rng, _weights),
    "sample_trace": (_rng, _weights),
    "expected_accuracy": (_weights,),
    "star_filter": (_traces,),
    "rationalize": (_traces,),
    "finetune": (_weights, _traces, 0.6),
    "star_round": (_rng, _weights, 2000),
    "vstar_select": (_traces, lambda t: 1.0 if t["rationale_sound"] else 0.0),
}
