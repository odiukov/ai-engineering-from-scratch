"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_stages = {f"stage_{i}": 10.0 + i for i in range(200)}
_stages["prefill"] = 150.0

# 20 секунд диалога при 50 Гц: звук, кадры камеры и текст вперемешку
_events = (
    [(i / 50.0, "audio", f"a{i}") for i in range(1000)]
    + [(i / 4.0, "vision", f"v{i}") for i in range(80)]
    + [(random.random() * 20.0, "text", f"t{i}") for i in range(50)]
)
random.shuffle(_events)

# 30 секунд кадров по 20 мс, тишина только в самом конце
_energies = [random.random() for _ in range(1450)] + [0.0] * 50

BENCH = {
    "ttfab_ms": (_stages,),
    "scaled_budget": (_stages, 70.0),
    "speech_tokens_needed": (30.0,),
    "talker_keeps_up": (64,),
    "pipeline_total_ms": (5000, 40.0, 20.0, 100.0),
    "tmrope_positions": (_events, 25),
    "interleave_by_time": (_events,),
    "turn_end_frame": (_energies,),
}
