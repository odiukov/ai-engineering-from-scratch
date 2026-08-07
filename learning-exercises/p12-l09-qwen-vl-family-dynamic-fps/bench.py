"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_DIM = 96  # столько же, сколько в примере M-RoPE из урока
_vec = [random.uniform(-1.0, 1.0) for _ in range(_DIM)]
_freqs = [10000.0 ** (-2.0 * i / _DIM) for i in range(_DIM // 2)]

# Упаковка «промпт + картинка + короткое видео»: 6400 позиций, столько же,
# сколько реально уезжает в LLM за один запрос.
_sequence = [("text", 64), ("image", 40, 24), ("video", 32, 12, 12), ("text", 32)]

_response = (
    "Хорошо, кликаю по кнопке.\n```json\n"
    '{"tool": "mouse_click", "coords": [1024, 512], '
    '"button": "left", "note": "кнопка {Отправить}"}\n```\n'
)

BENCH = {
    "rope_frequencies": (_DIM,),
    "rotate_pairs": (_vec, 137.0, _freqs),
    "mrope_positions": (_sequence,),
    "mrope_rotate": (_vec, (137, 40, 24)),
    "pick_fps": (300.0, 200000, 81, "high"),
    "frame_timestamps": (3600.0, 8),
    "parse_tool_call": (_response,),
}
