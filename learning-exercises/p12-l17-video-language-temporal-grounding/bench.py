"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

# 600 кадров по 1024 «пикселя» — минута видео при 10 FPS
_FRAMES = [[random.random() for _ in range(1024)] for _ in range(600)]

# посекундное движение получасового ролика
_MOTION = [random.random() for _ in range(1800)]

# по сотне предсказаний и эталонных событий: сопоставление квадратичное
_NAMES = ("jump", "turn", "sit", "wave")
_TRUTHS = []
_PREDS = []
for _i in range(200):
    _start = random.uniform(0.0, 1790.0)
    _name = _NAMES[_i % len(_NAMES)]
    _TRUTHS.append((_name, _start, _start + 0.5))
    _PREDS.append((_name, _start + random.uniform(-0.3, 0.3), _start + 0.6))

_TIMES = [i * 0.37 for i in range(5000)]
_TEXT = " ".join(f"event {i} at <time>{i * 0.5}</time>" for i in range(2000))

BENCH = {
    "frame_difference": (_FRAMES,),
    "uniform_sample": (1800.0, 5000),
    "dynamic_sample": (_MOTION, 5000, 4),
    "pooled_tokens": (27, 3),
    "temporal_iou": (4.0, 4.5, 4.1, 4.7),
    "grounding_recall": (_PREDS, _TRUTHS, 0.3),
    "position_ids": (_TIMES, "time"),
    "parse_time_tokens": (_TEXT,),
}
