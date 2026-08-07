"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

# Плотная страница: 2000 слов с координатами, как после OCR листа A4
_tokens = []
for _i in range(2000):
    _x = random.randrange(0, 1900)
    _y = random.randrange(0, 2700)
    _tokens.append((f"w{_i}", (_x, _y, _x + 60, _y + 24)))

_record = {f"field_{i}": f"value {i}" for i in range(300)}
_markup = "".join(f"<s_{k}>{v}</s_{k}>" for k, v in _record.items())

BENCH = {
    "normalize_bbox": ((100, 50, 300, 80), 2000, 2800),
    "iou": ((10, 10, 40, 30), (20, 15, 60, 50)),
    "reading_order": (_tokens, 30),
    "layoutlm_input": (_tokens, 2000, 2800),
    "donut_serialize": (_record,),
    "donut_parse": (_markup,),
    "anyres_tokens": (2500, 3500),
    "pick_stack": ({"pages_per_day": 10_000_000, "handwriting": False},),
}
