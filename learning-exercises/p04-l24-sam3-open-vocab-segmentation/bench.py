"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_SIZE = 256

# маска-«клякса»: сплошные куски дают длинные runs, как у настоящей сегментации
_mask_a = [
    [1 if (x - 90) ** 2 + (y - 110) ** 2 < 70 ** 2 else 0 for x in range(_SIZE)]
    for y in range(_SIZE)
]
_mask_b = [
    [1 if (x - 140) ** 2 + (y - 130) ** 2 < 60 ** 2 else 0 for x in range(_SIZE)]
    for y in range(_SIZE)
]

_DECODE_WIDTH = 8
_runs = [(i % 2, random.randint(1, 40)) for i in range(4000)]
# суммарная длина обязана делиться на ширину, иначе rle_decode честно
# бросит ValueError и замер не состоится
_tail = (-sum(length for _, length in _runs)) % _DECODE_WIDTH
if _tail:
    _runs[-1] = (_runs[-1][0], _runs[-1][1] + _tail)
_rle = ";".join(f"{value}x{length}" for value, length in _runs)

_sentence = ", ".join(["yellow school bus", "striped red umbrella and sandwich"] * 200)

_detections = [
    {"score": random.random(), "mask_rle": "1x10;0x20"} for _ in range(2000)
]
_per_concept = {f"concept-{i}": _detections[i * 100 : (i + 1) * 100] for i in range(20)}

BENCH = {
    "split_concepts": (_sentence,),
    "rle_encode": (_mask_a,),
    "rle_decode": (_rle, _DECODE_WIDTH),
    "mask_area": (_rle,),
    "mask_to_box": (_mask_a,),
    "mask_iou": (_mask_a, _mask_b),
    "presence_gate": (_detections, 0.9),
    "merge_concept_results": (_per_concept,),
}
