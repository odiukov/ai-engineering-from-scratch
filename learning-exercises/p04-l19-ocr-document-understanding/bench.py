"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_VOCAB = ["_"] + list("0123456789abcdefghijklmnopqrstuvwxyz")
_C = len(_VOCAB)

# 2000 шагов времени: как строка на длинном скане после CNN-энкодера
_frames = [random.randrange(_C) for _ in range(2000)]
_log_probs = [
    [0.0 if c == i else -9.0 + random.random() for c in range(_C)] for i in _frames
]

_ref = "".join(random.choice("abcdefghij ") for _ in range(600))
_hyp = "".join(random.choice("abcdefghij ") for _ in range(600))

_boxes = [
    (random.randrange(0, 900), 20 * (i // 12) + random.randrange(0, 3), 0, 0)
    for i in range(600)
]
_boxes = [(b[0], b[1], b[0] + 40, b[1] + 14) for b in _boxes]

_pred = {f"field_{i}": str(i) for i in range(2000)}
_gold = {f"field_{i}": str(i * (i % 2)) for i in range(2000)}

BENCH = {
    "ctc_collapse": (_frames,),
    "greedy_ctc_decode": (_log_probs,),
    "decode_text": (_log_probs, _VOCAB),
    "levenshtein": (_ref, _hyp),
    "cer": (_ref, _hyp),
    "wer": (_ref, _hyp),
    "reading_order": (_boxes,),
    "field_f1": (_pred, _gold),
}
