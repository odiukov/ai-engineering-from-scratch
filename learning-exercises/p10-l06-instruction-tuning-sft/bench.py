"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_WORDS = ["model", "token", "answer", "gravity", "python", "paris", "learning"]


def _sentence(n):
    return " ".join(random.choice(_WORDS) for _ in range(n))


_INSTRUCTION = _sentence(40)
_RESPONSE = _sentence(120)

_TOKENS = (
    [256] + list(_INSTRUCTION.encode("utf-8")) + [257, 258]
    + list(_RESPONSE.encode("utf-8")) + [259]
)
_MASK = [0.0] * (len(_TOKENS) - len(_RESPONSE.encode("utf-8")) - 1) + [1.0] * (
    len(_RESPONSE.encode("utf-8")) + 1
)

_VOCAB = 260
_ROWS = [[random.gauss(0.0, 1.0) for _ in range(_VOCAB)] for _ in range(64)]
_TARGETS = [random.randrange(_VOCAB) for _ in range(64)]
_ROW_MASK = [1.0 if i % 3 else 0.0 for i in range(64)]

_TURNS = [
    {"role": "user" if i % 2 == 0 else "assistant", "content": _sentence(20)}
    for i in range(12)
]

_DATASET = [{"instruction": _sentence(20), "response": _sentence(40)} for _ in range(200)]
_RAW = [_sentence(60) for _ in range(20)]

BENCH = {
    "tokenize_instruction_pair": (_INSTRUCTION, _RESPONSE),
    "create_loss_mask": (_TOKENS,),
    "tokenize_conversation": (_TURNS,),
    "shift_for_training": (_TOKENS, _MASK),
    "masked_cross_entropy": (_ROWS, _TARGETS, _ROW_MASK),
    "d_masked_cross_entropy": (_ROWS, _TARGETS, _ROW_MASK),
    "dataset_quality": (_DATASET[0],),
    "mix_pretraining_data": (_DATASET, _RAW, 0.05, random.Random(0)),
}
