"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_VOCAB, _N = 512, 128

_tokens = [random.randrange(3, _VOCAB) for _ in range(_N)]
_logits = [[random.uniform(-3.0, 3.0) for _ in range(_VOCAB)] for _ in range(_N)]
_labels = [
    random.randrange(_VOCAB) if random.random() < 0.15 else -100 for _ in range(_N)
]
_hidden = [[random.uniform(-1.0, 1.0) for _ in range(64)] for _ in range(_N)]
_W = [[random.uniform(-0.5, 0.5) for _ in range(64)] for _ in range(4)]

BENCH = {
    "softmax": ([random.uniform(-3.0, 3.0) for _ in range(_VOCAB)],),
    "build_bert_input": (_tokens, 1, 2, _tokens),
    "create_mlm_batch": (_tokens, _VOCAB, _VOCAB, random.Random(0)),
    "mlm_loss": (_logits, _labels),
    "mlm_loss_grad": (_logits, _labels),
    "mlm_accuracy": (_logits, _labels),
    "classify_from_cls": (_hidden, _W, [0.0] * 4),
}
