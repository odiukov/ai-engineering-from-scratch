"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)          # обязательно: замер должен быть воспроизводим

_frames = [[_rng.uniform(-5, 5) for _ in range(13)] for _ in range(300)]
_bank = [[_rng.uniform(-1, 1) for _ in range(26)] for _ in range(400)]
_labels = [f"c{i % 8}" for i in range(400)]
_query = [_rng.uniform(-1, 1) for _ in range(26)]

_spec = [[_rng.uniform(0.1, 1.0) for _ in range(80)] for _ in range(200)]

_y_true = [_rng.randrange(10) for _ in range(3000)]
_y_pred = [_rng.randrange(10) for _ in range(3000)]
_cm = [[_rng.randrange(1, 50) for _ in range(10)] for _ in range(10)]

_scores = [_rng.random() for _ in range(3000)]
_binary = [1 if _rng.random() < 0.1 else 0 for _ in range(3000)]

BENCH = {
    "summarize": (_frames,),
    "cosine_similarity": (_query, _bank[0]),
    "knn_classify": (_query, _bank, _labels, 5),
    "mixup": (_bank[0], [1.0, 0.0], _bank[1], [0.0, 1.0], 0.4),
    "spec_augment": (_spec, random.Random(1), 2, 2, 20, 10),
    "confusion_matrix": (_y_true, _y_pred, 10),
    "macro_f1": (_cm,),
    "average_precision": (_scores, _binary),
}
