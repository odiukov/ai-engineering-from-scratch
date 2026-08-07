"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_IMAGE = [
    [[random.random() for _ in range(3)] for _ in range(96)]
    for _ in range(96)
]
_PATCHES = [[random.random() for _ in range(48)] for _ in range(144)]
_W_E = [[random.random() for _ in range(48)] for _ in range(64)]
_TOKENS = [[random.random() for _ in range(64)] for _ in range(144)]
_TABLE = [[random.random() for _ in range(64)] for _ in range(144)]

BENCH = {
    "grid_shape": (224, 224, 16),
    "sequence_length": (896, 896, 14),
    "extract_patches": (_IMAGE, 4),
    "project_patches": (_PATCHES, _W_E),
    "add_position_embeddings": (_TOKENS, _TABLE),
    "mean_pool": (_TOKENS,),
    "vit_param_count": (224, 16, 768, 12),
}
