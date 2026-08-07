"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_PATCH_DIM = 32
_LLM_DIM = 48
_PATCHES = [[random.gauss(0, 1) for _ in range(_PATCH_DIM)] for _ in range(256)]
_QUERIES = [[random.gauss(0, 1) for _ in range(_PATCH_DIM)] for _ in range(32)]
_W_PROJ = [[random.gauss(0, 0.1) for _ in range(_PATCH_DIM)] for _ in range(_LLM_DIM)]
_ATTN = [[random.random() for _ in range(256)] for _ in range(32)]

BENCH = {
    "softmax": ([random.gauss(0, 3) for _ in range(4096)],),
    "scaled_dot_attention": (_QUERIES[0], _PATCHES, _PATCHES),
    "cross_attention": (_QUERIES, _PATCHES, _PATCHES),
    "linear_project": (_QUERIES, _W_PROJ),
    "qformer_forward": (_PATCHES, _QUERIES, _W_PROJ),
    "top_patches_per_query": (_ATTN, 3),
    "visual_token_budget": (60, 576, 32),
    "pick_bridge": (60, 576, 32, 32768),
}
