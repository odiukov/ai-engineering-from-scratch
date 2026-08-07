"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_SHAPE = (120, 360, 640)      # 5 секунд 360p при 24 кадрах в секунду
_PATCH = (2, 8, 8)

# сетка поменьше: группы и позиционное кодирование материализуют все токены
_GRID = (16, 16, 16)

_STATE = [random.uniform(-1.0, 1.0) for _ in range(64)]
_NEXT = [random.uniform(-1.0, 1.0) for _ in range(64)]
_ACTIONS = [[random.uniform(-0.1, 0.1) for _ in range(64)] for _ in range(500)]
_STEP = lambda s, a: [x + y for x, y in zip(s, a)]

BENCH = {
    "token_grid": (_SHAPE, _PATCH),
    "token_count": (_SHAPE, _PATCH),
    "flat_index": (_GRID, 8, 8, 8),
    "divided_attention_groups": (_GRID, "time"),
    "attention_pairs": (_GRID, "divided"),
    "axis_position_encoding": (_GRID, 16, 16, 16),
    "inverse_dynamics": (_STATE, _NEXT),
    "imagine_rollout": (_STATE, _ACTIONS, _STEP),
}
