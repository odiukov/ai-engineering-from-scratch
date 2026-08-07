"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

# 96x96 патчей — сетка крупнее любой реальной, зато пулинг успевает
# показать разницу между поблочным обходом и наивной пересборкой списков.
_grid = [[random.random() for _ in range(96)] for _ in range(96)]

_scenarios = {
    "single": (9, 24, True),
    "multi": (6, 24, False),
    "video": (32, 24, False),
    "long_video": (64, 24, False),
}

BENCH = {
    "pool_grid": (_grid, 3),
    "pooled_tokens": (24, 2),
    "scenario_tokens": (32, 24, 3),
    "best_pool_factor": (32, 24, 2600),
    "allocate_budget": (_scenarios, 2600),
    "is_valid_curriculum": (("si", "ov", "tt"),),
    "stage_steps": (100000, {"si": 0.5, "ov": 0.3, "tt": 0.2}),
}
