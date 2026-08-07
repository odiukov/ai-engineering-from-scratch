"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_CRITERIA = ("throughput", "setup_simplicity", "ecosystem", "prefix_reuse",
             "model_coverage", "future_proof")
_HARDWARE = ("AMD", "Apple Silicon", "CPU", "NVIDIA Blackwell", "NVIDIA Hopper")

# Синтетический каталог: в реальном мире движков шесть, но перебор кандидатов —
# именно то место, которое обязано остаться линейным, а не квадратичным.
_support = {}
_scores = {}
for _i in range(4000):
    _name = "engine-%04d" % _i
    _support[_name] = tuple(hw for hw in _HARDWARE if random.random() < 0.5) or ("AMD",)
    _scores[_name] = {c: random.uniform(0.0, 5.0) for c in _CRITERIA}

_since = {"engine-%04d" % i: "2025-12-11" for i in range(0, 4000, 17)}

_weights = {c: random.uniform(0.1, 1.0) for c in _CRITERIA}

# pipeline_plan берёт формат весов из настоящей таблицы движков, поэтому
# синтетический каталог ему не подсунуть — гоняем реальные шесть на длинной
# цепочке стадий.
_stages = [
    ("dev-%d" % i, hw, profile)
    for i, (hw, profile) in enumerate(
        [("Apple Silicon", "dev"), ("CPU", "staging"),
         ("NVIDIA Hopper", "prod_general"), ("AMD", "prod_agentic"),
         ("NVIDIA Blackwell", "ecosystem_first")] * 400
    )
]

BENCH = {
    "supports": ("engine-0001", "AMD", _support),
    "eligible_engines": ("AMD", _support),
    "normalize_weights": (_weights,),
    "maintenance_multiplier": ("engine-0000", "2026-08-07", _since),
    "weighted_score": ("engine-0001", _weights, "2026-08-07", _scores, _since),
    "rank_engines": ("AMD", _weights, "2026-08-07", _support, _scores, _since),
    "pick_engine": ("AMD", _weights, "2026-08-07", _support, _scores, _since),
    "pipeline_plan": (_stages, "2026-08-07"),
}
