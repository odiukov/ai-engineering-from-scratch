"""Входные данные для замера скорости."""

import math
import random

_rng = random.Random(0)  # обязательно: замер должен быть воспроизводим

_DIM = 64


def _unit(seed_rng):
    v = [seed_rng.gauss(0.0, 1.0) for _ in range(_DIM)]
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


# Кэш из тысячи записей: линейный перебор здесь заметно медленнее, чем
# ленивая арифметика внутри cosine, и разница видна на глаз.
_entries = [{"vector": _unit(_rng), "answer": f"ans_{i}"} for i in range(1000)]
_probe = _unit(_rng)

# Поток запросов с повторами: часть векторов дублирует уже виденные, так что
# кэш действительно наполняется и попадания случаются.
_pool = [_unit(_rng) for _ in range(120)]
_queries = [
    {"vector": list(_pool[_rng.randrange(len(_pool))]), "answer": f"a{_rng.randrange(120)}"}
    for _ in range(600)
]

_records = [
    {"served": "cache" if i % 3 else "llm", "answer": "x", "expected": "x",
     "correct": i % 7 != 0, "similarity": 0.9}
    for i in range(20000)
]

_long_a = [f"tok{i}" for i in range(20000)]
_long_b = _long_a[:19000] + [f"other{i}" for i in range(1000)]

BENCH = {
    "cosine": (_entries[0]["vector"], _probe),
    "nearest_entry": (_entries, _probe),
    "semantic_lookup": (_entries, _probe, 0.95),
    "run_semantic_cache": (_queries, 0.95),
    "cache_stats": (_records,),
    "common_prefix_tokens": (_long_a, _long_b),
    "l2_request_cost": (4000, 200, 200, "5min", False),
    "parallel_wave_cost": (10, 4000, 50, 50, "1hr", False),
}
