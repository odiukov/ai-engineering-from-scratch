"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)  # обязательно: замер должен быть воспроизводим

# Кэш, в который рабочее множество заведомо не влезает: вытеснение работает
# почти на каждом запросе, и каскад вниз по уровням — тоже.
_cache = {
    "capacity": {"gpu": 100_000, "cpu": 400_000, "disk": 2_000_000},
    "tiers": {"gpu": {}, "cpu": {}, "disk": {}},
}

# Зипфоподобный поток: горячая двадцатка префиксов и длинный хвост. Наивная
# реализация, которая на каждом обращении пересчитывает занятость уровня
# суммой по всему словарю, на таком размере уже заметно тормозит.
_requests = []
for _i in range(3000):
    _hot = _rng.random() < 0.6
    _key = f"doc{_rng.randrange(20) if _hot else _rng.randrange(20, 1500)}"
    _requests.append((_key, _rng.randrange(500, 16_000)))

BENCH = {
    "recompute_ms": (4000,),
    "restore_ms": (4000, "disk"),
    "effective_hit_ms": (4000, "disk"),
    "make_cache": (100_000, 400_000, 2_000_000),
    "cache_lookup": (_cache, "doc7"),
    "cache_put": (_cache, "doc7", 4000),
    "serve_request": (_cache, "doc7", 4000),
    "run_workload": (_cache, _requests),
}
